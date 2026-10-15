"""The LLM judge. Two jobs, one rubric, and a hard rule about the gold labels.

**Job 1 — false-negative filtering.** Hard negatives are mined from the top of
the retrieval list (``concierge.synth.negatives``), which means some of them are
genuinely relevant: other quiet cafés with wifi really do exist. Training the
cross-encoder to score those as irrelevant teaches it something false, and it
would be *penalised at evaluation time* for the very behaviour we taught it. So a
judge grades the hardest negatives and the ones it calls highly relevant are
removed before training. The removal count is a reported number.

**Job 2 — label-quality validation.** Job 1 only makes sense if the judge's
grades mean roughly the same thing as the owner's. So the same judge, the same
prompt and the same rubric are run over a sample of pairs the owner labeled by
hand, and agreement is reported as Cohen's kappa with its confusion matrix (via
``eval.metrics``). That measurement is the entire justification for using LLM
labels in training data at all; without it, "we used an LLM judge" is an
unsupported claim, and with a bad kappa the honest conclusion is that the
filtering step should be dropped.

**THE HARD RULE: this module is read-only with respect to gold labels.** Only the
owner's labeling CLI writes ``eval/qrels.csv``. A judge that could write labels
would make the evaluation circular — the system would be graded against grades it
produced — and no amount of care in a docstring prevents that in six months. It is
enforced structurally, in three independent ways:

1. **No writer exists here.** This module contains no file-opening-for-write call
   of any kind. Every write in the synthetic-data package goes through
   ``concierge.synth.generate.JsonlCache``, which cannot be pointed at a CSV
   schema, and ``tests/test_synth.py`` asserts that this file's source contains no
   write mode and no reference to the qrels path.
2. **The labels arrive as an immutable value.** ``GoldLabels`` is a frozen
   dataclass of frozen rows; the judge functions take one and have no path.
3. **A tripwire.** ``GoldLabels`` records the file's SHA-256 at load time, and
   ``validate_label_quality`` re-reads and re-hashes the file before returning —
   if anything changed it under us, the result is an exception, not a report.

Everything network-facing is injected, so the tests run with no API key.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from concierge.config import REPO_ROOT, Config
from concierge.query.llm import prompt_version_for
from concierge.synth.generate import (
    ConfirmFn,
    JsonlCache,
    LlmJsonClient,
    QuotaPlan,
    SynthBadResponseError,
    SynthError,
    SynthRateLimitError,
    require_confirmation,
)
from concierge.synth.negatives import TrainingExample
from concierge.types import (
    RELEVANCE_HIGH,
    RELEVANCE_IRRELEVANT,
    RELEVANCE_PARTIAL,
    GoldQuery,
)
from eval.metrics import ORDINAL_GRADES, KappaResult, cohens_kappa

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_relevance.txt"
RUBRIC_PLACEHOLDER = "{{RUBRIC}}"
PAIRS_PLACEHOLDER = "{{PAIRS}}"

# The rubric, injected rather than written in the prompt file, for the same reason
# the attribute vocabulary is injected in concierge/query/llm.py: these three
# sentences must be the SAME three sentences the owner labels by. If the prompt
# drifted from the owner's rubric, a low kappa would measure the drift rather than
# the judge, and the conclusion drawn from it would be wrong.
RUBRIC: dict[int, str] = {
    RELEVANCE_IRRELEVANT: (
        "irrelevant — a user who typed this query would not want this place, or the "
        "place clearly violates something the query asked for"
    ),
    RELEVANCE_PARTIAL: (
        "partially relevant — the right kind of place and useful to see, but it misses "
        "part of what was asked, or the description does not say whether it matches"
    ),
    RELEVANCE_HIGH: (
        "highly relevant — this place satisfies what the query asked for, and a user "
        "would be happy to see it as the first result"
    ),
}

# Which judge grade marks a mined negative as a FALSE negative, i.e. one to remove.
#
# Default: only grade 2. This is a real trade-off and worth being able to defend.
# Removing grade-1 negatives too would give a cleaner training signal in the narrow
# sense, but grade 1 is "the right kind of place, missing part of the request" —
# which is exactly the near-boundary contrast a reranker has to learn, and it is
# still genuinely worse than the source business that satisfies the whole request.
# Removing those would strip out most of what makes a hard negative useful and
# leave a training set closer to random negatives. Only a grade-2 negative is an
# outright label error, because it would teach "this place that fully satisfies the
# query is irrelevant". The constant is exposed so the stricter variant can be run
# and measured rather than argued about.
DEFAULT_FALSE_NEGATIVE_GRADE = RELEVANCE_HIGH

# Judge temperature: 0.0, unlike generation. Grading is a measurement, and a
# measurement that changes between runs cannot be put in a README. We want the
# model's modal label, not a sample from its distribution.
JUDGE_TEMPERATURE = 0.0

# How many of a query's negatives to grade by default, hardest (best-ranked) first.
# False negatives concentrate at the top of the retrieval list — the whole reason a
# negative is hard is that it looks like the positive — so the first few ranks are
# where the grading budget buys the most. Negatives below this depth are KEPT
# UNGRADED, which is a known, reported limitation and not a silent one: the report
# carries n_graded alongside n_negatives_in.
DEFAULT_JUDGE_TOP_N = 3


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────


def render_rubric() -> str:
    return "\n".join(f"  {grade} = {text}" for grade, text in sorted(RUBRIC.items()))


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def judge_prompt_version() -> str:
    """``<declared>+<hash of body and rubric>``; part of every cache key."""
    return prompt_version_for(prompt_text(), render_rubric())


def _strip_header(text: str) -> str:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip() + "\n"


def judge_response_schema() -> dict[str, Any]:
    """Constrained decoding: a grade outside 0/1/2 is refused by the decoder."""
    return {
        "type": "object",
        "properties": {
            "grades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_id": {"type": "string"},
                        "grade": {"type": "integer", "enum": list(ORDINAL_GRADES)},
                        "reason": {"type": "string"},
                    },
                    "required": ["pair_id", "grade"],
                },
            }
        },
        "required": ["grades"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pairs and verdicts
# ─────────────────────────────────────────────────────────────────────────────


def pair_id(query_id: str, business_id: str) -> str:
    """Stable id for one (query, business) pair — the judge cache key.

    Content-addressed so a cached verdict survives re-mining, reordering and
    resumption. The prompt version is folded in by the cache path, not here, so a
    pair id is stable across prompt edits while its verdict is not reused.
    """
    digest = hashlib.sha256(f"{query_id}\x00{business_id}".encode()).hexdigest()
    return f"p-{digest[:16]}"


@dataclass(frozen=True, slots=True)
class JudgePair:
    """One thing to grade: a query, a business, and the document the judge reads."""

    query_id: str
    query: str
    business_id: str
    document: str

    @property
    def pair_id(self) -> str:
        return pair_id(self.query_id, self.business_id)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    grade: int
    reason: str = ""
    pair_id: str = ""

    def __post_init__(self) -> None:
        if self.grade not in ORDINAL_GRADES:
            raise SynthError(f"judge grade {self.grade!r} outside {list(ORDINAL_GRADES)}")


# ─────────────────────────────────────────────────────────────────────────────
# The judge
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class JudgeRun:
    """Verdicts plus what the grading cost and what it failed to grade."""

    verdicts: Mapping[str, JudgeVerdict]
    n_api_calls: int
    n_from_cache: int
    n_failed_batches: int
    n_unparsed_pairs: int
    stopped_early: bool = False
    stop_reason: str = ""


class LlmJudge:
    """Grades ``JudgePair``s, batching and caching, and never writing a label file.

    Caching is per *pair*, not per batch: batches get re-formed whenever the
    negative set changes, and a per-batch cache would miss on every reshuffle and
    re-spend quota on pairs already graded. The cache path contains the prompt
    version, so editing the rubric invalidates old verdicts instead of silently
    mixing two judges' grades inside one reported kappa.
    """

    def __init__(
        self,
        client: LlmJsonClient | None,
        *,
        batch_size: int = 8,
        cache: JsonlCache | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self.client = client
        self.batch_size = max(1, batch_size)
        self.prompt_version = judge_prompt_version() if prompt_version is None else prompt_version
        self.cache = cache
        self._memo: dict[str, JudgeVerdict] = {}
        if cache is not None:
            for key, row in cache.load().items():
                try:
                    self._memo[key] = JudgeVerdict(
                        grade=int(row["grade"]),
                        reason=str(row.get("reason", "")),
                        pair_id=key,
                    )
                except (KeyError, SynthError, TypeError, ValueError):
                    log.warning("discarding unusable cached verdict for %s", key)

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        client: LlmJsonClient | None = None,
        root: Path | None = None,
    ) -> LlmJudge:
        """Build the real judge. The cache lives beside the generation cache under
        ``.llm_cache/`` because, like it, it is spent money that ``make clean``
        must not be able to delete."""
        from concierge.synth.generate import GeminiJsonClient

        version = judge_prompt_version()
        base = REPO_ROOT if root is None else Path(root)
        safe_version = version.replace("/", "_")
        return cls(
            client=(
                client
                if client is not None
                else GeminiJsonClient.from_config(
                    cfg,
                    response_schema=judge_response_schema(),
                    temperature=JUDGE_TEMPERATURE,
                )
            ),
            batch_size=cfg.synth.batch_size,
            cache=JsonlCache(base / ".llm_cache" / f"synth_judge_{safe_version}.jsonl", "pair_id"),
            prompt_version=version,
        )

    # ── prompt ───────────────────────────────────────────────────────────────

    def render_prompt(self, pairs: Sequence[JudgePair], max_chars: int | None = None) -> str:
        body = _strip_header(prompt_text())
        for placeholder in (RUBRIC_PLACEHOLDER, PAIRS_PLACEHOLDER):
            if placeholder not in body:
                raise SynthError(f"judge_relevance.txt is missing {placeholder}")
        blocks = []
        for pair in pairs:
            document = pair.document if max_chars is None else pair.document[:max_chars]
            blocks.append(
                f"--- PAIR {pair.pair_id} ---\n"
                f"QUERY: {pair.query.strip()}\n"
                f"PLACE:\n{document.strip()}"
            )
        return body.replace(RUBRIC_PLACEHOLDER, render_rubric()).replace(
            PAIRS_PLACEHOLDER, "\n\n".join(blocks)
        )

    # ── grading ──────────────────────────────────────────────────────────────

    def plan(self, pairs: Sequence[JudgePair], job: str) -> QuotaPlan:
        """The quota question, asked before anything is spent."""
        todo = [p for p in pairs if p.pair_id not in self._memo]
        n_batches_todo = -(-len(todo) // self.batch_size)  # ceil division
        n_cached_pairs = len(pairs) - len(todo)
        return QuotaPlan(
            job=job,
            n_items=len(pairs),
            items_label="(query, place) pairs to grade",
            # Batches are counted over the ungraded pairs only; cached pairs never
            # form a batch, so counting them here would overstate the cost.
            n_batches=n_batches_todo,
            n_cached_batches=0,
            n_api_calls=n_batches_todo,
            batch_size=self.batch_size,
            note=f"{n_cached_pairs} pair(s) already graded and served from cache",
        )

    def grade(
        self,
        pairs: Sequence[JudgePair],
        *,
        job: str = "LLM judge grading",
        confirm: ConfirmFn | None = None,
        assume_yes: bool = False,
        max_document_chars: int | None = None,
    ) -> JudgeRun:
        """Grade every pair, serving what is cached and batching the rest.

        Stops cleanly on a rate limit and returns what it has: the cache already
        holds every verdict earned so far, so a rerun resumes rather than restarts.
        Partial results are usable — the caller sees ``stopped_early`` and can
        decide whether to filter on a partial grading or wait for quota.
        """
        unique: dict[str, JudgePair] = {}
        for pair in pairs:
            unique.setdefault(pair.pair_id, pair)
        ordered = sorted(unique.values(), key=lambda p: p.pair_id)

        require_confirmation(self.plan(ordered, job), confirm, assume_yes=assume_yes)

        todo = [p for p in ordered if p.pair_id not in self._memo]
        n_cached = len(ordered) - len(todo)
        n_api = n_failed = n_unparsed = 0
        stopped_early = False
        stop_reason = ""

        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            if self.client is None:
                raise SynthError(
                    f"{len(todo)} pair(s) are not cached and no judge client was provided"
                )
            try:
                payload = self.client.complete_json(self.render_prompt(batch, max_document_chars))
            except SynthRateLimitError as exc:
                stopped_early = True
                stop_reason = f"rate limited after {n_api} API call(s): {exc}"
                log.warning("%s; graded verdicts are already cached", stop_reason)
                break
            except SynthError as exc:
                n_failed += 1
                log.warning("judge batch failed (%s); continuing", exc)
                continue
            n_api += 1
            verdicts, unparsed = parse_judge_response(payload, batch)
            n_unparsed += unparsed
            self._memo.update(verdicts)
            if self.cache is not None and verdicts:
                self.cache.append(
                    [
                        {"pair_id": key, "grade": v.grade, "reason": v.reason}
                        for key, v in verdicts.items()
                    ]
                )

        return JudgeRun(
            verdicts=MappingProxyType(
                {p.pair_id: self._memo[p.pair_id] for p in ordered if p.pair_id in self._memo}
            ),
            n_api_calls=n_api,
            n_from_cache=n_cached,
            n_failed_batches=n_failed,
            n_unparsed_pairs=n_unparsed,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )


def parse_judge_response(
    payload: Mapping[str, Any],
    batch: Sequence[JudgePair],
) -> tuple[dict[str, JudgeVerdict], int]:
    """Read verdicts out of one reply, keeping only pair ids we asked about.

    Unknown pair ids are dropped rather than matched by position. A verdict
    attached to the wrong pair is a wrong label, and a wrong label here either
    removes a good negative or keeps a false one — both invisible afterwards.
    """
    asked = {p.pair_id for p in batch}
    out: dict[str, JudgeVerdict] = {}
    rows = payload.get("grades")
    if not isinstance(rows, list):
        raise SynthBadResponseError("judge reply had no 'grades' array")
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("pair_id", "")).strip()
        if key not in asked:
            continue
        try:
            grade = int(row["grade"])
        except (KeyError, TypeError, ValueError):
            continue
        if grade not in ORDINAL_GRADES:
            continue
        out[key] = JudgeVerdict(grade=grade, reason=str(row.get("reason", "")).strip(), pair_id=key)
    return out, len(asked - set(out))


# ─────────────────────────────────────────────────────────────────────────────
# Job 1: false-negative filtering
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RemovedNegative:
    query_id: str
    query: str
    business_id: str
    grade: int
    retrieval_rank: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class FalseNegativeReport:
    """How many mined negatives were actually relevant. A required README number."""

    n_examples: int
    n_negatives_in: int
    n_negatives_out: int
    n_graded: int
    n_removed: int
    n_examples_left_empty: int
    threshold_grade: int
    grade_counts: Mapping[int, int]
    removed: tuple[RemovedNegative, ...]
    judge_stopped_early: bool = False

    @property
    def removal_rate(self) -> float:
        """Share of *graded* negatives removed. Graded, not mined: dividing by the
        mined total would quietly dilute the rate with negatives nobody looked at."""
        return self.n_removed / self.n_graded if self.n_graded else 0.0

    def summary(self) -> str:
        line = (
            f"false-negative filter: graded {self.n_graded} of {self.n_negatives_in} mined "
            f"negatives across {self.n_examples} queries; removed {self.n_removed} "
            f"({self.removal_rate:.1%} of graded) at grade >= {self.threshold_grade}; "
            f"{self.n_negatives_out} negatives remain; "
            f"judge grade counts {dict(sorted(self.grade_counts.items()))}"
        )
        if self.n_examples_left_empty:
            line += f"; {self.n_examples_left_empty} query/queries left with no negatives"
        if self.judge_stopped_early:
            line += "; JUDGE STOPPED EARLY (rate limit) — this is a partial filter"
        return line


def build_judge_pairs(
    examples: Sequence[TrainingExample],
    documents: Mapping[str, str],
    *,
    top_n: int | None = DEFAULT_JUDGE_TOP_N,
) -> list[JudgePair]:
    """The negatives worth grading: the ``top_n`` hardest of each example.

    Hardest = best retrieval rank. Negatives with no document are skipped — the
    judge cannot grade text it does not have, and asking it to guess from an id
    would manufacture labels out of nothing.
    """
    pairs: list[JudgePair] = []
    for example in examples:
        ranked = sorted(example.negatives, key=lambda n: n.retrieval_rank)
        chosen = ranked if top_n is None else ranked[:top_n]
        for negative in chosen:
            document = documents.get(negative.business_id)
            if not document:
                continue
            pairs.append(
                JudgePair(
                    query_id=example.query_id,
                    query=example.query,
                    business_id=negative.business_id,
                    document=document,
                )
            )
    return pairs


def filter_false_negatives(
    examples: Sequence[TrainingExample],
    documents: Mapping[str, str],
    judge: LlmJudge,
    *,
    top_n: int | None = DEFAULT_JUDGE_TOP_N,
    threshold_grade: int = DEFAULT_FALSE_NEGATIVE_GRADE,
    confirm: ConfirmFn | None = None,
    assume_yes: bool = False,
    max_document_chars: int | None = None,
) -> tuple[list[TrainingExample], FalseNegativeReport]:
    """Remove mined negatives the judge says are actually relevant.

    Ungraded negatives are kept. That is the conservative direction: keeping a
    possibly-false negative costs some label noise, whereas dropping every
    negative we did not pay to grade would throw away most of the training signal
    on the strength of no evidence at all.
    """
    pairs = build_judge_pairs(examples, documents, top_n=top_n)
    run = judge.grade(
        pairs,
        job="false-negative filtering",
        confirm=confirm,
        assume_yes=assume_yes,
        max_document_chars=max_document_chars,
    )

    grade_counts: dict[int, int] = {g: 0 for g in ORDINAL_GRADES}
    removed: list[RemovedNegative] = []
    filtered: list[TrainingExample] = []
    n_in = n_out = n_empty = 0

    for example in examples:
        n_in += example.n_negatives
        drop: list[str] = []
        for negative in example.negatives:
            verdict = run.verdicts.get(pair_id(example.query_id, negative.business_id))
            if verdict is None:
                continue
            grade_counts[verdict.grade] = grade_counts.get(verdict.grade, 0) + 1
            if verdict.grade >= threshold_grade:
                drop.append(negative.business_id)
                removed.append(
                    RemovedNegative(
                        query_id=example.query_id,
                        query=example.query,
                        business_id=negative.business_id,
                        grade=verdict.grade,
                        retrieval_rank=negative.retrieval_rank,
                        reason=verdict.reason,
                    )
                )
        kept = example.without_negatives(drop) if drop else example
        n_out += kept.n_negatives
        if kept.n_negatives == 0:
            n_empty += 1
        filtered.append(kept)

    report = FalseNegativeReport(
        n_examples=len(examples),
        n_negatives_in=n_in,
        n_negatives_out=n_out,
        n_graded=sum(grade_counts.values()),
        n_removed=len(removed),
        n_examples_left_empty=n_empty,
        threshold_grade=threshold_grade,
        grade_counts=MappingProxyType(grade_counts),
        removed=tuple(removed),
        judge_stopped_early=run.stopped_early,
    )
    log.info("%s", report.summary())
    return filtered, report


# ─────────────────────────────────────────────────────────────────────────────
# Job 2: label-quality validation (READ-ONLY on the gold labels)
# ─────────────────────────────────────────────────────────────────────────────


class QrelsMutatedError(SynthError):
    """The qrels file changed while the judge was reading it.

    The tripwire described in the module docstring. If this is ever raised, the
    label-quality number must be discarded: it was computed against labels that no
    longer exist, and the more worrying possibility is that something in this
    process wrote them.
    """


@dataclass(frozen=True, slots=True)
class LabeledPair:
    """One owner-provided label. Immutable, like everything else from qrels."""

    query_id: str
    business_id: str
    grade: int


@dataclass(frozen=True, slots=True)
class GoldLabels:
    """An immutable, read-only view of ``eval/qrels.csv``.

    There is no ``save``, no ``put`` and no mutable field, by design: the judge
    receives *this object*, never a path it could open for writing. ``sha256``
    exists so a caller can prove the file was untouched (``verify_unchanged``).
    """

    path: Path
    sha256: str
    rows: tuple[LabeledPair, ...]

    @classmethod
    def load(cls, path: str | Path) -> GoldLabels:
        """Read qrels. Opened read-only; a missing or malformed grade is an error.

        A stray grade outside 0/1/2 is refused rather than clamped: it would mean
        the labeling CLI or the file is broken, and silently repairing it would
        change a reported kappa by an amount nobody could reconstruct.
        """
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise SynthError(
                f"qrels file not found: {p}. The owner labels pooled results with "
                "`make label`; label quality cannot be validated before that exists."
            )
        raw = p.read_bytes()
        rows: list[LabeledPair] = []
        for line_no, row in enumerate(csv.DictReader(raw.decode("utf-8").splitlines()), start=2):
            query_id = (row.get("query_id") or "").strip()
            business_id = (row.get("business_id") or "").strip()
            raw_grade = (row.get("grade") or "").strip()
            if not query_id or not business_id:
                continue
            try:
                grade = int(raw_grade)
            except ValueError as exc:
                raise SynthError(f"{p}:{line_no}: grade {raw_grade!r} is not an integer") from exc
            if grade not in ORDINAL_GRADES:
                raise SynthError(
                    f"{p}:{line_no}: grade {grade} outside the labeling scheme "
                    f"{list(ORDINAL_GRADES)}"
                )
            rows.append(LabeledPair(query_id=query_id, business_id=business_id, grade=grade))
        return cls(path=p, sha256=hashlib.sha256(raw).hexdigest(), rows=tuple(rows))

    def as_qrels(self) -> Mapping[str, Mapping[str, int]]:
        """``{query_id: {business_id: grade}}`` as read-only mappings.

        The shape ``eval.metrics`` expects. Wrapped in ``MappingProxyType`` so a
        caller that tried to add a label would raise instead of succeeding.
        """
        nested: dict[str, dict[str, int]] = {}
        for row in self.rows:
            nested.setdefault(row.query_id, {})[row.business_id] = row.grade
        return MappingProxyType({qid: MappingProxyType(v) for qid, v in nested.items()})

    def verify_unchanged(self) -> None:
        """Re-hash the file; raise if it differs from load time."""
        current = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if current != self.sha256:
            raise QrelsMutatedError(
                f"{self.path} changed while the judge was running "
                f"(was {self.sha256[:12]}, now {current[:12]}). "
                "Label-quality results from this run must be discarded."
            )


@dataclass(frozen=True, slots=True)
class LabelQualityReport:
    """Agreement between the owner's labels and the judge's on the same pairs.

    Both weightings are reported. Quadratic kappa is the flattering number on a
    3-point ordinal scale (it forgives 1-vs-2 near misses), so quoting it alone
    without saying it is weighted would overstate agreement; unweighted kappa is
    the strict reading. The confusion matrix is what actually tells you whether
    the judge is usable: confusing 1 and 2 is survivable for training data,
    calling 0s highly relevant is not.
    """

    n_sampled: int
    n_requested: int
    kappa_quadratic: KappaResult
    kappa_unweighted: KappaResult
    owner_grade_counts: Mapping[int, int]
    judge_grade_counts: Mapping[int, int]
    n_skipped_no_document: int
    n_skipped_no_query: int
    n_ungraded: int
    qrels_sha256: str
    judge_stopped_early: bool = False

    @property
    def matrix(self) -> tuple[tuple[int, ...], ...]:
        """Rows = owner grade, columns = judge grade, over categories 0/1/2."""
        return self.kappa_quadratic.matrix

    def summary(self) -> str:
        def fmt(result: KappaResult) -> str:
            return "undefined" if result.kappa is None else f"{result.kappa:.3f}"

        line = (
            f"label quality on {self.n_sampled} owner-labeled pairs "
            f"(requested {self.n_requested}): "
            f"kappa quadratic-weighted {fmt(self.kappa_quadratic)}, "
            f"unweighted {fmt(self.kappa_unweighted)}; "
            f"owner grades {dict(sorted(self.owner_grade_counts.items()))}, "
            f"judge grades {dict(sorted(self.judge_grade_counts.items()))}; "
            f"matrix (rows owner, cols judge) {self.matrix}"
        )
        if self.n_ungraded:
            line += f"; {self.n_ungraded} sampled pair(s) the judge did not grade"
        if self.judge_stopped_early:
            line += "; JUDGE STOPPED EARLY (rate limit) — sample is smaller than requested"
        return line


def sample_labeled_pairs(
    labels: GoldLabels,
    gold_queries: Mapping[str, GoldQuery] | Iterable[GoldQuery],
    documents: Mapping[str, str],
    *,
    n: int,
    seed: int,
) -> tuple[list[JudgePair], list[int], int, int]:
    """Pick a reproducible sample of owner-labeled pairs to re-grade.

    Returns the pairs, the owner's grades in the same order, and the two skip
    counts. Random (seeded) rather than "the first N" so the sample is an unbiased
    read on the whole label set: taking the first N would sample whatever order
    the labeling CLI happened to write, which correlates with query and with
    pooling depth.
    """
    by_id = (
        gold_queries if isinstance(gold_queries, Mapping) else {g.query_id: g for g in gold_queries}
    )
    eligible: list[tuple[LabeledPair, GoldQuery]] = []
    n_no_query = n_no_doc = 0
    # Sorted so the sample depends only on the seed, not on file order.
    for row in sorted(labels.rows, key=lambda r: (r.query_id, r.business_id)):
        query = by_id.get(row.query_id)
        if query is None:
            n_no_query += 1
            continue
        if not documents.get(row.business_id):
            n_no_doc += 1
            continue
        eligible.append((row, query))

    if n < len(eligible):
        eligible = random.Random(seed).sample(eligible, n)

    pairs = [
        JudgePair(
            query_id=row.query_id,
            query=query.query,
            business_id=row.business_id,
            document=documents[row.business_id],
        )
        for row, query in eligible
    ]
    owner_grades = [row.grade for row, _ in eligible]
    return pairs, owner_grades, n_no_doc, n_no_query


def validate_label_quality(
    labels: GoldLabels,
    gold_queries: Mapping[str, GoldQuery] | Iterable[GoldQuery],
    documents: Mapping[str, str],
    judge: LlmJudge,
    cfg: Config,
    *,
    sample_size: int | None = None,
    seed: int | None = None,
    confirm: ConfirmFn | None = None,
    assume_yes: bool = False,
    max_document_chars: int | None = None,
) -> LabelQualityReport:
    """Measure judge-vs-owner agreement on a sample of the owner's own labels.

    This is the number that licenses everything else the judge does. It is also
    the one place where the judge touches gold data, so it is the one place the
    read-only rule has to be enforced at runtime: the qrels file is re-hashed
    before this function returns and a change raises ``QrelsMutatedError``.

    Pairs the judge failed to grade are excluded from the kappa (there is nothing
    to compare) and counted in ``n_ungraded`` — never defaulted to 0, which would
    invent agreement or disagreement out of an API failure.
    """
    n = cfg.evaluation.label_subset_size if sample_size is None else sample_size
    pairs, owner_grades, n_no_doc, n_no_query = sample_labeled_pairs(
        labels,
        gold_queries,
        documents,
        n=n,
        seed=cfg.seed if seed is None else seed,
    )
    if not pairs:
        raise SynthError(
            "no owner-labeled pair is usable for validation: every label was for an "
            "unknown query or a business with no document"
        )

    run = judge.grade(
        pairs,
        job="label-quality validation",
        confirm=confirm,
        assume_yes=assume_yes,
        max_document_chars=max_document_chars,
    )

    owner: list[int] = []
    judged: list[int] = []
    for pair, grade in zip(pairs, owner_grades, strict=True):
        verdict = run.verdicts.get(pair.pair_id)
        if verdict is None:
            continue
        owner.append(grade)
        judged.append(verdict.grade)

    if not owner:
        raise SynthError("the judge graded none of the sampled pairs; no kappa to report")

    # categories pinned to 0/1/2 so the matrix has a fixed shape even when one
    # rater never used a grade — otherwise the matrix silently changes dimensions
    # between runs and cannot be compared or tabulated.
    quadratic = cohens_kappa(owner, judged, weights="quadratic", categories=ORDINAL_GRADES)
    unweighted = cohens_kappa(owner, judged, weights="none", categories=ORDINAL_GRADES)

    report = LabelQualityReport(
        n_sampled=len(owner),
        n_requested=n,
        kappa_quadratic=quadratic,
        kappa_unweighted=unweighted,
        owner_grade_counts=MappingProxyType({g: owner.count(g) for g in ORDINAL_GRADES}),
        judge_grade_counts=MappingProxyType({g: judged.count(g) for g in ORDINAL_GRADES}),
        n_skipped_no_document=n_no_doc,
        n_skipped_no_query=n_no_query,
        n_ungraded=len(pairs) - len(owner),
        qrels_sha256=labels.sha256,
        judge_stopped_early=run.stopped_early,
    )
    # The tripwire. Last thing before returning, so it covers everything above.
    labels.verify_unchanged()
    log.info("%s", report.summary())
    return report
