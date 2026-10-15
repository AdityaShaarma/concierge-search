"""`make eval` — the command that produces every number in the README.

This module computes; it does not render. ``eval/report.py`` turns the result
objects defined here into markdown, a plot and JSON. Keeping the split means a
number can be re-rendered without being recomputed, and a rendering bug can
never change a measurement.

Four required outputs (CLAUDE.md §6), all produced by one run:

1. the **ablation ladder** — BM25 → +dense (RRF) → +cross-encoder (pretrained)
   → +cross-encoder (fine-tuned) → +query parsing → +LambdaMART;
2. a **per-query-type breakdown** (vibe / constraint / specific / messy) of
   every row, because the stages help different kinds of query by very
   different amounts;
3. a **quality-vs-latency curve** over ``candidates.sweep_retrieve`` by
   ``candidates.sweep_rerank``;
4. a **failure analysis** that says, for the worst queries, whether the good
   result was never retrieved or was retrieved and then mis-ranked.

Three rules this file is built around
-------------------------------------

**No number without its denominator.** Every metric travels as a
:class:`MetricCell` carrying a bootstrap confidence interval, the number of
queries it is defined on, and the number excluded. A mean with no *n* is not a
result, so the shape of the data makes it impossible to report one.

**No number at all when the labels are not there.** If ``qrels.csv`` is missing
or too sparse to support a conclusion, this module raises
:class:`LabelsNotReady` and nothing is written. It never emits an example, a
simulated value or an illustrative table — a fabricated number that reached the
README would destroy the only thing this project is selling.

**No reimplementation of the cascade.** Every ablation row is the *real*
pipeline run with a different ``concierge.pipeline.Variant``. A row that
re-scored candidates locally would be measuring this file instead of the system.

Where the ablation is defined
-----------------------------

``concierge.pipeline.ABLATION`` is the list of rows, and it lives there rather
than here on purpose: the table must not be able to drift from the code that
produced it. This module takes that tuple as given — its order *is* the ladder —
and adds only what evaluation owns: the display label for each row, which row
each is compared against, and the measurement itself.

The pipeline is driven through its documented surface only:

.. code-block:: python

    pipeline = SearchPipeline.from_config(cfg, variant=..., ...)   # loads once
    response = pipeline.search(query, variant=variant, k=depth)    # per query

Cascade widths come from ``cfg.candidates``, so the quality-vs-latency sweep
varies them with a config copy and rebuilds the pipeline *around the already
loaded models* (see :class:`PipelineFactory`) rather than reloading a FAISS
index and a cross-encoder fifteen times.

One thing this module deliberately does not inherit from the pipeline: its
missing-model *degradation*. ``SearchPipeline`` falls back (fine-tuned →
pretrained → no reranker) so that it stays runnable mid-project, which is right
for serving and wrong for an ablation — a "+cross-encoder (fine-tuned)" row that
silently ran without the fine-tuned model would be a fabricated measurement. So
:class:`PipelineFactory` refuses to build a variant whose artifacts are absent,
and the row is reported as *not run*, with the reason.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from concierge.config import REPO_ROOT, Config, load_config
from concierge.pipeline import (
    ABLATION,
    VARIANTS,
    SearchPipeline,
    Stage,
    Variant,
)
from concierge.types import (
    RELEVANCE_HIGH,
    RELEVANCE_IRRELEVANT,
    RELEVANCE_PARTIAL,
    GoldQuery,
    ParsedQuery,
    QueryType,
    SearchResponse,
)
from eval import metrics

# ─────────────────────────────────────────────────────────────────────────────
# The ablation ladder: labels and comparisons for the pipeline's own variants
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CandidateDepths:
    """How wide each stage of the cascade runs for one measurement."""

    retrieve: int
    rerank: int
    final: int

    def applied_to(self, cfg: Config) -> Config:
        """A config copy whose cascade widths are these depths.

        ``SearchPipeline`` reads its widths from ``cfg.candidates``, so varying
        them is a config change rather than a call argument. Done as a copy
        (the config is frozen) so the caller's config is never mutated and the
        fingerprint of the run that produced a number stays truthful.
        """
        return cfg.model_copy(
            update={
                "candidates": cfg.candidates.model_copy(
                    update={
                        "retrieve": self.retrieve,
                        "rerank": self.rerank,
                        "final": self.final,
                    }
                )
            }
        )


def stage_names(variant: Variant) -> tuple[str, ...]:
    """The enabled stages of a variant, as readable names for the report.

    ``Stage`` is a flag enum, so its raw value is an integer; a JSON file or a
    markdown table holding ``27`` instead of ``BM25, DENSE, FILTERS`` would be
    unreadable to the one person who has to defend these rows.
    """
    return tuple(flag.name or "?" for flag in Stage if flag.value and flag in variant.stages)


@dataclass(frozen=True, slots=True)
class SystemSpec:
    """One row of the ablation: a pipeline variant plus how to present it.

    The variant comes from ``concierge.pipeline``; everything else here is
    evaluation's business.

    ``compare_to`` names the row this one is paired-bootstrapped against. For
    the ladder that is simply the previous row, so each row's confidence
    interval answers "did *this stage* help?" rather than "is the whole system
    better than BM25?", which is a much easier question that hides which stage
    did the work.

    ``supplementary`` rows sit outside the cumulative ladder. They answer side
    questions (what do the hard filters contribute? how does dense retrieval
    do alone?) and must never be read as further steps up the ladder.
    """

    key: str
    label: str
    variant: Variant
    compare_to: str | None = None
    supplementary: bool = False
    rationale: str = ""

    @property
    def stages(self) -> tuple[str, ...]:
        return stage_names(self.variant)

    @property
    def reranker(self) -> str | None:
        """Which cross-encoder checkpoint this row uses, or None if Stage 3 is off.

        Read from the variant only when ``Stage.RERANK`` is enabled: the
        pipeline leaves the field set but unused otherwise, and reporting
        "fine-tuned" on a row with no reranker would be a false label.
        """
        if not self.variant.enabled(Stage.RERANK):
            return None
        return self.variant.reranker.value


# How each pipeline variant is labelled in the README table, and why the row
# exists. Keyed by variant name so that `concierge.pipeline.ABLATION` stays the
# single definition of *which* rows there are and in what order — this dict only
# says how to present them. A variant with no entry here still appears, under
# its own name.
LADDER_PRESENTATION: dict[str, tuple[str, str]] = {
    "bm25": (
        "BM25",
        "Lexical retrieval with the rule-parsed constraints applied. The floor "
        "every later stage has to beat.",
    ),
    "hybrid_rrf": (
        "+dense (RRF)",
        "Adds dense retrieval, fused with BM25 by reciprocal rank fusion. RRF "
        "uses ranks only, so the two incomparable score scales never have to be "
        "normalised against each other (DECISIONS.md D010).",
    ),
    "rerank_pretrained": (
        "+cross-encoder (pretrained)",
        "The off-the-shelf ms-marco cross-encoder. Kept as its own row because "
        "it is what separates 'a reranker helps' from 'our fine-tuning helps'.",
    ),
    "rerank_finetuned": (
        "+cross-encoder (fine-tuned)",
        "The reranker we trained, against the identical pretrained setup — the "
        "only difference between these two rows is the checkpoint.",
    ),
    "llm_parse": (
        "+query parsing (LLM)",
        "Stage 1 switches from the keyword rules to the LLM parser. Both rows "
        "apply hard constraints; this measures the parser, not the filters, and "
        "it is the row where the LLM's latency cost shows up.",
    ),
    "full": (
        "+LambdaMART",
        "Stage 4 blends every upstream score with rating, popularity and "
        "constraint-match features.",
    ),
}

#: Supplementary variants, with the row each is compared against. These are
#: diagnostics the project needs (DECISIONS.md D006/D007 asks what the hard
#: filters are worth) but they are not rungs, so they are reported apart.
SUPPLEMENTARY_PRESENTATION: tuple[tuple[str, str, str, str], ...] = (
    (
        "no_filters",
        "full system without hard filters",
        "full",
        "How much of the quality is arithmetic constraint checking rather than "
        "similarity. If this row is close to the full system, the filters are not "
        "earning their complexity; if it is far below, the three-valued "
        "constraint logic is doing real work.",
    ),
    (
        "dense_only",
        "dense retrieval alone",
        "bm25",
        "The counterpart to the BM25 row: which half of the hybrid is stronger on "
        "its own, which is what makes the fusion row's gain interpretable.",
    ),
)


def ablation_specs(include_supplementary: bool = True) -> tuple[SystemSpec, ...]:
    """The ablation rows, in the pipeline's order.

    Order and membership come from ``concierge.pipeline.ABLATION``; this
    function only attaches presentation and sets each row's comparison baseline
    to the row above it. A variant the pipeline adds later shows up here
    automatically, which is the point of not restating the list.
    """
    specs: list[SystemSpec] = []
    previous: str | None = None
    for variant in ABLATION:
        label, rationale = LADDER_PRESENTATION.get(variant.name, (variant.name, ""))
        specs.append(
            SystemSpec(
                key=variant.name,
                label=label,
                variant=variant,
                compare_to=previous,
                rationale=rationale,
            )
        )
        previous = variant.name
    if include_supplementary:
        for name, label, compare_to, rationale in SUPPLEMENTARY_PRESENTATION:
            extra = VARIANTS.get(name)
            if extra is None:  # the pipeline dropped it; nothing to report
                continue
            specs.append(
                SystemSpec(
                    key=name,
                    label=label,
                    variant=extra,
                    compare_to=compare_to,
                    supplementary=True,
                    rationale=rationale,
                )
            )
    return tuple(specs)


#: The full system: the last rung of the ladder. The latency sweep uses it.
FULL_SYSTEM_KEY = ABLATION[-1].name


# ─────────────────────────────────────────────────────────────────────────────
# The searcher the ablation drives
# ─────────────────────────────────────────────────────────────────────────────


class Searcher(Protocol):
    """The one method this module needs: a variant-and-depth-bound search.

    ``SearchPipeline.search`` takes the variant and the depth as keyword
    arguments; binding them (see :class:`PipelineFactory`) keeps the timing loop
    free of configuration and lets a test inject a few canned result lists.
    """

    def search(self, query: str) -> SearchResponse: ...


#: Built per ablation row. Injected in tests so the whole harness is unit
#: testable with no indexes, no models and no dataset on disk.
SearcherFactory = Callable[[Config, Variant, CandidateDepths], Searcher]


class EvalError(RuntimeError):
    """Base class for the conditions that must stop a run rather than be
    papered over with a default value."""


class LabelsNotReady(EvalError):
    """Not enough hand-labeled relevance judgements to report anything."""


class StageUnavailable(EvalError):
    """A variant asks for a model or index that has not been built yet.

    Raised by :class:`PipelineFactory` rather than letting the pipeline's own
    graceful degradation run: for serving, falling back to a shorter cascade is
    the right behaviour; for an ablation row it would silently relabel one
    system as another.
    """


class NoSystemRan(EvalError):
    """Every ablation row failed to build, so there is nothing to report."""


@dataclass
class _BoundSearcher:
    """A pipeline with its variant and result depth already bound."""

    pipeline: SearchPipeline
    variant: Variant
    k: int

    def search(self, query: str) -> SearchResponse:
        return self.pipeline.search(query, variant=self.variant, k=self.k)


class PipelineFactory:
    """Builds a searcher per (variant, depths), loading the models once.

    Two jobs beyond calling ``SearchPipeline.from_config``:

    **Load once.** Twenty-two measurements (seven ablation rows plus fifteen
    sweep points) would otherwise mean twenty-two loads of the embedding model,
    the FAISS index and the cross-encoder. The first call loads; the rest reuse
    those objects through the public constructor with a different config, which
    is how the sweep varies cascade widths cheaply.

    **Refuse to fake a row.** ``SearchPipeline`` degrades when an artifact is
    missing (fine-tuned → pretrained → none). That is correct for a service and
    fatal for an ablation: the "+cross-encoder (fine-tuned)" row would run
    without the fine-tuned model and report a number that belongs to a
    different system. So the required artifacts are checked up front and the
    row is refused, which ``execute_system`` records as *not run* with the
    reason.
    """

    def __init__(
        self,
        *,
        load_models: bool = True,
        now: datetime | None = None,
        artifacts_root: Path | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        self.load_models = load_models
        self.now = now
        self.artifacts_root = artifacts_root
        self.processed_dir = processed_dir
        self._base: SearchPipeline | None = None
        self._base_error: Exception | None = None

    def base(self, cfg: Config) -> SearchPipeline:
        """The one fully-loaded pipeline, built on first use.

        A failure is cached as well as a success. Without that, a missing
        ``data/processed`` would be re-discovered — slowly — once per ablation
        row and once per sweep point, and the output would repeat the same
        traceback twenty-two times instead of saying it once per row.
        """
        if self._base_error is not None:
            raise self._base_error
        if self._base is None:
            try:
                self._base = SearchPipeline.from_config(
                    cfg,
                    load_models=self.load_models,
                    now=self.now,
                    artifacts_root=self.artifacts_root,
                    processed_dir=self.processed_dir,
                )
            except Exception as exc:
                self._base_error = exc
                raise
        return self._base

    @staticmethod
    def missing_artifacts(pipe: SearchPipeline, variant: Variant) -> list[str]:
        """What this variant needs and the loaded pipeline does not have."""
        missing: list[str] = []
        if variant.enabled(Stage.BM25) and pipe.bm25_index is None:
            missing.append("BM25 index (run `make index`)")
        if variant.enabled(Stage.DENSE) and pipe.dense_index is None:
            missing.append("dense/FAISS index (run `make index`)")
        if variant.enabled(Stage.RERANK) and pipe.rerankers.get(variant.reranker) is None:
            missing.append(
                f"{variant.reranker.value} cross-encoder "
                "(run `make train-reranker` for the fine-tuned checkpoint)"
            )
        if variant.enabled(Stage.LTR) and pipe.ltr_ranker is None:
            missing.append("LambdaMART model (run `make train-ltr`)")
        return missing

    def __call__(self, cfg: Config, variant: Variant, depths: CandidateDepths) -> Searcher:
        base = self.base(cfg)
        missing = self.missing_artifacts(base, variant)
        if missing:
            raise StageUnavailable(f"variant {variant.name!r} needs " + "; ".join(missing))
        # A new pipeline object around the same loaded models, with the widths
        # this measurement asks for. `search(k=...)` sets only the final depth;
        # retrieve and rerank come from the config.
        pipe = SearchPipeline(
            depths.applied_to(cfg),
            businesses=base.businesses,
            documents=base.doc_text,
            bm25_index=base.bm25_index,
            dense_index=base.dense_index,
            reranker=base.rerankers,
            ltr_ranker=base.ltr_ranker,
            variant=variant,
            now=self.now,
        )
        return _BoundSearcher(pipeline=pipe, variant=variant, k=depths.final)


# ─────────────────────────────────────────────────────────────────────────────
# Loading the owner's gold set and labels
# ─────────────────────────────────────────────────────────────────────────────

GOLD_COLUMNS = ("query_id", "query", "query_type")

# ``qrels.csv`` is written by the labeling CLI. The canonical header is
# query_id,business_id,grade; the aliases are accepted because a grade column
# is just as naturally called "relevance" or "label", and failing a 25-hour
# labeling effort over a column name would be absurd. Anything else is an
# error, not a guess.
QRELS_GRADE_ALIASES = ("grade", "relevance", "label", "rel")

#: Below this many labeled queries, the run refuses to report. Not a
#: statistical threshold — a floor below which a mean over queries says nothing
#: at all, and the right action is to label more rather than to publish a
#: number with a CI wider than the metric's range.
MIN_LABELED_QUERIES = 10


def _resolve(path: Path) -> Path:
    """Resolve a config path against the repo root, as ``Config`` paths are
    written relative to it."""
    return path if path.is_absolute() else REPO_ROOT / path


def load_gold_queries(path: Path) -> list[GoldQuery]:
    """Read ``gold_queries.csv``.

    Strict about ``query_type``: a typo there would silently create a fifth
    query type, quietly shrink one of the four reported breakdowns, and move a
    headline number. Fails with the row number instead.
    """
    p = _resolve(path)
    if not p.exists():
        raise LabelsNotReady(
            f"gold query file not found: {p}\n"
            "The owner writes these queries themselves (CLAUDE.md §6: never "
            "write or 'improve' them). Create the file with columns "
            f"{','.join(GOLD_COLUMNS)} and label a pool before running eval."
        )
    valid_types = {t.value for t in QueryType}
    out: list[GoldQuery] = []
    seen: set[str] = set()
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in GOLD_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise LabelsNotReady(
                f"{p} is missing column(s) {missing}; expected {list(GOLD_COLUMNS)}"
            )
        for lineno, row in enumerate(reader, start=2):
            qid = (row["query_id"] or "").strip()
            text = (row["query"] or "").strip()
            qtype = (row["query_type"] or "").strip().lower()
            if not qid and not text:
                continue  # blank line at the end of a hand-edited CSV
            if not qid or not text:
                raise LabelsNotReady(f"{p}:{lineno}: query_id and query must both be set")
            if qtype not in valid_types:
                raise LabelsNotReady(
                    f"{p}:{lineno}: query_type {qtype!r} is not one of {sorted(valid_types)}"
                )
            if qid in seen:
                raise LabelsNotReady(f"{p}:{lineno}: duplicate query_id {qid!r}")
            seen.add(qid)
            out.append(GoldQuery(query_id=qid, query=text, query_type=QueryType(qtype)))
    if not out:
        raise LabelsNotReady(f"{p} has no query rows")
    return out


#: The header the labeling CLI writes (``eval/pooling.QRELS_COLUMNS``). When a
#: file carries it, that module's reader is used, so there is exactly one
#: parser for the format its own writer produces.
POOLING_QRELS_COLUMNS = frozenset({"query_id", "business_id", "relevance", "labeled_at", "labeler"})

#: Grades the labeling scheme defines. A stray 3 or -1 would change every DCG
#: gain without any visible error, so it is rejected here rather than scored.
VALID_GRADES = frozenset({RELEVANCE_IRRELEVANT, RELEVANCE_PARTIAL, RELEVANCE_HIGH})


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read ``qrels.csv`` into ``{query_id: {business_id: grade}}``.

    Two readers, chosen by the header, and only ever one parser per file shape:

    * a file written by the labeling CLI carries that tool's full header, and
      ``eval.pooling.load_qrels`` reads it — the writer's own module owns the
      format, and a second interpretation of it here is exactly the kind of
      seam that produces two disagreeing numbers;
    * a minimal hand-written ``query_id,business_id,grade`` file is read below,
      because a labeled set assembled by hand (or in a test) is still valid
      input and should not require the CLI's bookkeeping columns.

    **Last row wins** for a repeated pair, matching the labeling CLI: that file
    is append-only, so a correction is a new row rather than an edit, and the
    most recent judgement is the owner's current opinion.
    """
    p = _resolve(path)
    if not p.exists():
        raise LabelsNotReady(
            f"relevance labels not found: {p}\n"
            "Nothing can be measured without them, and this command will not "
            "invent example numbers. Run `make label` to grade the pooled "
            "results for the gold queries first (CLAUDE.md §6: labeling is the "
            "owner's own work)."
        )
    with p.open(newline="", encoding="utf-8-sig") as fh:
        fields = set(csv.DictReader(fh).fieldnames or [])

    if fields >= POOLING_QRELS_COLUMNS:
        try:
            from eval import pooling  # deliberate lazy import: optional peer module
        except ImportError:
            pass  # fall through to the minimal reader below
        else:
            try:
                return pooling.load_qrels(p)
            except ValueError as exc:
                # A genuine data problem in a CLI-written file. Its message
                # already names the line and the rule; re-raised as the
                # not-ready error so `make eval` exits with the owner-action
                # exit code rather than a traceback.
                raise LabelsNotReady(str(exc)) from exc

    qrels: dict[str, dict[str, int]] = {}
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        grade_col = next((c for c in QRELS_GRADE_ALIASES if c in header), None)
        if "query_id" not in header or "business_id" not in header or grade_col is None:
            raise LabelsNotReady(
                f"{p} must have columns query_id, business_id and a grade column "
                f"(one of {list(QRELS_GRADE_ALIASES)}); found {header}"
            )
        for lineno, row in enumerate(reader, start=2):
            qid = (row["query_id"] or "").strip()
            bid = (row["business_id"] or "").strip()
            raw_grade = (row[grade_col] or "").strip()
            if not qid and not bid and not raw_grade:
                continue
            if not qid or not bid:
                raise LabelsNotReady(f"{p}:{lineno}: query_id and business_id must both be set")
            try:
                grade = int(raw_grade)
            except ValueError as exc:
                raise LabelsNotReady(
                    f"{p}:{lineno}: grade {raw_grade!r} is not an integer"
                ) from exc
            if grade not in VALID_GRADES:
                raise LabelsNotReady(
                    f"{p}:{lineno}: grade {grade} is outside the labeling scheme "
                    f"{sorted(VALID_GRADES)} (0 = irrelevant, 1 = partially relevant, "
                    "2 = highly relevant)"
                )
            qrels.setdefault(qid, {})[bid] = grade
    return qrels


@dataclass(frozen=True, slots=True)
class LabelCoverage:
    """How much of the gold set is actually judged.

    Printed on every table. "NDCG@10 = 0.61" means nothing without "over 48 of
    150 gold queries, 912 labeled pairs".

    The per-type maps exist because the per-query-type breakdown is its own
    table, and a table pasted into the README on its own has to carry its own
    denominators. "vibe NDCG@10 = 0.71" over six queries and eighteen labeled
    pairs is a different claim from the same number over sixty and nine hundred.
    """

    n_gold_queries: int
    n_queries_with_labels: int
    n_queries_with_relevant: int
    n_labeled_pairs: int
    n_positive_pairs: int
    unlabeled_query_ids: tuple[str, ...]
    orphan_query_ids: tuple[str, ...]  # labels for queries not in the gold set
    #: ``{query_type: count}`` for the queries in scope for this run.
    labeled_pairs_by_type: Mapping[str, int] = field(default_factory=dict)
    positive_pairs_by_type: Mapping[str, int] = field(default_factory=dict)

    @property
    def mean_pairs_per_labeled_query(self) -> float | None:
        if not self.n_queries_with_labels:
            return None
        return self.n_labeled_pairs / self.n_queries_with_labels


def summarize_labels(
    gold: Sequence[GoldQuery],
    qrels: Mapping[str, Mapping[str, int]],
    relevance_threshold: int = metrics.DEFAULT_RELEVANCE_THRESHOLD,
) -> LabelCoverage:
    """Count what has been judged, including the awkward parts."""
    gold_ids = [g.query_id for g in gold]
    gold_set = set(gold_ids)
    pairs_by_type: dict[str, int] = {t.value: 0 for t in QueryType}
    positives_by_type: dict[str, int] = {t.value: 0 for t in QueryType}
    for gq in gold:
        judged = qrels.get(gq.query_id, {})
        pairs_by_type[gq.query_type.value] += len(judged)
        positives_by_type[gq.query_type.value] += sum(
            1 for g in judged.values() if g >= relevance_threshold
        )
    with_labels = [qid for qid in gold_ids if qrels.get(qid)]
    with_relevant = [
        qid
        for qid in gold_ids
        if any(g >= relevance_threshold for g in qrels.get(qid, {}).values())
    ]
    n_pairs = sum(len(qrels.get(qid, {})) for qid in gold_ids)
    n_positive = sum(
        1 for qid in gold_ids for g in qrels.get(qid, {}).values() if g >= relevance_threshold
    )
    return LabelCoverage(
        n_gold_queries=len(gold_ids),
        n_queries_with_labels=len(with_labels),
        n_queries_with_relevant=len(with_relevant),
        n_labeled_pairs=n_pairs,
        n_positive_pairs=n_positive,
        unlabeled_query_ids=tuple(qid for qid in gold_ids if not qrels.get(qid)),
        orphan_query_ids=tuple(sorted(set(qrels) - gold_set)),
        labeled_pairs_by_type=pairs_by_type,
        positive_pairs_by_type=positives_by_type,
    )


def require_enough_labels(coverage: LabelCoverage, minimum: int = MIN_LABELED_QUERIES) -> None:
    """Stop the run when the labels cannot support a conclusion.

    The check is on queries with at least one *relevant* label, not on labeled
    pairs: a query whose whole pool was graded 0 has an undefined NDCG (see
    ``metrics.ndcg_at_k``) and contributes nothing to any mean, however many
    rows it added to the CSV.
    """
    if coverage.n_queries_with_relevant >= minimum:
        return
    raise LabelsNotReady(
        f"only {coverage.n_queries_with_relevant} of {coverage.n_gold_queries} gold "
        f"queries have at least one relevant label (need {minimum} before any metric "
        "is worth reporting).\n"
        f"Labeled pairs so far: {coverage.n_labeled_pairs} "
        f"({coverage.n_positive_pairs} relevant).\n"
        "Run `make label` and grade more of the pool. This command will not "
        "produce example, simulated or illustrative numbers to fill the gap."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Executing one system over the gold set
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    """One result as the eval saw it, kept for the failure analysis."""

    business_id: str
    name: str
    score: float
    constraints: tuple[str, ...]  # ConstraintMatch.explanation strings


@dataclass
class SystemRun:
    """Everything one toggle configuration produced over the whole gold set.

    Mutable (unlike the project's frozen contract types) because it is filled
    in query by query inside this module and never crosses a stage boundary.
    """

    key: str
    label: str
    variant: Variant
    depths: CandidateDepths
    ran: bool = False
    skip_reason: str = ""
    docs: dict[str, tuple[RetrievedDoc, ...]] = field(default_factory=dict)
    parsed: dict[str, ParsedQuery | None] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    cold_start_ms: float | None = None
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ranked(self) -> dict[str, list[str]]:
        """``{query_id: [business_id, ...]}`` — the run, in metrics' shape.

        Every gold query is a key even when the search failed, so that the
        paired bootstrap (which refuses mismatched query sets) always sees
        aligned systems. A failed query contributes an empty list, which scores
        0.0 rather than being quietly dropped: a system that cannot answer a
        query did not answer it.
        """
        return {qid: [d.business_id for d in docs] for qid, docs in self.docs.items()}

    def rank_of(self, query_id: str, business_id: str) -> int | None:
        """1-based rank of a business in this run, or ``None`` if absent."""
        for rank, doc in enumerate(self.docs.get(query_id, ()), start=1):
            if doc.business_id == business_id:
                return rank
        return None


def execute_system(
    spec: SystemSpec,
    gold: Sequence[GoldQuery],
    factory: SearcherFactory,
    cfg: Config,
    depths: CandidateDepths,
    exclude_warmup: bool = True,
) -> SystemRun:
    """Run every gold query through one toggle configuration.

    A row that cannot be built (a model not trained yet, an index missing) is
    recorded as *not run* with the reason, and the rest of the ablation
    continues. That is deliberate: ``make eval`` has to be useful in Phase 2
    when three of the six rows do not exist yet, and a half-table with stated
    gaps is honest where a crash is merely unhelpful.
    """
    run = SystemRun(key=spec.key, label=spec.label, variant=spec.variant, depths=depths)
    try:
        searcher = factory(cfg, spec.variant, depths)
    except Exception as exc:  # broad: the reason is the output, not a crash
        run.skip_reason = f"could not build: {type(exc).__name__}: {exc}"
        return run

    # The first successful search pays for model loading, lazy index mmap and a
    # cold page cache. Including it would put a one-off multi-second cost into
    # a latency distribution that is meant to describe steady-state serving, so
    # it is measured separately and reported as the cold start.
    first_timed = True
    for gq in gold:
        started = time.perf_counter()
        try:
            response = searcher.search(gq.query)
        except Exception as exc:  # broad: recorded per query, not hidden
            run.errors[gq.query_id] = f"{type(exc).__name__}: {exc}"
            run.docs[gq.query_id] = ()
            run.parsed[gq.query_id] = None
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if first_timed and exclude_warmup:
            run.cold_start_ms = elapsed_ms
            first_timed = False
        else:
            run.latency_ms[gq.query_id] = elapsed_ms
            first_timed = False
        run.docs[gq.query_id] = tuple(
            RetrievedDoc(
                business_id=r.business.business_id,
                name=r.business.name,
                score=r.score,
                constraints=tuple(c.explanation for c in r.constraints),
            )
            for r in response.results
        )
        run.parsed[gq.query_id] = response.parsed

    if run.errors and len(run.errors) == len(gold):
        # Not "a system that scores 0" — a system that did not run. Reporting
        # zeros here would put a real-looking row in the ablation table.
        run.skip_reason = (
            f"every one of {len(gold)} queries raised; first error: "
            f"{next(iter(run.errors.values()))}"
        )
        return run
    run.ran = True
    return run


# ─────────────────────────────────────────────────────────────────────────────
# Latency percentiles
# ─────────────────────────────────────────────────────────────────────────────

#: Below this many samples, p95 is not really a 95th percentile — it is the
#: largest one or two observations. Reported with a caveat rather than silently.
MIN_SAMPLES_FOR_P95 = 20


def percentile_nearest_rank(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest observation with at least ``q`` of
    the samples at or below it.

    Deliberately *not* the interpolated percentile ``metrics._percentile`` uses
    for bootstrap intervals. A latency percentile is an SLO statement — "95% of
    requests finished within this many milliseconds" — and must therefore be a
    value that was actually observed, not an average of two neighbours.
    """
    if not values:
        return None
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(values)
    index = math.ceil(q * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """End-to-end wall clock for one system, with its sample count.

    Measured around the ``search()`` call with ``perf_counter`` rather than
    read from ``SearchResponse.total_ms``: this is the number a caller
    experiences, and it does not depend on the pipeline's own bookkeeping being
    right.
    """

    n_samples: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    mean_ms: float | None
    cold_start_ms: float | None
    caveat: str = ""


def summarize_latency(run: SystemRun) -> LatencySummary:
    values = list(run.latency_ms.values())
    caveat = ""
    if 0 < len(values) < MIN_SAMPLES_FOR_P95:
        caveat = (
            f"p95 over only {len(values)} samples is the top observation, not a "
            "stable 95th percentile"
        )
    return LatencySummary(
        n_samples=len(values),
        p50_ms=percentile_nearest_rank(values, 0.50),
        p95_ms=percentile_nearest_rank(values, 0.95),
        max_ms=max(values) if values else None,
        mean_ms=sum(values) / len(values) if values else None,
        cold_start_ms=run.cold_start_ms,
        caveat=caveat,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric cells: a number, its interval, and its denominators
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MetricCell:
    """One metric for one system, with everything needed to quote it honestly.

    ``mean is None`` means the metric is undefined on this slice (no query has
    a relevant label), and ``note`` says so in words. There is no code path
    that turns that into a 0.0.
    """

    name: str
    mean: float | None
    low: float | None
    high: float | None
    n_defined: int
    n_undefined: int
    confidence: float
    n_resamples: int
    note: str = ""

    @property
    def n_total(self) -> int:
        return self.n_defined + self.n_undefined


def metric_cell(
    name: str,
    per_query_scores: Mapping[str, float | None],
    cfg: Config,
) -> MetricCell:
    """Aggregate per-query scores and bootstrap a CI around the mean."""
    agg = metrics.aggregate(per_query_scores.values())
    if agg.mean is None:
        return MetricCell(
            name=name,
            mean=None,
            low=None,
            high=None,
            n_defined=0,
            n_undefined=agg.n_undefined,
            confidence=cfg.evaluation.confidence,
            n_resamples=cfg.evaluation.bootstrap_samples,
            note=(
                f"undefined: none of the {agg.n_undefined} queries in this slice "
                "has a relevant label"
            ),
        )
    ci = metrics.bootstrap_ci_from_config(per_query_scores, cfg)
    note = ""
    if agg.n_defined == 1:
        # One observation gives a degenerate interval (every resample is the
        # same value). Saying so stops a zero-width CI being read as precision.
        note = "single query: the interval is degenerate, not narrow"
    return MetricCell(
        name=name,
        mean=ci.mean,
        low=ci.low,
        high=ci.high,
        n_defined=agg.n_defined,
        n_undefined=agg.n_undefined,
        confidence=ci.confidence,
        n_resamples=ci.n_resamples,
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Comparisons, and the words that must accompany them
# ─────────────────────────────────────────────────────────────────────────────

#: The exact phrase a CI crossing zero must produce. Named so the report and
#: the test that guards this behaviour cannot drift apart.
NOT_DISTINGUISHABLE_PHRASE = "not distinguishable from"

#: Below this many paired queries, no difference may be called distinguishable,
#: whatever the bootstrap interval says.
#:
#: The reason is arithmetic, not taste. Take the most favourable evidence that
#: can exist: system B beats system A on *every* one of n paired queries. The
#: two-sided sign test for that is ``2 * (1/2)**n = 1 / 2**(n-1)`` — n = 5
#: gives 1/16 = 0.0625, still above 0.05, and only n = 6 reaches
#: 1/32 = 0.031. Meanwhile a percentile bootstrap over those same five values
#: cheerfully returns an interval that excludes zero, because every resample of
#: a uniformly positive set is positive. The bootstrap has no way to say "there
#: were barely any observations", so the floor is imposed here. It matters most
#: in the per-query-type breakdown, where one type can hold a handful of
#: queries and a zero-width interval would otherwise read as certainty.
MIN_PAIRS_FOR_A_CLAIM = 6


@dataclass(frozen=True, slots=True)
class Comparison:
    """A paired-bootstrap difference against another system, already in words.

    ``sentence`` is part of the data, not a rendering detail. The requirement
    is that an interval crossing zero *says so in words* wherever the number
    appears — markdown, JSON, or a terminal — so the sentence is built once,
    here, next to the arithmetic that decides it.
    """

    previous_key: str
    previous_label: str
    metric: str
    computed: bool
    mean_diff: float | None
    low: float | None
    high: float | None
    p_value: float | None
    confidence: float
    n_queries: int
    n_dropped: int
    n_resamples: int
    distinguishable: bool
    sentence: str


def _format_p_value(p_value: float, n_resamples: int) -> str:
    """Format a bootstrap p-value, refusing to overstate its resolution.

    ``n`` resamples cannot distinguish "very unlikely" from "impossible", so
    once the estimate bottoms out it is reported as a bound (see the note in
    ``metrics.paired_bootstrap``).
    """
    floor = 2.0 / (n_resamples + 1)
    if p_value <= floor * 1.000001:
        return f"p <= {floor:.4f} (bootstrap resolution limit)"
    return f"p = {p_value:.4f}"


def compare_systems(
    scores_a: Mapping[str, float | None],
    scores_b: Mapping[str, float | None],
    previous_key: str,
    previous_label: str,
    metric: str,
    cfg: Config,
) -> Comparison:
    """Paired bootstrap of B against A, with the hedging decided here.

    Paired because both systems ran the same queries: pairing cancels query
    difficulty, which is a confound and not a signal (see
    ``metrics.paired_bootstrap``).
    """
    try:
        result = metrics.paired_bootstrap_from_config(scores_a, scores_b, cfg)
    except ValueError as exc:
        # Raised when the two systems have different query sets, or when no
        # query has a defined score on both sides. Either way there is no
        # comparison to report, and the reason belongs in the output.
        return Comparison(
            previous_key=previous_key,
            previous_label=previous_label,
            metric=metric,
            computed=False,
            mean_diff=None,
            low=None,
            high=None,
            p_value=None,
            confidence=cfg.evaluation.confidence,
            n_queries=0,
            n_dropped=0,
            n_resamples=cfg.evaluation.bootstrap_samples,
            distinguishable=False,
            sentence=f"no paired comparison against {previous_label}: {exc}",
        )

    pct = f"{result.confidence:.0%}"
    queries = "query" if result.n_queries == 1 else "queries"
    head = (
        f"{metric} {result.mean_diff:+.4f} vs {previous_label} "
        f"({pct} CI {result.low:+.4f} to {result.high:+.4f}, "
        f"{_format_p_value(result.p_value, result.n_resamples)}, "
        f"n = {result.n_queries} paired {queries}"
    )
    if result.n_dropped:
        head += f", {result.n_dropped} dropped as undefined on one side"
    head += ")"

    too_few = result.n_queries < MIN_PAIRS_FOR_A_CLAIM
    # A degenerate interval is worth flagging but is not itself a reason to
    # hedge: if every paired difference really is identical, the bootstrap's
    # zero width is the honest answer to "how much would this move if I had
    # drawn different queries?" given those observations.
    degenerate = result.low == result.high
    if result.significant and not too_few:
        direction = "better than" if result.mean_diff > 0 else "worse than"
        sentence = (
            f"{head} — distinguishable: the interval excludes zero, so this stage "
            f"is measurably {direction} {previous_label} on this query set."
        )
        if degenerate:
            sentence += (
                " Note that the interval has zero width because every paired "
                "difference is identical; it describes these queries exactly and "
                "says nothing about variability between them."
            )
    elif too_few:
        sentence = (
            f"{head} — {NOT_DISTINGUISHABLE_PHRASE} {previous_label}: only "
            f"{result.n_queries} paired {queries} "
            f"{'carries' if result.n_queries == 1 else 'carry'} a defined score on "
            f"both sides, below the {MIN_PAIRS_FOR_A_CLAIM} "
            "needed for any interval here to support a claim either way (even a "
            "clean sweep of 5 queries is only two-sided p = 0.06 by a sign test). "
            "Label more queries of this kind before reading this difference."
        )
    else:
        sentence = (
            f"{head} — {NOT_DISTINGUISHABLE_PHRASE} {previous_label} at this sample "
            f"size: the {pct} interval includes zero, so the observed difference is "
            "consistent with noise and must not be reported as an improvement."
        )
    return Comparison(
        previous_key=previous_key,
        previous_label=previous_label,
        metric=metric,
        computed=True,
        mean_diff=result.mean_diff,
        low=result.low,
        high=result.high,
        p_value=result.p_value,
        confidence=result.confidence,
        n_queries=result.n_queries,
        n_dropped=result.n_dropped,
        n_resamples=result.n_resamples,
        # Both conditions: the interval must exclude zero *and* there must be
        # enough paired queries for that to mean anything.
        distinguishable=result.significant and not too_few,
        sentence=sentence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result objects the report renders
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TypeBreakdown:
    """One system's metrics restricted to one query type."""

    query_type: str
    n_queries: int
    ndcg: MetricCell
    recall: MetricCell
    mrr: MetricCell
    unjudged_fraction: float | None
    vs_previous: Comparison | None


@dataclass(frozen=True, slots=True)
class AblationRow:
    """One row of the ablation table, or a stated gap where a row could not run."""

    key: str
    label: str
    variant_name: str
    stages: tuple[str, ...]
    reranker: str | None
    rationale: str
    supplementary: bool
    ran: bool
    skip_reason: str
    n_queries_attempted: int
    n_query_errors: int
    ndcg: MetricCell | None
    recall: MetricCell | None
    mrr: MetricCell | None
    latency: LatencySummary | None
    unjudged_fraction: float | None
    n_unjudged: int
    n_top_k_results: int
    by_type: tuple[TypeBreakdown, ...]
    vs_previous: Comparison | None


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One (retrieve, rerank) operating point on the quality-vs-latency curve."""

    retrieve_k: int
    rerank_k: int
    final_k: int
    ran: bool
    skip_reason: str
    ndcg: MetricCell | None
    latency: LatencySummary | None


@dataclass(frozen=True, slots=True)
class RelevantDocTrace:
    """Where one known-relevant business landed in every system's ranking.

    This is the whole diagnostic value of the failure analysis: a relevant
    place absent everywhere was never retrieved (a Stage 2 or filter problem),
    while one sitting at rank 40 was retrieved and then mis-ranked (a Stage 3/4
    problem). The two call for completely different fixes.
    """

    business_id: str
    name: str
    grade: int
    rank_by_system: Mapping[str, int | None]


@dataclass(frozen=True, slots=True)
class FailureCase:
    """One badly-answered query, with everything needed to see why."""

    query_id: str
    query: str
    query_type: str
    system_key: str
    ndcg: float | None
    parse_source: str | None
    parse_latency_ms: float | None
    parsed_summary: str
    top_results: tuple[tuple[int, str, str, int, bool], ...]
    # (rank, business_id, name, grade, judged)
    relevant_traces: tuple[RelevantDocTrace, ...]
    lost_stage: str
    diagnosis: str


@dataclass(frozen=True, slots=True)
class EvalResults:
    """Everything one ``make eval`` produced. Rendered by ``eval/report.py``."""

    generated_at: str
    config_path: str
    config_fingerprint: str
    ndcg_k: int
    recall_k: int
    relevance_threshold: int
    eval_depth: int
    operating_depths: CandidateDepths
    coverage: LabelCoverage
    rows: tuple[AblationRow, ...]
    sweep: tuple[SweepPoint, ...]
    sweep_note: str
    failures: tuple[FailureCase, ...]
    failure_reference_system: str
    query_type_counts: Mapping[str, int]
    partial_run_note: str
    warnings: tuple[str, ...]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def ndcg_k_from_config(cfg: Config) -> int:
    """The headline cutoff, taken from config rather than hardcoded.

    ``primary_metric`` is the thing the README quotes, so parsing it (instead
    of writing ``10`` in this file) means the config cannot drift away from the
    number that gets reported.
    """
    raw = cfg.evaluation.primary_metric.strip().lower()
    prefix = "ndcg@"
    if not raw.startswith(prefix):
        raise EvalError(
            f"evaluation.primary_metric is {cfg.evaluation.primary_metric!r}; this "
            "harness reports NDCG@k as the headline metric and needs it in the "
            "form 'ndcg@10'"
        )
    try:
        k = int(raw[len(prefix) :])
    except ValueError as exc:
        raise EvalError(
            f"cannot read a cutoff from primary_metric {cfg.evaluation.primary_metric!r}"
        ) from exc
    if k not in tuple(cfg.evaluation.cutoffs):
        raise EvalError(
            f"primary_metric is ndcg@{k} but evaluation.cutoffs is "
            f"{list(cfg.evaluation.cutoffs)}; the headline cutoff must be one of them"
        )
    return k


def eval_depth_for(cfg: Config) -> int:
    """How deep a ranked list each system must return for scoring.

    Recall@100 cannot be computed from a top-10 response, so during evaluation
    the final stage is asked for ``recall_cutoff`` results. The consequence is
    stated in the report: in the ablation table the reranked depth is at least
    the recall cutoff, so that table's latency column describes producing a
    depth-100 list. The production operating point (200/50/10) is measured in
    the quality-vs-latency sweep instead.
    """
    return max(
        cfg.evaluation.recall_cutoff,
        cfg.evaluation.pool_depth,
        ndcg_k_from_config(cfg),
        cfg.candidates.final,
    )


def depths_for_eval(cfg: Config) -> CandidateDepths:
    """Candidate depths used for the ablation rows.

    Each stage is widened only as far as the eval depth forces, so the system
    under test stays as close to its configured shape as scoring allows.
    """
    depth = eval_depth_for(cfg)
    return CandidateDepths(
        retrieve=max(cfg.candidates.retrieve, depth),
        rerank=max(cfg.candidates.rerank, depth),
        final=depth,
    )


def _unjudged_fraction(
    ranked: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    query_ids: Iterable[str],
    k: int,
) -> float | None:
    """Unjudged share of the top-k over a subset of queries."""
    ids = set(query_ids)
    sub = {qid: r for qid, r in ranked.items() if qid in ids}
    if not sub:
        return None
    return metrics.report_unjudged(sub, qrels, k).fraction


def score_rows(
    specs: Sequence[SystemSpec],
    runs: Mapping[str, SystemRun],
    gold: Sequence[GoldQuery],
    qrels: Mapping[str, Mapping[str, int]],
    cfg: Config,
) -> list[AblationRow]:
    """Turn executed runs into ablation rows with CIs, breakdowns and hedging."""
    ndcg_k = ndcg_k_from_config(cfg)
    recall_k = cfg.evaluation.recall_cutoff
    threshold = metrics.DEFAULT_RELEVANCE_THRESHOLD

    ids_by_type: dict[str, list[str]] = {t.value: [] for t in QueryType}
    for gq in gold:
        ids_by_type[gq.query_type.value].append(gq.query_id)

    reports: dict[str, metrics.RunReport] = {}
    rows: list[AblationRow] = []

    for spec in specs:
        run = runs[spec.key]
        if not run.ran:
            rows.append(
                AblationRow(
                    key=spec.key,
                    label=spec.label,
                    variant_name=spec.variant.name,
                    stages=spec.stages,
                    reranker=spec.reranker,
                    rationale=spec.rationale,
                    supplementary=spec.supplementary,
                    ran=False,
                    skip_reason=run.skip_reason or "not run",
                    n_queries_attempted=len(gold),
                    n_query_errors=len(run.errors),
                    ndcg=None,
                    recall=None,
                    mrr=None,
                    latency=None,
                    unjudged_fraction=None,
                    n_unjudged=0,
                    n_top_k_results=0,
                    by_type=(),
                    vs_previous=None,
                )
            )
            continue

        ranked = run.ranked
        report = metrics.evaluate_run(
            ranked, qrels, ndcg_k=ndcg_k, recall_k=recall_k, relevance_threshold=threshold
        )
        reports[spec.key] = report

        # The comparison baseline: the nearest earlier row that actually ran.
        # Comparing against a row that never executed would be meaningless, and
        # silently comparing against two rows back would mis-attribute the gain
        # — so the previous label says which row it really is.
        vs_previous: Comparison | None = None
        if spec.compare_to and spec.compare_to in reports:
            prev_spec = next(s for s in specs if s.key == spec.compare_to)
            vs_previous = compare_systems(
                reports[spec.compare_to].ndcg_per_query,
                report.ndcg_per_query,
                previous_key=prev_spec.key,
                previous_label=prev_spec.label,
                metric=f"NDCG@{ndcg_k}",
                cfg=cfg,
            )
        elif spec.compare_to:
            vs_previous = Comparison(
                previous_key=spec.compare_to,
                previous_label=spec.compare_to,
                metric=f"NDCG@{ndcg_k}",
                computed=False,
                mean_diff=None,
                low=None,
                high=None,
                p_value=None,
                confidence=cfg.evaluation.confidence,
                n_queries=0,
                n_dropped=0,
                n_resamples=cfg.evaluation.bootstrap_samples,
                distinguishable=False,
                sentence=(
                    f"no comparison: the baseline row '{spec.compare_to}' did not run, "
                    "so this row's contribution is unattributed"
                ),
            )

        by_type: list[TypeBreakdown] = []
        for qtype in QueryType:
            type_ids = ids_by_type[qtype.value]
            if not type_ids:
                continue
            type_prev: Comparison | None = None
            if spec.compare_to and spec.compare_to in reports:
                prev_spec = next(s for s in specs if s.key == spec.compare_to)
                type_prev = compare_systems(
                    metrics.subset(reports[spec.compare_to].ndcg_per_query, type_ids),
                    metrics.subset(report.ndcg_per_query, type_ids),
                    previous_key=prev_spec.key,
                    previous_label=prev_spec.label,
                    metric=f"NDCG@{ndcg_k} ({qtype.value})",
                    cfg=cfg,
                )
            by_type.append(
                TypeBreakdown(
                    query_type=qtype.value,
                    n_queries=len(type_ids),
                    ndcg=metric_cell(
                        f"NDCG@{ndcg_k}",
                        metrics.subset(report.ndcg_per_query, type_ids),
                        cfg,
                    ),
                    recall=metric_cell(
                        f"Recall@{recall_k}",
                        metrics.subset(report.recall_per_query, type_ids),
                        cfg,
                    ),
                    mrr=metric_cell("MRR", metrics.subset(report.rr_per_query, type_ids), cfg),
                    unjudged_fraction=_unjudged_fraction(ranked, qrels, type_ids, ndcg_k),
                    vs_previous=type_prev,
                )
            )

        rows.append(
            AblationRow(
                key=spec.key,
                label=spec.label,
                variant_name=spec.variant.name,
                stages=spec.stages,
                reranker=spec.reranker,
                rationale=spec.rationale,
                supplementary=spec.supplementary,
                ran=True,
                skip_reason="",
                n_queries_attempted=len(gold),
                n_query_errors=len(run.errors),
                ndcg=metric_cell(f"NDCG@{ndcg_k}", report.ndcg_per_query, cfg),
                recall=metric_cell(f"Recall@{recall_k}", report.recall_per_query, cfg),
                mrr=metric_cell("MRR", report.rr_per_query, cfg),
                latency=summarize_latency(run),
                unjudged_fraction=report.unjudged.fraction,
                n_unjudged=report.unjudged.n_unjudged,
                n_top_k_results=report.unjudged.n_results,
                by_type=tuple(by_type),
                vs_previous=vs_previous,
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Quality vs latency sweep
# ─────────────────────────────────────────────────────────────────────────────


def sweep_quality_latency(
    gold: Sequence[GoldQuery],
    qrels: Mapping[str, Mapping[str, int]],
    factory: SearcherFactory,
    cfg: Config,
    spec: SystemSpec | None = None,
) -> tuple[list[SweepPoint], str]:
    """NDCG@10 against p95 latency across the configured candidate counts.

    Run at the *production* final depth (``candidates.final``), not the eval
    depth, because this curve is the one that answers "what should we actually
    serve?" — so its latency numbers have to be the ones a user would pay.
    NDCG@10 only: Recall@100 is not computable from a top-10 list, and asking
    for a deeper list would change the very latency being measured.

    Combinations where ``rerank > retrieve`` are skipped: a cascade that widens
    at a later stage is a bug, not an operating point (``CandidatesConfig``
    rejects it too).
    """
    target = spec or next(s for s in ablation_specs() if s.key == FULL_SYSTEM_KEY)
    ndcg_k = ndcg_k_from_config(cfg)
    final_k = max(cfg.candidates.final, ndcg_k)
    points: list[SweepPoint] = []
    for retrieve_k in cfg.candidates.sweep_retrieve:
        for rerank_k in cfg.candidates.sweep_rerank:
            if rerank_k > retrieve_k or final_k > rerank_k:
                continue
            depths = CandidateDepths(retrieve=retrieve_k, rerank=rerank_k, final=final_k)
            run = execute_system(target, gold, factory, cfg, depths)
            if not run.ran:
                points.append(
                    SweepPoint(
                        retrieve_k=retrieve_k,
                        rerank_k=rerank_k,
                        final_k=final_k,
                        ran=False,
                        skip_reason=run.skip_reason or "not run",
                        ndcg=None,
                        latency=None,
                    )
                )
                continue
            report = metrics.evaluate_run(run.ranked, qrels, ndcg_k=ndcg_k)
            points.append(
                SweepPoint(
                    retrieve_k=retrieve_k,
                    rerank_k=rerank_k,
                    final_k=final_k,
                    ran=True,
                    skip_reason="",
                    ndcg=metric_cell(f"NDCG@{ndcg_k}", report.ndcg_per_query, cfg),
                    latency=summarize_latency(run),
                )
            )
    note = (
        f"Swept system: {target.label}. Each point returns {final_k} results "
        f"(the production final depth), so NDCG@{ndcg_k} and latency are both "
        "measured at the operating point a user would actually hit. "
        f"{len(points)} of {len(cfg.candidates.sweep_retrieve) * len(cfg.candidates.sweep_rerank)} "
        "grid combinations are valid (the rest would widen the cascade)."
    )
    return points, note


# ─────────────────────────────────────────────────────────────────────────────
# Failure analysis
# ─────────────────────────────────────────────────────────────────────────────

#: Worst queries shown per query type. Three is enough to see a pattern and few
#: enough that the owner actually reads them.
FAILURES_PER_TYPE = 3


def _summarize_parse(parsed: ParsedQuery | None) -> str:
    """One line describing what Stage 1 extracted, for the failure write-up."""
    if parsed is None:
        return "no parse recorded (the search itself failed)"
    parts: list[str] = [f"semantic_text={parsed.semantic_text!r}"]
    if parsed.open_at is not None:
        hour, minute = divmod(parsed.open_at.minute_of_day, 60)
        parts.append(f"open_at={parsed.open_at.day} {hour:02d}:{minute:02d}")
    if parsed.attributes:
        parts.append(f"attributes={parsed.attributes}")
    if parsed.price_max is not None:
        parts.append(f"price_max={parsed.price_max}")
    if parsed.categories:
        parts.append(f"categories={list(parsed.categories)}")
    if parsed.geo is not None:
        parts.append(f"geo={parsed.geo.label or 'point'} r={parsed.geo.radius_km}km")
    parts.append(f"source={parsed.source.value}")
    return "; ".join(parts)


def diagnose_query(
    query_id: str,
    relevant: Mapping[str, int],
    runs_in_order: Sequence[SystemRun],
    reference_key: str,
    ndcg_k: int,
    depth: int,
    ndcg: float | None = None,
) -> tuple[str, str, tuple[RelevantDocTrace, ...]]:
    """Decide *which stage lost the good result*, and say it in words.

    Returns ``(lost_stage, diagnosis, traces)``. The distinction the whole
    failure analysis exists for:

    * absent from every system's candidate list → **never retrieved**. Stage 2
      or a hard filter dropped it, and no reranker can recover it.
    * present in a list but never in the top k → **retrieved, never ranked up**.
      A ranking problem; the candidate pool was fine.
    * in the top k under an earlier system and out of it under a later one →
      **mis-ranked by a specific stage**, named.

    ``ndcg`` is the reference system's score for this query. It is passed in so
    that a *perfect* ranking is never described as a failure: the worst query
    of a type can still be a good one, and inventing a fault for it would be
    exactly as dishonest as inventing a number.
    """
    traces: list[RelevantDocTrace] = []
    name_by_id: dict[str, str] = {}
    for run in runs_in_order:
        for doc in run.docs.get(query_id, ()):
            name_by_id.setdefault(doc.business_id, doc.name)
    for bid, grade in sorted(relevant.items(), key=lambda kv: (-kv[1], kv[0])):
        traces.append(
            RelevantDocTrace(
                business_id=bid,
                name=name_by_id.get(bid, "(never returned by any system)"),
                grade=grade,
                rank_by_system={run.key: run.rank_of(query_id, bid) for run in runs_in_order},
            )
        )

    def any_in_pool(run: SystemRun) -> bool:
        return any(t.rank_by_system[run.key] is not None for t in traces)

    def any_in_top_k(run: SystemRun) -> bool:
        return any(
            (rank := t.rank_by_system[run.key]) is not None and rank <= ndcg_k for t in traces
        )

    if ndcg is not None and ndcg >= 1.0:
        # NDCG@k of 1.0 means the top k is already the ideal ordering of the
        # labels. This query is only here because it is the lowest-scoring one
        # of its type, which is a statement about the type, not a fault.
        return (
            "none",
            (
                f"Nothing to diagnose: NDCG@{ndcg_k} is 1.0, so the top {ndcg_k} is "
                "already the best ordering the labels allow. This query is listed "
                "because it is the lowest-scoring one of its type — in the labeled "
                "set, this type has no failure to show."
            ),
            tuple(traces),
        )

    if not any(any_in_pool(run) for run in runs_in_order):
        return (
            "retrieval",
            (
                f"Never retrieved: none of the {len(traces)} labeled-relevant places "
                f"appears anywhere in any system's depth-{depth} candidate list. No "
                "amount of reranking could fix this query — it is a retrieval or "
                "hard-filter failure (wrong vocabulary, an over-strict filter, or the "
                "place is genuinely not indexed)."
            ),
            tuple(traces),
        )

    reference = next(run for run in runs_in_order if run.key == reference_key)
    if any_in_top_k(reference):
        best = min(
            (t for t in traces if t.rank_by_system[reference_key] is not None),
            key=lambda t: t.rank_by_system[reference_key] or 0,
        )
        return (
            "ordering",
            (
                f"Mis-ordered inside the top {ndcg_k}: a relevant place "
                f"({best.name}, grade {best.grade}) is at rank "
                f"{best.rank_by_system[reference_key]}, so retrieval and the shortlist "
                "both did their job. The loss is in the order of the top "
                f"{ndcg_k} — lower-graded or unjudged results are ranked above "
                "higher-graded ones — which is Stage 3/4 work, or a labeling gap if "
                "the results above it are unjudged."
            ),
            tuple(traces),
        )

    # In a pool somewhere, but not in the reference system's top k. Find
    # whether an earlier stage had it up there and a later one pushed it out.
    had_it = [run.key for run in runs_in_order if any_in_top_k(run)]
    if had_it:
        last_good = had_it[-1]
        order = [run.key for run in runs_in_order]
        dropped_at = order[order.index(last_good) + 1] if last_good != order[-1] else last_good
        best = max(
            (t for t in traces if t.rank_by_system[last_good] is not None),
            key=lambda t: (t.grade, -(t.rank_by_system[last_good] or 0)),
        )
        after = best.rank_by_system[dropped_at]
        where_now = f"rank {after}" if after is not None else "gone from the list entirely"
        return (
            dropped_at,
            (
                f"Retrieved and then mis-ranked: {best.name} (grade {best.grade}) was at "
                f"rank {best.rank_by_system[last_good]} under '{last_good}' and is "
                f"{where_now} under '{dropped_at}', which pushed it out of the top "
                f"{ndcg_k}. The candidate pool was fine; a ranking stage lost it."
            ),
            tuple(traces),
        )

    best = max(
        (t for t in traces if any(r is not None for r in t.rank_by_system.values())),
        key=lambda t: t.grade,
    )
    best_rank = min(r for r in best.rank_by_system.values() if r is not None)
    return (
        "ranking",
        (
            f"Retrieved but never ranked up: {best.name} (grade {best.grade}) reaches "
            f"rank {best_rank} at best and no system ever put a relevant place in the "
            f"top {ndcg_k}. The candidate pool contains the answer, so this is a "
            "ranking failure, not a retrieval failure."
        ),
        tuple(traces),
    )


def build_failure_analysis(
    gold: Sequence[GoldQuery],
    qrels: Mapping[str, Mapping[str, int]],
    runs_in_order: Sequence[SystemRun],
    cfg: Config,
    reference_key: str,
    per_type: int = FAILURES_PER_TYPE,
) -> list[FailureCase]:
    """The worst queries per type under the reference system, fully unpacked.

    "Worst" is by NDCG@k of the reference system, and only over queries where
    NDCG is *defined*: a query with no relevant label has no NDCG to be worst
    at, and including it would fill the analysis with unlabeled queries instead
    of real failures.
    """
    ndcg_k = ndcg_k_from_config(cfg)
    threshold = metrics.DEFAULT_RELEVANCE_THRESHOLD
    reference = next((r for r in runs_in_order if r.key == reference_key), None)
    if reference is None or not reference.ran:
        return []
    depth = reference.depths.final

    per_query_ndcg = metrics.per_query(
        reference.ranked, qrels, lambda r, j: metrics.ndcg_at_k(r, j, ndcg_k)
    )
    gold_by_id = {g.query_id: g for g in gold}

    cases: list[FailureCase] = []
    for qtype in QueryType:
        scored = [
            (score, qid)
            for qid, score in per_query_ndcg.items()
            if score is not None and qid in gold_by_id and gold_by_id[qid].query_type is qtype
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        for score, qid in scored[:per_type]:
            judged = qrels.get(qid, {})
            relevant = {bid: g for bid, g in judged.items() if g >= threshold}
            lost_stage, diagnosis, traces = diagnose_query(
                qid, relevant, runs_in_order, reference_key, ndcg_k, depth, ndcg=score
            )
            parsed = reference.parsed.get(qid)
            cases.append(
                FailureCase(
                    query_id=qid,
                    query=gold_by_id[qid].query,
                    query_type=qtype.value,
                    system_key=reference_key,
                    ndcg=score,
                    parse_source=parsed.source.value if parsed else None,
                    parse_latency_ms=parsed.parse_latency_ms if parsed else None,
                    parsed_summary=_summarize_parse(parsed),
                    top_results=tuple(
                        (
                            rank,
                            doc.business_id,
                            doc.name,
                            judged.get(doc.business_id, 0),
                            doc.business_id in judged,
                        )
                        for rank, doc in enumerate(reference.docs.get(qid, ())[:ndcg_k], start=1)
                    ),
                    relevant_traces=traces,
                    lost_stage=lost_stage,
                    diagnosis=diagnosis,
                )
            )
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# The whole run
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(
    cfg: Config,
    config_path: str = "configs/base.yaml",
    factory: SearcherFactory | None = None,
    specs: Sequence[SystemSpec] | None = None,
    run_sweep: bool = True,
    max_queries: int | None = None,
    min_labeled_queries: int = MIN_LABELED_QUERIES,
) -> EvalResults:
    """Produce all four required outputs, or refuse and explain why.

    Order matters: the labels are checked *before* any search runs. Discovering
    after twenty minutes of GPU-free CPU reranking that nothing could be scored
    would be a waste, and refusing up front with "label more queries" is the
    actionable message.
    """
    specs = ablation_specs() if specs is None else specs
    searcher_factory = factory if factory is not None else PipelineFactory()
    warnings: list[str] = []

    gold = load_gold_queries(cfg.evaluation.gold_queries)
    qrels = load_qrels(cfg.evaluation.qrels)
    coverage = summarize_labels(gold, qrels)
    require_enough_labels(coverage, min_labeled_queries)

    if coverage.orphan_query_ids:
        warnings.append(
            f"{len(coverage.orphan_query_ids)} query id(s) in qrels.csv are not in the "
            f"gold set and were ignored: {list(coverage.orphan_query_ids[:5])}"
            + (" ..." if len(coverage.orphan_query_ids) > 5 else "")
        )
    if coverage.unlabeled_query_ids:
        warnings.append(
            f"{len(coverage.unlabeled_query_ids)} gold query/queries have no labels at "
            "all; every metric is undefined for them and they are excluded from all "
            "means (the counts are on every table)"
        )

    partial_note = ""
    if max_queries is not None and max_queries < len(gold):
        # A deliberately crippled run for smoke-testing the harness. It must be
        # impossible to mistake its output for a result, so the note travels in
        # the result object and the report stamps it on every table.
        gold = list(gold)[:max_queries]
        coverage = summarize_labels(gold, qrels)
        partial_note = (
            f"PARTIAL RUN — only the first {max_queries} gold queries were evaluated "
            "(--max-queries). These numbers are a smoke test of the harness, not a "
            "result, and must not be quoted anywhere."
        )
        warnings.append(partial_note)

    depths = depths_for_eval(cfg)
    runs: dict[str, SystemRun] = {}
    for spec in specs:
        runs[spec.key] = execute_system(spec, gold, searcher_factory, cfg, depths)
        run = runs[spec.key]
        if not run.ran:
            warnings.append(f"system '{spec.key}' ({spec.label}) did not run: {run.skip_reason}")
        elif run.errors:
            warnings.append(
                f"system '{spec.key}' failed on {len(run.errors)} of {len(gold)} queries; "
                "those queries scored as if nothing was returned"
            )

    if not any(r.ran for r in runs.values()):
        raise NoSystemRan(
            "no ablation row could be executed, so there is nothing to report.\n"
            + "\n".join(f"  - {spec.key}: {runs[spec.key].skip_reason}" for spec in specs)
        )

    rows = score_rows(specs, runs, gold, qrels, cfg)

    ladder_order = [s.key for s in specs if not s.supplementary]
    runs_in_order = [runs[k] for k in ladder_order if runs[k].ran]
    reference_key = runs_in_order[-1].key if runs_in_order else ""

    sweep: list[SweepPoint] = []
    sweep_note = "sweep skipped (--no-sweep)"
    if run_sweep:
        # Sweep the most complete system that actually ran, not the nominal
        # full system: in an earlier project phase the top rungs do not exist,
        # and a curve for a system that cannot be built is no curve at all.
        swept_spec = next((s for s in specs if s.key == reference_key), None)
        if swept_spec is None:
            sweep_note = (
                "sweep not run: no row of the ablation ladder executed, so there is "
                "no system to trace a quality-vs-latency curve for"
            )
            warnings.append(sweep_note)
        else:
            sweep, sweep_note = sweep_quality_latency(
                gold, qrels, searcher_factory, cfg, swept_spec
            )
            if not any(p.ran for p in sweep):
                warnings.append(
                    "no point on the quality-vs-latency curve could be measured: "
                    + (sweep[0].skip_reason if sweep else "no valid grid combinations")
                )

    failures = build_failure_analysis(gold, qrels, runs_in_order, cfg, reference_key)

    counts: dict[str, int] = {t.value: 0 for t in QueryType}
    for gq in gold:
        counts[gq.query_type.value] += 1

    return EvalResults(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config_path=str(config_path),
        config_fingerprint=cfg.fingerprint(),
        ndcg_k=ndcg_k_from_config(cfg),
        recall_k=cfg.evaluation.recall_cutoff,
        relevance_threshold=metrics.DEFAULT_RELEVANCE_THRESHOLD,
        eval_depth=depths.final,
        operating_depths=CandidateDepths(
            retrieve=cfg.candidates.retrieve,
            rerank=cfg.candidates.rerank,
            final=cfg.candidates.final,
        ),
        coverage=coverage,
        rows=tuple(rows),
        sweep=tuple(sweep),
        sweep_note=sweep_note,
        failures=tuple(failures),
        failure_reference_system=reference_key,
        query_type_counts=counts,
        partial_run_note=partial_note,
        warnings=tuple(warnings),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description=(
            "Produce the ablation table, per-query-type breakdown, "
            "quality-vs-latency curve and failure analysis."
        ),
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="override evaluation.out_dir (results land in <out-dir>/<config fingerprint>)",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the quality-vs-latency sweep (it reruns the full system once per grid point)",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="comma-separated system keys to run instead of the full ablation",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "smoke-test the harness on the first N gold queries; output is stamped "
            "PARTIAL RUN and must not be quoted"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="write the files without printing the report"
    )
    return parser


def _select_specs(keys: str | None) -> tuple[SystemSpec, ...]:
    if not keys:
        return ablation_specs()
    wanted = [k.strip() for k in keys.split(",") if k.strip()]
    all_specs = ablation_specs()
    by_key = {s.key: s for s in all_specs}
    unknown = [k for k in wanted if k not in by_key]
    if unknown:
        raise EvalError(f"unknown system key(s) {unknown}; choose from {list(by_key)}")
    # Keep the canonical ladder order whatever order the flag listed them in,
    # so "previous row" still means the row above in the architecture.
    return tuple(s for s in all_specs if s.key in set(wanted))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make eval``.

    Returns 0 on success, 2 when the owner needs to do something (label more
    queries, write the gold set, train a stage) and 1 on an unexpected failure.
    A distinct exit code for "not ready" matters because that case is a normal
    state of this project, not a crash.
    """
    args = build_arg_parser().parse_args(argv)
    from eval import report  # lazy: keeps matplotlib out of import time

    try:
        cfg = load_config(args.config)
        results = run_eval(
            cfg,
            config_path=args.config,
            specs=_select_specs(args.systems),
            run_sweep=not args.no_sweep,
            max_queries=args.max_queries,
        )
    except EvalError as exc:
        print(f"\nmake eval stopped: {exc}\n", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"\nmake eval stopped: {exc}\n", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else cfg.evaluation.out_dir
    paths = report.write_report(results, _resolve(out_dir))
    if not args.quiet:
        print(paths.markdown.read_text())
    print("\nwrote:")
    for path in paths.all_paths():
        print(f"  {path}")
    for warning in results.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
