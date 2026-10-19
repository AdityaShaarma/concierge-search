"""Ranking metrics and the statistics used to compare systems.

Every headline number in this project is produced here, which makes this the
most dangerous file in the repo: a wrong NDCG does not crash, it just quietly
makes every ablation row, every resume bullet, and every conclusion wrong while
looking entirely plausible. Three rules follow from that:

1. **Pure functions, no I/O.** Nothing here reads a file, touches a config file
   on disk, or depends on the dataset. Loading ``qrels.csv`` is ``run_eval``'s
   job. That separation is what makes every number in this file verifiable
   against a hand-computed example in ``tests/test_metrics.py``.
2. **Undefined is not zero.** A query with no relevant documents has *no*
   NDCG — not an NDCG of 0.0. Such queries return ``None`` and are excluded
   from means (see ``aggregate``). Scoring them 0.0 would silently punish every
   system equally and drag all reported numbers down by however many
   unlabelable queries happen to be in the gold set.
3. **Standard-library only.** No numpy here. 2000 bootstrap resamples over
   ~150 queries is a few hundred thousand operations, which plain Python does
   in well under a second, and the owner has to be able to read and defend the
   bootstrap loop line by line in an interview.

Data shapes used throughout:

* a **run** is ``{query_id: [business_id, ...]}`` in rank order, best first;
* **qrels** are ``{query_id: {business_id: grade}}`` with graded relevance
  ``0`` irrelevant / ``1`` partially relevant / ``2`` highly relevant.

A business id missing from a query's qrels is *unjudged*, which is treated as
grade 0 for scoring but counted and reported separately — see
``report_unjudged`` for why that caveat must stay visible.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from concierge.types import (
    RELEVANCE_HIGH,
    RELEVANCE_IRRELEVANT,
    RELEVANCE_PARTIAL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shapes and defaults
# ─────────────────────────────────────────────────────────────────────────────

QueryId = str
BusinessId = str
Grade = int

#: ``{query_id: {business_id: grade}}``
Qrels = Mapping[QueryId, Mapping[BusinessId, Grade]]
#: ``{query_id: [business_id, ...]}`` in rank order, best first.
Run = Mapping[QueryId, Sequence[BusinessId]]
#: ``{query_id: score | None}``; ``None`` means the metric is undefined there.
PerQueryScores = Mapping[QueryId, float | None]

DEFAULT_NDCG_K = 10
DEFAULT_RECALL_K = 100

# What counts as "relevant" when a metric needs a yes/no answer from a graded
# label. MRR and Recall both need this. We use >= 1 (partial counts) rather
# than == 2, because for a concierge-style query a partially relevant place is
# a genuinely useful result for the user — a quiet café that happens to lack
# outdoor seating is not a failure in the way an auto body shop is. The
# threshold is a parameter everywhere so the stricter "highly relevant only"
# variant can be reported alongside it without touching this module.
DEFAULT_RELEVANCE_THRESHOLD = RELEVANCE_PARTIAL

# Mirrors the defaults in configs/base.yaml. Duplicated as plain numbers (not
# imported from Config) so this module stays importable and testable with no
# config object at all; `*_from_config` wrappers below bridge the two.
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42

ORDINAL_GRADES: tuple[Grade, ...] = (RELEVANCE_IRRELEVANT, RELEVANCE_PARTIAL, RELEVANCE_HIGH)

#: Cutoffs that turn the graded scale into a usable yes/no answer. Derived from
#: ``ORDINAL_GRADES`` rather than written out, so adding a grade widens this
#: automatically; grade 0 is dropped because "relevant" cannot include the
#: documents the owner graded irrelevant. See ``_validate_relevance_threshold``.
USABLE_RELEVANCE_THRESHOLDS: tuple[int, ...] = tuple(
    g for g in ORDINAL_GRADES if g > RELEVANCE_IRRELEVANT
)

# Which column of ``eval/qrels.csv`` holds the grade, in preference order.
# ``eval.pooling.QrelsWriter`` writes "relevance"; a hand-written qrels file is
# just as naturally headed "grade" or "label". This tuple lives here, next to the
# grading scheme itself, because every reader of that file needs it and three of
# them previously disagreed: the labeling CLI wrote "relevance" while the LLM
# judge read only "grade", so the label-quality kappa — a headline deliverable —
# failed on the owner's own labels. One list, imported by every reader.
QRELS_GRADE_COLUMNS: tuple[str, ...] = ("relevance", "grade", "label", "rel")


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A metric averaged over queries, with the exclusions made explicit.

    ``n_undefined`` is part of the result rather than a log line: "NDCG@10 was
    x over n of the N gold queries" is an honest sentence, "NDCG@10 was x" is
    not, when N - n queries were silently dropped.
    """

    mean: float | None
    n_defined: int
    n_undefined: int

    @property
    def n_total(self) -> int:
        return self.n_defined + self.n_undefined


@dataclass(frozen=True, slots=True)
class UnjudgedReport:
    """How much of a system's top-k nobody has labeled.

    Pooled judging labels the union of what the *compared* systems returned. A
    newly added system surfaces documents that were never in the pool, those
    documents score 0 by default, and the new system therefore looks worse than
    it is. This number is the size of that bias, so it belongs next to the
    metric and not in a footnote.
    """

    fraction: float
    n_unjudged: int
    n_results: int
    per_query: Mapping[QueryId, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    confidence: float
    n_queries: int
    n_resamples: int


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Result of comparing two systems on the same queries."""

    mean_a: float
    mean_b: float
    mean_diff: float  # b - a, i.e. positive means system B is better
    low: float
    high: float
    p_value: float
    confidence: float
    n_queries: int
    n_resamples: int
    n_dropped: int  # pairs excluded because either side was undefined

    @property
    def significant(self) -> bool:
        """Whether the CI for the difference excludes zero.

        Deliberately a property and not a field: it is a reading of the
        interval, not a separate measurement, and the interval is what gets
        reported.
        """
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True, slots=True)
class KappaResult:
    """Inter-rater agreement, with the matrix that produced it.

    ``kappa is None`` means agreement is undefined for this sample — see
    ``cohens_kappa``. Note that ``observed_agreement`` is *weighted* agreement
    (1 - weighted disagreement) and only equals plain percent agreement when
    ``weights == "none"``; the weighting scheme travels in the result so a
    report can never quote the number without saying which one it is.
    """

    kappa: float | None
    observed_agreement: float
    expected_agreement: float
    weights: str
    categories: tuple[int, ...]
    matrix: tuple[tuple[int, ...], ...]
    n: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class RunReport:
    """Everything one system scores on one qrels set.

    Per-query scores are kept, not just the means, because the bootstrap and
    the paired comparison both need them and because the failure analysis is
    driven by the worst queries.
    """

    ndcg: Aggregate
    recall: Aggregate
    mrr: Aggregate
    unjudged: UnjudgedReport
    ndcg_per_query: Mapping[QueryId, float | None]
    recall_per_query: Mapping[QueryId, float | None]
    rr_per_query: Mapping[QueryId, float | None]
    ndcg_k: int
    recall_k: int
    relevance_threshold: int


# ─────────────────────────────────────────────────────────────────────────────
# Small shared pieces
# ─────────────────────────────────────────────────────────────────────────────


def _reject_bad_grade(grade: Grade, subject: str) -> None:
    """Reject any grade outside the labeling scheme — at both ends.

    A stray grade in qrels.csv (a typo in the labeling CLI, a column read in the
    wrong position) would change every gain in the DCG without any visible
    error, so it must fail loudly here.

    Both bounds matter, and the upper one arguably matters more: ``gain`` is
    exponential in the grade, so a label *above* the scheme does not merely
    perturb a DCG, it dominates it — one such document can outweigh every
    correctly labeled result in the ranking. The check is therefore membership
    in the scheme itself rather than a one-sided bound, so that adding a grade
    to ``ORDINAL_GRADES`` is the only way to widen what this function accepts.

    ``subject`` names what is being graded, so the message points at the row to
    fix: a business id when the grade came from a qrels mapping, or the
    position in the list when it came from an already-flattened grade sequence.
    """
    # Type first, deliberately: True == 1 in Python, so a bool would pass the
    # membership test below and silently become "partially relevant".
    if not isinstance(grade, int) or isinstance(grade, bool):
        raise TypeError(f"grade for {subject} must be an int, got {grade!r}")
    if grade not in ORDINAL_GRADES:
        raise ValueError(
            f"grade for {subject} is outside the labeling scheme {ORDINAL_GRADES}: {grade}"
        )


def _validate_grade(business_id: BusinessId, grade: Grade) -> None:
    """``_reject_bad_grade`` for a grade that arrived keyed by a business id."""
    _reject_bad_grade(grade, repr(business_id))


def _validate_relevance_threshold(threshold: int) -> None:
    """Reject a binarization cutoff that is not a usable grade.

    ``relevance_threshold`` turns the graded scale into the yes/no answer that
    Recall and MRR need, as ``grade >= threshold``. Only two values on this
    scheme mean anything:

    * ``1`` — partial counts (the project default), and
    * ``2`` — highly relevant only, the stricter variant reported alongside it.

    Everything else fails here rather than producing a number:

    * ``0`` would make every *judged* document relevant, including the ones the
      owner graded irrelevant on purpose. Recall would then measure how much of
      the pool a system returned and MRR would reward the first judged result
      whatever its grade — both still floats, neither the advertised metric.
    * ``3`` or above can never be reached by any grade, so every query becomes
      undefined and the run reports ``mean=None over n_defined=0``. That is
      honest but opaque: it reads as "nothing is labeled yet" when the real
      cause is one bad argument.

    Validated where it enters the metric rather than only at the config
    boundary, because these functions are public and the eval harness, the LLM
    judge and the tests all call them directly.
    """
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise TypeError(f"relevance_threshold must be an int, got {threshold!r}")
    if threshold not in USABLE_RELEVANCE_THRESHOLDS:
        raise ValueError(
            f"relevance_threshold must be one of {USABLE_RELEVANCE_THRESHOLDS} "
            f"(grades {ORDINAL_GRADES}), got {threshold}"
        )


def dedupe_preserving_order(ids: Iterable[BusinessId]) -> list[BusinessId]:
    """Drop repeated ids, keeping the first (best-ranked) occurrence.

    A buggy fusion step can emit the same business twice. Deduplicating
    *before* truncating to k mirrors what a user would actually see (one card
    per place) and stops a system from being rewarded twice for one good hit.
    Doing it before truncation matters: dropping duplicates after cutting to k
    would leave a short list and understate recall.
    """
    seen: set[BusinessId] = set()
    out: list[BusinessId] = []
    for bid in ids:
        if bid in seen:
            continue
        seen.add(bid)
        out.append(bid)
    return out


def gain(grade: Grade) -> float:
    """Exponential gain, ``2**grade - 1``.

    This is a real choice and a standard interview question. Linear gain
    (``gain = grade``) makes a highly relevant result worth twice a partially
    relevant one; exponential gain makes it worth three times as much
    (``3`` vs ``1``). We use exponential because it is the formulation in
    Burges et al.'s LambdaMART line of work and the default in most IR
    toolkits, so our numbers are comparable to published ones; and because for
    this task the preference really is sharply non-linear — a user asking for
    "a quiet café with wifi open past 10pm" wants the place that satisfies all
    three far more than two places that satisfy some. The cost is that NDCG
    becomes more sensitive to a single mislabeled 2, which is exactly why label
    agreement (``cohens_kappa``) is measured too.

    The grade is validated here rather than only in the callers that read
    qrels. ``gain`` is public, so it is reachable without passing through
    ``grades_for`` / ``ideal_dcg_at_k``, and the exponent is exactly what makes
    an out-of-scheme grade dangerous: ``gain(5) = 31`` would outweigh ten
    correctly graded 2s and still return a plausible float. The one guard the
    whole DCG rests on therefore sits at the innermost point every path
    reaches, not only at the outer ones.
    """
    _reject_bad_grade(grade, "a gain")
    return float(2**grade - 1)


def discount(rank: int) -> float:
    """Positional discount ``log2(rank + 1)`` for a 1-based rank.

    rank 1 -> log2(2) = 1 (no discount), rank 2 -> log2(3) ~ 1.585, and so on.
    """
    if rank < 1:
        raise ValueError(f"rank must be 1-based, got {rank}")
    return math.log2(rank + 1)


def grades_for(
    ranked_ids: Sequence[BusinessId],
    judged: Mapping[BusinessId, Grade],
) -> list[Grade]:
    """Grades of a ranked list, with unjudged results scored 0.

    Scoring unjudged as 0 is the standard assumption (and the only one
    available without more labeling), but it is a *bias*, not a fact. Its size
    is measured separately by ``report_unjudged``.
    """
    out: list[Grade] = []
    for bid in ranked_ids:
        grade = judged.get(bid, RELEVANCE_IRRELEVANT)
        _validate_grade(bid, grade)
        out.append(grade)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-query ranking metrics
# ─────────────────────────────────────────────────────────────────────────────


def dcg_at_k(grades: Sequence[Grade], k: int) -> float:
    """Discounted cumulative gain of a graded, rank-ordered list.

    Fewer than k grades is normal (a filter may leave a short candidate list)
    and simply contributes nothing beyond the end of the list; it is not padded
    and not an error.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    total = 0.0
    for rank, grade in enumerate(grades[:k], start=1):
        # Named by position, because by this point the grades have been
        # flattened out of their qrels mapping and there is no business id left
        # to point at. ``gain`` validates too; this call site adds the position
        # so the message says *which* entry of the list is wrong.
        _reject_bad_grade(grade, f"rank {rank} of a graded list")
        total += gain(grade) / discount(rank)
    return total


def ideal_dcg_at_k(judged: Mapping[BusinessId, Grade], k: int) -> float:
    """Best achievable DCG@k for a query, from *all* of its labels.

    Built from the full qrels row rather than from the grades the system
    happened to retrieve. Using only retrieved documents would normalize each
    system against its own ceiling, so a system that missed every good result
    could still score 1.0.
    """
    all_grades = sorted(judged.values(), reverse=True)
    for bid, grade in judged.items():
        _validate_grade(bid, grade)
    return dcg_at_k(all_grades, k)


def ndcg_at_k(
    ranked_ids: Sequence[BusinessId],
    judged: Mapping[BusinessId, Grade],
    k: int = DEFAULT_NDCG_K,
) -> float | None:
    """NDCG@k, or ``None`` when the query has no relevant documents.

    The ``None`` is the honest answer rather than a dodge. If every label for a
    query is 0, the ideal DCG is 0 and NDCG is 0/0 — literally undefined, not
    zero. Returning 0.0 instead would mean a system that correctly found there
    is nothing good to return is scored identically to one that returned
    garbage, and every such query would pull the reported mean down by a
    constant that says nothing about ranking quality. ``aggregate`` drops these
    and reports how many were dropped.
    """
    ideal = ideal_dcg_at_k(judged, k)
    if ideal <= 0.0:
        return None
    actual = dcg_at_k(grades_for(dedupe_preserving_order(ranked_ids), judged), k)
    return actual / ideal


def recall_at_k(
    ranked_ids: Sequence[BusinessId],
    judged: Mapping[BusinessId, Grade],
    k: int = DEFAULT_RECALL_K,
    relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Fraction of known-relevant documents found in the top k.

    This is the retrieval-stage metric: Stage 2's job is not to rank well, it
    is to not lose the good results before the reranker ever sees them. Recall
    here is recall *against the labeled pool*, so it is an upper bound on true
    recall: the denominator counts only the relevant businesses somebody
    actually graded, and no labeling session covers every business in scope for
    every gold query. Relevant places nobody ever surfaced are missing from the
    denominator entirely, which makes this number comparable between systems
    but not readable as absolute coverage.

    Returns ``None`` when no document is labeled relevant for the query, for
    the same reason NDCG does.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _validate_relevance_threshold(relevance_threshold)
    for bid, grade in judged.items():
        _validate_grade(bid, grade)
    relevant = {bid for bid, g in judged.items() if g >= relevance_threshold}
    if not relevant:
        return None
    top_k = dedupe_preserving_order(ranked_ids)[:k]
    found = sum(1 for bid in top_k if bid in relevant)
    return found / len(relevant)


def reciprocal_rank(
    ranked_ids: Sequence[BusinessId],
    judged: Mapping[BusinessId, Grade],
    relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
    k: int | None = None,
) -> float | None:
    """1 / rank of the first relevant result; 0.0 if none is found.

    "Relevant" is explicitly ``grade >= relevance_threshold``, which defaults
    to 1 (partially relevant counts). The stricter ``relevance_threshold=2``
    variant answers a different question — "how fast do we surface a *great*
    result" — and both are worth reporting; neither is the single true MRR, so
    the threshold is a parameter and the chosen value is printed with the
    number.

    ``k`` optionally caps the depth searched (``None`` = the whole list). It is
    distinct from ``0.0``-vs-``None``: 0.0 means "we looked and found nothing
    relevant", ``None`` means "this query has no relevant documents at all, so
    there was nothing to find".
    """
    _validate_relevance_threshold(relevance_threshold)
    for bid, grade in judged.items():
        _validate_grade(bid, grade)
    if not any(g >= relevance_threshold for g in judged.values()):
        return None
    candidates = dedupe_preserving_order(ranked_ids)
    if k is not None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        candidates = candidates[:k]
    for rank, bid in enumerate(candidates, start=1):
        if judged.get(bid, RELEVANCE_IRRELEVANT) >= relevance_threshold:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation over a run
# ─────────────────────────────────────────────────────────────────────────────


def aggregate(scores: Iterable[float | None]) -> Aggregate:
    """Mean over queries where the metric is defined, counting the rest.

    Excluding undefined queries (rather than treating them as 0.0) is the whole
    point; see ``ndcg_at_k``. The exclusion count travels with the mean so a
    report can never quote the mean without the denominator.

    ``scores`` is materialized once, before anything is counted, because the
    annotation admits any iterable and a caller may reasonably pass a generator
    expression. Two passes over a one-shot iterable would see the scores in the
    first pass and nothing in the second: the mean would still be right and
    ``n_undefined`` would come back 0, so the ``Aggregate`` would claim every
    query was defined. That is the exact dishonesty this container exists to
    prevent, and it would be invisible in the output.
    """
    values = list(scores)
    defined = [s for s in values if s is not None]
    mean = sum(defined) / len(defined) if defined else None
    return Aggregate(
        mean=mean,
        n_defined=len(defined),
        n_undefined=len(values) - len(defined),
    )


def per_query(
    run: Run,
    qrels: Qrels,
    metric: Callable[[Sequence[BusinessId], Mapping[BusinessId, Grade]], float | None],
) -> dict[QueryId, float | None]:
    """Apply a per-query metric across a whole run.

    Queries absent from qrels get an empty label row, so every metric sees "no
    relevant documents" and returns ``None`` rather than a misleading 0.0.
    """
    return {qid: metric(ranked, qrels.get(qid, {})) for qid, ranked in run.items()}


def report_unjudged(
    run: Run,
    qrels: Qrels,
    k: int = DEFAULT_NDCG_K,
) -> UnjudgedReport:
    """Fraction of the top-k results that carry no label.

    Must be reported next to every NDCG in the ablation table. With incremental
    pooling, a newly added system is the only one whose freshly surfaced
    documents are unlabeled; those are scored 0 and the system is penalized for
    finding things nobody looked at yet. A high fraction here means the
    comparison is not yet fair and more labeling is needed — a conclusion that
    is invisible if this number is not printed.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_unjudged = 0
    n_results = 0
    per_q: dict[QueryId, float] = {}
    for qid, ranked in run.items():
        judged = qrels.get(qid, {})
        top_k = dedupe_preserving_order(ranked)[:k]
        missing = sum(1 for bid in top_k if bid not in judged)
        n_unjudged += missing
        n_results += len(top_k)
        # A query that returned nothing has no unjudged share; 0.0 keeps the
        # per-query dict total and avoids a 0/0.
        per_q[qid] = missing / len(top_k) if top_k else 0.0
    fraction = n_unjudged / n_results if n_results else 0.0
    return UnjudgedReport(
        fraction=fraction,
        n_unjudged=n_unjudged,
        n_results=n_results,
        per_query=per_q,
    )


def evaluate_run(
    run: Run,
    qrels: Qrels,
    ndcg_k: int = DEFAULT_NDCG_K,
    recall_k: int = DEFAULT_RECALL_K,
    relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> RunReport:
    """Score one system: the three headline metrics plus the unjudged caveat.

    Per-query scores are returned alongside the means because the bootstrap,
    the paired comparison, the per-query-type breakdown and the failure
    analysis all need them, and recomputing them in four places is how two
    reported numbers end up disagreeing.
    """
    # Up front, not inside the per-query lambdas: a run with an empty qrels row
    # for every query would never reach the metric functions, and the harness
    # would report "nothing is labeled" rather than "your threshold is wrong".
    _validate_relevance_threshold(relevance_threshold)
    ndcg_scores = per_query(run, qrels, lambda r, j: ndcg_at_k(r, j, ndcg_k))
    recall_scores = per_query(
        run, qrels, lambda r, j: recall_at_k(r, j, recall_k, relevance_threshold)
    )
    rr_scores = per_query(run, qrels, lambda r, j: reciprocal_rank(r, j, relevance_threshold))
    return RunReport(
        ndcg=aggregate(ndcg_scores.values()),
        recall=aggregate(recall_scores.values()),
        mrr=aggregate(rr_scores.values()),
        unjudged=report_unjudged(run, qrels, ndcg_k),
        ndcg_per_query=ndcg_scores,
        recall_per_query=recall_scores,
        rr_per_query=rr_scores,
        ndcg_k=ndcg_k,
        recall_k=recall_k,
        relevance_threshold=relevance_threshold,
    )


def subset(scores: PerQueryScores, query_ids: Iterable[QueryId]) -> dict[QueryId, float | None]:
    """Restrict per-query scores to a set of query ids (for the query-type
    breakdown). Ids not present in ``scores`` are skipped rather than invented.
    """
    wanted = set(query_ids)
    return {qid: s for qid, s in scores.items() if qid in wanted}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────


def _defined_values(scores: PerQueryScores | Sequence[float | None]) -> list[float]:
    """Pull out the usable per-query scores from either a dict or a sequence.

    Both shapes appear naturally (``RunReport`` holds dicts; ad-hoc analysis
    holds lists), and accepting both keeps call sites from juggling conversions.
    """
    values = scores.values() if isinstance(scores, Mapping) else scores
    return [v for v in values if v is not None]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    Written out rather than pulled from numpy so the interpolation rule is
    visible: this is the same convention numpy uses by default, which matters
    only for reproducing a published interval exactly.
    """
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    per_query_scores: PerQueryScores | Sequence[float | None],
    n: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for a mean metric, resampling *queries*.

    Why bootstrap at all: on a gold set this small, the difference between two
    nearby NDCG@10 values can easily be nothing more than which queries happen
    to be in the set. The bootstrap answers "how much would this mean move if I
    had drawn a different sample of queries from the same population?" by
    resampling the queries with replacement ``n`` times and reading the middle
    ``confidence`` of the resulting means.

    Why queries are the resampling unit, not query-document pairs: the queries
    are the independent observations. Documents within a query are not
    independent of each other, and resampling them would produce an interval
    far too narrow.

    Seeded from the config seed so a rerun reproduces the same interval
    exactly; a CI that moves between runs cannot be put in a README.
    """
    if n < 1:
        raise ValueError(f"n resamples must be >= 1, got {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    values = _defined_values(per_query_scores)
    if not values:
        raise ValueError("no defined per-query scores to bootstrap")

    observed_mean = sum(values) / len(values)
    rng = random.Random(seed)
    size = len(values)
    means: list[float] = []
    for _ in range(n):
        resample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(resample) / size)
    means.sort()

    alpha = 1.0 - confidence
    return ConfidenceInterval(
        mean=observed_mean,
        low=_percentile(means, alpha / 2.0),
        high=_percentile(means, 1.0 - alpha / 2.0),
        confidence=confidence,
        n_queries=size,
        n_resamples=n,
    )


def paired_bootstrap(
    scores_a: PerQueryScores,
    scores_b: PerQueryScores,
    n: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Compare two systems on the same queries: mean difference, CI, p-value.

    **Why paired.** Both systems are evaluated on an identical query set, so
    each query's difficulty affects both equally. Resampling the *pairs* lets
    that shared difficulty cancel: an easy query that lifts both systems'
    scores contributes ~0 to the difference. Two independent CIs on the two
    means would instead include query-difficulty variance in both, be much
    wider, and routinely overlap even when every single query improved. Query
    difficulty is a confound, and pairing removes it for free.

    The difference is reported as ``b - a``, so a positive value means system B
    (the later, usually more expensive stage) is better.

    **The p-value.** It is the bootstrap achieved significance level: the share
    of resamples whose mean difference falls on the opposite side of zero from
    the observed one, doubled for a two-sided test. A ``(count + 1) / (n + 1)``
    convention is used so the result is never exactly 0 — ``n`` resamples
    cannot distinguish "very unlikely" from "impossible", and a literal p = 0
    in a README is a claim the method cannot support. Report it as
    "p <= 1/(n+1)" when it bottoms out. The CI on the difference is the
    primary thing to read; the p-value is for people who ask for one.

    Queries where either system's metric is undefined are dropped as a pair
    (comparing a defined score against a missing one is not a comparison), and
    the count is reported in ``n_dropped``.
    """
    if n < 1:
        raise ValueError(f"n resamples must be >= 1, got {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    shared = set(scores_a) & set(scores_b)
    if set(scores_a) != set(scores_b):
        # Hard error, not a silent intersection. Quietly comparing the overlap
        # would report a paired test over a query set neither system was
        # actually evaluated on, and the caller would never know. Aligning the
        # sets is the caller's decision to make explicitly.
        raise ValueError(
            "paired comparison needs the same query ids on both sides; "
            f"a has {len(scores_a)}, b has {len(scores_b)}, {len(shared)} in common"
        )

    pairs: list[tuple[float, float]] = []
    n_dropped = 0
    for qid in sorted(shared):  # sorted so the resampling is order-independent
        a, b = scores_a[qid], scores_b[qid]
        if a is None or b is None:
            n_dropped += 1
            continue
        pairs.append((a, b))
    if not pairs:
        raise ValueError("no query has a defined score for both systems")

    size = len(pairs)
    mean_a = sum(a for a, _ in pairs) / size
    mean_b = sum(b for _, b in pairs) / size
    observed_diff = mean_b - mean_a

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n):
        # One index draw per resample row, used for BOTH systems. This is the
        # pairing: system A and system B always see the same resampled queries.
        total = 0.0
        for _ in range(size):
            a, b = pairs[rng.randrange(size)]
            total += b - a
        diffs.append(total / size)

    alpha = 1.0 - confidence
    ordered = sorted(diffs)
    n_wrong_side = (
        sum(1 for d in diffs if d <= 0.0)
        if observed_diff > 0
        else sum(1 for d in diffs if d >= 0.0)
    )
    p_value = min(1.0, 2.0 * (n_wrong_side + 1) / (n + 1))

    return PairedComparison(
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=observed_diff,
        low=_percentile(ordered, alpha / 2.0),
        high=_percentile(ordered, 1.0 - alpha / 2.0),
        p_value=p_value,
        confidence=confidence,
        n_queries=size,
        n_resamples=n,
        n_dropped=n_dropped,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Label agreement
# ─────────────────────────────────────────────────────────────────────────────


def confusion_matrix(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    categories: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Counts of (rater A label, rater B label), rows = A, columns = B.

    Reported alongside kappa because kappa alone hides the *shape* of the
    disagreement, and the shape is the useful part: an LLM judge that confuses
    1 and 2 is usable for training data, one that calls 0s highly relevant is
    not.

    ``categories`` fixes the axes. ``None`` derives them from the labels that
    actually appear, which is the right default *for a free-standing display
    matrix on an unknown scale* and nothing more: a sample that happens to
    contain no 1s then yields a 2x2 matrix whose middle grade has silently
    vanished. Pass the scale explicitly whenever the axes have to mean the same
    thing across two samples — which is why ``cohens_kappa`` always passes it
    (see the note on weighting there) and why ``label_cli`` passes ``(0, 1, 2)``
    when it prints the matrix for the owner.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"label lists differ in length: {len(labels_a)} vs {len(labels_b)}")
    if not labels_a:
        raise ValueError("cannot build a confusion matrix from zero labels")
    cats = list(categories) if categories is not None else sorted(set(labels_a) | set(labels_b))
    index = {c: i for i, c in enumerate(cats)}
    matrix = [[0] * len(cats) for _ in cats]
    for a, b in zip(labels_a, labels_b, strict=True):
        if a not in index or b not in index:
            raise ValueError(f"label pair ({a!r}, {b!r}) outside categories {cats}")
        matrix[index[a]][index[b]] += 1
    return tuple(tuple(row) for row in matrix)


def cohens_kappa(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    weights: str = "quadratic",
    categories: Sequence[int] = ORDINAL_GRADES,
) -> KappaResult:
    """Chance-corrected agreement between two raters over ordinal grades.

    Used for the label-quality check: an LLM judge grades a sample of the pairs
    the owner graded by hand, and this says how much to trust LLM labels in the
    training data. Raw percent agreement is not enough: whenever one grade
    dominates the pool — and for pooled relevance judging the dominant grade is
    0 — a rater that always answers that grade matches the other rater on every
    such pair and scores high agreement while knowing nothing. Kappa subtracts
    the agreement that the label distribution alone already explains.

    **Why weighted, and why quadratic by default.** Our grades 0/1/2 are
    ordinal, not categorical: labeling a "2" as a "1" is a near miss, labeling
    it "0" is a real error. Unweighted kappa treats both as equally wrong,
    which understates agreement on exactly the kind of data we have. Quadratic
    weights ``w_ij = (i - j)^2 / (C - 1)^2`` penalize the two-step
    disagreement four times as much as the one-step one, matching the intuition
    that 0-vs-2 is the mistake that matters. ``"linear"`` weights
    (``|i - j| / (C - 1)``) are the milder ordinal alternative, and
    ``"none"`` gives the original unweighted Cohen's kappa — pass it to report
    both, since quadratic weighting on a 3-point scale is the more flattering
    number and should not be quoted alone without saying it is weighted.

    **Why ``categories`` defaults to the scale and not to the sample.** The
    weights above are positions on a *declared* scale: ``i`` and ``j`` index
    ``categories``, and the normalizer is ``C - 1`` where ``C =
    len(categories)``. Deriving that list from the labels a sample happens to
    contain — the earlier default here — makes the weighting depend on the
    sample. A batch with no 1s in it collapses to ``categories = (0, 2)``,
    ``C = 2``, and the single worst disagreement a relevance judge can make
    (0 vs 2) gets weight ``1.0`` instead of its place on the 0/1/2 scale: the
    weighting has silently become unweighted while the result still says
    ``weights="quadratic"``.

    Be precise about what that costs, because it is tempting to overstate it.
    On *this* scale the kappa ratio happens to survive. Collapsing the grades
    rescales every weight by the same factor — the normalizer ``(C - 1)^2`` is
    common to all of them — and a uniform rescale cancels in
    ``1 - observed/expected``. What breaks the cancellation is a collapse that
    changes the *relative* distances between the surviving grades, and that
    needs an unused grade with grades on both sides of it. A three-point scale
    cannot produce one: drop the interior 1 and only the 0-vs-2 magnitude is
    left, drop an end grade and the remaining two are adjacent. Four points
    can, and then the kappa itself moves (see the four-point case in the
    tests).

    So the ratio is safe here and the rest is not. ``observed_agreement`` and
    ``expected_agreement`` are weighted quantities and change outright,
    ``categories`` records a scale the labeling scheme does not have, and the
    confusion matrix — a required deliverable in its own right — loses the row
    and column for the missing grade, so "nobody ever graded a 1" becomes
    invisible precisely when it is the finding. None of that is worth defending
    for a default that saves one argument, and relying on the cancellation
    would be relying on an accident of ``len(ORDINAL_GRADES) == 3``.

    Hence: declare the scale up front and check the sample against it. An
    unused grade is then a fact about the sample, not a change to the metric.
    A caller measuring agreement on a genuinely different scale (a binary
    relevant/irrelevant collapse, say) passes that scale explicitly and the
    result records it. This matters here more than it would elsewhere because
    this kappa is the project's only evidence that LLM labels may be used in
    training data at all.

    **Degenerate case.** If chance-expected disagreement is zero — both raters'
    marginals sitting on the same single category — kappa is 0/0. We return
    ``kappa=None`` with a note instead of dividing by zero or quietly reporting
    1.0: when one grade explains every label, there is no evidence of agreement
    beyond chance, because chance already explains everything. Note that this
    now depends only on the *labels*, not on how many categories were declared:
    declaring the full 0/1/2 scale over a sample where both raters only ever
    said 0 still yields ``None``, because the marginals, not the axes, are what
    chance agreement is computed from.
    """
    if weights not in {"quadratic", "linear", "none"}:
        raise ValueError(f"unknown weights {weights!r}; use quadratic, linear or none")
    if categories is None:
        # Not silently re-derived from the sample: that was the bug this
        # default replaced, and failing loudly is the only way a caller that
        # relied on it finds out.
        raise ValueError(
            "categories must be the scale the grades live on, e.g. "
            f"{ORDINAL_GRADES}; None is no longer accepted because the "
            "ordinal weights are positions on that scale"
        )
    cats = tuple(categories)
    if not cats:
        raise ValueError("categories must not be empty")
    if list(cats) != sorted(set(cats)):
        # The weights read |i - j| as "how far apart on the scale", so the list
        # has to *be* the scale in order. (2, 0, 1) would make 0-vs-1 the
        # two-step disagreement and 0-vs-2 the one-step one.
        raise ValueError(f"categories must be strictly increasing, got {cats}")
    matrix = confusion_matrix(labels_a, labels_b, cats)
    n = len(labels_a)
    c = len(cats)

    # Weight matrix: w = 0 means full agreement, w = 1 means maximal disagreement.
    def disagreement_weight(i: int, j: int) -> float:
        if i == j:
            return 0.0
        if weights == "none":
            return 1.0
        if c == 1:  # cannot normalize by (c - 1); no disagreement is expressible
            return 0.0
        step = abs(i - j) / (c - 1)
        return step**2 if weights == "quadratic" else step

    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[i][j] for i in range(c)) for j in range(c)]

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(c):
        for j in range(c):
            w = disagreement_weight(i, j)
            observed_disagreement += w * matrix[i][j] / n
            # Expected counts under independent raters keeping their own
            # marginal label frequencies — that is what "by chance" means here.
            expected_disagreement += w * (row_totals[i] / n) * (col_totals[j] / n)

    observed_agreement = 1.0 - observed_disagreement
    expected_agreement = 1.0 - expected_disagreement

    if expected_disagreement <= 0.0:
        # Say which grades were actually used, not how many were declared: with
        # the scale now fixed by default, "3 categories" would be true of every
        # call and would explain nothing about why this one is undefined.
        used = tuple(cat for i, cat in enumerate(cats) if row_totals[i] or col_totals[i])
        note = (
            "kappa undefined: expected disagreement is zero. Over the declared "
            f"scale {cats}, both raters' labels fall on {used}, so chance alone "
            "explains the agreement and there is nothing left to measure"
        )
        return KappaResult(
            kappa=None,
            observed_agreement=observed_agreement,
            expected_agreement=expected_agreement,
            weights=weights,
            categories=cats,
            matrix=matrix,
            n=n,
            note=note,
        )

    kappa = 1.0 - (observed_disagreement / expected_disagreement)
    return KappaResult(
        kappa=kappa,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        weights=weights,
        categories=cats,
        matrix=matrix,
        n=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config bridges
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_ci_from_config(
    per_query_scores: PerQueryScores | Sequence[float | None],
    cfg: object,
) -> ConfidenceInterval:
    """``bootstrap_ci`` with ``n``, ``confidence`` and ``seed`` taken from config.

    A thin wrapper on purpose: the metric functions stay free of the config
    object so they can be unit-tested with literals, while ``run_eval`` never
    has to restate the resample count and risk two reported intervals using
    different settings. Typed as ``object`` to avoid importing the pydantic
    Config at module import time; the attribute access below is the contract.
    """
    return bootstrap_ci(
        per_query_scores,
        n=cfg.evaluation.bootstrap_samples,  # type: ignore[attr-defined]
        confidence=cfg.evaluation.confidence,  # type: ignore[attr-defined]
        seed=cfg.seed,  # type: ignore[attr-defined]
    )


def paired_bootstrap_from_config(
    scores_a: PerQueryScores,
    scores_b: PerQueryScores,
    cfg: object,
) -> PairedComparison:
    """``paired_bootstrap`` with resamples, confidence and seed from config."""
    return paired_bootstrap(
        scores_a,
        scores_b,
        n=cfg.evaluation.bootstrap_samples,  # type: ignore[attr-defined]
        confidence=cfg.evaluation.confidence,  # type: ignore[attr-defined]
        seed=cfg.seed,  # type: ignore[attr-defined]
    )
