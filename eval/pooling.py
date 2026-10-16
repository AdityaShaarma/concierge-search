"""Pooling, label-subset selection, and the qrels file format.

This module is the bookkeeping half of the labeling workflow; ``label_cli.py``
is the interface half. Everything that decides *which* (query, business) pairs a
human is asked about lives here, because those decisions are the ones an
interviewer will question:

1. **Pooling.** Grading every gold query against every business in scope is out
   of the question by hand: that is the gold set size times the whole searchable
   index. The standard alternative (TREC's method since 1992) is to pool: take
   the top-``depth`` results from every system being compared, take the union,
   and judge only that. Anything outside the pool is treated as irrelevant when
   scoring. This
   is biased in a known direction — a *new* system that surfaces good documents
   nobody pooled looks worse than it is — so ``eval/metrics.py`` reports the
   unjudged fraction next to every metric rather than hiding it.

2. **Incremental pooling.** When a new stage lands, only the newly surfaced
   pairs are judged; previously graded pairs are excluded here, in
   :func:`build_pool`. The owner must never be asked to re-grade a pair. That
   is not only about their time: re-grading invites a different answer on a
   second look, and then the same pair has two labels and every metric computed
   from the file depends on which row won.

3. **Random label subset.** Only ~50 of ~150 queries get labeled first (D004).
   Which 50 is chosen by a hash of the query id — see
   :func:`select_label_subset` for why anything convenience-based (the first 50,
   the short ones, the ones that "look interesting") would quietly bias every
   number in the project.

The qrels file is append-only CSV. Not sqlite, not JSON: a human can open it,
grep it, and see that nothing was lost, and a crash mid-session can at worst
lose the row being written.

Data shapes are the ones ``eval/metrics.py`` already defines — ``run``,
``qrels`` — so a pool built here and a metric computed there never disagree
about what a query id is.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from types import TracebackType
from typing import Final, TextIO

from concierge.types import (
    RELEVANCE_HIGH,
    RELEVANCE_IRRELEVANT,
    RELEVANCE_PARTIAL,
    GoldQuery,
    QueryType,
)
from eval.metrics import (
    BusinessId,
    Grade,
    Qrels,
    QueryId,
    Run,
    dedupe_preserving_order,
)

__all__ = [
    "EXAMPLE_ID_PREFIX",
    "QRELS_COLUMNS",
    "VALID_GRADES",
    "GoldSet",
    "Judgement",
    "Pool",
    "PoolPair",
    "PoolStats",
    "QrelsWriter",
    "Runs",
    "agreement_pairs",
    "build_pool",
    "build_retest_pool",
    "estimate_minutes",
    "format_pool_report",
    "hash_key",
    "judged_pairs",
    "load_gold_queries",
    "load_qrels",
    "measure_pace",
    "read_judgements",
    "retest_subset",
    "select_label_subset",
]

#: ``{system_name: run}``. One entry per system in the ablation.
Runs = Mapping[str, Run]

QRELS_COLUMNS: Final[tuple[str, ...]] = (
    "query_id",
    "business_id",
    "relevance",
    "labeled_at",
    "labeler",
)

VALID_GRADES: Final[frozenset[int]] = frozenset(
    {RELEVANCE_IRRELEVANT, RELEVANCE_PARTIAL, RELEVANCE_HIGH}
)

#: Rows in ``gold_queries.csv`` whose id starts with this are shipped examples,
#: not real test queries, and are skipped by :func:`load_gold_queries`. The file
#: has to carry examples (the owner needs to see the format) and those examples
#: must be incapable of reaching a metric: a query nobody wrote should never
#: appear in a reported number.
EXAMPLE_ID_PREFIX: Final = "EXAMPLE"

# Timestamp gaps longer than this are treated as "the owner stood up", not as
# one very slow judgement, when measuring labeling pace. Three minutes is long
# enough for a genuinely hard pair (reading two reviews and a weekly hours
# table) and short enough to exclude lunch.
DEFAULT_MAX_JUDGEMENT_GAP_S: Final = 180.0

# A pace estimate from one or two intervals is noise. Below this many usable
# intervals we report "unmeasured" rather than a number — the whole project
# rests on not inventing numbers, including the small ones.
MIN_INTERVALS_FOR_PACE: Final = 3


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic randomness
# ─────────────────────────────────────────────────────────────────────────────


def hash_key(seed: int, *parts: str) -> str:
    """A stable pseudorandom sort key for the given strings.

    SHA-256 rather than :func:`hash`: Python's string hash is salted per
    process, so a pool or a subset built with it would differ between runs and
    the owner could not resume a labeling session or reproduce a reported
    number. SHA-256 of ``seed|parts`` is uniform, deterministic forever, and
    depends on *nothing* but its inputs — not on row order, not on insertion
    order, not on how many other queries exist.
    """
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Gold queries
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GoldSet:
    """The owner's test queries, plus what was dropped while reading them.

    ``n_examples_skipped`` is part of the result so the CLI can say "you still
    have 3 example rows in there" out loud. Silently dropping rows from a file
    the owner is actively editing is how a query goes missing without anyone
    noticing.
    """

    queries: tuple[GoldQuery, ...]
    n_examples_skipped: int = 0

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def by_id(self) -> dict[QueryId, GoldQuery]:
        return {q.query_id: q for q in self.queries}

    def counts_by_type(self) -> dict[QueryType, int]:
        counts = dict.fromkeys(QueryType, 0)
        for q in self.queries:
            counts[q.query_type] += 1
        return counts


def load_gold_queries(path: str | Path, *, keep_examples: bool = False) -> GoldSet:
    """Read ``gold_queries.csv``.

    Args:
        path: the CSV, with a ``query_id,query,query_type`` header.
        keep_examples: include the shipped ``EXAMPLE*`` rows. Only the tests
            and a format check want these.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: a bad header, a duplicate ``query_id``, an empty query, or a
            ``query_type`` outside the four defined in ``concierge.types``.

    Every one of those is raised rather than skipped. A duplicate query id in
    particular is silent poison: two different queries would share one row of
    labels, and the merged result is neither query's score.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"gold queries not found: {p}. The owner writes this file by hand — see eval/README.md."
        )

    queries: list[GoldQuery] = []
    seen: set[str] = set()
    skipped = 0
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"query_id", "query", "query_type"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{p} is missing column(s) {sorted(missing)}; "
                f"header must be query_id,query,query_type"
            )
        for lineno, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            text = (row.get("query") or "").strip()
            raw_type = (row.get("query_type") or "").strip().lower()
            if not query_id and not text:
                continue  # a blank line left behind by an editor
            if query_id.upper().startswith(EXAMPLE_ID_PREFIX) and not keep_examples:
                skipped += 1
                continue
            if not query_id:
                raise ValueError(f"{p}:{lineno} has no query_id")
            if not text:
                raise ValueError(f"{p}:{lineno} ({query_id}) has an empty query")
            if query_id in seen:
                raise ValueError(
                    f"{p}:{lineno} repeats query_id {query_id!r}. Ids must be unique — "
                    f"labels are keyed by them."
                )
            try:
                query_type = QueryType(raw_type)
            except ValueError as exc:
                valid = ", ".join(t.value for t in QueryType)
                raise ValueError(
                    f"{p}:{lineno} ({query_id}) has query_type {raw_type!r}; "
                    f"must be one of: {valid}"
                ) from exc
            seen.add(query_id)
            queries.append(GoldQuery(query_id=query_id, query=text, query_type=query_type))

    return GoldSet(queries=tuple(queries), n_examples_skipped=skipped)


# ─────────────────────────────────────────────────────────────────────────────
# Label subset selection
# ─────────────────────────────────────────────────────────────────────────────


def select_label_subset(
    gold_queries: Sequence[GoldQuery],
    size: int,
    seed: int,
) -> tuple[GoldQuery, ...]:
    """Choose which queries to label first — a RANDOM, reproducible subset.

    ### Why random is non-negotiable

    Labeling all ~150 queries at pool depth 20 is many hours of work, so the
    first reported numbers come from a subset. The temptation is to pick that
    subset by convenience: the first 50 rows, the ones that are quick to judge,
    the ones the owner finds interesting. Every one of those is a biased sample,
    and the bias lands directly in the headline metric:

    * **First 50 rows** — queries get written in the order they occur to you,
      which means the easy, prototypical ones come first ("good pho") and the
      awkward multi-intent ones come later. Grading the first 50 reports the
      system's performance on its easiest third and calls it NDCG@10.
    * **Quick to judge** — a query is quick to judge when the results are
      obviously right or obviously wrong. Selecting for that selects *against*
      the borderline cases where ranking quality actually differs between
      systems, so every ablation gap shrinks toward noise.
    * **Interesting-looking** — selects for whatever the owner suspects is
      broken, which inflates the measured benefit of whichever stage fixes it.

    A uniform random sample has none of those problems: the subset's expected
    score is the full set's score, so "NDCG@10 = X over a random 50 of 150" is
    an honest, defensible sentence, and the confidence interval in
    ``metrics.bootstrap_ci`` correctly describes the remaining uncertainty.

    ### Why a hash and not ``random.sample``

    Both are random. The hash ordering has one extra property that matters here:
    it is **nested**. ``size=50`` returns the first 50 of the same total order
    that ``size=100`` takes 100 from, so when the owner expands the labeled set
    (D004 plans exactly that), every label already collected stays inside the
    new subset. With ``random.sample`` the 100-query draw is a different set and
    some of the first 50 would fall outside it — labels paid for in hours of
    work, no longer part of the reported sample.

    It is also independent of file order: adding, removing, or reordering rows
    cannot change whether some *other* query is in the subset, because a query's
    key depends only on its id and the seed.

    Args:
        gold_queries: the owner's queries. Order is irrelevant to the result.
        size: how many to label first. Clamped to the number available.
        seed: fixed in ``configs/base.yaml``; changing it reshuffles everything
            and should not be done once labeling has started.

    Returns:
        The chosen queries, in the hash order they were chosen in. That order is
        also the order they are served for labeling, which keeps a resumed
        session on the same path through the set.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if len({q.query_id for q in gold_queries}) != len(gold_queries):
        raise ValueError("duplicate query_id in gold_queries; cannot sample reproducibly")
    ordered = sorted(gold_queries, key=lambda q: hash_key(seed, "subset", q.query_id))
    return tuple(ordered[:size])


def retest_subset(
    qrels: Qrels,
    size: int,
    seed: int,
) -> tuple[QueryId, ...]:
    """Pick already-labeled queries to re-label for the self-agreement check.

    The gold set has a single annotator, so there is no second rater to compare
    against. The honest substitute is the same rater, later: re-label a random
    subset weeks after the first pass and report agreement with yourself
    (``metrics.cohens_kappa``). That number is the project's only evidence that
    the labels mean something stable, so the *subset* has to be random for the
    same reasons :func:`select_label_subset` does — re-labeling the pairs you
    remember being easy would report an agreement figure that measures memory,
    not consistency.

    Only queries that already have at least one judged pair are eligible;
    there is nothing to re-test on an unlabeled query. The ``"retest"`` salt
    makes this ordering independent of the labeling order, so the re-tested
    queries are not simply the ones labeled first.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    eligible = [qid for qid, judged in qrels.items() if judged]
    ordered = sorted(eligible, key=lambda qid: hash_key(seed, "retest", qid))
    return tuple(ordered[:size])


# ─────────────────────────────────────────────────────────────────────────────
# The pool
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PoolPair:
    """One (query, business) pair awaiting a grade.

    ``systems`` and ``best_rank`` are provenance, kept for the pool statistics
    and for failure analysis ("this pair was only ever surfaced by the dense
    retriever"). Neither is shown while labeling and neither affects the order
    pairs are served in — see :func:`build_pool` for why that separation is
    deliberate.
    """

    query_id: QueryId
    business_id: BusinessId
    systems: tuple[str, ...]
    best_rank: int  # 1-based, the best rank this pair reached in any system


@dataclass(frozen=True, slots=True)
class PoolStats:
    """What the pool costs and where it came from.

    Reported before a labeling session starts, because "this is 420 judgements"
    is information the owner needs *before* committing an evening, and because
    the per-system contribution answers an interview question directly: how much
    of the pool did each stage uniquely surface?
    """

    n_systems: int
    depth: int
    n_queries: int
    n_pooled: int  # distinct pairs in the pool, before excluding judged ones
    n_already_judged: int
    n_new: int  # == len(pool.pairs): what the owner is actually asked
    new_per_query: Mapping[QueryId, int] = field(default_factory=dict)
    # Pairs (new ones only) that exactly one system surfaced, per system. A
    # system with a high number here is the reason the pool grew.
    unique_contribution: Mapping[str, int] = field(default_factory=dict)
    # New pairs each system surfaced at all, per system; they overlap.
    contribution: Mapping[str, int] = field(default_factory=dict)

    @property
    def mean_new_per_query(self) -> float | None:
        """Average judgements per query, or None when there are no queries.

        None rather than 0.0 — a mean over nothing is undefined, and a printed
        0.0 reads like a measurement.
        """
        if not self.new_per_query:
            return None
        return sum(self.new_per_query.values()) / len(self.new_per_query)


@dataclass(frozen=True, slots=True)
class Pool:
    """Pairs to judge, in the order they will be served, plus the statistics."""

    pairs: tuple[PoolPair, ...]
    stats: PoolStats

    def __len__(self) -> int:
        return len(self.pairs)

    def pairs_for(self, query_id: QueryId) -> tuple[PoolPair, ...]:
        return tuple(p for p in self.pairs if p.query_id == query_id)

    @property
    def query_ids(self) -> tuple[QueryId, ...]:
        """Query ids with at least one pair, in serving order."""
        return tuple(dict.fromkeys(p.query_id for p in self.pairs))


def build_pool(
    runs: Runs,
    depth: int,
    *,
    qrels: Qrels | None = None,
    query_ids: Sequence[QueryId] | None = None,
    shuffle_seed: int = 42,
) -> Pool:
    """Pool the top-``depth`` results of every system into pairs to judge.

    Args:
        runs: ``{system_name: {query_id: [business_id, ...]}}``, each run in
            rank order, best first.
        depth: how deep to pool per system (``cfg.evaluation.pool_depth``, 20).
            Deeper pools are less biased against future systems and cost
            linearly more labeling time; 20 is TREC-ish and matches the 10
            results the system actually shows.
        qrels: pairs already graded. These are **excluded** — incremental
            pooling. Pass the loaded ``eval/qrels.csv``.
        query_ids: restrict and order the output by these ids (the label
            subset). Ids absent from every run contribute nothing. When omitted,
            every query id appearing in any run is pooled, in sorted id order.
        shuffle_seed: seed for the within-query serving order.

    Returns:
        A :class:`Pool`, grouped by query in ``query_ids`` order.

    ### Two ordering decisions

    **Grouped by query.** All of one query's candidates are served together.
    Judging "cozy spot to read" means holding a standard in your head — how
    quiet is quiet enough — and swapping queries every pair means rebuilding
    that standard every pair. Grouping is both faster and more self-consistent,
    and self-consistency is exactly what the test-retest check measures.

    **Shuffled within a query, not ranked.** Pairs are ordered by a hash of
    (seed, query id, business id), which is deterministic but unrelated to how
    any system ranked them. Serving them best-first instead would be faster to
    judge and would corrupt the labels: a grader who sees good results first
    anchors high and drifts down the list, so the *label* would encode the
    system's own ranking. The pool would then agree with whichever system
    supplied the order, and the evaluation would be partly marking its own
    homework. Determinism is kept because a resumed session must continue along
    the same path, and because the pool has to be reproducible months later.
    """
    if depth < 1:
        raise ValueError(f"pool depth must be at least 1, got {depth}")
    if not runs:
        raise ValueError("no runs to pool; pooling needs at least one system's results")

    judged: Qrels = qrels or {}

    if query_ids is None:
        all_ids: set[QueryId] = set()
        for run in runs.values():
            all_ids.update(run)
        ordered_query_ids: list[QueryId] = sorted(all_ids)
    else:
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_ids contains duplicates")
        ordered_query_ids = list(query_ids)

    pairs: list[PoolPair] = []
    new_per_query: dict[QueryId, int] = {}
    contribution: dict[str, int] = dict.fromkeys(runs, 0)
    unique_contribution: dict[str, int] = dict.fromkeys(runs, 0)
    n_pooled = 0
    n_already_judged = 0

    for query_id in ordered_query_ids:
        # business_id -> systems that surfaced it, and its best rank anywhere.
        surfaced: dict[BusinessId, list[str]] = {}
        best_rank: dict[BusinessId, int] = {}
        for system, run in runs.items():
            ranked = run.get(query_id)
            if not ranked:
                continue
            # Dedupe before truncating: a run that repeats a business id would
            # otherwise get a shallower effective pool than the others, making
            # the depth inconsistent across systems.
            top = dedupe_preserving_order(ranked)[:depth]
            for rank, business_id in enumerate(top, start=1):
                surfaced.setdefault(business_id, []).append(system)
                if business_id not in best_rank or rank < best_rank[business_id]:
                    best_rank[business_id] = rank

        n_pooled += len(surfaced)
        already = judged.get(query_id, {})
        unjudged = [bid for bid in surfaced if bid not in already]
        n_already_judged += len(surfaced) - len(unjudged)

        # Hash order: deterministic, and uncorrelated with every system's rank.
        unjudged.sort(key=lambda bid: hash_key(shuffle_seed, "pool", query_id, bid))
        for business_id in unjudged:
            systems = tuple(surfaced[business_id])
            pairs.append(
                PoolPair(
                    query_id=query_id,
                    business_id=business_id,
                    systems=systems,
                    best_rank=best_rank[business_id],
                )
            )
            for system in systems:
                contribution[system] += 1
            if len(systems) == 1:
                unique_contribution[systems[0]] += 1

        new_per_query[query_id] = len(unjudged)

    stats = PoolStats(
        n_systems=len(runs),
        depth=depth,
        n_queries=len(ordered_query_ids),
        n_pooled=n_pooled,
        n_already_judged=n_already_judged,
        n_new=len(pairs),
        new_per_query=new_per_query,
        unique_contribution=unique_contribution,
        contribution=contribution,
    )
    return Pool(pairs=tuple(pairs), stats=stats)


def build_retest_pool(
    qrels: Qrels,
    already_retested: Qrels,
    *,
    size: int,
    seed: int,
) -> Pool:
    """Re-serve already-graded pairs for the test-retest self-agreement check.

    Deliberately *not* :func:`build_pool`: that function's entire job is to
    exclude judged pairs, and this mode's job is to re-serve exactly them. Pairs
    already present in the retest file are excluded, so a retest session resumes
    like a normal one.

    The original grades are not an argument to this function and never reach the
    caller through it — not hidden in the interface, simply absent. There is no
    path by which the previous grade could be displayed while re-labeling, which
    is the one property that makes the resulting agreement number mean anything.
    """
    query_ids = retest_subset(qrels, size, seed)
    pairs: list[PoolPair] = []
    per_query: dict[QueryId, int] = {}
    for query_id in query_ids:
        done = already_retested.get(query_id, {})
        candidates = [bid for bid in qrels.get(query_id, {}) if bid not in done]
        # A different salt from build_pool's: re-serving in the original order
        # would cue the labeler to the original session's rhythm.
        candidates.sort(key=lambda bid: hash_key(seed, "retest-order", query_id, bid))
        per_query[query_id] = len(candidates)
        pairs.extend(
            PoolPair(query_id=query_id, business_id=bid, systems=("retest",), best_rank=1)
            for bid in candidates
        )
    stats = PoolStats(
        n_systems=1,
        depth=0,  # not a depth-limited pool: it re-serves what was already judged
        n_queries=len(query_ids),
        n_pooled=sum(len(qrels.get(q, {})) for q in query_ids),
        n_already_judged=sum(len(already_retested.get(q, {})) for q in query_ids),
        n_new=len(pairs),
        new_per_query=per_query,
        unique_contribution={"retest": len(pairs)},
        contribution={"retest": len(pairs)},
    )
    return Pool(pairs=tuple(pairs), stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# Labeling effort
# ─────────────────────────────────────────────────────────────────────────────


def measure_pace(
    judgements: Sequence[Judgement],
    *,
    max_gap_s: float = DEFAULT_MAX_JUDGEMENT_GAP_S,
) -> float | None:
    """Median seconds per judgement, measured from the owner's own qrels.

    Returns None when there is not enough history to measure
    (``MIN_INTERVALS_FOR_PACE`` usable intervals). None means "unmeasured" and
    callers must print it as such: a default like "15 seconds per pair" would be
    an invented number, and the moment it appears in an ETA the owner starts
    planning an evening around a figure nobody measured.

    Median rather than mean, and gaps over ``max_gap_s`` dropped entirely: the
    distribution has a long right tail of interruptions, and one coffee break
    would otherwise double the estimate.
    """
    parsed = (j.labeled_at_dt for j in judgements)
    stamps = sorted(dt for dt in parsed if dt is not None)
    gaps = [
        (b - a).total_seconds()
        for a, b in pairwise(stamps)
        if 0.0 < (b - a).total_seconds() <= max_gap_s
    ]
    if len(gaps) < MIN_INTERVALS_FOR_PACE:
        return None
    return float(median(gaps))


def estimate_minutes(n_pairs: int, seconds_per_pair: float | None) -> float | None:
    """Estimated labeling time, or None when the pace is unmeasured.

    Deliberately returns None rather than guessing. Every caller has to handle
    the None branch and say "unmeasured" out loud.
    """
    if seconds_per_pair is None or n_pairs < 0:
        return None
    return n_pairs * seconds_per_pair / 60.0


def format_pool_report(
    pool: Pool,
    *,
    seconds_per_pair: float | None = None,
) -> list[str]:
    """Plain-text pool statistics, one line per fact.

    Returns lines instead of printing them so this stays testable and so the
    Rich-formatted CLI and a plain ``make`` log can share the same wording.
    """
    s = pool.stats
    lines = [
        f"Systems pooled:        {s.n_systems} at depth {s.depth}",
        f"Queries:               {s.n_queries}",
        f"Distinct pairs pooled: {s.n_pooled}",
        f"Already judged:        {s.n_already_judged} (skipped — never re-graded)",
        f"New judgements:        {s.n_new}",
    ]
    mean_new = s.mean_new_per_query
    if mean_new is not None:
        per_query = sorted(s.new_per_query.values())
        lines.append(
            f"New per query:         mean {mean_new:.1f}, min {per_query[0]}, max {per_query[-1]}"
        )
    if s.n_systems > 1:
        for system in sorted(s.contribution):
            lines.append(
                f"  {system}: {s.contribution[system]} new pairs, "
                f"{s.unique_contribution[system]} surfaced by it alone"
            )

    minutes = estimate_minutes(s.n_new, seconds_per_pair)
    if minutes is None:
        lines.append(
            "Estimated time:        unmeasured (no labeling pace on record yet; "
            "it is measured from the timestamps in your qrels file once you have "
            f"graded {MIN_INTERVALS_FOR_PACE + 1} pairs)"
        )
    else:
        lines.append(
            f"Estimated time:        {minutes:.0f} min "
            f"at your measured median of {seconds_per_pair:.1f}s per pair"
        )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# qrels: the label file
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Judgement:
    """One graded pair, as one row of ``eval/qrels.csv``.

    ``labeled_at`` and ``labeler`` are not decoration. The timestamps are what
    :func:`measure_pace` reads to produce a real ETA and what dates the
    test-retest gap ("relabeled 18 days later"), and ``labeler`` is how a second
    annotator could ever be added without the two being silently merged.
    """

    query_id: QueryId
    business_id: BusinessId
    relevance: Grade
    labeled_at: str  # ISO-8601, UTC
    labeler: str

    @property
    def labeled_at_dt(self) -> datetime | None:
        """Parsed timestamp, or None if the string is not ISO-8601.

        Tolerant on read because a hand-edited row should not crash the pace
        estimate; the grade itself is validated strictly.
        """
        try:
            return datetime.fromisoformat(self.labeled_at)
        except (TypeError, ValueError):
            return None

    @property
    def key(self) -> tuple[QueryId, BusinessId]:
        return (self.query_id, self.business_id)


def _parse_grade(raw: str, where: str) -> Grade:
    try:
        grade = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: relevance {raw!r} is not an integer") from exc
    if grade not in VALID_GRADES:
        raise ValueError(
            f"{where}: relevance {grade} is outside {sorted(VALID_GRADES)}. "
            f"0 = irrelevant, 1 = partially relevant, 2 = highly relevant."
        )
    return grade


def read_judgements(path: str | Path) -> list[Judgement]:
    """Every row of a qrels CSV, in file order, duplicates included.

    A missing file is an empty list, not an error: before the first labeling
    session there is legitimately no qrels file, and the CLI has to be able to
    start from nothing.

    Rows are returned in order *with* duplicates because order carries meaning
    here — a later row supersedes an earlier one for the same pair (see
    :func:`load_qrels`) and the raw sequence is what the pace estimate needs.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[Judgement] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(QRELS_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{p} is missing column(s) {sorted(missing)}; "
                f"header must be {','.join(QRELS_COLUMNS)}"
            )
        for lineno, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            business_id = (row.get("business_id") or "").strip()
            if not query_id and not business_id:
                continue
            if not query_id or not business_id:
                raise ValueError(f"{p}:{lineno} has an empty query_id or business_id")
            out.append(
                Judgement(
                    query_id=query_id,
                    business_id=business_id,
                    relevance=_parse_grade(row.get("relevance", ""), f"{p}:{lineno}"),
                    labeled_at=(row.get("labeled_at") or "").strip(),
                    labeler=(row.get("labeler") or "").strip(),
                )
            )
    return out


def load_qrels(path: str | Path) -> dict[QueryId, dict[BusinessId, Grade]]:
    """qrels in the shape ``eval/metrics.py`` expects.

    **Last row wins** for a repeated pair. The file is append-only, so a
    correction is a new row rather than an edit, and the most recent opinion is
    the one that counts. The labeling CLI's undo truncates its own row instead of
    appending a correction, so in normal use duplicates do not occur at all —
    this rule exists for hand-edited files and for a crash between an undo and
    the following grade.
    """
    qrels: dict[QueryId, dict[BusinessId, Grade]] = {}
    for j in read_judgements(path):
        qrels.setdefault(j.query_id, {})[j.business_id] = j.relevance
    return qrels


def judged_pairs(path: str | Path) -> set[tuple[QueryId, BusinessId]]:
    """Pairs with any grade on record. Used to resume and to exclude."""
    return {(qid, bid) for qid, row in load_qrels(path).items() for bid in row}


class QrelsWriter:
    """Append-only writer for a qrels CSV, with a single-step undo.

    **Why append-and-flush after every single judgement.** A labeling session is
    hours of irreplaceable human work. Buffering it in memory and writing at the
    end means a crash, a closed laptop lid, or a stray Ctrl-C destroys all of
    it, and the owner will not label those pairs a second time with the same
    care. So every grade is written, flushed, and ``fsync``-ed before the next
    candidate is shown. The cost is one tiny synchronous write per keystroke,
    which is invisible next to the seconds a human spends reading a candidate.

    **Why undo truncates rather than appending a correction.** A mis-keyed grade
    is the one error the owner will want to fix immediately. Appending a
    superseding row would work (``load_qrels`` takes the last one) but leaves the
    file with two rows for one pair, which makes "how many pairs have I judged?"
    a question you have to think about. Truncating back to the byte offset
    recorded before the write removes it cleanly, and is only ever applied to
    rows this process wrote — never to pre-existing history.
    """

    def __init__(self, path: str | Path, labeler: str) -> None:
        self.path = Path(path)
        self.labeler = labeler
        self._fh: TextIO | None = None
        # Byte offsets and rows written in this session, most recent last.
        self._offsets: list[int] = []
        self._written: list[Judgement] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> QrelsWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a+", newline="", encoding="utf-8")
        if is_new:
            csv.writer(self._fh).writerow(QRELS_COLUMNS)
            self._sync()
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> QrelsWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── writing ──────────────────────────────────────────────────────────────

    def append(
        self,
        query_id: QueryId,
        business_id: BusinessId,
        relevance: Grade,
        *,
        when: datetime | None = None,
    ) -> Judgement:
        """Write one judgement and make it durable before returning."""
        if self._fh is None:
            raise RuntimeError("QrelsWriter is not open; use it as a context manager")
        if relevance not in VALID_GRADES:
            raise ValueError(f"relevance {relevance!r} outside {sorted(VALID_GRADES)}")
        stamp = (when or datetime.now(UTC)).isoformat(timespec="seconds")
        judgement = Judgement(
            query_id=query_id,
            business_id=business_id,
            relevance=relevance,
            labeled_at=stamp,
            labeler=self.labeler,
        )
        self._fh.seek(0, os.SEEK_END)
        self._offsets.append(self._fh.tell())
        csv.writer(self._fh).writerow(
            [
                judgement.query_id,
                judgement.business_id,
                judgement.relevance,
                judgement.labeled_at,
                judgement.labeler,
            ]
        )
        self._sync()
        self._written.append(judgement)
        return judgement

    def undo(self) -> Judgement | None:
        """Remove the most recent judgement written by this session.

        Returns the removed judgement, or None when this session has written
        nothing left to undo. Rows from earlier sessions are never touched: an
        undo that could reach into last week's labels is a footgun, and the
        owner can edit the CSV by hand if they really mean it.
        """
        if self._fh is None:
            raise RuntimeError("QrelsWriter is not open; use it as a context manager")
        if not self._written:
            return None
        offset = self._offsets.pop()
        removed = self._written.pop()
        self._fh.truncate(offset)
        self._fh.seek(offset)
        self._sync()
        return removed

    @property
    def n_written(self) -> int:
        """Judgements written in this session and not undone."""
        return len(self._written)

    @property
    def written(self) -> tuple[Judgement, ...]:
        return tuple(self._written)

    def _sync(self) -> None:
        """flush() hands the bytes to the OS; fsync() gets them onto the disk.

        Without the fsync a power loss or a hard kill can lose minutes of
        already-"saved" labels, which is precisely the failure this class exists
        to prevent.
        """
        if self._fh is None:  # pragma: no cover - callers check first
            raise RuntimeError("QrelsWriter is not open")
        self._fh.flush()
        os.fsync(self._fh.fileno())


def agreement_pairs(
    first: Iterable[Judgement],
    second: Iterable[Judgement],
) -> tuple[list[Grade], list[Grade], tuple[tuple[QueryId, BusinessId], ...]]:
    """Align two label sets on the pairs both graded, for the kappa check.

    Returns ``(grades_first, grades_second, keys)`` — two parallel grade lists
    ready for ``metrics.cohens_kappa``, plus the keys they came from so a
    disagreement can be looked up and argued with. Pairs only one side graded
    are dropped: there is no agreement to measure on them, and filling the gap
    with a 0 would manufacture agreement out of absence.

    If a side grades the same pair twice, its last grade is used — the same
    last-write-wins rule :func:`load_qrels` applies.
    """
    a = {j.key: j.relevance for j in first}
    b = {j.key: j.relevance for j in second}
    keys = sorted(a.keys() & b.keys())
    return [a[k] for k in keys], [b[k] for k in keys], tuple(keys)
