"""Tests for the labeling infrastructure: pooling, subset selection, qrels I/O.

These tests guard properties that, if they broke, would not crash anything and
would not be noticed until the numbers were already in a README:

* a pool that silently re-serves a judged pair wastes the owner's hours and
  produces two labels for one pair;
* a "random" subset that correlates with query order or type biases every
  reported metric in a direction nobody can see from the metric;
* a qrels writer that loses the last row loses irreplaceable human work.

The statistical tests (``TestSubsetIsUnbiased``) are deliberately aggregate: a
single draw from a random sampler tells you nothing, so they average over many
seeds and assert against the population value with a tolerance several standard
errors wide. They are deterministic — the seeds are fixed — so they cannot
flake.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from concierge.types import GoldQuery, QueryType
from eval import pooling
from eval.pooling import (
    Judgement,
    QrelsWriter,
    agreement_pairs,
    build_pool,
    build_retest_pool,
    estimate_minutes,
    format_pool_report,
    load_gold_queries,
    load_qrels,
    measure_pace,
    read_judgements,
    retest_subset,
    select_label_subset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────


def gold(n: int, *, types: Sequence[QueryType] | None = None) -> list[GoldQuery]:
    """``n`` gold queries, ids ``q001``… and types cycling through the four.

    Ids are zero-padded so sorted order equals written order; several tests use
    the position of a query in this list as "the order the owner wrote them in".
    """
    cycle = tuple(types) if types else tuple(QueryType)
    return [
        GoldQuery(
            query_id=f"q{i:03d}",
            query=f"test query {i}",
            query_type=cycle[i % len(cycle)],
        )
        for i in range(1, n + 1)
    ]


def judgement(
    query_id: str,
    business_id: str,
    relevance: int,
    *,
    at: datetime | None = None,
    labeler: str = "owner",
) -> Judgement:
    stamp = (at or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)).isoformat(timespec="seconds")
    return Judgement(
        query_id=query_id,
        business_id=business_id,
        relevance=relevance,
        labeled_at=stamp,
        labeler=labeler,
    )


def write_qrels(path: Path, rows: Sequence[Sequence[object]]) -> Path:
    """Write a qrels CSV by hand, bypassing the writer, for read-path tests."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(pooling.QRELS_COLUMNS)
        writer.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Pool arithmetic
# ─────────────────────────────────────────────────────────────────────────────


class TestPoolArithmetic:
    def test_pool_is_the_union_of_each_system_top_depth(self) -> None:
        runs = {
            "bm25": {"q001": ["a", "b", "c"]},
            "dense": {"q001": ["c", "d", "e"]},
        }
        pool = build_pool(runs, depth=2)
        # depth 2: bm25 gives a,b and dense gives c,d. "e" is below depth and
        # "c" is pooled once even though bm25 would have reached it at depth 3.
        assert {p.business_id for p in pool.pairs} == {"a", "b", "c", "d"}
        assert pool.stats.n_pooled == 4
        assert pool.stats.n_new == 4
        assert len(pool) == 4

    def test_depth_truncates_every_system_independently(self) -> None:
        runs = {
            "shallow": {"q001": ["a"]},
            "deep": {"q001": ["b", "c", "d", "e", "f"]},
        }
        pool = build_pool(runs, depth=3)
        assert {p.business_id for p in pool.pairs} == {"a", "b", "c", "d"}

    def test_repeated_id_in_a_run_does_not_shrink_its_effective_depth(self) -> None:
        # A buggy run that lists "a" twice must still contribute `depth`
        # distinct businesses, or the pool depth differs between systems and
        # the comparison is unfair in a way no metric would reveal.
        runs = {"sloppy": {"q001": ["a", "a", "b", "c"]}}
        pool = build_pool(runs, depth=3)
        assert [p.business_id for p in sorted(pool.pairs, key=lambda p: p.business_id)] == [
            "a",
            "b",
            "c",
        ]

    def test_provenance_records_every_system_and_the_best_rank(self) -> None:
        runs = {
            "bm25": {"q001": ["x", "y"]},
            "dense": {"q001": ["y", "z"]},
        }
        pool = build_pool(runs, depth=2)
        by_id = {p.business_id: p for p in pool.pairs}
        assert set(by_id["y"].systems) == {"bm25", "dense"}
        # bm25 ranked y second, dense ranked it first → best rank 1.
        assert by_id["y"].best_rank == 1
        assert by_id["z"].systems == ("dense",)
        assert by_id["z"].best_rank == 2

    def test_pairs_are_grouped_by_query(self) -> None:
        runs = {"bm25": {"q001": ["a", "b"], "q002": ["c", "d"]}}
        pool = build_pool(runs, depth=2, query_ids=["q002", "q001"])
        assert [p.query_id for p in pool.pairs] == ["q002", "q002", "q001", "q001"]
        assert pool.query_ids == ("q002", "q001")

    def test_query_ids_restricts_the_pool(self) -> None:
        runs = {"bm25": {"q001": ["a"], "q002": ["b"], "q003": ["c"]}}
        pool = build_pool(runs, depth=5, query_ids=["q001", "q003"])
        assert {p.query_id for p in pool.pairs} == {"q001", "q003"}
        assert pool.stats.n_queries == 2

    def test_a_requested_query_with_no_results_contributes_nothing(self) -> None:
        runs = {"bm25": {"q001": ["a"]}}
        pool = build_pool(runs, depth=5, query_ids=["q001", "q999"])
        assert pool.stats.new_per_query == {"q001": 1, "q999": 0}
        assert pool.pairs_for("q999") == ()

    @pytest.mark.parametrize("depth", [0, -1])
    def test_depth_below_one_is_rejected(self, depth: int) -> None:
        with pytest.raises(ValueError, match="depth"):
            build_pool({"bm25": {"q001": ["a"]}}, depth=depth)

    def test_empty_runs_are_rejected(self) -> None:
        # Pooling nothing would hand the owner an empty queue and look like
        # "all done" instead of "you forgot to produce a run".
        with pytest.raises(ValueError, match="at least one system"):
            build_pool({}, depth=10)

    def test_duplicate_query_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            build_pool({"bm25": {"q001": ["a"]}}, depth=5, query_ids=["q001", "q001"])


class TestIncrementalPooling:
    def test_already_judged_pairs_are_excluded(self) -> None:
        runs = {"bm25": {"q001": ["a", "b", "c"]}}
        qrels = {"q001": {"b": 2}}
        pool = build_pool(runs, depth=3, qrels=qrels)
        assert {p.business_id for p in pool.pairs} == {"a", "c"}
        assert pool.stats.n_pooled == 3
        assert pool.stats.n_already_judged == 1
        assert pool.stats.n_new == 2

    def test_a_grade_of_zero_still_counts_as_judged(self) -> None:
        # Guards the classic falsy-value bug: `if qrels[qid][bid]` would treat a
        # 0 as unjudged and re-serve every irrelevant pair forever.
        pool = build_pool({"bm25": {"q001": ["a"]}}, depth=1, qrels={"q001": {"a": 0}})
        assert pool.pairs == ()
        assert pool.stats.n_already_judged == 1

    def test_judgements_for_another_query_do_not_leak_across(self) -> None:
        runs = {"bm25": {"q001": ["a"], "q002": ["a"]}}
        pool = build_pool(runs, depth=1, qrels={"q001": {"a": 2}})
        assert [(p.query_id, p.business_id) for p in pool.pairs] == [("q002", "a")]

    def test_adding_a_system_asks_only_about_what_it_newly_surfaced(self) -> None:
        # The scenario the owner actually lives through: label the BM25 pool,
        # then add dense retrieval and label again.
        bm25 = {"q001": ["a", "b"]}
        first = build_pool({"bm25": bm25}, depth=2)
        labeled = {"q001": {p.business_id: 1 for p in first.pairs}}

        second = build_pool(
            {"bm25": bm25, "dense": {"q001": ["b", "c"]}},
            depth=2,
            qrels=labeled,
        )
        assert [p.business_id for p in second.pairs] == ["c"]
        assert second.stats.n_already_judged == 2
        assert second.stats.unique_contribution == {"bm25": 0, "dense": 1}
        assert second.stats.contribution == {"bm25": 0, "dense": 1}

    def test_relabeling_is_never_requested_twice_in_a_row(self) -> None:
        runs = {"bm25": {"q001": ["a", "b", "c"]}}
        pool = build_pool(runs, depth=3)
        labeled = {"q001": {p.business_id: 2 for p in pool.pairs}}
        again = build_pool(runs, depth=3, qrels=labeled)
        assert again.pairs == ()
        assert again.stats.n_new == 0


class TestServingOrder:
    """The order candidates are shown in must not come from any system's rank."""

    def test_order_is_deterministic_across_calls(self) -> None:
        runs = {"bm25": {"q001": [f"b{i:02d}" for i in range(20)]}}
        first = build_pool(runs, depth=20, shuffle_seed=42)
        second = build_pool(runs, depth=20, shuffle_seed=42)
        assert [p.business_id for p in first.pairs] == [p.business_id for p in second.pairs]

    def test_order_does_not_follow_the_system_ranking(self) -> None:
        ranked = [f"b{i:02d}" for i in range(20)]
        pool = build_pool({"bm25": {"q001": ranked}}, depth=20, shuffle_seed=42)
        served = [p.business_id for p in pool.pairs]
        assert set(served) == set(ranked)
        # Anchoring is the risk: a labeler shown the best results first grades
        # high and drifts down. The served order must differ from rank order.
        assert served != ranked
        assert served != list(reversed(ranked))

    def test_a_different_seed_gives_a_different_order(self) -> None:
        ranked = [f"b{i:02d}" for i in range(20)]
        a = build_pool({"bm25": {"q001": ranked}}, depth=20, shuffle_seed=1)
        b = build_pool({"bm25": {"q001": ranked}}, depth=20, shuffle_seed=2)
        assert [p.business_id for p in a.pairs] != [p.business_id for p in b.pairs]


class TestPoolStats:
    def test_counts_and_mean(self) -> None:
        runs = {
            "bm25": {"q001": ["a", "b"], "q002": ["c"]},
            "dense": {"q001": ["b"], "q002": ["c", "d", "e"]},
        }
        pool = build_pool(runs, depth=2)
        stats = pool.stats
        assert stats.n_systems == 2
        assert stats.depth == 2
        assert stats.n_queries == 2
        # q001 pools a, b; q002 pools c, d — dense's "e" is below depth 2.
        assert stats.new_per_query == {"q001": 2, "q002": 2}
        assert stats.n_new == 4
        assert stats.mean_new_per_query == pytest.approx(2.0)
        # Only "a" (bm25) and "d" (dense) were surfaced by a single system.
        assert stats.unique_contribution == {"bm25": 1, "dense": 1}
        assert stats.contribution == {"bm25": 3, "dense": 3}

    def test_mean_is_none_rather_than_zero_without_queries(self) -> None:
        # A printed 0.0 reads like a measurement; None reads like "no data".
        pool = build_pool({"bm25": {"q001": ["a"]}}, depth=1, query_ids=[])
        assert pool.stats.mean_new_per_query is None

    def test_report_says_unmeasured_without_a_pace(self) -> None:
        pool = build_pool({"bm25": {"q001": ["a", "b"]}}, depth=2)
        text = "\n".join(format_pool_report(pool))
        assert "unmeasured" in text
        assert "New judgements:        2" in text

    def test_report_quotes_the_measured_pace(self) -> None:
        pool = build_pool({"bm25": {"q001": ["a", "b"]}}, depth=2)
        text = "\n".join(format_pool_report(pool, seconds_per_pair=30.0))
        assert "1 min" in text
        assert "30.0s per pair" in text


# ─────────────────────────────────────────────────────────────────────────────
# Label subset selection
# ─────────────────────────────────────────────────────────────────────────────


class TestSubsetSelection:
    def test_fixed_seed_is_stable(self) -> None:
        queries = gold(150)
        first = select_label_subset(queries, 50, seed=42)
        second = select_label_subset(queries, 50, seed=42)
        assert [q.query_id for q in first] == [q.query_id for q in second]
        assert len(first) == 50

    def test_subsets_are_nested_as_the_size_grows(self) -> None:
        # D004 plans to expand the labeled set. Labels already collected must
        # stay inside the larger subset or those hours are wasted.
        queries = gold(150)
        fifty = [q.query_id for q in select_label_subset(queries, 50, seed=42)]
        hundred = [q.query_id for q in select_label_subset(queries, 100, seed=42)]
        assert hundred[:50] == fifty

    def test_result_is_independent_of_row_order(self) -> None:
        queries = gold(60)
        shuffled = list(reversed(queries))
        assert [q.query_id for q in select_label_subset(queries, 20, seed=7)] == [
            q.query_id for q in select_label_subset(shuffled, 20, seed=7)
        ]

    def test_adding_a_query_does_not_evict_unrelated_queries(self) -> None:
        # Membership depends only on a query's own id, so appending a query can
        # at worst displace one at the boundary — it cannot reshuffle the set.
        queries = gold(60)
        before = {q.query_id for q in select_label_subset(queries, 20, seed=3)}
        extended = [*queries, GoldQuery("q999", "a new query", QueryType.MESSY)]
        after = {q.query_id for q in select_label_subset(extended, 20, seed=3)}
        assert len(before - after) <= 1

    def test_size_is_clamped_to_what_exists(self) -> None:
        queries = gold(5)
        assert len(select_label_subset(queries, 50, seed=1)) == 5
        assert select_label_subset(queries, 0, seed=1) == ()
        assert select_label_subset([], 10, seed=1) == ()

    def test_different_seeds_select_differently(self) -> None:
        queries = gold(150)
        a = [q.query_id for q in select_label_subset(queries, 50, seed=1)]
        b = [q.query_id for q in select_label_subset(queries, 50, seed=2)]
        assert a != b

    def test_duplicate_ids_are_rejected(self) -> None:
        duplicated = [*gold(3), GoldQuery("q001", "same id again", QueryType.VIBE)]
        with pytest.raises(ValueError, match="duplicate query_id"):
            select_label_subset(duplicated, 2, seed=1)

    def test_negative_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            select_label_subset(gold(3), -1, seed=1)


class TestSubsetIsUnbiased:
    """The subset must be a genuine random sample, not merely reproducible.

    A deterministic-but-biased selector (the first 50, the shortest, every third
    row) would pass every test above. These check the property that actually
    matters: the sample looks like the population.
    """

    SEEDS = range(200)

    def test_not_correlated_with_the_order_queries_were_written_in(self) -> None:
        queries = gold(120)
        position = {q.query_id: i / (len(queries) - 1) for i, q in enumerate(queries)}
        means = []
        for seed in self.SEEDS:
            chosen = select_label_subset(queries, 20, seed=seed)
            means.append(sum(position[q.query_id] for q in chosen) / len(chosen))
        # Uniform over [0, 1] → expected mean position 0.5. The standard error
        # of this average over 200 seeds is well under 0.01, so 0.03 is a wide
        # tolerance that still fails loudly for any order-based selector
        # (first-50 would give 0.08, last-50 would give 0.92).
        assert abs(sum(means) / len(means) - 0.5) < 0.03

    def test_query_type_proportions_match_the_population(self) -> None:
        queries = gold(120)  # 30 of each of the four types
        fractions: dict[QueryType, list[float]] = {t: [] for t in QueryType}
        for seed in self.SEEDS:
            chosen = select_label_subset(queries, 20, seed=seed)
            for query_type in QueryType:
                share = sum(1 for q in chosen if q.query_type is query_type) / len(chosen)
                fractions[query_type].append(share)
        for query_type, shares in fractions.items():
            mean = sum(shares) / len(shares)
            assert abs(mean - 0.25) < 0.03, f"{query_type} over/under-sampled: {mean:.3f}"

    def test_every_query_can_be_selected(self) -> None:
        # A selector keyed on something structural (an id prefix, say) would
        # make some queries unreachable no matter the seed.
        queries = gold(40)
        reachable: set[str] = set()
        for seed in self.SEEDS:
            reachable.update(q.query_id for q in select_label_subset(queries, 10, seed=seed))
        assert reachable == {q.query_id for q in queries}


# ─────────────────────────────────────────────────────────────────────────────
# Effort estimates
# ─────────────────────────────────────────────────────────────────────────────


class TestEffortEstimates:
    def test_estimate_is_none_when_the_pace_is_unknown(self) -> None:
        assert estimate_minutes(500, None) is None

    def test_estimate_arithmetic(self) -> None:
        assert estimate_minutes(120, 30.0) == pytest.approx(60.0)
        assert estimate_minutes(0, 30.0) == 0.0

    def test_pace_needs_enough_history(self) -> None:
        base = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
        two_intervals = [
            judgement("q001", f"b{i}", 1, at=base + timedelta(seconds=20 * i)) for i in range(3)
        ]
        assert measure_pace(two_intervals) is None
        assert measure_pace([]) is None

    def test_pace_is_the_median_interval(self) -> None:
        base = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
        offsets = [0, 10, 20, 30, 50]  # intervals: 10, 10, 10, 20
        rows = [
            judgement("q001", f"b{i}", 1, at=base + timedelta(seconds=o))
            for i, o in enumerate(offsets)
        ]
        assert measure_pace(rows) == pytest.approx(10.0)

    def test_long_breaks_are_not_counted_as_slow_judgements(self) -> None:
        base = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
        offsets = [0, 15, 30, 45, 4000, 4015]  # one lunch break in the middle
        rows = [
            judgement("q001", f"b{i}", 1, at=base + timedelta(seconds=o))
            for i, o in enumerate(offsets)
        ]
        assert measure_pace(rows) == pytest.approx(15.0)

    def test_unparseable_timestamps_are_ignored_not_fatal(self) -> None:
        base = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
        rows = [
            judgement("q001", f"b{i}", 1, at=base + timedelta(seconds=12 * i)) for i in range(5)
        ]
        rows.append(Judgement("q001", "bX", 1, "not-a-timestamp", "owner"))
        assert measure_pace(rows) == pytest.approx(12.0)


# ─────────────────────────────────────────────────────────────────────────────
# qrels file
# ─────────────────────────────────────────────────────────────────────────────


class TestQrelsReading:
    def test_a_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "qrels.csv"
        assert read_judgements(missing) == []
        assert load_qrels(missing) == {}

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = write_qrels(
            tmp_path / "qrels.csv",
            [
                ["q001", "a", 2, "2026-01-01T12:00:00+00:00", "owner"],
                ["q001", "b", 0, "2026-01-01T12:00:30+00:00", "owner"],
                ["q002", "a", 1, "2026-01-01T12:01:00+00:00", "owner"],
            ],
        )
        assert load_qrels(path) == {"q001": {"a": 2, "b": 0}, "q002": {"a": 1}}
        assert pooling.judged_pairs(path) == {("q001", "a"), ("q001", "b"), ("q002", "a")}

    def test_the_last_row_for_a_pair_wins(self, tmp_path: Path) -> None:
        path = write_qrels(
            tmp_path / "qrels.csv",
            [
                ["q001", "a", 0, "2026-01-01T12:00:00+00:00", "owner"],
                ["q001", "a", 2, "2026-02-01T12:00:00+00:00", "owner"],
            ],
        )
        assert load_qrels(path) == {"q001": {"a": 2}}
        # The raw history keeps both rows: the file is an append-only log.
        assert [j.relevance for j in read_judgements(path)] == [0, 2]

    @pytest.mark.parametrize("bad", ["3", "-1", "", "two", "1.5"])
    def test_a_grade_outside_the_scale_is_an_error(self, tmp_path: Path, bad: str) -> None:
        # A stray grade would flow straight into NDCG's gain function and
        # inflate or deflate a reported number with no visible symptom.
        path = write_qrels(
            tmp_path / "qrels.csv", [["q001", "a", bad, "2026-01-01T12:00:00+00:00", "owner"]]
        )
        with pytest.raises(ValueError, match="relevance"):
            read_judgements(path)

    def test_a_missing_column_names_itself(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        path.write_text("query_id,business_id,relevance\nq001,a,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="labeled_at"):
            read_judgements(path)

    def test_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        path.write_text(
            "query_id,business_id,relevance,labeled_at,labeler\n"
            "q001,a,2,2026-01-01T12:00:00+00:00,owner\n"
            ",,,,\n",
            encoding="utf-8",
        )
        assert len(read_judgements(path)) == 1


class TestQrelsWriter:
    def test_creates_the_header_once(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "a", 2)
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "b", 1)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == ",".join(pooling.QRELS_COLUMNS)
        assert len(lines) == 3

    def test_every_judgement_is_on_disk_immediately(self, tmp_path: Path) -> None:
        # The crash-safety guarantee: a reader that opens the file mid-session
        # sees every grade already entered, with no flush from the writer.
        path = tmp_path / "qrels.csv"
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "a", 2)
            assert load_qrels(path) == {"q001": {"a": 2}}
            writer.append("q001", "b", 0)
            assert load_qrels(path) == {"q001": {"a": 2, "b": 0}}

    def test_records_the_labeler_and_a_timestamp(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        when = datetime(2026, 3, 4, 17, 30, tzinfo=UTC)
        with QrelsWriter(path, "aditya") as writer:
            written = writer.append("q001", "a", 1, when=when)
        assert written.labeler == "aditya"
        assert written.labeled_at.startswith("2026-03-04T17:30:00")
        assert read_judgements(path)[0] == written

    def test_undo_removes_only_the_last_row(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "a", 2)
            writer.append("q001", "b", 1)
            removed = writer.undo()
        assert removed is not None
        assert removed.business_id == "b"
        assert load_qrels(path) == {"q001": {"a": 2}}
        assert writer.n_written == 1

    def test_undo_then_regrade_leaves_one_clean_row(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "a", 0)
            writer.undo()
            writer.append("q001", "a", 2)
        assert [j.relevance for j in read_judgements(path)] == [2]

    def test_undo_cannot_reach_a_previous_session(self, tmp_path: Path) -> None:
        # Deliberate: an undo that could delete last week's labels is a footgun.
        path = tmp_path / "qrels.csv"
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "a", 2)
        with QrelsWriter(path, "owner") as writer:
            assert writer.undo() is None
        assert load_qrels(path) == {"q001": {"a": 2}}

    def test_undo_on_an_empty_session_is_harmless(self, tmp_path: Path) -> None:
        with QrelsWriter(tmp_path / "qrels.csv", "owner") as writer:
            assert writer.undo() is None
            assert writer.n_written == 0

    def test_an_invalid_grade_is_refused_at_the_keystroke(self, tmp_path: Path) -> None:
        path = tmp_path / "qrels.csv"
        with QrelsWriter(path, "owner") as writer, pytest.raises(ValueError, match="relevance"):
            writer.append("q001", "a", 5)
        assert load_qrels(path) == {}

    def test_writing_without_opening_is_an_error(self, tmp_path: Path) -> None:
        writer = QrelsWriter(tmp_path / "qrels.csv", "owner")
        with pytest.raises(RuntimeError, match="not open"):
            writer.append("q001", "a", 1)

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "qrels.csv"
        with QrelsWriter(path, "owner") as writer:
            writer.append("q001", "a", 1)
        assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Gold queries
# ─────────────────────────────────────────────────────────────────────────────


class TestGoldQueryFile:
    def write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "gold_queries.csv"
        path.write_text("query_id,query,query_type\n" + body, encoding="utf-8")
        return path

    def test_reads_queries(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "q001,cozy spot to read,vibe\nq002,good pho,specific\n")
        goldset = load_gold_queries(path)
        assert len(goldset) == 2
        assert goldset.queries[0].query_type is QueryType.VIBE
        assert goldset.by_id["q002"].query == "good pho"
        assert goldset.counts_by_type()[QueryType.SPECIFIC] == 1

    def test_example_rows_are_skipped_and_counted(self, tmp_path: Path) -> None:
        # Examples exist so the owner can see the format; they must never be
        # able to reach a metric.
        path = self.write(tmp_path, "EXAMPLE001,cozy spot to read,vibe\nq001,my real query,vibe\n")
        goldset = load_gold_queries(path)
        assert [q.query_id for q in goldset.queries] == ["q001"]
        assert goldset.n_examples_skipped == 1

    def test_examples_can_be_kept_for_a_format_check(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "EXAMPLE001,cozy spot to read,vibe\n")
        assert len(load_gold_queries(path, keep_examples=True)) == 1

    def test_an_unknown_query_type_names_the_line(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "q001,something,atmosphere\n")
        with pytest.raises(ValueError, match=r"gold_queries\.csv:2"):
            load_gold_queries(path)

    def test_a_duplicate_id_is_fatal(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "q001,first,vibe\nq001,second,messy\n")
        with pytest.raises(ValueError, match="repeats query_id"):
            load_gold_queries(path)

    def test_an_empty_query_is_fatal(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "q001,,vibe\n")
        with pytest.raises(ValueError, match="empty query"):
            load_gold_queries(path)

    def test_a_missing_file_explains_who_writes_it(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="owner writes this file"):
            load_gold_queries(tmp_path / "nope.csv")

    def test_the_shipped_file_contains_only_examples(self) -> None:
        # Guards the deliverable itself: the ~150 real queries are the owner's
        # work (CLAUDE.md section 6), so this file must ship with nothing but a
        # handful of clearly-marked examples.
        path = REPO_ROOT / "eval" / "gold_queries.csv"
        with_examples = load_gold_queries(path, keep_examples=True)
        assert len(with_examples) <= 5
        assert all(q.query_id.startswith(pooling.EXAMPLE_ID_PREFIX) for q in with_examples.queries)
        assert load_gold_queries(path).queries == ()
        # The examples should demonstrate all four types, since the owner copies
        # them as a template.
        assert {q.query_type for q in with_examples.queries} == set(QueryType)


# ─────────────────────────────────────────────────────────────────────────────
# Retest
# ─────────────────────────────────────────────────────────────────────────────


class TestRetest:
    def test_only_labeled_queries_are_eligible(self) -> None:
        qrels = {"q001": {"a": 2}, "q002": {}, "q003": {"b": 0}}
        assert set(retest_subset(qrels, 10, seed=42)) == {"q001", "q003"}

    def test_selection_is_stable_and_clamped(self) -> None:
        qrels = {f"q{i:03d}": {"a": 1} for i in range(1, 41)}
        first = retest_subset(qrels, 30, seed=42)
        assert first == retest_subset(qrels, 30, seed=42)
        assert len(first) == 30
        assert len(retest_subset(qrels, 100, seed=42)) == 40

    def test_retest_order_differs_from_the_label_order(self) -> None:
        # The retest ordering uses its own salt so re-serving does not replay
        # the original session's sequence, which would cue recall.
        queries = gold(60)
        qrels = {q.query_id: {"a": 1} for q in queries}
        labeling_order = [q.query_id for q in select_label_subset(queries, 30, seed=42)]
        assert list(retest_subset(qrels, 30, seed=42)) != labeling_order

    def test_pool_reserves_exactly_the_already_judged_pairs(self) -> None:
        qrels = {"q001": {"a": 2, "b": 0}, "q002": {"c": 1}}
        pool = build_retest_pool(qrels, {}, size=10, seed=42)
        assert {(p.query_id, p.business_id) for p in pool.pairs} == {
            ("q001", "a"),
            ("q001", "b"),
            ("q002", "c"),
        }
        assert pool.stats.n_new == 3

    def test_pool_resumes_past_what_was_already_retested(self) -> None:
        qrels = {"q001": {"a": 2, "b": 0}}
        pool = build_retest_pool(qrels, {"q001": {"a": 1}}, size=10, seed=42)
        assert [p.business_id for p in pool.pairs] == ["b"]
        assert pool.stats.n_already_judged == 1

    def test_a_pair_carries_no_previous_grade(self) -> None:
        # Structural guarantee, not a UI promise: there is no field on the pair
        # through which the earlier grade could be displayed.
        pool = build_retest_pool({"q001": {"a": 2}}, {}, size=1, seed=42)
        pair = pool.pairs[0]
        assert not hasattr(pair, "relevance")
        assert not hasattr(pair, "grade")

    def test_nothing_labeled_means_nothing_to_retest(self) -> None:
        pool = build_retest_pool({}, {}, size=30, seed=42)
        assert pool.pairs == ()
        assert pool.stats.n_queries == 0


class TestRetestPairCap:
    """``--retest-size`` counts queries; the *work* is pairs, and it is capped.

    Without the cap, a retest over 30 queries re-served every pair ever judged
    for each of them — a second labeling job behind a flag documented as a quick
    second pass. These tests pin the two properties that make the cap honest:
    the served count obeys it, and the cap is spread rather than truncated.
    """

    def qrels(self, n_queries: int, per_query: int) -> dict[str, dict[str, int]]:
        return {
            f"q{i:03d}": {f"b{i:03d}-{j}": 1 for j in range(per_query)} for i in range(n_queries)
        }

    def test_the_cap_limits_the_pairs_served(self) -> None:
        pool = build_retest_pool(self.qrels(10, 20), {}, size=10, seed=42, max_pairs=25)
        assert len(pool.pairs) == 25
        assert pool.stats.n_new == 25
        # The eligible pairs that did not fit are counted, so the report can say
        # why n_new is smaller than what was pooled.
        assert pool.stats.n_over_cap == 10 * 20 - 25

    def test_the_cap_is_spread_across_queries_not_truncated(self) -> None:
        pool = build_retest_pool(self.qrels(10, 20), {}, size=10, seed=42, max_pairs=25)
        per_query = pool.stats.new_per_query
        assert len(per_query) == 10, "every chosen query must contribute"
        assert max(per_query.values()) - min(per_query.values()) <= 1

    def test_pairs_stay_grouped_by_query(self) -> None:
        # Interleaving queries would make the labeler rebuild the request in
        # their head on every judgement; the cap must not cost that.
        pool = build_retest_pool(self.qrels(5, 10), {}, size=5, seed=42, max_pairs=15)
        order = [p.query_id for p in pool.pairs]
        assert order == sorted(order, key=order.index)
        assert len(dict.fromkeys(order)) == 5

    def test_no_cap_serves_everything(self) -> None:
        pool = build_retest_pool(self.qrels(4, 7), {}, size=4, seed=42)
        assert len(pool.pairs) == 28
        assert pool.stats.n_over_cap == 0

    def test_a_cap_above_what_exists_is_harmless(self) -> None:
        pool = build_retest_pool(self.qrels(3, 2), {}, size=3, seed=42, max_pairs=1000)
        assert len(pool.pairs) == 6
        assert pool.stats.n_over_cap == 0

    def test_resuming_counts_against_the_cap_fresh(self) -> None:
        # Already-retested pairs are excluded first, so a resumed session serves
        # up to the cap of what is *left* rather than re-counting graded pairs.
        qrels = self.qrels(2, 10)
        done = {qid: dict(list(pairs.items())[:8]) for qid, pairs in qrels.items()}
        pool = build_retest_pool(qrels, done, size=2, seed=42, max_pairs=10)
        assert len(pool.pairs) == 4
        assert pool.stats.n_over_cap == 0


class TestSpreadCap:
    def test_round_robin_fills_evenly(self) -> None:
        assert pooling.spread_cap([5, 5, 5], 7) == (3, 2, 2)

    def test_a_small_bucket_does_not_hold_back_the_cap(self) -> None:
        assert pooling.spread_cap([1, 10], 6) == (1, 5)

    def test_no_cap_returns_what_is_available(self) -> None:
        assert pooling.spread_cap([2, 3], None) == (2, 3)

    def test_a_cap_larger_than_the_supply_stops_rather_than_spinning(self) -> None:
        assert pooling.spread_cap([2, 1], 99) == (2, 1)

    def test_empty(self) -> None:
        assert pooling.spread_cap([], 10) == ()
        assert pooling.spread_cap([3, 3], 0) == (0, 0)

    def test_a_negative_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pooling.spread_cap([1], -1)


class TestFrozenLabelSubset:
    """The label subset is pinned on first use and only ever added to.

    ``select_label_subset`` is a rank cut over a hash ordering, so adding gold
    queries displaces queries at the boundary — graded pools leave the subset
    while fresh ones arrive. eval/README.md tells the owner to add a later
    batch, so that is an expected path, not a corner case.
    """

    def test_the_draw_can_move_without_the_manifest(self) -> None:
        # The defect being fixed, pinned so nobody "simplifies" the manifest
        # away: this is the behaviour the freeze exists to stop.
        base = gold(130)
        extra = [GoldQuery(f"g{i:03d}", "later batch", QueryType.VIBE) for i in range(20)]
        before = {q.query_id for q in select_label_subset(base, 50, seed=42)}
        after = {q.query_id for q in select_label_subset([*base, *extra], 50, seed=42)}
        assert before != after
        assert before - after, "adding queries evicts some already-chosen ones"

    def test_the_first_call_freezes_the_pure_draw(self) -> None:
        queries = gold(130)
        resolved = pooling.resolve_label_subset(queries, 50, 42, None)
        assert [q.query_id for q in resolved.queries] == [
            q.query_id for q in select_label_subset(queries, 50, seed=42)
        ]
        assert resolved.changed is True
        assert resolved.manifest.n_gold_at_draw == 130
        assert resolved.manifest.size == 50

    def test_adding_gold_queries_does_not_move_the_frozen_subset(self) -> None:
        base = gold(130)
        first = pooling.resolve_label_subset(base, 50, 42, None)
        extra = [GoldQuery(f"g{i:03d}", "later batch", QueryType.VIBE) for i in range(20)]
        second = pooling.resolve_label_subset([*base, *extra], 50, 42, first.manifest)
        assert {q.query_id for q in second.queries} == {q.query_id for q in first.queries}
        assert second.changed is False
        # And it says the population the draw describes is no longer the file's.
        assert any("gold file has changed" in note for note in second.notes)

    def test_raising_the_size_adds_without_removing(self) -> None:
        queries = gold(130)
        first = pooling.resolve_label_subset(queries, 50, 42, None)
        second = pooling.resolve_label_subset(queries, 80, 42, first.manifest)
        assert {q.query_id for q in first.queries} <= {q.query_id for q in second.queries}
        assert len(second.queries) == 80
        assert second.changed is True
        assert any("added 30" in note for note in second.notes)

    def test_lowering_the_size_removes_nothing(self) -> None:
        queries = gold(130)
        first = pooling.resolve_label_subset(queries, 50, 42, None)
        second = pooling.resolve_label_subset(queries, 20, 42, first.manifest)
        assert len(second.queries) == 50
        assert any("already in the frozen subset" in note for note in second.notes)

    def test_a_deleted_gold_query_is_not_replaced(self) -> None:
        # Substituting another query would change the population silently; the
        # honest answer is a shorter run and a note saying so.
        queries = gold(60)
        first = pooling.resolve_label_subset(queries, 20, 42, None)
        dropped = first.manifest.query_ids[0]
        remaining = [q for q in queries if q.query_id != dropped]
        second = pooling.resolve_label_subset(remaining, 20, 42, first.manifest)
        assert len(second.queries) == 19
        assert dropped in second.manifest.query_ids
        assert any("no longer in the gold file" in note for note in second.notes)

    def test_serving_order_is_the_hash_order(self) -> None:
        queries = gold(60)
        first = pooling.resolve_label_subset(queries, 20, 42, None)
        again = pooling.resolve_label_subset(queries, 20, 42, first.manifest)
        assert [q.query_id for q in again.queries] == [q.query_id for q in first.queries]

    def test_a_changed_seed_is_refused(self) -> None:
        queries = gold(60)
        first = pooling.resolve_label_subset(queries, 20, 42, None)
        with pytest.raises(ValueError, match="seed"):
            pooling.resolve_label_subset(queries, 20, 7, first.manifest)

    def test_duplicate_ids_are_rejected(self) -> None:
        duplicated = [*gold(3), GoldQuery("q001", "same id again", QueryType.VIBE)]
        with pytest.raises(ValueError, match="duplicate query_id"):
            pooling.resolve_label_subset(duplicated, 2, 1, None)

    def test_round_trips_through_the_manifest_file(self, tmp_path: Path) -> None:
        path = tmp_path / "artifacts" / pooling.LABEL_SUBSET_FILENAME
        assert pooling.read_label_subset_manifest(path) is None
        resolved = pooling.resolve_label_subset(gold(40), 10, 42, None)
        pooling.write_label_subset_manifest(path, resolved.manifest)
        assert pooling.read_label_subset_manifest(path) == resolved.manifest

    @pytest.mark.parametrize("body", ["not json", "[]", '{"seed": 1}'])
    def test_a_damaged_manifest_raises_rather_than_re_drawing(
        self, tmp_path: Path, body: str
    ) -> None:
        # The opposite trade from the snippet cache: re-deriving this file
        # changes what a reported number describes, so it must not be guessed.
        path = tmp_path / pooling.LABEL_SUBSET_FILENAME
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            pooling.read_label_subset_manifest(path)


class TestAgreementPairs:
    def test_aligns_on_the_intersection(self) -> None:
        first = [judgement("q001", "a", 2), judgement("q001", "b", 0), judgement("q002", "c", 1)]
        second = [judgement("q001", "a", 1), judgement("q001", "b", 0)]
        a, b, keys = agreement_pairs(first, second)
        assert keys == (("q001", "a"), ("q001", "b"))
        assert a == [2, 0]
        assert b == [1, 0]

    def test_a_pair_only_one_side_graded_is_dropped(self) -> None:
        # Filling the gap with a 0 would manufacture agreement from absence.
        a, b, keys = agreement_pairs([judgement("q001", "a", 2)], [judgement("q001", "z", 2)])
        assert (a, b, keys) == ([], [], ())

    def test_last_grade_wins_on_each_side(self) -> None:
        first = [judgement("q001", "a", 0), judgement("q001", "a", 2)]
        second = [judgement("q001", "a", 1)]
        a, b, _ = agreement_pairs(first, second)
        assert (a, b) == ([2], [1])


# ─────────────────────────────────────────────────────────────────────────────
# The CLI's pure logic
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli():
    """The labeling CLI module.

    ``importorskip`` because this module imports typer and rich: if the venv is
    still being built, the pooling tests above should still run and report
    rather than erroring at collection.
    """
    return pytest.importorskip("eval.label_cli")


@pytest.fixture
def businesses(cli, raw_businesses):
    return {b["business_id"]: cli.business_from_raw(b) for b in raw_businesses}


class TestBusinessDisplay:
    def test_parses_the_datasets_awkward_attribute_encodings(self, businesses) -> None:
        # b001 ships "u'free'" and a stringified nested dict. The labeling screen
        # has to show what the search system sees, so it uses the same parsers.
        lantern = businesses["b001"]
        assert lantern.name == "Lantern Coffee"
        assert "Coffee & Tea" in lantern.categories
        assert lantern.attributes["WiFi"] == "free"
        assert lantern.attributes["NoiseLevel"] == "quiet"

    def test_a_business_with_no_data_is_empty_not_invented(self, businesses) -> None:
        bistro = businesses["b005"]
        assert bistro.attributes == {}
        assert bistro.hours == {}

    def test_missing_coordinates_are_none_not_zero(self, cli) -> None:
        business = cli.business_from_raw(
            {"business_id": "bX", "name": "N", "latitude": None, "longitude": None}
        )
        assert business.latitude is None and business.longitude is None

    def test_attributes_render_as_readable_phrases(self, cli, businesses) -> None:
        panel = cli.candidate_panel(businesses["b001"], ["a snippet"])
        assert panel is not None  # rendering must not raise on the real quirks


class TestHoursDisplay:
    def test_overnight_hours_say_next_day(self, cli, businesses) -> None:
        # "17:0-2:0". A bare "17:00-02:00" invites the reader to see a gap in the
        # afternoon, and "open late" queries are a large share of the gold set.
        rows = dict(cli.format_hours(businesses["b002"]))
        assert rows["Mon"] == "17:00-02:00 (next day)"

    def test_an_interval_ending_at_midnight_is_not_overnight(self, cli, businesses) -> None:
        rows = dict(cli.format_hours(businesses["b002"]))
        assert rows["Sun"] == "17:00-midnight"

    def test_open_all_day(self, cli, businesses) -> None:
        rows = dict(cli.format_hours(businesses["b007"]))
        assert rows["Mon"] == "open 24 hours"

    def test_a_day_with_no_entry_reads_as_closed(self, cli, businesses) -> None:
        rows = dict(cli.format_hours(businesses["b008"]))
        assert rows["Mon"] == "closed"

    def test_no_hours_data_is_distinguishable_from_closed(self, cli, businesses) -> None:
        # The labeler must be able to tell "shut on Monday" from "the dataset
        # says nothing", because grading the second as a failure bakes a wrong
        # penalty into the ground truth.
        rows = cli.format_hours(businesses["b005"])
        assert rows == [("—", "no hours data in the dataset")]


class TestSnippets:
    def test_truncates_at_a_word_boundary(self, cli) -> None:
        text = "word " * 100
        out = cli.truncate_snippet(text, limit=20)
        assert len(out) <= 21
        assert out.endswith("…")
        assert "wor…" not in out

    def test_short_text_is_untouched(self, cli) -> None:
        assert cli.truncate_snippet("  short   review ") == "short review"

    def test_picks_the_most_useful_reviews(self, cli, fixture_dir) -> None:
        snippets = cli.load_snippets(fixture_dir / "reviews.json", {"b001"}, per_business=1)
        # r001 has 12 useful votes, r002 has 2.
        assert "Fast wifi" in snippets["b001"][0]


class TestSnippetCache:
    """The review scan is paid once per pool, not once per sitting.

    ``load_snippets`` cannot stop early (it ranks by ``useful``, so the best
    review may be the last record), and eval/README.md encourages short
    sittings. Without a cache the same full scan ran at the start of every one.
    """

    def test_the_second_call_does_not_read_the_file(
        self, cli, fixture_dir, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "snippets.json"
        path = fixture_dir / "reviews.json"
        first, note = cli.load_snippets_cached(path, {"b001"}, cache_path=cache)
        assert "Fast wifi" in first["b001"][0]
        assert "scanned" in note

        def fail(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the cache should have answered without a scan")

        monkeypatch.setattr(cli, "load_snippets", fail)
        second, note = cli.load_snippets_cached(path, {"b001"}, cache_path=cache)
        assert second == first
        assert "already in" in note

    def test_only_the_new_ids_are_scanned_when_the_pool_grows(
        self, cli, fixture_dir, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "snippets.json"
        path = fixture_dir / "reviews.json"
        cli.load_snippets_cached(path, {"b001"}, cache_path=cache)
        asked: list[set[str]] = []
        real = cli.load_snippets

        def spy(p, keep, **kwargs):
            asked.append(set(keep))
            return real(p, keep, **kwargs)

        monkeypatch.setattr(cli, "load_snippets", spy)
        cli.load_snippets_cached(path, {"b001", "b002"}, cache_path=cache)
        assert asked == [{"b002"}]

    def test_a_business_with_no_reviews_is_not_looked_for_twice(
        self, cli, fixture_dir, tmp_path: Path, monkeypatch
    ) -> None:
        # The case that would otherwise defeat the cache entirely: an id that is
        # absent from review.json is never written, so every session re-scans
        # 5 GB to not find it again.
        cache = tmp_path / "snippets.json"
        path = fixture_dir / "reviews.json"
        found, _ = cli.load_snippets_cached(path, {"b-nonexistent"}, cache_path=cache)
        assert found == {}

        def fail(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("an absent id must be remembered as absent")

        monkeypatch.setattr(cli, "load_snippets", fail)
        found, note = cli.load_snippets_cached(path, {"b-nonexistent"}, cache_path=cache)
        assert found == {}
        assert "already in" in note

    def test_a_changed_source_file_invalidates_the_cache(
        self, cli, fixture_dir, tmp_path: Path
    ) -> None:
        source = tmp_path / "reviews.json"
        source.write_text(
            json.dumps({"business_id": "b001", "useful": 1, "text": "first text"}) + "\n",
            encoding="utf-8",
        )
        cache = tmp_path / "snippets.json"
        first, _ = cli.load_snippets_cached(source, {"b001"}, cache_path=cache)
        assert first["b001"] == ["first text"]
        with source.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"business_id": "b001", "useful": 99, "text": "much better text"}) + "\n"
            )
        second, note = cli.load_snippets_cached(source, {"b001"}, cache_path=cache)
        assert second["b001"][0] == "much better text"
        assert "has changed" in note

    def test_a_damaged_cache_is_rebuilt_rather_than_raised_on(
        self, cli, fixture_dir, tmp_path: Path
    ) -> None:
        # Derived data: nothing in it cannot be recomputed, so a labeling
        # session must not die over it.
        cache = tmp_path / "snippets.json"
        cache.write_text("{not json", encoding="utf-8")
        found, _ = cli.load_snippets_cached(
            fixture_dir / "reviews.json", {"b001"}, cache_path=cache
        )
        assert "Fast wifi" in found["b001"][0]

    def test_the_cache_is_named_after_its_source(self, cli, tmp_path: Path) -> None:
        # review.json and tip.json are ranked by different keys, so one cache
        # for both would serve tip snippets as reviews.
        reviews = cli.snippet_cache_path(tmp_path, Path("/data/review.json"))
        tips = cli.snippet_cache_path(tmp_path, Path("/data/tip.json"))
        assert reviews != tips
        assert reviews.parent == tmp_path


class TestRunsFile:
    def test_reads_a_valid_file(self, cli, tmp_path: Path) -> None:
        path = tmp_path / "runs.json"
        path.write_text(json.dumps({"bm25": {"q001": ["a", "b"]}}), encoding="utf-8")
        assert cli.load_runs(path) == {"bm25": {"q001": ["a", "b"]}}

    @pytest.mark.parametrize(
        "payload",
        [{}, [], {"bm25": ["a"]}, {"bm25": {"q001": "a"}}, {"bm25": {"q001": [1, 2]}}],
    )
    def test_rejects_a_malformed_file(self, cli, tmp_path: Path, payload) -> None:
        # Produced by another part of the pipeline; a silent misread here would
        # pool the wrong thing and only surface as a strange metric weeks later.
        path = tmp_path / "runs.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            cli.load_runs(path)


class TestLabelSession:
    def session(self, cli, tmp_path: Path, n: int = 3):
        pool = build_pool({"bm25": {"q001": [f"b{i}" for i in range(n)]}}, depth=n)
        writer = QrelsWriter(tmp_path / "qrels.csv", "owner").open()
        return cli.LabelSession(pairs=pool.pairs, writer=writer), writer

    def test_grading_advances_and_writes(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path)
        session.grade(2)
        assert session.index == 1
        assert session.n_graded == 1
        assert len(read_judgements(tmp_path / "qrels.csv")) == 1
        writer.close()

    def test_skipping_writes_nothing_so_the_pair_returns(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path)
        session.skip()
        assert session.n_skipped == 1
        assert read_judgements(tmp_path / "qrels.csv") == []
        writer.close()

    def test_undo_reverses_a_grade_in_the_file_too(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path)
        first = session.current()
        session.grade(0)
        assert session.undo() is True
        assert session.current() == first
        assert session.n_graded == 0
        assert read_judgements(tmp_path / "qrels.csv") == []
        writer.close()

    def test_undo_reverses_a_skip(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path)
        session.skip()
        assert session.undo() is True
        assert session.n_skipped == 0
        assert session.index == 0
        writer.close()

    def test_undo_with_no_history_is_a_no_op(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path)
        assert session.undo() is False
        assert session.index == 0
        writer.close()

    def test_position_within_the_query(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path, n=3)
        assert session.position_in_query() == (1, 3)
        session.grade(1)
        assert session.position_in_query() == (2, 3)
        writer.close()

    def test_eta_is_unmeasured_until_there_is_a_pace(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path)
        assert session.pace_s is None
        assert session.eta_s is None
        assert "ETA —" in session.progress_line().plain
        writer.close()

    def test_a_historical_pace_is_used_before_this_session_has_one(
        self, cli, tmp_path: Path
    ) -> None:
        pool = build_pool({"bm25": {"q001": ["a", "b", "c", "d"]}}, depth=4)
        writer = QrelsWriter(tmp_path / "qrels.csv", "owner").open()
        session = cli.LabelSession(pairs=pool.pairs, writer=writer, historical_pace_s=20.0)
        assert session.eta_s == pytest.approx(80.0)
        writer.close()

    def test_runs_to_completion(self, cli, tmp_path: Path) -> None:
        session, writer = self.session(cli, tmp_path, n=2)
        session.grade(2)
        session.grade(0)
        assert session.done
        assert session.current() is None
        assert session.remaining == 0
        writer.close()


class TestRunSession:
    """The keyboard loop, driven by a scripted key reader."""

    def drive(self, cli, tmp_path: Path, keys: list[str], n: int = 3):
        """Run the loop over `n` fixture businesses with a scripted keyboard.

        Returns the served order too: `build_pool` shuffles within a query on
        purpose, so a test that assumed "b001 comes first" would be asserting
        the very anchoring behaviour we removed.
        """
        pool = build_pool({"bm25": {"q001": [f"b00{i}" for i in range(1, n + 1)]}}, depth=n)
        served = [p.business_id for p in pool.pairs]
        queries = {"q001": GoldQuery("q001", "quiet cafe with wifi", QueryType.CONSTRAINT)}
        path = tmp_path / "qrels.csv"
        pressed = iter(keys)
        console = cli.Console(file=io.StringIO(), width=100)
        with QrelsWriter(path, "owner") as writer:
            session = cli.LabelSession(pairs=pool.pairs, writer=writer)
            cli.run_session(
                session,
                queries,
                self.records(cli),
                {},
                key_reader=lambda: next(pressed, "q"),
                out=console,
            )
        return session, path, console.file.getvalue(), served

    def records(self, cli):
        import json as _json

        path = REPO_ROOT / "tests" / "fixtures" / "businesses.json"
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = _json.loads(line)
                out[record["business_id"]] = cli.business_from_raw(record)
        # The pool in `drive` uses b001..b003, which exist in the fixtures.
        return out

    def test_each_keystroke_grades_the_candidate_on_screen(self, cli, tmp_path: Path) -> None:
        session, path, _, served = self.drive(cli, tmp_path, ["2", "1", "0"])
        assert session.n_graded == 3
        assert load_qrels(path) == {
            "q001": dict(zip(served, [2, 1, 0], strict=True)),
        }

    def test_quitting_keeps_everything_graded_so_far(self, cli, tmp_path: Path) -> None:
        session, path, _, served = self.drive(cli, tmp_path, ["2", "q", "1"])
        assert session.n_graded == 1
        assert load_qrels(path) == {"q001": {served[0]: 2}}

    def test_undo_from_the_keyboard(self, cli, tmp_path: Path) -> None:
        session, path, _, served = self.drive(cli, tmp_path, ["2", "u", "0", "q"])
        assert load_qrels(path) == {"q001": {served[0]: 0}}
        assert session.n_graded == 1

    def test_an_unknown_key_does_nothing_but_say_so(self, cli, tmp_path: Path) -> None:
        session, _, output, _ = self.drive(cli, tmp_path, ["x", "2", "q"])
        assert session.n_graded == 1
        assert "unknown key" in output

    def test_the_grade_scale_is_always_on_screen(self, cli, tmp_path: Path) -> None:
        # Drift in what "1" means between sessions is invisible in the data and
        # fatal to it, so re-reading the definition must cost nothing.
        _, _, output, _ = self.drive(cli, tmp_path, ["2", "q"])
        assert "highly relevant" in output
        assert "partially relevant" in output
        assert "irrelevant" in output

    def test_the_query_and_the_evidence_are_shown(self, cli, tmp_path: Path) -> None:
        _, _, output, served = self.drive(cli, tmp_path, ["q"])
        names = {"b001": "Lantern Coffee", "b002": "Midnight Ramen", "b003": "The Quiet Shelf"}
        assert "quiet cafe with wifi" in output
        assert names[served[0]] in output
        assert "q001" in output and "constraint" in output

    def test_attributes_and_hours_are_on_the_first_screen(self, cli, tmp_path: Path) -> None:
        # Judging "open past 10pm with wifi" is impossible without them, so
        # their presence is a correctness property of this tool, not polish.
        pool = build_pool({"bm25": {"q001": ["b001"]}}, depth=1)
        queries = {"q001": GoldQuery("q001", "quiet cafe with wifi", QueryType.CONSTRAINT)}
        console = cli.Console(file=io.StringIO(), width=120)
        with QrelsWriter(tmp_path / "qrels.csv", "owner") as writer:
            session = cli.LabelSession(pairs=pool.pairs, writer=writer)
            cli.run_session(
                session,
                queries,
                self.records(cli),
                {"b001": ["Fast wifi, soft music."]},
                key_reader=lambda: "q",
                out=console,
            )
        output = console.file.getvalue()
        assert "Lantern Coffee" in output
        assert "wifi" in output.lower()
        assert "quiet" in output.lower()
        assert "07:00" in output  # Monday hours, from the weekly table
        assert "4.5 stars" in output and "312 reviews" in output
        assert "Fast wifi" in output

    def test_a_pooled_business_with_no_record_is_skipped_and_reported(
        self, cli, tmp_path: Path
    ) -> None:
        # Runs file and business file built from different data: a real bug that
        # must be visible, not silently swallowed.
        pool = build_pool({"bm25": {"q001": ["ghost"]}}, depth=1)
        queries = {"q001": GoldQuery("q001", "anything", QueryType.VIBE)}
        console = cli.Console(file=io.StringIO(), width=100)
        with QrelsWriter(tmp_path / "qrels.csv", "owner") as writer:
            session = cli.LabelSession(pairs=pool.pairs, writer=writer)
            cli.run_session(session, queries, {}, {}, key_reader=lambda: "q", out=console)
        assert session.n_skipped == 1
        assert session.done
        assert load_qrels(tmp_path / "qrels.csv") == {}


class TestFrozenSubsetInTheCLI:
    """The interface half of the freeze: where the manifest lives, when it is written."""

    def config(self, cli, tmp_path: Path, size: int = 10):
        from concierge.config import load_config

        cfg = load_config(Path("configs/base.yaml"))
        return cfg.model_copy(
            update={
                "evaluation": cfg.evaluation.model_copy(
                    update={"out_dir": tmp_path, "label_subset_size": size}
                )
            }
        )

    def run(self, cli, cfg, queries, *, write: bool, monkeypatch):
        monkeypatch.setattr(cli, "console", cli.Console(file=io.StringIO(), width=120))
        chosen, scope = cli._frozen_subset(cfg, queries, cfg.seed, write=write)
        return chosen, scope, cli.console.file.getvalue()

    def test_the_manifest_is_written_and_reused(self, cli, tmp_path: Path, monkeypatch) -> None:
        cfg = self.config(cli, tmp_path)
        queries = gold(60)
        first, scope, _ = self.run(cli, cfg, queries, write=True, monkeypatch=monkeypatch)
        manifest = tmp_path / pooling.LABEL_SUBSET_FILENAME
        assert manifest.exists()
        assert "frozen random subset of 10" in scope
        # A later run with 20 more queries serves the same ten.
        extra = [GoldQuery(f"g{i:03d}", "later batch", QueryType.MESSY) for i in range(20)]
        again, _, output = self.run(
            cli, cfg, [*queries, *extra], write=True, monkeypatch=monkeypatch
        )
        assert [q.query_id for q in again] == [q.query_id for q in first]
        assert "gold file has changed" in output

    def test_stats_does_not_freeze_the_subset(self, cli, tmp_path: Path, monkeypatch) -> None:
        # Looking at the pool is not starting to label it, and freezing on a
        # --stats run would pin the subset before the gold set is finished.
        cfg = self.config(cli, tmp_path)
        _, _, output = self.run(cli, cfg, gold(60), write=False, monkeypatch=monkeypatch)
        assert not (tmp_path / pooling.LABEL_SUBSET_FILENAME).exists()
        assert "not written" in output

    def test_a_changed_seed_exits_rather_than_re_drawing(
        self, cli, tmp_path: Path, monkeypatch
    ) -> None:
        cfg = self.config(cli, tmp_path)
        queries = gold(60)
        self.run(cli, cfg, queries, write=True, monkeypatch=monkeypatch)
        monkeypatch.setattr(cli, "console", cli.Console(file=io.StringIO(), width=120))
        with pytest.raises(cli.typer.Exit):
            cli._frozen_subset(cfg, queries, cfg.seed + 1, write=True)
        assert "seed" in cli.console.file.getvalue()


class TestAgreementReport:
    def test_reports_nothing_to_compare(self, cli, tmp_path: Path) -> None:
        original = write_qrels(
            tmp_path / "qrels.csv", [["q001", "a", 2, "2026-01-01T12:00:00+00:00", "owner"]]
        )
        retest = tmp_path / "qrels_retest.csv"
        console = cli.Console(file=io.StringIO(), width=100)
        assert cli.report_agreement(original, retest, out=console) == 0
        assert "No pairs graded in both" in console.file.getvalue()

    def test_reports_kappa_both_weighted_and_unweighted(self, cli, tmp_path: Path) -> None:
        # Quadratic weighting is the more flattering number on a 3-point scale,
        # so the report must never show it alone.
        rows_first = [
            ["q001", f"b{i}", g, f"2026-01-0{i + 1}T12:00:00+00:00", "owner"]
            for i, g in enumerate([2, 2, 1, 0, 0])
        ]
        rows_second = [
            ["q001", f"b{i}", g, f"2026-02-0{i + 1}T12:00:00+00:00", "owner"]
            for i, g in enumerate([2, 1, 1, 0, 1])
        ]
        original = write_qrels(tmp_path / "qrels.csv", rows_first)
        retest = write_qrels(tmp_path / "qrels_retest.csv", rows_second)
        console = cli.Console(file=io.StringIO(), width=120)
        n = cli.report_agreement(original, retest, out=console)
        output = console.file.getvalue()
        assert n == 5
        assert "quadratic" in output
        assert "unweighted" in output
        assert "Exact agreement" in output
        assert "Days between first and second label" in output
