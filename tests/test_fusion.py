"""Unit tests for Reciprocal Rank Fusion.

Every expected number here is written out as the arithmetic that produces it
(``1/61 + 1/62``, not ``0.0325``) and asserted exactly where exact assertion is
sound. The point is that the test encodes the *formula*, so a change in
behaviour has to argue with the definition rather than with a magic constant
somebody once copied out of a passing run.
"""

from __future__ import annotations

import pytest

from concierge.retrieval.fusion import (
    fuse,
    rank_map,
    rrf_scores,
    weighted_sum_scores,
)
from concierge.types import RANK_RRF, SCORE_BM25, SCORE_DENSE, SCORE_RRF

K = 60  # cfg.retrieval.fusion.k default; spelled out so the sums below read


def _with_fusion(config, **updates):
    """Return a copy of the config with `retrieval.fusion` fields overridden."""
    fusion = config.retrieval.fusion.model_copy(update=updates)
    return config.model_copy(
        update={"retrieval": config.retrieval.model_copy(update={"fusion": fusion})}
    )


# ── rank_map ─────────────────────────────────────────────────────────────────


def test_ranks_are_one_based() -> None:
    """RRF's 1/(k + rank) assumes 1-based ranks. A 0-based rank would hand the
    top document 1/k, shifting every weight by one position and quietly
    disagreeing with the published formula."""
    assert rank_map([("a", 9.0), ("b", 4.0), ("c", 1.0)]) == {"a": 1, "b": 2, "c": 3}


def test_duplicate_ids_keep_their_best_rank() -> None:
    """Should not happen with a correct retriever; if it does, the alternative
    (last occurrence wins) would silently demote a top hit."""
    assert rank_map([("a", 9.0), ("b", 4.0), ("a", 0.5)]) == {"a": 1, "b": 2}


def test_rank_map_ignores_scores() -> None:
    """Scale-invariance is the reason RRF is the default: the same ordering
    must produce the same ranks whatever the score magnitudes are."""
    assert rank_map([("a", 1e9), ("b", 0.001)]) == rank_map([("a", 0.2), ("b", 0.1)])


# ── rrf_scores: the arithmetic, by hand ──────────────────────────────────────


def test_rrf_sum_over_two_systems() -> None:
    """A document at rank 1 in both lists scores 1/(60+1) + 1/(60+1) = 2/61."""
    rankings = {"bm25": {"a": 1}, "dense": {"a": 1}}
    assert rrf_scores(rankings, K)["a"] == pytest.approx(1 / 61 + 1 / 61)


def test_rrf_worked_example() -> None:
    """bm25 = [a, b, c], dense = [a, d]. By hand, with k = 60:

        a: 1/(60+1) + 1/(60+1) = 1/61 + 1/61 = 2/61
        b: 1/(60+2)                          = 1/62
        c: 1/(60+3)                          = 1/63
        d:             1/(60+2)              = 1/62
    """
    rankings = {
        "bm25": {"a": 1, "b": 2, "c": 3},
        "dense": {"a": 1, "d": 2},
    }
    scores = rrf_scores(rankings, K)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)
    # b and d are found at the same rank by one system each, so RRF cannot tell
    # them apart — a real and intended property, not a bug.
    assert scores["b"] == pytest.approx(scores["d"])


def test_single_system_document_gets_no_phantom_bonus() -> None:
    """THE classic RRF implementation bug.

    "c" appears only in the bm25 list, at rank 3, so its score must be exactly
    1/63 — one term, from one system. The two tempting wrong answers:

      * treat the missing rank as 0  -> 1/63 + 1/60, the *largest possible*
        contribution, awarded for being absent;
      * treat it as "rank = len(list) + 1" -> 1/63 + 1/63, a smaller unearned
        bonus that also makes the score depend on how long each list happens
        to be.

    Either way, a document found by one system would beat a document found by
    one system plus a near-miss in the other, which inverts the whole point of
    fusing.
    """
    rankings = {"bm25": {"c": 3}, "dense": {"x": 1}}
    scores = rrf_scores(rankings, K)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["c"] != pytest.approx(1 / 63 + 1 / 60)  # missing-as-rank-0
    assert scores["c"] != pytest.approx(1 / 63 + 1 / 63)  # missing-as-last


def test_a_document_in_one_system_still_beats_a_lower_pair() -> None:
    """Being found once near the top can legitimately outrank being found twice
    further down; both must simply be *fusable*.

        single: rank 1 in bm25 only       -> 1/61            ≈ 0.016393
        pair:   rank 40 in both systems   -> 2 * 1/100 = 0.02
    """
    scores = rrf_scores({"bm25": {"single": 1, "pair": 40}, "dense": {"pair": 40}}, K)
    assert scores["single"] == pytest.approx(1 / 61)
    assert scores["pair"] == pytest.approx(2 / 100)
    assert scores["pair"] > scores["single"]


def test_k_damps_the_top_of_the_list() -> None:
    """Small k makes rank 1 nearly unbeatable; large k flattens towards
    "appeared in many lists". Asserted as the ratio between rank 1 and rank 2,
    which is the cleanest way to see the effect."""
    small = rrf_scores({"s": {"a": 1, "b": 2}}, 1)
    large = rrf_scores({"s": {"a": 1, "b": 2}}, 1000)
    assert small["a"] / small["b"] == pytest.approx((1 / 2) / (1 / 3))
    assert large["a"] / large["b"] == pytest.approx((1 / 1001) / (1 / 1002))
    assert small["a"] / small["b"] > large["a"] / large["b"]


def test_no_systems_yields_no_scores() -> None:
    assert rrf_scores({}, K) == {}


# ── fuse(): candidates, scores, ranks ────────────────────────────────────────


def test_fuse_writes_canonical_score_and_rank_keys(config) -> None:
    """Stage 4 reads these by name, so the key constants are part of the
    contract, not an implementation detail."""
    candidates = fuse(
        {SCORE_BM25: [("a", 8.2), ("b", 3.1)], SCORE_DENSE: [("a", 0.81), ("c", 0.42)]},
        config,
    )
    by_id = {c.business_id: c for c in candidates}

    assert by_id["a"].scores[SCORE_BM25] == pytest.approx(8.2)
    assert by_id["a"].scores[SCORE_DENSE] == pytest.approx(0.81)
    assert by_id["a"].scores[SCORE_RRF] == pytest.approx(1 / 61 + 1 / 61)
    assert by_id["a"].ranks[SCORE_BM25] == 1
    assert by_id["a"].ranks[SCORE_DENSE] == 1
    assert by_id["a"].ranks[RANK_RRF] == 1

    # "b" was never returned by the dense system: no dense score, no dense rank.
    # Writing a 0.0 there would be a lie that Stage 4 would learn from.
    assert SCORE_DENSE not in by_id["b"].scores
    assert SCORE_DENSE not in by_id["b"].ranks
    assert by_id["b"].scores[SCORE_RRF] == pytest.approx(1 / 62)


def test_fuse_orders_by_fused_score_with_deterministic_ties(config) -> None:
    """bm25 = [a, b, c], dense = [a, d] (the worked example above):

        a: 2/61 ≈ 0.032787
        b: 1/62 ≈ 0.016129   tie
        d: 1/62 ≈ 0.016129   tie, broken by business_id -> b before d
        c: 1/63 ≈ 0.015873
    """
    candidates = fuse(
        {
            SCORE_BM25: [("a", 3.0), ("b", 2.0), ("c", 1.0)],
            SCORE_DENSE: [("a", 0.9), ("d", 0.5)],
        },
        config,
    )
    assert [c.business_id for c in candidates] == ["a", "b", "d", "c"]
    assert [c.ranks[RANK_RRF] for c in candidates] == [1, 2, 3, 4]


def test_fuse_tie_break_does_not_depend_on_input_order(config) -> None:
    """Dict iteration order must not be able to change a result; otherwise an
    A/B difference cannot be attributed to the change under test."""
    forward = fuse({"s1": [("z", 1.0)], "s2": [("a", 1.0)]}, config)
    backward = fuse({"s2": [("a", 1.0)], "s1": [("z", 1.0)]}, config)
    assert [c.business_id for c in forward] == [c.business_id for c in backward] == ["a", "z"]


def test_fuse_respects_the_candidate_budget(config) -> None:
    systems = {SCORE_BM25: [(f"b{i:03d}", 10.0 - i) for i in range(25)]}
    assert len(fuse(systems, config, top_k=10)) == 10
    assert len(fuse(systems, config, top_k=None)) == 25  # under the 200 default


def test_fuse_defaults_top_k_to_the_configured_retrieve_budget(config) -> None:
    systems = {SCORE_BM25: [(f"b{i:04d}", float(1000 - i)) for i in range(300)]}
    assert len(fuse(systems, config)) == config.candidates.retrieve


def test_fuse_with_one_system_is_that_system(config) -> None:
    """The BM25-only ablation row goes through this same code path, so it must
    reduce cleanly to "just BM25"."""
    candidates = fuse({SCORE_BM25: [("a", 9.0), ("b", 1.0)]}, config)
    assert [c.business_id for c in candidates] == ["a", "b"]
    assert candidates[0].scores[SCORE_RRF] == pytest.approx(1 / 61)


def test_fuse_with_empty_systems_returns_nothing(config) -> None:
    assert fuse({SCORE_BM25: [], SCORE_DENSE: []}, config) == []


# ── weighted_sum, the ablation alternative ───────────────────────────────────


def test_weighted_sum_minmax_normalizes_per_system() -> None:
    """bm25 = [(a,10), (b,5), (c,0)] -> min 0, max 10 -> a=1.0, b=0.5, c=0.0
       dense = [(a,0.9), (d,0.1)]    -> min .1, max .9 -> a=1.0, d=0.0

       a = 1.0 + 1.0 = 2.0 ; b = 0.5 ; c = 0.0 ; d = 0.0
    """
    scores = weighted_sum_scores(
        {"bm25": [("a", 10.0), ("b", 5.0), ("c", 0.0)], "dense": [("a", 0.9), ("d", 0.1)]}
    )
    assert scores == {
        "a": pytest.approx(2.0),
        "b": pytest.approx(0.5),
        "c": pytest.approx(0.0),
        "d": pytest.approx(0.0),
    }


def test_weighted_sum_applies_weights() -> None:
    scores = weighted_sum_scores(
        {"bm25": [("a", 10.0), ("b", 0.0)], "dense": [("b", 1.0), ("a", 0.0)]},
        weights={"bm25": 3.0, "dense": 1.0},
    )
    assert scores["a"] == pytest.approx(3.0)  # 3*1.0 + 1*0.0
    assert scores["b"] == pytest.approx(1.0)  # 3*0.0 + 1*1.0


def test_weighted_sum_degenerate_range_maps_to_one() -> None:
    """A single result, or an all-equal list, has no range to normalize over.
    Mapping it to 0.0 instead would turn "the only result" into "worthless"."""
    assert weighted_sum_scores({"s": [("a", 7.0)]}) == {"a": pytest.approx(1.0)}
    assert weighted_sum_scores({"s": [("a", 2.0), ("b", 2.0)]}) == {
        "a": pytest.approx(1.0),
        "b": pytest.approx(1.0),
    }


def test_weighted_sum_is_selected_by_config(config) -> None:
    cfg = _with_fusion(config, method="weighted_sum")
    candidates = fuse(
        {"bm25": [("a", 10.0), ("b", 5.0), ("c", 0.0)], "dense": [("a", 0.9), ("d", 0.1)]},
        cfg,
    )
    assert [c.business_id for c in candidates] == ["a", "b", "c", "d"]
    assert candidates[0].scores[SCORE_RRF] == pytest.approx(2.0)


def test_weighted_sum_score_depends_on_unrelated_results(config) -> None:
    """The documented reason RRF is the default, asserted rather than claimed.

    Min-max normalization is computed over whatever population the system
    happened to return, so a document's fused score moves when a *different*
    document leaves the list — even though nothing about its own retrieval
    changed. RRF cannot do that, because it reads only this document's rank.

    Setup (bm25 identical in both cases; only the dense tail differs):

        dense = [b:0.9, a:0.5, c:0.1]  -> lo .1, hi .9 -> b=1.0, a=0.5, c=0.0
        dense = [b:0.9, a:0.5]         -> lo .5, hi .9 -> b=1.0, a=0.0

    "a"'s dense contribution halves to zero because "c" went away. Its dense
    *rank* is 2 in both, so RRF is unmoved.
    """
    bm25 = [("a", 10.0), ("b", 9.0)]
    wide = {"bm25": bm25, "dense": [("b", 0.9), ("a", 0.5), ("c", 0.1)]}
    narrow = {"bm25": bm25, "dense": [("b", 0.9), ("a", 0.5)]}

    ws_cfg = _with_fusion(config, method="weighted_sum")
    ws_wide = {c.business_id: c.scores[SCORE_RRF] for c in fuse(wide, ws_cfg)}
    ws_narrow = {c.business_id: c.scores[SCORE_RRF] for c in fuse(narrow, ws_cfg)}
    assert ws_wide["a"] == pytest.approx(1.0 + 0.5)
    assert ws_narrow["a"] == pytest.approx(1.0 + 0.0)

    rrf_cfg = _with_fusion(config, method="rrf")
    rrf_wide = {c.business_id: c.scores[SCORE_RRF] for c in fuse(wide, rrf_cfg)}
    rrf_narrow = {c.business_id: c.scores[SCORE_RRF] for c in fuse(narrow, rrf_cfg)}
    assert rrf_wide["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf_narrow["a"] == pytest.approx(rrf_wide["a"])
    assert rrf_narrow["b"] == pytest.approx(rrf_wide["b"])
