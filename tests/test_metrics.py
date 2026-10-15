"""Known-answer tests for every metric and statistic in ``eval/metrics.py``.

House rule for this file, and the reason it exists: **no expected value here was
obtained by running the code.** Every number is derived by hand in a comment
above its assertion, with the arithmetic written out, so the owner can
re-derive it on a whiteboard in an interview and so a refactor that changes a
formula fails loudly instead of quietly agreeing with itself.

Where a closed form is clearer than a decimal, the assertion is written against
the symbolic expression (e.g. ``3.5 / (3 + 1 / math.log2(3))``) *and* against
the decimal that long division gives, so neither a transcription slip nor a
formula change can pass unnoticed.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    Aggregate,
    aggregate,
    bootstrap_ci,
    bootstrap_ci_from_config,
    cohens_kappa,
    confusion_matrix,
    dcg_at_k,
    dedupe_preserving_order,
    discount,
    evaluate_run,
    gain,
    grades_for,
    ideal_dcg_at_k,
    ndcg_at_k,
    paired_bootstrap,
    paired_bootstrap_from_config,
    per_query,
    recall_at_k,
    reciprocal_rank,
    report_unjudged,
    subset,
)

# log2(3) appears in nearly every ideal-DCG computation below (the rank-2
# discount). Named once so the hand arithmetic in comments reads cleanly.
LOG2_3 = math.log2(3)  # 1.5849625007211562


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks: gain and discount
# ─────────────────────────────────────────────────────────────────────────────


def test_gain_is_exponential_not_linear():
    # gain(g) = 2**g - 1  ->  g=0: 1-1=0,  g=1: 2-1=1,  g=2: 4-1=3
    assert gain(0) == 0.0
    assert gain(1) == 1.0
    assert gain(2) == 3.0
    # The decision this encodes: a highly relevant result is worth THREE
    # partially relevant ones, not two. Under linear gain the ratio would be
    # 2/1 = 2.0. If this assertion ever reads 2.0, the gain function silently
    # became linear and every NDCG in the repo changed meaning.
    assert gain(2) / gain(1) == 3.0


def test_discount_is_log2_of_rank_plus_one():
    # rank 1 -> log2(2) = 1   (top result is undiscounted)
    # rank 3 -> log2(4) = 2   (counts half as much as rank 1)
    # rank 7 -> log2(8) = 3
    assert discount(1) == 1.0
    assert discount(3) == 2.0
    assert discount(7) == 3.0


def test_discount_rejects_zero_based_rank():
    # log2(0 + 1) = 0 would divide by zero; a 0-based rank is a caller bug.
    with pytest.raises(ValueError):
        discount(0)


# ─────────────────────────────────────────────────────────────────────────────
# DCG and NDCG
# ─────────────────────────────────────────────────────────────────────────────


def test_dcg_at_k_hand_computed():
    # grades [2, 0, 1] at k=3:
    #   rank 1: (2**2 - 1) / log2(2) = 3 / 1   = 3.0
    #   rank 2: (2**0 - 1) / log2(3) = 0 / 1.585 = 0.0
    #   rank 3: (2**1 - 1) / log2(4) = 1 / 2   = 0.5
    #   DCG@3 = 3.0 + 0.0 + 0.5 = 3.5
    assert dcg_at_k([2, 0, 1], k=3) == 3.5


def test_dcg_truncates_at_k():
    # Same list at k=2 drops the rank-3 contribution of 0.5: 3.0 + 0.0 = 3.0
    assert dcg_at_k([2, 0, 1], k=2) == 3.0


def test_dcg_with_fewer_results_than_k_does_not_pad():
    # A one-document run at k=10 is just its rank-1 term: 3 / 1 = 3.0
    assert dcg_at_k([2], k=10) == 3.0


def test_dcg_of_all_zero_grades_is_zero():
    # gain(0) = 0 at every rank, so the sum is 0 regardless of length.
    assert dcg_at_k([0, 0, 0], k=3) == 0.0


def test_ideal_dcg_uses_all_labels_sorted_desc():
    # qrels {a:2, b:0, c:1} sorted descending -> [2, 1, 0]
    #   3/log2(2) + 1/log2(3) + 0/log2(4) = 3 + 1/1.5849625 = 3.6309297535714574
    judged = {"a": 2, "b": 0, "c": 1}
    expected = 3 + 1 / LOG2_3
    assert ideal_dcg_at_k(judged, k=3) == pytest.approx(expected)
    assert ideal_dcg_at_k(judged, k=3) == pytest.approx(3.63092975, abs=1e-8)


def test_ideal_dcg_is_independent_of_what_the_system_retrieved():
    # The ceiling comes from the labels, never from the run. Two systems
    # scored against the same qrels must share one denominator, otherwise a
    # system that retrieved only its own best hit could score 1.0.
    judged = {"a": 2, "b": 1}
    assert ideal_dcg_at_k(judged, k=10) == pytest.approx(3 + 1 / LOG2_3)


def test_ndcg_at_3_hand_computed():
    # THE worked example. Run [a, b, c] with grades [2, 0, 1]:
    #   DCG@3   = 3/log2(2) + 0/log2(3) + 1/log2(4) = 3 + 0 + 0.5 = 3.5
    #   ideal   = labels sorted [2, 1, 0] = 3 + 1/log2(3) + 0 = 3.6309297535714574
    #   NDCG@3  = 3.5 / 3.6309297535714574 = 0.9639397...
    judged = {"a": 2, "b": 0, "c": 1}
    got = ndcg_at_k(["a", "b", "c"], judged, k=3)
    assert got == pytest.approx(3.5 / (3 + 1 / LOG2_3))
    assert got == pytest.approx(0.96394, abs=1e-5)


def test_ndcg_of_the_perfect_ranking_is_exactly_one():
    # Run [a, c, b] puts grades in order [2, 1, 0], which IS the ideal list,
    # so numerator and denominator are the same number: 3 + 1/log2(3).
    judged = {"a": 2, "b": 0, "c": 1}
    assert ndcg_at_k(["a", "c", "b"], judged, k=3) == pytest.approx(1.0)


def test_ndcg_of_the_reversed_ranking_hand_computed():
    # Run [b, c, a] -> grades [0, 1, 2]:
    #   DCG = 0/1 + 1/log2(3) + 3/log2(4) = 0.6309297535714574 + 1.5
    #       = 2.1309297535714574
    #   NDCG = 2.1309297535714574 / 3.6309297535714574 = 0.5868825...
    judged = {"a": 2, "b": 0, "c": 1}
    got = ndcg_at_k(["b", "c", "a"], judged, k=3)
    assert got == pytest.approx((1 / LOG2_3 + 1.5) / (3 + 1 / LOG2_3))
    assert got == pytest.approx(0.58688, abs=1e-5)


def test_ndcg_penalizes_a_short_run_that_misses_a_relevant_doc():
    # Run returns only [a] (grade 2); qrels also hold b at grade 1.
    #   DCG@10   = 3 / log2(2) = 3.0
    #   ideal@10 = 3 + 1/log2(3) = 3.6309297535714574
    #   NDCG@10  = 3 / 3.6309297535714574 = 0.8262363...
    # This is the point of a label-derived ceiling: a run can be perfect in
    # what it ordered and still score below 1 for what it never returned.
    judged = {"a": 2, "b": 1}
    got = ndcg_at_k(["a"], judged, k=10)
    assert got == pytest.approx(3 / (3 + 1 / LOG2_3))
    assert got == pytest.approx(0.82624, abs=1e-5)


def test_ndcg_truncation_can_cut_the_only_relevant_result():
    # Run [b, c, a] with qrels {a:2}; at k=2 the only relevant doc (a, rank 3)
    # is below the cutoff.
    #   DCG@2   = 0 + 0 = 0
    #   ideal@2 = [2] -> 3/log2(2) = 3
    #   NDCG@2  = 0 / 3 = 0.0
    assert ndcg_at_k(["b", "c", "a"], {"a": 2}, k=2) == 0.0


def test_ndcg_is_zero_not_none_when_the_run_finds_nothing_relevant():
    # Relevant documents exist (so NDCG is well defined) but none was
    # retrieved: a real, measured 0.0. Distinguishing this from the undefined
    # case below is the whole reason ndcg returns an Optional.
    assert ndcg_at_k(["x", "y"], {"a": 2, "b": 1}, k=10) == 0.0


def test_ndcg_is_none_when_no_document_is_relevant():
    # Every label is 0, so ideal DCG = 0 and NDCG = 0/0 — undefined.
    # Returning 0.0 here would mean "this system ranked badly", which is a
    # false statement about a query where there was nothing to rank well.
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=10) is None


def test_ndcg_is_none_for_an_entirely_unjudged_query():
    assert ndcg_at_k(["a", "b"], {}, k=10) is None


def test_duplicate_ids_are_not_scored_twice():
    # Run [a, a, c] with qrels {a:2, c:1, b:0}. Without deduplication:
    #   3/log2(2) + 3/log2(3) + 1/log2(4) = 3 + 1.8927893 + 0.5 = 5.3927893
    # against an ideal of 3.6309297535714574 — an NDCG of 1.485, which is
    # nonsense. After deduplication the run is [a, c]:
    #   3/log2(2) + 1/log2(3) = 3 + 0.6309297535714574 = ideal  ->  NDCG = 1.0
    judged = {"a": 2, "c": 1, "b": 0}
    got = ndcg_at_k(["a", "a", "c"], judged, k=3)
    assert got == pytest.approx(1.0)
    assert got <= 1.0  # the invariant a duplicate bug would break


def test_dedupe_keeps_the_best_ranked_occurrence():
    assert dedupe_preserving_order(["c", "a", "c", "b", "a"]) == ["c", "a", "b"]


def test_dedupe_runs_before_the_k_cutoff():
    # [a, a, b] at k=2 must score the deduplicated top 2 ([a, b]), not [a, a]
    # truncated. Deduping after truncation would discard b and lose its gain.
    #   DCG@2 = 3/log2(2) + 1/log2(3) = 3.6309297535714574 = ideal -> 1.0
    assert ndcg_at_k(["a", "a", "b"], {"a": 2, "b": 1}, k=2) == pytest.approx(1.0)


def test_unjudged_results_are_graded_zero():
    # "x" has no label, so it contributes gain(0) = 0 rather than raising.
    assert grades_for(["a", "x", "b"], {"a": 2, "b": 1}) == [2, 0, 1]


def test_negative_grade_in_qrels_raises():
    # A -1 in qrels.csv (a typo, or a column read in the wrong order) would
    # make gain() = 2**-1 - 1 = -0.5 and quietly corrupt every DCG.
    with pytest.raises(ValueError):
        grades_for(["a"], {"a": -1})


def test_boolean_grade_is_rejected():
    # True == 1 in Python, so a boolean would pass a numeric check and silently
    # become "partially relevant".
    with pytest.raises(TypeError):
        grades_for(["a"], {"a": True})


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        dcg_at_k([2, 1], k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 2}, k=0)


# ─────────────────────────────────────────────────────────────────────────────
# Recall
# ─────────────────────────────────────────────────────────────────────────────


def test_recall_at_k_hand_computed():
    # qrels {a:2, b:1, c:0, d:2}; relevant at threshold 1 = {a, b, d} -> 3 docs.
    # Run [c, a, x, d] at k=4 contains a and d -> 2 of 3 found = 0.6666...
    judged = {"a": 2, "b": 1, "c": 0, "d": 2}
    assert recall_at_k(["c", "a", "x", "d"], judged, k=4) == pytest.approx(2 / 3)


def test_recall_threshold_changes_the_denominator_and_the_numerator():
    # Same run, threshold 2: relevant = {a, d} -> 2 docs, both found = 2/2 = 1.0
    # This is why the threshold must be reported with the number: the same run
    # scores 0.667 or 1.0 depending on a choice nobody can see in the metric.
    judged = {"a": 2, "b": 1, "c": 0, "d": 2}
    assert recall_at_k(["c", "a", "x", "d"], judged, k=4, relevance_threshold=2) == 1.0


def test_recall_respects_the_cutoff():
    # k=2 truncates the run to [c, a]; only a is relevant -> 1 of 3 = 0.3333...
    judged = {"a": 2, "b": 1, "c": 0, "d": 2}
    assert recall_at_k(["c", "a", "x", "d"], judged, k=2) == pytest.approx(1 / 3)


def test_recall_counts_a_duplicate_hit_once():
    # Run [a, a] with relevant {a, b}: 1 of 2 = 0.5, not 2/2 = 1.0.
    assert recall_at_k(["a", "a"], {"a": 2, "b": 1}, k=10) == 0.5


def test_recall_is_none_when_nothing_is_relevant():
    # 0 / 0. Same reasoning as NDCG: excluded from the mean, not scored 0.
    assert recall_at_k(["a"], {"a": 0}, k=10) is None
    assert recall_at_k(["a"], {}, k=10) is None


def test_recall_is_zero_when_relevant_docs_exist_but_none_was_retrieved():
    assert recall_at_k(["x", "y"], {"a": 2}, k=10) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal rank / MRR
# ─────────────────────────────────────────────────────────────────────────────


def test_reciprocal_rank_at_threshold_one():
    # Run [c, b, a], qrels {a:2, b:1, c:0}. At threshold 1 the first relevant
    # result is b at rank 2 -> 1/2 = 0.5
    assert reciprocal_rank(["c", "b", "a"], {"a": 2, "b": 1, "c": 0}) == 0.5


def test_reciprocal_rank_at_threshold_two():
    # Same run, threshold 2: b (grade 1) no longer counts, so the first
    # relevant result is a at rank 3 -> 1/3 = 0.3333...
    got = reciprocal_rank(["c", "b", "a"], {"a": 2, "b": 1, "c": 0}, relevance_threshold=2)
    assert got == pytest.approx(1 / 3)


def test_reciprocal_rank_is_one_when_the_top_result_is_relevant():
    assert reciprocal_rank(["a", "b"], {"a": 1}) == 1.0


def test_reciprocal_rank_is_zero_when_we_looked_and_found_nothing():
    # Relevant documents exist but are not in the returned list: a measured 0.
    assert reciprocal_rank(["x", "y"], {"a": 2}) == 0.0


def test_reciprocal_rank_is_none_when_no_document_is_relevant():
    # Nothing to find, so "how fast did we find it" has no answer.
    assert reciprocal_rank(["x"], {"x": 0}) is None
    assert reciprocal_rank(["x"], {}) is None


def test_reciprocal_rank_depth_cap():
    # Run [c, c, b]: dedupe -> [c, b], so b sits at rank 2.
    #   k=1 searches only [c] (grade 0) -> 0.0
    #   k=2 reaches b -> 1/2 = 0.5
    judged = {"b": 2, "c": 0}
    assert reciprocal_rank(["c", "c", "b"], judged, k=1) == 0.0
    assert reciprocal_rank(["c", "c", "b"], judged, k=2) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_excludes_undefined_queries_from_the_mean():
    # Scores [0.5, None, 1.0]:
    #   honest mean  = (0.5 + 1.0) / 2 = 0.75  over 2 defined queries
    #   wrong mean   = (0.5 + 0 + 1.0) / 3 = 0.5  (None treated as 0.0)
    # The second number is what a naive implementation reports, and it is
    # lower than anything any system actually achieved on any query.
    got = aggregate([0.5, None, 1.0])
    assert got == Aggregate(mean=0.75, n_defined=2, n_undefined=1)
    assert got.mean != pytest.approx(0.5)
    assert got.n_total == 3


def test_aggregate_of_all_undefined_has_no_mean():
    got = aggregate([None, None])
    assert got.mean is None
    assert got.n_defined == 0
    assert got.n_undefined == 2


def test_per_query_treats_a_missing_qrels_row_as_unjudged():
    run = {"q1": ["a"], "q2": ["b"]}
    qrels = {"q1": {"a": 2}}
    scores = per_query(run, qrels, lambda r, j: ndcg_at_k(r, j, 10))
    assert scores["q1"] == pytest.approx(1.0)
    assert scores["q2"] is None  # q2 has no labels at all


def test_subset_restricts_to_known_query_ids():
    scores = {"q1": 1.0, "q2": None, "q3": 0.5}
    assert subset(scores, ["q1", "q3", "q_missing"]) == {"q1": 1.0, "q3": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# Unjudged reporting (the pooling caveat)
# ─────────────────────────────────────────────────────────────────────────────


def test_report_unjudged_hand_computed():
    # q1 returns [a, b, c] and only a is labeled   -> 2 of 3 unjudged
    # q2 returns [d, e] and has no qrels row at all -> 2 of 2 unjudged
    # overall = (2 + 2) / (3 + 2) = 4/5 = 0.8
    run = {"q1": ["a", "b", "c"], "q2": ["d", "e"]}
    qrels = {"q1": {"a": 2}}
    got = report_unjudged(run, qrels, k=3)
    assert got.n_unjudged == 4
    assert got.n_results == 5
    assert got.fraction == pytest.approx(0.8)
    assert got.per_query["q1"] == pytest.approx(2 / 3)
    assert got.per_query["q2"] == 1.0


def test_a_zero_graded_label_is_judged_not_unjudged():
    # "b" was looked at and graded irrelevant. That is a judgement, and
    # conflating it with "never shown to a labeler" would make the pooling
    # caveat look worse than it is.
    run = {"q1": ["a", "b"]}
    qrels = {"q1": {"a": 2, "b": 0}}
    assert report_unjudged(run, qrels, k=2).fraction == 0.0


def test_report_unjudged_only_counts_the_top_k():
    # k=1 looks at [a] only, which is labeled -> 0 unjudged, 1 result.
    run = {"q1": ["a", "b", "c"]}
    qrels = {"q1": {"a": 2}}
    got = report_unjudged(run, qrels, k=1)
    assert (got.n_unjudged, got.n_results, got.fraction) == (0, 1, 0.0)


def test_report_unjudged_handles_an_empty_result_list():
    got = report_unjudged({"q1": []}, {"q1": {"a": 2}}, k=10)
    assert got.n_results == 0
    assert got.fraction == 0.0  # 0/0 reported as 0, not a crash
    assert got.per_query["q1"] == 0.0


def test_report_unjudged_deduplicates_before_counting():
    # [a, a, x] -> [a, x]: 1 unjudged of 2, not 1 of 3.
    got = report_unjudged({"q1": ["a", "a", "x"]}, {"q1": {"a": 2}}, k=3)
    assert (got.n_unjudged, got.n_results) == (1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_run: the three headline numbers together
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_run_end_to_end_hand_computed():
    # q1: run [a, b, c], qrels {a:2, b:0, c:1}
    #   NDCG@3 = 3.5 / (3 + 1/log2(3)) = 0.9639397   (the worked example above)
    #   relevant at threshold 1 = {a, c} -> both in top 100 -> recall = 2/2 = 1.0
    #   first relevant is a at rank 1 -> RR = 1.0
    # q2: run [x, y], qrels {} (entirely unjudged)
    #   NDCG, recall and RR are all undefined -> None, excluded from the means
    # Means over the one defined query: NDCG 0.9639397, recall 1.0, MRR 1.0
    # Unjudged at k=3: q1 has 0 of 3, q2 has 2 of 2 -> 2/5 = 0.4
    run = {"q1": ["a", "b", "c"], "q2": ["x", "y"]}
    qrels = {"q1": {"a": 2, "b": 0, "c": 1}}
    report = evaluate_run(run, qrels, ndcg_k=3)

    assert report.ndcg.mean == pytest.approx(3.5 / (3 + 1 / LOG2_3))
    assert report.ndcg.n_defined == 1
    assert report.ndcg.n_undefined == 1
    assert report.recall.mean == 1.0
    assert report.mrr.mean == 1.0
    assert report.ndcg_per_query["q2"] is None
    assert report.unjudged.fraction == pytest.approx(0.4)
    # The cutoffs used are carried in the report so a table cannot mislabel them.
    assert (report.ndcg_k, report.recall_k, report.relevance_threshold) == (3, 100, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────


def test_bootstrap_ci_of_identical_scores_is_a_point():
    # Every query scores 0.6, so every resample — whatever queries it draws —
    # has mean 0.6. A zero-width interval is the correct answer and a useful
    # sanity check that no noise is being injected anywhere.
    got = bootstrap_ci({f"q{i}": 0.6 for i in range(20)}, n=200, seed=42)
    assert got.mean == pytest.approx(0.6)
    assert got.low == pytest.approx(0.6)
    assert got.high == pytest.approx(0.6)
    assert got.n_queries == 20
    assert got.n_resamples == 200


def test_bootstrap_ci_on_two_extreme_scores_spans_the_whole_range():
    # With scores {0.0, 1.0} a resample of size 2 has mean 0.0 (p=0.25),
    # 0.5 (p=0.5) or 1.0 (p=0.25). Since 25% of resamples sit at each end —
    # far more than the 2.5% tail — the 95% percentile interval is exactly
    # [0.0, 1.0], and the observed mean is (0 + 1)/2 = 0.5.
    got = bootstrap_ci({"q1": 0.0, "q2": 1.0}, n=2000, seed=42)
    assert got.mean == 0.5
    assert got.low == 0.0
    assert got.high == 1.0


def test_bootstrap_ci_of_a_single_query_is_that_query():
    # Resampling one value with replacement can only ever draw that value.
    # The interval is degenerate, which is the honest answer: one query tells
    # you nothing about variability across queries.
    got = bootstrap_ci({"q1": 0.42}, n=50, seed=1)
    assert (got.mean, got.low, got.high) == (0.42, 0.42, 0.42)


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    # A CI that moves between runs cannot be quoted in a README.
    scores = {f"q{i}": i / 10 for i in range(10)}
    first = bootstrap_ci(scores, n=500, seed=7)
    second = bootstrap_ci(scores, n=500, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_bootstrap_ci_ignores_undefined_queries():
    # The three defined scores are 0.6 each; the Nones must not be drawn and
    # must not shrink the mean. Mean stays 0.6, n_queries counts 3, not 5.
    got = bootstrap_ci({"q1": 0.6, "q2": None, "q3": 0.6, "q4": None, "q5": 0.6}, n=100)
    assert got.mean == pytest.approx(0.6)
    assert got.n_queries == 3


def test_bootstrap_ci_brackets_the_observed_mean():
    scores = {f"q{i}": v for i, v in enumerate([0.1, 0.4, 0.5, 0.55, 0.9, 0.2, 0.7])}
    got = bootstrap_ci(scores, n=1000, seed=42)
    assert got.low <= got.mean <= got.high


def test_bootstrap_ci_accepts_a_plain_sequence():
    got = bootstrap_ci([0.5, 0.5, 0.5], n=50)
    assert got.mean == pytest.approx(0.5)


def test_bootstrap_ci_rejects_degenerate_input():
    with pytest.raises(ValueError):
        bootstrap_ci({}, n=100)
    with pytest.raises(ValueError):
        bootstrap_ci({"q1": None}, n=100)
    with pytest.raises(ValueError):
        bootstrap_ci({"q1": 0.5}, n=0)
    with pytest.raises(ValueError):
        bootstrap_ci({"q1": 0.5}, confidence=1.0)


def test_bootstrap_ci_from_config_uses_config_values(config):
    got = bootstrap_ci_from_config({"q1": 0.5, "q2": 0.5}, config)
    assert got.n_resamples == config.evaluation.bootstrap_samples
    assert got.confidence == config.evaluation.confidence
    # Same seed as the config means the wrapper and a direct call must agree.
    direct = bootstrap_ci(
        {"q1": 0.5, "q2": 0.5},
        n=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence,
        seed=config.seed,
    )
    assert (got.low, got.high) == (direct.low, direct.high)


# ─────────────────────────────────────────────────────────────────────────────
# Paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def test_paired_bootstrap_with_a_constant_improvement():
    # System B beats A by exactly 0.1 on every query. Because the SAME
    # resampled queries are used for both systems, every resample's mean
    # difference is exactly 0.1 — the pairing cancels query difficulty
    # completely, so the interval has zero width.
    a = {"q1": 0.1, "q2": 0.5, "q3": 0.9}
    b = {"q1": 0.2, "q2": 0.6, "q3": 1.0}
    got = paired_bootstrap(a, b, n=999, seed=42)
    assert got.mean_diff == pytest.approx(0.1)
    assert got.low == pytest.approx(0.1)
    assert got.high == pytest.approx(0.1)
    # mean_a = (0.1 + 0.5 + 0.9)/3 = 0.5 ; mean_b = (0.2 + 0.6 + 1.0)/3 = 0.6
    assert got.mean_a == pytest.approx(0.5)
    assert got.mean_b == pytest.approx(0.6)
    # No resample lands on the wrong side of zero, so with the (count+1)/(n+1)
    # convention: p = 2 * (0 + 1) / (999 + 1) = 0.002
    assert got.p_value == pytest.approx(0.002)
    assert got.significant is True


def test_paired_bootstrap_of_a_system_against_itself_finds_nothing():
    # Every difference is 0, so the observed difference is 0, the interval is
    # [0, 0], and p is capped at 1.0: 2 * (2000 + 1)/(2000 + 1) = 2 -> 1.0
    a = {"q1": 0.3, "q2": 0.8}
    got = paired_bootstrap(a, dict(a), n=2000, seed=42)
    assert got.mean_diff == 0.0
    assert (got.low, got.high) == (0.0, 0.0)
    assert got.p_value == 1.0
    assert got.significant is False


def test_paired_bootstrap_sign_convention_is_b_minus_a():
    # B is worse everywhere, so the difference must be negative. Getting this
    # backwards would turn every regression in the ablation into a win.
    a = {"q1": 0.9, "q2": 0.9}
    b = {"q1": 0.4, "q2": 0.4}
    got = paired_bootstrap(a, b, n=199, seed=3)
    assert got.mean_diff == pytest.approx(-0.5)
    assert got.high < 0


def test_paired_bootstrap_drops_pairs_where_either_side_is_undefined():
    # q2 is undefined for A, so the pair is dropped: the comparison runs on
    # q1 and q3 only. Keeping q2 by substituting 0.0 for A would invent a
    # half-point win for B out of a missing measurement.
    a = {"q1": 0.4, "q2": None, "q3": 0.6}
    b = {"q1": 0.5, "q2": 0.5, "q3": 0.7}
    got = paired_bootstrap(a, b, n=99, seed=1)
    assert got.n_queries == 2
    assert got.n_dropped == 1
    # mean over the surviving pairs: ((0.5-0.4) + (0.7-0.6)) / 2 = 0.1
    assert got.mean_diff == pytest.approx(0.1)


def test_paired_bootstrap_requires_the_same_query_set():
    with pytest.raises(ValueError):
        paired_bootstrap({"q1": 0.5}, {"q1": 0.5, "q2": 0.5}, n=10)


def test_paired_bootstrap_rejects_a_comparison_with_no_usable_pair():
    with pytest.raises(ValueError):
        paired_bootstrap({"q1": None}, {"q1": 0.5}, n=10)


def test_paired_bootstrap_is_reproducible_for_a_fixed_seed():
    a = {f"q{i}": (i % 5) / 10 for i in range(12)}
    b = {f"q{i}": (i % 7) / 10 for i in range(12)}
    first = paired_bootstrap(a, b, n=300, seed=11)
    second = paired_bootstrap(a, b, n=300, seed=11)
    assert (first.low, first.high, first.p_value) == (second.low, second.high, second.p_value)


def test_paired_bootstrap_is_insensitive_to_dict_ordering():
    # Query ids are sorted internally, so two dicts holding the same pairs in
    # a different insertion order must give the same interval. Without that,
    # a run whose queries arrived in a different order would report a
    # different CI from identical data.
    a1 = {"q1": 0.2, "q2": 0.6, "q3": 0.4}
    b1 = {"q1": 0.3, "q2": 0.5, "q3": 0.9}
    a2 = {"q3": 0.4, "q1": 0.2, "q2": 0.6}
    b2 = {"q2": 0.5, "q3": 0.9, "q1": 0.3}
    first = paired_bootstrap(a1, b1, n=200, seed=5)
    second = paired_bootstrap(a2, b2, n=200, seed=5)
    assert (first.low, first.high, first.p_value) == (second.low, second.high, second.p_value)


def test_paired_bootstrap_from_config_uses_config_values(config):
    got = paired_bootstrap_from_config({"q1": 0.4}, {"q1": 0.5}, config)
    assert got.n_resamples == config.evaluation.bootstrap_samples
    assert got.confidence == config.evaluation.confidence


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix and Cohen's kappa
# ─────────────────────────────────────────────────────────────────────────────


def _two_by_two_labels() -> tuple[list[int], list[int]]:
    """The textbook 2x2 example: confusion matrix [[20, 5], [10, 15]].

    20 pairs both rated 0, 5 rated (0 by A, 1 by B), 10 rated (1 by A, 0 by B),
    15 both rated 1. n = 50.
    """
    a = [0] * 20 + [0] * 5 + [1] * 10 + [1] * 15
    b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
    return a, b


def test_confusion_matrix_textbook_example():
    a, b = _two_by_two_labels()
    assert confusion_matrix(a, b) == ((20, 5), (10, 15))


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])


def test_confusion_matrix_rejects_empty_input():
    with pytest.raises(ValueError):
        confusion_matrix([], [])


def test_confusion_matrix_rejects_a_label_outside_the_declared_categories():
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 0], categories=(0, 1, 2))


def test_unweighted_kappa_textbook_2x2():
    # Matrix [[20, 5], [10, 15]], n = 50.
    #   observed agreement  Po = (20 + 15) / 50 = 35/50 = 0.70
    #   A's marginals: row totals 25, 25 -> 0.5, 0.5
    #   B's marginals: col totals 30, 20 -> 0.6, 0.4
    #   chance agreement   Pe = 0.5*0.6 + 0.5*0.4 = 0.30 + 0.20 = 0.50
    #   kappa = (Po - Pe) / (1 - Pe) = (0.70 - 0.50) / 0.50 = 0.40
    a, b = _two_by_two_labels()
    got = cohens_kappa(a, b, weights="none")
    assert got.kappa == pytest.approx(0.4)
    assert got.observed_agreement == pytest.approx(0.7)
    assert got.expected_agreement == pytest.approx(0.5)
    assert got.n == 50
    assert got.matrix == ((20, 5), (10, 15))


def test_quadratic_weights_equal_unweighted_on_a_two_category_scale():
    # With C = 2 the only disagreement step is |i-j|/(C-1) = 1, and 1**2 = 1,
    # so quadratic weighting cannot differ from unweighted here. A useful
    # guard: it proves the weighting code reduces to the classic formula.
    a, b = _two_by_two_labels()
    assert cohens_kappa(a, b, weights="quadratic").kappa == pytest.approx(0.4)


def test_quadratic_kappa_on_ordinal_grades_hand_computed():
    # A = [0, 0, 1, 2, 2], B = [0, 1, 1, 1, 2], n = 5. Matrix (rows A, cols B):
    #        B=0  B=1  B=2        row totals
    #   A=0 [  1    1    0 ]  -> 2
    #   A=1 [  0    1    0 ]  -> 1
    #   A=2 [  0    1    1 ]  -> 2
    #   col totals: 1, 3, 1
    #
    # Quadratic disagreement weights, C = 3, w_ij = ((i-j)/2)**2:
    #   w(0,1) = w(1,0) = w(1,2) = w(2,1) = 0.25 ; w(0,2) = w(2,0) = 1 ; w(i,i) = 0
    #
    # Observed weighted disagreement = sum(w_ij * O_ij) / n
    #   = (0.25*1 [A0,B1] + 0.25*1 [A2,B1]) / 5 = 0.5 / 5 = 0.10
    #
    # Expected weighted disagreement = sum(w_ij * p_row_i * p_col_j)
    #   row fractions 0.4, 0.2, 0.4 ; col fractions 0.2, 0.6, 0.2
    #   (0,1): 0.25 * 0.4 * 0.6 = 0.06
    #   (0,2): 1.00 * 0.4 * 0.2 = 0.08
    #   (1,0): 0.25 * 0.2 * 0.2 = 0.01
    #   (1,2): 0.25 * 0.2 * 0.2 = 0.01
    #   (2,0): 1.00 * 0.4 * 0.2 = 0.08
    #   (2,1): 0.25 * 0.4 * 0.6 = 0.06
    #   total = 0.30
    #
    # kappa_w = 1 - 0.10 / 0.30 = 1 - 1/3 = 0.666666...
    a = [0, 0, 1, 2, 2]
    b = [0, 1, 1, 1, 2]
    got = cohens_kappa(a, b, weights="quadratic")
    assert got.kappa == pytest.approx(2 / 3)
    assert got.matrix == ((1, 1, 0), (0, 1, 0), (0, 1, 1))
    assert got.categories == (0, 1, 2)
    assert got.weights == "quadratic"


def test_unweighted_kappa_on_the_same_ordinal_example_is_lower():
    # Same data, unweighted:
    #   Po = (1 + 1 + 1) / 5 = 0.60        (the three diagonal cells)
    #   Pe = 0.4*0.2 + 0.2*0.6 + 0.4*0.2 = 0.08 + 0.12 + 0.08 = 0.28
    #   kappa = (0.60 - 0.28) / (1 - 0.28) = 0.32 / 0.72 = 0.444444...
    # Lower than the quadratic 0.667 because both disagreements here are
    # one-step near misses (0-vs-1, 2-vs-1), which unweighted kappa punishes
    # as harshly as a 0-vs-2 blunder. This gap is exactly why the weighting
    # scheme must be reported with the number.
    got = cohens_kappa([0, 0, 1, 2, 2], [0, 1, 1, 1, 2], weights="none")
    assert got.kappa == pytest.approx(4 / 9)


def test_linear_kappa_sits_between_unweighted_and_quadratic():
    # Same data, linear weights w_ij = |i-j|/2:
    #   observed = (0.5 [A0,B1] + 0.5 [A2,B1]) / 5 = 1.0 / 5 = 0.20
    #   expected = (0,1) 0.5*0.24 = 0.12 ; (0,2) 1*0.08 = 0.08
    #            + (1,0) 0.5*0.04 = 0.02 ; (1,2) 0.5*0.04 = 0.02
    #            + (2,0) 1*0.08 = 0.08  ; (2,1) 0.5*0.24 = 0.12
    #            = 0.44
    #   kappa = 1 - 0.20/0.44 = 1 - 5/11 = 6/11 = 0.545454...
    got = cohens_kappa([0, 0, 1, 2, 2], [0, 1, 1, 1, 2], weights="linear")
    assert got.kappa == pytest.approx(6 / 11)
    assert 4 / 9 < got.kappa < 2 / 3


def test_perfect_agreement_is_kappa_one():
    # Po = 1, so (1 - Pe)/(1 - Pe) = 1 for any Pe < 1.
    got = cohens_kappa([0, 1, 2, 0, 1], [0, 1, 2, 0, 1], weights="quadratic")
    assert got.kappa == pytest.approx(1.0)
    assert got.observed_agreement == pytest.approx(1.0)


def test_systematic_opposite_labels_give_kappa_minus_one():
    # Matrix [[0, 2], [2, 0]]: Po = 0, marginals all 0.5 -> Pe = 0.5,
    # kappa = (0 - 0.5) / (1 - 0.5) = -1.0 (worse than chance).
    got = cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0], weights="none")
    assert got.kappa == pytest.approx(-1.0)


def test_kappa_is_none_when_only_one_category_was_used():
    # Both raters labeled everything 0. Percent agreement is 100%, but chance
    # agreement is also 100%: a rater that always says 0 agrees perfectly with
    # another that always says 0 while demonstrating no skill. kappa = 0/0.
    # Returning 1.0 here would be the most flattering possible lie about label
    # quality, which is why this case returns None with a note instead.
    got = cohens_kappa([0, 0, 0], [0, 0, 0])
    assert got.kappa is None
    assert got.observed_agreement == pytest.approx(1.0)
    assert "undefined" in got.note


def test_kappa_is_none_for_one_used_category_even_with_declared_categories():
    # Declaring the full 0/1/2 scale does not rescue it: the expected
    # disagreement is still zero because both marginals put all mass on one
    # category.
    got = cohens_kappa([0, 0], [0, 0], categories=(0, 1, 2))
    assert got.kappa is None
    assert got.categories == (0, 1, 2)
    assert got.matrix == ((2, 0, 0), (0, 0, 0), (0, 0, 0))


def test_kappa_rejects_an_unknown_weighting_scheme():
    with pytest.raises(ValueError):
        cohens_kappa([0, 1], [0, 1], weights="cubic")


def test_kappa_rejects_mismatched_label_lists():
    with pytest.raises(ValueError):
        cohens_kappa([0, 1, 2], [0, 1])
