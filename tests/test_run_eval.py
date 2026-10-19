"""Tests for ``make eval`` — the harness that produces every README number.

House rule, inherited from ``tests/test_metrics.py``: **no expected value in
this file was obtained by running the code.** Each one is derived by hand in a
comment above its assertion, so the owner can re-derive it on a whiteboard and
so a refactor that changes a formula fails instead of quietly agreeing with
itself.

The synthetic world used throughout is defined once in ``SYNTHETIC_QRELS`` and
``RUN_A`` / ``RUN_B`` below, with its arithmetic worked out in the comments
there. Three things are being guarded:

* the ablation table reproduces metrics that were computed by hand;
* a confidence interval that crosses zero produces hedged *words*, not just a
  smaller number;
* when labels are missing the harness refuses to emit anything rather than
  filling the table with examples.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from concierge.pipeline import ABLATION, VARIANTS, RerankCheckpoint, Stage, Variant
from concierge.types import (
    RELEVANCE_HIGH,
    Business,
    ParsedQuery,
    ParseSource,
    QueryType,
    SearchResponse,
    SearchResult,
)
from eval import report
from eval.run_eval import (
    FULL_SYSTEM_KEY,
    GOLD_SWEEP_WARNING,
    NOT_DISTINGUISHABLE_PHRASE,
    SWEEP_ON_GOLD,
    SWEEP_ON_VALIDATION,
    VALIDATION_SPLIT,
    CandidateDepths,
    EvalError,
    LabelsNotReady,
    NoSystemRan,
    PipelineFactory,
    RetrievedDoc,
    SweepSplitNotReady,
    SystemRun,
    SystemSpec,
    ablation_specs,
    depths_for_eval,
    diagnose_query,
    main,
    ndcg_k_from_config,
    percentile_nearest_rank,
    run_eval,
    stage_names,
    validation_sweep_query_set,
)

LOG2_3 = math.log2(3)  # 1.5849625007211562 — the rank-2 discount


# ─────────────────────────────────────────────────────────────────────────────
# A fake pipeline: canned result lists, no indexes, no models, no dataset
# ─────────────────────────────────────────────────────────────────────────────


def _business(business_id: str) -> Business:
    """A minimal Business. Only the id and name are read by the harness."""
    return Business(
        business_id=business_id,
        name=f"Place {business_id}",
        city="Metroville",
        state="XX",
        latitude=None,
        longitude=None,
        stars=4.0,
        review_count=10,
        categories=("Cafes",),
        attributes={},
        hours={},
        is_open=True,
    )


def _response(query: str, business_ids: Sequence[str]) -> SearchResponse:
    return SearchResponse(
        parsed=ParsedQuery(
            raw=query, semantic_text=query, source=ParseSource.RULES, parse_latency_ms=1.5
        ),
        results=tuple(
            SearchResult(
                business=_business(bid),
                rank=rank,
                score=1.0 / rank,
                constraints=(),
            )
            for rank, bid in enumerate(business_ids, start=1)
        ),
        timings=(),
        total_ms=1.0,
    )


class FakeSearcher:
    """Returns a canned ranked list per query text."""

    def __init__(self, runs: Mapping[str, Sequence[str]], raise_on: Sequence[str] = ()) -> None:
        self.runs = runs
        self.raise_on = set(raise_on)

    def search(self, query: str) -> SearchResponse:
        if query in self.raise_on:
            raise RuntimeError("fake stage exploded")
        return _response(query, self.runs.get(query, ()))


def make_factory(
    by_variant: Mapping[Variant, Mapping[str, Sequence[str]]],
    raise_on: Sequence[str] = (),
):
    """A searcher factory that only knows the variants it was given.

    Anything else raises, which is exactly what ``PipelineFactory`` does for a
    variant whose model has not been trained yet.
    """

    def factory(cfg, variant: Variant, depths: CandidateDepths) -> FakeSearcher:
        if variant not in by_variant:
            raise RuntimeError(f"stage not available for {variant.name}")
        return FakeSearcher(by_variant[variant], raise_on=raise_on)

    return factory


# ─────────────────────────────────────────────────────────────────────────────
# The synthetic world, with its arithmetic worked out
# ─────────────────────────────────────────────────────────────────────────────

# Five gold queries, one per type plus a second vibe query, chosen so that the
# awkward cases are all present:
#   q1  two relevant results, both found at the top
#   q2  one relevant result, found at rank 2 behind an unjudged result
#   q3  two relevant results
#   q4  a pool that was judged and is entirely irrelevant -> NDCG undefined
#   q5  no labels at all                                  -> NDCG undefined
GOLD_ROWS = [
    ("q1", "quiet cafe to read in", "vibe"),
    ("q2", "open past 10pm with outdoor seating", "constraint"),
    ("q3", "good pho", "specific"),
    ("q4", "sumthing nice idk", "messy"),
    ("q5", "cozy bar", "vibe"),
]
QUERY_TEXT = {qid: text for qid, text, _ in GOLD_ROWS}

SYNTHETIC_QRELS: dict[str, dict[str, int]] = {
    "q1": {"b1": 2, "b2": 1},
    "q2": {"b3": 2},
    "q3": {"b4": 2, "b5": 1},
    "q4": {"b6": 0},
    # q5 deliberately absent from qrels.
}

# System A. Ranked lists keyed by query text, as the pipeline receives them.
RUN_A: dict[str, list[str]] = {
    QUERY_TEXT["q1"]: ["b1", "b2"],
    QUERY_TEXT["q2"]: ["bX", "b3"],
    QUERY_TEXT["q3"]: ["b4", "b5"],
    QUERY_TEXT["q4"]: ["b6"],
    QUERY_TEXT["q5"]: ["b9"],
}

# System B differs from A on exactly one query: q2's relevant result moves from
# rank 2 to rank 1. One changed query out of three defined ones is what makes
# the paired bootstrap below land on "not distinguishable".
RUN_B: dict[str, list[str]] = dict(RUN_A) | {QUERY_TEXT["q2"]: ["b3", "bX"]}

# Two real pipeline variants: BM25 alone, then BM25 + dense fused by RRF. Real
# `Variant` objects rather than stand-ins, so the harness is exercised against
# the same toggle type `make eval` will hand it.
VARIANT_A = VARIANTS["bm25"]
VARIANT_B = VARIANTS["hybrid_rrf"]

SPEC_A = SystemSpec(key="a", label="A", variant=VARIANT_A)
SPEC_B = SystemSpec(key="b", label="B", variant=VARIANT_B, compare_to="a")
TWO_SYSTEMS = (SPEC_A, SPEC_B)

# ── Hand-computed metrics for system A ───────────────────────────────────────
#
# NDCG@10 uses exponential gain 2**g - 1 and discount log2(rank + 1).
#
# q1: returned [b1(2), b2(1)]
#     DCG   = 3/log2(2) + 1/log2(3) = 3 + 1/1.5849625 = 3.6309297535714574
#     ideal = same two labels in the same order                = 3.6309297535714574
#     NDCG  = 1.0
# q2: returned [bX(unjudged -> 0), b3(2)]
#     DCG   = 0 + 3/log2(3) = 1.8927892607143717
#     ideal = 3/log2(2) = 3.0
#     NDCG  = 1.8927892607143717 / 3 = 0.6309297535714572
# q3: returned [b4(2), b5(1)] -> same shape as q1 -> NDCG = 1.0
# q4: every label is 0 -> ideal DCG = 0 -> NDCG undefined (None)
# q5: no labels        -> ideal DCG = 0 -> NDCG undefined (None)
#
# mean over the 3 defined queries = (1.0 + 0.6309297535714572 + 1.0) / 3
NDCG_A_Q2 = 1.0 / LOG2_3  # = (3/log2(3)) / 3
NDCG_A_MEAN = (1.0 + NDCG_A_Q2 + 1.0) / 3  # = 0.8769765845238191

# Recall@100, relevance threshold 1 (partially relevant counts):
#   q1: {b1,b2} both in the list -> 2/2 = 1.0
#   q2: {b3} in the list        -> 1/1 = 1.0
#   q3: {b4,b5} both            -> 2/2 = 1.0
#   q4, q5: no relevant labels  -> undefined
RECALL_A_MEAN = 1.0

# MRR, threshold 1:
#   q1: first relevant at rank 1 -> 1.0
#   q2: b3 at rank 2             -> 0.5
#   q3: rank 1                   -> 1.0
#   mean = 2.5 / 3
MRR_A_MEAN = 2.5 / 3  # = 0.8333333333333334

# Unjudged share of the top 10, counted over all five queries:
#   q1 [b1,b2]  -> 0 unjudged of 2
#   q2 [bX,b3]  -> 1 of 2   (bX was never pooled)
#   q3 [b4,b5]  -> 0 of 2
#   q4 [b6]     -> 0 of 1   (graded 0 is still a label)
#   q5 [b9]     -> 1 of 1   (q5 has no qrels row at all)
#   total = 2 unjudged / 8 returned = 0.25
UNJUDGED_A = 2 / 8


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def write_gold(path: Path, rows: Sequence[tuple[str, str, str]] = tuple(GOLD_ROWS)) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "query", "query_type"])
        writer.writerows(rows)
    return path


def write_qrels(path: Path, qrels: Mapping[str, Mapping[str, int]] = SYNTHETIC_QRELS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "business_id", "grade"])
        for qid, judged in qrels.items():
            for bid, grade in judged.items():
                writer.writerow([qid, bid, grade])
    return path


@pytest.fixture
def eval_config(config, tmp_path: Path):
    """The shared config fixture, pointed at a synthetic gold set and labels.

    ``bootstrap_samples`` is lowered to keep the suite fast. Every assertion
    about an interval below is argued from the *shape* of the resample
    distribution, so it holds for any resample count.
    """
    gold = write_gold(tmp_path / "gold_queries.csv")
    qrels = write_qrels(tmp_path / "qrels.csv")
    return config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "gold_queries": gold,
                    "qrels": qrels,
                    "out_dir": tmp_path / "artifacts",
                    "bootstrap_samples": 500,
                }
            )
        }
    )


def run_two_systems(cfg, runs_a=RUN_A, runs_b=RUN_B, **kwargs):
    """Run the two-system ablation over the synthetic world.

    The label floor defaults to 3 (the synthetic world has exactly three
    judgeable queries) and individual tests override it.
    """
    factory = make_factory({VARIANT_A: runs_a, VARIANT_B: runs_b})
    kwargs.setdefault("min_labeled_queries", 3)
    kwargs.setdefault("run_sweep", False)
    return run_eval(cfg, factory=factory, specs=TWO_SYSTEMS, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# The ablation ladder is the architecture
# ─────────────────────────────────────────────────────────────────────────────


def test_the_ladder_is_the_pipelines_own_ablation_in_its_own_order():
    # The rows are defined in concierge/pipeline.py so the table cannot drift
    # from the code that produced it. This harness must take that list as given
    # rather than keep a second copy of it.
    ladder = [s for s in ablation_specs() if not s.supplementary]
    assert [s.key for s in ladder] == [v.name for v in ABLATION]
    assert [s.variant for s in ladder] == list(ABLATION)
    assert ABLATION[-1].name == FULL_SYSTEM_KEY
    # And it is the ladder CLAUDE.md §6 asks for: BM25 -> +dense (RRF) ->
    # +cross-encoder (pretrained) -> +cross-encoder (fine-tuned) ->
    # +query parsing -> +LambdaMART.
    assert [s.key for s in ladder] == [
        "bm25",
        "hybrid_rrf",
        "rerank_pretrained",
        "rerank_finetuned",
        "llm_parse",
        "full",
    ]


def test_each_rung_changes_exactly_one_thing_from_the_one_above():
    # The ladder's whole claim is that the paired difference between
    # consecutive rows is *that stage's* contribution. A row that changed two
    # things at once would make its difference unattributable.
    ladder = [s for s in ablation_specs() if not s.supplementary]
    for previous, spec in itertools.pairwise(ladder):
        added = spec.variant.stages & ~previous.variant.stages
        removed = previous.variant.stages & ~spec.variant.stages
        checkpoint_changed = (
            spec.variant.enabled(Stage.RERANK)
            and previous.variant.enabled(Stage.RERANK)
            and spec.variant.reranker != previous.variant.reranker
        )
        assert not removed, f"{spec.key} turns a stage off"
        n_added = sum(1 for flag in Stage if flag.value and flag in added)
        # Exactly one of: one stage switched on, or the cross-encoder
        # checkpoint swapped (which is the pretrained -> fine-tuned row).
        assert n_added + int(checkpoint_changed) == 1, (
            f"{spec.key} changes {n_added} stage(s) and "
            f"{int(checkpoint_changed)} checkpoint(s), not exactly one thing"
        )


def test_every_rung_compares_against_the_row_directly_above_it():
    ladder = [s for s in ablation_specs() if not s.supplementary]
    keys = [s.key for s in ladder]
    for index, spec in enumerate(ladder):
        expected = None if index == 0 else keys[index - 1]
        assert spec.compare_to == expected


def test_supplementary_rows_are_marked_and_kept_off_the_ladder():
    specs = ablation_specs()
    supplementary = [s for s in specs if s.supplementary]
    # They exist (the filters question is a project requirement) and none of
    # them is the comparison baseline for a ladder row.
    assert {s.key for s in supplementary} == {"no_filters", "dense_only"}
    ladder_baselines = {s.compare_to for s in specs if not s.supplementary}
    assert ladder_baselines.isdisjoint({s.key for s in supplementary})
    assert ablation_specs(include_supplementary=False) == tuple(
        s for s in specs if not s.supplementary
    )


def test_a_variant_whose_model_is_missing_is_refused_rather_than_degraded(eval_config):
    # SearchPipeline deliberately degrades (fine-tuned -> pretrained -> none)
    # so it stays runnable mid-project. For an ablation row that would be a
    # fabricated measurement: "+cross-encoder (fine-tuned)" would be reported
    # from a run with no fine-tuned model. So the artifacts are checked first.
    class _Loaded:
        """Stands in for a loaded pipeline with the dense half and the
        fine-tuned checkpoint still missing."""

        bm25_index = object()
        dense_index = None
        rerankers: ClassVar[dict] = {RerankCheckpoint.PRETRAINED: object()}
        ltr_ranker = None

    loaded = _Loaded()
    assert PipelineFactory.missing_artifacts(loaded, VARIANTS["bm25"]) == []

    dense_missing = PipelineFactory.missing_artifacts(loaded, VARIANTS["hybrid_rrf"])
    assert len(dense_missing) == 1 and "dense" in dense_missing[0]

    # The pretrained checkpoint is present, so that row can run ...
    assert PipelineFactory.missing_artifacts(loaded, VARIANTS["rerank_pretrained"]) == [
        dense_missing[0]
    ]
    # ... and the fine-tuned one cannot, because its own weights are absent.
    finetuned = PipelineFactory.missing_artifacts(loaded, VARIANTS["rerank_finetuned"])
    assert any("finetuned cross-encoder" in reason for reason in finetuned)
    assert any("make train-reranker" in reason for reason in finetuned)

    full = PipelineFactory.missing_artifacts(loaded, VARIANTS["full"])
    assert any("LambdaMART" in reason for reason in full)
    assert any("make train-ltr" in reason for reason in full)


def test_depths_are_applied_as_a_config_copy_not_a_mutation(eval_config):
    # The pipeline reads its cascade widths from cfg.candidates, so the sweep
    # varies them with a copy. The caller's config must come back untouched, or
    # the fingerprint recorded with a result would describe the wrong settings.
    depths = CandidateDepths(retrieve=400, rerank=100, final=10)
    tuned = depths.applied_to(eval_config)
    assert (tuned.candidates.retrieve, tuned.candidates.rerank, tuned.candidates.final) == (
        400,
        100,
        10,
    )
    assert eval_config.candidates.retrieve == 200  # configs/base.yaml, unchanged
    assert eval_config.candidates.rerank == 50
    assert eval_config.candidates.final == 10


def test_a_row_reports_which_stages_and_which_checkpoint_it_ran():
    # The JSON and the markdown have to be readable by the one person who must
    # defend these rows, so the flag integer is rendered as stage names and the
    # checkpoint is reported only where Stage 3 is actually on.
    specs = {s.key: s for s in ablation_specs()}
    assert stage_names(VARIANTS["bm25"]) == ("BM25", "FILTERS")
    assert specs["bm25"].reranker is None
    assert specs["rerank_pretrained"].reranker == RerankCheckpoint.PRETRAINED.value
    assert specs["rerank_finetuned"].reranker == RerankCheckpoint.FINETUNED.value
    assert "DENSE" in specs["hybrid_rrf"].stages
    assert "FILTERS" not in specs["no_filters"].stages


# ─────────────────────────────────────────────────────────────────────────────
# The table reproduces hand-computed metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_ablation_row_metrics_match_the_hand_computed_values(eval_config):
    results = run_two_systems(eval_config)
    row_a = next(r for r in results.rows if r.key == "a")

    # mean NDCG@10 = (1.0 + 0.6309297535714572 + 1.0) / 3
    assert row_a.ndcg is not None
    assert row_a.ndcg.mean == pytest.approx(NDCG_A_MEAN)
    assert row_a.ndcg.mean == pytest.approx(0.8769765845238191)
    assert row_a.recall is not None and row_a.recall.mean == pytest.approx(RECALL_A_MEAN)
    assert row_a.mrr is not None and row_a.mrr.mean == pytest.approx(MRR_A_MEAN)
    assert row_a.mrr.mean == pytest.approx(0.8333333333333334)

    # System B fixes q2's ranking, so all three defined queries score 1.0.
    row_b = next(r for r in results.rows if r.key == "b")
    assert row_b.ndcg is not None and row_b.ndcg.mean == pytest.approx(1.0)
    assert row_b.mrr is not None and row_b.mrr.mean == pytest.approx(1.0)


def test_every_metric_carries_the_number_of_queries_it_is_defined_on(eval_config):
    results = run_two_systems(eval_config)
    row_a = next(r for r in results.rows if r.key == "a")
    # q4 (all labels 0) and q5 (no labels) have no defined NDCG: 3 of 5.
    assert (row_a.ndcg.n_defined, row_a.ndcg.n_undefined) == (3, 2)
    assert row_a.ndcg.n_total == 5
    assert (row_a.recall.n_defined, row_a.recall.n_undefined) == (3, 2)
    assert (row_a.mrr.n_defined, row_a.mrr.n_undefined) == (3, 2)
    # Coverage counts travel with the result so no table can omit them.
    assert results.coverage.n_gold_queries == 5
    assert results.coverage.n_queries_with_labels == 4  # q5 has no qrels row
    assert results.coverage.n_queries_with_relevant == 3  # q4's pool is all zeros
    # 2 + 1 + 2 + 1 labeled pairs
    assert results.coverage.n_labeled_pairs == 6
    assert results.coverage.n_positive_pairs == 5  # b6's grade 0 is not positive


def test_unjudged_fraction_of_the_top_k_is_hand_computed(eval_config):
    results = run_two_systems(eval_config)
    row_a = next(r for r in results.rows if r.key == "a")
    # 2 unjudged results (bX for q2, b9 for q5) out of 8 returned.
    assert row_a.n_unjudged == 2
    assert row_a.n_top_k_results == 8
    assert row_a.unjudged_fraction == pytest.approx(UNJUDGED_A)
    assert row_a.unjudged_fraction == pytest.approx(0.25)

    # B returns the same documents in a different order for q2, so its
    # unjudged share is identical — reordering does not change what is labeled.
    row_b = next(r for r in results.rows if r.key == "b")
    assert row_b.n_unjudged == 2
    assert row_b.unjudged_fraction == pytest.approx(0.25)


def test_a_query_that_returns_an_unjudged_only_list_counts_as_fully_unjudged(eval_config):
    # q1's two judged results replaced by two results nobody pooled: the top-10
    # unjudged count rises from 2/8 to 4/8 while the ranking "looks" the same.
    runs = dict(RUN_A) | {QUERY_TEXT["q1"]: ["bY", "bZ"]}
    results = run_two_systems(eval_config, runs_a=runs)
    row_a = next(r for r in results.rows if r.key == "a")
    assert row_a.n_unjudged == 4
    assert row_a.unjudged_fraction == pytest.approx(0.5)
    # ... and NDCG for q1 collapses to 0.0, which is defined, not undefined:
    # the query has relevant labels, the system just did not find them.
    assert row_a.ndcg.n_defined == 3
    assert row_a.ndcg.mean == pytest.approx((0.0 + NDCG_A_Q2 + 1.0) / 3)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical honesty
# ─────────────────────────────────────────────────────────────────────────────


def test_a_ci_crossing_zero_is_hedged_in_words(eval_config):
    results = run_two_systems(eval_config)
    row_b = next(r for r in results.rows if r.key == "b")
    comparison = row_b.vs_previous
    assert comparison is not None and comparison.computed

    # Hand reasoning: three queries have a defined NDCG under both systems and
    # the paired differences are (q1) 0, (q2) 1 - 0.6309... = 0.3690702..., (q3) 0.
    # Observed mean difference = 0.3690702464285428 / 3 = 0.12302341547618093.
    assert comparison.mean_diff == pytest.approx((1.0 - NDCG_A_Q2) / 3)
    # A bootstrap resample draws 3 pairs with replacement. It misses q2 entirely
    # with probability (2/3)**3 = 8/27 ~ 29.6% of the time, and those resamples
    # have a mean difference of exactly 0. So the 2.5th percentile of the
    # resample distribution is 0 — the interval touches zero and the difference
    # is not distinguishable, however large the point estimate looks.
    assert comparison.low == pytest.approx(0.0)
    assert comparison.distinguishable is False
    assert NOT_DISTINGUISHABLE_PHRASE in comparison.sentence
    # The n is in the sentence too: a difference without its sample size is not
    # a result.
    assert "n = 3 paired queries" in comparison.sentence
    # Three pairs is also below the claim floor, so the sentence says that as
    # well — see test_too_few_paired_queries_can_never_be_called_distinguishable.
    assert "below the 6" in comparison.sentence


def test_an_interval_spanning_zero_over_enough_queries_says_it_may_be_noise(eval_config):
    # Eight queries, each with one highly relevant place. B moves the relevant
    # place to rank 1 on w0 and w1 (+0.3690702464285428 each) and drops it to
    # rank 2 on w2 (-0.3690702464285428); the other five are identical.
    #   mean difference = 0.3690702464285428 / 8 = 0.046133780803567846
    # Hand argument that the interval must include zero: a resample draws 8
    # pairs with replacement, and the single all-w2-and-no-w0/w1 family of
    # resamples (exactly one w2 draw, seven neutral draws) alone has probability
    # 8 * (1/8) * (5/8)**7 = (5/8)**7 = 0.0373 > 0.025, and every one of those
    # resamples has a negative mean difference. So more than 2.5% of the
    # resample distribution is below zero and the 2.5th percentile is negative.
    rows = [(f"w{i}", f"wide query {i}", "vibe") for i in range(8)]
    write_gold(Path(eval_config.evaluation.gold_queries), rows)
    write_qrels(Path(eval_config.evaluation.qrels), {f"w{i}": {f"b{i}": 2} for i in range(8)})
    worse = {f"wide query {i}": [f"b{i}"] for i in range(8)}
    worse["wide query 0"] = ["bX", "b0"]
    worse["wide query 1"] = ["bX", "b1"]
    better = {f"wide query {i}": [f"b{i}"] for i in range(8)}
    better["wide query 2"] = ["bX", "b2"]

    results = run_two_systems(eval_config, runs_a=worse, runs_b=better, min_labeled_queries=8)
    comparison = next(r for r in results.rows if r.key == "b").vs_previous
    assert comparison.n_queries == 8
    assert comparison.mean_diff == pytest.approx((1.0 - NDCG_A_Q2) / 8)
    assert comparison.low < 0 < comparison.high
    assert comparison.distinguishable is False
    assert NOT_DISTINGUISHABLE_PHRASE in comparison.sentence
    assert "includes zero" in comparison.sentence
    assert "noise" in comparison.sentence
    assert "must not be reported as an improvement" in comparison.sentence


def eight_query_world(cfg):
    """Eight queries of one type, each with a single highly relevant place.

    Six paired queries is the floor below which no difference may be claimed
    (``MIN_PAIRS_FOR_A_CLAIM``), so a test about *claiming* a difference needs
    a world above it.

    System A puts one unjudged result in front of the relevant place on every
    query; system B returns the relevant place first:

        A: [bX, bi] -> DCG = 0 + 3/log2(3); ideal = 3 -> NDCG = 0.6309297535714572
        B: [bi]     -> NDCG = 1.0

    so all eight paired differences equal 1 - 1/log2(3) = 0.3690702464285428.
    """
    rows = [(f"w{i}", f"wide query {i}", "vibe") for i in range(8)]
    write_gold(Path(cfg.evaluation.gold_queries), rows)
    write_qrels(Path(cfg.evaluation.qrels), {f"w{i}": {f"b{i}": 2} for i in range(8)})
    worse = {f"wide query {i}": ["bX", f"b{i}"] for i in range(8)}
    better = {f"wide query {i}": [f"b{i}"] for i in range(8)}
    return run_two_systems(cfg, runs_a=worse, runs_b=better, min_labeled_queries=8)


def test_a_uniform_improvement_over_enough_queries_is_called_distinguishable(eval_config):
    # Every paired difference is strictly positive, so every bootstrap resample
    # mean is strictly positive, so the 2.5th percentile cannot be <= 0 — no
    # resample count can make this interval touch zero. With 8 pairs that is
    # above the claim floor, so it may be stated.
    results = eight_query_world(eval_config)
    comparison = next(r for r in results.rows if r.key == "b").vs_previous
    assert comparison is not None and comparison.computed
    assert comparison.n_queries == 8
    assert comparison.mean_diff == pytest.approx(1.0 - NDCG_A_Q2)
    assert comparison.low > 0
    assert comparison.distinguishable is True
    assert NOT_DISTINGUISHABLE_PHRASE not in comparison.sentence
    assert "distinguishable" in comparison.sentence
    assert "excludes zero" in comparison.sentence
    # Every difference is identical here, so the interval has zero width. That
    # has to be said rather than read as precision.
    assert comparison.low == pytest.approx(comparison.high)
    assert "zero width" in comparison.sentence


def test_too_few_paired_queries_can_never_be_called_distinguishable(eval_config):
    # The per-type breakdown is where this bites: 'constraint' holds exactly one
    # query (q2), B strictly beats A on it, and a bootstrap over one observation
    # returns a zero-width interval that excludes zero. That is not evidence —
    # a clean sweep of five queries is still only two-sided p = 1/16 = 0.0625
    # by a sign test — so the claim is refused at any interval width.
    results = run_two_systems(eval_config)
    row_b = next(r for r in results.rows if r.key == "b")
    constraint = next(bt for bt in row_b.by_type if bt.query_type == "constraint")
    comparison = constraint.vs_previous
    assert comparison is not None and comparison.computed
    assert comparison.n_queries == 1
    assert comparison.low > 0  # the raw interval does exclude zero ...
    assert comparison.distinguishable is False  # ... and is refused anyway
    assert NOT_DISTINGUISHABLE_PHRASE in comparison.sentence
    assert "only 1 paired query" in comparison.sentence
    assert "Label more queries" in comparison.sentence


def test_a_bottomed_out_p_value_is_reported_as_a_bound_not_as_zero(eval_config):
    # With every paired difference positive, no resample lands on the wrong
    # side of zero, so the estimator bottoms out at 2/(n+1) and must be printed
    # as a bound: n resamples cannot distinguish "very unlikely" from
    # "impossible", and a literal p = 0 would be a claim the method cannot make.
    results = eight_query_world(eval_config)
    comparison = next(r for r in results.rows if r.key == "b").vs_previous
    # 500 resamples -> floor = 2/501 = 0.003992...
    assert comparison.p_value == pytest.approx(2 / 501)
    assert "bootstrap resolution limit" in comparison.sentence
    assert "p = 0.0000" not in comparison.sentence


def test_a_single_query_interval_is_labelled_degenerate_not_narrow(eval_config):
    # Only q1 keeps a relevant label, so every metric is a mean of one
    # observation and every bootstrap resample is identical. A zero-width
    # interval must not read as precision.
    one_relevant = {"q1": {"b1": 2, "b2": 1}, "q4": {"b6": 0}}
    write_qrels(Path(eval_config.evaluation.qrels), one_relevant)
    results = run_two_systems(eval_config, min_labeled_queries=1)
    row_a = next(r for r in results.rows if r.key == "a")
    assert row_a.ndcg.n_defined == 1
    assert row_a.ndcg.low == row_a.ndcg.high == pytest.approx(1.0)
    assert "degenerate" in row_a.ndcg.note


# ─────────────────────────────────────────────────────────────────────────────
# Refusing to invent numbers
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_qrels_refuses_to_report(eval_config, tmp_path):
    cfg = eval_config.model_copy(
        update={
            "evaluation": eval_config.evaluation.model_copy(
                update={"qrels": tmp_path / "does_not_exist.csv"}
            )
        }
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(cfg)
    message = str(excinfo.value)
    assert "does_not_exist.csv" in message
    assert "make label" in message
    # The message must point at labeling, and must promise no numbers.
    assert "invent" in message or "will not" in message


def test_missing_gold_queries_refuses_and_does_not_write_them_for_the_owner(eval_config, tmp_path):
    cfg = eval_config.model_copy(
        update={
            "evaluation": eval_config.evaluation.model_copy(
                update={"gold_queries": tmp_path / "no_gold.csv"}
            )
        }
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(cfg)
    assert "no_gold.csv" in str(excinfo.value)
    assert not (tmp_path / "no_gold.csv").exists()


def test_shipped_example_rows_reach_no_denominator_and_the_two_readers_agree(eval_config):
    # `eval/gold_queries.csv` ships with EXAMPLE* rows so the owner can see the
    # format, and `eval.pooling.load_gold_queries` — the reader the *labeling*
    # side uses — skips them. This reader did not, so the pool that got labeled
    # and the query set that got scored disagreed about which rows exist.
    #
    # The consequence was not a crash but a quiet overstatement: an example row
    # counted in n_gold_queries and in the per-type denominators the report
    # prints, yet could never be labeled, so it was silently dropped from every
    # mean. "NDCG@10 over n of N gold queries" then overstated N — unevenly,
    # since the shipped examples are not one per type.
    rows = [*GOLD_ROWS, ("EXAMPLE_1", "a worked example", "constraint")]
    write_gold(Path(eval_config.evaluation.gold_queries), rows)
    results = run_two_systems(eval_config)

    # Not in the headline denominator...
    assert results.coverage.n_gold_queries == len(GOLD_ROWS)
    # ...nor in the per-type counts (the example above is a 'constraint' row,
    # and the synthetic world has exactly one real constraint query).
    assert results.query_type_counts["constraint"] == 1
    # ...nor scored, nor listed as an unlabeled gold query.
    assert "EXAMPLE_1" not in results.coverage.unlabeled_query_ids
    assert all(c.query_id != "EXAMPLE_1" for c in results.failures)

    # The count is carried and surfaced, not silently dropped: a reader who
    # cannot reconcile the file's row count with this run's N has no way to
    # tell a skipped example from a parsing bug.
    assert results.coverage.n_example_rows_skipped == 1
    assert any("EXAMPLE" in w and "skipped" in w for w in results.warnings)
    assert "EXAMPLE" in report.render_markdown(results)

    # And the two readers now agree by construction, because the prefix has one
    # definition. This is the assertion that would fail if someone re-spelled
    # it in either module.
    from eval import pooling

    assert pooling.EXAMPLE_ID_PREFIX == "EXAMPLE"
    pooled = pooling.load_gold_queries(Path(eval_config.evaluation.gold_queries))
    assert pooled.n_examples_skipped == results.coverage.n_example_rows_skipped
    assert [q.query_id for q in pooled.queries] == [qid for qid, _, _ in GOLD_ROWS]


def test_an_example_row_is_skipped_before_it_can_fail_validation(eval_config):
    # Both readers skip the example rows *before* the query_type check, so a
    # shipped example can never block a run over a field nobody will score.
    # (pooling does the same, at the same point, for the same reason.)
    rows = [*GOLD_ROWS, ("EXAMPLE_bad", "a worked example", "not-a-real-type")]
    write_gold(Path(eval_config.evaluation.gold_queries), rows)
    results = run_two_systems(eval_config)
    assert results.coverage.n_gold_queries == len(GOLD_ROWS)
    assert results.coverage.n_example_rows_skipped == 1


def test_a_gold_file_of_nothing_but_examples_says_so(eval_config, tmp_path):
    # The state the repo actually ships in: five EXAMPLE rows and nothing else.
    # "has no query rows" would be baffling over a file the owner can see has
    # rows in it, so the message has to name the cause.
    path = write_gold(
        tmp_path / "examples_only.csv",
        [("EXAMPLE_1", "a worked example", "vibe")],
    )
    cfg = eval_config.model_copy(
        update={"evaluation": eval_config.evaluation.model_copy(update={"gold_queries": path})}
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(cfg)
    message = str(excinfo.value)
    assert "EXAMPLE" in message
    assert "only the 1 shipped" in message
    # And it points at who owns the work, rather than offering to do it.
    assert "the owner writes all of them" in message


def test_nearly_empty_labels_refuse_to_report(eval_config):
    # Two judged queries, which is below the default floor. The message has to
    # say what to do, and the run must stop before any search happens.
    write_qrels(Path(eval_config.evaluation.qrels), {"q1": {"b1": 2}, "q2": {"b3": 2}})
    factory = make_factory({})  # would raise if it were ever called
    with pytest.raises(LabelsNotReady) as excinfo:
        run_eval(eval_config, factory=factory, specs=TWO_SYSTEMS, run_sweep=False)
    message = str(excinfo.value)
    assert "2 of 5 gold queries" in message
    assert "make label" in message
    assert "simulated" in message or "illustrative" in message


def test_a_pool_graded_entirely_zero_does_not_count_as_a_labeled_query(eval_config):
    # 12 queries' worth of labels, all graded 0: plenty of rows in the CSV and
    # nothing that can be measured. The floor is on queries with a *relevant*
    # label for exactly this reason.
    rows = [(f"z{i}", f"query {i}", "vibe") for i in range(12)]
    write_gold(Path(eval_config.evaluation.gold_queries), rows)
    write_qrels(Path(eval_config.evaluation.qrels), {f"z{i}": {f"b{i}": 0} for i in range(12)})
    with pytest.raises(LabelsNotReady) as excinfo:
        run_eval(eval_config, factory=make_factory({}), specs=TWO_SYSTEMS, run_sweep=False)
    assert "0 of 12 gold queries" in str(excinfo.value)


def test_main_exits_with_a_distinct_code_and_writes_nothing_when_labels_are_missing(
    tmp_path, capsys
):
    # "Not ready" is a normal state of this project, not a crash, so it gets
    # its own exit code — and nothing may be written.
    out_dir = tmp_path / "artifacts"
    gold = write_gold(tmp_path / "gold_queries.csv")
    config_path = tmp_path / "eval_only.yaml"
    config_path.write_text(
        "evaluation:\n"
        f"  gold_queries: {gold}\n"
        f"  qrels: {tmp_path / 'missing_qrels.csv'}\n"
        f"  out_dir: {out_dir}\n",
        encoding="utf-8",
    )
    exit_code = main(["--config", str(config_path), "--no-sweep"])
    assert exit_code == 2
    assert "make eval stopped" in capsys.readouterr().err
    assert not out_dir.exists()


def test_a_stage_that_cannot_run_is_a_stated_gap_not_a_zero(eval_config):
    # Only system A is buildable: B's model does not exist yet, which is the
    # normal state in the middle phases of the project.
    factory = make_factory({VARIANT_A: RUN_A})
    results = run_eval(
        eval_config, factory=factory, specs=TWO_SYSTEMS, run_sweep=False, min_labeled_queries=3
    )
    row_b = next(r for r in results.rows if r.key == "b")
    assert row_b.ran is False
    assert row_b.ndcg is None and row_b.recall is None and row_b.mrr is None
    assert row_b.latency is None
    assert row_b.unjudged_fraction is None
    assert "stage not available" in row_b.skip_reason
    assert any("did not run" in w for w in results.warnings)
    # And the row renders as a dash with the reason, never as a number.
    markdown = report.render_markdown(results)
    assert "— (not run:" in markdown
    assert "0.000" not in markdown.split("## 2.")[0].split("| B |")[-1].split("\n")[0]


def test_every_system_failing_stops_the_run(eval_config):
    with pytest.raises(NoSystemRan) as excinfo:
        run_eval(
            eval_config,
            factory=make_factory({}),
            specs=TWO_SYSTEMS,
            run_sweep=False,
            min_labeled_queries=3,
        )
    assert "nothing to report" in str(excinfo.value)


def test_a_system_that_raises_on_every_query_is_not_run_rather_than_scored_zero(eval_config):
    factory = make_factory({VARIANT_A: RUN_A}, raise_on=list(QUERY_TEXT.values()))
    with pytest.raises(NoSystemRan):
        run_eval(
            eval_config,
            factory=factory,
            specs=(SPEC_A,),
            run_sweep=False,
            min_labeled_queries=3,
        )


def test_a_query_that_raises_is_recorded_and_scored_as_an_empty_answer(eval_config):
    # One query of five explodes. The system did not answer it, so it scores as
    # if it returned nothing (0.0, which is defined) — and the error count is
    # reported rather than hidden.
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B}, raise_on=[QUERY_TEXT["q1"]])
    results = run_eval(
        eval_config, factory=factory, specs=TWO_SYSTEMS, run_sweep=False, min_labeled_queries=3
    )
    row_a = next(r for r in results.rows if r.key == "a")
    assert row_a.ran is True
    assert row_a.n_query_errors == 1
    # q1's NDCG becomes 0.0; q2 and q3 are unchanged.
    assert row_a.ndcg.mean == pytest.approx((0.0 + NDCG_A_Q2 + 1.0) / 3)
    assert any("failed on 1 of 5 queries" in w for w in results.warnings)
    assert "raised an exception" in report.render_markdown(results)


def test_a_partial_run_is_stamped_so_it_cannot_be_quoted(eval_config):
    results = run_two_systems(eval_config, max_queries=3, min_labeled_queries=1)
    assert "PARTIAL RUN" in results.partial_run_note
    assert "must not be quoted" in results.partial_run_note
    assert "PARTIAL RUN" in report.render_markdown(results)


# ─────────────────────────────────────────────────────────────────────────────
# Per-query-type breakdown
# ─────────────────────────────────────────────────────────────────────────────


def test_breakdown_covers_every_query_type_with_its_own_counts(eval_config):
    results = run_two_systems(eval_config)
    row_a = next(r for r in results.rows if r.key == "a")
    by_type = {bt.query_type: bt for bt in row_a.by_type}
    assert set(by_type) == {t.value for t in QueryType}
    # Two vibe queries (q1, q5), one each of the others.
    assert by_type["vibe"].n_queries == 2
    assert by_type["constraint"].n_queries == 1
    assert by_type["specific"].n_queries == 1
    assert by_type["messy"].n_queries == 1
    assert results.query_type_counts == {
        "vibe": 2,
        "constraint": 1,
        "specific": 1,
        "messy": 1,
    }


def test_breakdown_metrics_are_restricted_to_the_right_queries(eval_config):
    results = run_two_systems(eval_config)
    by_type = {bt.query_type: bt for bt in next(r for r in results.rows if r.key == "a").by_type}
    # vibe = {q1 (NDCG 1.0), q5 (undefined)} -> mean 1.0 over 1 of 2 queries.
    assert by_type["vibe"].ndcg.mean == pytest.approx(1.0)
    assert (by_type["vibe"].ndcg.n_defined, by_type["vibe"].ndcg.n_undefined) == (1, 2 - 1)
    # constraint = {q2} -> the hand-computed 0.6309297535714572.
    assert by_type["constraint"].ndcg.mean == pytest.approx(NDCG_A_Q2)
    assert by_type["constraint"].ndcg.mean == pytest.approx(0.6309297535714572)
    # messy = {q4}, whose pool is graded entirely 0 -> no defined NDCG at all.
    assert by_type["messy"].ndcg.mean is None
    assert "undefined" in by_type["messy"].ndcg.note


def test_breakdown_undefined_slice_renders_without_a_number(eval_config):
    results = run_two_systems(eval_config)
    markdown = report.render_by_type(results)
    assert "### messy (1 gold query)" in markdown
    assert "— (undefined" in markdown


def test_each_query_type_carries_its_own_label_counts(eval_config):
    # Hand-counted from SYNTHETIC_QRELS against GOLD_ROWS:
    #   vibe       = q1 {b1:2, b2:1} + q5 {} -> 2 pairs, 2 relevant
    #   constraint = q2 {b3:2}               -> 1 pair,  1 relevant
    #   specific   = q3 {b4:2, b5:1}         -> 2 pairs, 2 relevant
    #   messy      = q4 {b6:0}               -> 1 pair,  0 relevant
    cov = run_two_systems(eval_config).coverage
    assert cov.labeled_pairs_by_type == {"vibe": 2, "constraint": 1, "specific": 2, "messy": 1}
    assert cov.positive_pairs_by_type == {"vibe": 2, "constraint": 1, "specific": 2, "messy": 0}
    # And they sum to the overall counts, so the breakdown cannot drift from
    # the headline denominators.
    assert sum(cov.labeled_pairs_by_type.values()) == cov.n_labeled_pairs
    assert sum(cov.positive_pairs_by_type.values()) == cov.n_positive_pairs


def test_every_table_carries_its_denominators_on_its_own(eval_config):
    # Each table is pasted into the README separately, so each one has to state
    # what it was computed over; the document header alone is not enough.
    results = run_two_systems(eval_config)
    for section in (
        report.render_ablation(results),
        report.render_by_type(results),
        report.render_failures(results),
    ):
        assert "n = " in section
        assert "labeled pair" in section
    # The per-type note carries that type's own counts, not the global ones.
    by_type = report.render_by_type(results)
    assert "n = 1 messy query in scope, 1 labeled pair (0 graded relevant)" in by_type
    assert "n = 2 vibe queries in scope, 2 labeled pairs (2 graded relevant)" in by_type


def test_a_partial_run_stamp_rides_along_with_every_table(eval_config):
    # A smoke-test table lifted out of the document must still say it is not a
    # result, so the stamp is repeated under each table rather than only at the
    # top of the report.
    results = run_two_systems(eval_config, max_queries=4, min_labeled_queries=2)
    assert results.partial_run_note
    assert report.render_ablation(results).count("PARTIAL RUN") == 1
    assert "PARTIAL RUN" in report.render_by_type(results)


# ─────────────────────────────────────────────────────────────────────────────
# Latency
# ─────────────────────────────────────────────────────────────────────────────


def test_percentile_nearest_rank_is_an_observed_value():
    # Ten samples, 1..10. p95: ceil(0.95 * 10) = 10 -> the 10th smallest = 10.
    assert percentile_nearest_rank([float(v) for v in range(1, 11)], 0.95) == 10.0
    # p50: ceil(0.5 * 10) = 5 -> the 5th smallest = 5 (lower median, no
    # interpolation: a latency percentile must be a value that was observed).
    assert percentile_nearest_rank([float(v) for v in range(1, 11)], 0.50) == 5.0
    # 20 samples 1..20: p95 -> ceil(19) = 19 -> the 19th smallest = 19.
    assert percentile_nearest_rank([float(v) for v in range(1, 21)], 0.95) == 19.0
    assert percentile_nearest_rank([], 0.95) is None
    assert percentile_nearest_rank([7.0], 0.95) == 7.0


def test_percentile_rejects_a_quantile_outside_the_unit_interval():
    with pytest.raises(ValueError):
        percentile_nearest_rank([1.0], 0.0)
    with pytest.raises(ValueError):
        percentile_nearest_rank([1.0], 1.5)


def test_the_cold_start_is_separated_from_the_steady_state_samples(eval_config):
    results = run_two_systems(eval_config)
    row_a = next(r for r in results.rows if r.key == "a")
    assert row_a.latency is not None
    # Five queries, the first excluded as the cold start.
    assert row_a.latency.n_samples == 4
    assert row_a.latency.cold_start_ms is not None
    # Four samples is nowhere near enough for a 95th percentile, and the result
    # has to say so rather than quietly print the maximum as "p95".
    assert "not a stable 95th percentile" in row_a.latency.caveat
    assert "Latency caveats" in report.render_markdown(results)


# ─────────────────────────────────────────────────────────────────────────────
# Quality-vs-latency sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_runs_the_valid_grid_at_the_production_final_depth(eval_config):
    cfg = eval_config.model_copy(
        update={
            "candidates": eval_config.candidates.model_copy(
                update={"sweep_retrieve": (20, 40), "sweep_rerank": (10, 25)}
            )
        }
    )
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B})
    specs = (SPEC_A, SPEC_B)
    results = run_eval(
        cfg,
        factory=factory,
        specs=specs,
        run_sweep=True,
        min_labeled_queries=3,
    )
    # rerank must not exceed retrieve, and final (10) must not exceed rerank:
    # (20,10), (20,25)x, (40,10), (40,25) -> three valid points.
    assert [(p.retrieve_k, p.rerank_k) for p in results.sweep] == [(20, 10), (40, 10), (40, 25)]
    assert all(p.final_k == 10 for p in results.sweep)
    # The sweep drives the last rung of the ladder that the spec list defines.
    assert all(p.ran for p in results.sweep)
    assert all(p.ndcg is not None and p.ndcg.mean == pytest.approx(1.0) for p in results.sweep)
    assert "production final depth" in results.sweep_note


def test_sweep_can_be_skipped_and_says_so(eval_config):
    results = run_two_systems(eval_config)
    assert results.sweep == ()
    assert "skipped" in results.sweep_note
    assert "No sweep was run." in report.render_sweep(results, "")


# ─────────────────────────────────────────────────────────────────────────────
# Where the serving operating point is allowed to be chosen
#
# The candidate depths are a tuned parameter. Choosing them on the gold set and
# then reporting the headline NDCG@10 at the chosen depths would be test-set
# tuning, which is the one methodology error this project cannot survive. These
# tests pin the split: depths are chosen on the synthetic ltr_val split, and the
# gold curve is a measurement of an already-chosen point that says so.
# ─────────────────────────────────────────────────────────────────────────────

MEASUREMENT_MARKER = "MEASUREMENT, NOT SELECTION"


def _small_sweep_grid(cfg):
    """Two retrieve widths, one rerank width — enough for a curve, fast."""
    return cfg.model_copy(
        update={
            "candidates": cfg.candidates.model_copy(
                update={"sweep_retrieve": (20, 40), "sweep_rerank": (10,)}
            )
        }
    )


def _plant_synthetic_queries(cfg, root: Path, n: int = 40):
    """Write n synthetic queries and their verified split manifest under ``root``.

    The real writers and the real splitter are used rather than hand-built JSON,
    so this exercises the same files ``make synth`` would produce. The split is
    hash-based and therefore deterministic for these ids and this seed: which
    queries land in ``ltr_val`` is fixed, not sampled.
    """
    from concierge.synth import splits as synth_splits
    from concierge.synth.generate import GenerationStore, SyntheticQuery

    queries = [
        SyntheticQuery(
            query_id=f"s{i:03d}",
            query=f"synthetic query number {i}",
            query_type=QueryType.VIBE,
            business_id=f"sb{i:03d}",
        )
        for i in range(n)
    ]
    GenerationStore.from_config(cfg, root=root).queries.append(q.to_json() for q in queries)
    manifest, _ = synth_splits.build_and_save(queries, cfg, root=root)
    return queries, manifest


def test_the_depth_selection_set_is_the_synthetic_split_and_carries_no_gold_label(
    eval_config, tmp_path
):
    root = tmp_path / "synth_root"
    queries, manifest = _plant_synthetic_queries(eval_config, root)
    val_ids = set(manifest.ids(VALIDATION_SPLIT))
    assert val_ids, "the planted split must put at least one query in the validation split"

    query_set = validation_sweep_query_set(eval_config, root)

    assert not query_set.is_gold
    assert {q.query_id for q in query_set.queries} == val_ids
    # The whole point: no gold query and no owner-graded pair can reach a curve
    # that the served depths are chosen from.
    assert set(query_set.qrels).isdisjoint(SYNTHETIC_QRELS)
    assert set(query_set.qrels).isdisjoint(QUERY_TEXT)
    # Exactly one positive per query — the business the query was generated
    # from — graded highly relevant because it is a positive by construction.
    business_of = {q.query_id: q.business_id for q in queries}
    assert query_set.qrels == {qid: {business_of[qid]: RELEVANCE_HIGH} for qid in sorted(val_ids)}
    assert VALIDATION_SPLIT in query_set.provenance


def test_the_validation_sweep_measures_the_grid_on_the_synthetic_split(eval_config, tmp_path):
    cfg = _small_sweep_grid(eval_config)
    root = tmp_path / "synth_root"
    queries, manifest = _plant_synthetic_queries(cfg, root)
    # Every synthetic query's own positive returned at rank 1, so each point's
    # NDCG@10 is exactly 1.0: DCG = (2**2 - 1)/log2(1 + 1) = 3, and the ideal
    # ranking of a single grade-2 positive has the same DCG.
    synthetic_runs = {q.query: [q.business_id] for q in queries}
    factory = make_factory({VARIANT_A: RUN_A | synthetic_runs, VARIANT_B: RUN_B | synthetic_runs})
    results = run_eval(
        cfg,
        factory=factory,
        specs=TWO_SYSTEMS,
        run_sweep=True,
        sweep_on=SWEEP_ON_VALIDATION,
        synth_root=root,
        min_labeled_queries=3,
    )
    assert [(p.retrieve_k, p.rerank_k) for p in results.sweep] == [(20, 10), (40, 10)]
    assert all(p.ran and p.ndcg is not None for p in results.sweep)
    assert all(p.ndcg.mean == pytest.approx(1.0) for p in results.sweep)
    assert all(p.ndcg.n_defined == len(manifest.ids(VALIDATION_SPLIT)) for p in results.sweep)
    # A selection curve must not be stamped with the gold-set warning, and must
    # name the split it was measured on so the two curves can never be swapped.
    assert VALIDATION_SPLIT in results.sweep_note
    assert MEASUREMENT_MARKER not in results.sweep_note
    assert GOLD_SWEEP_WARNING not in results.warnings


def test_choosing_depths_refuses_to_fall_back_to_gold_when_the_split_is_absent(
    eval_config, tmp_path
):
    # The dangerous failure mode is a silent fallback: a run that was asked for
    # the selection curve and quietly answered with the test set.
    cfg = _small_sweep_grid(eval_config)
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B})
    with pytest.raises(SweepSplitNotReady):
        run_eval(
            cfg,
            factory=factory,
            specs=TWO_SYSTEMS,
            run_sweep=True,
            sweep_on=SWEEP_ON_VALIDATION,
            synth_root=tmp_path / "nothing_here",
            min_labeled_queries=3,
        )


def test_the_gold_curve_is_published_as_a_measurement_not_a_selection(eval_config, capsys):
    cfg = _small_sweep_grid(eval_config)
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B})
    results = run_eval(
        cfg,
        factory=factory,
        specs=TWO_SYSTEMS,
        run_sweep=True,
        min_labeled_queries=3,
    )
    # The gold curve is still produced — it is a required deliverable.
    assert [(p.retrieve_k, p.rerank_k) for p in results.sweep] == [(20, 10), (40, 10)]
    # And it is unambiguously labeled, in all three places a reader might look:
    # the caption the report renders, the warnings list, and stderr.
    assert MEASUREMENT_MARKER in results.sweep_note
    assert GOLD_SWEEP_WARNING in results.warnings
    assert MEASUREMENT_MARKER in report.render_sweep(results, "")
    assert MEASUREMENT_MARKER in capsys.readouterr().err
    assert f"--sweep-on {SWEEP_ON_VALIDATION}" in results.sweep_note


def test_an_unknown_sweep_set_is_rejected_rather_than_defaulted(eval_config):
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B})
    with pytest.raises(EvalError):
        run_eval(
            eval_config,
            factory=factory,
            specs=TWO_SYSTEMS,
            run_sweep=True,
            sweep_on="whatever_looks_best",
            min_labeled_queries=3,
        )
    # The default is the gold *measurement*, because `make eval` must keep
    # producing the required deliverable with no extra flag.
    assert SWEEP_ON_GOLD == "gold"


# ─────────────────────────────────────────────────────────────────────────────
# Failure analysis
# ─────────────────────────────────────────────────────────────────────────────


def test_absence_is_not_called_never_retrieved_when_retrieval_ran_wider(eval_config):
    # q3's only relevant place is b99, which no system returns.
    #
    # Under the shipped config this absence CANNOT be attributed to retrieval.
    # `depths_for_eval` gives retrieve=200 / final=100, so `run.docs` holds only
    # the top 100 of a 200-candidate pool: b99 may have been retrieved at rank
    # 150 and left below the cut. Both cases are unfixable by reranking as
    # configured, but only one is a Stage 2 bug, and they have different fixes.
    # The whole point of this deliverable is that distinction, so the wording
    # has to stop where the evidence does.
    #
    # Previously this returned lost_stage='retrieval' and the words "Never
    # retrieved" unconditionally, which was a confident claim the data could
    # not support — on *every* run of the shipped config, since retrieve is
    # always wider than final there.
    assert eval_config.candidates.retrieve > depths_for_eval(eval_config).final
    qrels = dict(SYNTHETIC_QRELS) | {"q3": {"b99": 2}}
    write_qrels(Path(eval_config.evaluation.qrels), qrels)
    results = run_two_systems(eval_config)
    case = next(c for c in results.failures if c.query_id == "q3")
    assert case.ndcg == pytest.approx(0.0)
    assert case.lost_stage == "retrieval_or_shortlist"
    assert "Never retrieved" not in case.diagnosis
    # The useful half of the finding still gets said plainly.
    assert "Reranking cannot fix this query" in case.diagnosis
    # And the ambiguity is named, with both alternatives and a way to settle it.
    assert "never retrieved" in case.diagnosis
    assert "never promoted" in case.diagnosis
    assert str(eval_config.candidates.retrieve) in case.diagnosis
    # The trace shows the absence explicitly in every system.
    trace = next(t for t in case.relevant_traces if t.business_id == "b99")
    assert trace.rank_by_system == {"a": None, "b": None}


def test_a_crashed_query_is_diagnosed_as_an_error_not_as_a_recall_failure(eval_config):
    # The trap this guard exists for. `execute_system` records a per-query
    # exception and keeps going, leaving an EMPTY result list. An empty list
    # scores NDCG 0.0 — defined, not None — so the query sorts to the very
    # bottom of its type and is a *guaranteed* pick for the failure analysis.
    # Arriving there with no trace of the good result in any list, the
    # retrieval branch used to write a Python traceback up as a Stage 2 recall
    # failure, naming a stage and offering a confident diagnosis. A bug in our
    # code is not a finding about the ranking system, and the failure analysis
    # is the deliverable a reader trusts most.
    #
    # q3 is the only 'specific' query, so whatever happens to it is what that
    # type's failure row says.
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B}, raise_on=[QUERY_TEXT["q3"]])
    results = run_eval(
        eval_config, factory=factory, specs=TWO_SYSTEMS, run_sweep=False, min_labeled_queries=3
    )
    case = next(c for c in results.failures if c.query_id == "q3")
    assert case.ndcg == pytest.approx(0.0)
    assert case.lost_stage == "error"
    # The recorded exception text, so the owner can fix it rather than guess.
    assert "fake stage exploded" in case.diagnosis
    assert "RuntimeError" in case.diagnosis
    # And none of the ranking vocabulary, which would be a false diagnosis.
    for phrase in (
        "Never retrieved",
        "never retrieved",
        "mis-ranked",
        "never ranked up",
        "Mis-ordered",
        "retrieval or hard-filter failure",
    ):
        assert phrase not in case.diagnosis
    assert case.lost_stage not in {"retrieval", "retrieval_or_shortlist", "ranking", "ordering"}
    # The 0.0 is explicitly disowned as a measurement of ranking quality.
    assert "measures the exception, not the ranking" in case.diagnosis
    # The row-level count is still reported, so the deflated mean beside this
    # is not read as a clean measurement either.
    assert next(r for r in results.rows if r.key == "b").n_query_errors == 1
    assert "raised an exception" in report.render_markdown(results)


def test_queries_that_did_not_crash_are_still_diagnosed_normally(eval_config):
    # The short-circuit must be keyed to the query that actually raised, not
    # applied to the whole row: a system that failed one query out of five is
    # still a measurement on the other four, and their failures are real.
    swapped = dict(RUN_A) | {QUERY_TEXT["q1"]: ["b2", "b1"]}
    factory = make_factory({VARIANT_A: swapped, VARIANT_B: swapped}, raise_on=[QUERY_TEXT["q3"]])
    results = run_eval(
        eval_config, factory=factory, specs=TWO_SYSTEMS, run_sweep=False, min_labeled_queries=3
    )
    by_id = {c.query_id: c for c in results.failures}
    assert by_id["q3"].lost_stage == "error"
    assert by_id["q1"].lost_stage == "ordering"
    assert "Mis-ordered inside the top 10" in by_id["q1"].diagnosis


def test_absence_is_a_proven_retrieval_failure_once_the_scored_depth_covers_stage_2(
    eval_config,
):
    # The other side of the same branch. Narrow Stage 2 to below the scored
    # depth and the scored list now contains everything retrieval produced, so
    # "never retrieved" becomes a fact rather than a guess and the diagnosis is
    # allowed to say it. This is the configuration the owner can re-run to
    # settle the ambiguity the previous test pins down.
    cfg = eval_config.model_copy(
        update={"candidates": eval_config.candidates.model_copy(update={"retrieve": 5})}
    )
    depths = depths_for_eval(cfg)
    assert depths.retrieve <= depths.final
    qrels = dict(SYNTHETIC_QRELS) | {"q3": {"b99": 2}}
    write_qrels(Path(cfg.evaluation.qrels), qrels)
    results = run_two_systems(cfg)
    case = next(c for c in results.failures if c.query_id == "q3")
    assert case.lost_stage == "retrieval"
    assert "Never retrieved" in case.diagnosis
    assert "no amount of reranking" in case.diagnosis.lower()
    assert "hard-filter failure" in case.diagnosis


def test_a_result_pushed_out_of_the_top_k_names_the_stage_that_lost_it(eval_config):
    # b4 is relevant and rank 1 under A; under B (the reference system, being
    # the last rung) it has fallen to rank 11, outside the top 10.
    padding = [f"pad{i}" for i in range(10)]
    runs_a = dict(RUN_A) | {QUERY_TEXT["q3"]: ["b4", "b5"]}
    runs_b = dict(RUN_B) | {QUERY_TEXT["q3"]: [*padding, "b4", "b5"]}
    results = run_two_systems(eval_config, runs_a=runs_a, runs_b=runs_b)
    case = next(c for c in results.failures if c.query_id == "q3")
    assert case.system_key == "b"
    assert case.lost_stage == "b"
    assert "Retrieved and then mis-ranked" in case.diagnosis
    assert "rank 1 under 'a'" in case.diagnosis
    assert "rank 11 under 'b'" in case.diagnosis
    trace = next(t for t in case.relevant_traces if t.business_id == "b4")
    assert trace.rank_by_system == {"a": 1, "b": 11}


def _diag_run(key: str, ranked: Sequence[str]) -> SystemRun:
    """A SystemRun carrying one query's result list, for diagnosing directly."""
    run = SystemRun(
        key=key,
        label=key,
        variant=VARIANT_A,
        depths=CandidateDepths(retrieve=10, rerank=10, final=10),
        ran=True,
    )
    run.docs["q"] = tuple(
        RetrievedDoc(business_id=bid, name=f"Place {bid}", score=0.0, constraints=())
        for bid in ranked
    )
    return run


def test_the_stage_that_lost_a_result_is_never_named_as_itself():
    # `diagnose_query` is public, and a caller may diagnose a system that is not
    # the last rung of the ladder it passes. The scan for "which stage lost it"
    # used to run over every run and fall back to `last_good` when the best
    # candidate was the final one, so with the reference anywhere but last the
    # sentence compared a run to itself: "was at rank 1 under 'c' and is rank 1
    # under 'c', which pushed it out of the top 3" — self-contradictory, and
    # attributing the loss to a stage that still had the result.
    #
    # Ladder a -> b -> c, diagnosing b. b has the relevant place below the top
    # 3; a had it at rank 1; c has it at rank 1 again. Only a and b can be
    # answers, because c runs after the system under test.
    padding = ["p1", "p2", "p3"]
    runs = [
        _diag_run("a", ["b1", *padding]),
        _diag_run("b", [*padding, "b1"]),
        _diag_run("c", ["b1", *padding]),
    ]
    lost_stage, diagnosis, _ = diagnose_query(
        "q", {"b1": 2}, runs, "b", ndcg_k=3, depth=10, ndcg=0.0, retrieve_depth=10
    )
    assert lost_stage == "b"
    assert "rank 1 under 'a'" in diagnosis
    assert "rank 4 under 'b'" in diagnosis
    # The stage after the one being diagnosed cannot be blamed, and cannot be
    # credited with having had it either.
    assert "'c'" not in diagnosis
    # And no self-comparison survives.
    assert "under 'b' and is rank 4 under 'b'" not in diagnosis


def test_a_result_in_the_pool_but_never_surfaced_is_a_ranking_failure(eval_config):
    # b4 sits at rank 11 under both systems: in the candidate pool, never in the
    # top 10. Nothing "lost" it — the ranking never found it.
    padding = [f"pad{i}" for i in range(10)]
    deep = {**RUN_A, QUERY_TEXT["q3"]: [*padding, "b4"]}
    results = run_two_systems(eval_config, runs_a=deep, runs_b=deep)
    case = next(c for c in results.failures if c.query_id == "q3")
    assert case.lost_stage == "ranking"
    assert "never ranked up" in case.diagnosis
    assert "not a retrieval failure" in case.diagnosis


def test_a_relevant_result_inside_the_top_k_is_diagnosed_as_mis_ordering(eval_config):
    # The grade-1 place is above the grade-2 place: both are retrieved and both
    # are shown, so the only thing wrong is their order.
    #   DCG   = 1/log2(2) + 3/log2(3) = 1 + 1.8927892607143717 = 2.8927892607143717
    #   ideal = 3 + 1/log2(3)         = 3.6309297535714574
    #   NDCG  = 0.7966...  -> below 1.0, so this really is a failure to diagnose
    swapped = dict(RUN_A) | {QUERY_TEXT["q1"]: ["b2", "b1"]}
    results = run_two_systems(eval_config, runs_a=swapped, runs_b=swapped)
    case = next(c for c in results.failures if c.query_id == "q1")
    assert case.ndcg == pytest.approx((1.0 + 3 / LOG2_3) / (3.0 + 1 / LOG2_3))
    assert case.lost_stage == "ordering"
    assert "Mis-ordered inside the top 10" in case.diagnosis


def test_a_perfectly_ranked_worst_query_is_not_given_an_invented_fault(eval_config):
    # q1 is the worst 'vibe' query under the reference system and still scores
    # a perfect 1.0 — the top 10 is the best ordering the labels allow. Calling
    # that a mis-ranking would be fabricating a failure.
    results = run_two_systems(eval_config)
    case = next(c for c in results.failures if c.query_id == "q1")
    assert case.ndcg == pytest.approx(1.0)
    assert case.lost_stage == "none"
    assert "Nothing to diagnose" in case.diagnosis
    assert "no failure to show" in case.diagnosis


def test_failure_cases_show_the_actual_results_with_their_grades(eval_config):
    results = run_two_systems(eval_config)
    case = next(c for c in results.failures if c.query_id == "q2")
    # (rank, business_id, name, grade, judged) for B's q2 list ["b3", "bX"].
    assert case.top_results == (
        (1, "b3", "Place b3", 2, True),
        (2, "bX", "Place bX", 0, False),
    )
    assert "semantic_text=" in case.parsed_summary
    assert case.parse_source == "rules"


def test_failure_analysis_skips_queries_with_no_defined_ndcg(eval_config):
    results = run_two_systems(eval_config)
    ids = {c.query_id for c in results.failures}
    # q4 (pool graded all 0) and q5 (unlabeled) have no NDCG to be worst at.
    assert "q4" not in ids and "q5" not in ids
    assert ids == {"q1", "q2", "q3"}


def test_failure_analysis_reports_at_most_the_configured_number_per_type(eval_config):
    rows = [(f"v{i}", f"vibe query {i}", "vibe") for i in range(6)]
    write_gold(Path(eval_config.evaluation.gold_queries), rows)
    write_qrels(Path(eval_config.evaluation.qrels), {f"v{i}": {f"b{i}": 2} for i in range(6)})
    # Every query returns nothing relevant, so all six are equally bad.
    runs = {f"vibe query {i}": ["bX"] for i in range(6)}
    results = run_two_systems(eval_config, runs_a=runs, runs_b=runs, min_labeled_queries=6)
    assert len(results.failures) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Loading the owner's files
# ─────────────────────────────────────────────────────────────────────────────


def test_an_unknown_query_type_is_rejected_with_its_line_number(eval_config):
    write_gold(
        Path(eval_config.evaluation.gold_queries),
        [("q1", "a query", "vibes"), ("q2", "another", "vibe")],
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(eval_config)
    assert ":2:" in str(excinfo.value)
    assert "'vibes'" in str(excinfo.value)


def test_duplicate_query_ids_are_rejected(eval_config):
    write_gold(
        Path(eval_config.evaluation.gold_queries),
        [("q1", "a query", "vibe"), ("q1", "another", "vibe")],
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(eval_config)
    assert "duplicate query_id" in str(excinfo.value)


def test_a_repeated_pair_takes_the_last_row_as_the_current_judgement(eval_config):
    # qrels.csv is append-only: the labeling CLI records a correction as a new
    # row rather than editing the old one, so the most recent row is the
    # owner's current opinion. Both readers must agree on that rule.
    path = Path(eval_config.evaluation.qrels)
    path.write_text(
        "query_id,business_id,grade\nq1,b1,2\nq1,b2,1\nq2,b3,2\nq3,b4,2\nq1,b1,0\n",
        encoding="utf-8",
    )
    results = run_two_systems(eval_config)
    row_a = next(r for r in results.rows if r.key == "a")
    # q1 is now [b1(0), b2(1)]: DCG = 0 + 1/log2(3); ideal = 1/log2(2) = 1.
    # NDCG(q1) = 1/log2(3) = 0.6309297535714574, and q2/q3 are unchanged at
    # 0.6309297535714572 and 1.0.
    expected = (1.0 / LOG2_3 + NDCG_A_Q2 + 1.0) / 3
    assert row_a.ndcg.mean == pytest.approx(expected)


def test_a_file_written_by_the_labeling_cli_is_read_by_its_own_module(eval_config):
    # The labeling CLI's header (query_id,business_id,relevance,labeled_at,
    # labeler) is handed to eval.pooling, so the format has exactly one parser.
    path = Path(eval_config.evaluation.qrels)
    rows = ["query_id,business_id,relevance,labeled_at,labeler"]
    for qid, judged in SYNTHETIC_QRELS.items():
        rows += [
            f"{qid},{bid},{grade},2026-10-15T12:00:00+00:00,owner" for bid, grade in judged.items()
        ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    results = run_two_systems(eval_config)
    assert results.coverage.n_labeled_pairs == 6
    row_a = next(r for r in results.rows if r.key == "a")
    assert row_a.ndcg.mean == pytest.approx(NDCG_A_MEAN)


def test_a_grade_outside_the_labeling_scheme_is_rejected(eval_config):
    # A stray 3 would change every DCG gain (2**3 - 1 = 7) with no visible
    # error, so it must stop the run.
    Path(eval_config.evaluation.qrels).write_text(
        "query_id,business_id,grade\nq1,b1,3\n", encoding="utf-8"
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(eval_config)
    assert "outside the labeling scheme" in str(excinfo.value)


def test_a_relevance_column_alias_is_accepted(eval_config):
    path = Path(eval_config.evaluation.qrels)
    rows = ["query_id,business_id,relevance"]
    for qid, judged in SYNTHETIC_QRELS.items():
        rows += [f"{qid},{bid},{grade}" for bid, grade in judged.items()]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    results = run_two_systems(eval_config)
    assert results.coverage.n_labeled_pairs == 6


def test_a_non_integer_grade_is_rejected_with_its_line_number(eval_config):
    Path(eval_config.evaluation.qrels).write_text(
        "query_id,business_id,grade\nq1,b1,very relevant\n", encoding="utf-8"
    )
    with pytest.raises(LabelsNotReady) as excinfo:
        run_two_systems(eval_config)
    assert ":2:" in str(excinfo.value)
    assert "not an integer" in str(excinfo.value)


def test_labels_for_queries_outside_the_gold_set_are_ignored_but_reported(eval_config):
    qrels = dict(SYNTHETIC_QRELS) | {"not_a_gold_query": {"b1": 2}}
    write_qrels(Path(eval_config.evaluation.qrels), qrels)
    results = run_two_systems(eval_config)
    assert results.coverage.orphan_query_ids == ("not_a_gold_query",)
    assert any("not in the gold set" in w for w in results.warnings)
    # The orphan must not inflate any denominator.
    assert results.coverage.n_labeled_pairs == 6


def test_the_headline_cutoff_comes_from_the_config_not_from_this_code(eval_config):
    assert ndcg_k_from_config(eval_config) == 10
    cfg = eval_config.model_copy(
        update={
            "evaluation": eval_config.evaluation.model_copy(
                update={"primary_metric": "ndcg@5", "cutoffs": (5, 10)}
            )
        }
    )
    assert ndcg_k_from_config(cfg) == 5
    results = run_two_systems(cfg)
    assert results.ndcg_k == 5
    assert "NDCG@5" in report.render_markdown(results)


def test_a_headline_cutoff_absent_from_the_cutoffs_list_is_an_error(eval_config):
    cfg = eval_config.model_copy(
        update={
            "evaluation": eval_config.evaluation.model_copy(update={"primary_metric": "ndcg@7"})
        }
    )
    with pytest.raises(Exception, match="cutoffs"):
        ndcg_k_from_config(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering and the files that get written
# ─────────────────────────────────────────────────────────────────────────────


def test_the_report_states_n_on_the_tables_and_keeps_the_hedge(eval_config):
    results = run_two_systems(eval_config)
    markdown = report.render_markdown(results)
    assert "n queries" in markdown  # the ablation table's denominator column
    assert "**5** gold queries" in markdown
    assert "**6** labeled (query, business) pairs" in markdown
    assert NOT_DISTINGUISHABLE_PHRASE in markdown
    # Each of the four required outputs has a section.
    for heading in ("## 1. Ablation", "## 2. Per query type", "## 3. Quality vs latency"):
        assert heading in markdown
    assert "## 4. Failure analysis" in markdown


def test_write_report_writes_markdown_json_and_the_curve_data(eval_config, tmp_path):
    results = run_two_systems(eval_config)
    paths = report.write_report(results, tmp_path / "out")
    assert paths.markdown.exists() and paths.json.exists() and paths.sweep_csv.exists()
    # Artifacts are namespaced by the config fingerprint so any figure can be
    # traced back to the settings that produced it (DECISIONS.md D008).
    assert paths.directory.name == results.config_fingerprint
    payload = json.loads(paths.json.read_text())
    assert payload["coverage"]["n_gold_queries"] == 5
    row_a = next(r for r in payload["rows"] if r["key"] == "a")
    assert row_a["ndcg"]["mean"] == pytest.approx(NDCG_A_MEAN)
    # Enums must be values, not reprs, so the README generator never has to
    # parse "QueryType.VIBE".
    assert {bt["query_type"] for bt in row_a["by_type"]} == {
        "vibe",
        "constraint",
        "specific",
        "messy",
    }


def test_a_missing_matplotlib_costs_the_figure_and_nothing_else(eval_config, tmp_path, monkeypatch):
    # The numbers are the deliverable; the plot is a convenience. Losing an
    # eval run to a plotting dependency would be absurd.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    cfg = eval_config.model_copy(
        update={
            "candidates": eval_config.candidates.model_copy(
                update={"sweep_retrieve": (20,), "sweep_rerank": (10,)}
            )
        }
    )
    factory = make_factory({VARIANT_A: RUN_A, VARIANT_B: RUN_B})
    results = run_eval(
        cfg, factory=factory, specs=TWO_SYSTEMS, run_sweep=True, min_labeled_queries=3
    )
    paths = report.write_report(results, tmp_path / "out")
    assert paths.plot is None
    text = paths.markdown.read_text()
    assert "matplotlib is not installed" in text
    assert "latency_curve.csv" in text
    # The curve's data survives regardless.
    rows = list(csv.DictReader(paths.sweep_csv.read_text().splitlines()))
    assert len(rows) == 1
    assert float(rows[0]["ndcg_mean"]) == pytest.approx(1.0)


def test_nothing_in_the_report_is_a_placeholder_number(eval_config):
    # A guard against the single worst outcome for this project: an example
    # table reaching the README. Every numeric cell must trace back to a
    # MetricCell with a mean, and an unrun row must carry none.
    factory = make_factory({VARIANT_A: RUN_A})
    results = run_eval(
        eval_config, factory=factory, specs=TWO_SYSTEMS, run_sweep=False, min_labeled_queries=3
    )
    for row in results.rows:
        if row.ran:
            assert row.ndcg is not None and row.ndcg.mean is not None
        else:
            assert (row.ndcg, row.recall, row.mrr, row.latency) == (None, None, None, None)
    markdown = report.render_markdown(results)
    for forbidden in ("example", "placeholder", "illustrative", "e.g. 0.", "TODO"):
        assert forbidden not in markdown.lower()
