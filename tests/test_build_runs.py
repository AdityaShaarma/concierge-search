"""The pool builder: what gets written, and what refuses to be written.

`eval/build_runs.py` sits between two things the owner cannot do without: a
built index and 25 hours of labeling. It therefore has one property that matters
more than any other — **a system's name in the runs file must mean that system
produced those results**. Pooling records which system surfaced each candidate,
so a mislabeled run does not merely misreport a row: it bakes the error into
hand-graded labels that then outlive every later correction.

Everything here tests either that property or the arithmetic around it (pool
depth, merging, the quota estimate). The dataset is never needed: the searcher
factory is injected, the same way `run_eval` injects it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from concierge.pipeline import ABLATION, VARIANTS, Stage, Variant
from concierge.types import GoldQuery, QueryType
from eval import build_runs
from eval.run_eval import CandidateDepths, SystemSpec, ablation_specs
from tests.test_run_eval import make_factory

BM25 = next(v for v in ABLATION if v.name == "bm25")
HYBRID = next(v for v in ABLATION if v.name == "hybrid_rrf")
FULL = next(v for v in ABLATION if v.name == "full")


def _gold(n: int = 3) -> list[GoldQuery]:
    return [
        GoldQuery(query_id=f"q{i}", query=f"query number {i}", query_type=QueryType.VIBE)
        for i in range(1, n + 1)
    ]


def _spec(variant: Variant) -> SystemSpec:
    return SystemSpec(key=variant.name, label=variant.name, variant=variant)


def _runs_for(gold: Sequence[GoldQuery], ids: Sequence[str]) -> Mapping[str, Sequence[str]]:
    return {gq.query: list(ids) for gq in gold}


# ─────────────────────────────────────────────────────────────────────────────
# A system's name means that system ran
# ─────────────────────────────────────────────────────────────────────────────


def test_an_unavailable_system_is_omitted_not_degraded(config) -> None:
    """The property this module exists to protect.

    `SearchPipeline` degrades gracefully when a model is missing — right for
    serving, wrong here. If `full` were written using the pretrained reranker
    and no LambdaMART, the owner would grade those results believing the full
    system produced them.
    """
    gold = _gold()
    factory = make_factory({BM25: _runs_for(gold, ["b1", "b2"])})

    runs, notes = build_runs.build(
        config,
        gold=gold,
        specs=[_spec(BM25), _spec(FULL)],
        depth=10,
        factory=factory,
    )

    assert set(runs) == {"bm25"}
    assert any("full" in note for note in notes)


def test_a_partial_pool_is_a_success_not_a_failure(config) -> None:
    """Phase 1 has exactly one system. Incremental pooling is the design."""
    gold = _gold()
    runs, _ = build_runs.build(
        config,
        gold=gold,
        specs=list(ablation_specs()),
        depth=10,
        factory=make_factory({BM25: _runs_for(gold, ["b1"])}),
    )
    assert runs == {"bm25": {"q1": ["b1"], "q2": ["b1"], "q3": ["b1"]}}


def test_the_skip_reason_is_recorded_for_every_omitted_system(config) -> None:
    """A silently short pool is the failure mode; each gap must be named."""
    gold = _gold()
    specs = list(ablation_specs())
    runs, notes = build_runs.build(
        config,
        gold=gold,
        specs=specs,
        depth=5,
        factory=make_factory({BM25: _runs_for(gold, ["b1"])}),
    )
    assert len(notes) == len(specs) - len(runs)


def test_a_system_returning_nothing_at_all_is_omitted(config) -> None:
    """An empty run would add a system name to the pool and no candidates.

    Worse than absent: the labeling tool would report a system as pooled, and
    its "unjudged in top 10" count would later be computed against results that
    were never offered for grading.
    """
    gold = _gold()
    runs, notes = build_runs.build(
        config,
        gold=gold,
        specs=[_spec(BM25)],
        depth=5,
        factory=make_factory({BM25: {}}),
    )
    assert runs == {}
    assert any("no results" in note or "returned nothing" in note for note in notes)


# ─────────────────────────────────────────────────────────────────────────────
# Depth
# ─────────────────────────────────────────────────────────────────────────────


def test_the_builder_does_not_truncate_results_itself(config) -> None:
    """Depth is applied by the pipeline, through `CandidateDepths`, and nowhere else.

    Asserted explicitly because a second truncation here would be invisible:
    the pool would simply be shorter than the systems it claims to pool, and
    the labeling queue would look merely cheaper. The fake searcher returns 30
    results regardless of depth, and all 30 survive — so the only cut in the
    real path is the one the config carries (see the test below).
    """
    gold = _gold(1)
    deep = [f"b{i}" for i in range(1, 31)]
    runs, _ = build_runs.build(
        config,
        gold=gold,
        specs=[_spec(BM25)],
        depth=20,
        factory=make_factory({BM25: _runs_for(gold, deep)}),
    )
    assert len(runs["bm25"]["q1"]) == 30


def test_retrieve_and_rerank_widths_come_from_the_config(config) -> None:
    """Pooling must happen at the depths the ablation will be measured at.

    A pool built at other widths grades a different system than the one
    reported — the results would be real, but they would not be the shipped
    cascade's results.
    """
    seen: list[CandidateDepths] = []

    def factory(cfg, variant, depths):
        seen.append(depths)
        return make_factory({BM25: {}})(cfg, variant, depths)

    build_runs.build(config, gold=_gold(1), specs=[_spec(BM25)], depth=7, factory=factory)
    assert seen[0].retrieve == config.candidates.retrieve
    assert seen[0].rerank == config.candidates.rerank
    assert seen[0].final == 7


def test_the_default_pool_depth_is_twenty() -> None:
    """CLAUDE.md §6 says top 20. It is a labeling-cost decision, so it is pinned."""
    assert build_runs.DEFAULT_POOL_DEPTH == 20


# ─────────────────────────────────────────────────────────────────────────────
# Merging — what makes incremental pooling work
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_keeps_systems_that_did_not_run_this_time() -> None:
    merged = build_runs.merge_runs(
        {"bm25": {"q1": ["b1"]}},
        {"hybrid_rrf": {"q1": ["b2"]}},
    )
    assert merged == {"bm25": {"q1": ["b1"]}, "hybrid_rrf": {"q1": ["b2"]}}


def test_merge_replaces_a_system_wholesale_rather_than_per_query() -> None:
    """Half-updated runs describe a system that never existed.

    The old run had two queries; the new one answers only `q1`. Keeping the
    stale `q2` would make the merged run a mixture of two checkouts, and any
    metric over it would describe neither.
    """
    merged = build_runs.merge_runs(
        {"bm25": {"q1": ["old"], "q2": ["stale"]}},
        {"bm25": {"q1": ["new"]}},
    )
    assert merged == {"bm25": {"q1": ["new"]}}


def test_merge_does_not_mutate_its_inputs() -> None:
    existing = {"bm25": {"q1": ["b1"]}}
    build_runs.merge_runs(existing, {"bm25": {"q1": ["b9"]}})
    assert existing == {"bm25": {"q1": ["b1"]}}


# ─────────────────────────────────────────────────────────────────────────────
# The quota gate
# ─────────────────────────────────────────────────────────────────────────────


def test_rule_parsing_systems_cost_no_api_calls(config) -> None:
    plan = build_runs.plan_quota([_spec(BM25), _spec(HYBRID)], _gold(150), config)
    assert plan.n_api_calls == 0
    assert "rule-based" in plan.note


def test_llm_systems_are_counted_once_per_query_not_once_per_system(config) -> None:
    """Two LLM variants share one parse cache, so they cost one set of calls.

    Counting per system would roughly double the stated cost. Overstating is
    not the safe direction: a confirmation prompt that is routinely wrong is a
    prompt the owner learns to wave through.
    """
    gold = _gold(10)
    llm_specs = [_spec(v) for v in ABLATION if v.enabled(Stage.QUERY_LLM)]
    assert len(llm_specs) >= 2, "the ablation should have more than one LLM-parsing row"

    plan = build_runs.plan_quota(llm_specs, gold, config)
    assert plan.n_api_calls == len(gold)


def test_the_plan_names_which_systems_will_spend_quota(config) -> None:
    plan = build_runs.plan_quota([_spec(v) for v in ABLATION], _gold(5), config)
    assert "llm_parse" in plan.note and "full" in plan.note


# ─────────────────────────────────────────────────────────────────────────────
# System selection
# ─────────────────────────────────────────────────────────────────────────────


def test_default_selection_is_every_ablation_row_and_diagnostic() -> None:
    keys = {s.key for s in build_runs.select_specs(None)}
    assert {v.name for v in ABLATION} <= keys
    assert "no_filters" in keys and "dense_only" in keys


def test_an_unknown_system_is_an_error_naming_what_exists() -> None:
    """A typo that silently pooled five systems instead of six would surface
    only much later, as an unexplained rise in unjudged results."""
    with pytest.raises(SystemExit, match="unknown system"):
        build_runs.select_specs(["bm25", "bm52"])


def test_selection_preserves_the_requested_order() -> None:
    specs = build_runs.select_specs(["hybrid_rrf", "bm25"])
    assert [s.key for s in specs] == ["hybrid_rrf", "bm25"]


def test_every_named_variant_is_selectable() -> None:
    """`VARIANTS` and the selectable set must not drift apart."""
    selectable = {s.key for s in build_runs.select_specs(None)}
    assert selectable == set(VARIANTS)


# ─────────────────────────────────────────────────────────────────────────────
# The written file
# ─────────────────────────────────────────────────────────────────────────────


def test_the_written_file_is_what_the_labeling_tool_reads(tmp_path: Path, config) -> None:
    """End to end against the labeling CLI's own reader, not a copy of it.

    This is the test that would have caught the gap: the two sides previously
    agreed on a format by documentation alone, because nothing wrote the file.
    """
    from eval.label_cli import load_runs

    gold = _gold(2)
    runs, _ = build_runs.build(
        config,
        gold=gold,
        specs=[_spec(BM25)],
        depth=20,
        factory=make_factory({BM25: _runs_for(gold, ["b1", "b2", "b3"])}),
    )
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loaded = load_runs(path)
    assert loaded == {"bm25": {"q1": ["b1", "b2", "b3"], "q2": ["b1", "b2", "b3"]}}


def test_a_query_that_errored_is_not_written_as_an_empty_list(config) -> None:
    """An empty list is a *judgement* — "this system answered nothing" — and
    the metrics read it that way. For a pool it is just noise to grade, so
    failed queries are dropped from the runs file rather than recorded as
    answered-with-nothing."""
    gold = _gold(2)
    factory = make_factory(
        {BM25: _runs_for(gold, ["b1"])},
        raise_on=[gold[0].query],
    )
    runs, _ = build_runs.build(config, gold=gold, specs=[_spec(BM25)], depth=10, factory=factory)
    assert runs["bm25"] == {"q2": ["b1"]}


# ─────────────────────────────────────────────────────────────────────────────
# Availability is settled before the quota prompt
# ─────────────────────────────────────────────────────────────────────────────


class _LoadedStub:
    """Stands in for a loaded `SearchPipeline`, with chosen pieces present.

    `PipelineFactory.missing_artifacts` reads four attributes and nothing else,
    so a stub is honest here and avoids needing an index on disk.
    """

    def __init__(self, *, bm25=None, dense=None, rerankers=None, ltr=None) -> None:
        self.bm25_index = bm25
        self.dense_index = dense
        self.rerankers = rerankers or {}
        self.ltr_ranker = ltr


def test_a_system_missing_its_index_is_reported_with_what_it_needs() -> None:
    runnable, unavailable = build_runs.partition_available(
        [_spec(BM25), _spec(HYBRID)],
        _LoadedStub(bm25=object()),
    )
    assert [s.key for s in runnable] == ["bm25"]
    assert unavailable == [("hybrid_rrf", "dense/FAISS index (run `make index`)")]


def test_availability_uses_the_same_check_as_the_ablation() -> None:
    """Not a reimplementation of "which stage needs which file".

    If the pool and `make eval` disagreed about availability, the pool could
    hold a system the ablation will not measure, or omit one it will — and the
    owner would only find out after grading.
    """
    from eval.run_eval import PipelineFactory

    loaded = _LoadedStub(bm25=object())
    _, unavailable = build_runs.partition_available([_spec(HYBRID)], loaded)
    assert unavailable[0][1] == "; ".join(
        PipelineFactory.missing_artifacts(loaded, HYBRID)  # type: ignore[arg-type]
    )


def test_the_quota_plan_covers_only_the_systems_that_will_run(config) -> None:
    """The prompt must quote the cost of the work that will actually happen.

    An LLM system that cannot be built contributes no API calls. Counting it
    would overstate the cost, and a confirmation that is routinely wrong is one
    the owner learns to wave through — which is the whole thing the gate is
    for.
    """
    specs = [_spec(v) for v in ABLATION]
    runnable, unavailable = build_runs.partition_available(specs, _LoadedStub(bm25=object()))
    assert any(s.variant.enabled(Stage.QUERY_LLM) for s in specs)
    assert not any(s.variant.enabled(Stage.QUERY_LLM) for s in runnable)
    assert build_runs.plan_quota(runnable, _gold(5), config).n_api_calls == 0
    assert unavailable, "the stub has no dense index, so some systems must be unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# The clock
# ─────────────────────────────────────────────────────────────────────────────


def test_the_pool_and_the_report_share_one_default_instant() -> None:
    """The subtlest way this whole thing can go wrong.

    Hours constraints resolve against an instant, so "open after 10pm" has a
    different answer on a Saturday than on a Tuesday. If the pool were built
    against the wall clock while `make eval` scores at its pinned instant, the
    report would surface businesses the pool never offered for grading, score
    them 0 for every system, and look entirely normal doing it.

    Asserted as a shared *default*, not two equal constants: equal constants
    drift.
    """
    from eval.run_eval import DEFAULT_EVAL_NOW, parse_now

    assert build_runs.DEFAULT_EVAL_NOW is DEFAULT_EVAL_NOW
    assert parse_now(None) == DEFAULT_EVAL_NOW
    assert DEFAULT_EVAL_NOW.strftime("%A") == "Monday"


def test_the_resolved_instant_reaches_the_pipeline(config) -> None:
    """`now` must arrive at the factory, not merely be parsed and dropped."""
    from datetime import datetime

    pinned = datetime(2024, 1, 6, 23, 30)  # a Saturday night
    seen: list[datetime | None] = []

    class _RecordingFactory:
        def __init__(self, now=None) -> None:
            seen.append(now)

        def __call__(self, cfg, variant, depths):
            return make_factory({BM25: {}})(cfg, variant, depths)

    build_runs.build(
        config,
        gold=_gold(1),
        specs=[_spec(BM25)],
        depth=5,
        factory=_RecordingFactory(now=pinned),
    )
    assert seen == [pinned]


def test_merging_onto_a_malformed_pool_is_refused(tmp_path: Path) -> None:
    """A corrupt existing pool must not be copied forward and rewritten.

    `{"bm25": "oops"}` *is* a JSON object, so an `isinstance(..., dict)` check
    would wave it through and `--merge` would rewrite it as though it were
    sound. Validation is delegated to the labeling CLI's own reader so the two
    sides cannot hold different opinions about the format.
    """
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"bm25": "oops"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="not a valid runs file"):
        build_runs.read_existing_pool(runs_path)


def test_merging_onto_nothing_says_so_instead_of_writing_a_fresh_pool_quietly(
    tmp_path: Path,
) -> None:
    """`--merge` with no existing pool is allowed, but never silent.

    Writing a fresh pool is the right outcome. Doing it without a word is not:
    `--merge` means "keep the systems already judged", so if the path is wrong —
    a typo in `--out`, a config whose `out_dir` moved — the owner gets a fresh
    pool and goes on believing earlier judgements are in it. The cost lands much
    later, as an unexplained jump in the unjudged-in-top-10 count, after the
    grading hours are spent.
    """
    missing = tmp_path / "runs.json"
    runs, notes = build_runs.merge_onto(missing, {"bm25": {"q1": ["b1"]}})

    assert runs == {"bm25": {"q1": ["b1"]}}, "the new runs must survive unchanged"
    assert len(notes) == 1
    assert "nothing to merge onto" in notes[0]
    assert str(missing) in notes[0], "the message has to name the path it looked at"


def test_merging_onto_an_existing_pool_reports_what_it_carried_forward(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"bm25": {"q1": ["b1"]}}), encoding="utf-8")

    runs, notes = build_runs.merge_onto(runs_path, {"hybrid_rrf": {"q1": ["b2"]}})

    assert runs == {"bm25": {"q1": ["b1"]}, "hybrid_rrf": {"q1": ["b2"]}}
    assert notes == ["merged: kept 1 earlier system(s) (bm25)"]


def test_merging_a_rebuild_of_every_system_reports_nothing(tmp_path: Path) -> None:
    """Nothing was carried forward, so a "kept 0" line would be noise."""
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"bm25": {"q1": ["old"]}}), encoding="utf-8")

    runs, notes = build_runs.merge_onto(runs_path, {"bm25": {"q1": ["new"]}})

    assert runs == {"bm25": {"q1": ["new"]}}
    assert notes == []


def test_merging_onto_a_malformed_pool_is_still_refused_through_merge_onto(
    tmp_path: Path,
) -> None:
    """The validation must sit on the path `main` actually takes.

    `read_existing_pool` is tested directly above; this pins that `merge_onto`
    goes through it rather than reading the file some other way.
    """
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"bm25": "oops"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="not a valid runs file"):
        build_runs.merge_onto(runs_path, {"hybrid_rrf": {"q1": ["b2"]}})


def test_a_sound_existing_pool_reads_back_for_merging(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"bm25": {"q1": ["b1", "b2"]}}), encoding="utf-8")
    assert build_runs.read_existing_pool(runs_path) == {"bm25": {"q1": ["b1", "b2"]}}
