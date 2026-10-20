"""End-to-end pipeline tests, and the processed-data round trip.

These run against the eight synthetic fixture businesses with a fake embedder
and a fake cross-encoder, so the whole cascade is exercised in milliseconds with
no model download and no Yelp data. What that buys is the thing the ablation
depends on: proof that a stage toggle actually changes what runs, not just what
the response says ran. The reranker and the retrievers are therefore wrapped in
spies, and "BM25 only did not call the reranker" is asserted against the spy,
not against the output.

The real BM25 (`bm25s`) and the real FAISS index are used — only the *models*
are faked. A fake retriever would make these tests pass while the fusion and
filter plumbing stayed unverified.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from concierge.data import build as build_module
from concierge.data.build import (
    BUSINESSES_FILE,
    DOCUMENTS_FILE,
    business_from_record,
    read_businesses,
    read_documents,
    resolve_processed_dir,
    write_businesses,
    write_documents,
)
from concierge.data.documents import build_documents
from concierge.pipeline import (
    ABLATION,
    CASCADE,
    COUNT_FINAL,
    DROPPED_PREFIX,
    STAGE_ASSEMBLE,
    STAGE_FILTER,
    STAGE_LTR,
    STAGE_PARSE,
    STAGE_RERANK,
    STAGE_RETRIEVE,
    VARIANT_FULL,
    VARIANTS,
    RerankCheckpoint,
    SearchPipeline,
    Stage,
    Variant,
)
from concierge.rerank.model import CrossEncoderReranker
from concierge.retrieval import bm25 as bm25_mod
from concierge.retrieval.dense import DenseEmbedder, DenseIndex
from concierge.types import (
    SCORE_CROSS_ENCODER,
    SCORE_LTR,
    Business,
    BusinessDocument,
    Candidate,
    ParsedQuery,
    ParseSource,
)

# A query with no hard constraints at all, so the filter stage keeps everything
# it is given and the cascade widths are what narrow the candidate list.
UNCONSTRAINED_QUERY = "somewhere nice to sit and read for a while"
# Rule-parseable constraints: NoiseLevel=quiet, WiFi=free (see test_query_rules).
CONSTRAINED_QUERY = "quiet cafe with wifi"


# ─────────────────────────────────────────────────────────────────────────────
# Fakes and spies
# ─────────────────────────────────────────────────────────────────────────────


class HashingEncoder:
    """A deterministic stand-in for the sentence-transformers encoder.

    Hashed bag of words: two texts sharing words get a higher inner product.
    That is enough for the pipeline's purposes (a *plausible* ranked list that
    changes with the query) and it means the test suite never downloads a model
    — the real model's quality is not what these tests are about.

    `zlib.crc32` rather than `hash()`: Python salts string hashing per process,
    which would make these rankings differ between runs.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(token.encode()) % self._dim] += 1.0
        return out


@dataclass
class SpyRetriever:
    """Wraps a real index and records every search call."""

    inner: Any
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        self.calls.append((query_text, top_k))
        return self.inner.search(query_text, top_k)


def _reverse_scores(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Score ascending, so the shortlist comes out exactly reversed.

    Chosen so a test can prove the reranker *reordered* rather than merely ran.
    """
    return [float(i) for i in range(len(pairs))]


@dataclass
class SpyPairScorer:
    """Stand-in for the fine-tuned cross-encoder's forward pass.

    The spy sits at the model boundary (`PairScorer`), not at the reranker
    boundary, and the real `CrossEncoderReranker` is wrapped around it. So these
    tests exercise the actual Stage 3 ordering contract — stable sort, top_k,
    score bookkeeping — while still proving, per call, whether the expensive
    part ran at all.
    """

    scores_fn: Callable[[Sequence[tuple[str, str]]], Sequence[float]] = _reverse_scores
    calls: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    max_length: int = 512

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(tuple(pairs))
        return [float(s) for s in self.scores_fn(pairs)]

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def document_token_budget(self, query_text: str) -> int:
        return self.max_length - self.count_tokens(query_text)


def _fake_reranker(scorer: SpyPairScorer | None = None) -> CrossEncoderReranker:
    """The real Stage 3 object, driven by a fake forward pass."""
    return CrossEncoderReranker(scorer or SpyPairScorer(), weights_source="fake scorer (test)")


@dataclass
class SpyLtr:
    """Stand-in for LambdaMART: puts `winner` on top, everything else below."""

    winner: str
    calls: list[int] = field(default_factory=list)

    def score(
        self,
        parsed: ParsedQuery,
        candidates: Sequence[Candidate],
        businesses: Mapping[str, Business],
    ) -> Sequence[float]:
        self.calls.append(len(candidates))
        return [1.0 if c.business_id == self.winner else 0.0 for c in candidates]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pipeline_config(config):
    """Fixture config with a narrow cascade and no network parsing.

    * `provider: rules` so no test can reach the Gemini API, whatever variant
      it runs. The "LLM stage forces the rule parser" behaviour is tested
      separately by inspecting the config the parser is handed.
    * Narrow candidate widths (6/4/2) because there are only eight fixture
      businesses: with the production 200/50/10 no stage would ever truncate
      and the cascade's narrowing would be untested.
    """
    return config.model_copy(
        update={
            "query_parsing": config.query_parsing.model_copy(update={"provider": "rules"}),
            "candidates": config.candidates.model_copy(
                update={"retrieve": 6, "rerank": 4, "final": 2}
            ),
        }
    )


@pytest.fixture
def businesses(raw_businesses: list[dict]) -> list[Business]:
    return [business_from_record(record) for record in raw_businesses]


@pytest.fixture
def documents(
    businesses: list[Business],
    raw_reviews: list[dict],
    raw_tips: list[dict],
    pipeline_config,
) -> list[BusinessDocument]:
    reviews: dict[str, list[dict]] = {}
    tips: dict[str, list[dict]] = {}
    for review in raw_reviews:
        reviews.setdefault(review["business_id"], []).append(review)
    for tip in raw_tips:
        tips.setdefault(tip["business_id"], []).append(tip)
    return build_documents(businesses, reviews, tips, pipeline_config)


@dataclass
class Harness:
    """A built pipeline plus the spies wired into it."""

    pipeline: SearchPipeline
    bm25: SpyRetriever
    dense: SpyRetriever
    #: The spy sits inside the real reranker: `scorer.calls` is empty exactly
    #: when no cross-encoder forward pass happened.
    scorer: SpyPairScorer
    ltr: SpyLtr


def _harness(
    cfg,
    businesses: Sequence[Business],
    documents: Sequence[BusinessDocument],
    *,
    business_map: Mapping[str, Business] | None = None,
    rerankers: Any = None,
    ltr: Any = None,
) -> Harness:
    bm25 = SpyRetriever(bm25_mod.build_index(documents, cfg))
    embedder = DenseEmbedder(encoder=HashingEncoder(), query_prefix="query: ", batch_size=4)
    dense = SpyRetriever(DenseIndex.build(documents, embedder, cfg))
    scorer = SpyPairScorer()
    ltr_ranker = SpyLtr(winner="b007") if ltr is None else ltr
    pipeline = SearchPipeline(
        cfg,
        businesses=(
            {b.business_id: b for b in businesses} if business_map is None else business_map
        ),
        documents=documents,
        bm25_index=bm25,
        dense_index=dense,
        reranker=_fake_reranker(scorer) if rerankers is None else rerankers,
        ltr_ranker=ltr_ranker,
    )
    return Harness(pipeline=pipeline, bm25=bm25, dense=dense, scorer=scorer, ltr=ltr_ranker)


@pytest.fixture
def harness(pipeline_config, businesses, documents) -> Harness:
    return _harness(pipeline_config, businesses, documents)


# ─────────────────────────────────────────────────────────────────────────────
# The ablation definition itself
# ─────────────────────────────────────────────────────────────────────────────


def test_ablation_rows_add_exactly_one_thing_at_a_time():
    """Each row must be the row above it plus one change.

    This is what makes the table interpretable: if two rows differed in two
    stages, the difference between them could not be attributed to either.
    """
    names = [v.name for v in ABLATION]
    assert len(set(names)) == len(names), "ablation row names must be unique"

    for previous, current in itertools.pairwise(ABLATION):
        added = current.stages & ~previous.stages
        removed = previous.stages & ~current.stages
        assert removed == Stage.NONE, f"{current.name} removes stages from {previous.name}"
        # The checkpoint only counts as a change once Stage 3 is actually on:
        # `Variant.reranker` is ignored while RERANK is off, so rows 1-2 carry
        # the default value without it meaning anything.
        changed_checkpoint = (
            previous.enabled(Stage.RERANK) and current.reranker is not previous.reranker
        )
        n_new_stages = bin(added.value).count("1")
        # Either one new stage, or the same stages with a different checkpoint
        # (the pretrained -> fine-tuned row).
        assert n_new_stages + int(changed_checkpoint) == 1, (
            f"{current.name} changes {n_new_stages} stages and "
            f"{int(changed_checkpoint)} checkpoints relative to {previous.name}"
        )


def test_ablation_covers_the_six_configurations_the_readme_reports():
    assert [v.name for v in ABLATION] == [
        "bm25",
        "hybrid_rrf",
        "rerank_pretrained",
        "rerank_finetuned",
        "llm_parse",
        "full",
    ]
    assert VARIANT_FULL.stages == (
        Stage.BM25 | Stage.DENSE | Stage.FILTERS | Stage.RERANK | Stage.LTR | Stage.QUERY_LLM
    )
    # The diagnostics are reachable by name but are not table rows.
    assert "no_filters" in VARIANTS
    assert not VARIANTS["no_filters"].enabled(Stage.FILTERS)


def test_a_variant_with_no_retriever_is_rejected():
    """Caught when the experiment is written, not when it produces zeros."""
    with pytest.raises(ValueError, match="no retriever"):
        Variant(name="broken", stages=Stage.RERANK | Stage.LTR)


# ─────────────────────────────────────────────────────────────────────────────
# A search end to end
# ─────────────────────────────────────────────────────────────────────────────


def test_search_returns_ranked_results(harness: Harness):
    response = harness.pipeline.search(UNCONSTRAINED_QUERY)

    assert response.results, "the unconstrained query should return something"
    assert len(response.results) <= harness.pipeline.cfg.candidates.final
    assert [r.rank for r in response.results] == list(range(1, len(response.results) + 1))
    for result in response.results:
        assert result.business.is_open
        assert result.stage_scores, "every result carries its upstream stage scores"


def test_timings_are_recorded_for_every_stage(harness: Harness):
    """Latency is a reported result, so it is measured by default."""
    response = harness.pipeline.search(UNCONSTRAINED_QUERY)

    stages = [t.stage for t in response.timings]
    assert stages == [
        STAGE_PARSE,
        STAGE_RETRIEVE,
        STAGE_FILTER,
        STAGE_RERANK,
        STAGE_LTR,
        STAGE_ASSEMBLE,
    ]
    assert all(t.ms >= 0.0 for t in response.timings)
    assert response.total_ms > 0.0
    # The total is wall-clock around the whole search, so it cannot be less than
    # the stages it contains.
    assert response.total_ms >= sum(t.ms for t in response.timings) - 1e-6


def test_candidate_counts_narrow_down_the_cascade(harness: Harness):
    response = harness.pipeline.search(UNCONSTRAINED_QUERY)
    counts = [response.n_candidates[stage] for stage in CASCADE]

    assert counts == sorted(counts, reverse=True), f"cascade widened: {counts}"
    assert response.n_candidates[STAGE_RETRIEVE] <= harness.pipeline.cfg.candidates.retrieve
    assert response.n_candidates[STAGE_RERANK] <= harness.pipeline.cfg.candidates.rerank
    assert response.n_candidates[COUNT_FINAL] == len(response.results)


# ─────────────────────────────────────────────────────────────────────────────
# `candidates()`: the public entry point the offline stages run
# ─────────────────────────────────────────────────────────────────────────────
#
# `candidates()` exists so that `concierge.ltr.build_examples` can build Stage
# 4's training rows from the served cascade instead of restating its order
# (DECISIONS.md D228). These tests hold it to the three things that makes it
# worth having: `search()` really does run through it, it stops before Stage 4,
# and the candidates it hands back still carry the per-stage ranks a
# `SearchResult` has dropped.


def _parsed(harness: Harness, query: str) -> ParsedQuery:
    """Parse through the public parser with the pipeline's own config.

    `pipeline_config` sets `provider: rules`, so this reaches no network; using
    the real parser rather than a hand-built `ParsedQuery` keeps these tests
    about the cascade and not about a fixture's idea of a parse.
    """
    from concierge.query import parser as query_parser

    return query_parser.parse(query, harness.pipeline.cfg)


def test_candidates_are_the_stages_search_runs(harness: Harness):
    """One cascade, two callers: `search()` is `candidates()` plus Stages 4 and 5.

    Run with Stage 4 off, `search()`'s results must be `candidates()`'s list
    rendered — same businesses, same order, same per-stage scores. If the two ever
    diverged, the rows Stage 4 trains on would describe a system that never ran.
    """
    variant = Variant(name="cascade_only", stages=VARIANT_FULL.stages & ~Stage.LTR)
    width = harness.pipeline.cfg.candidates.rerank

    response = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=variant, k=width)
    candidates = harness.pipeline.candidates(response.parsed, variant=variant)

    assert response.results, "nothing was returned, so nothing was compared"
    assert [r.business.business_id for r in response.results] == [c.business_id for c in candidates]
    for result, candidate in zip(response.results, candidates, strict=True):
        assert result.stage_scores == pytest.approx(candidate.scores)


def test_candidates_keep_the_per_stage_ranks_a_result_drops(harness: Harness):
    """The reason this method returns `Candidate`s rather than `SearchResult`s.

    `rrf_rank` and the cross-encoder rank are two of Stage 4's features, and
    `SearchResult` carries neither — that is the gap that forced the offline
    producer to run its own cascade.
    """
    from concierge.types import RANK_CROSS_ENCODER, RANK_RRF

    candidates = harness.pipeline.candidates(
        _parsed(harness, UNCONSTRAINED_QUERY), variant=VARIANTS["rerank_finetuned"]
    )

    assert candidates
    for candidate in candidates:
        assert RANK_RRF in candidate.ranks
        assert RANK_CROSS_ENCODER in candidate.ranks


def test_candidates_stops_before_stage_four(harness: Harness):
    """Stage 4 must not run, even under the full variant.

    These candidates are Stage 4's *training input*: scoring them with Stage 4
    would feed the model its own prediction back as a feature. Asserted against
    the spy, not against the absence of a score key, because a model that ran and
    whose score was discarded would still have cost the latency.
    """
    candidates = harness.pipeline.candidates(
        _parsed(harness, UNCONSTRAINED_QUERY), variant=VARIANT_FULL
    )

    assert candidates
    assert harness.ltr.calls == []
    assert all(SCORE_LTR not in c.scores for c in candidates)


def test_candidates_records_the_counts_search_reports(harness: Harness):
    """The producer's reported counts are the pipeline's own, not a second tally."""
    from concierge.pipeline import StageRecorder

    variant = VARIANTS["rerank_finetuned"]
    response = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=variant)
    record = StageRecorder()
    harness.pipeline.candidates(response.parsed, variant=variant, record=record)

    for stage in (STAGE_RETRIEVE, STAGE_FILTER, STAGE_RERANK):
        assert record.counts[stage] == response.n_candidates[stage]
    assert [t.stage for t in record.timings] == [STAGE_RETRIEVE, STAGE_FILTER, STAGE_RERANK]


def test_candidates_limit_truncates_the_list_but_not_the_counts(harness: Harness):
    """`limit` is what the caller wants to see; the counts are what the stages did.

    A count that moved with the caller's `limit` would make the latency analysis
    report the shortlist sizes somebody asked for rather than the ones the cascade
    produced.
    """
    from concierge.pipeline import StageRecorder

    parsed = _parsed(harness, UNCONSTRAINED_QUERY)
    full = harness.pipeline.candidates(parsed)
    assert len(full) > 1, "truncation needs more than one candidate"

    record = StageRecorder()
    limited = harness.pipeline.candidates(parsed, limit=1, record=record)

    assert limited == full[:1]
    assert record.counts[STAGE_RERANK] == len(full)
    assert harness.pipeline.candidates(parsed, limit=0) == ()
    with pytest.raises(ValueError, match="limit must be"):
        harness.pipeline.candidates(parsed, limit=-1)


def test_candidates_returns_nothing_when_there_is_nothing_to_search_for(harness: Harness):
    """The same honest answer `search()` gives: no results, no exception."""
    parsed = _parsed(harness, "   ")

    assert harness.pipeline.candidates(parsed) == ()
    assert harness.bm25.calls == []


def test_constraint_matches_are_attached_for_the_why_this_result_ui(harness: Harness):
    response = harness.pipeline.search(CONSTRAINED_QUERY)

    assert response.parsed.attributes, "the rule parser should find the quiet/wifi asks"
    assert response.results
    for result in response.results:
        names = {c.name for c in result.constraints}
        assert set(response.parsed.attributes) <= names | {"category"}


def test_explanations_exist_even_with_filters_disabled(harness: Harness):
    """The explanation is a property of the answer, not of the filter stage."""
    response = harness.pipeline.search(CONSTRAINED_QUERY, variant=VARIANTS["no_filters"])

    assert response.results
    assert all(result.constraints for result in response.results)


# ─────────────────────────────────────────────────────────────────────────────
# Stage toggles actually change what runs
# ─────────────────────────────────────────────────────────────────────────────


def test_bm25_only_variant_calls_neither_dense_nor_the_reranker(harness: Harness):
    """The claim the whole ablation rests on, asserted against spies.

    A response that merely *reports* a shorter cascade would look identical
    while the expensive stages still ran — and the latency column of the table
    would be wrong.
    """
    response = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["bm25"])

    assert harness.bm25.calls, "the lexical half must still run"
    assert harness.dense.calls == []
    assert harness.scorer.calls == []
    assert harness.ltr.calls == []
    assert all(SCORE_CROSS_ENCODER not in r.stage_scores for r in response.results)
    assert all(SCORE_LTR not in r.stage_scores for r in response.results)


def test_hybrid_variant_queries_both_retrievers(harness: Harness):
    harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])

    assert len(harness.bm25.calls) == 1
    assert len(harness.dense.calls) == 1
    assert harness.scorer.calls == []


def test_dense_only_variant_does_not_call_bm25(harness: Harness):
    harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["dense_only"])

    assert harness.bm25.calls == []
    assert len(harness.dense.calls) == 1


def test_reranker_runs_once_on_the_shortlist_and_reorders_it(harness: Harness):
    cfg = harness.pipeline.cfg
    retrieved = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])
    reranked = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["rerank_finetuned"])

    assert len(harness.scorer.calls) == 1, "one cross-encoder call per search"
    scored_pairs = harness.scorer.calls[0]
    # Stage 3 scores everything Stage 2 retrieved and keeps the best `rerank` of
    # them — that is what lets it rescue a candidate retrieval ranked last.
    assert len(scored_pairs) == retrieved.n_candidates[STAGE_FILTER]
    assert reranked.n_candidates[STAGE_RERANK] <= cfg.candidates.rerank

    # The spy scores ascending, so the shortlist comes back reversed: the two
    # orderings must differ, which is the point of having a Stage 3 at all.
    assert [r.business.business_id for r in reranked.results] != [
        r.business.business_id for r in retrieved.results
    ]
    assert all(SCORE_CROSS_ENCODER in r.stage_scores for r in reranked.results)


def test_ltr_stage_decides_the_final_order_when_enabled(harness: Harness):
    """Stage 4 must be able to promote a business Stage 3 ranked last.

    The winner is read off the Stage 3 ordering rather than hard-coded, so the
    test asserts "LambdaMART decided the order" instead of accidentally
    asserting something about the fixture corpus.
    """
    width = harness.pipeline.cfg.candidates.rerank
    reranked = harness.pipeline.search(
        UNCONSTRAINED_QUERY, variant=VARIANTS["rerank_finetuned"], k=width
    )
    underdog = reranked.results[-1].business.business_id
    assert underdog != reranked.results[0].business.business_id

    ranker = SpyLtr(winner=underdog)
    harness.pipeline.ltr_ranker = ranker
    response = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANT_FULL)

    assert ranker.calls, "Stage 4 must have been asked to score"
    assert response.results[0].business.business_id == underdog
    assert response.results[0].score == response.results[0].stage_scores[SCORE_LTR]


def test_llm_stage_decides_which_provider_the_parser_sees(
    pipeline_config, businesses, documents, monkeypatch
):
    """`Stage.QUERY_LLM` off must mean no API call, not just a relabelled row.

    Asserted on the provider in the config handed to the parser, because that
    is the thing that decides whether an HTTP request happens — the one place
    where "the LLM row" and "the rules row" become physically different.

    The stub never calls the real parser, so this test cannot reach the network
    even for the gemini half.
    """
    import concierge.pipeline as pipeline_module
    from concierge.query.rules import parse_rules

    cfg = pipeline_config.model_copy(
        update={
            "query_parsing": pipeline_config.query_parsing.model_copy(update={"provider": "gemini"})
        }
    )
    harness = _harness(cfg, businesses, documents)

    seen: list[str] = []

    def stub_parse(query, parse_cfg, **kwargs):
        seen.append(parse_cfg.query_parsing.provider)
        return parse_rules(query, today="Friday")

    monkeypatch.setattr(pipeline_module.query_parser, "parse", stub_parse)

    harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])
    assert seen == ["rules"]

    harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANT_FULL)
    assert seen[-1] == "gemini", "the LLM row must actually ask for the LLM"


def test_rule_parsed_source_is_reported(harness: Harness):
    """The ablation cannot attribute a row to a parser it cannot identify."""
    response = harness.pipeline.search(CONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])
    assert response.parsed.source is ParseSource.RULES


# ─────────────────────────────────────────────────────────────────────────────
# Hard constraints
# ─────────────────────────────────────────────────────────────────────────────


def test_permanently_closed_business_never_reaches_the_results(
    pipeline_config, businesses, documents
):
    """b004 is `is_open == 0`. The only query it matches lexically is "diner".

    The result is empty *and* the counts say why — which is exactly what the
    failure analysis needs to distinguish "we ranked badly" from "there was
    nothing to rank".
    """
    harness = _harness(pipeline_config, businesses, documents)
    response = harness.pipeline.search("diner", variant=VARIANTS["bm25"])

    ids = [r.business.business_id for r in response.results]
    assert "b004" not in ids
    assert response.n_candidates[f"{DROPPED_PREFIX}permanently_closed"] == 1


def test_violated_constraints_remove_candidates(harness: Harness):
    """b002 is loud with no wifi, so it cannot answer "quiet cafe with wifi"."""
    response = harness.pipeline.search(CONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])

    ids = [r.business.business_id for r in response.results]
    assert "b002" not in ids
    assert response.n_candidates[STAGE_FILTER] <= response.n_candidates[STAGE_RETRIEVE]


# ─────────────────────────────────────────────────────────────────────────────
# Empty and degraded paths
# ─────────────────────────────────────────────────────────────────────────────


def test_zero_surviving_candidates_returns_an_empty_response(
    pipeline_config, businesses, documents
):
    """No business records at all: every candidate is unrenderable.

    Must return cleanly rather than raise — a search engine that 500s on a
    narrow query is worse than one that says "nothing matched".
    """
    harness = _harness(pipeline_config, businesses, documents, business_map={})
    response = harness.pipeline.search(UNCONSTRAINED_QUERY)

    assert response.results == ()
    assert response.n_candidates[COUNT_FINAL] == 0
    assert response.n_candidates[STAGE_FILTER] == 0
    assert response.n_candidates[STAGE_RETRIEVE] > 0, "candidates were retrieved, then dropped"
    assert response.n_candidates[f"{DROPPED_PREFIX}unknown_business"] > 0
    assert response.total_ms > 0.0


def test_blank_query_returns_an_empty_response_without_retrieving(harness: Harness):
    response = harness.pipeline.search("   ")

    assert response.results == ()
    assert harness.bm25.calls == []
    assert harness.dense.calls == []
    assert response.n_candidates[COUNT_FINAL] == 0
    assert [t.stage for t in response.timings] == [STAGE_PARSE]


def test_missing_finetuned_checkpoint_falls_back_to_pretrained(
    pipeline_config, businesses, documents, caplog
):
    """The project must stay runnable before the cross-encoder is fine-tuned."""
    pretrained = SpyPairScorer()
    harness = _harness(
        pipeline_config,
        businesses,
        documents,
        rerankers={RerankCheckpoint.PRETRAINED: _fake_reranker(pretrained)},
    )

    with caplog.at_level(logging.WARNING, logger="concierge.pipeline"):
        response = harness.pipeline.search(
            UNCONSTRAINED_QUERY, variant=VARIANTS["rerank_finetuned"]
        )

    assert len(pretrained.calls) == 1
    assert any("falling back to the pretrained" in r.message for r in caplog.records)
    assert all(SCORE_CROSS_ENCODER in r.stage_scores for r in response.results)


def test_missing_reranker_entirely_keeps_retrieval_order(
    pipeline_config, businesses, documents, caplog
):
    harness = _harness(pipeline_config, businesses, documents, rerankers={})

    with caplog.at_level(logging.WARNING, logger="concierge.pipeline"):
        degraded = harness.pipeline.search(
            UNCONSTRAINED_QUERY, variant=VARIANTS["rerank_finetuned"]
        )
    retrieved = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])

    assert any("skipping Stage 3" in r.message for r in caplog.records)
    assert [r.business.business_id for r in degraded.results] == [
        r.business.business_id for r in retrieved.results
    ]


def test_missing_ltr_model_keeps_cross_encoder_order(
    pipeline_config, businesses, documents, caplog
):
    harness = _harness(pipeline_config, businesses, documents, ltr=None)
    harness.pipeline.ltr_ranker = None

    with caplog.at_level(logging.WARNING, logger="concierge.pipeline"):
        full = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANT_FULL)
    reranked = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["rerank_finetuned"])

    assert any("no LambdaMART model" in r.message for r in caplog.records)
    assert [r.business.business_id for r in full.results] == [
        r.business.business_id for r in reranked.results
    ]
    assert all(SCORE_LTR not in r.stage_scores for r in full.results)


def test_a_misaligned_reranker_fails_loudly(pipeline_config, businesses, documents):
    """Wrong number of scores means every score is attached to the wrong
    business, and the ranking would still look plausible. Never silent."""
    broken = _fake_reranker(SpyPairScorer(scores_fn=lambda pairs: [1.0]))
    harness = _harness(pipeline_config, businesses, documents, rerankers=broken)

    with pytest.raises(ValueError, match="scores for"):
        harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANTS["rerank_finetuned"])


# ─────────────────────────────────────────────────────────────────────────────
# concierge.data.build: the processed-data round trip
# ─────────────────────────────────────────────────────────────────────────────


def test_business_parquet_round_trip_is_exact(businesses, tmp_path: Path):
    """Attributes and hours survive the JSON-in-parquet encoding unchanged.

    This is the contract the filters depend on: an overnight interval or a
    normalized "u'quiet'" that changed shape on the way through disk would
    break constraint checking in production while every parser test still
    passed.
    """
    write_businesses(businesses, tmp_path)
    loaded = read_businesses(tmp_path)

    assert loaded == businesses


def test_round_trip_preserves_the_awkward_fixtures(businesses, tmp_path: Path):
    write_businesses(businesses, tmp_path)
    by_id = {b.business_id: b for b in read_businesses(tmp_path)}

    # b002: overnight hours (17:00-02:00) must still cross midnight.
    friday = by_id["b002"].hours["Friday"][0]
    assert friday.crosses_midnight and friday.covers(30)
    # b005: no attributes and no hours at all.
    assert by_id["b005"].attributes == {} and by_id["b005"].hours == {}
    # b007: Yelp's open-24-hours encoding.
    assert by_id["b007"].hours["Monday"][0].covers(3 * 60)
    # b004: closed businesses are still *representable*; excluding them is the
    # build's job, not the serializer's.
    assert by_id["b004"].is_open is False


def test_documents_parquet_matches_what_the_index_builder_reads(documents, tmp_path: Path):
    write_documents(documents, tmp_path)

    import polars as pl

    columns = set(pl.read_parquet(tmp_path / DOCUMENTS_FILE).columns)
    # `concierge.retrieval.build_index.load_documents` requires these names.
    assert {"business_id", "text", "n_reviews_used", "n_tips_used", "truncated"} <= columns
    assert read_documents(tmp_path) == list(documents)


def test_writers_refuse_to_write_nothing(tmp_path: Path):
    """An empty parquet is a successful-looking build with no data in it."""
    with pytest.raises(ValueError):
        write_businesses([], tmp_path)
    with pytest.raises(ValueError):
        write_documents([], tmp_path)


def test_reading_before_building_says_what_to_run(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="make data"):
        read_businesses(tmp_path)
    with pytest.raises(FileNotFoundError, match="make data"):
        read_documents(tmp_path)


def test_processed_dir_resolves_relative_paths_against_the_repo(pipeline_config):
    resolved = resolve_processed_dir(pipeline_config)
    assert resolved.is_absolute()
    assert resolved.name == Path(pipeline_config.data.processed_dir).name


def _write_config(path: Path, raw_dir: Path, metro: str | None) -> Path:
    import yaml

    payload: dict[str, Any] = {
        "data": {"raw_dir": str(raw_dir), "processed_dir": str(path.parent / "processed")}
    }
    if metro is not None:
        payload["data"]["metro"] = metro
    path.write_text(yaml.safe_dump(payload))
    return path


def test_build_cli_without_a_metro_exits_with_an_actionable_error(tmp_path: Path, capsys):
    """`data.metro` is the owner's decision (D009); the build must not guess."""
    cfg_path = _write_config(tmp_path / "cfg.yaml", tmp_path / "raw", metro=None)
    assert build_module.main(["--config", str(cfg_path)]) == 2
    assert "metro" in capsys.readouterr().out


def test_build_cli_without_raw_data_prints_the_download_instructions(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg_path = _write_config(tmp_path / "cfg.yaml", raw, metro="Phoenix")
    assert build_module.main(["--config", str(cfg_path)]) == 2
    assert "yelp.com/dataset" in capsys.readouterr().out


@pytest.mark.needs_data
def test_build_over_the_real_dataset(tmp_path: Path):
    """The only test that needs the 5 GB download; skipped without it.

    Everything above runs on the fixtures, so this exists to catch the things
    only real data can show: a column type we guessed wrong, or a metro name
    that matches nothing.
    """
    from concierge.config import load_config
    from concierge.data.build import build
    from concierge.data.load import BUSINESS_FILE, raw_path

    cfg = load_config("configs/base.yaml")
    if not raw_path(cfg, BUSINESS_FILE).is_file():
        pytest.skip("Yelp Open Dataset not present in data/raw")
    if not cfg.data.metro:
        pytest.skip("configs/base.yaml has no metro yet (owner's Phase 1 decision)")

    summary = build(cfg, limit=200, processed_dir=tmp_path)

    assert summary.n_businesses > 0
    assert summary.n_documents == summary.n_businesses
    assert summary.limit_applied == 200
    assert json.loads((tmp_path / "build_report.json").read_text())["metro"] == summary.metro
    assert (tmp_path / BUSINESSES_FILE).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Loading from disk: the path `make data && make index && make serve` takes
# ─────────────────────────────────────────────────────────────────────────────


def test_from_config_loads_indexes_and_data_from_disk(
    pipeline_config, businesses, documents, tmp_path: Path
):
    """`from_config` is what the API calls once at startup, so it is tested.

    Models are skipped (`load_models=False`) and the encoder is injected, so
    this exercises the real loading path — parquet, BM25 artifact, FAISS
    artifact — without downloading anything.
    """
    from concierge.retrieval.dense import DenseEmbedder, DenseIndex, dense_index_dir

    cfg = pipeline_config
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    write_businesses(businesses, processed)
    write_documents(documents, processed)

    # The dense index is the one artifact `from_config` will not rebuild for
    # itself, because rebuilding it means embedding every document.
    embedder = DenseEmbedder.from_config(cfg, encoder=HashingEncoder())
    DenseIndex.build(documents, embedder, cfg).save(dense_index_dir(cfg, artifacts))

    pipeline = SearchPipeline.from_config(
        cfg,
        encoder=HashingEncoder(),
        artifacts_root=artifacts,
        processed_dir=processed,
        load_models=False,
    )

    assert len(pipeline.businesses) == len(businesses)
    assert pipeline.doc_text, "document text is what the cross-encoder reads"

    response = pipeline.search(CONSTRAINED_QUERY, variant=VARIANTS["hybrid_rrf"])
    assert response.results
    assert (artifacts / "bm25").exists(), "a missing BM25 index is rebuilt, not fatal"


def test_from_config_without_a_dense_index_degrades_to_lexical(
    pipeline_config, businesses, documents, tmp_path: Path, caplog
):
    """Before `make index` has ever run, search must still work."""
    processed = tmp_path / "processed"
    write_businesses(businesses, processed)
    write_documents(documents, processed)

    with caplog.at_level(logging.WARNING, logger="concierge.pipeline"):
        pipeline = SearchPipeline.from_config(
            pipeline_config,
            encoder=HashingEncoder(),
            artifacts_root=tmp_path / "artifacts",
            processed_dir=processed,
            load_models=False,
        )
        # "ramen" rather than CONSTRAINED_QUERY: without PyStemmer installed
        # BM25 indexes unstemmed tokens, so "cafe" does not match "Cafes" and
        # the lexical half legitimately returns nothing. That is a real property
        # of this environment (bm25.py warns about it), not something to paper
        # over — so the test asks a question BM25 alone can answer.
        response = pipeline.search("ramen", variant=VARIANTS["bm25"])

    assert pipeline.dense_index is None
    assert any("dense" in r.message for r in caplog.records)
    assert response.results, "the lexical half alone must still answer"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 wiring: the saved LambdaMART artifact
# ─────────────────────────────────────────────────────────────────────────────


class _FakeBooster:
    """A LightGBM booster's inference surface, without LightGBM.

    Used because importing *and running* lightgbm in the same process as faiss
    segfaults in this environment (duplicate OpenMP runtimes — the same clash
    `concierge.retrieval.build_index.warn_on_duplicate_libomp` reports). That
    leaves one surface genuinely untested here: reading a real `model.txt`
    written by `make train-ltr`. It is untested, not assumed to work — see
    DECISIONS/blocked_on, and `tests/test_ltr.py` covers the training side.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self._names = list(names)
        self.seen: list[Any] = []

    def feature_name(self) -> list[str]:
        return list(self._names)

    def predict(self, values: np.ndarray) -> np.ndarray:
        self.seen.append(values)
        # Rank by star rating, so the ordering is checkable from the fixtures.
        from concierge.ltr.features import F_STARS

        return values[:, self._names.index(F_STARS)]


def test_ltr_adapter_builds_features_in_the_saved_column_order(
    pipeline_config, businesses, documents
):
    """Stage 4's serving path: features by name, in the model's own order.

    Asserted on the matrix the booster is handed, because feature skew between
    training and serving is invisible in the output — it just ranks worse.
    """
    from concierge.ltr.features import F_STARS, FEATURE_NAMES
    from concierge.pipeline import LightGbmRanker

    booster = _FakeBooster(FEATURE_NAMES)
    harness = _harness(
        pipeline_config,
        businesses,
        documents,
        ltr=LightGbmRanker(booster, FEATURE_NAMES),
    )

    response = harness.pipeline.search(UNCONSTRAINED_QUERY, variant=VARIANT_FULL)

    assert booster.seen, "the booster must have been asked to predict"
    matrix = booster.seen[0]
    assert matrix.shape[1] == len(FEATURE_NAMES)
    # The star column really is the star rating: proof the names line up.
    stars_column = matrix[:, list(FEATURE_NAMES).index(F_STARS)]
    assert set(stars_column) <= {b.stars for b in businesses}

    assert response.results
    assert all(SCORE_LTR in r.stage_scores for r in response.results)
    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True)


@dataclass
class RecordingLtr:
    """Records what Stage 4 was handed, then delegates to the real adapter.

    Two things have to be captured to test Stage 4's inputs rather than its
    outputs: the exact `(parsed, candidates)` the pipeline handed it, and the
    matrix `LightGbmRanker` built from them. The delegate is the production
    adapter, so the matrix is the one serving really computes and not a
    re-derivation in the test.
    """

    inner: Any
    seen_parsed: list[ParsedQuery] = field(default_factory=list)
    seen_candidates: list[tuple[Candidate, ...]] = field(default_factory=list)

    def score(
        self,
        parsed: ParsedQuery,
        candidates: Sequence[Candidate],
        businesses: Mapping[str, Business],
    ) -> Sequence[float]:
        self.seen_parsed.append(parsed)
        self.seen_candidates.append(tuple(candidates))
        return self.inner.score(parsed, candidates, businesses)


def _stage4_recorder(pipeline_config, businesses, documents) -> tuple[Harness, RecordingLtr, Any]:
    """A harness whose Stage 4 is the real adapter behind a recorder."""
    from concierge.ltr.features import FEATURE_NAMES
    from concierge.pipeline import LightGbmRanker

    booster = _FakeBooster(FEATURE_NAMES)
    recorder = RecordingLtr(LightGbmRanker(booster, FEATURE_NAMES))
    harness = _harness(pipeline_config, businesses, documents, ltr=recorder)
    return harness, recorder, booster


def _n_distinct(column: np.ndarray) -> int:
    """Distinct values in a feature column, counting NaN as one value.

    Written out because `set(column)` would count every NaN separately
    (NaN != NaN) and so would report a constant all-NaN column as varying —
    which is the exact failure this helper is used to detect.
    """
    values = {float(v) for v in column if not np.isnan(v)}
    return len(values) + (1 if any(np.isnan(v) for v in column) else 0)


@pytest.mark.parametrize("variant_name", ["full", "no_filters"])
def test_constraint_features_vary_across_candidates_when_stage4_scores(
    pipeline_config, businesses, documents, variant_name: str
):
    """The constraint features must carry signal *at ranking time*.

    This is a regression test for a bug with no symptom. The constraint checks
    used to be attached during result assembly, which runs *after* Stage 4, so
    in any variant with the filter stage off LambdaMART saw an empty constraint
    list on every candidate: `n_constraints_satisfied` constant at 0,
    `frac_constraints_satisfied` constant at NaN, `has_unknown_constraint_data`
    constant at 0. A constant column is one LightGBM can never split on, so
    three of the model's features would simply have been missing from the
    feature-importance chart — a listed deliverable — and the honest reading of
    that chart ("the constraint features did not matter") would have been a
    false conclusion drawn from features that were never populated.

    So the assertion is on *variation*, not well-formedness. A test that
    checked the row had the right width and no exceptions would have passed
    throughout the bug's entire life.

    Both variants are checked because the one with filters off is where the bug
    lived, and the one with filters on is where it must not appear later: the
    two paths attach constraints through different code (`apply_filters` and
    `_with_constraints`) and both call the same `evaluate_constraints`.
    """
    from concierge.ltr.features import F_FRAC_SATISFIED, F_HAS_UNKNOWN, F_N_SATISFIED, FEATURE_NAMES

    harness, recorder, booster = _stage4_recorder(pipeline_config, businesses, documents)
    # The whole shortlist, not the final two: the features Stage 4 ranks *by*
    # are the subject here, so the test must see every row it scored.
    harness.pipeline.search(
        CONSTRAINED_QUERY,
        variant=VARIANTS[variant_name],
        k=harness.pipeline.cfg.candidates.rerank,
    )

    assert booster.seen, "Stage 4 must have been asked to predict"
    matrix = booster.seen[0]
    assert len(recorder.seen_candidates[0]) > 1, "variation needs more than one candidate"

    # Checked first because it is the bug's own shape: if no candidate carries
    # constraint checks at ranking time, all three features are *necessarily*
    # constant, and saying so is a sharper failure than "this column is flat".
    assert any(c.constraints for c in recorder.seen_candidates[0]), (
        "no candidate carried constraint checks when Stage 4 scored it, so the "
        "three constraint features cannot be anything but constant"
    )

    for name in (F_N_SATISFIED, F_FRAC_SATISFIED, F_HAS_UNKNOWN):
        column = matrix[:, list(FEATURE_NAMES).index(name)]
        assert _n_distinct(column) > 1, (
            f"{name} is constant at {column[0]} across "
            f"{len(column)} candidates of {CONSTRAINED_QUERY!r} — LightGBM "
            "cannot split on it, so the feature is dead"
        )


@pytest.mark.parametrize("variant_name", ["full", "no_filters"])
def test_every_candidate_stage4_scores_carries_its_constraint_checks(
    pipeline_config, businesses, documents, variant_name: str
):
    """The invariant behind the two tests around this one, asserted directly.

    Stated as an invariant rather than as a property of one query because the
    failure it guards against is *structural*: any future candidate source that
    joins the cascade after the filter stage — a constraint-only retriever, a
    backfill for thin result sets, a cache — would hand Stage 4 candidates with
    no checks attached, and three of its features would go dead again for
    exactly those candidates. That is a worse version of the original bug,
    because it would affect only some rows and so would not even show up as a
    constant column.

    `_rank_ltr` deliberately does not repair such a candidate: a repair there
    would make the Stage-2 placement untestable and hide the plumbing mistake.
    This test is the alarm instead.
    """
    harness, recorder, _ = _stage4_recorder(pipeline_config, businesses, documents)
    response = harness.pipeline.search(
        CONSTRAINED_QUERY,
        variant=VARIANTS[variant_name],
        k=harness.pipeline.cfg.candidates.rerank,
    )

    expected = set(response.parsed.attributes) | (
        {"category"} if response.parsed.categories else set()
    )
    assert expected, "the fixture query must parse to at least one hard constraint"

    for candidate in recorder.seen_candidates[0]:
        assert {m.name for m in candidate.constraints} == expected, (
            f"{candidate.business_id} reached Stage 4 without the query's "
            f"constraint checks; LambdaMART's constraint features are dead for it"
        )


def test_stage4_sees_the_same_features_the_trainer_would_build(
    pipeline_config, businesses, documents
):
    """Train/serve parity, asserted on the numbers rather than on the design.

    `concierge.ltr.features` is one pure function precisely so the training job
    and the serving path cannot drift. That argument is only worth anything if
    the two are fed the same *inputs*, which is a property of this pipeline and
    not of the feature module: the trainer reads `ConstraintMatch`es recorded in
    its examples file, so serving has to hand the model constraint data too, and
    on the same stage of the cascade.

    The test therefore takes the exact `(parsed, candidate, business)` triples
    Stage 4 was handed at serving time, wraps them in `LtrExample`s as the
    training loop does, and runs them through `build_ranking_data` — the
    training entry point. The two matrices must be identical value for value,
    NaNs included.
    """
    from concierge.ltr.features import FEATURE_NAMES
    from concierge.ltr.train import LtrExample, build_ranking_data

    harness, recorder, booster = _stage4_recorder(pipeline_config, businesses, documents)
    harness.pipeline.search(
        CONSTRAINED_QUERY, variant=VARIANT_FULL, k=harness.pipeline.cfg.candidates.rerank
    )

    served = booster.seen[0]
    parsed = recorder.seen_parsed[0]
    candidates = recorder.seen_candidates[0]
    assert candidates

    trained = build_ranking_data(
        [
            LtrExample(
                query_id="q1",  # one query, so the group array is a single block
                query_text=parsed.raw,
                label=0,  # labels do not enter the feature matrix
                parsed=parsed,
                candidate=candidate,
                business=harness.pipeline.businesses[candidate.business_id],
            )
            for candidate in candidates
        ],
        names=FEATURE_NAMES,
    ).features

    assert trained.names == tuple(FEATURE_NAMES)
    # assert_array_equal, not ==: it treats NaN as equal to NaN, which is what
    # "the same feature value" means for a deliberately-missing signal.
    np.testing.assert_array_equal(trained.values, served)


def test_permuted_feature_order_is_refused():
    """Columns in the wrong order produce numbers, not errors — so check them."""
    from concierge.ltr.features import FEATURE_NAMES
    from concierge.pipeline import check_feature_order

    check_feature_order(FEATURE_NAMES, FEATURE_NAMES)  # the agreeing case
    check_feature_order(FEATURE_NAMES, [])  # an older model file carries no names

    with pytest.raises(ValueError, match="feature order"):
        check_feature_order(FEATURE_NAMES, tuple(reversed(FEATURE_NAMES)))


def test_untrained_lambdamart_says_which_command_to_run(pipeline_config, tmp_path: Path):
    from concierge.pipeline import LightGbmRanker

    with pytest.raises(FileNotFoundError, match="make train-ltr"):
        LightGbmRanker.load(pipeline_config, tmp_path / "nothing-here")


def test_a_stage_recorder_refuses_reuse_across_queries():
    """Reuse would silently corrupt the latency numbers.

    `timings` is a list and appends; `counts` is a dict keyed by stage and
    overwrites. A caller reusing one recorder therefore collects every stage
    several times while keeping only the last query's counts — nothing raises,
    and the per-stage p50/p95 computed from it is simply wrong. Latency is a
    reported result here, so reuse is an error rather than a documented mistake.
    """
    from concierge.pipeline import StageRecorder

    recorder = StageRecorder()
    with recorder.measure("retrieve"):
        pass
    assert len(recorder.timings) == 1

    with pytest.raises(RuntimeError, match="already measured"), recorder.measure("retrieve"):
        pass

    # The refusal must not leave a partial second entry behind.
    assert len(recorder.timings) == 1

    # A different stage on the same run is still fine — that is the normal case.
    with recorder.measure("filter"):
        pass
    assert [t.stage for t in recorder.timings] == ["retrieve", "filter"]
