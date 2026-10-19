"""Tests for the constraint-only retrieval path (DECISIONS.md D224).

The defect under test, restated so a failure here is self-explanatory: Stage 1
strips constraint phrases out of the query, the residue sometimes matches no
document, retrieval returns nothing, and the query's *correctly parsed* hard
constraints are therefore never applied — because `filters.apply_filters`
filters a candidate list and there is no candidate list. The symptom is an empty
page, which is the failure shape least likely to be noticed in a demo.

Three things these tests are deliberate about.

1. **The queries are the measured ones.** `D224_TABLE` holds the exact queries
   from the DECISIONS.md table, with the businesses that genuinely satisfy each
   one worked out from the fixture corpus by hand (see the comment on each row).
   If the fix stops returning them, this file says which query and why.

2. **Non-displacement is asserted on the score, not on list position.**
   `apply_filters` re-sorts its survivors by `SCORE_RRF`, so "appended at the
   end" is not a guarantee of anything. The guarantee has to live in the score,
   so that is what is asserted.

3. **Agreement with the filters is asserted end to end.** A backfilled candidate
   that `apply_filters` then drops would leave the page empty again — the bug,
   one stage later. So the central test runs the real `apply_filters` over the
   backfill's output under all three missing-data policies.
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from concierge.config import Config
from concierge.data.build import business_from_record
from concierge.data.documents import build_documents
from concierge.pipeline import Retriever, SearchPipeline, Stage, Variant
from concierge.query.rules import parse_rules
from concierge.retrieval import bm25 as bm25_mod
from concierge.retrieval.constraints_index import (
    SCORE_CONSTRAINT,
    TRIGGER_FEW_CANDIDATES,
    TRIGGER_FEW_SATISFIED,
    ConstraintIndex,
    backfill,
    shrunk_rating,
)
from concierge.retrieval.filters import apply_filters, evaluate_constraints
from concierge.types import (
    RANK_RRF,
    SCORE_RRF,
    Business,
    Candidate,
    ConstraintStatus,
    GeoConstraint,
    OpenAt,
    ParsedQuery,
)

# The weekday every parse in this file is resolved against, so that "open past
# 10pm" means a fixed night and not whichever day the suite happens to run on.
TODAY = "Monday"


# ─────────────────────────────────────────────────────────────────────────────
# The measured defect, as data
# ─────────────────────────────────────────────────────────────────────────────

#: ``(query, business_ids that genuinely satisfy every constraint)``.
#:
#: Worked out from `tests/conftest.py`'s fixture corpus by hand — these are the
#: businesses whose stored data *establishes* the constraint, i.e. whose every
#: `ConstraintMatch` is SATISFIED. Businesses whose data is merely silent are a
#: separate group and are covered by `test_established_results_come_before_...`.
D224_TABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # OutdoorSeating=True: b001 and b006 store "True". b002 stores "False" (so
    # it is refuted, never returned) and b008 stores the string "None", which
    # normalizes to missing.
    ("somewhere with a patio", ("b001", "b006")),
    # price_max=2: b001 and b003 are "2"; b002, b006, b007 are "1". b008 is "4",
    # refuted. b005 has no attributes at all.
    ("cheap place", ("b001", "b003", "b006", "b007", "b002")),
    # WiFi=free: b001, b003, b006. b002 is "no" and b008 is "paid", both refuted.
    ("free wifi", ("b001", "b003", "b006")),
    # NoiseLevel=quiet AND WiFi=free AND category Cafes — all three established
    # only by b001 and b003.
    ("quiet cafe with wifi", ("b001", "b003")),
)

#: A query whose semantic text is genuinely meaningful: "ramen" is a literal
#: token in exactly one fixture document, so the lexical retriever has real
#: evidence that must not be diluted.
MEANINGFUL_QUERY = "ramen"
MEANINGFUL_QUERY_BEST = "b002"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def businesses(raw_businesses: list[dict]) -> list[Business]:
    return [business_from_record(record) for record in raw_businesses]


@pytest.fixture
def index(businesses: list[Business]) -> ConstraintIndex:
    return ConstraintIndex.build(businesses)


@pytest.fixture
def narrow_config(config: Config) -> Config:
    """The fixture corpus is eight businesses, so the production 200/50/10
    widths would never make any stage narrow. These widths do, and they put
    `candidates.final` (the default backfill floor) in range of what the lexical
    retriever actually returns on this corpus.
    """
    return config.model_copy(
        update={
            "query_parsing": config.query_parsing.model_copy(update={"provider": "rules"}),
            "candidates": config.candidates.model_copy(
                update={"retrieve": 6, "rerank": 4, "final": 2}
            ),
        }
    )


def with_policy(cfg: Config, policy: str) -> Config:
    return cfg.model_copy(
        update={"retrieval": cfg.retrieval.model_copy(update={"missing_attribute_policy": policy})}
    )


def parse(query: str) -> ParsedQuery:
    """The rule parser, pinned to one weekday. The rule parser rather than the
    LLM one because the defect was measured on it, and because a test must not
    need an API key."""
    return parse_rules(query, today=TODAY)


class _HashingEncoder:
    """Deterministic stand-in for the sentence-transformers encoder, copied in
    spirit from `tests/test_pipeline.py`: `zlib.crc32` rather than `hash()`
    because Python salts string hashing per process."""

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


@pytest.fixture
def lexical_pipeline(
    narrow_config: Config,
    businesses: list[Business],
    raw_reviews: list[dict],
    raw_tips: list[dict],
) -> SearchPipeline:
    """A real BM25 index, no dense index, no reranker, no Stage 4.

    That is the configuration the D224 numbers were measured in, and it is also
    ablation row 1 — the row the project's first real numbers come from. Using
    the real `bm25s` index matters: a fake retriever would let these tests pass
    while the actual plumbing stayed unverified.
    """
    reviews: dict[str, list[dict]] = {}
    tips: dict[str, list[dict]] = {}
    for review in raw_reviews:
        reviews.setdefault(review["business_id"], []).append(review)
    for tip in raw_tips:
        tips.setdefault(tip["business_id"], []).append(tip)
    documents = build_documents(businesses, reviews, tips, narrow_config)
    return SearchPipeline(
        narrow_config,
        businesses={b.business_id: b for b in businesses},
        documents=documents,
        bm25_index=bm25_mod.build_index(documents, narrow_config),
    )


LEXICAL_ONLY = Variant(name="bm25_only", stages=Stage.BM25 | Stage.FILTERS)


# ─────────────────────────────────────────────────────────────────────────────
# The index itself
# ─────────────────────────────────────────────────────────────────────────────


def test_a_permanently_closed_business_is_never_indexed(
    index: ConstraintIndex, businesses: list[Business]
) -> None:
    """b004 has `is_open == 0`. This path has no gate in front of it, so it has
    to exclude closed businesses itself — showing one is the most embarrassing
    wrong answer available."""
    assert any(not b.is_open for b in businesses), "fixture must contain a closed business"
    assert "b004" not in {b.business_id for b in index.ordered}
    assert index.n_indexed == sum(1 for b in businesses if b.is_open)


@pytest.mark.parametrize(("query", "expected"), D224_TABLE)
def test_the_constraint_index_returns_everything_that_establishes_the_constraints(
    index: ConstraintIndex, narrow_config: Config, query: str, expected: tuple[str, ...]
) -> None:
    """The headline requirement: every business that genuinely satisfies the
    query's constraints must come back, and nothing refuted may.

    Asserted as a set, because *which* of them is first is the quality prior's
    job and is tested separately — mixing the two would make a reordering look
    like a recall bug.
    """
    parsed = parse(query)
    selected = index.select(parsed, index.n_indexed, narrow_config)
    ids = [business_id for business_id, _score in selected]

    established = set()
    for business_id in ids:
        matches = evaluate_constraints(index.by_id[business_id], parsed)
        if all(m.status is ConstraintStatus.SATISFIED for m in matches):
            established.add(business_id)
    assert established == set(expected)


@pytest.mark.parametrize(("query", "expected"), D224_TABLE)
def test_a_refuted_business_is_never_selected(
    index: ConstraintIndex, narrow_config: Config, query: str, expected: tuple[str, ...]
) -> None:
    """VIOLATED means the stored data contradicts the request. `apply_filters`
    drops such a candidate unconditionally, so selecting one here would only
    waste a slot — and under a looser downstream it would be a wrong answer."""
    del expected
    parsed = parse(query)
    for business_id, _score in index.select(parsed, index.n_indexed, narrow_config):
        matches = evaluate_constraints(index.by_id[business_id], parsed)
        assert not any(m.status is ConstraintStatus.VIOLATED for m in matches), business_id


def test_established_results_come_before_merely_unrefuted_ones(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """ "This place definitely has a patio" must outrank "this place might".

    That is the same judgement `missing_attribute_policy=penalize` makes in the
    fused score, and it is the whole reason the selection is in two groups rather
    than one sort by popularity.
    """
    parsed = parse("somewhere with a patio")
    ids = [b for b, _ in index.select(parsed, index.n_indexed, narrow_config)]

    n_unknown = [
        sum(
            1
            for m in evaluate_constraints(index.by_id[business_id], parsed)
            if m.status is ConstraintStatus.UNKNOWN
        )
        for business_id in ids
    ]
    pairs = list(zip(ids, n_unknown, strict=True))
    assert n_unknown == sorted(n_unknown), f"unknown counts not non-decreasing: {pairs}"
    # And the established group is exactly the two businesses that store True.
    assert [b for b, n in pairs if n == 0] == ["b001", "b006"]


def test_the_returned_score_is_monotone_with_the_returned_order(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """`search`/`select` promise "best first". A caller that min-max scales the
    scores — `fusion.weighted_sum_scores` does exactly that — would silently
    reorder the list if the score disagreed with the order."""
    for query, _expected in D224_TABLE:
        scores = [s for _b, s in index.select(parse(query), index.n_indexed, narrow_config)]
        assert scores == sorted(scores, reverse=True), query


def test_selection_is_empty_when_the_query_constrains_nothing(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """With no constraint there is nothing to select on, and this path has no
    relevance signal of its own. Returning a popularity list would answer a
    different question than the one asked — and would dilute every unconstrained
    query, which is the failure mode the design has to avoid."""
    parsed = parse("somewhere nice to sit and read for a while")
    assert parsed.n_hard_constraints == 0
    assert parsed.categories == ()
    assert index.select(parsed, 10, narrow_config) == []


def test_a_category_only_query_still_selects(index: ConstraintIndex, narrow_config: Config) -> None:
    """Guards a real discrepancy: `ParsedQuery.n_hard_constraints` does not count
    `categories`, but `filters.evaluate_constraints` does check them. So the
    gate in this module counts the fields *it* will check rather than reusing
    that property — a "simplification" to `n_hard_constraints` would silently
    switch the whole category path off.
    """
    parsed = ParsedQuery(raw="good pho", semantic_text="pho", categories=("Ramen",))
    assert parsed.n_hard_constraints == 0
    selected = index.select(parsed, index.n_indexed, narrow_config)
    assert selected, "a categories-only query must still reach the constraint index"
    assert selected[0][0] == "b002", "the Ramen-tagged business must come first"


def test_a_drift_between_the_field_list_and_the_checks_selects_nothing(
    index: ConstraintIndex, narrow_config: Config, monkeypatch, caplog
) -> None:
    """The field list in this module mirrors `evaluate_constraints` by hand.

    If it ever claims a constraint that function does not check, every business
    comes back unrefuted and the selection quietly becomes a popularity ranking
    for a query that constrained nothing — which would dilute exactly the
    queries this module exists to protect. So the drift is detected and the
    answer is "nothing", loudly.
    """
    monkeypatch.setattr(
        "concierge.retrieval.constraints_index.evaluate_constraints",
        lambda business, parsed: (),
    )
    with caplog.at_level("ERROR"):
        assert index.select(parse("cheap place"), 10, narrow_config) == []
    assert "produced no checks" in caplog.text


def test_top_k_is_respected_and_a_non_positive_k_returns_nothing(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    parsed = parse("cheap place")
    assert len(index.select(parsed, 3, narrow_config)) == 3
    assert index.select(parsed, 0, narrow_config) == []
    assert index.select(parsed, -1, narrow_config) == []


def test_selection_is_deterministic(index: ConstraintIndex, narrow_config: Config) -> None:
    """Ties are common (same stars, same review count) and dict iteration order
    is not a ranking. An unreproducible order would make an A/B difference in
    the ablation un-attributable, which is the same reason `fusion.fuse` breaks
    ties on business_id."""
    parsed = parse("cheap place")
    first = index.select(parsed, index.n_indexed, narrow_config)
    assert first == index.select(parsed, index.n_indexed, narrow_config)
    # And rebuilding the index from a differently-ordered input must not change it.
    shuffled = ConstraintIndex.build(reversed(index.ordered))
    assert shuffled.select(parsed, shuffled.n_indexed, narrow_config) == first


def test_an_empty_index_selects_nothing_rather_than_raising(narrow_config: Config) -> None:
    """An over-narrow metro scope is a legitimate state. The constraint path is
    a fallback and must never be the thing that breaks a search."""
    empty = ConstraintIndex.build([])
    assert empty.n_indexed == 0
    assert empty.select(parse("cheap place"), 10, narrow_config) == []


def test_overnight_hours_are_handled_because_the_checks_are_not_reimplemented(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """b002 is open Monday "17:0-2:0" — the interval crosses midnight.

    The point is not the calendar arithmetic (that is tested in
    `tests/test_hours.py`); it is that this module gets it right *for free*
    because it calls `filters.evaluate_constraints` instead of reimplementing
    the checks. b001 closes at 20:00 and must be refuted.
    """
    parsed = parse("open past 10pm")
    assert parsed.open_at == OpenAt(day=TODAY, minute_of_day=22 * 60)
    ids = [b for b, _ in index.select(parsed, index.n_indexed, narrow_config)]

    established = [
        business_id
        for business_id in ids
        if all(
            m.status is ConstraintStatus.SATISFIED
            for m in evaluate_constraints(index.by_id[business_id], parsed)
        )
    ]
    # b002's Monday interval is "17:0-2:0" and 22:00 falls inside it; b007's
    # "0:0-0:0" is Yelp's encoding for open 24 hours.
    assert set(established) == {"b002", "b006", "b007"}
    assert ids[: len(established)] == established, "established results must come first"
    assert "b001" not in ids, "b001 closes at 20:00 and is refuted"
    assert "b003" not in ids, "b003 closes at 21:00 on Monday and is refuted"


def test_a_geo_constraint_narrows_through_the_same_checks(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """Geo is in the index because `evaluate_constraints` checks it; a business
    outside the radius is refuted and must not be selected.

    Both directions are asserted, because a geo path that always returned
    nothing would pass the negative case alone.
    """
    anchor = index.ordered[0]
    assert anchor.latitude is not None and anchor.longitude is not None

    near = GeoConstraint(
        latitude=anchor.latitude, longitude=anchor.longitude, radius_km=1.0, label="here"
    )
    selected = index.select(
        ParsedQuery(raw="near here", semantic_text="", geo=near), index.n_indexed, narrow_config
    )
    assert selected and selected[0][0] == anchor.business_id

    far = GeoConstraint(latitude=0.0, longitude=0.0, radius_km=1.0, label="nowhere")
    assert (
        index.select(
            ParsedQuery(raw="near nowhere", semantic_text="", geo=far),
            index.n_indexed,
            narrow_config,
        )
        == []
    )


# ─────────────────────────────────────────────────────────────────────────────
# The quality prior
# ─────────────────────────────────────────────────────────────────────────────


def test_shrinkage_pulls_a_sparsely_reviewed_rating_towards_the_corpus_mean() -> None:
    """The property the prior exists for: with the same rating, the business
    with less evidence sits closer to the corpus mean."""
    mean, weight = 4.0, 100.0
    sparse = shrunk_rating(5.0, 2, corpus_mean=mean, weight=weight)
    dense = shrunk_rating(5.0, 5_000, corpus_mean=mean, weight=weight)
    assert mean < sparse < dense < 5.0
    # With equal evidence, the better rating still wins.
    assert shrunk_rating(4.5, 50, corpus_mean=mean, weight=weight) > shrunk_rating(
        4.0, 50, corpus_mean=mean, weight=weight
    )
    # At the shrinkage weight itself, own rating and corpus mean weigh equally.
    assert shrunk_rating(5.0, int(weight), corpus_mean=mean, weight=weight) == pytest.approx(
        (5.0 + mean) / 2
    )


def test_a_zero_shrinkage_weight_is_rejected() -> None:
    """It would collapse the formula to the raw rating and divide by zero for a
    business with no reviews, so `ConstraintIndex.build` floors it."""
    with pytest.raises(ValueError, match="positive"):
        shrunk_rating(4.0, 10, corpus_mean=4.0, weight=0.0)


def test_the_prior_is_not_raw_stars(index: ConstraintIndex) -> None:
    """On the fixture corpus the shrinkage actually changes the order: b003
    (higher stars, far fewer reviews) ends up *below* b001. If this test ever
    reads the same both ways, the prior has stopped doing anything."""
    actual = [b.business_id for b in index.ordered]
    by_raw_stars = [
        b.business_id for b in sorted(index.ordered, key=lambda b: (-b.stars, b.business_id))
    ]
    assert actual != by_raw_stars
    assert actual.index("b001") < actual.index("b003")
    assert by_raw_stars.index("b003") < by_raw_stars.index("b001")


# ─────────────────────────────────────────────────────────────────────────────
# The backfill: non-displacement
# ─────────────────────────────────────────────────────────────────────────────


def _fused(*pairs: tuple[str, float]) -> list[Candidate]:
    return [
        Candidate(business_id=bid, scores={SCORE_RRF: score}, ranks={RANK_RRF: rank})
        for rank, (bid, score) in enumerate(pairs, start=1)
    ]


def test_a_full_candidate_list_is_returned_untouched(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """The trigger compares against `candidates.final`, the number of results
    the user is actually promised: with a full page there is no empty-page
    failure to repair, so the backfill must do nothing at all."""
    fused = _fused(("b002", 0.03), ("b003", 0.02))
    assert len(fused) >= narrow_config.candidates.final
    out = backfill(fused, index=index, parsed=parse("free wifi"), cfg=narrow_config)
    assert out == fused


def test_an_unconstrained_query_is_returned_untouched(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    fused = _fused(("b002", 0.03))
    parsed = parse("somewhere nice to sit and read for a while")
    assert backfill(fused, index=index, parsed=parsed, cfg=narrow_config) == fused


def test_the_missing_data_penalty_may_reorder_the_two_blocks_and_that_is_the_policy(
    index: ConstraintIndex, narrow_config: Config, businesses: list[Business]
) -> None:
    """Pins the exact scope of the non-displacement guarantee.

    It holds on the fused Stage-2 score. `apply_filters` runs afterwards and,
    under `penalize`, discounts a retrieved candidate whose constraint data is
    merely *missing* — which can leave it below a backfilled candidate that
    establishes the constraint. That is D007's policy doing its job, not the
    backfill displacing anything: the retrieved candidate would have taken the
    same discount with no backfill at all. Asserted so that nobody later
    "fixes" it by exempting backfilled candidates from the ordering.
    """
    cfg = with_policy(narrow_config, "penalize")
    parsed = parse("somewhere with a patio")
    # b005 has no attributes at all, so its OutdoorSeating check is UNKNOWN and
    # the penalty applies to it; b001 stores True and is established.
    fused = _fused(("b005", 0.02))
    out = backfill(fused, index=index, parsed=parsed, cfg=cfg, min_candidates=5)
    assert out[0].business_id == "b005"
    assert "b001" in {c.business_id for c in out[1:]}

    report = apply_filters(out, {b.business_id: b for b in businesses}, parsed, cfg)
    order = [c.business_id for c in report.kept]
    assert order.index("b001") < order.index("b005")
    assert report.n_penalized >= 1


@pytest.mark.parametrize(("query", "expected"), D224_TABLE)
def test_backfilled_candidates_score_strictly_below_every_retrieved_one(
    index: ConstraintIndex, narrow_config: Config, query: str, expected: tuple[str, ...]
) -> None:
    """The non-displacement guarantee, asserted where it lives.

    `apply_filters` re-sorts survivors by `SCORE_RRF`, so appending is not
    enough — the backfill block has to be *scored* below the retrieved block.
    Asserted on the score, and on the retrieved prefix being identical.
    """
    del expected
    fused = _fused(("b002", 0.0328), ("b003", 0.0161))
    wide = narrow_config.model_copy(
        update={"candidates": narrow_config.candidates.model_copy(update={"final": 6})}
    )
    out = backfill(fused, index=index, parsed=parse(query), cfg=wide)

    assert len(out) > len(fused), "the backfill should have fired"
    assert out[: len(fused)] == fused, "retrieved candidates must be untouched, in order"

    floor = min(c.scores[SCORE_RRF] for c in fused)
    added = out[len(fused) :]
    for candidate in added:
        assert 0.0 < candidate.scores[SCORE_RRF] < floor, candidate
    # Strictly decreasing within the block, so `select`'s order survives the
    # re-sort in `apply_filters`.
    scores = [c.scores[SCORE_RRF] for c in added]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_backfilled_scores_stay_positive_so_the_penalty_cannot_invert_them(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """`apply_filters` under `penalize` multiplies `SCORE_RRF` by a factor below
    one. A negative encoding of "below everything retrieved" would move such a
    candidate *up*, inverting the penalty — so the encoding must be positive."""
    out = backfill([], index=index, parsed=parse("somewhere with a patio"), cfg=narrow_config)
    assert out, "with nothing retrieved the backfill must supply candidates"
    penalty = narrow_config.retrieval.missing_attribute_penalty
    assert 0.0 < penalty < 1.0
    for candidate in out:
        score = candidate.scores[SCORE_RRF]
        assert score > 0.0
        assert score * penalty < score


def test_a_retrieved_candidate_scoring_zero_does_not_crash_the_backfill(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """Under `fusion.method = "weighted_sum"` the worst document a system
    returned scores exactly 0.0 — that method cannot tell "returned last" from
    "absent", as `fusion.py` says itself. The backfill must still produce
    strictly positive, ordered scores rather than dividing by it.
    """
    fused = _fused(("b002", 0.0))
    out = backfill(fused, index=index, parsed=parse("free wifi"), cfg=narrow_config)
    assert out[0] == fused[0]
    added = [c.scores[SCORE_RRF] for c in out[1:]]
    assert added and all(s > 0.0 for s in added)
    assert added == sorted(added, reverse=True)


def test_the_backfill_never_duplicates_a_retrieved_business(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """b001 satisfies "free wifi" and would be `select`'s top hit, so this is the
    case that would duplicate. A repeated business_id would give one business two
    fusion ranks and show it twice on the page."""
    fused = _fused(("b001", 0.03))
    wide = narrow_config.model_copy(
        update={"candidates": narrow_config.candidates.model_copy(update={"final": 6})}
    )
    out = backfill(fused, index=index, parsed=parse("free wifi"), cfg=wide)
    ids = [c.business_id for c in out]
    assert len(ids) == len(set(ids))
    assert ids[0] == "b001"


def test_the_backfill_never_exceeds_the_stage_two_budget(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """The candidate widths are part of the latency analysis, so Stage 2 must
    not hand on more than `candidates.retrieve` however empty retrieval was."""
    tight = narrow_config.model_copy(
        update={
            "candidates": narrow_config.candidates.model_copy(
                update={"retrieve": 3, "rerank": 3, "final": 3}
            )
        }
    )
    out = backfill([], index=index, parsed=parse("cheap place"), cfg=tight)
    assert len(out) == 3


def test_every_backfilled_candidate_carries_a_rank_and_its_own_score(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """`RANK_RRF` because Stage 4 reads it as a feature and a missing value
    there is not the same as a bad rank; `SCORE_CONSTRAINT` so a reader can
    always tell a backfilled candidate from a retrieved one."""
    fused = _fused(("b002", 0.03))
    wide = narrow_config.model_copy(
        update={"candidates": narrow_config.candidates.model_copy(update={"final": 6})}
    )
    out = backfill(fused, index=index, parsed=parse("cheap place"), cfg=wide)
    added = out[len(fused) :]
    assert added
    assert [c.ranks[RANK_RRF] for c in added] == list(range(2, 2 + len(added)))
    assert all(SCORE_CONSTRAINT in c.scores for c in added)
    assert SCORE_CONSTRAINT not in out[0].scores


# ─────────────────────────────────────────────────────────────────────────────
# The backfill agrees with the filters
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("policy", ["neutral", "penalize", "exclude"])
@pytest.mark.parametrize(("query", "expected"), D224_TABLE)
def test_backfilled_candidates_survive_the_real_filters(
    index: ConstraintIndex,
    narrow_config: Config,
    businesses: list[Business],
    policy: str,
    query: str,
    expected: tuple[str, ...],
) -> None:
    """The test that matters most.

    A backfilled candidate that `apply_filters` then drops would leave the page
    empty again — the original bug, one stage later. Agreement is by
    construction (this module calls `evaluate_constraints`, the same function
    the filters call, and applies the same policy rule), and this asserts it
    against the real `apply_filters` under all three policies.
    """
    cfg = with_policy(narrow_config, policy)
    parsed = parse(query)
    added = backfill([], index=index, parsed=parsed, cfg=cfg)
    assert added, f"{query} under {policy} must produce candidates"

    report = apply_filters(added, {b.business_id: b for b in businesses}, parsed, cfg)
    assert report.dropped == {}, f"{policy}: filters dropped backfilled candidates"
    assert {c.business_id for c in report.kept} == {c.business_id for c in added}
    # Whatever the policy, everything that genuinely satisfies the query is there.
    assert set(expected) <= {c.business_id for c in report.kept}


def test_under_the_exclude_policy_only_established_results_are_offered(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """`exclude` drops a candidate whose UNKNOWN the policy *governs*, so
    offering one would hand the filters a candidate it is certain to delete.

    "somewhere with a patio" constrains only `OutdoorSeating`, an attribute the
    policy does govern, so here every UNKNOWN is governed and only the
    established group may be offered. The exempt case — a category miss, which
    `apply_filters` keeps under `exclude` — is the test below; this one is
    deliberately a query with no category constraint so the two rules are
    pinned separately rather than as one.
    """
    cfg = with_policy(narrow_config, "exclude")
    parsed = parse("somewhere with a patio")
    assert not parsed.categories, "this test is about governed unknowns only"
    ids = [b for b, _ in index.select(parsed, index.n_indexed, cfg)]
    assert ids == ["b001", "b006"]
    # The looser policies offer the same two first, then the unrefuted rest.
    neutral = [
        b for b, _ in index.select(parsed, index.n_indexed, with_policy(narrow_config, "neutral"))
    ]
    assert neutral[:2] == ids
    assert len(neutral) > len(ids)


@pytest.mark.parametrize("policy", ["exclude", "penalize", "neutral"])
def test_selection_admits_exactly_what_the_filters_keep_under_every_policy(
    index: ConstraintIndex,
    narrow_config: Config,
    businesses: list[Business],
    policy: str,
) -> None:
    """The agreement this module promises, asserted as set equality in both
    directions and under every policy.

    This is a regression test for a cross-module seam rather than for a single
    function. `filters.POLICY_EXEMPT_CONSTRAINTS` exempts a category miss from
    the missing-data policy, because "no category matched the words the parser
    emitted" is a fact about our matcher and Yelp's taxonomy rather than about
    the dataset's coverage of the business. `select` read *every* UNKNOWN as
    policy-governed, so under `exclude` it offered only the businesses whose
    categories matched lexically — six of the eight fixture businesses that
    `apply_filters` keeps for "ramen" were withheld. Nothing caught it: the
    existing agreement test checks only that nothing offered gets dropped, and
    being over-strict passes that.

    Why that mattered beyond recall: `exclude` is one of the three arms of the
    D007 missing-data comparison, so the shortfall would have been read as a
    property of the policy instead of as a disagreement between two modules.

    "ramen" is the query that exercises it — a category constraint no fixture
    business satisfies except `b002` — and the comparison is on sets because the
    two sides order by different things on purpose (`select` by its quality
    prior, `apply_filters` by the fused Stage-2 score).
    """
    cfg = with_policy(narrow_config, policy)
    by_id = {b.business_id: b for b in businesses}
    parsed = parse("ramen")
    assert parsed.categories, "this test needs a category constraint to be meaningful"

    offered = {b for b, _ in index.select(parsed, index.n_indexed, cfg)}
    # Everything the filters would keep if handed the whole open corpus.
    everything = [
        Candidate(business_id=b.business_id, scores={SCORE_RRF: 1.0})
        for b in businesses
        if b.is_open
    ]
    kept = {c.business_id for c in apply_filters(everything, by_id, parsed, cfg).kept}

    assert offered == kept, (
        f"policy={policy}: the constraint path and the filters disagree. "
        f"withheld={sorted(kept - offered)} offered-but-dropped={sorted(offered - kept)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The trigger
# ─────────────────────────────────────────────────────────────────────────────


def test_the_satisfaction_aware_trigger_fires_where_the_count_trigger_does_not(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """The documented limitation of the default trigger, pinned as behaviour.

    `b002` stores WiFi "no" and `b008` stores "paid", so a full page of
    retrieved candidates can contain nothing that establishes "free wifi". The
    count trigger stays quiet (the page is full); `TRIGGER_FEW_SATISFIED`
    notices. It is not the default because it changes the candidate set of every
    constrained query and that trade has not been measured against labels — so
    this test records what each one does, not which is better.
    """
    fused = _fused(("b002", 0.03), ("b008", 0.02))
    parsed = parse("free wifi")
    assert len(fused) >= narrow_config.candidates.final

    quiet = backfill(
        fused, index=index, parsed=parsed, cfg=narrow_config, trigger=TRIGGER_FEW_CANDIDATES
    )
    assert quiet == fused

    fired = backfill(
        fused, index=index, parsed=parsed, cfg=narrow_config, trigger=TRIGGER_FEW_SATISFIED
    )
    assert [c.business_id for c in fired[: len(fused)]] == ["b002", "b008"]
    assert "b001" in {c.business_id for c in fired[len(fused) :]}


def test_the_floor_is_overridable_for_a_sweep(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """`Config` is a frozen `extra="forbid"` contract owned elsewhere, so the
    floor cannot be a config key. An experiment that wants to sweep it passes
    it in, and `min_candidates=0` turns the backfill off entirely."""
    fused = _fused(("b002", 0.03))
    parsed = parse("cheap place")
    assert backfill(fused, index=index, parsed=parsed, cfg=narrow_config, min_candidates=0) == fused
    assert len(backfill(fused, index=index, parsed=parsed, cfg=narrow_config, min_candidates=5)) > 1


# ─────────────────────────────────────────────────────────────────────────────
# End to end, through the pipeline
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("query", "expected"), D224_TABLE)
def test_the_measured_defect_is_closed_end_to_end(
    lexical_pipeline: SearchPipeline, query: str, expected: tuple[str, ...]
) -> None:
    """The D224 table, run through the real pipeline on ablation row 1.

    Before the fix, `somewhere with a patio` and `cheap place` returned an empty
    list here: the rule parser stripped the constraint, the residue matched no
    document, and the correctly-parsed constraints had nothing to filter.
    """
    response = lexical_pipeline.search(query, variant=LEXICAL_ONLY)
    returned = [r.business.business_id for r in response.results]
    assert returned, f"{query!r} returned nothing"
    # Established results come first, so as far down the page as there are
    # businesses that genuinely satisfy the query, that is what the page holds.
    # (Written as a prefix rather than `set(returned) <= set(expected)` so it
    # stays correct if `candidates.final` is widened past len(expected), where
    # the merely-unrefuted group legitimately starts filling the tail.)
    head = min(len(expected), len(returned))
    assert set(returned[:head]) <= set(expected), f"{query!r} returned {returned}"


@pytest.mark.parametrize("policy", ["exclude", "penalize", "neutral"])
def test_a_constraint_only_query_returns_results_under_every_policy(
    narrow_config: Config,
    businesses: list[Business],
    raw_reviews: list[dict],
    raw_tips: list[dict],
    policy: str,
) -> None:
    """The D224 defect is "the page is empty", so the end-to-end property is
    that the page is *not* empty — and it has to hold in all three arms of the
    missing-data comparison, not just the default one.

    The test above runs only the default `penalize`. That left the `exclude`
    arm unpinned, which is where `select` and `apply_filters` had drifted apart
    over the category exemption: a constraint query naming a category came back
    short under `exclude` while every assertion in the file still passed.

    `quiet cafe with wifi` is the query that carries both kinds of constraint at
    once — two governed attributes and a category — so it is the one that fails
    if either rule is wrong. Under `exclude` only businesses whose attribute
    data is complete may appear; under the looser policies the unrefuted rest
    may fill the tail. Both are asserted, rather than just "non-empty", so this
    cannot pass by returning the wrong businesses.
    """
    reviews: dict[str, list[dict]] = {}
    tips: dict[str, list[dict]] = {}
    for review in raw_reviews:
        reviews.setdefault(review["business_id"], []).append(review)
    for tip in raw_tips:
        tips.setdefault(tip["business_id"], []).append(tip)
    cfg = with_policy(narrow_config, policy)
    documents = build_documents(businesses, reviews, tips, cfg)
    pipeline = SearchPipeline(
        cfg,
        businesses={b.business_id: b for b in businesses},
        documents=documents,
        bm25_index=bm25_mod.build_index(documents, cfg),
    )

    response = pipeline.search("quiet cafe with wifi", variant=LEXICAL_ONLY)
    returned = [r.business.business_id for r in response.results]

    assert returned, f"policy={policy}: a constraint-only query returned an empty page"
    # b001 and b003 are the two whose data establishes all three constraints
    # (NoiseLevel=quiet, WiFi=free, category Cafes); see D224_TABLE.
    assert set(returned[:2]) == {"b001", "b003"}, f"policy={policy}: got {returned}"
    if policy == "exclude":
        # b005 has no attributes at all, so `exclude` must not offer it.
        assert "b005" not in returned, f"policy={policy}: got {returned}"


def test_a_meaningful_query_keeps_its_lexical_match_first(
    lexical_pipeline: SearchPipeline,
) -> None:
    """The counterpart requirement: the fallback must not dilute real evidence.

    "ramen" is a literal token in exactly one fixture document and the query
    also carries a category constraint, so this is precisely the case where an
    over-eager constraint path would push a popular, constraint-satisfying but
    off-topic business above the right answer.
    """
    response = lexical_pipeline.search(MEANINGFUL_QUERY, variant=LEXICAL_ONLY)
    returned = [r.business.business_id for r in response.results]
    assert returned[0] == MEANINGFUL_QUERY_BEST


def test_a_query_retrieval_already_answers_is_byte_identical_with_and_without_the_index(
    lexical_pipeline: SearchPipeline,
    narrow_config: Config,
    businesses: list[Business],
) -> None:
    """The strongest form of "no dilution": for a query the lexical retriever
    already answers, substituting an empty constraint index changes nothing.

    `free wifi` survives its own stripping ("free wifi" is left in
    `semantic_text`) and matches four documents, so the trigger stays quiet and
    the two runs must agree exactly — ids *and* scores.
    """
    disabled = SearchPipeline(
        narrow_config,
        businesses={b.business_id: b for b in businesses},
        documents=lexical_pipeline.doc_text,
        bm25_index=lexical_pipeline.bm25_index,
        constraint_index=ConstraintIndex.build([]),
    )
    with_index = lexical_pipeline.search("free wifi", variant=LEXICAL_ONLY)
    without = disabled.search("free wifi", variant=LEXICAL_ONLY)
    assert [(r.business.business_id, r.score) for r in with_index.results] == [
        (r.business.business_id, r.score) for r in without.results
    ]


def test_the_no_filters_variant_does_not_receive_constraint_candidates(
    lexical_pipeline: SearchPipeline,
) -> None:
    """A `no_filters` row that was handed constraint-driven candidates would be
    measuring constraint handling under another name, and the row exists to
    measure its absence."""
    no_filters = Variant(name="no_filters_lexical", stages=Stage.BM25)
    response = lexical_pipeline.search("somewhere with a patio", variant=no_filters)
    assert response.n_candidates["retrieve"] == 0
    assert response.results == ()


def test_the_cascade_counts_stay_non_increasing_with_the_backfill(
    lexical_pipeline: SearchPipeline,
) -> None:
    """The backfill runs inside Stage 2, so its candidates are counted at
    `retrieve`. Counting them later would make a downstream stage appear to hand
    on more candidates than it received, which `tests/test_pipeline.py` asserts
    is a bug."""
    for query, _expected in D224_TABLE:
        response = lexical_pipeline.search(query, variant=LEXICAL_ONLY)
        counts = response.n_candidates
        assert counts["retrieve"] >= counts["filter"] >= counts["final"]
        assert counts["retrieve"] <= lexical_pipeline.cfg.candidates.retrieve


# ─────────────────────────────────────────────────────────────────────────────
# Interface conformance
# ─────────────────────────────────────────────────────────────────────────────


def test_the_bound_index_satisfies_the_retriever_protocol(
    index: ConstraintIndex, narrow_config: Config
) -> None:
    """D224's option (c) — fusing this list by RRF as a third system — stays
    reachable without touching this module, because the adapter matches
    `BM25Index` and `DenseIndex` exactly. The query text is accepted and
    ignored: a constraint has no query text, and pretending otherwise is what
    got this stage into trouble.
    """
    parsed = parse("cheap place")
    retriever = index.as_retriever(parsed, narrow_config)
    assert isinstance(retriever, Retriever)
    assert retriever.search("anything at all", 3) == retriever.search("", 3)
    assert retriever.search("ignored", 3) == index.select(parsed, 3, narrow_config)


def test_the_fixture_corpus_still_contains_the_quirks_these_tests_rely_on(
    businesses: list[Business], fixture_dir: Path
) -> None:
    """Pins the fixture facts the hand-worked `D224_TABLE` depends on, so a
    change to the corpus fails here with a readable message instead of failing
    a dozen expectation assertions."""
    by_id = {b.business_id: b for b in businesses}
    assert by_id["b002"].attributes["OutdoorSeating"] is False
    assert by_id["b002"].attributes["WiFi"] == "no"
    assert by_id["b008"].attributes["WiFi"] == "paid"
    assert str(by_id["b008"].attributes["RestaurantsPriceRange2"]) == "4"
    assert by_id["b005"].attributes == {}
    assert by_id["b005"].hours == {}
    # b008 stores the *string* "None" for OutdoorSeating, which must normalize
    # to missing rather than reading as "yes".
    raw = [
        json.loads(line)
        for line in (fixture_dir / "businesses.json").read_text().splitlines()
        if line.strip()
    ]
    assert {r["business_id"]: r for r in raw}["b008"]["attributes"]["OutdoorSeating"] == "None"
