"""Unit tests for the Stage 2 hard-constraint filters.

Two things these tests are careful about:

1. **Three-valued outcomes.** Every check is asserted in all three states where
   the fixture corpus supports it, because conflating UNKNOWN with VIOLATED (or
   with SATISFIED) is the failure this whole design exists to prevent.
2. **Not testing someone else's module.** The hours arithmetic lives in
   `concierge.data.hours` and is tested there. What is under test here is the
   *mapping* from its answer onto a ConstraintMatch and onto a keep/drop
   decision — so the hours cases below are chosen to exercise that mapping in
   all three states, not to re-verify the calendar maths.
"""

from __future__ import annotations

import ast
import re

import pytest

from concierge.retrieval.filters import (
    CONSTRAINT_CATEGORY,
    CONSTRAINT_GEO,
    CONSTRAINT_OPEN_AT,
    CONSTRAINT_PRICE_MAX,
    DROP_MISSING_DATA,
    DROP_PERMANENTLY_CLOSED,
    DROP_UNKNOWN_BUSINESS,
    DROP_VIOLATED,
    apply_filters,
    check_attribute,
    check_categories,
    check_geo,
    check_open_at,
    check_price_max,
    evaluate_constraints,
    haversine_km,
)
from concierge.types import (
    RANK_RRF,
    SCORE_RRF,
    Business,
    Candidate,
    ConstraintStatus,
    GeoConstraint,
    HoursInterval,
    OpenAt,
    ParsedQuery,
)

SATISFIED = ConstraintStatus.SATISFIED
VIOLATED = ConstraintStatus.VIOLATED
UNKNOWN = ConstraintStatus.UNKNOWN

_LITERAL = re.compile(r"^u?'(?P<inner>.*)'$")

# Attribute values deliberately left as raw strings instead of being normalized
# by the helper below, so the filters' own defensive handling is what gets
# tested. "None" is the dangerous one: it is truthy in Python, so a leaked value
# would read as "yes" and produce a confidently wrong answer.
_KEEP_RAW = {("b008", "OutdoorSeating")}


# ─────────────────────────────────────────────────────────────────────────────
# Building `Business` objects from the raw fixtures.
#
# The real loader is another module's job. This is the smallest normalization
# that satisfies the `Business` contract ("attributes holds normalized values
# only"), kept local so a filter test fails for filter reasons.
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_attribute(business_id: str, key: str, value: object) -> object:
    if (business_id, key) in _KEEP_RAW:
        return value
    if not isinstance(value, str):
        return value
    match = _LITERAL.match(value)
    if match:
        return match.group("inner")
    if value.startswith("{"):
        return ast.literal_eval(value)
    if value in {"True", "False"}:
        return value == "True"
    return value


def _parse_hours(hours: dict[str, str] | None) -> dict[str, tuple[HoursInterval, ...]]:
    """"7:0-20:0" -> one HoursInterval in minutes from midnight."""
    if not hours:
        return {}
    parsed: dict[str, tuple[HoursInterval, ...]] = {}
    for day, span in hours.items():
        start, end = span.split("-")
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        parsed[day] = (HoursInterval(sh * 60 + sm, eh * 60 + em),)
    return parsed


def _business(raw: dict) -> Business:
    attributes = {
        key: _normalize_attribute(raw["business_id"], key, value)
        for key, value in (raw["attributes"] or {}).items()
    }
    return Business(
        business_id=raw["business_id"],
        name=raw["name"],
        city=raw["city"],
        state=raw["state"],
        latitude=raw["latitude"],
        longitude=raw["longitude"],
        stars=raw["stars"],
        review_count=raw["review_count"],
        categories=tuple(c.strip() for c in raw["categories"].split(",")),
        attributes=attributes,
        hours=_parse_hours(raw["hours"]),
        is_open=bool(raw["is_open"]),
    )


@pytest.fixture
def businesses(raw_businesses: list[dict]) -> dict[str, Business]:
    return {b["business_id"]: _business(b) for b in raw_businesses}


# 01:00 on a Saturday, i.e. the morning half of Friday's overnight shift. This
# is the hour that separates a correct hours implementation from a naive one.
SAT_1AM = OpenAt(day="Saturday", minute_of_day=60)
MON_NOON = OpenAt(day="Monday", minute_of_day=12 * 60)
TUE_NOON = OpenAt(day="Tuesday", minute_of_day=12 * 60)


# ─────────────────────────────────────────────────────────────────────────────
# haversine
# ─────────────────────────────────────────────────────────────────────────────


def test_haversine_zero_distance() -> None:
    assert haversine_km(33.4484, -112.0740, 33.4484, -112.0740) == pytest.approx(0.0)


def test_haversine_one_degree_of_longitude_at_the_equator() -> None:
    """A known value: one degree along the equator is R * 1 degree in radians
    = 6371.0088 * 0.0174533 km."""
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, abs=0.01)


def test_haversine_is_symmetric() -> None:
    there = haversine_km(33.4484, -112.0740, 36.1699, -115.1398)
    back = haversine_km(36.1699, -115.1398, 33.4484, -112.0740)
    assert there == pytest.approx(back)


# ─────────────────────────────────────────────────────────────────────────────
# open_at — the overnight case is the one that matters
# ─────────────────────────────────────────────────────────────────────────────


def test_b002_is_open_at_1am_via_its_overnight_interval(businesses) -> None:
    """Midnight Ramen's Friday hours are "17:0-3:0", i.e. 5pm Friday to 3am
    Saturday, so it is open at Saturday 01:00. Naive endpoint comparison would
    call that closed, wrongly dropping exactly the businesses an "open late"
    query is asking for."""
    match = check_open_at(businesses["b002"], SAT_1AM)
    assert match.status is SATISFIED
    assert match.name == CONSTRAINT_OPEN_AT


def test_b001_is_closed_at_1am(businesses) -> None:
    """Lantern Coffee closes at 23:00 on Friday and opens at 08:00 on Saturday,
    so Saturday 01:00 is a genuine VIOLATED — the data says no, which is
    different from the data being absent."""
    assert check_open_at(businesses["b001"], SAT_1AM).status is VIOLATED


def test_b001_is_open_at_noon(businesses) -> None:
    assert check_open_at(businesses["b001"], MON_NOON).status is SATISFIED


def test_b005_open_at_is_unknown_not_closed(businesses) -> None:
    """No Data Bistro has `hours: null`. Treating that as closed would delete
    every business with missing hours from every time-constrained query, and
    time-constrained queries are a whole query_type in the gold set."""
    match = check_open_at(businesses["b005"], SAT_1AM)
    assert match.status is UNKNOWN
    assert match.actual is None


def test_b007_open_24_hours(businesses) -> None:
    """Yelp encodes 24-hour operation as "0:0-0:0"."""
    assert check_open_at(businesses["b007"], SAT_1AM).status is SATISFIED
    assert check_open_at(businesses["b007"], MON_NOON).status is SATISFIED


def test_b003_missing_weekday_is_closed_not_unknown(businesses) -> None:
    """The Quiet Shelf lists Monday, Saturday and Sunday only. It has hours
    data; Tuesday is simply not one of them, which is "closed", not "no data".
    Only a business with *no* hours dict at all is UNKNOWN."""
    assert check_open_at(businesses["b003"], TUE_NOON).status is VIOLATED


# ─────────────────────────────────────────────────────────────────────────────
# price
# ─────────────────────────────────────────────────────────────────────────────


def test_price_max_is_an_inequality(businesses) -> None:
    """Someone asking for cheap ramen is not annoyed by cheaper ramen."""
    assert check_price_max(businesses["b002"], 1).status is SATISFIED  # $ <= $
    assert check_price_max(businesses["b002"], 3).status is SATISFIED  # $ <= $$$
    assert check_price_max(businesses["b001"], 2).status is SATISFIED  # $$ <= $$


def test_price_max_violation(businesses) -> None:
    match = check_price_max(businesses["b008"], 2)
    assert match.status is VIOLATED
    assert match.actual == 4


def test_price_max_unknown_when_absent(businesses) -> None:
    match = check_price_max(businesses["b005"], 2)
    assert match.status is UNKNOWN
    assert match.actual is None


def test_unparseable_price_is_unknown() -> None:
    weird = Business(
        business_id="x",
        name="x",
        city="Metroville",
        state="AZ",
        latitude=0.0,
        longitude=0.0,
        stars=3.0,
        review_count=1,
        categories=("Cafes",),
        attributes={"RestaurantsPriceRange2": "cheap-ish"},
        hours={},
        is_open=True,
    )
    assert check_price_max(weird, 2).status is UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# attributes
# ─────────────────────────────────────────────────────────────────────────────


def test_string_attribute_match_and_violation(businesses) -> None:
    assert check_attribute(businesses["b001"], "WiFi", "free").status is SATISFIED
    assert check_attribute(businesses["b002"], "WiFi", "free").status is VIOLATED
    assert check_attribute(businesses["b008"], "WiFi", "free").status is VIOLATED


def test_string_attribute_match_is_case_insensitive(businesses) -> None:
    assert check_attribute(businesses["b001"], "NoiseLevel", "QUIET").status is SATISFIED


def test_boolean_attribute(businesses) -> None:
    assert check_attribute(businesses["b001"], "OutdoorSeating", True).status is SATISFIED
    assert check_attribute(businesses["b002"], "OutdoorSeating", True).status is VIOLATED
    assert check_attribute(businesses["b002"], "OutdoorSeating", False).status is SATISFIED


def test_b008_none_string_is_unknown_not_false(businesses) -> None:
    """b008 stores OutdoorSeating as the *string* "None".

    The string "None" is truthy in Python, so a naive boolean read reports
    "yes, outdoor seating" — a confidently wrong answer, which is worse than no
    answer. It must come back UNKNOWN, and it must not be reported as False
    either, because that would be an equally invented fact.
    """
    match = check_attribute(businesses["b008"], "OutdoorSeating", True)
    assert match.status is UNKNOWN
    assert match.actual is None
    assert check_attribute(businesses["b008"], "OutdoorSeating", False).status is UNKNOWN


def test_missing_attribute_key_is_unknown(businesses) -> None:
    assert check_attribute(businesses["b005"], "WiFi", "free").status is UNKNOWN
    assert check_attribute(businesses["b002"], "Ambience", "casual").status is UNKNOWN


def test_nested_dict_attribute(businesses) -> None:
    """b001's Ambience is "{'romantic': False, 'intimate': False, 'casual': True}".

    A request naming one sub-key must check that sub-key, and a sub-key the dict
    does not mention is UNKNOWN rather than False — the dict is not a complete
    description of the venue.
    """
    assert check_attribute(businesses["b001"], "Ambience", "casual").status is SATISFIED
    assert check_attribute(businesses["b001"], "Ambience", "romantic").status is VIOLATED
    assert check_attribute(businesses["b001"], "Ambience", "trendy").status is UNKNOWN


def test_nested_dict_attribute_with_dict_request(businesses) -> None:
    biz = businesses["b001"]
    assert check_attribute(biz, "Ambience", {"casual": True}).status is SATISFIED
    assert check_attribute(biz, "Ambience", {"casual": True, "romantic": True}).status is VIOLATED
    assert check_attribute(biz, "Ambience", {"casual": True, "trendy": True}).status is UNKNOWN


def test_shape_mismatch_is_unknown_not_a_guess(businesses) -> None:
    """Requested booleans against a stored vocabulary word ("free"/"paid"/"no")
    cannot be compared without guessing. Emitting values in the dataset's own
    vocabulary is the parser's job; this returns UNKNOWN rather than inventing
    a mapping here, where it would be invisible."""
    assert check_attribute(businesses["b001"], "WiFi", True).status is UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# categories
# ─────────────────────────────────────────────────────────────────────────────


def test_category_match_is_separator_and_case_insensitive(businesses) -> None:
    biz = businesses["b001"]  # "Coffee & Tea, Cafes, Breakfast & Brunch"
    for requested in ("Coffee & Tea", "coffee and tea", "COFFEE-TEA", "coffeetea"):
        assert check_categories(biz, (requested,)).status is SATISFIED, requested


def test_category_match_accepts_a_token_subset(businesses) -> None:
    """An LLM parser emits "coffee", not Yelp's verbatim "Coffee & Tea"."""
    assert check_categories(businesses["b001"], ("coffee",)).status is SATISFIED
    assert check_categories(businesses["b002"], ("ramen",)).status is SATISFIED


def test_categories_are_any_of_not_all_of(businesses) -> None:
    """A parser asked about ramen may emit several near-synonyms; requiring all
    of them would reject the obvious answer."""
    match = check_categories(businesses["b002"], ("Ramen", "Pho", "Udon"))
    assert match.status is SATISFIED
    assert match.actual == "Ramen"


def test_category_violation(businesses) -> None:
    assert check_categories(businesses["b001"], ("Ramen",)).status is VIOLATED


def test_category_unknown_when_the_business_has_none() -> None:
    biz = Business(
        business_id="x",
        name="x",
        city="Metroville",
        state="AZ",
        latitude=None,
        longitude=None,
        stars=3.0,
        review_count=1,
        categories=(),
        attributes={},
        hours={},
        is_open=True,
    )
    assert check_categories(biz, ("Cafes",)).status is UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# geo
# ─────────────────────────────────────────────────────────────────────────────


def test_geo_inside_radius(businesses) -> None:
    biz = businesses["b001"]
    here = GeoConstraint(latitude=biz.latitude, longitude=biz.longitude, radius_km=1.0)
    match = check_geo(biz, here)
    assert match.status is SATISFIED
    assert match.name == CONSTRAINT_GEO
    assert match.actual == pytest.approx(0.0)


def test_geo_outside_radius_is_violated(businesses) -> None:
    """b002 is a bit over a kilometre from b001, so the same point with a tight
    radius is a real VIOLATED and with a loose one is SATISFIED."""
    origin = businesses["b001"]
    far = businesses["b002"]
    distance = haversine_km(far.latitude, far.longitude, origin.latitude, origin.longitude)
    tight = GeoConstraint(origin.latitude, origin.longitude, radius_km=distance / 2)
    loose = GeoConstraint(origin.latitude, origin.longitude, radius_km=distance * 2)
    assert check_geo(far, tight).status is VIOLATED
    assert check_geo(far, loose).status is SATISFIED


def test_geo_without_coordinates_is_unknown_never_violated() -> None:
    """A business with no coordinates must not be dropped *by the geo check*.
    What happens to it is the missing-data policy's call, which is where that
    decision is measured."""
    biz = Business(
        business_id="x",
        name="x",
        city="Metroville",
        state="AZ",
        latitude=None,
        longitude=None,
        stars=3.0,
        review_count=1,
        categories=("Cafes",),
        attributes={},
        hours={},
        is_open=True,
    )
    match = check_geo(biz, GeoConstraint(33.4484, -112.0740, radius_km=1.0))
    assert match.status is UNKNOWN
    assert match.actual is None


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_constraints
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_returns_one_match_per_constraint(businesses) -> None:
    parsed = ParsedQuery(
        raw="quiet cafe with free wifi, cheap, open at noon, near here",
        semantic_text="quiet cafe with free wifi",
        open_at=MON_NOON,
        attributes={"WiFi": "free", "NoiseLevel": "quiet"},
        price_max=2,
        categories=("Cafes",),
        geo=GeoConstraint(33.4484, -112.0740, radius_km=5.0),
    )
    matches = evaluate_constraints(businesses["b001"], parsed)
    assert {m.name for m in matches} == {
        CONSTRAINT_OPEN_AT,
        CONSTRAINT_PRICE_MAX,
        "WiFi",
        "NoiseLevel",
        CONSTRAINT_CATEGORY,
        CONSTRAINT_GEO,
    }
    assert all(m.status is SATISFIED for m in matches)
    assert len(matches) == parsed.n_hard_constraints + 1  # +1: categories


def test_b005_is_unknown_for_every_checkable_constraint(businesses) -> None:
    """No Data Bistro has `attributes: null` and `hours: null`. Every constraint
    that depends on that data must be UNKNOWN — not one of them may come back
    SATISFIED or VIOLATED."""
    parsed = ParsedQuery(
        raw="cheap quiet place with wifi and outdoor seating open at 1am",
        semantic_text="cheap quiet place with wifi and outdoor seating",
        open_at=SAT_1AM,
        attributes={"WiFi": "free", "NoiseLevel": "quiet", "OutdoorSeating": True},
        price_max=1,
    )
    matches = evaluate_constraints(businesses["b005"], parsed)
    assert len(matches) == 5
    assert all(m.status is UNKNOWN for m in matches)


def test_price_is_not_checked_twice(businesses) -> None:
    """`price_max` is an inequality; the same key arriving in `attributes` would
    be compared for equality and contradict it, so the dedicated field wins."""
    parsed = ParsedQuery(
        raw="cheap",
        semantic_text="cheap",
        price_max=3,
        attributes={"RestaurantsPriceRange2": 3},
    )
    matches = evaluate_constraints(businesses["b001"], parsed)
    assert [m.name for m in matches] == [CONSTRAINT_PRICE_MAX]
    assert matches[0].status is SATISFIED  # $$ <= $$$, not "== $$$"


def test_no_constraints_means_no_matches(businesses) -> None:
    parsed = ParsedQuery(raw="somewhere nice", semantic_text="somewhere nice")
    assert evaluate_constraints(businesses["b001"], parsed) == ()


# ─────────────────────────────────────────────────────────────────────────────
# apply_filters: the keep/drop decision and the three missing-data policies
# ─────────────────────────────────────────────────────────────────────────────


def _candidates(*pairs: tuple[str, float]) -> list[Candidate]:
    """Candidates as fusion would hand them over: 1-based RRF ranks."""
    return [
        Candidate(business_id=bid, scores={SCORE_RRF: score}, ranks={RANK_RRF: i})
        for i, (bid, score) in enumerate(pairs, start=1)
    ]


OUTDOOR_QUERY = ParsedQuery(
    raw="somewhere with outdoor seating",
    semantic_text="somewhere with outdoor seating",
    attributes={"OutdoorSeating": True},
)

# b001 True (SATISFIED) · b002 False (VIOLATED) · b005 no attributes (UNKNOWN)
# · b008 the string "None" (UNKNOWN). One query, all three statuses present.
POLICY_CANDIDATES = (("b001", 0.9), ("b002", 0.8), ("b005", 0.7), ("b008", 0.6))


def _policy_config(config, policy: str, penalty: float = 0.5):
    retrieval = config.retrieval.model_copy(
        update={"missing_attribute_policy": policy, "missing_attribute_penalty": penalty}
    )
    return config.model_copy(update={"retrieval": retrieval})


def test_violated_constraint_always_removes_the_business(config, businesses) -> None:
    """b002's OutdoorSeating is False. No policy and no ranking signal may
    rescue it: the data says it fails."""
    for policy in ("exclude", "penalize", "neutral"):
        report = apply_filters(
            _candidates(*POLICY_CANDIDATES),
            businesses,
            OUTDOOR_QUERY,
            _policy_config(config, policy),
        )
        assert "b002" not in {c.business_id for c in report.kept}, policy
        assert report.dropped["b002"] == DROP_VIOLATED, policy


def test_policy_exclude_drops_unknowns(config, businesses) -> None:
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "exclude"),
    )
    assert [c.business_id for c in report.kept] == ["b001"]
    assert report.dropped["b005"] == DROP_MISSING_DATA
    assert report.dropped["b008"] == DROP_MISSING_DATA
    assert report.n_penalized == 0


def test_policy_penalize_keeps_unknowns_with_a_multiplied_score(config, businesses) -> None:
    """0.9 stays; 0.7 -> 0.35 and 0.6 -> 0.30. The penalty does not reorder
    these three, but it does pull them below anything scoring above 0.35."""
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "penalize", penalty=0.5),
    )
    kept = {c.business_id: c.scores[SCORE_RRF] for c in report.kept}
    assert kept["b001"] == pytest.approx(0.9)
    assert kept["b005"] == pytest.approx(0.35)
    assert kept["b008"] == pytest.approx(0.30)
    assert report.n_penalized == 2
    assert not report.dropped.keys() - {"b002"}


def test_policy_neutral_leaves_scores_untouched(config, businesses) -> None:
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "neutral"),
    )
    kept = {c.business_id: c.scores[SCORE_RRF] for c in report.kept}
    assert kept == {
        "b001": pytest.approx(0.9),
        "b005": pytest.approx(0.7),
        "b008": pytest.approx(0.6),
    }
    assert report.n_penalized == 0


def test_penalty_can_reorder_results(config, businesses) -> None:
    """The point of `penalize`: a well-retrieved business with no data must be
    able to fall below a worse-retrieved business that provably satisfies the
    constraint. b005 (UNKNOWN, 0.9) halves to 0.45, below b001 (SATISFIED,
    0.5)."""
    report = apply_filters(
        _candidates(("b005", 0.9), ("b001", 0.5)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "penalize", penalty=0.5),
    )
    assert [c.business_id for c in report.kept] == ["b001", "b005"]
    assert [c.ranks[RANK_RRF] for c in report.kept] == [1, 2]


def test_a_penalty_of_one_is_equivalent_to_neutral(config, businesses) -> None:
    """A sanity property, and the identity the penalty sweep starts from."""
    candidates = _candidates(*POLICY_CANDIDATES)
    penalized = apply_filters(
        candidates, businesses, OUTDOOR_QUERY, _policy_config(config, "penalize", penalty=1.0)
    )
    neutral = apply_filters(
        candidates, businesses, OUTDOOR_QUERY, _policy_config(config, "neutral")
    )
    assert [(c.business_id, c.scores[SCORE_RRF]) for c in penalized.kept] == [
        (c.business_id, c.scores[SCORE_RRF]) for c in neutral.kept
    ]


def test_b004_is_dropped_as_permanently_closed(config, businesses) -> None:
    """`is_open == 0`. The index build already excludes these; this is the
    second line of defence, because showing a closed business is the most
    embarrassing possible wrong answer. Note it is dropped under `neutral`
    too — it is not a missing-data case."""
    parsed = ParsedQuery(raw="diner", semantic_text="diner")
    report = apply_filters(
        _candidates(("b004", 0.9), ("b001", 0.1)),
        businesses,
        parsed,
        _policy_config(config, "neutral"),
    )
    assert [c.business_id for c in report.kept] == ["b001"]
    assert report.dropped["b004"] == DROP_PERMANENTLY_CLOSED


def test_candidate_without_a_business_record_is_dropped(config, businesses) -> None:
    """An id with no Business is a data bug — we could neither check it nor
    render it, so it must not reach the user."""
    parsed = ParsedQuery(raw="anything", semantic_text="anything")
    report = apply_filters(
        _candidates(("ghost", 0.9), ("b001", 0.1)), businesses, parsed, config
    )
    assert [c.business_id for c in report.kept] == ["b001"]
    assert report.dropped["ghost"] == DROP_UNKNOWN_BUSINESS


def test_constraints_are_attached_to_survivors(config, businesses) -> None:
    """The "why this result" UI and the Stage 4 features both read these off
    the candidate, so they have to survive filtering."""
    report = apply_filters(
        _candidates(("b001", 0.9)), businesses, OUTDOOR_QUERY, _policy_config(config, "neutral")
    )
    (candidate,) = report.kept
    assert [c.name for c in candidate.constraints] == ["OutdoorSeating"]
    assert candidate.constraints[0].status is SATISFIED


def test_ranks_are_recomputed_contiguously(config, businesses) -> None:
    """Filtering is part of Stage 2, so its output ranks are the post-filter
    ranks. Leaving a gap where b002 was would hand Stage 4 a "rank" feature
    that corresponds to no list that ever existed."""
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "neutral"),
    )
    assert [c.ranks[RANK_RRF] for c in report.kept] == [1, 2, 3]


def test_upstream_scores_and_ranks_are_preserved(config, businesses) -> None:
    """Only the fused score is touched. Stage 4 needs the per-system numbers as
    features, so nothing else may be rewritten here."""
    candidate = Candidate(
        business_id="b005",
        scores={SCORE_RRF: 0.8, "bm25": 7.5, "dense": 0.61},
        ranks={RANK_RRF: 1, "bm25": 3, "dense": 2},
    )
    report = apply_filters(
        [candidate], businesses, OUTDOOR_QUERY, _policy_config(config, "penalize", penalty=0.5)
    )
    (kept,) = report.kept
    assert kept.scores["bm25"] == pytest.approx(7.5)
    assert kept.scores["dense"] == pytest.approx(0.61)
    assert kept.scores[SCORE_RRF] == pytest.approx(0.4)
    assert kept.ranks["bm25"] == 3
    assert kept.ranks["dense"] == 2


def test_ties_break_deterministically(config, businesses) -> None:
    """The penalty creates exact ties, so an unreproducible order would make an
    A/B difference un-attributable."""
    forward = apply_filters(
        _candidates(("b005", 0.6), ("b008", 0.6)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "penalize"),
    )
    backward = apply_filters(
        _candidates(("b008", 0.6), ("b005", 0.6)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "penalize"),
    )
    assert [c.business_id for c in forward.kept] == ["b005", "b008"]
    assert [c.business_id for c in backward.kept] == ["b005", "b008"]


def test_drop_counts_summarize_the_reasons(config, businesses) -> None:
    """Needed by the failure analysis: "this query returned three results" has
    to come with an explanation."""
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES, ("b004", 0.5), ("ghost", 0.4)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "exclude"),
    )
    assert report.drop_counts == {
        DROP_VIOLATED: 1,
        DROP_MISSING_DATA: 2,
        DROP_PERMANENTLY_CLOSED: 1,
        DROP_UNKNOWN_BUSINESS: 1,
    }


def test_an_unconstrained_query_keeps_everything_open(config, businesses) -> None:
    """A query with no hard constraints has no UNKNOWNs to police, so the
    policy must be inert — only the permanently closed b004 goes."""
    parsed = ParsedQuery(raw="somewhere nice", semantic_text="somewhere nice")
    for policy in ("exclude", "penalize", "neutral"):
        report = apply_filters(
            _candidates(*[(bid, 1.0) for bid in businesses]),
            businesses,
            parsed,
            _policy_config(config, policy),
        )
        assert len(report.kept) == len(businesses) - 1, policy
        assert report.drop_counts == {DROP_PERMANENTLY_CLOSED: 1}, policy
