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

from concierge.data.attributes import ATTRIBUTE_PATH_SEP
from concierge.retrieval.filters import (
    CONSTRAINT_CATEGORY,
    CONSTRAINT_GEO,
    CONSTRAINT_OPEN_AT,
    CONSTRAINT_PRICE_MAX,
    DROP_MISSING_DATA,
    DROP_PERMANENTLY_CLOSED,
    DROP_UNKNOWN_BUSINESS,
    DROP_VIOLATED,
    POLICY_EXEMPT_CONSTRAINTS,
    apply_filters,
    check_attribute,
    check_categories,
    check_geo,
    check_open_at,
    check_price_max,
    drop_reason,
    evaluate_constraints,
    haversine_km,
    reason_constraint,
    reason_kind,
)
from concierge.types import (
    RANK_CROSS_ENCODER,
    RANK_RRF,
    SCORE_CROSS_ENCODER,
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
    if value in {"True", "False"}:
        return value == "True"
    return value


def _flat_attributes(raw: dict) -> dict[str, object]:
    """Normalize one fixture's attributes the way the data layer does.

    The important part is the flattening: `concierge.data.attributes` turns a
    nested blob ("{'casual': True}") into dotted leaf keys ("Ambience.casual")
    and keeps no parent key, so `Business.attributes` never contains a dict.
    These fixtures must obey that contract, or a filter test would exercise a
    code path the real data cannot reach — which is exactly how the dead
    nested-dict branch in `_compare_attribute` survived as long as it did.
    """
    flat: dict[str, object] = {}
    for key, value in (raw["attributes"] or {}).items():
        if isinstance(value, str) and value.startswith("{"):
            for sub_key, sub_value in ast.literal_eval(value).items():
                flat[f"{key}{ATTRIBUTE_PATH_SEP}{sub_key}"] = sub_value
            continue
        flat[key] = _normalize_attribute(raw["business_id"], key, value)
    return flat


def _parse_hours(hours: dict[str, str] | None) -> dict[str, tuple[HoursInterval, ...]]:
    """ "7:0-20:0" -> one HoursInterval in minutes from midnight."""
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
    attributes = _flat_attributes(raw)
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
    # b002 has no Ambience data at all, so neither the leaf nor anything else
    # under it is known.
    assert check_attribute(businesses["b002"], "Ambience.casual", True).status is UNKNOWN


def test_nested_attributes_arrive_flattened_and_are_not_filterable(businesses) -> None:
    """b001's raw Ambience is "{'romantic': False, 'intimate': False, 'casual': True}".

    The data layer flattens it to dotted leaf keys, so the parent key is gone by
    the time a filter sees it. The leaves *are* comparable if something asks for
    one, which is asserted below so the flattening contract is pinned — but
    nothing can ask: `concierge.query.schema.SUPPORTED_ATTRIBUTES` lists no
    dotted key, so a `ParsedQuery` cannot name one. That is the honest state of
    ambience in this system: semantic, via the business document, not a hard
    constraint. The next test pins the half that is enforced.
    """
    biz = businesses["b001"]
    assert "Ambience" not in biz.attributes
    assert biz.attributes[f"Ambience{ATTRIBUTE_PATH_SEP}casual"] is True

    # Reachable only from a hand-written key, but it must behave if it is used.
    assert check_attribute(biz, "Ambience.casual", True).status is SATISFIED
    assert check_attribute(biz, "Ambience.romantic", True).status is VIOLATED
    # A leaf the raw dict never mentioned is simply an absent key: UNKNOWN, not
    # False. The dict was never a complete description of the venue.
    assert check_attribute(biz, "Ambience.trendy", True).status is UNKNOWN


def test_the_parser_cannot_request_a_nested_attribute(businesses) -> None:
    """The other half of the claim above, asserted against the schema itself so
    that adding a dotted key to `SUPPORTED_ATTRIBUTES` without revisiting the
    filters fails here rather than quietly changing behaviour."""
    from concierge.query.schema import SUPPORTED_ATTRIBUTES

    assert not [key for key in SUPPORTED_ATTRIBUTES if ATTRIBUTE_PATH_SEP in key]


def test_a_dict_attribute_value_is_a_loud_contract_breach(businesses) -> None:
    """If a dict ever reaches the filters, the data layer's flattening broke.

    It must raise rather than fall through to the string comparison, which would
    compare a dict's repr to a vocabulary word, get False, and drop the business
    as VIOLATED — a confident wrong answer with no trace. Same reasoning on the
    requested side: a parser that emitted a dict is broken, and hiding it behind
    an UNKNOWN would leave a broken parser looking healthy.
    """
    biz = businesses["b001"]
    leaked = Business(
        business_id=biz.business_id,
        name=biz.name,
        city=biz.city,
        state=biz.state,
        latitude=biz.latitude,
        longitude=biz.longitude,
        stars=biz.stars,
        review_count=biz.review_count,
        categories=biz.categories,
        attributes={"Ambience": {"casual": True}},
        hours=biz.hours,
        is_open=biz.is_open,
    )
    with pytest.raises(TypeError, match="flat"):
        check_attribute(leaked, "Ambience", "casual")
    with pytest.raises(TypeError, match="flat"):
        check_attribute(biz, "WiFi", {"free": True})


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


def test_a_category_miss_is_unknown_never_violated(businesses) -> None:
    """A lexical miss must not be VIOLATED, because VIOLATED is an unconditional
    drop and this matcher is not entitled to one.

    b001 is a coffee shop, so "Ramen" is about as clear a miss as the fixture
    corpus offers — and even here the only true statement is "this filter could
    not establish a match", not "the data says no". `actual` carries the
    business's own categories so the explanation can say what we matched
    against.
    """
    match = check_categories(businesses["b001"], ("Ramen",))
    assert match.status is UNKNOWN
    assert match.actual == businesses["b001"].categories


def test_a_taxonomy_gap_does_not_delete_the_right_answer(config) -> None:
    """The recall bug this status change exists to prevent.

    Yelp category lists are incomplete, and the matcher is lexical. A ramen shop
    tagged only "Japanese, Noodles" misses a request for "Ramen" — so if the
    miss were VIOLATED, the single best answer to "cheap ramen" would be deleted
    from the candidate list before any ranker could see it, and the dense
    retriever named as the mitigation in `check_categories` could not rescue it.
    Ten plausible results would come back without it, which is the hardest kind
    of failure to notice.
    """
    shop = Business(
        business_id="noodle-bar",
        name="Noodle Bar",
        city="Metroville",
        state="AZ",
        latitude=33.4484,
        longitude=-112.0740,
        stars=4.5,
        review_count=200,
        categories=("Japanese", "Noodles"),
        attributes={"RestaurantsPriceRange2": "1"},
        hours={},
        is_open=True,
    )
    parsed = ParsedQuery(raw="cheap ramen", semantic_text="ramen", categories=("Ramen",))
    assert check_categories(shop, parsed.categories).status is UNKNOWN

    # And it survives the keep/drop decision under the default policy instead of
    # being dropped unconditionally.
    report = apply_filters(_candidates(("noodle-bar", 0.9)), {"noodle-bar": shop}, parsed, config)
    assert [c.business_id for c in report.kept] == ["noodle-bar"]


def test_a_category_miss_is_exempt_from_the_missing_data_policy(config) -> None:
    """A lexical category miss must not route through `missing_attribute_policy`.

    This is the confound the three-way D007 comparison cannot be run with. The
    shop below has *complete* attribute data for everything the query asks about
    — there is no missing attribute anywhere on it — and fails only the lexical
    category match, which `check_categories` returns as UNKNOWN by design. If
    that UNKNOWN reached the policy, then under `exclude` the shop would be
    counted as removed by the missing-data policy and under `penalize` it would
    be counted in `n_penalized`, and the knob would be partly measuring gaps in
    Yelp's taxonomy rather than gaps in the data.

    So: survives under all three policies, discounted under none of them, and
    the miss is still visible — in `unknown_by_constraint`, as its own number.
    """
    shop = Business(
        business_id="noodle-bar",
        name="Noodle Bar",
        city="Metroville",
        state="AZ",
        latitude=33.4484,
        longitude=-112.0740,
        stars=4.5,
        review_count=200,
        categories=("Japanese", "Noodles"),
        attributes={"RestaurantsPriceRange2": "1"},
        hours={},
        is_open=True,
    )
    parsed = ParsedQuery(
        raw="cheap ramen", semantic_text="ramen", price_max=2, categories=("Ramen",)
    )
    for policy in ("exclude", "penalize", "neutral"):
        report = apply_filters(
            _candidates(("noodle-bar", 0.9)),
            {"noodle-bar": shop},
            parsed,
            _policy_config(config, policy),
        )
        assert [c.business_id for c in report.kept] == ["noodle-bar"], policy
        assert report.dropped == {}, policy
        assert report.n_penalized == 0, policy
        assert report.penalized_by_constraint == {}, policy
        # Exempt from the policy, not invisible to the report.
        assert report.unknown_by_constraint == {CONSTRAINT_CATEGORY: 1}, policy
        assert report.kept[0].scores[SCORE_RRF] == pytest.approx(0.9), policy


def test_category_is_the_only_exempt_constraint() -> None:
    """Pins the exemption list so it cannot grow by accident.

    Geo in particular is deliberately *not* exempt: "this business has no
    coordinates" is a coverage fact like a missing attribute, so the policy is
    the right place to decide what happens to it. Widening this set would
    silently take a constraint out of the D007 measurement.
    """
    assert {CONSTRAINT_CATEGORY} == POLICY_EXEMPT_CONSTRAINTS
    assert CONSTRAINT_GEO not in POLICY_EXEMPT_CONSTRAINTS
    assert CONSTRAINT_OPEN_AT not in POLICY_EXEMPT_CONSTRAINTS
    assert CONSTRAINT_PRICE_MAX not in POLICY_EXEMPT_CONSTRAINTS


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
    # Same status as a miss, but `actual` is None rather than a category list,
    # so "we have no categories for this place" stays distinguishable from "we
    # have categories and none of them matched".
    match = check_categories(biz, ("Cafes",))
    assert match.status is UNKNOWN
    assert match.actual is None


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
        # The reason names the constraint that refuted it, not just the kind:
        # "hours removed it" and "price removed it" are different findings.
        assert report.dropped["b002"] == drop_reason(DROP_VIOLATED, "OutdoorSeating"), policy


def test_policy_exclude_drops_unknowns(config, businesses) -> None:
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "exclude"),
    )
    assert [c.business_id for c in report.kept] == ["b001"]
    missing_outdoor = drop_reason(DROP_MISSING_DATA, "OutdoorSeating")
    assert report.dropped["b005"] == missing_outdoor
    assert report.dropped["b008"] == missing_outdoor
    assert report.n_penalized == 0
    assert report.penalized_by_constraint == {}
    # The census counts the population the policy governed, whatever it did to
    # them — so it is the same number `penalize` and `neutral` would report.
    assert report.unknown_by_constraint == {"OutdoorSeating": 2}


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


def test_penalize_changes_the_order_but_not_the_candidate_set(config, businesses) -> None:
    """Pins the documented scope of `penalize`, because an ablation will be read
    against it.

    `penalize` discounts the fused Stage-2 score, so it reorders. It is *not* a
    soft exclude: it removes nothing, and `apply_filters` does not truncate, so
    Stage 3 is handed exactly the candidates `neutral` would have handed it and
    rescores all of them. `exclude` is the only policy that changes the set —
    asserted here side by side so the three are not confused.

    How far that reordering travels down the cascade is a separate question, and
    it is answered by execution in the last section of this file rather than by
    argument here: it reaches the final answer through the Stage-4 ``rrf_rank``
    feature, and does not reach it at all when Stage 4 is off.
    """
    candidates = _candidates(("b005", 0.9), ("b001", 0.5))

    def kept(policy: str) -> list[str]:
        report = apply_filters(
            candidates, businesses, OUTDOOR_QUERY, _policy_config(config, policy, penalty=0.5)
        )
        return [c.business_id for c in report.kept]

    assert set(kept("penalize")) == set(kept("neutral"))
    assert kept("penalize") != kept("neutral"), "the penalty must at least reorder"
    assert set(kept("exclude")) < set(kept("neutral")), "only exclude shrinks the set"


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
    report = apply_filters(_candidates(("ghost", 0.9), ("b001", 0.1)), businesses, parsed, config)
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
    to come with an explanation, and the explanation has to name a check."""
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES, ("b004", 0.5), ("ghost", 0.4)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "exclude"),
    )
    assert report.drop_counts == {
        drop_reason(DROP_VIOLATED, "OutdoorSeating"): 1,
        drop_reason(DROP_MISSING_DATA, "OutdoorSeating"): 2,
        DROP_PERMANENTLY_CLOSED: 1,
        DROP_UNKNOWN_BUSINESS: 1,
    }
    # One candidate dropped, one reason recorded: the counts must still sum to
    # the number of dropped candidates, which is what `n_candidates` is read as.
    assert sum(report.drop_counts.values()) == len(report.dropped)


def test_drop_counts_by_kind_recovers_the_coarse_view(config, businesses) -> None:
    """Per-constraint reasons must not cost a caller the four-way summary.

    `drop_counts_by_kind` is a separate property rather than making
    `drop_counts` coarse, because the coarse version is the one that lost the
    information the report exists to carry.
    """
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES, ("b004", 0.5), ("ghost", 0.4)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "exclude"),
    )
    assert report.drop_counts_by_kind == {
        DROP_VIOLATED: 1,
        DROP_MISSING_DATA: 2,
        DROP_PERMANENTLY_CLOSED: 1,
        DROP_UNKNOWN_BUSINESS: 1,
    }


def test_reason_readers_round_trip_what_the_writer_wrote(config, businesses) -> None:
    """`drop_reason` / `reason_kind` / `reason_constraint` are one contract.

    Pinned together because the readers are what the failure analysis will use
    to group the counters, and a drift between writer and readers would show up
    as a quietly empty table rather than as an error. The two non-constraint
    kinds have no constraint half, and must read back as None rather than as an
    empty string.
    """
    report = apply_filters(
        _candidates(*POLICY_CANDIDATES, ("b004", 0.5), ("ghost", 0.4)),
        businesses,
        OUTDOOR_QUERY,
        _policy_config(config, "exclude"),
    )
    assert reason_kind(report.dropped["b002"]) == DROP_VIOLATED
    assert reason_constraint(report.dropped["b002"]) == "OutdoorSeating"
    assert reason_kind(report.dropped["b005"]) == DROP_MISSING_DATA
    assert reason_constraint(report.dropped["b005"]) == "OutdoorSeating"
    assert reason_kind(report.dropped["b004"]) == DROP_PERMANENTLY_CLOSED
    assert reason_constraint(report.dropped["b004"]) is None
    assert reason_kind(report.dropped["ghost"]) == DROP_UNKNOWN_BUSINESS
    assert reason_constraint(report.dropped["ghost"]) is None


def test_two_constraints_are_reported_as_two_different_reasons(config) -> None:
    """The point of the per-constraint reasons, in one assertion.

    Two candidates, two different failing checks: one is refuted by price, the
    other has no hours data at all. Before the reasons named a constraint, both
    of these collapsed into a single counter and the sentence "the hours check
    removed x% of candidates and the price check y%" was unwriteable from this
    report.
    """
    expensive = Business(
        business_id="expensive",
        name="Expensive",
        city="Metroville",
        state="AZ",
        latitude=33.4484,
        longitude=-112.0740,
        stars=4.0,
        review_count=10,
        categories=("Cafes",),
        attributes={"RestaurantsPriceRange2": "4"},
        hours={"Monday": (HoursInterval(0, 24 * 60),)},
        is_open=True,
    )
    no_hours = Business(
        business_id="no-hours",
        name="No Hours",
        city="Metroville",
        state="AZ",
        latitude=33.4484,
        longitude=-112.0740,
        stars=4.0,
        review_count=10,
        categories=("Cafes",),
        attributes={"RestaurantsPriceRange2": "1"},
        hours={},
        is_open=True,
    )
    parsed = ParsedQuery(
        raw="cheap cafe open monday noon",
        semantic_text="cafe",
        open_at=MON_NOON,
        price_max=2,
    )
    report = apply_filters(
        _candidates(("expensive", 0.9), ("no-hours", 0.8)),
        {"expensive": expensive, "no-hours": no_hours},
        parsed,
        _policy_config(config, "exclude"),
    )
    assert report.dropped == {
        "expensive": drop_reason(DROP_VIOLATED, CONSTRAINT_PRICE_MAX),
        "no-hours": drop_reason(DROP_MISSING_DATA, CONSTRAINT_OPEN_AT),
    }


def test_penalties_are_counted_per_constraint(config) -> None:
    """`n_penalized` counts candidates; `penalized_by_constraint` counts causes.

    They are not expected to agree: the candidate below is missing data for two
    different constraints, so it is one penalized candidate and two causes. A
    single integer cannot answer "which check is doing the penalising", which is
    what the D007 write-up has to say.
    """
    blank = Business(
        business_id="blank",
        name="Blank",
        city="Metroville",
        state="AZ",
        latitude=33.4484,
        longitude=-112.0740,
        stars=4.0,
        review_count=10,
        categories=("Cafes",),
        attributes={},
        hours={},
        is_open=True,
    )
    parsed = ParsedQuery(
        raw="quiet cafe open monday noon",
        semantic_text="cafe",
        open_at=MON_NOON,
        attributes={"NoiseLevel": "quiet"},
    )
    report = apply_filters(
        _candidates(("blank", 0.8)), {"blank": blank}, parsed, _policy_config(config, "penalize")
    )
    assert report.n_penalized == 1
    assert report.penalized_by_constraint == {CONSTRAINT_OPEN_AT: 1, "NoiseLevel": 1}
    # One discount per candidate, not one per missing constraint — the penalty
    # must not compound into 0.25 for a business that is missing two fields.
    assert report.kept[0].scores[SCORE_RRF] == pytest.approx(0.4)


def test_the_unknown_census_is_identical_under_all_three_policies(config, businesses) -> None:
    """What makes the census usable as the denominator of the D007 comparison.

    `unknown_by_constraint` describes the *data*, so it must not move when the
    policy moves — otherwise the three runs would each report a different
    population and "x of y candidates had no data" would mean something
    different in each row.
    """
    census = {}
    for policy in ("exclude", "penalize", "neutral"):
        report = apply_filters(
            _candidates(*POLICY_CANDIDATES),
            businesses,
            OUTDOOR_QUERY,
            _policy_config(config, policy),
        )
        census[policy] = report.unknown_by_constraint
    assert census["exclude"] == census["penalize"] == census["neutral"]
    assert census["penalize"] == {"OutdoorSeating": 2}


def test_the_census_separates_a_taxonomy_gap_from_a_missing_attribute(config) -> None:
    """Both arrive as UNKNOWN; they are two different findings.

    One business has no NoiseLevel field (the dataset does not cover it); the
    other has NoiseLevel but its categories miss the lexicon (our matcher does
    not cover it). Only the first is the policy's business, and the report has
    to be able to say which is which — that is the whole reason the census is
    keyed by constraint.
    """
    common = {
        "city": "Metroville",
        "state": "AZ",
        "latitude": 33.4484,
        "longitude": -112.0740,
        "stars": 4.0,
        "review_count": 10,
        "hours": {},
        "is_open": True,
    }
    no_noise_field = Business(
        business_id="no-noise",
        name="No Noise Field",
        categories=("Ramen",),
        attributes={},
        **common,
    )
    wrong_taxonomy = Business(
        business_id="odd-tags",
        name="Odd Tags",
        categories=("Japanese", "Noodles"),
        attributes={"NoiseLevel": "quiet"},
        **common,
    )
    parsed = ParsedQuery(
        raw="quiet ramen",
        semantic_text="ramen",
        attributes={"NoiseLevel": "quiet"},
        categories=("Ramen",),
    )
    report = apply_filters(
        _candidates(("no-noise", 0.9), ("odd-tags", 0.8)),
        {"no-noise": no_noise_field, "odd-tags": wrong_taxonomy},
        parsed,
        _policy_config(config, "penalize"),
    )
    assert report.unknown_by_constraint == {"NoiseLevel": 1, CONSTRAINT_CATEGORY: 1}
    # Only the attribute gap is the policy's business, so only one candidate is
    # discounted — and `penalized_by_constraint` names which one and why.
    assert report.n_penalized == 1
    assert report.penalized_by_constraint == {"NoiseLevel": 1}
    scores = {c.business_id: c.scores[SCORE_RRF] for c in report.kept}
    assert scores["no-noise"] == pytest.approx(0.45)
    assert scores["odd-tags"] == pytest.approx(0.8)


def test_penalize_does_not_invent_a_fused_score(config, businesses) -> None:
    """A candidate with no `SCORE_RRF` must not come out of here with one.

    Multiplying a defaulted 0.0 by the penalty would write a fused score for a
    candidate that fusion never scored, and the Stage 4 feature builder reads an
    absent score as NaN ("this stage never scored this candidate"), which is the
    honest value. Not reachable through `fusion.fuse`; asserted because the
    penalty must not be the thing that creates one.
    """
    candidate = Candidate(business_id="b005", scores={"bm25": 7.5}, ranks={"bm25": 1})
    report = apply_filters(
        [candidate], businesses, OUTDOOR_QUERY, _policy_config(config, "penalize")
    )
    (kept,) = report.kept
    assert SCORE_RRF not in kept.scores
    assert kept.scores["bm25"] == pytest.approx(7.5)
    # Nothing was discounted, so nothing is reported as discounted.
    assert report.n_penalized == 0
    assert report.penalized_by_constraint == {}
    # The census still records that the data was missing, which is the fact.
    assert report.unknown_by_constraint == {"OutdoorSeating": 1}


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


# ─────────────────────────────────────────────────────────────────────────────
# How far the missing-data policy reaches: Stage 2 -> Stage 4 -> the answer
#
# These tests exist because of a specific reporting hazard. DECISIONS.md D007
# promises to *measure* the three policies against each other, and `penalize`
# only ever multiplies the fused Stage-2 score. If that discount turned out to
# be invisible by the time the pipeline produces its top ten, the promised
# experiment would be comparing two real behaviours (`exclude`, `neutral`)
# against a placebo — and the natural reading of "penalize ≈ neutral" is
# "missing attribute data does not matter", which would be a wrong conclusion
# drawn from a perfectly real number.
#
# So the reach is established here by execution rather than by argument, and in
# both directions: where the policy does change the final answer and where it
# does not. The route is `apply_filters` recomputing `RANK_RRF` -> the Stage-4
# `rrf_rank` feature -> the final ordering. Stage 3 is *not* on that route: it
# rescores every survivor from scratch.
#
# What these tests deliberately do NOT claim: how large the Stage-4 effect is.
# That depends on how much the trained LambdaMART weights `rrf_rank`, which is
# unmeasured, and no number for it is asserted here or anywhere else.
# ─────────────────────────────────────────────────────────────────────────────

QUIET_QUERY = ParsedQuery(
    raw="quiet cafe",
    semantic_text="cafe",
    attributes={"NoiseLevel": "quiet"},
)

#: Retrieved best but missing the requested attribute — the business the policy
#: is about.
_MISSING = "quiet-nodata"
#: Retrieved worse but provably satisfying it.
_SATISFIES = "quiet-known"
#: Retrieved worst and refuted by the data, so dropped under every policy. Here
#: to keep the scenario honest: all three constraint statuses are present.
_VIOLATES = "loud-known"


def _cafe(business_id: str, noise: str | None) -> Business:
    return Business(
        business_id=business_id,
        name=business_id,
        city="Metroville",
        state="AZ",
        latitude=33.4484,
        longitude=-112.0740,
        stars=4.0,
        review_count=100,
        # "Cafes" is what the rule parser emits for "quiet cafe", so the category
        # check is SATISFIED for all three and the only UNKNOWN in the scenario
        # is the attribute the test is about.
        categories=("Cafes",),
        attributes={} if noise is None else {"NoiseLevel": noise},
        hours={},
        is_open=True,
    )


@pytest.fixture
def cafes() -> dict[str, Business]:
    return {
        _MISSING: _cafe(_MISSING, None),
        _SATISFIES: _cafe(_SATISFIES, "quiet"),
        _VIOLATES: _cafe(_VIOLATES, "loud"),
    }


#: Fusion order handed to `apply_filters`: the business with no data is ranked
#: first, and its score is close enough to the next one that a 0.5 discount
#: crosses it. That closeness is the scenario, not a coincidence — a penalty
#: only reorders when it moves a candidate past a neighbour.
_RETRIEVED = ((_MISSING, 0.9), (_SATISFIES, 0.5), (_VIOLATES, 0.4))


def test_the_three_policies_reach_the_stage_4_feature_matrix(config, cafes) -> None:
    """The load-bearing, stub-free half of the demonstration.

    `rrf_rank` is a Stage-4 feature (`concierge.ltr.features.FEATURE_NAMES`), and
    `apply_filters` recomputes `RANK_RRF` over its survivors after applying the
    policy. So the three policies hand the final model three different feature
    matrices: the business with missing data is row 1 under `neutral`, row 2
    under `penalize`, and absent under `exclude`.

    This is what makes the D007 comparison a real experiment on the full
    pipeline rather than on retrieval-only rows alone. It says nothing about how
    much the model weights the feature — only that the information arrives.
    """
    from concierge.ltr.features import F_RRF_RANK, build_feature_row

    def rrf_rank_feature(policy: str) -> dict[str, float]:
        report = apply_filters(
            _candidates(*_RETRIEVED), cafes, QUIET_QUERY, _policy_config(config, policy)
        )
        return {
            c.business_id: build_feature_row(QUIET_QUERY, c, cafes[c.business_id])[F_RRF_RANK]
            for c in report.kept
        }

    assert rrf_rank_feature("neutral") == {_MISSING: 1.0, _SATISFIES: 2.0}
    assert rrf_rank_feature("penalize") == {_SATISFIES: 1.0, _MISSING: 2.0}
    assert rrf_rank_feature("exclude") == {_SATISFIES: 1.0}


class _FixedScoreReranker:
    """Stage 3 stand-in whose scores do not depend on the Stage-2 ordering.

    That independence is the point. A real cross-encoder reads (query, document)
    and nothing else, so its scores are a function of the text alone — which is
    precisely why it erases the Stage-2 ordering, and why `penalize` cannot
    reach the answer through Stage 3. A stub that peeked at `SCORE_RRF` would
    hide the very property these tests are here to establish.

    Implements the Stage 3 contract that `concierge.pipeline.Reranker` names:
    add `SCORE_CROSS_ENCODER` and `RANK_CROSS_ENCODER` without overwriting
    anything upstream, stable-sort by descending score, truncate to `top_k`.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = dict(scores)

    def rerank(self, parsed_query, candidates, documents, top_k):
        scored = [
            c.with_score(SCORE_CROSS_ENCODER, self._scores[c.business_id]) for c in candidates
        ]
        scored.sort(key=lambda c: -c.scores[SCORE_CROSS_ENCODER])
        return tuple(
            Candidate(
                business_id=c.business_id,
                scores=c.scores,
                ranks={**c.ranks, RANK_CROSS_ENCODER: position},
                constraints=c.constraints,
            )
            for position, c in enumerate(scored[:top_k], start=1)
        )


class _LinearFeatureRanker:
    """Stage 4 stand-in: a transparent linear model over the *real* feature row.

    It calls `concierge.ltr.features.build_feature_row` — the same function the
    trained model is fed — so the only thing being stubbed is the learned
    weights, not the feature extraction. The weights are passed in and stated by
    the test that uses them, so there is no hidden "it works because the stub
    was written to make it work": what a test asserts is that a final ranker
    which uses `rrf_rank` *at all* orders the three policies differently.

    A real LambdaMART would weight `rrf_rank` by some learned amount. Whether
    that amount is large is exactly the open question D007 asks, and nothing
    here answers it.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = dict(weights)

    def score(self, parsed, candidates, businesses):
        from concierge.ltr.features import build_feature_row

        out = []
        for candidate in candidates:
            row = build_feature_row(parsed, candidate, businesses[candidate.business_id])
            out.append(sum(weight * row[name] for name, weight in self._weights.items()))
        return out


class _FixedRetriever:
    """Stage 2 stand-in: a fixed `(business_id, score)` list, best first."""

    def __init__(self, results) -> None:
        self._results = list(results)

    def search(self, query_text: str, top_k: int):
        return self._results[:top_k]


def _pipeline(config, cafes, *, stages):
    """A whole pipeline over the three cafés, with Stages 3 and 4 stubbed.

    Everything the policy has to travel through is the real code: the rule
    parser, `fusion.fuse`, `apply_filters`, the pipeline's own stage plumbing
    and result assembly. Stage 3 and Stage 4 are stubs because neither a
    cross-encoder checkpoint nor a trained LightGBM model exists in this
    environment — see `_FixedScoreReranker` and `_LinearFeatureRanker` for what
    each one does and does not stand in for.
    """
    from concierge.pipeline import SearchPipeline, Variant

    return SearchPipeline(
        config,
        businesses=cafes,
        documents={bid: f"{bid} cafe" for bid in cafes},
        bm25_index=_FixedRetriever(_RETRIEVED),
        # Scores the business with missing data highest, independently of
        # Stage 2. With Stage 4 off this is the final order under both
        # `penalize` and `neutral`, which is the caveat the second test pins.
        reranker=_FixedScoreReranker({_MISSING: 0.9, _SATISFIES: 0.1, _VIOLATES: 0.0}),
        # Lower `rrf_rank` is better, hence the negative weight. Every other
        # feature is weighted zero so that the assertion is about this one
        # route and not about a mixture of signals.
        ltr_ranker=_LinearFeatureRanker({"rrf_rank": -1.0}),
        variant=Variant(name="test_variant", stages=stages),
    )


def test_the_three_policies_produce_three_different_final_orderings(config, cafes) -> None:
    """D007's comparison, run end to end: three policies, three answers.

    Scenario: a café retrieved first has no `NoiseLevel` data, and a café
    retrieved second is recorded as quiet. The cross-encoder prefers the one
    with no data (so Stage 3 cannot be what distinguishes the policies), and the
    final ranker uses `rrf_rank`.

      neutral  -> the undocumented café keeps rank 1 and stays on top
      penalize -> its discounted score falls below the quiet café, which takes
                  rank 1 and wins the final ordering
      exclude  -> the undocumented café is gone from the answer entirely

    Three genuinely different top-k lists, so none of the three rows of the
    promised ablation is a placebo.
    """
    from concierge.pipeline import Stage

    stages = Stage.BM25 | Stage.FILTERS | Stage.RERANK | Stage.LTR

    def final_order(policy: str) -> list[str]:
        pipeline = _pipeline(_policy_config(config, policy), cafes, stages=stages)
        return [r.business.business_id for r in pipeline.search("quiet cafe").results]

    neutral = final_order("neutral")
    penalize = final_order("penalize")
    exclude = final_order("exclude")

    assert neutral == [_MISSING, _SATISFIES]
    assert penalize == [_SATISFIES, _MISSING]
    assert exclude == [_SATISFIES]
    assert len({tuple(neutral), tuple(penalize), tuple(exclude)}) == 3

    # The refuted café is absent from all three: a VIOLATED constraint is not a
    # policy question, and no ranking signal may outvote it.
    assert _VIOLATES not in neutral + penalize + exclude


def test_with_stage_4_off_penalize_and_neutral_give_the_same_order(config, cafes) -> None:
    """The other half of the honest answer, and the reason it is written down.

    Same scenario, Stage 4 switched off. Stage 3 rescores every survivor and
    re-sorts by its own score, so the Stage-2 ordering `penalize` changed is
    simply gone: `penalize` and `neutral` return the identical list. `exclude`
    still differs, because it is the only policy that changes the candidate set.

    This is pinned as a *property*, not left as a caveat in a docstring, so that
    nobody reads a penalize-vs-neutral gap on a reranked-but-no-Stage-4 ablation
    row as evidence about missing attribute data. On that row there is nothing
    to read.
    """
    from concierge.pipeline import Stage

    stages = Stage.BM25 | Stage.FILTERS | Stage.RERANK

    def final_order(policy: str) -> list[str]:
        pipeline = _pipeline(_policy_config(config, policy), cafes, stages=stages)
        return [r.business.business_id for r in pipeline.search("quiet cafe").results]

    assert final_order("penalize") == final_order("neutral") == [_MISSING, _SATISFIES]
    assert final_order("exclude") == [_SATISFIES]


def test_without_the_reranker_the_policy_orders_the_answer_directly(config, cafes) -> None:
    """The retrieval-only rows, where `SCORE_RRF` *is* the final ranking.

    With Stages 3 and 4 both off, the policy's effect on the answer is its
    effect on the fused score, undiluted. These are the rows D007's three-way
    comparison is cleanest on, and the contrast with the test above is the whole
    reason D007 has to name which rows it is talking about.
    """
    from concierge.pipeline import Stage

    stages = Stage.BM25 | Stage.FILTERS

    def final_order(policy: str) -> list[str]:
        pipeline = _pipeline(_policy_config(config, policy), cafes, stages=stages)
        return [r.business.business_id for r in pipeline.search("quiet cafe").results]

    assert final_order("neutral") == [_MISSING, _SATISFIES]
    assert final_order("penalize") == [_SATISFIES, _MISSING]
    assert final_order("exclude") == [_SATISFIES]
