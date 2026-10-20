"""Tests for Stage 1 without the LLM: the JSON schema/validator and the rule parser.

Two things are being pinned here, and they are different kinds of claim.

For ``schema.py`` the claim is *nothing junk gets through*. Every test that asserts a
``SchemaValidationError`` is protecting a number in the ablation table: a bad filter value
silently empties or widens the result set, which looks exactly like a ranking failure and
is almost impossible to find later.

For ``rules.py`` the claim is *predictable degradation*. The rule parser is allowed to miss
constraints — that gap against the LLM parser is a measurement the project wants. What it
is not allowed to do is invent a constraint, throw, or be unpredictable about which words
reach retrieval. The tests below therefore assert the stripped ``semantic_text`` exactly,
and the known failure modes are tested as the behaviour they actually have, not as
aspirations.

``TestTheStripRule`` carries the one cross-cutting claim: a phrase the parser lifted leaves
``semantic_text`` if and only if it became a constraint the filters can reject a business
for (``rules.strips_phrase``). It is parametrized over every constraint kind the rule table
declares, so a kind added later cannot quietly acquire its own policy. ``semantic_text`` may
therefore be empty — an all-constraint query has nothing left to embed — and that is a
state, not a failure: ``concierge.rerank.model.query_text_for`` is the one definition of the
string a stage scores and falls back to ``raw``, so nothing is ever handed the empty string.

No test here needs the Yelp dataset or an API key, so nothing is marked ``needs_data`` or
``needs_api``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from concierge.data.hours import is_open_at, parse_hours
from concierge.query import rules
from concierge.query.rules import (
    CATEGORY_LEXICON,
    LATE_NIGHT_MINUTE,
    REJECTING_CONSTRAINTS,
    VAGUE_CHEAP_PRICE_MAX,
    Writes,
    parse_rules,
    strip_phrases,
    strips_phrase,
)
from concierge.query.schema import (
    MAX_CATEGORIES,
    SMALL_HOURS_END_MINUTE,
    SUPPORTED_ATTRIBUTES,
    WEEKDAYS,
    SchemaValidationError,
    describe_attributes,
    in_small_hours,
    night_order_key,
    normalize_day,
    parse_clock_time,
    query_json_schema,
    resolve_open_at_day,
    validate_parse_payload,
    validate_parsed_query,
)
from concierge.retrieval.filters import (
    PRICE_ATTRIBUTE,
    check_attribute,
    check_categories,
    check_open_at,
    check_price_max,
)
from concierge.types import (
    Business,
    ConstraintStatus,
    DayOfWeek,
    HoursInterval,
    OpenAt,
    ParsedQuery,
    ParseSource,
)

# ─────────────────────────────────────────────────────────────────────────────
# Normalizers
# ─────────────────────────────────────────────────────────────────────────────


class TestParseClockTime:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("22:00", 22 * 60),
            ("22:0", 22 * 60),  # the dataset's own style
            ("10pm", 22 * 60),
            ("10 PM", 22 * 60),
            ("10 p.m.", 22 * 60),
            ("7:30pm", 19 * 60 + 30),
            ("12am", 0),
            ("12pm", 12 * 60),
            ("midnight", 0),
            ("noon", 12 * 60),
            ("0:00", 0),
            ("19", 19 * 60),
            ("24:00", 0),  # same instant as midnight, expressed usably
        ],
    )
    def test_accepts_the_forms_models_and_people_write(self, text: str, expected: int) -> None:
        assert parse_clock_time(text) == expected

    @pytest.mark.parametrize("text", ["25:00", "22:75", "later", "", "13pm", "0pm"])
    def test_rejects_junk(self, text: str) -> None:
        with pytest.raises(SchemaValidationError):
            parse_clock_time(text)

    def test_rejects_a_boolean(self) -> None:
        # bool is an int subclass, so this would otherwise parse as 60 minutes.
        with pytest.raises(SchemaValidationError):
            parse_clock_time(True)

    def test_every_result_is_a_usable_minute_of_day(self) -> None:
        # HoursInterval.covers() takes a minute-of-day, so 1440 would never match.
        for text in ("midnight", "24:00", "23:59", "noon"):
            assert 0 <= parse_clock_time(text) < 24 * 60


class TestNormalizeDay:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("fri", "Friday"), ("Friday", "Friday"), ("FRIDAY", "Friday"), ("thurs", "Thursday")],
    )
    def test_normalizes_to_the_datasets_spelling(self, text: str, expected: str) -> None:
        assert normalize_day(text) == expected

    @pytest.mark.parametrize("text", ["weekend", "weekday", "tomorrow", "Funday", ""])
    def test_rejects_anything_that_is_not_one_weekday(self, text: str) -> None:
        # "weekend" is two days and OpenAt holds one, so accepting it would quietly drop
        # half the request.
        with pytest.raises(SchemaValidationError):
            normalize_day(text)


class TestResolveOpenAtDay:
    """The night-to-calendar-day convention, tested directly.

    Both parsers route through this function, so these cases are the contract: a wrong
    answer here is a filter that asks about the wrong night and reports no error.

    ``day_was_named`` selects between the two halves of the contract and is keyword-only
    and required, so every case below has to say which half it is testing.
    """

    def test_the_small_hours_boundary_is_the_stated_convention(self) -> None:
        # Pinned against the constant, not against a literal: the threshold is an
        # unmeasured convention the failure analysis is expected to move, and a test that
        # restated the number would have to be edited in lockstep for no benefit.
        assert in_small_hours(0), "midnight is part of the night that preceded it"
        assert in_small_hours(SMALL_HOURS_END_MINUTE - 1)
        assert not in_small_hours(SMALL_HOURS_END_MINUTE)

    def test_a_small_hours_time_belongs_to_the_following_calendar_day(self) -> None:
        assert resolve_open_at_day("Friday", 2 * 60, day_was_named=False) == "Saturday"
        assert resolve_open_at_day("Friday", 0, day_was_named=False) == "Saturday"

    def test_a_daytime_or_evening_time_stays_on_its_own_day(self) -> None:
        assert (
            resolve_open_at_day("Friday", SMALL_HOURS_END_MINUTE, day_was_named=False) == "Friday"
        )
        assert resolve_open_at_day("Friday", 22 * 60, day_was_named=False) == "Friday"

    def test_the_clock_only_matters_inside_the_small_hours(self) -> None:
        # Asked at 01:00, the night is already underway: "2am" is an hour away, still the
        # same calendar day. Asked in the evening, it is tomorrow's small hours.
        assert (
            resolve_open_at_day("Saturday", 2 * 60, day_was_named=False, now_minute=60)
            == "Saturday"
        )
        assert (
            resolve_open_at_day("Saturday", 2 * 60, day_was_named=False, now_minute=20 * 60)
            == "Sunday"
        )
        # An evening constraint is unaffected by either.
        assert (
            resolve_open_at_day("Saturday", 22 * 60, day_was_named=False, now_minute=60)
            == "Saturday"
        )

    def test_the_week_wraps_in_both_directions(self) -> None:
        assert resolve_open_at_day("Sunday", 60, day_was_named=False) == "Monday"
        assert resolve_open_at_day("Monday", 60, day_was_named=False, now_minute=30) == "Monday"

    def test_a_named_day_is_taken_at_face_value(self) -> None:
        """A day the request named is the calendar day a clock would read, so no shift.

        The alternative — reading a named day as a *night* anchor — was the earlier
        behaviour and is wrong for the one case the prompt pins down: it tells the model
        that "tomorrow" means the day after today, so the model has already done the
        calendar arithmetic and a shift here would land two days out.
        """
        for minute in (0, 60, 2 * 60, SMALL_HOURS_END_MINUTE - 1):
            assert resolve_open_at_day("Friday", minute, day_was_named=True) == "Friday", minute

    def test_the_clock_can_never_move_a_day_the_request_named(self) -> None:
        """`now_minute` is ignored for a named day, whatever the hour.

        Suppressed inside this function rather than at each call site, so neither parser
        can forget to do it — the asymmetry that let the two paths disagree before.
        """
        for now_minute in (None, 0, 60, 13 * 60, 23 * 60):
            for minute in (60, 10 * 60, 22 * 60):
                assert (
                    resolve_open_at_day(
                        "Wednesday", minute, day_was_named=True, now_minute=now_minute
                    )
                    == "Wednesday"
                ), (now_minute, minute)


class TestNightOrderKey:
    """Ordering two requested closing times, which clock order gets wrong."""

    def test_a_small_hours_time_outranks_any_evening_time(self) -> None:
        # The whole point: 02:00 is later *into the night* than 23:59, but earlier on the
        # clock. A comparison that used minute-of-day would pick the evening hour and
        # relax a constraint the query stated outright.
        assert night_order_key(2 * 60) > night_order_key(23 * 60 + 59)
        assert night_order_key(0) > night_order_key(22 * 60)

    def test_daytime_and_evening_times_keep_their_clock_order(self) -> None:
        assert night_order_key(9 * 60) < night_order_key(22 * 60) < night_order_key(23 * 60)

    def test_the_boundary_follows_the_small_hours_convention(self) -> None:
        # One minute apart on the clock, a day apart in night order — and the split is the
        # same constant `in_small_hours` uses, not a second copy of the convention.
        assert night_order_key(SMALL_HOURS_END_MINUTE) == SMALL_HOURS_END_MINUTE
        assert night_order_key(SMALL_HOURS_END_MINUTE - 1) > night_order_key(SMALL_HOURS_END_MINUTE)

    def test_it_is_an_ordering_key_and_never_a_minute_of_day(self) -> None:
        # Guards against the key leaking into `OpenAt`, which must hold a real
        # minute-of-day: the shifted values are deliberately outside that range.
        assert night_order_key(60) >= 24 * 60


# ─────────────────────────────────────────────────────────────────────────────
# The schema dict sent to the API
# ─────────────────────────────────────────────────────────────────────────────


class TestQuerySchemaDict:
    def test_openapi_dialect_marks_optional_fields_nullable(self) -> None:
        schema = query_json_schema("openapi")
        assert schema["properties"]["open_at"]["nullable"] is True
        assert schema["required"] == ["semantic_text"]

    def test_strict_jsonschema_dialect_uses_type_unions_and_closes_objects(self) -> None:
        schema = query_json_schema("jsonschema")
        assert schema["properties"]["open_at"]["type"] == ["object", "null"]
        assert "nullable" not in schema["properties"]["open_at"]
        assert schema["additionalProperties"] is False
        # OpenAI-style strict mode requires every property in `required`; optionality is
        # carried by the null in the type union.
        assert set(schema["required"]) == set(schema["properties"])

    def test_strict_mode_leaves_a_legal_way_to_say_not_requested(self) -> None:
        # Strict mode forces every attribute into `required`, so each one must also accept
        # null — otherwise the model is obliged to assert something about WiFi, dogs and
        # happy hour on every query, which is exactly the hallucinated-constraint failure
        # the validator exists to prevent.
        attributes = query_json_schema("jsonschema")["properties"]["attributes"]
        assert set(attributes["required"]) == set(attributes["properties"])
        for key, spec in SUPPORTED_ATTRIBUTES.items():
            prop = attributes["properties"][key]
            assert "null" in prop["type"], key
            if spec.kind == "enum":
                assert None in prop["enum"], key

    def test_strict_mode_keeps_genuinely_required_fields_non_nullable(self) -> None:
        schema = query_json_schema("jsonschema")
        assert schema["properties"]["semantic_text"]["type"] == "string"
        # `time` is what makes an hours constraint checkable, so it stays mandatory inside
        # an open_at block, while `day` may be null.
        assert schema["properties"]["open_at"]["properties"]["time"]["type"] == "string"

    def test_every_supported_attribute_is_advertised_with_its_vocabulary(self) -> None:
        props = query_json_schema()["properties"]["attributes"]["properties"]
        assert set(props) == set(SUPPORTED_ATTRIBUTES)
        for key, spec in SUPPORTED_ATTRIBUTES.items():
            if spec.kind == "enum":
                assert props[key]["enum"] == list(spec.values)
            else:
                assert props[key]["type"] == "boolean"

    def test_each_call_returns_an_independent_copy(self) -> None:
        # A client that mutates a shared dict breaks the *other* provider, which only
        # shows up when the fallback runs.
        first = query_json_schema()
        first["properties"].pop("semantic_text")
        assert "semantic_text" in query_json_schema()["properties"]

    def test_unknown_dialect_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError):
            query_json_schema("protobuf")  # type: ignore[arg-type]

    def test_describe_attributes_lists_the_same_vocabulary(self) -> None:
        # The prompt and the validator must come from one table or the advertised value
        # fails validation 100% of the time.
        rendered = describe_attributes()
        for key, spec in SUPPORTED_ATTRIBUTES.items():
            assert key in rendered
            for value in spec.values:
                assert value in rendered


# ─────────────────────────────────────────────────────────────────────────────
# Validating a model reply
# ─────────────────────────────────────────────────────────────────────────────

RAW = "quiet cafe with wifi open past 10pm"

GOOD_PAYLOAD = {
    "semantic_text": "cafe",
    "open_at": {"day": "fri", "time": "10pm"},
    "attributes": {"NoiseLevel": "quiet", "WiFi": "free", "OutdoorSeating": True},
    "price_max": 2,
    "categories": ["Cafes"],
    "geo": None,
}


class TestValidateHappyPath:
    def test_builds_a_parsed_query_with_normalized_values(self) -> None:
        parsed = validate_parsed_query(GOOD_PAYLOAD, raw_query=RAW)
        assert parsed.raw == RAW
        assert parsed.semantic_text == "cafe"
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == ("Friday", 22 * 60)
        assert parsed.attributes == {
            "NoiseLevel": "quiet",
            "WiFi": "free",
            "OutdoorSeating": True,
        }
        assert parsed.price_max == 2
        assert parsed.categories == ("Cafes",)
        assert parsed.geo is None

    def test_accepts_raw_json_text(self) -> None:
        parsed = validate_parsed_query(json.dumps(GOOD_PAYLOAD), raw_query=RAW)
        assert parsed.semantic_text == "cafe"

    def test_carries_source_and_measured_latency_through(self) -> None:
        parsed = validate_parsed_query(
            GOOD_PAYLOAD, raw_query=RAW, source=ParseSource.CACHE, parse_latency_ms=1.5
        )
        # A cache hit has LLM quality at a fraction of the latency, so the ablation
        # reports it as its own row.
        assert parsed.source is ParseSource.CACHE
        assert parsed.parse_latency_ms == 1.5

    def test_counts_hard_constraints_for_the_failure_breakdown(self) -> None:
        parsed = validate_parsed_query(GOOD_PAYLOAD, raw_query=RAW)
        # open_at + 3 attributes + price_max
        assert parsed.n_hard_constraints == 5

    def test_a_minimal_reply_is_valid(self) -> None:
        parsed = validate_parsed_query({"semantic_text": "good pho"}, raw_query="good pho")
        assert parsed.n_hard_constraints == 0
        assert parsed.semantic_text == "good pho"

    def test_nulls_are_treated_as_absent(self) -> None:
        payload = {
            "semantic_text": "tacos",
            "open_at": None,
            "attributes": None,
            "price_max": None,
            "categories": None,
            "geo": None,
        }
        parsed = validate_parsed_query(payload, raw_query="tacos")
        assert parsed.n_hard_constraints == 0


class TestValidateRejectsJunk:
    def test_non_json_text(self) -> None:
        with pytest.raises(SchemaValidationError):
            validate_parsed_query("I'd be happy to help!", raw_query=RAW)

    def test_a_json_array(self) -> None:
        with pytest.raises(SchemaValidationError):
            validate_parsed_query("[1, 2]", raw_query=RAW)

    def test_an_unexpected_top_level_key(self) -> None:
        # Usually means the model answered a different question, so the fields we did
        # read are suspect too.
        with pytest.raises(SchemaValidationError, match="unexpected keys"):
            validate_parsed_query({"semantic_text": "x", "cuisine": "thai"}, raw_query=RAW)

    def test_an_unknown_attribute_key(self) -> None:
        with pytest.raises(SchemaValidationError, match="unknown attribute"):
            validate_parsed_query(
                {"semantic_text": "x", "attributes": {"Cozy": True}}, raw_query=RAW
            )

    def test_an_out_of_vocabulary_enum_value(self) -> None:
        # "pretty quiet" matches no business at all and would look like a ranking bug.
        with pytest.raises(SchemaValidationError, match="NoiseLevel"):
            validate_parsed_query(
                {"semantic_text": "x", "attributes": {"NoiseLevel": "pretty quiet"}},
                raw_query=RAW,
            )

    def test_a_non_boolean_for_a_flag(self) -> None:
        with pytest.raises(SchemaValidationError, match="OutdoorSeating"):
            validate_parsed_query(
                {"semantic_text": "x", "attributes": {"OutdoorSeating": "maybe"}},
                raw_query=RAW,
            )

    @pytest.mark.parametrize("value", [0, 5, "lots", 2.5, True])
    def test_a_price_outside_yelps_scale(self, value: object) -> None:
        with pytest.raises(SchemaValidationError):
            validate_parsed_query({"semantic_text": "x", "price_max": value}, raw_query=RAW)

    def test_an_unexpected_key_inside_open_at(self) -> None:
        with pytest.raises(SchemaValidationError, match="open_at"):
            validate_parsed_query(
                {"semantic_text": "x", "open_at": {"day": "Fri", "time": "22:00", "tz": "MST"}},
                raw_query=RAW,
            )

    def test_a_geo_block_with_no_label(self) -> None:
        with pytest.raises(SchemaValidationError, match="label"):
            validate_parsed_query({"semantic_text": "x", "geo": {"radius_km": 2}}, raw_query=RAW)

    @pytest.mark.parametrize("radius", [0, -1, 1000, "near"])
    def test_an_implausible_radius(self, radius: object) -> None:
        with pytest.raises(SchemaValidationError):
            validate_parsed_query(
                {"semantic_text": "x", "geo": {"label": "downtown", "radius_km": radius}},
                raw_query=RAW,
            )

    def test_a_non_string_category(self) -> None:
        with pytest.raises(SchemaValidationError, match="categories"):
            validate_parsed_query({"semantic_text": "x", "categories": [7]}, raw_query=RAW)


class TestValidateForgivesWhatIsSafeToForgive:
    def test_attribute_key_casing(self) -> None:
        # The model got the concept right and the casing wrong; the lookup is otherwise
        # exact, so there is nothing to guess.
        parsed = validate_parsed_query(
            {"semantic_text": "x", "attributes": {"wifi": "FREE"}}, raw_query=RAW
        )
        assert parsed.attributes == {"WiFi": "free"}

    def test_stringified_booleans(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "x", "attributes": {"OutdoorSeating": "true"}}, raw_query=RAW
        )
        assert parsed.attributes == {"OutdoorSeating": True}

    def test_enum_values_written_with_spaces_or_hyphens(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "x", "attributes": {"Alcohol": "full bar"}}, raw_query=RAW
        )
        assert parsed.attributes == {"Alcohol": "full_bar"}

    def test_a_price_as_a_digit_string(self) -> None:
        parsed = validate_parsed_query({"semantic_text": "x", "price_max": "2"}, raw_query=RAW)
        assert parsed.price_max == 2

    def test_a_bare_category_string_instead_of_a_list(self) -> None:
        parsed = validate_parsed_query({"semantic_text": "x", "categories": "Ramen"}, raw_query=RAW)
        assert parsed.categories == ("Ramen",)

    def test_categories_are_deduplicated_and_capped(self) -> None:
        payload = {
            "semantic_text": "x",
            "categories": ["Ramen", "Ramen", "Noodles", "Japanese", "Soup", "Asian Fusion"],
        }
        parsed = validate_parsed_query(payload, raw_query=RAW)
        assert len(parsed.categories) == MAX_CATEGORIES
        assert parsed.categories[0] == "Ramen"

    def test_unknown_categories_are_dropped_when_a_vocabulary_is_supplied(self) -> None:
        # A hallucinated category is advisory noise; a hallucinated *attribute* would
        # silently empty the result set. Hence drop here, raise there.
        parsed = validate_parsed_query(
            {"semantic_text": "x", "categories": ["Ramen", "Late Night Cafes"]},
            raw_query=RAW,
            known_categories=frozenset({"Ramen", "Cafes"}),
        )
        assert parsed.categories == ("Ramen",)

    def test_a_null_attribute_value_is_not_a_constraint(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "x", "attributes": {"WiFi": None}}, raw_query=RAW
        )
        assert parsed.attributes == {}


class TestValidateSemanticTextFallback:
    def test_empty_text_falls_back_to_the_categories(self) -> None:
        # A fully-constrained request can legitimately leave nothing to embed, and the
        # nearest neighbours of an empty string are arbitrary.
        parsed = validate_parsed_query(
            {"semantic_text": "  ", "categories": ["Ramen"]}, raw_query="open late ramen"
        )
        assert parsed.semantic_text == "Ramen"

    def test_empty_text_with_no_categories_falls_back_to_the_raw_query(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": ""}, raw_query="open after 10pm with outdoor seating"
        )
        assert parsed.semantic_text == "open after 10pm with outdoor seating"

    def test_a_missing_semantic_text_key_is_tolerated(self) -> None:
        parsed = validate_parsed_query({"price_max": 1}, raw_query="cheap eats")
        assert parsed.semantic_text == "cheap eats"


class TestValidateParsePayload:
    """The payload layer: what the LLM client validates with and the cache stores."""

    def test_returns_all_six_keys_canonicalized(self) -> None:
        payload = validate_parse_payload(
            {
                "semantic_text": "  cafe  ",
                "open_at": {"day": "fri", "time": "10pm"},
                "attributes": {"wifi": "FREE"},
                "price_max": "2",
                "categories": ["Cafes", "Cafes"],
            }
        )
        assert payload == {
            "semantic_text": "cafe",
            "open_at": {"day": "Friday", "time": "22:00"},
            "attributes": {"WiFi": "free"},
            "price_max": 2,
            "categories": ["Cafes"],
            "geo": None,
        }

    def test_the_canonical_payload_survives_a_json_round_trip(self) -> None:
        # It is written straight into the SQLite parse cache, so an object in here would
        # break the round trip and silently disable caching.
        payload = validate_parse_payload(GOOD_PAYLOAD)
        assert json.loads(json.dumps(payload)) == payload

    def test_the_weekday_is_left_unresolved(self) -> None:
        # "open past 10pm" names a time and no day, and the day it means is the day the
        # question is asked. Resolving it here would bake the first caller's weekday into
        # every later cache hit — a bug that only shows up as wrong results the next day.
        payload = validate_parse_payload({"semantic_text": "x", "open_at": {"time": "22:00"}})
        assert payload["open_at"] == {"day": None, "time": "22:00"}

    def test_a_day_with_no_time_is_refused(self) -> None:
        # Minute-based intervals cannot be checked against a bare day, and inventing 00:00
        # would filter on a time nobody asked about. The whole parse is refused rather than
        # just this field: an open_at block means the query almost certainly has an hours
        # constraint, and the rule parser reads "open past 10pm" correctly, so falling back
        # beats salvaging a parse with the constraint silently missing.
        with pytest.raises(SchemaValidationError, match="time"):
            validate_parse_payload({"semantic_text": "x", "open_at": {"day": "Friday"}})

    def test_it_is_reachable_under_the_name_the_llm_client_resolves(self) -> None:
        # concierge.query.llm resolves the validator by name; a mismatch silently sends
        # every query down the rule-fallback path.
        from concierge.query import schema

        assert schema.validate_payload is schema.validate_parse_payload

    def test_junk_is_rejected_at_the_payload_layer_too(self) -> None:
        with pytest.raises(SchemaValidationError):
            validate_parse_payload({"semantic_text": "x", "attributes": {"Cozy": True}})

    def test_a_canonical_payload_revalidates_unchanged(self) -> None:
        # The cache stores a canonical payload and it comes back through the same door, so
        # validation has to be idempotent.
        once = validate_parse_payload(GOOD_PAYLOAD)
        assert validate_parse_payload(once) == once


class TestValidateOpenAtResolution:
    def test_today_supplies_the_day_the_model_left_out(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "open_at": {"time": "23:00"}},
            raw_query="ramen open past 11",
            today="Thursday",
        )
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == ("Thursday", 23 * 60)

    def test_the_models_day_wins_over_today(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "open_at": {"day": "Sunday", "time": "13:00"}},
            raw_query="ramen sunday lunch",
            today="Thursday",
        )
        assert parsed.open_at is not None
        assert parsed.open_at.day == "Sunday"

    def test_without_a_day_or_today_the_constraint_is_declined(self) -> None:
        # Guessing a weekday filters on hours the user never mentioned. Same choice the
        # rule parser makes when given no `today`.
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "open_at": {"time": "23:00"}}, raw_query="ramen late"
        )
        assert parsed.open_at is None
        assert parsed.semantic_text == "ramen"

    def test_a_small_hours_time_lands_on_the_next_day_here_too(self) -> None:
        """The LLM path must agree with the rule path, or the ablation compares two
        different definitions of "tonight" and calls the difference parser quality."""
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "open_at": {"time": "02:00"}},
            raw_query="ramen open past 2am",
            today="Friday",
        )
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == ("Saturday", 2 * 60)

        rule_parsed = parse_rules("ramen open past 2am", today="Friday")
        assert rule_parsed.open_at == parsed.open_at

    def test_a_small_hours_time_asked_after_midnight_stays_on_today(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "open_at": {"time": "02:00"}},
            raw_query="ramen open past 2am",
            today="Saturday",
            now_minute=60,
        )
        assert parsed.open_at is not None
        assert parsed.open_at.day == "Saturday"

    def test_a_small_hours_time_on_a_weekday_the_model_named_is_read_literally(self) -> None:
        """A named weekday is a calendar day, and the shift applies to clock-supplied days
        only.

        This branch had no test before and the behaviour has changed: the day used to be
        shifted here too, so a reply naming a day was read as the night *beginning* on it.
        That reading contradicts the prompt, which tells the model that "tomorrow" means
        the day after today — the model has already moved the day, and a shift on top of
        that lands two days out. The cost, stated in `resolve_open_at_day`: "2am on friday"
        is now Friday's own 02:00 rather than Friday night's, which needs a field the
        payload does not have to express.

        `now_minute` is ignored here on purpose: the clock may only move a day that came
        from the clock in the first place.
        """
        parsed = validate_parsed_query(
            {"semantic_text": "bar", "open_at": {"day": "Friday", "time": "02:00"}},
            raw_query="bar open until 2am friday",
            today="Tuesday",
            now_minute=60,
        )
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == ("Friday", 2 * 60)

    def test_the_same_request_is_resolved_the_same_way_whether_or_not_the_day_is_filled(
        self,
    ) -> None:
        """Asked at 01:00 on Saturday, "open past 2am" means Saturday 02:00.

        The model is told to leave `day` null when the request names no weekday, but a
        model that fills in the day it was handed anyway must not produce a different
        answer — before, these two payloads resolved a day apart, so the parse depended on
        a choice the prompt leaves to the model.
        """
        payloads = ({"day": None, "time": "01:00"}, {"day": "Saturday", "time": "01:00"})
        resolved = {
            validate_parsed_query(
                {"semantic_text": "bar", "open_at": block},
                raw_query="bar open past 1am",
                today="Saturday",
                now_minute=60,
            ).open_at
            for block in payloads
        }
        assert resolved == {OpenAt(day="Saturday", minute_of_day=60)}

    def test_a_named_weekday_with_a_daytime_hour_is_untouched(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "brunch", "open_at": {"day": "Sunday", "time": "10:00"}},
            raw_query="sunday brunch at 10",
            today="Tuesday",
            now_minute=60,
        )
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == ("Sunday", 10 * 60)


class TestValidateGeo:
    def test_a_resolver_turns_a_label_into_a_constraint(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "geo": {"label": "downtown", "radius_km": 2}},
            raw_query="ramen downtown",
            geo_resolver=lambda label: (33.45, -112.07) if label == "downtown" else None,
        )
        assert parsed.geo is not None
        assert (parsed.geo.latitude, parsed.geo.radius_km, parsed.geo.label) == (
            33.45,
            2.0,
            "downtown",
        )

    def test_a_missing_radius_gets_the_default(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "geo": {"label": "downtown"}},
            raw_query="ramen downtown",
            geo_resolver=lambda _: (33.45, -112.07),
        )
        assert parsed.geo is not None
        assert parsed.geo.radius_km > 0

    def test_an_unresolvable_label_goes_back_into_the_embeddable_text(self) -> None:
        # Dropping it silently would make a geo query look like a ranking failure. Putting
        # the words back at least lets BM25 match an address or city field.
        parsed = validate_parsed_query(
            {"semantic_text": "ramen", "geo": {"label": "Roosevelt Row", "radius_km": 1}},
            raw_query="ramen in Roosevelt Row",
        )
        assert parsed.geo is None
        assert parsed.semantic_text == "ramen Roosevelt Row"

    def test_the_label_is_not_duplicated_if_it_is_already_there(self) -> None:
        parsed = validate_parsed_query(
            {"semantic_text": "ramen downtown", "geo": {"label": "downtown"}},
            raw_query="ramen downtown",
        )
        assert parsed.semantic_text == "ramen downtown"


# ─────────────────────────────────────────────────────────────────────────────
# The rule parser: the three example queries from CLAUDE.md section 1
# ─────────────────────────────────────────────────────────────────────────────

FRIDAY = "Friday"


class TestClaudeMdExampleQueries:
    def test_quiet_cafe_with_wifi_open_past_10pm(self) -> None:
        parsed = parse_rules("quiet café with wifi that's open past 10pm", today=FRIDAY)
        assert parsed.attributes == {"NoiseLevel": "quiet", "WiFi": "free"}
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == (FRIDAY, 22 * 60)
        assert parsed.categories == ("Cafes",)
        # Every constraint phrase is gone and the dangling connectors with it; the
        # category word stays because the category filter is coarse and "café" still
        # carries meaning for retrieval.
        assert parsed.semantic_text == "café"

    def test_cheap_ramen_near_downtown_with_outdoor_seating(self) -> None:
        parsed = parse_rules("cheap ramen near downtown with outdoor seating")
        assert parsed.price_max == VAGUE_CHEAP_PRICE_MAX
        assert parsed.attributes == {"OutdoorSeating": True}
        assert parsed.categories == ("Ramen",)
        # "near downtown" survives on purpose: the rule parser has no gazetteer, so the
        # phrase has to stay where retrieval can still make soft use of it.
        assert parsed.semantic_text == "ramen near downtown"
        assert parsed.geo is None

    def test_somewhere_good_for_a_first_date_that_isnt_loud(self) -> None:
        parsed = parse_rules("somewhere good for a first date that isn't loud")
        # The negation must win over the bare word "loud", or the filter inverts the
        # request — the single most damaging failure this parser could have.
        assert parsed.attributes == {"NoiseLevel": "quiet"}
        assert parsed.semantic_text == "somewhere good for a first date"
        assert parsed.price_max is None


# ─────────────────────────────────────────────────────────────────────────────
# The rule parser, constraint by constraint
# ─────────────────────────────────────────────────────────────────────────────


class TestHoursRules:
    @pytest.mark.parametrize(
        ("query", "expected_minute"),
        [
            ("open past 10pm", 22 * 60),
            ("open after 11", 23 * 60),  # bare hour in this construction means pm
            ("open until 2am", 2 * 60),
            ("open till midnight", 0),
            ("open past 12", 0),  # colloquially midnight
            ("open past 22:30", 22 * 60 + 30),
            ("closes after 1am", 60),
            ("late night tacos", LATE_NIGHT_MINUTE),
            ("somewhere open late", LATE_NIGHT_MINUTE),
            ("ramen after midnight", 0),  # the verb is implied
            ("open at 7pm", 19 * 60),
        ],
    )
    def test_encodes_the_hour_asked_for(self, query: str, expected_minute: int) -> None:
        parsed = parse_rules(query, today=FRIDAY)
        assert parsed.open_at is not None, query
        assert parsed.open_at.minute_of_day == expected_minute

    def test_a_small_hours_request_lands_on_the_next_calendar_day(self) -> None:
        """ "open until 2am" on a Friday is about Friday *night*, whose 2am is Saturday's.

        This is the case the project set out to handle, and getting it wrong is invisible:
        OpenAt(Friday, 02:00) is a well-formed constraint that quietly asks about Thursday
        night. `is_open_at` is day-aware — for an interval that crosses midnight it only
        accepts the evening half on the interval's own day, and reaches the morning half
        through the *previous* day — so it has to be handed the day the clock would read.
        """
        parsed = parse_rules("open until 2am", today=FRIDAY)
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == ("Saturday", 2 * 60)

    def test_an_evening_request_stays_on_the_day_it_was_asked(self) -> None:
        """The other direction: only the small hours move. 10pm Friday is Friday."""
        parsed = parse_rules("open past 10pm", today=FRIDAY)
        assert parsed.open_at is not None
        assert (parsed.open_at.day, parsed.open_at.minute_of_day) == (FRIDAY, 22 * 60)

    def test_the_friday_night_constraint_matches_friday_night_hours(self) -> None:
        """End to end against the hours matcher, in both directions.

        The parse is only right if it selects the venue whose Friday shift runs into
        Saturday morning and rejects the one whose Thursday shift did. Before the day
        attribution was fixed this assertion held with the two businesses swapped, which
        is what "confidently wrong with no error anywhere" looks like.
        """
        parsed = parse_rules("ramen open until 2am", today=FRIDAY)
        assert parsed.open_at is not None

        friday_night = parse_hours({"Friday": "17:0-3:0"})
        thursday_night = parse_hours({"Thursday": "17:0-3:0"})
        day, minute = parsed.open_at.day, parsed.open_at.minute_of_day

        assert is_open_at(friday_night, day, minute) is ConstraintStatus.SATISFIED
        assert is_open_at(thursday_night, day, minute) is ConstraintStatus.VIOLATED

    def test_a_query_asked_after_midnight_means_the_night_already_underway(self) -> None:
        """At 01:00 on Saturday the Friday night in progress is the one being asked about.

        Without the clock minute, "open until 2am" would be read as the night *beginning*
        on Saturday and resolve to Sunday 02:00 — a day late, the mirror image of the bug
        above. `now_minute` is the only thing that distinguishes the two.
        """
        after_midnight = parse_rules("open until 2am", today="Saturday", now_minute=60)
        assert after_midnight.open_at is not None
        assert after_midnight.open_at.day == "Saturday"

        before_midnight = parse_rules("open until 2am", today="Saturday", now_minute=23 * 60)
        assert before_midnight.open_at is not None
        assert before_midnight.open_at.day == "Sunday"

    def test_an_evening_request_ignores_the_clock_minute(self) -> None:
        """`now_minute` may only move a small-hours time. 10pm is 10pm whenever you ask."""
        for now_minute in (60, 13 * 60, 23 * 60):
            parsed = parse_rules("open past 10pm", today=FRIDAY, now_minute=now_minute)
            assert parsed.open_at is not None
            assert parsed.open_at.day == FRIDAY, now_minute

    def test_the_week_wraps(self) -> None:
        """Sunday night's 1am is Monday, not a KeyError and not Saturday."""
        parsed = parse_rules("open until 1am", today="Sunday")
        assert parsed.open_at is not None
        assert parsed.open_at.day == "Monday"

    def test_the_latest_hour_asked_for_wins(self) -> None:
        parsed = parse_rules("late night spot open past 11pm", today=FRIDAY)
        assert parsed.open_at is not None
        assert parsed.open_at.minute_of_day == 23 * 60

    def test_a_stated_small_hours_hour_beats_a_vague_open_late(self) -> None:
        """ "Latest wins" has to mean latest *into the night*, not largest minute-of-day.

        "late night ramen open until 2am" asks for both 22:00 (the `late_night` default)
        and 02:00. Compared as minutes, 120 < 1320 and the hour the query stated outright
        loses to the convention — the constraint is silently relaxed to 22:00, so places
        that shut at 23:00 pass a filter written to exclude them. It is a loosening rather
        than an inversion, so the results stay plausible and nothing downstream complains.
        This phrasing is also exactly the doubled-up kind the `messy` query type collects,
        so the small-hours day attribution would never be exercised where it matters most.
        """
        parsed = parse_rules("late night ramen open until 2am", today=FRIDAY, now_minute=21 * 60)
        assert parsed.open_at == OpenAt(day="Saturday", minute_of_day=2 * 60)

    def test_the_order_the_two_hour_phrases_appear_in_does_not_matter(self) -> None:
        # Neither "first match wins" nor "last match wins": the stricter hour wins.
        for query in ("open until 2am, late night ramen", "late night ramen open until 2am"):
            parsed = parse_rules(query, today=FRIDAY, now_minute=21 * 60)
            assert parsed.open_at is not None, query
            assert parsed.open_at.minute_of_day == 2 * 60, query

    def test_both_hour_phrases_leave_the_embeddable_text(self) -> None:
        # The losing phrase is a constraint too, and the hours filter now owns that part of
        # the request; leaving it in would double-count it against the business text.
        parsed = parse_rules("late night ramen open until 2am", today=FRIDAY)
        assert parsed.semantic_text == "ramen"

    def test_without_a_weekday_the_constraint_is_declined_not_guessed(self) -> None:
        parsed = parse_rules("quiet cafe open past 10pm")  # no `today`
        assert parsed.open_at is None
        # Declined, so the phrase stays where retrieval can still use it. The request is
        # weakened, never falsified.
        assert "open past 10pm" in parsed.semantic_text
        assert parsed.attributes == {"NoiseLevel": "quiet"}

    def test_open_now_is_recognized_and_deliberately_not_resolved(self) -> None:
        # Resolving "now" needs a clock and a timezone; this module has neither by design.
        parsed = parse_rules("coffee open now", today=FRIDAY)
        assert parsed.open_at is None
        assert "open now" in parsed.semantic_text

    def test_an_ambiguous_bare_hour_after_open_at_is_refused(self) -> None:
        # "open at 9" could be 9am or 9pm and the construction gives no hint. A coin flip
        # would remove good results with no visible symptom.
        parsed = parse_rules("open at 9", today=FRIDAY)
        assert parsed.open_at is None
        assert parsed.semantic_text == "open at 9"

    def test_an_impossible_hour_is_ignored_rather_than_clamped(self) -> None:
        parsed = parse_rules("open past 99", today=FRIDAY)
        assert parsed.open_at is None


class TestWeekdayRules:
    """The weekday a query names, which must outrank the caller's clock.

    The parser had no weekday lexicon at all before this, so "open until 2am on friday"
    produced a well-formed constraint anchored on *today* — usually several days from the
    night the user named — while the word "friday" stayed in the embeddable text for BM25
    and the embedder to score. Nothing reported an error: the filter ran and returned
    another night's venues. It also contaminated the LLM-vs-rules ablation, because the
    rules row lost every day-naming query for a reason that reads as parser quality.
    """

    def test_a_named_weekday_anchors_the_constraint_instead_of_today(self) -> None:
        parsed = parse_rules("pho open until 2am on friday", today="Tuesday", now_minute=20 * 60)
        assert parsed.open_at == OpenAt(day=FRIDAY, minute_of_day=2 * 60)

    def test_an_evening_hour_on_a_named_weekday(self) -> None:
        parsed = parse_rules("bar open past 10pm on saturday", today="Tuesday", now_minute=20 * 60)
        assert parsed.open_at == OpenAt(day="Saturday", minute_of_day=22 * 60)

    def test_the_clock_minute_cannot_move_a_day_the_query_named(self) -> None:
        """The asymmetry that keeps this parser honest: `now_minute` resolves "tonight",
        and a query that named its day is not asking about tonight."""
        asked = [
            parse_rules("bar open until 1am on saturday", today="Tuesday", now_minute=now)
            for now in (None, 30, 12 * 60, 23 * 60)
        ]
        assert {p.open_at for p in asked} == {OpenAt(day="Saturday", minute_of_day=60)}

    def test_a_named_weekday_works_with_no_today_at_all(self) -> None:
        # The query supplied the day, so there is nothing left for the caller to supply.
        parsed = parse_rules("tacos open past 11pm on monday")
        assert parsed.open_at == OpenAt(day="Monday", minute_of_day=23 * 60)

    @pytest.mark.parametrize(
        ("query", "expected_day"),
        [
            ("open past 11pm on fri", FRIDAY),
            ("open past 11pm friday", FRIDAY),
            ("open past 11pm fridays", FRIDAY),
            ("open past 11pm this thurs", "Thursday"),
            ("open past 11pm next wed", "Wednesday"),
            ("open past 11pm on tues", "Tuesday"),
            ("open past 11pm on sat", "Saturday"),
            ("open past 11pm this sun", "Sunday"),
            ("open past 11pm every monday", "Monday"),
        ],
    )
    def test_the_spellings_people_write(self, query: str, expected_day: str) -> None:
        parsed = parse_rules(query, today="Tuesday")
        assert parsed.open_at is not None, query
        assert parsed.open_at.day == expected_day

    @pytest.mark.parametrize(
        "query",
        [
            "patio where we sat outside open past 11pm",
            "cafe in the sun open past 11pm",
            "a sunny spot open past 11pm",
        ],
    )
    def test_sat_and_sun_as_ordinary_words_are_not_weekdays(self, query: str) -> None:
        """Why those two abbreviations need an introducer and the others do not.

        "sat" and "sun" are ordinary English words, and a wrong day is an invisible wrong
        result set — far worse than missing an abbreviation some users write. So they count
        as weekdays only after a word that can introduce a day, and these queries fall back
        to the caller's weekday.
        """
        parsed = parse_rules(query, today="Tuesday")
        assert parsed.open_at is not None, query
        assert parsed.open_at.day == "Tuesday", query

    def test_every_spelling_the_weekday_rule_matches_is_a_known_day(self) -> None:
        """`_weekday` normalizes through `schema.normalize_day`, which raises on an unknown
        spelling. Pinned so a new entry in the pattern cannot turn a parse into a 500."""
        for word in (*rules._UNAMBIGUOUS_DAY_WORDS, "sat", "sun"):
            assert normalize_day(word) in WEEKDAYS, word

    def test_the_day_word_leaves_the_embeddable_text_once_it_is_a_filter(self) -> None:
        # Same reasoning as the hour phrase: the filter enforces the day exactly, so
        # scoring it a second time as text rewards a document that merely says "friday".
        # The joining "on" goes with it, which is why the pattern matches it.
        parsed = parse_rules("pho open until 2am on friday", today="Tuesday")
        assert parsed.semantic_text == "pho"

    def test_a_day_word_with_no_hour_is_left_in_the_text(self) -> None:
        # Nothing was encoded — a day alone cannot be checked against an hours interval —
        # so the word must stay where retrieval can still make soft use of it.
        parsed = parse_rules("friday brunch", today="Tuesday")
        assert parsed.open_at is None
        assert parsed.semantic_text == "friday brunch"

    def test_two_different_weekdays_decline_the_whole_constraint(self) -> None:
        """`OpenAt` holds one day, so "friday or saturday" cannot be encoded.

        Taking the first would enforce half the request and silently drop the other half.
        Declining keeps every word in the text, which is the same choice
        `schema.normalize_day` makes in refusing "weekend".

        This is the one place the two parsers can still differ on a day-naming query: the
        model is free to pick one of the two days, and the rule parser declines. That is a
        difference of coverage, which is what the LLM-vs-rules ablation is supposed to
        measure — not a difference of day arithmetic, which is what would have been
        mistaken for quality.
        """
        parsed = parse_rules("bar open past 1am on friday or saturday", today="Tuesday")
        assert parsed.open_at is None
        assert "friday" in parsed.semantic_text and "saturday" in parsed.semantic_text
        assert "1am" in parsed.semantic_text

    def test_the_same_weekday_twice_is_not_a_conflict(self) -> None:
        parsed = parse_rules("friday drinks, open past 1am on friday", today="Tuesday")
        assert parsed.open_at == OpenAt(day=FRIDAY, minute_of_day=60)


# ─────────────────────────────────────────────────────────────────────────────
# The two parsers must agree about which night a query means
# ─────────────────────────────────────────────────────────────────────────────

# Why this matters more than it looks. The ablation has a row for rule parsing and a row
# for LLM parsing, and the gap between them is reported as parser *quality*. If the two
# paths resolved the day of an hours constraint differently, part of that gap would be day
# arithmetic instead — and no metric downstream could tell the two apart. So agreement is
# a property to enumerate, not a case to spot-check: the parsers share
# `resolve_open_at_day`, and these tests are what keeps them sharing it.

# The hours to cross with every weekday. Daytime, evening, small hours, and both sides of
# the small-hours boundary — derived from the constant rather than written out, because the
# boundary is an unmeasured convention the failure analysis may move.
_AGREEMENT_MINUTES: tuple[int, ...] = (
    0,  # midnight: the first minute of the night
    60,
    2 * 60,
    SMALL_HOURS_END_MINUTE - 1,  # last minute that still belongs to the night
    SMALL_HOURS_END_MINUTE,  # first minute that does not
    SMALL_HOURS_END_MINUTE + 60,
    9 * 60,
    12 * 60,
    14 * 60 + 30,
    22 * 60,
    23 * 60 + 45,
)

# Instants a query might be asked at: no clock, inside the small hours, on both sides of
# the boundary, midday and late evening.
_AGREEMENT_NOW_MINUTES: tuple[int | None, ...] = (
    None,
    0,
    60,
    SMALL_HOURS_END_MINUTE - 1,
    SMALL_HOURS_END_MINUTE,
    12 * 60,
    23 * 60,
)


def _as_a_user_writes_it(minute_of_day: int) -> str:
    """ "2:30pm" for 870. A 12-hour clock with an explicit meridiem, so the rule parser
    never has to guess am/pm — this is a test about day arithmetic, not about that."""
    hour, minute = divmod(minute_of_day, 60)
    return f"{hour % 12 or 12}:{minute:02d}{'am' if hour < 12 else 'pm'}"


def _contract_payload(day_word: str | None, minute_of_day: int) -> dict[str, object]:
    """The `open_at` block the prompt's contract asks the model for.

    Transcribed from concierge/query/prompts/parse_query.txt: a weekday when the request
    names one, null when it does not ("the caller resolves it to the day the search is
    actually run"), and the time in 24-hour "HH:MM". Written out here rather than imported
    so that a change to either parser has to be reconciled against the contract, not
    against the other parser's current behaviour.
    """
    return {
        "day": day_word,
        "time": f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}",
    }


@pytest.mark.parametrize("day_word", [None, *(day.lower() for day in WEEKDAYS)])
@pytest.mark.parametrize("minute_of_day", _AGREEMENT_MINUTES)
def test_both_parsers_resolve_the_same_open_at(day_word: str | None, minute_of_day: int) -> None:
    """Enumerated: every weekday (and no weekday) crossed with every kind of hour, at
    every kind of reference instant, through both Stage 1 paths.

    The rule parser reads the query text; the schema validator reads the payload the prompt
    asks the model to produce for that same query. They must land on the same `OpenAt` —
    including the same calendar day, which is the half that was wrong in both directions
    before: the rule path ignored a named weekday, and the schema path shifted one.
    """
    time_text = _as_a_user_writes_it(minute_of_day)
    query = f"ramen open until {time_text}"
    if day_word is not None:
        query += f" on {day_word}"
    payload = {"semantic_text": "ramen", "open_at": _contract_payload(day_word, minute_of_day)}

    for today in WEEKDAYS:
        for now_minute in _AGREEMENT_NOW_MINUTES:
            from_rules = parse_rules(query, today=today, now_minute=now_minute).open_at
            from_schema = validate_parsed_query(
                payload, raw_query=query, today=today, now_minute=now_minute
            ).open_at

            where = (query, today, now_minute)
            # Both produced a constraint at all — otherwise two Nones would "agree" and
            # the enumeration would prove nothing.
            assert from_rules is not None, where
            assert from_rules.minute_of_day == minute_of_day, where
            assert from_rules == from_schema, where


@pytest.mark.parametrize(
    ("day_phrase", "day_word"),
    [("on fri", "Friday"), ("this sat", "Saturday"), ("on tues", "Tuesday"), ("sunday", "Sunday")],
)
def test_the_parsers_agree_on_abbreviated_and_bare_day_words_too(
    day_phrase: str, day_word: DayOfWeek
) -> None:
    """The rule parser accepts day spellings the LLM payload never carries ("fri", "sat"),
    so the abbreviation has to resolve to the same day the model's full weekday would."""
    for minute_of_day in (60, 22 * 60):
        query = f"bar open until {_as_a_user_writes_it(minute_of_day)} {day_phrase}"
        payload = {"semantic_text": "bar", "open_at": _contract_payload(day_word, minute_of_day)}

        from_rules = parse_rules(query, today="Wednesday", now_minute=60).open_at
        from_schema = validate_parsed_query(
            payload, raw_query=query, today="Wednesday", now_minute=60
        ).open_at

        assert from_rules is not None, query
        assert from_rules == from_schema, query


def test_both_parsers_decline_an_hours_constraint_with_no_day_anywhere() -> None:
    """The other side of agreement: when neither the query nor the caller names a day,
    both paths decline rather than guessing, so the ablation rows stay comparable."""
    query = "ramen open until 2am"
    payload = {"semantic_text": "ramen", "open_at": _contract_payload(None, 2 * 60)}

    assert parse_rules(query).open_at is None
    assert validate_parsed_query(payload, raw_query=query).open_at is None


class TestNoiseRules:
    @pytest.mark.parametrize(
        "query",
        [
            "quiet place to work",
            "a peaceful spot",
            "not loud",
            "nothing too loud",
            "somewhere that isn't noisy",
        ],
    )
    def test_quiet_requests(self, query: str) -> None:
        assert parse_rules(query).attributes == {"NoiseLevel": "quiet"}

    def test_an_explicit_request_for_loud(self) -> None:
        assert parse_rules("a loud sports bar").attributes == {"NoiseLevel": "loud"}


class TestAmenityRules:
    @pytest.mark.parametrize("query", ["with wifi", "has wi-fi", "free wi fi", "WIFI"])
    def test_wifi_variants(self, query: str) -> None:
        assert parse_rules(query).attributes == {"WiFi": "free"}

    @pytest.mark.parametrize(
        "query",
        [
            "with outdoor seating",
            "has a patio",
            "outdoor tables",
            "somewhere I can sit outside",
            "al fresco dining",
        ],
    )
    def test_outdoor_variants(self, query: str) -> None:
        assert parse_rules(query).attributes == {"OutdoorSeating": True}


class TestPriceRules:
    @pytest.mark.parametrize("query", ["cheap eats", "affordable dinner", "budget-friendly"])
    def test_vague_cheapness_uses_the_looser_ceiling(self, query: str) -> None:
        assert parse_rules(query).price_max == VAGUE_CHEAP_PRICE_MAX

    @pytest.mark.parametrize(
        ("query", "expected"), [("$ tacos", 1), ("$$ dinner", 2), ("$$$$ tasting menu", 4)]
    )
    def test_dollar_tiers_are_taken_at_face_value(self, query: str, expected: int) -> None:
        assert parse_rules(query).price_max == expected

    def test_a_literal_dollar_amount_is_not_a_price_tier(self) -> None:
        assert parse_rules("dinner under $30").price_max is None

    def test_the_tightest_ceiling_wins(self) -> None:
        assert parse_rules("cheap $ tacos").price_max == 1

    def test_expensive_is_recognized_but_cannot_be_encoded(self) -> None:
        # ParsedQuery has price_max and no price_min, so a price *floor* has nowhere to
        # live. Reported as a contract gap rather than faked with price_max=4, which would
        # filter nothing while looking like a working filter in the ablation.
        parsed = parse_rules("an expensive tasting menu")
        assert parsed.price_max is None
        assert "expensive" in parsed.semantic_text

    def test_not_expensive_reads_as_cheap_not_as_expensive(self) -> None:
        parsed = parse_rules("somewhere not expensive")
        assert parsed.price_max == VAGUE_CHEAP_PRICE_MAX


class TestCategoryRules:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("good pho", ("Vietnamese",)),
            ("a wine bar", ("Wine Bars",)),  # longest phrase wins over "bar"
            ("sandwiches", ("Sandwiches",)),  # trailing "es" tolerated
            ("cafes", ("Cafes",)),
            ("bbq", ("Barbeque",)),
        ],
    )
    def test_simple_category_words(self, query: str, expected: tuple[str, ...]) -> None:
        assert parse_rules(query).categories == expected

    def test_category_words_are_never_stripped_from_the_embeddable_text(self) -> None:
        # Not an exception to the strip rule but a consequence of it: a category check
        # cannot reject a business (`filters.check_categories` returns SATISFIED or
        # UNKNOWN, never VIOLATED), so nothing is being counted twice — and the word is
        # the more specific signal of the two, "ramen" saying more than "Japanese". The
        # rule and its premise are tested in `TestTheStripRule`.
        parsed = parse_rules("cheap ramen")
        assert parsed.categories == ("Ramen",)
        assert parsed.semantic_text == "ramen"

    def test_categories_are_capped(self) -> None:
        parsed = parse_rules("sushi ramen pizza tacos pho")
        assert len(parsed.categories) == MAX_CATEGORIES


class TestCategoryLexiconHygiene:
    """The lexicon is a parameter of an evaluated condition, so it gets guard rails.

    What is mechanically checkable is checked here. The part that is not — that entries
    are derived from the dataset's category taxonomy and never from reading the gold
    queries — is a human rule, written out at ``CATEGORY_LEXICON`` along with the overlap
    that already exists and why the entries are kept rather than quietly deleted.
    """

    def test_every_key_is_lowercase_so_the_lookup_cannot_raise(self) -> None:
        # The pattern matches case-insensitively and `_category` then looks the phrase up
        # by `.lower()`. An uppercase key would match and KeyError, which `parser.py`
        # degrades to a pass-through parse — a constraint silently lost, not a crash.
        for phrase in CATEGORY_LEXICON:
            assert phrase == phrase.lower(), phrase

    def test_every_value_is_a_non_empty_taxonomy_string(self) -> None:
        # The right-hand side is compared against the dataset's own category spellings
        # (`Business.categories`), so whitespace or emptiness here is a filter that can
        # never match anything.
        for phrase, category in CATEGORY_LEXICON.items():
            assert category and category == category.strip(), phrase

    def test_no_gold_query_appears_in_the_rule_parser(self) -> None:
        """Leakage guard, the same one the LLM prompt has (see ``test_query_llm.py``).

        Exact substring matching catches the realistic failure — pasting a gold query into
        a new rule or lexicon entry. It cannot detect a lexicon grown word by word to fit
        the gold set, which is why that rule is stated at the table itself and has to be
        kept by a human reviewing the diff.
        """
        gold = Path(__file__).resolve().parent.parent / "eval" / "gold_queries.csv"
        if not gold.exists():
            pytest.skip("eval/gold_queries.csv not written yet (owner-owned work)")

        source = Path(rules.__file__).read_text(encoding="utf-8").casefold()
        with gold.open(newline="") as fh:
            for row in csv.DictReader(fh):
                query = (row.get("query") or "").strip().casefold()
                if len(query) < 8:  # too short to be a meaningful match
                    continue
                assert query not in source, f"gold query leaked into rules.py: {query!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Messy input: the parser may miss, but must never invent or throw
# ─────────────────────────────────────────────────────────────────────────────


class TestMessyInput:
    def test_nothing_parseable_degrades_to_the_raw_query(self) -> None:
        query = "somewhere nice for my mom's birthday"
        parsed = parse_rules(query, today=FRIDAY)
        assert parsed.semantic_text == query
        assert parsed.n_hard_constraints == 0
        assert parsed.categories == ()

    @pytest.mark.parametrize("query", ["", "   ", "???", "a"])
    def test_degenerate_input_does_not_throw(self, query: str) -> None:
        parsed = parse_rules(query, today=FRIDAY)
        assert parsed.n_hard_constraints == 0
        assert parsed.raw == query

    def test_a_typo_is_missed_rather_than_guessed(self) -> None:
        # This is the measured gap against the LLM parser, not a bug: fuzzy matching a
        # constraint word risks inventing a filter the user never asked for. The phrase
        # stays in the text, where the embedding has a fair chance at it.
        parsed = parse_rules("quiett cafe with wiifi", today=FRIDAY)
        assert parsed.attributes == {}
        assert "quiett" in parsed.semantic_text
        assert "wiifi" in parsed.semantic_text

    def test_a_typo_in_the_verb_still_parses_but_a_typo_in_the_time_does_not(self) -> None:
        # The bare-preposition rule ("past <time>") carries no verb, so misspelling
        # "open" costs nothing — the hour is still read. That asymmetry is the point of
        # having the looser rule at all.
        parsed = parse_rules("opne past 10pm", today=FRIDAY)
        assert parsed.open_at is not None
        assert parsed.open_at.minute_of_day == 22 * 60
        assert parsed.semantic_text == "opne"

        # A typo inside the time itself has nothing left to match, and the phrase is left
        # for retrieval rather than guessed at.
        garbled = parse_rules("open past 1Opm", today=FRIDAY)
        assert garbled.open_at is None
        assert "1Opm" in garbled.semantic_text

    def test_multi_intent_collects_every_constraint_it_can(self) -> None:
        parsed = parse_rules(
            "cheap quiet cafe with wifi and outdoor seating open past 11pm", today=FRIDAY
        )
        assert parsed.attributes == {
            "NoiseLevel": "quiet",
            "WiFi": "free",
            "OutdoorSeating": True,
        }
        assert parsed.price_max == VAGUE_CHEAP_PRICE_MAX
        assert parsed.open_at is not None
        assert parsed.open_at.minute_of_day == 23 * 60
        assert parsed.categories == ("Cafes",)
        assert parsed.semantic_text == "cafe"

    def test_an_all_constraint_query_leaves_nothing_to_embed(self) -> None:
        # This test previously asserted the opposite — that the raw query came back — on
        # the reasoning that the nearest neighbours of an empty string are arbitrary.
        # That fallback is what made the strip rule non-uniform: identical constraint
        # phrases were removed from "somewhere with a patio" and restored in "free wifi",
        # depending only on whether the query happened to contain another word. The
        # behaviour is deliberately different now (`rules.strips_phrase`), and the
        # arbitrary-neighbours worry is handled where it belongs: no model is ever handed
        # this field directly — `concierge.rerank.model.query_text_for` is the single
        # definition of a stage's query text and already falls back to `raw`.
        parsed = parse_rules("open past 10pm with wifi", today=FRIDAY)
        assert parsed.semantic_text == ""
        assert parsed.attributes == {"WiFi": "free"}
        assert parsed.open_at is not None
        # Empty text, but never an empty request: the constraint-only retriever has both
        # of these to generate candidates from.
        assert parsed.n_hard_constraints == 2

    def test_a_proper_noun_containing_a_constraint_word_is_a_known_failure(self) -> None:
        # "The Quiet Shelf" is a business name, not a noise request. The rule parser cannot
        # tell, and this test records the real behaviour rather than wishing it away: the
        # name loses a word and gains a NoiseLevel filter. Handling it is exactly what the
        # LLM parser is for, and this is the kind of case the ablation should separate.
        parsed = parse_rules("The Quiet Shelf")
        assert parsed.attributes == {"NoiseLevel": "quiet"}
        assert parsed.semantic_text == "Shelf"


class TestPurityAndMeasurement:
    def test_the_same_input_always_gives_the_same_constraints(self) -> None:
        # No clock reads, so behaviour cannot drift with the time of day.
        first = parse_rules("quiet cafe open past 10pm", today=FRIDAY)
        second = parse_rules("quiet cafe open past 10pm", today=FRIDAY)
        assert (first.semantic_text, first.attributes, first.open_at) == (
            second.semantic_text,
            second.attributes,
            second.open_at,
        )

    def test_the_source_is_reported_as_rules(self) -> None:
        # The ablation reports rule parsing and LLM parsing as separate rows.
        assert parse_rules("cheap tacos").source is ParseSource.RULES

    def test_latency_is_measured_not_assumed(self) -> None:
        parsed = parse_rules("quiet cafe with wifi open past 10pm", today=FRIDAY)
        assert parsed.parse_latency_ms >= 0.0

    def test_a_day_abbreviation_is_accepted_for_today(self) -> None:
        parsed = parse_rules("open past 10pm", today="fri")
        assert parsed.open_at is not None
        assert parsed.open_at.day == FRIDAY

    def test_an_invalid_today_fails_loudly(self) -> None:
        # A silently-wrong weekday filters on hours the user never asked about.
        with pytest.raises(SchemaValidationError):
            parse_rules("open past 10pm", today="Funday")

    def test_the_rule_parser_only_ever_emits_supported_attributes(self) -> None:
        # Guards against a pattern inventing a key the filter stage cannot check.
        queries = [
            "quiet cafe with wifi and a patio open past 10pm",
            "loud cheap bar",
            "peaceful bookstore",
        ]
        for query in queries:
            for key in parse_rules(query, today=FRIDAY).attributes:
                assert key in SUPPORTED_ATTRIBUTES


# ─────────────────────────────────────────────────────────────────────────────
# The strip rule, over every constraint kind the parser can write
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StripCase:
    """One query that exercises one constraint kind, and what must survive it.

    `lifted` is the words the matching rule claimed. Whether they may still appear in
    `semantic_text` is not written here: it is read off `rules.strips_phrase(kind)`, so
    these cases check the rule rather than restating it.
    """

    kind: Writes
    query: str
    lifted: tuple[str, ...]
    residue: str
    today: DayOfWeek | None = FRIDAY


#: At least one case per `Writes` member, and `test_every_constraint_kind_has_a_case`
#: fails if a kind is ever added without one. Each query carries the non-constraint word
#: "place" so the residue is observable; the all-constraint queries are separately pinned
#: in `TestTheStripRuleOnTheMeasuredQueries`.
STRIP_RULE_CASES: tuple[StripCase, ...] = (
    StripCase(Writes.OPEN_AT, "place open past 10pm", ("open", "past", "10pm"), "place"),
    StripCase(Writes.OPEN_AT, "place open until 2am on friday", ("2am", "friday"), "place"),
    StripCase(Writes.ATTRIBUTES, "place with free wifi", ("free", "wifi"), "place"),
    StripCase(Writes.ATTRIBUTES, "place with a patio", ("patio",), "place"),
    StripCase(Writes.ATTRIBUTES, "quiet place", ("quiet",), "place"),
    StripCase(Writes.ATTRIBUTES, "loud place", ("loud",), "place"),
    StripCase(Writes.PRICE_MAX, "cheap place", ("cheap",), "place"),
    StripCase(Writes.PRICE_MAX, "$$ place", ("$$",), "place"),
    # Kept, and not as an exception: a category cannot reject (see the premise test).
    StripCase(Writes.CATEGORIES, "ramen place", ("ramen",), "ramen place"),
    # Recognized, unencodable, therefore never lifted — so nothing to remove.
    StripCase(Writes.NOTHING, "place open now", ("open", "now"), "place open now"),
    StripCase(Writes.NOTHING, "expensive place", ("expensive",), "expensive place"),
)


def _constraint_was_written(parsed: ParsedQuery, kind: Writes) -> bool:
    """Did the parse actually fill the field the rule claims to write?

    Without this, a case could "pass" the strip rule by failing to parse at all.
    """
    if kind is Writes.OPEN_AT:
        return parsed.open_at is not None
    if kind is Writes.ATTRIBUTES:
        return bool(parsed.attributes)
    if kind is Writes.PRICE_MAX:
        return parsed.price_max is not None
    if kind is Writes.CATEGORIES:
        return bool(parsed.categories)
    # NOTHING: the phrase is understood and deliberately not encoded, so a filter
    # appearing here would mean the parser invented one.
    return parsed.n_hard_constraints == 0 and not parsed.categories


class TestTheStripRule:
    """`rules.strips_phrase` is the whole policy; these tests check it holds everywhere.

    The property under test, in one sentence: a phrase the parser lifted leaves
    `semantic_text` if and only if it became a constraint the filters can reject a
    business for. The cases are parametrized over `Writes`, so a constraint kind added
    later cannot quietly acquire a different policy — it either appears in
    `STRIP_RULE_CASES` and is held to the rule, or `test_every_constraint_kind_has_a_case`
    fails.
    """

    @pytest.mark.parametrize("case", STRIP_RULE_CASES, ids=lambda c: c.query)
    def test_the_rule_holds_for_every_constraint_kind(self, case: StripCase) -> None:
        parsed = parse_rules(case.query, today=case.today)

        # A case that silently stopped parsing would make the rule check meaningless.
        assert _constraint_was_written(parsed, case.kind), case.query

        must_be_gone = strips_phrase(case.kind)
        for word in case.lifted:
            is_present = word.lower() in parsed.semantic_text.lower()
            assert is_present is not must_be_gone, (case.query, word, parsed.semantic_text)
        assert parsed.semantic_text == case.residue

    def test_every_constraint_kind_has_a_case(self) -> None:
        # The guard that makes the class above a property test rather than a list of
        # examples: a new rule declares a `writes`, and that kind must be exercised here.
        assert {case.kind for case in STRIP_RULE_CASES} == {rule.writes for rule in rules._RULES}

    def test_every_constraint_kind_is_classified_by_the_strip_rule(self) -> None:
        # Adding a `Writes` member forces a decision about rejection rather than
        # defaulting to "kept", which is how the inconsistency this rule replaced grew.
        not_rejecting = {kind for kind in Writes if kind not in REJECTING_CONSTRAINTS}
        assert not_rejecting == {Writes.CATEGORIES, Writes.NOTHING}

    def test_the_strip_rules_premise_holds_in_the_filters(self) -> None:
        """The premise, asserted against the real filter functions rather than assumed.

        The rule keeps category words because a category check cannot reject a business,
        and removes the others because theirs can. Both halves are facts about
        `concierge.retrieval.filters` (DECISIONS.md D225), so if a category miss ever
        starts returning VIOLATED, this is the test that says the rule must be revisited.
        """
        rejecting = (
            check_price_max(_business(attributes={PRICE_ATTRIBUTE: "4"}), 1),
            check_attribute(_business(attributes={"WiFi": "no"}), "WiFi", "free"),
            check_open_at(
                _business(hours={"Friday": (HoursInterval(9 * 60, 17 * 60),)}),
                OpenAt(day="Friday", minute_of_day=22 * 60),
            ),
        )
        for match in rejecting:
            assert match.status is ConstraintStatus.VIOLATED, match.name

        # Both shapes of category miss: a business tagged something else, and one with no
        # category data at all. Neither can be refuted, only left unknown.
        for business in (_business(categories=("Hardware Stores",)), _business(categories=())):
            assert check_categories(business, ["Ramen"]).status is not ConstraintStatus.VIOLATED

    def test_every_lifted_phrase_leaves_a_multi_constraint_query_at_once(self) -> None:
        # One query carrying every rejecting kind plus a category word. The residue is
        # exactly the words no filter took over, which is the whole claim of the rule.
        parsed = parse_rules(
            "a quiet cheap ramen place with wifi and a patio open past 10pm", today=FRIDAY
        )
        assert parsed.semantic_text == "ramen place"
        assert parsed.attributes == {
            "NoiseLevel": "quiet",
            "WiFi": "free",
            "OutdoorSeating": True,
        }
        assert parsed.price_max == VAGUE_CHEAP_PRICE_MAX
        assert parsed.open_at is not None
        assert parsed.categories == ("Ramen",)


class TestTheStripRuleOnTheMeasuredQueries:
    """The five queries from DECISIONS.md D224's table, which is where the defect was
    measured: three of them were stripped and two were not, for no stateable reason.

    Note which direction the fix went. The inconsistency was not in the per-rule flags —
    "wifi" and "patio" were both marked for removal — but in the raw-query fallback that
    used to fire when stripping emptied the text, re-inserting exactly the phrase the
    filter had just taken over. So the fix is that an all-constraint query now yields an
    empty `semantic_text`, which is safe for the reasons set out in `parse_rules`.
    """

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("somewhere with a patio", "somewhere"),
            ("cheap place", "place"),
            # Was "free wifi" — the phrase the WiFi filter already enforces.
            ("free wifi", ""),
            # Was "outdoor seating".
            ("outdoor seating", ""),
            # Was "open past 10pm". With a day to anchor it the hours filter owns it.
            ("open past 10pm", ""),
        ],
    )
    def test_the_measured_queries_now_strip_uniformly(self, query: str, expected: str) -> None:
        assert parse_rules(query, today=FRIDAY).semantic_text == expected

    def test_an_hours_phrase_with_no_day_is_kept_because_nothing_was_lifted(self) -> None:
        # The fifth row of the table read "open past 10pm" -> "open past 10pm", and with
        # no weekday anywhere that is *conformant*: `resolve_open_at` declines the
        # constraint, so the parser lifted nothing and has nothing to remove. The same
        # query with a day lands in the parametrized case above. Both behaviours follow
        # from one rule, which is the point.
        parsed = parse_rules("open past 10pm")  # no `today`
        assert parsed.open_at is None
        assert parsed.semantic_text == "open past 10pm"

    @pytest.mark.parametrize(
        "query",
        [
            "free wifi",
            "outdoor seating",
            "open past 10pm",
            "cheap",
            "quiet",
            "with wifi and a patio",
            "late night",
            "$",
        ],
    )
    def test_an_empty_residue_always_carries_a_hard_constraint(self, query: str) -> None:
        """Why an empty `semantic_text` is a usable state and not a dead end.

        Stripping only removes a phrase that became a rejecting filter, so emptying the
        text proves the query has at least one hard constraint — which is exactly what
        `concierge.retrieval.constraints_index` generates candidates from when the
        retrievers under-deliver (DECISIONS.md D224). The page therefore has a source of
        candidates in precisely the case the text no longer provides one.
        """
        parsed = parse_rules(query, today=FRIDAY)
        if not parsed.semantic_text:
            assert parsed.n_hard_constraints >= 1, query

    def test_a_query_with_no_constraints_is_never_emptied(self) -> None:
        # The converse guard: stripping must not be able to delete ordinary words.
        query = "somewhere nice to read a book"
        assert parse_rules(query, today=FRIDAY).semantic_text == query


def _business(**overrides: Any) -> Business:
    """A minimal business for the premise test. Only the overridden field matters."""
    fields: dict[str, Any] = {
        "business_id": "b-strip-rule",
        "name": "Test Place",
        "city": "Metro",
        "state": "XX",
        "latitude": None,
        "longitude": None,
        "stars": 4.0,
        "review_count": 10,
        "categories": ("Hardware Stores",),
        "attributes": {},
        "hours": {},
        "is_open": True,
    }
    fields.update(overrides)
    return Business(**fields)


class TestStripPhrases:
    def test_removes_spans_and_the_connectors_left_dangling(self) -> None:
        text = "a cafe with wifi and a patio"
        spans = [(text.index("wifi"), text.index("wifi") + 4), (text.index("patio"), len(text))]
        # "with", "and" and the second "a" all touched a removed phrase and go with it.
        # The leading "a" does not: string boundaries are deliberately not treated as
        # holes, so the parser never edits words that were not next to a constraint. A
        # stray leading article costs retrieval nothing.
        assert strip_phrases(text, spans) == "a cafe"

    def test_leaves_untouched_text_alone(self) -> None:
        assert strip_phrases("cozy spot to read a book", []) == "cozy spot to read a book"

    def test_tidies_punctuation_left_behind(self) -> None:
        text = "quiet, cheap cafe"
        assert strip_phrases(text, [(0, 5), (7, 12)]) == "cafe"

    def test_a_connector_far_from_a_removal_survives(self) -> None:
        # Only connectors touching a hole are dropped, so ordinary sentences keep their
        # function words.
        text = "a place with a view and good coffee"
        assert strip_phrases(text, []) == text
