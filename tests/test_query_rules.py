"""Tests for Stage 1 without the LLM: the JSON schema/validator and the rule parser.

Two things are being pinned here, and they are different kinds of claim.

For ``schema.py`` the claim is *nothing junk gets through*. Every test that asserts a
``SchemaValidationError`` is protecting a number in the ablation table: a bad filter value
silently empties or widens the result set, which looks exactly like a ranking failure and
is almost impossible to find later.

For ``rules.py`` the claim is *predictable degradation*. The rule parser is allowed to miss
constraints — that gap against the LLM parser is a measurement the project wants. What it
is not allowed to do is invent a constraint, throw, or hand retrieval an empty string. The
tests below therefore assert the stripped ``semantic_text`` exactly, and the known failure
modes are tested as the behaviour they actually have, not as aspirations.

No test here needs the Yelp dataset or an API key, so nothing is marked ``needs_data`` or
``needs_api``.
"""

from __future__ import annotations

import json

import pytest

from concierge.query.rules import (
    LATE_NIGHT_MINUTE,
    VAGUE_CHEAP_PRICE_MAX,
    parse_rules,
    strip_phrases,
)
from concierge.query.schema import (
    MAX_CATEGORIES,
    SUPPORTED_ATTRIBUTES,
    SchemaValidationError,
    describe_attributes,
    normalize_day,
    parse_clock_time,
    query_json_schema,
    validate_parse_payload,
    validate_parsed_query,
)
from concierge.types import ParseSource

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

    def test_the_day_is_the_evening_day_not_the_calendar_day(self) -> None:
        # Yelp files Friday's "17:0-3:0" under Friday, so a 2am request on Friday night is
        # OpenAt(Friday, 02:00) and HoursInterval.covers() resolves it.
        parsed = parse_rules("open until 2am", today=FRIDAY)
        assert parsed.open_at is not None
        assert parsed.open_at.day == FRIDAY

    def test_the_latest_hour_asked_for_wins(self) -> None:
        parsed = parse_rules("late night spot open past 11pm", today=FRIDAY)
        assert parsed.open_at is not None
        assert parsed.open_at.minute_of_day == 23 * 60

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
        # The category filter is coarse and the word is more specific than the category
        # ("ramen" vs "Japanese"), so retrieval must still see it. Stripping it would also
        # leave "cheap ramen" with nothing to embed.
        parsed = parse_rules("cheap ramen")
        assert parsed.categories == ("Ramen",)
        assert parsed.semantic_text == "ramen"

    def test_categories_are_capped(self) -> None:
        parsed = parse_rules("sushi ramen pizza tacos pho")
        assert len(parsed.categories) == MAX_CATEGORIES


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

    def test_an_all_constraint_query_keeps_something_to_embed(self) -> None:
        # Stripping everything would leave an empty string, whose nearest neighbours are
        # arbitrary. A double-counted constraint is the smaller error.
        parsed = parse_rules("open past 10pm with wifi", today=FRIDAY)
        assert parsed.semantic_text == "open past 10pm with wifi"
        assert parsed.attributes == {"WiFi": "free"}
        assert parsed.open_at is not None

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
