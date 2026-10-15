"""Tests for the Yelp attribute normalizer.

The fixture corpus (see ``tests/conftest.py``) carries one business per dataset
quirk, so most tests here read a real-shaped record rather than a hand-built
dict. Hand-built dicts are used only for cases the fixtures cannot contain, such
as a hostile payload.
"""

from __future__ import annotations

import ast

import pytest

from concierge.data.attributes import (
    ATTRIBUTE_PATH_SEP,
    ATTRIBUTE_VOCABULARY,
    PRICE_ATTRIBUTE,
    canonical_value,
    coverage,
    parse_attributes,
)
from concierge.types import Business

# ─────────────────────────────────────────────────────────────────────────────
# Decoding the stringified literals
# ─────────────────────────────────────────────────────────────────────────────


def test_u_prefixed_literal_is_valid_python3(raw_by_id):
    """Documents the premise this module is built on.

    The u-prefix is widely described as needing to be stripped before parsing.
    It does not: PEP 414 restored ``u'...'`` to Python 3, so ``literal_eval``
    handles it directly. This test pins that fact, because the module's regex
    fallback only runs on strings ``literal_eval`` rejects — if this ever
    stopped holding, that fallback would silently become the hot path.
    """
    assert raw_by_id["b001"]["attributes"]["WiFi"] == "u'free'"
    assert ast.literal_eval("u'free'") == "free"


def test_u_prefixed_values_normalize(raw_by_id):
    parsed = parse_attributes(raw_by_id["b001"]["attributes"])
    assert parsed["WiFi"] == "free"
    assert parsed["NoiseLevel"] == "quiet"


def test_plainly_quoted_values_normalize(raw_by_id):
    """b003 uses ``"'quiet'"`` rather than ``"u'quiet'"``."""
    parsed = parse_attributes(raw_by_id["b003"]["attributes"])
    assert parsed["NoiseLevel"] == "quiet"
    assert parsed["WiFi"] == "free"


def test_legacy_ur_prefix_uses_the_regex_fallback():
    """``ur'...'`` is genuinely invalid Python 3 syntax, unlike ``u'...'``."""
    with pytest.raises(SyntaxError):
        ast.literal_eval("ur'free'")
    assert parse_attributes({"WiFi": "ur'free'"}) == {"WiFi": "free"}


def test_booleans_and_ints_decode(raw_by_id):
    parsed = parse_attributes(raw_by_id["b001"]["attributes"])
    assert parsed["OutdoorSeating"] is True
    assert parsed[PRICE_ATTRIBUTE] == 2


def test_already_typed_values_pass_through():
    """Some exports hand back real JSON types instead of stringified literals."""
    parsed = parse_attributes({"OutdoorSeating": True, PRICE_ATTRIBUTE: 3})
    assert parsed["OutdoorSeating"] is True
    assert parsed[PRICE_ATTRIBUTE] == 3


def test_whole_blob_may_be_stringified():
    parsed = parse_attributes("{'WiFi': u'free', 'OutdoorSeating': 'True'}")
    assert parsed == {"WiFi": "free", "OutdoorSeating": True}


def test_literal_eval_does_not_execute_payloads(tmp_path):
    """The reason this module uses ``literal_eval`` and not ``eval``.

    These strings come from a third-party file. Under ``eval`` the payload below
    would write a file; under ``literal_eval`` it is rejected as a non-literal
    and kept as inert text.
    """
    marker = tmp_path / "pwned.txt"
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('x')"

    parsed = parse_attributes({"Hostile": payload})

    assert not marker.exists()
    assert isinstance(parsed["Hostile"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Missing vs. present — the load-bearing distinction
# ─────────────────────────────────────────────────────────────────────────────


def test_null_attributes_blob_is_empty(raw_by_id):
    assert raw_by_id["b005"]["attributes"] is None
    assert parse_attributes(raw_by_id["b005"]["attributes"]) == {}


def test_none_string_becomes_missing_not_a_value(raw_by_id):
    """b008 stores ``"None"`` as an attribute string.

    The key must be *absent*, so that a caller testing ``"OutdoorSeating" in
    attrs`` gets the right answer. A stored string ``"None"`` or a stored ``None``
    would both read as a value and could be mistaken for a fail.
    """
    assert raw_by_id["b008"]["attributes"]["OutdoorSeating"] == "None"
    parsed = parse_attributes(raw_by_id["b008"]["attributes"])
    assert "OutdoorSeating" not in parsed
    assert parsed["WiFi"] == "paid"
    assert parsed[PRICE_ATTRIBUTE] == 4


def test_json_null_value_becomes_missing():
    assert parse_attributes({"WiFi": None}) == {}


def test_empty_string_value_becomes_missing():
    assert parse_attributes({"WiFi": "", "NoiseLevel": "   "}) == {}


def test_false_is_a_value_not_a_missing(raw_by_id):
    """``False`` must survive: "has no patio" is not "we don't know"."""
    parsed = parse_attributes(raw_by_id["b002"]["attributes"])
    assert parsed["OutdoorSeating"] is False
    assert "OutdoorSeating" in parsed


def test_alcohol_none_is_a_real_value():
    """``Alcohol="none"`` means "serves no alcohol" and must not be dropped.

    This is why only Python ``None`` is treated as missing and the *string*
    ``"none"`` is kept — the two look alike in the raw file and mean opposite
    things.
    """
    assert parse_attributes({"Alcohol": "u'none'"}) == {"Alcohol": "none"}
    assert parse_attributes({"Alcohol": "None"}) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Nested dicts
# ─────────────────────────────────────────────────────────────────────────────


def test_nested_dicts_flatten_to_dotted_keys(raw_by_id):
    parsed = parse_attributes(raw_by_id["b001"]["attributes"])
    assert parsed["Ambience" + ATTRIBUTE_PATH_SEP + "casual"] is True
    assert parsed["Ambience.romantic"] is False
    assert parsed["Ambience.intimate"] is False
    assert parsed["BusinessParking.street"] is True
    assert parsed["BusinessParking.garage"] is False
    # The parent key is not kept: every consumer would otherwise have to re-walk
    # it and invent its own leaf spelling.
    assert "Ambience" not in parsed


def test_nested_none_leaves_are_dropped():
    parsed = parse_attributes({"Ambience": "{'casual': None, 'romantic': True}"})
    assert parsed == {"Ambience.romantic": True}


def test_nested_dict_of_only_none_leaves_yields_nothing():
    assert parse_attributes({"Ambience": "{'casual': None}"}) == {}


def test_deeply_nested_dicts_flatten_recursively():
    parsed = parse_attributes({"A": {"b": {"c": True}}})
    assert parsed == {"A.b.c": True}


# ─────────────────────────────────────────────────────────────────────────────
# Canonical vocabulary
# ─────────────────────────────────────────────────────────────────────────────


def test_spelling_variants_canonicalize():
    assert parse_attributes({"NoiseLevel": "u'very loud'"})["NoiseLevel"] == "very_loud"
    assert parse_attributes({"NoiseLevel": "'VERY_LOUD'"})["NoiseLevel"] == "very_loud"
    assert parse_attributes({"Alcohol": "u'beer & wine'"})["Alcohol"] == "beer_and_wine"
    assert parse_attributes({"Alcohol": "'full bar'"})["Alcohol"] == "full_bar"


def test_bare_unquoted_word_is_still_normalized():
    """``literal_eval`` rejects ``free``; we keep it as text and slug it."""
    assert parse_attributes({"WiFi": "free"})["WiFi"] == "free"


def test_wifi_boolean_is_translated_but_noiselevel_boolean_is_not():
    """WiFi=True has one sensible reading; NoiseLevel=True has none.

    Translating only where the reading is unambiguous keeps the module from
    inventing data, while not silently hiding every WiFi=True business from
    "cafe with wifi" queries.
    """
    assert parse_attributes({"WiFi": "True"})["WiFi"] == "free"
    assert parse_attributes({"WiFi": "False"})["WiFi"] == "no"
    assert parse_attributes({"NoiseLevel": "True"})["NoiseLevel"] is True


def test_out_of_vocabulary_value_is_kept_not_dropped():
    """Keeping it is the honest outcome: we know something, just not a spelling
    we recognize. Dropping it would read as UNKNOWN, a stronger claim."""
    assert parse_attributes({"NoiseLevel": "u'deafening'"})["NoiseLevel"] == "deafening"


def test_fixture_values_all_land_inside_the_declared_vocabulary(raw_businesses):
    """Guards the contract the query parser and the filters both depend on."""
    for business in raw_businesses:
        parsed = parse_attributes(business["attributes"])
        for key, allowed in ATTRIBUTE_VOCABULARY.items():
            if key in parsed:
                assert parsed[key] in allowed, (business["business_id"], key, parsed[key])


def test_canonical_value_matches_parse_attributes():
    """The query parser normalizes LLM output with this; it must agree."""
    assert canonical_value("NoiseLevel", "Very Loud") == "very_loud"
    assert canonical_value("WiFi", "free") == "free"
    assert canonical_value("WiFi", None) is None
    assert canonical_value(PRICE_ATTRIBUTE, "2") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Price
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["1", "2", "3", "4", 1, 4])
def test_price_coerces_to_int(raw):
    assert parse_attributes({PRICE_ATTRIBUTE: raw})[PRICE_ATTRIBUTE] == int(raw)


@pytest.mark.parametrize("raw", ["0", "5", "abc", "'$$'", "None"])
def test_unusable_price_is_missing(raw):
    """Price is compared arithmetically downstream, so a value outside 1-4 is
    not usable data and must read as missing rather than as a cheap place."""
    assert PRICE_ATTRIBUTE not in parse_attributes({PRICE_ATTRIBUTE: raw})


# ─────────────────────────────────────────────────────────────────────────────
# Coverage reporting
# ─────────────────────────────────────────────────────────────────────────────


def test_coverage_counts_all_businesses_in_the_denominator(raw_businesses):
    """b005 has no attributes at all and still counts, because the headline
    finding is "what fraction of the corpus can answer this constraint"."""
    assert len(raw_businesses) == 8
    rates = coverage(raw_businesses)

    # Hand-counted from tests/fixtures/businesses.json.
    assert rates[PRICE_ATTRIBUTE] == pytest.approx(7 / 8)
    assert rates["NoiseLevel"] == pytest.approx(6 / 8)
    assert rates["WiFi"] == pytest.approx(5 / 8)
    # b008's OutdoorSeating is the string "None" and must not be counted.
    assert rates["OutdoorSeating"] == pytest.approx(3 / 8)


def test_coverage_is_sorted_by_fill_rate_then_alphabetically(raw_businesses):
    rates = coverage(raw_businesses)
    keys = list(rates)
    assert keys == sorted(keys, key=lambda k: (-rates[k], k))


def test_coverage_includes_flattened_keys(raw_businesses):
    rates = coverage(raw_businesses)
    # Only b001 declares an Ambience dict.
    assert rates["Ambience.casual"] == pytest.approx(1 / 8)


def test_coverage_of_nothing_is_empty():
    assert coverage([]) == {}


def test_coverage_accepts_parsed_business_objects():
    business = Business(
        business_id="x1",
        name="Test",
        city="Metroville",
        state="AZ",
        latitude=None,
        longitude=None,
        stars=4.0,
        review_count=1,
        categories=("Cafes",),
        attributes={"WiFi": "free"},
        hours={},
        is_open=True,
    )
    assert coverage([business]) == {"WiFi": 1.0}
