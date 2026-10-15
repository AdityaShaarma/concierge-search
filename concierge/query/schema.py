"""Stage 1 contract: the JSON the LLM must return, and the validator that accepts it.

This module is the *boundary* between an untrusted text generator and a pipeline of
arithmetic filters. Everything here exists to keep one class of bug out of the system:
a plausible-looking but wrong filter value silently narrowing or widening the result
set, which would corrupt every number in the ablation table without ever raising.

Three design rules follow from that:

1. **Strict and total.** An unknown attribute key or an out-of-vocabulary value is an
   error, not something to pass through. ``NoiseLevel="pretty quiet"`` would match no
   business at all and would look exactly like a ranking failure; refusing it means the
   caller falls back to the rule parser and the search still returns something sane.
2. **One place for the vocabulary.** The JSON schema sent to the API and the validator
   that checks the reply are generated from the same ``SUPPORTED_ATTRIBUTES`` table, so
   they cannot drift apart. A prompt that advertises a value the validator rejects is a
   silent 100% failure rate on that constraint.
3. **Normalize on the way in.** Days become the dataset's own spelling ("Friday") and
   times become minutes from midnight, because that is what the hours matcher compares
   against. Doing it here means no downstream stage has to know that an LLM might say
   "fri" one time and "Friday" the next.

Canonical forms (see ``concierge.types``): ``DayOfWeek`` is the full weekday name as the
Yelp dataset spells it, and a time is an ``int`` minute-of-day in ``[0, 1440)`` so it can
be handed straight to ``HoursInterval.covers()``.

The prompt itself is deliberately NOT here — the LLM client owns it. ``describe_attributes()``
renders the vocabulary for that prompt so the client never has to restate it.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from concierge.types import DayOfWeek, GeoConstraint, OpenAt, ParsedQuery, ParseSource

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when a model reply cannot be turned into a trustworthy ``ParsedQuery``.

    A distinct type so the caller can write ``except SchemaValidationError: use_rules()``
    without also swallowing genuine bugs in the validator itself.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Attribute vocabulary
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    """One filterable business attribute.

    ``key`` is the Yelp attribute name exactly as it appears in ``business.json``, because
    ``ParsedQuery.attributes`` is matched against ``Business.attributes`` key for key. The
    moment we rename a key for prettiness, the filter stage has to carry a translation
    table, and translation tables rot.
    """

    key: str
    kind: Literal["bool", "enum"]
    values: tuple[str, ...] = ()
    description: str = ""


def _enum(key: str, values: tuple[str, ...], description: str) -> AttributeSpec:
    return AttributeSpec(key=key, kind="enum", values=values, description=description)


def _flag(key: str, description: str) -> AttributeSpec:
    return AttributeSpec(key=key, kind="bool", description=description)


# Why this list and not every Yelp attribute: each entry is a promise that the filter
# stage can check it and that the gold set plausibly asks for it. Attributes nobody
# queries cost prompt tokens and give the model more ways to be wrong. Values are the
# dataset's own normalized vocabulary (the data build resolves "u'free'" -> "free").
#
# Deliberately excluded: the nested attributes (Ambience, GoodForMeal, BusinessParking,
# Music). They are dicts-inside-strings in the raw data, so a request for one would need
# a dotted key path that `Business.attributes` does not express. "Good for a first date"
# therefore stays in `semantic_text`, which is the right home for a vibe anyway — an
# embedding handles "romantic" far better than a sparse boolean does.
SUPPORTED_ATTRIBUTES: dict[str, AttributeSpec] = {
    spec.key: spec
    for spec in (
        _enum("WiFi", ("free", "paid", "no"), "WiFi availability"),
        _enum(
            "NoiseLevel",
            ("quiet", "average", "loud", "very_loud"),
            "How loud the room is",
        ),
        _enum(
            "Alcohol",
            ("none", "beer_and_wine", "full_bar"),
            "What alcohol is served",
        ),
        _enum(
            "RestaurantsAttire",
            ("casual", "dressy", "formal"),
            "Expected dress code",
        ),
        _flag("OutdoorSeating", "Has outdoor/patio seating"),
        _flag("GoodForKids", "Suitable for children"),
        _flag("RestaurantsReservations", "Takes reservations"),
        _flag("RestaurantsTakeOut", "Offers takeout"),
        _flag("RestaurantsDelivery", "Offers delivery"),
        _flag("RestaurantsGoodForGroups", "Suitable for large groups"),
        _flag("RestaurantsTableService", "Has table service"),
        _flag("DogsAllowed", "Dogs allowed"),
        _flag("BikeParking", "Has bike parking"),
        _flag("HasTV", "Has a TV"),
        _flag("WheelchairAccessible", "Wheelchair accessible"),
        _flag("Caters", "Caters events"),
        _flag("HappyHour", "Has a happy hour"),
    )
}

# Yelp's RestaurantsPriceRange2 is 1-4 ($ to $$$$). It is carried as `price_max` rather
# than as an attribute because it is an *inequality* ("at most $$"), and the attribute
# dict only expresses equality.
PRICE_ATTRIBUTE_KEY = "RestaurantsPriceRange2"
PRICE_MIN = 1
PRICE_MAX = 4

# More than a handful of categories is a sign the model is listing synonyms rather than
# filtering, which over-constrains the search. Cap and keep the first few.
MAX_CATEGORIES = 3

# A radius is our knob, not the user's intent: nobody types "within 3.2 km". Unmeasured
# default — the right value depends on the metro's density and is a Phase 2 experiment.
DEFAULT_GEO_RADIUS_KM = 3.0
MAX_GEO_RADIUS_KM = 50.0

MINUTES_PER_DAY = 24 * 60

WEEKDAYS: tuple[DayOfWeek, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# Abbreviations an LLM actually emits. Anything outside this map is rejected rather than
# guessed at: a wrong day is a wrong result set with no visible symptom.
_DAY_ALIASES: dict[str, DayOfWeek] = {
    alias: day
    for day, aliases in {
        "Monday": ("monday", "mon", "mo"),
        "Tuesday": ("tuesday", "tue", "tues", "tu"),
        "Wednesday": ("wednesday", "wed", "we"),
        "Thursday": ("thursday", "thu", "thur", "thurs", "th"),
        "Friday": ("friday", "fri", "fr"),
        "Saturday": ("saturday", "sat", "sa"),
        "Sunday": ("sunday", "sun", "su"),
    }.items()
    for alias in aliases
}

_NAMED_TIMES = {"midnight": 0, "noon": 12 * 60, "midday": 12 * 60}


# ─────────────────────────────────────────────────────────────────────────────
# Normalizers — shared with the rule parser so both produce identical values
# ─────────────────────────────────────────────────────────────────────────────


def normalize_day(value: object) -> DayOfWeek:
    """Return the dataset's spelling of a weekday ("fri" -> "Friday").

    Raises:
        SchemaValidationError: for anything that is not a single weekday. "weekend" and
            "weekday" are refused on purpose — ``OpenAt`` holds one day, so accepting a
            set here would mean quietly dropping half of what was asked for.
    """
    if not isinstance(value, str):
        raise SchemaValidationError(f"open_at.day must be a string, got {type(value).__name__}")
    key = value.strip().lower().rstrip(".")
    if key in _DAY_ALIASES:
        return _DAY_ALIASES[key]
    raise SchemaValidationError(
        f"open_at.day must be one of {list(WEEKDAYS)} (or a common abbreviation), got {value!r}"
    )


def parse_clock_time(value: object) -> int:
    """Return minute-of-day in ``[0, 1440)`` for a clock time.

    Accepts the forms a model or a user actually writes: "22:00", "22:0" (the dataset's
    own style), "10pm", "7:30 PM", "midnight", "noon", and a bare 24-hour "19".

    Midnight is 0, not 1440: ``HoursInterval.covers()`` compares against a minute-of-day,
    so 1440 would never match anything. "24:00" is therefore folded to 0 — the same
    instant, expressed the way the rest of the pipeline can use.
    """
    if isinstance(value, bool):  # bool is an int subclass; a flag is not a time
        raise SchemaValidationError("time must be a clock time, got a boolean")
    if isinstance(value, int):
        return _validated_minute(value * 60 if value <= 24 else value, source=value)
    if not isinstance(value, str):
        raise SchemaValidationError(f"time must be a string, got {type(value).__name__}")

    text = value.strip().lower().replace(".", "")
    if text in _NAMED_TIMES:
        return _NAMED_TIMES[text]

    meridiem: str | None = None
    for suffix in ("am", "pm"):
        if text.endswith(suffix):
            meridiem = suffix
            text = text[: -len(suffix)].strip()
            break

    hour_text, _, minute_text = text.partition(":")
    try:
        hour = int(hour_text)
        minute = int(minute_text) if minute_text else 0
    except ValueError as exc:
        raise SchemaValidationError(f"could not read a clock time from {value!r}") from exc

    if not 0 <= minute < 60:
        raise SchemaValidationError(f"minutes out of range in {value!r}")
    if meridiem is not None:
        if not 1 <= hour <= 12:
            raise SchemaValidationError(f"hour out of range for a 12-hour time: {value!r}")
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    return _validated_minute(hour * 60 + minute, source=value)


def _validated_minute(minute: int, *, source: object) -> int:
    if minute == MINUTES_PER_DAY:  # "24:00" means midnight
        return 0
    if not 0 <= minute < MINUTES_PER_DAY:
        raise SchemaValidationError(f"time out of range: {source!r}")
    return minute


def normalize_attribute(key: object, value: object) -> tuple[str, Any]:
    """Validate one requested attribute and return its canonical ``(key, value)``.

    Raises:
        SchemaValidationError: if the key is not filterable or the value is outside that
            attribute's vocabulary. Both are unrecoverable here by design — see the
            module docstring on why junk must not pass through.
    """
    if not isinstance(key, str):
        raise SchemaValidationError(f"attribute keys must be strings, got {key!r}")
    spec = SUPPORTED_ATTRIBUTES.get(key) or SUPPORTED_ATTRIBUTES.get(_match_key_case(key))
    if spec is None:
        raise SchemaValidationError(
            f"unknown attribute {key!r}; supported: {sorted(SUPPORTED_ATTRIBUTES)}"
        )

    if spec.kind == "bool":
        return spec.key, _coerce_bool(spec.key, value)

    if not isinstance(value, str):
        raise SchemaValidationError(
            f"{spec.key} must be one of {list(spec.values)}, got {value!r}"
        )
    candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
    if candidate not in spec.values:
        raise SchemaValidationError(
            f"{spec.key} must be one of {list(spec.values)}, got {value!r}"
        )
    return spec.key, candidate


def _match_key_case(key: str) -> str:
    """Recover the canonical casing of an attribute key ("wifi" -> "WiFi").

    Case is the one deviation worth forgiving: the model gets the concept right and the
    casing wrong often enough that rejecting it would cost real recall, and there is no
    ambiguity to resolve — the lookup is exact apart from case.
    """
    lowered = key.strip().lower()
    for canonical in SUPPORTED_ATTRIBUTES:
        if canonical.lower() == lowered:
            return canonical
    return key


def _coerce_bool(key: str, value: object) -> bool:
    # JSON from an LLM sometimes stringifies booleans. Accepting "true" is safe because
    # the mapping is unambiguous; accepting 1/0 is not (a price of 1 is not "yes").
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise SchemaValidationError(f"{key} must be true or false, got {value!r}")


def normalize_price_max(value: object) -> int:
    """Validate a price ceiling on Yelp's 1-4 scale."""
    if isinstance(value, bool):
        raise SchemaValidationError("price_max must be an integer 1-4, got a boolean")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise SchemaValidationError(f"price_max must be an integer 1-4, got {value!r}")
    if not PRICE_MIN <= value <= PRICE_MAX:
        raise SchemaValidationError(f"price_max must be between 1 and 4, got {value}")
    return value


# ─────────────────────────────────────────────────────────────────────────────
# The JSON schema sent to the API
# ─────────────────────────────────────────────────────────────────────────────

Dialect = Literal["openapi", "jsonschema"]


def _attribute_properties() -> dict[str, dict[str, Any]]:
    """One property per supported attribute, rather than a free-form map.

    Enumerating them costs prompt tokens but buys two things: the model is told the exact
    vocabulary instead of inventing one, and the structured-output decoder can refuse an
    invalid value before it ever reaches us. A free-form object would push all of that
    onto the validator, where the only remedy is to throw the whole parse away.
    """
    props: dict[str, dict[str, Any]] = {}
    for spec in SUPPORTED_ATTRIBUTES.values():
        if spec.kind == "bool":
            props[spec.key] = {"type": "boolean", "description": spec.description}
        else:
            props[spec.key] = {
                "type": "string",
                "enum": list(spec.values),
                "description": spec.description,
            }
    return props


def _base_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "A place-search request split into embeddable text and hard filters.",
        "properties": {
            "semantic_text": {
                "type": "string",
                "description": (
                    "The request with every constraint expressed in the other fields "
                    "removed, leaving what the place should be like. Keep dish, cuisine "
                    "and vibe words here."
                ),
            },
            "open_at": {
                "type": "object",
                "nullable": True,
                "description": (
                    "Only when the request names a day or time the place must be open. "
                    "For 'open past 10pm' use the time the place must still be open. "
                    "Null if the request says nothing about hours."
                ),
                "properties": {
                    "day": {
                        "type": "string",
                        "enum": list(WEEKDAYS),
                        "nullable": True,
                        "description": (
                            "Weekday the request names. Null when it names only a time "
                            "('open past 10pm') — the day is then resolved at query time."
                        ),
                    },
                    "time": {
                        "type": "string",
                        "description": "24-hour HH:MM, e.g. '22:00'.",
                    },
                },
                "required": ["time"],
            },
            "attributes": {
                "type": "object",
                "nullable": True,
                "description": (
                    "Only attributes the request actually asks for. Omit everything else; "
                    "do not guess."
                ),
                "properties": _attribute_properties(),
            },
            "price_max": {
                "type": "integer",
                "nullable": True,
                "description": "Highest acceptable price tier, 1 ($) to 4 ($$$$).",
            },
            "categories": {
                "type": "array",
                "nullable": True,
                "description": (
                    f"At most {MAX_CATEGORIES} Yelp business categories, e.g. 'Ramen', "
                    "'Coffee & Tea'. Empty if the request names no business type."
                ),
                "items": {"type": "string"},
            },
            "geo": {
                "type": "object",
                "nullable": True,
                "description": "Only when the request names a place or neighbourhood.",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The location phrase as typed, e.g. 'downtown'.",
                    },
                    "radius_km": {
                        "type": "number",
                        "description": "How far out to search, in kilometres.",
                    },
                },
                "required": ["label"],
            },
        },
        "required": ["semantic_text"],
    }


def query_json_schema(dialect: Dialect = "openapi") -> dict[str, Any]:
    """The schema to send with the request, as a fresh dict.

    Two dialects because the two configured providers disagree. Gemini's ``response_schema``
    takes the OpenAPI 3.0 subset, where optional means ``"nullable": true``. OpenAI-style
    strict JSON-schema modes (Groq, our fallback provider) reject ``nullable`` and want a
    ``["object", "null"]`` type union plus every property listed in ``required``. Keeping
    one source and converting is how the two stay in step.

    Returns a copy on every call: a shared mutable dict that one provider's client edits
    in place is a bug that only shows up when the fallback runs.
    """
    schema = copy.deepcopy(_base_schema())
    if dialect == "openapi":
        return schema
    if dialect == "jsonschema":
        return _to_strict_jsonschema(schema)
    raise ValueError(f"unknown schema dialect: {dialect!r}")


def _to_strict_jsonschema(node: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``nullable`` as a type union and make every object closed and total."""
    if node.get("nullable"):
        del node["nullable"]
        node["type"] = _with_null(node["type"])
    if "object" in (node.get("type") or ()):
        properties: dict[str, Any] = node.get("properties", {})
        # Captured before it is overwritten: these are the fields that genuinely must be
        # answered, and they are the only ones that stay non-nullable.
        truly_required = set(node.get("required", ()))
        for name, child in properties.items():
            if not isinstance(child, dict):
                continue
            _to_strict_jsonschema(child)
            if name not in truly_required:
                _allow_null(child)
        node["additionalProperties"] = False
        # Strict mode has no notion of an optional property: every key must be listed in
        # `required`, and optionality is expressed by allowing null instead. Without the
        # nullifying step above, this would demand a value for *every* attribute — the
        # model would be forced to assert something about WiFi, dogs and happy hour on
        # every single query, which is exactly the hallucinated-constraint failure the
        # validator exists to prevent.
        node["required"] = list(properties)
    items = node.get("items")
    if isinstance(items, dict):
        _to_strict_jsonschema(items)
    return node


def _with_null(type_value: Any) -> list[str]:
    types = list(type_value) if isinstance(type_value, list) else [type_value]
    return types if "null" in types else [*types, "null"]


def _allow_null(node: dict[str, Any]) -> None:
    """Let a property be answered with null, enum included.

    An enumerated field needs null in the enum as well as in the type, or a strict decoder
    has no legal way to say "the request did not ask for this".
    """
    if "type" in node:
        node["type"] = _with_null(node["type"])
    enum = node.get("enum")
    if isinstance(enum, list) and None not in enum:
        enum.append(None)


# Convenience for callers that just want the Gemini-shaped dict. Built on demand so the
# module has no mutable shared state.
QUERY_JSON_SCHEMA: dict[str, Any] = query_json_schema("openapi")


def describe_attributes() -> str:
    """Render the attribute vocabulary as prompt-ready lines.

    Exists so the LLM client can state the vocabulary without copying it. A prompt that
    lists a value this module rejects fails 100% of the time on that constraint and looks
    like a model-quality problem, so the two must come from one table.
    """
    lines = []
    for spec in SUPPORTED_ATTRIBUTES.values():
        allowed = "true | false" if spec.kind == "bool" else " | ".join(spec.values)
        lines.append(f"- {spec.key} ({allowed}): {spec.description}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
#
# Two layers, because two callers need different things:
#
#   validate_parse_payload  raw reply -> a canonical, JSON-safe payload dict. This is what
#                           the LLM client validates with and what the parse cache stores.
#                           It deliberately does NOT resolve a weekday or coordinates:
#                           both depend on when and where the query was asked, and a cache
#                           entry that froze "Friday" would serve a Friday filter forever.
#   validate_parsed_query   that payload (or a raw reply) -> ParsedQuery, with the
#                           request-time context (today, a geo resolver) supplied.
#
# Strictness lives in the first layer, so the client, the cache and the pipeline cannot
# disagree about what counts as a valid parse.
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_TOP_LEVEL = frozenset(_base_schema()["properties"])

GeoResolver = Callable[[str], tuple[float, float] | None]


def validate_parse_payload(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    """Validate a model reply and return it canonicalized, or raise.

    The returned dict always has all six keys, with ``None`` where the request said
    nothing, and contains only JSON types — it is written straight into the parse cache,
    so an object in here would break the round trip.

    Args:
        payload: the decoded JSON object, or the raw JSON text. Markdown fences and other
            provider packaging are the client's problem, not this function's; keeping the
            boundary narrow means one place to look when a provider changes its output.

    Raises:
        SchemaValidationError: on any structural or vocabulary problem. Catch it and fall
            back to ``concierge.query.rules`` — see the module docstring for why junk must
            never be passed through instead.
    """
    data = _as_mapping(payload)

    unknown = set(data) - _ALLOWED_TOP_LEVEL
    if unknown:
        # Not pedantry: an unexpected key usually means the model answered a different
        # question (e.g. invented "cuisine"), and the fields we *did* read are then
        # suspect too.
        raise SchemaValidationError(f"unexpected keys in reply: {sorted(unknown)}")

    semantic_text = data.get("semantic_text")
    if semantic_text is not None and not isinstance(semantic_text, str):
        raise SchemaValidationError(
            f"semantic_text must be a string, got {type(semantic_text).__name__}"
        )

    return {
        # May legitimately be empty: a fully-constrained request ("open after 10pm with
        # outdoor seating") leaves nothing to embed. The *caller* supplies the fallback,
        # because only the caller knows the raw query.
        "semantic_text": (semantic_text or "").strip(),
        "open_at": _canonical_open_at(data.get("open_at")),
        "attributes": _validate_attributes(data.get("attributes")),
        "price_max": (
            None if data.get("price_max") is None else normalize_price_max(data["price_max"])
        ),
        "categories": list(_validate_categories(data.get("categories"), None)),
        "geo": _canonical_geo(data.get("geo")),
    }


# Alias for callers that resolve the validator by name. Kept because the symbol name was
# not fixed in the shared contract and two spellings are cheaper than a broken fallback.
validate_payload = validate_parse_payload


def validate_parsed_query(
    payload: str | Mapping[str, Any],
    *,
    raw_query: str,
    source: ParseSource = ParseSource.LLM,
    parse_latency_ms: float = 0.0,
    today: DayOfWeek | None = None,
    geo_resolver: GeoResolver | None = None,
    known_categories: frozenset[str] | None = None,
) -> ParsedQuery:
    """Turn a model reply (or an already-canonical payload) into a ``ParsedQuery``.

    Args:
        payload: a reply, raw or decoded, or a payload from ``validate_parse_payload``.
            Re-validating a canonical payload is cheap and idempotent, so there is one
            code path rather than a trusted and an untrusted one.
        raw_query: what the user typed. Carried through untouched for logging, caching and
            the ``semantic_text`` fallback.
        source: ``LLM`` or ``CACHE``. The ablation reports them separately because a cache
            hit has the same quality and a fraction of the latency.
        parse_latency_ms: measured by the caller, which owns the clock.
        today: the weekday to use when the request named a time but no day ("open past
            10pm"). A parameter, never a clock read: resolving it here would make the
            result depend on the day the code ran, and a cached payload would freeze it.
        geo_resolver: maps a location phrase to coordinates. Without one, a geo constraint
            cannot be built — see ``_resolve_geo``.
        known_categories: the real category vocabulary, when the caller has the index
            loaded. Out-of-vocabulary categories are dropped rather than rejected: a
            hallucinated category is advisory noise, while a hallucinated *attribute*
            would silently empty the result set.

    Raises:
        SchemaValidationError: on any structural or vocabulary problem.
    """
    data = validate_parse_payload(payload)

    categories = _filter_to_known(tuple(data["categories"]), known_categories)
    open_at = _open_at_from_payload(data["open_at"], today)
    geo, unresolved_geo_label = _resolve_geo(data["geo"], geo_resolver)

    semantic_text = data["semantic_text"] or _semantic_text_fallback(raw_query, categories)
    if unresolved_geo_label:
        semantic_text = _fold_text(semantic_text, unresolved_geo_label)

    return ParsedQuery(
        raw=raw_query,
        semantic_text=semantic_text,
        open_at=open_at,
        attributes=dict(data["attributes"]),
        price_max=data["price_max"],
        categories=categories,
        geo=geo,
        source=source,
        parse_latency_ms=parse_latency_ms,
    )


def _as_mapping(payload: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"reply was not valid JSON: {exc}") from exc
    else:
        decoded = payload
    if not isinstance(decoded, Mapping):
        raise SchemaValidationError(
            f"reply must be a JSON object, got {type(decoded).__name__}"
        )
    return decoded


def _semantic_text_fallback(raw_query: str, categories: tuple[str, ...]) -> str:
    """What to embed when the model left ``semantic_text`` empty.

    Embedding an empty string is the worst option: the nearest neighbours of nothing are
    arbitrary, so the dense arm would contribute pure noise to the fusion. Categories are
    the next best summary of intent; the raw query is the last resort, accepting the
    double-count explained in ``concierge.query.rules``.
    """
    fallback = ", ".join(categories) if categories else raw_query.strip()
    logger.debug("semantic_text was empty; falling back to %r", fallback)
    return fallback


def _validate_attributes(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaValidationError(
            f"attributes must be an object, got {type(value).__name__}"
        )
    normalized: dict[str, Any] = {}
    for key, raw_value in value.items():
        if raw_value is None:
            continue  # a null attribute is "not requested", not a constraint on null
        canonical_key, canonical_value = normalize_attribute(key, raw_value)
        normalized[canonical_key] = canonical_value
    return normalized


def _validate_categories(
    value: object, known_categories: frozenset[str] | None
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        raise SchemaValidationError(
            f"categories must be an array, got {type(value).__name__}"
        )
    kept: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SchemaValidationError(f"categories must be strings, got {item!r}")
        name = item.strip()
        if name and name not in kept:
            kept.append(name)
    return _filter_to_known(tuple(kept[:MAX_CATEGORIES]), known_categories)


def _filter_to_known(
    categories: tuple[str, ...], known_categories: frozenset[str] | None
) -> tuple[str, ...]:
    if known_categories is None:
        return categories
    kept = tuple(name for name in categories if name in known_categories)
    for name in categories:
        if name not in known_categories:
            logger.debug("dropping category %r: not in the index vocabulary", name)
    return kept


def _canonical_open_at(value: object) -> dict[str, Any] | None:
    """Canonicalize an hours constraint to ``{"day": "Friday" | None, "time": "HH:MM"}``.

    ``day`` stays optional and may be null. "open past 10pm" names a time and no day, and
    the day it means is the day the question is asked — so it is resolved at request time
    by the caller, after the cache. Resolving it here would bake the first caller's weekday
    into every later cache hit, a bug that only surfaces as wrong results the next morning.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"open_at must be an object, got {type(value).__name__}")
    unknown = set(value) - {"day", "time"}
    if unknown:
        raise SchemaValidationError(f"unexpected keys in open_at: {sorted(unknown)}")

    if value.get("time") is None:
        # A day with no time cannot be checked against an hours interval, which is
        # minute-based, and inventing 00:00 would filter on a time nobody asked about.
        #
        # Refusing the whole parse (rather than dropping just this field) is the deliberate
        # choice: an `open_at` block at all means the query almost certainly *has* an hours
        # constraint, and silently dropping it yields confidently wrong results — a place
        # that shuts at 6pm ranked top for "open past 10pm". The caller's fallback is the
        # rule parser, which reads exactly that phrase correctly, so refusing produces a
        # better parse than salvaging this one does.
        raise SchemaValidationError("open_at has a day but no 'time'; nothing to check")

    minute = parse_clock_time(value["time"])
    day = None if value.get("day") in (None, "") else normalize_day(value["day"])
    return {"day": day, "time": f"{minute // 60:02d}:{minute % 60:02d}"}


def _open_at_from_payload(block: Mapping[str, Any] | None, today: DayOfWeek | None) -> OpenAt | None:
    if block is None:
        return None
    day = block.get("day") or (normalize_day(today) if today is not None else None)
    if day is None:
        # Guessing a weekday would filter on hours the user never mentioned. Declining
        # weakens the request instead of falsifying it — the same choice the rule parser
        # makes when it is given no `today`.
        logger.info("hours constraint names no day and no `today` was given; dropping it")
        return None
    return OpenAt(day=day, minute_of_day=parse_clock_time(block["time"]))


def _canonical_geo(value: object) -> dict[str, Any] | None:
    """Canonicalize a location constraint to ``{"label": str, "radius_km": float}``.

    The model is asked for a *label* and never for coordinates, because it has no reliable
    way to know where "downtown" is for the metro in scope and a confidently wrong latitude
    is worse than no filter at all. Resolution is the caller's job (per DECISIONS D003, a
    city centroid plus a radius).
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"geo must be an object, got {type(value).__name__}")
    unknown = set(value) - {"label", "radius_km"}
    if unknown:
        raise SchemaValidationError(f"unexpected keys in geo: {sorted(unknown)}")

    label = value.get("label")
    if not isinstance(label, str) or not label.strip():
        raise SchemaValidationError("geo.label must be a non-empty string")

    radius = value.get("radius_km")
    radius_km = DEFAULT_GEO_RADIUS_KM if radius is None else _validate_radius(radius)
    return {"label": label.strip(), "radius_km": radius_km}


def _resolve_geo(
    block: Mapping[str, Any] | None, geo_resolver: GeoResolver | None
) -> tuple[GeoConstraint | None, str]:
    """Build a geo constraint, or report the label we could not resolve.

    When there is no resolver, or the label is unknown to it, the constraint is dropped and
    the label is returned so it can go back into ``semantic_text``. Silently losing part of
    the request would make a geo query look like a ranking failure; putting the words back
    at least lets BM25 match an address or city field.
    """
    if block is None:
        return None, ""
    label = str(block["label"])
    point = geo_resolver(label) if geo_resolver else None
    if point is None:
        logger.info("geo label %r not resolved; folding it back into semantic_text", label)
        return None, label
    latitude, longitude = point
    return (
        GeoConstraint(
            latitude=latitude,
            longitude=longitude,
            radius_km=float(block["radius_km"]),
            label=label,
        ),
        "",
    )


def _validate_radius(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaValidationError(f"geo.radius_km must be a number, got {value!r}")
    radius = float(value)
    if not 0 < radius <= MAX_GEO_RADIUS_KM:
        raise SchemaValidationError(
            f"geo.radius_km must be in (0, {MAX_GEO_RADIUS_KM}], got {radius}"
        )
    return radius


def _fold_text(text: str, phrase: str) -> str:
    """Append a phrase to the embeddable text unless it is already in there."""
    if phrase.lower() in text.lower():
        return text
    return f"{text} {phrase}".strip()
