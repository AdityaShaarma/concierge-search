"""Normalize the Yelp ``attributes`` blob into clean Python values.

The dataset stores attribute values as *stringified Python literals* rather than
JSON, e.g. ``"u'free'"``, ``"'quiet'"``, ``"True"``, ``"None"``, and nested
dictionaries as strings like ``"{'romantic': False, 'casual': True}"``. Every
stage downstream (filters, document rendering, constraint explanations, the
LambdaMART features) reads attributes, so this is parsed exactly once, here.

Three decisions in this module are load-bearing for correctness:

1. **``ast.literal_eval``, never ``eval``.** These strings come from a 5 GB
   third-party file. ``eval`` on them is arbitrary code execution: a value of
   ``"__import__('os').system('rm -rf ~')"`` would run. ``literal_eval`` walks
   the parsed AST and accepts only literal nodes, so the worst case is a
   ``ValueError``. This is cheap to get right and expensive to get wrong.

2. **Missing means the key is absent.** The string ``"None"`` parses to Python
   ``None``, which we drop from the result entirely. Nothing downstream may read
   missing data as a pass *or* a fail — that is what ``ConstraintStatus.UNKNOWN``
   exists for, and a ``False`` left lying around where there was no data would
   quietly turn "we don't know" into "this place has no patio".

3. **Nested dicts are flattened to dotted keys** (``Ambience.casual``), so a
   constraint can target one leaf directly.

Attribute coverage in this dataset is sparse; :func:`coverage` measures it,
because how sparse it is drives the missing-data policy in retrieval and is a
headline finding in the failure analysis.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

__all__ = [
    "ATTRIBUTE_PATH_SEP",
    "ATTRIBUTE_VOCABULARY",
    "PRICE_ATTRIBUTE",
    "PRICE_RANGE",
    "canonical_value",
    "coverage",
    "parse_attributes",
]

# Separator for flattened nested attributes. A constant rather than a literal "."
# so the query parser and the filters cannot disagree about the spelling of
# "Ambience.casual".
ATTRIBUTE_PATH_SEP: Final = "."

PRICE_ATTRIBUTE: Final = "RestaurantsPriceRange2"
PRICE_RANGE: Final[tuple[int, ...]] = (1, 2, 3, 4)

# The canonical vocabulary for every string-valued attribute we rely on, exposed
# so that the LLM query parser is prompted with these exact spellings and the
# hard filters compare against these exact spellings. A mismatch here
# ("very loud" vs "very_loud") silently drops every result for a constraint,
# which is the kind of bug that never raises and never gets noticed.
#
# Values are as the dataset spells them after normalization. Note that
# ``Alcohol="none"`` is a real value meaning "serves no alcohol" — it is NOT
# missing data. That distinction survives because only Python ``None`` (from the
# literal string ``"None"``) is treated as missing; the *string* ``"none"`` is
# kept as a value.
ATTRIBUTE_VOCABULARY: Final[dict[str, tuple[str, ...]]] = {
    "WiFi": ("free", "paid", "no"),
    "NoiseLevel": ("quiet", "average", "loud", "very_loud"),
    "Alcohol": ("none", "beer_and_wine", "full_bar"),
    "RestaurantsAttire": ("casual", "dressy", "formal"),
    "Smoking": ("no", "outdoor", "yes"),
    "BYOBCorkage": ("no", "yes_corkage", "yes_free"),
}

# Spelling variants seen in the wild, mapped onto the vocabulary above. Slugging
# (lowercase, non-alphanumerics to underscores) handles most of them; these are
# the cases slugging alone cannot fix.
_VALUE_ALIASES: Final[dict[str, dict[str, str]]] = {
    "WiFi": {"yes": "free", "wifi": "free", "none": "no", "not_free": "paid"},
    "NoiseLevel": {"very": "very_loud", "quite": "quiet"},
    "Alcohol": {
        "beer_wine": "beer_and_wine",
        "beer_and_wine_only": "beer_and_wine",
        "fullbar": "full_bar",
        "full": "full_bar",
    },
    "BYOBCorkage": {"yes": "yes_free"},
}

# A handful of records give a boolean where the vocabulary expects a string.
# We only translate where there is exactly one sensible reading; everywhere else
# the boolean is kept as a boolean rather than guessed at (see
# ``_canonical_vocabulary_value``). WiFi=True can only mean the place has WiFi,
# and leaving it as ``True`` would make it invisible to every "wifi" query.
_BOOL_ALIASES: Final[dict[str, dict[bool, str]]] = {
    "WiFi": {True: "free", False: "no"},
}

# Only applied to strings that ``literal_eval`` has already rejected. The u-prefix
# is in fact *valid* Python 3 (PEP 414 restored it), so ``literal_eval("u'free'")``
# returns ``'free'`` with no help from us — the widespread claim that this needs
# pre-stripping is wrong, and we have a test asserting it. What is genuinely
# invalid is the legacy combined prefix ``ur'...'``, so the regex exists for that
# residue only. Restricting it to the failure path matters: a blanket regex over
# every value would also rewrite the inside of real strings.
# ``\bu(['"])`` requires a word boundary before the ``u`` so that "menu'" and
# "you'" are left alone; a value ending in a bare word plus ``u'`` would still be
# rewritten, which we accept as vanishingly rare in a values-only field.
_U_PREFIX_RE: Final = re.compile(r"\b[uU][rR]?(['\"])")

_SLUG_RE: Final = re.compile(r"[^a-z0-9]+")


class _Missing:
    """Sentinel for "this attribute has no usable value".

    A sentinel rather than ``None`` because ``None`` is itself one of the inputs
    we are classifying, and ``if value is None`` would then be ambiguous.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<MISSING>"


_MISSING: Final = _Missing()


def _slug(text: str) -> str:
    """Lowercase, collapse non-alphanumerics to underscores, trim them.

    Applied to every string value, not only the known-vocabulary ones. Every
    string attribute in this dataset is a small enum, so consistent spelling is
    worth more than preserving the original casing; if a genuinely free-text
    string attribute ever appears it will be slugged too, which is visible in
    :func:`coverage` output rather than silent.
    """
    return _SLUG_RE.sub("_", text.strip().lower()).strip("_")


def _decode(raw: Any) -> Any:
    """Turn one raw attribute value into a Python object.

    Returns ``None`` for anything that carries no information, so the caller can
    drop the key. Never raises: a single malformed value in a 150K-row file must
    not take down the data build.
    """
    if raw is None:
        return None
    # Some exports hand back already-typed values (real JSON booleans, ints,
    # nested objects). Nothing to decode.
    if not isinstance(raw, str):
        return raw

    text = raw.strip()
    if not text:
        return None

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass

    # Second chance for legacy prefixes that are not valid Python 3 syntax.
    repaired = _U_PREFIX_RE.sub(r"\1", text)
    if repaired != text:
        try:
            return ast.literal_eval(repaired)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass

    # Not a literal at all — e.g. a bare word like ``free``. Keep the raw text:
    # it is still data, and normalization below will canonicalize its spelling.
    return text


def _canonical_vocabulary_value(key: str, value: Any) -> Any:
    """Map a value onto :data:`ATTRIBUTE_VOCABULARY` where we have one."""
    vocabulary = ATTRIBUTE_VOCABULARY.get(key)
    if vocabulary is None:
        return _slug(value) if isinstance(value, str) else value

    if isinstance(value, bool):
        # Deliberately not a guess: only the explicitly listed translations.
        return _BOOL_ALIASES.get(key, {}).get(value, value)

    if not isinstance(value, str):
        return value

    slug = _slug(value)
    slug = _VALUE_ALIASES.get(key, {}).get(slug, slug)
    # An out-of-vocabulary value is kept rather than dropped. Dropping it would
    # make the business read as UNKNOWN, which is a stronger claim than the data
    # supports — we know *something*, just not a spelling we recognize. A filter
    # comparing against the vocabulary will treat it as a mismatch, which is the
    # honest outcome.
    return slug


def canonical_value(key: str, value: Any) -> Any:
    """Normalize one already-decoded value the way :func:`parse_attributes` does.

    Public because the query parser must canonicalize the LLM's output with the
    exact same function the business side uses. Two parallel implementations of
    "how do you spell quiet" is how constraint matching silently breaks.
    """
    normalized = _normalize(key, value)
    return None if normalized is _MISSING else normalized


def _normalize(key: str, value: Any) -> Any:
    """Canonicalize one decoded value, or return :data:`_MISSING`."""
    if value is None:
        return _MISSING

    if key == PRICE_ATTRIBUTE:
        # Price is numeric everywhere downstream (``price_max`` comparisons), so
        # coerce here. A non-numeric price is unusable, hence missing.
        try:
            price = int(str(value).strip())
        except (TypeError, ValueError):
            return _MISSING
        return price if price in PRICE_RANGE else _MISSING

    if isinstance(value, bool):
        return _canonical_vocabulary_value(key, value)

    if isinstance(value, int | float):
        return value

    if isinstance(value, str):
        slug = _canonical_vocabulary_value(key, value)
        # The empty string survives slugging as "" and means nothing.
        return _MISSING if slug == "" else slug

    return value


def _flatten(prefix: str, nested: Mapping[str, Any], out: dict[str, Any]) -> None:
    """Flatten a nested attribute dict into ``out`` under dotted keys.

    Flatten rather than keep the dict because a constraint targets exactly one
    leaf ("casual ambience"). If the dict were kept whole, every consumer —
    filters, document renderer, the coverage report, the LambdaMART feature
    builder — would have to re-walk it and each would invent its own key
    spelling. Recursion handles arbitrary depth, though this dataset only nests
    one level.
    """
    for sub_key, sub_value in nested.items():
        if not isinstance(sub_key, str):
            continue
        key = f"{prefix}{ATTRIBUTE_PATH_SEP}{sub_key}"
        decoded = _decode(sub_value)
        if isinstance(decoded, Mapping):
            _flatten(key, decoded, out)
            continue
        normalized = _normalize(key, decoded)
        if normalized is not _MISSING:
            out[key] = normalized


def parse_attributes(raw: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """Normalize a raw Yelp attributes blob.

    Args:
        raw: the ``attributes`` field of a ``business.json`` record. ``None`` for
            the many businesses that have no attribute data at all. A string is
            accepted because some exports stringify the whole blob.

    Returns:
        A flat dict of canonical values. **A key is present only when the
        dataset gave us a usable value for it** — callers test membership to
        distinguish "no data" from a falsy value, which is why no key is ever
        mapped to ``None``.
    """
    if isinstance(raw, str):
        decoded_blob = _decode(raw)
        raw = decoded_blob if isinstance(decoded_blob, Mapping) else None

    if not isinstance(raw, Mapping):
        return {}

    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        decoded = _decode(value)
        if isinstance(decoded, Mapping):
            _flatten(key, decoded, parsed)
            continue
        normalized = _normalize(key, decoded)
        if normalized is not _MISSING:
            parsed[key] = normalized
    return parsed


def _attributes_of(business: Any) -> Mapping[str, Any]:
    """Accept either a raw record or an already-parsed :class:`Business`."""
    if isinstance(business, Mapping):
        return parse_attributes(business.get("attributes"))
    # ``Business.attributes`` is normalized at construction time.
    return getattr(business, "attributes", None) or {}


def coverage(businesses: Iterable[Any]) -> dict[str, float]:
    """Per-attribute fill rate, as a fraction of **all** businesses passed in.

    Args:
        businesses: raw ``business.json`` records or parsed ``Business`` objects.

    Returns:
        ``{attribute_key: fraction_present}``, highest fill rate first (ties
        broken alphabetically so the report is byte-stable across runs).

    The denominator is every business seen, including those whose ``attributes``
    field is entirely null — that is the number the failure analysis needs
    ("only N% of businesses declare a NoiseLevel at all"). Measuring fill rate
    only among businesses that have *some* attributes would flatter the dataset
    and misstate how often a constraint query can be answered.
    """
    total = 0
    counts: dict[str, int] = {}
    for business in businesses:
        total += 1
        for key in _attributes_of(business):
            counts[key] = counts.get(key, 0) + 1

    if total == 0:
        return {}

    return {key: counts[key] / total for key in sorted(counts, key=lambda k: (-counts[k], k))}
