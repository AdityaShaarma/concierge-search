"""Rule-based query parser: Stage 1 without the LLM.

Two jobs, both of which matter to the final numbers:

1. **Fallback.** The free Gemini tier rate-limits and occasionally times out. Without a
   fallback, a rate-limited request means a search with no filters at all, which is a
   much worse result than a search with the *easy* filters applied. This parser handles
   the constraints that are genuinely regular: hours, WiFi, outdoor seating, noise,
   price, and obvious category words.
2. **An ablation baseline.** "LLM query parsing" only earns its latency if it beats
   rules. Having a real rule parser turns "we used an LLM" into a measured comparison,
   and the gap between the two rows is the actual finding — most likely concentrated in
   the `messy` and `vibe` query types, which is a claim the eval can check rather than
   assert.

Design constraints, all deliberate:

* **Pure.** No I/O, no clock, no network. ``parse_rules`` called twice with the same
  arguments returns the same constraints, so a test can pin behaviour exactly. "Today"
  is a parameter for the same reason — a parser that reads the system clock cannot be
  unit-tested without freezing time, and the caller (the API layer) is the only place
  that knows the user's timezone anyway.
* **Narrow beats clever.** Every pattern here is one that the dataset can actually
  answer. A rule that fires on "cozy" would have to invent an attribute; that request
  belongs in ``semantic_text``, where an embedding handles it properly.
* **Recognize without encoding.** Some phrases are understood but deliberately not turned
  into filters ("open now", "expensive"). They are left in the embeddable text, because
  the honest thing to do with a constraint we cannot enforce is to let retrieval take a
  soft shot at it rather than drop it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from concierge.query.schema import (
    MAX_CATEGORIES,
    SUPPORTED_ATTRIBUTES,
    normalize_day,
    normalize_price_max,
)
from concierge.types import DayOfWeek, OpenAt, ParsedQuery, ParseSource

# ─────────────────────────────────────────────────────────────────────────────
# Conventions that are choices, not facts
# ─────────────────────────────────────────────────────────────────────────────

# "late night" / "open late" with no hour attached. 22:00 is a convention, not a measured
# threshold: it is the hour the gold set's own example query uses ("open past 10pm"). If
# the failure analysis shows late-night queries missing good places, this is the first
# number to sweep. UNMEASURED.
LATE_NIGHT_MINUTE = 22 * 60

# "cheap" is vague where "$" is precise, so they map to different ceilings. Yelp tier 1 is
# roughly under $10 a head and tier 2 is $11-30; most things a person calls cheap — ramen,
# tacos, a casual breakfast — sit in tier 2. Filtering vague cheapness at tier 1 would
# drop most of the corpus to satisfy a word the user used loosely. An explicit "$" is taken
# at face value. UNMEASURED: the right ceiling for vague cheapness is a sweep, not a fact.
VAGUE_CHEAP_PRICE_MAX = 2

# Hour mentioned with no am/pm, in an "open past N" construction. Such a request is
# practically always about the evening ("open past 11" never means 11am), so bare hours
# 1-11 are read as pm. 12 is read as midnight, which is what "open past 12" means
# colloquially. Hours 13-23 are already unambiguous.
_EVENING_PROMOTION_MAX_HOUR = 11

_NAMED_TIMES = {"midnight": 0, "noon": 12 * 60}

# Clock time as it appears in prose: "10pm", "22:00", "7:30 PM", "midnight", bare "23".
# The trailing lookahead is load-bearing: without it, "open past 1Opm" (letter O for zero)
# matches the leading "1" and silently becomes a 13:00 filter. A garbled time must fail to
# match so the phrase falls through to retrieval instead of inventing a constraint.
_TIME = (
    r"(?:(?P<word>midnight|noon)"
    r"|(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)?)"
    r"(?![\w:])"
)


# ─────────────────────────────────────────────────────────────────────────────
# Category lexicon
# ─────────────────────────────────────────────────────────────────────────────

# Phrase -> Yelp category string. Hand-built and deliberately small: a category filter is
# coarse (Yelp's category lists are noisy and incomplete), so a wrong guess here costs
# recall on a hard filter. Everything not listed falls through to retrieval, which handles
# unknown food words perfectly well on text.
#
# Matching tolerates a trailing "s"/"es", so only singular forms are listed.
CATEGORY_LEXICON: dict[str, str] = {
    "coffee shop": "Coffee & Tea",
    "coffee": "Coffee & Tea",
    "cafe": "Cafes",
    "café": "Cafes",
    "bakery": "Bakeries",
    "bookstore": "Bookstores",
    "wine bar": "Wine Bars",
    "cocktail bar": "Cocktail Bars",
    "sports bar": "Sports Bars",
    "gastropub": "Gastropubs",
    "brewery": "Breweries",
    "pub": "Pubs",
    "bar": "Bars",
    "ramen": "Ramen",
    "noodle": "Noodles",
    "sushi": "Sushi Bars",
    "pho": "Vietnamese",
    "taqueria": "Mexican",
    "taco": "Tacos",
    "pizza": "Pizza",
    "burger": "Burgers",
    "sandwich": "Sandwiches",
    "deli": "Delis",
    "diner": "Diners",
    "bistro": "Bistros",
    "steakhouse": "Steakhouses",
    "seafood": "Seafood",
    "barbecue": "Barbeque",
    "bbq": "Barbeque",
    "tapas": "Tapas",
    "dessert": "Desserts",
    "ice cream": "Ice Cream & Frozen Yogurt",
    "brunch": "Breakfast & Brunch",
    "breakfast": "Breakfast & Brunch",
    "food truck": "Food Trucks",
    "thai": "Thai",
    "indian": "Indian",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "vietnamese": "Vietnamese",
    "italian": "Italian",
    "mexican": "Mexican",
    "mediterranean": "Mediterranean",
    "ethiopian": "Ethiopian",
    "vegan": "Vegan",
    "vegetarian": "Vegetarian",
}

# Longest phrase first so "wine bar" wins over "bar".
_CATEGORY_PATTERN = re.compile(
    r"\b(?P<phrase>"
    + "|".join(re.escape(p) for p in sorted(CATEGORY_LEXICON, key=len, reverse=True))
    + r")(?:e?s)?\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Working state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Draft:
    """Mutable accumulator for one parse.

    Local to a single ``parse_rules`` call, so mutating it keeps the function pure from
    the outside. ``ParsedQuery`` is frozen and is built once at the end.
    """

    today: DayOfWeek | None
    attributes: dict[str, Any] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    price_max: int | None = None
    open_minute: int | None = None
    # Phrases understood but not encodable as a filter — kept for the explanation and for
    # the failure analysis, which wants to know *why* a query was under-constrained.
    unencodable: list[str] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> bool:
        # Guard against a typo in a pattern silently inventing an attribute the filter
        # stage cannot check. SUPPORTED_ATTRIBUTES is the single vocabulary (schema.py).
        assert key in SUPPORTED_ATTRIBUTES, f"rule produced unsupported attribute {key!r}"
        # First rule to fire wins: patterns are ordered most-specific first, so "not loud"
        # has already claimed its span before the plain "loud" pattern is tried.
        self.attributes.setdefault(key, value)
        return True

    def set_price_max(self, value: int) -> bool:
        value = normalize_price_max(value)
        # Tightest ceiling wins: "cheap" plus "$" should mean "$", not "$$".
        self.price_max = value if self.price_max is None else min(self.price_max, value)
        return True

    def set_open_minute(self, minute: int | None) -> bool:
        """Record an hours constraint, or decline it.

        Declines when the caller gave no weekday: ``OpenAt`` needs a day, and guessing one
        would filter on a day the user never mentioned. Declining leaves the phrase in the
        embeddable text, so the request is weakened rather than falsified.
        """
        if minute is None or self.today is None:
            return False
        # Keep the *latest* hour asked for, so "open past 10pm" and "late night" together
        # resolve to the stricter 22:00 rather than whichever matched first.
        self.open_minute = minute if self.open_minute is None else max(self.open_minute, minute)
        return True

    def add_category(self, name: str) -> bool:
        if name not in self.categories and len(self.categories) < MAX_CATEGORIES:
            self.categories.append(name)
        return True

    def note_unencodable(self, phrase: str) -> bool:
        self.unencodable.append(phrase)
        return False  # False == "do not strip it from the text"


@dataclass(frozen=True, slots=True)
class _Rule:
    """One pattern and what it does.

    ``strip`` says whether the matched phrase should be removed from ``semantic_text``
    *when the rule actually encodes a filter*. A rule that recognizes a phrase but cannot
    encode it (``apply`` returns False) never strips, which is what keeps unenforceable
    requests visible to retrieval.
    """

    name: str
    pattern: re.Pattern[str]
    apply: Callable[[re.Match[str], _Draft], bool]
    strip: bool = True


def _compile(pattern: str) -> re.Pattern[str]:
    # IGNORECASE rather than lowercasing the query, so match offsets always line up with
    # the original string. (`str.lower()` can change length for some Unicode characters,
    # which would silently shift every span.)
    return re.compile(pattern, re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Hours
# ─────────────────────────────────────────────────────────────────────────────


def _minute_from_match(match: re.Match[str], *, assume_evening: bool) -> int | None:
    """Read the ``_TIME`` fragment out of a match. ``None`` means "too ambiguous to use"."""
    word = match.group("word")
    if word:
        return _NAMED_TIMES[word.lower()]

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if minute >= 60:
        return None

    meridiem = match.group("meridiem")
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem.lower().startswith("p") else 0)
    elif hour > 23:
        return None
    elif hour <= _EVENING_PROMOTION_MAX_HOUR:
        if not assume_evening:
            # "open at 9" could be 9am or 9pm and the construction gives no hint. Refusing
            # is better than a coin flip: a wrong hour filter removes good results with no
            # visible symptom.
            return None
        hour += 12  # bare 1-11 in this construction means pm
    elif hour == 12 and assume_evening:
        hour = 0  # "open past 12" means midnight

    return hour * 60 + minute


def _open_until(match: re.Match[str], draft: _Draft) -> bool:
    return draft.set_open_minute(_minute_from_match(match, assume_evening=True))


def _open_at(match: re.Match[str], draft: _Draft) -> bool:
    return draft.set_open_minute(_minute_from_match(match, assume_evening=False))


def _late_night(_: re.Match[str], draft: _Draft) -> bool:
    return draft.set_open_minute(LATE_NIGHT_MINUTE)


def _open_now(match: re.Match[str], draft: _Draft) -> bool:
    """ "open now" is understood and deliberately not resolved.

    It needs a clock *and* a timezone, and this module has neither by design (see the
    module docstring on purity). The serving layer knows both, so it can pass
    ``today=<user's weekday>`` and translate "now" into an explicit time before calling —
    which is also the only version that is testable. Until then the phrase stays in the
    embeddable text.
    """
    return draft.note_unencodable(match.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# Attributes, price, categories
# ─────────────────────────────────────────────────────────────────────────────


def _wifi(_: re.Match[str], draft: _Draft) -> bool:
    # "with wifi" is read as *free* wifi, which is what people mean; "paid" exists in the
    # vocabulary but nobody asks for it. The alternative (accept free OR paid) cannot be
    # expressed — `ParsedQuery.attributes` holds one value per key — so this is the
    # tighter, and more defensible, of the two available readings.
    return draft.set_attribute("WiFi", "free")


def _outdoor(_: re.Match[str], draft: _Draft) -> bool:
    return draft.set_attribute("OutdoorSeating", True)


def _quiet(_: re.Match[str], draft: _Draft) -> bool:
    return draft.set_attribute("NoiseLevel", "quiet")


def _loud(_: re.Match[str], draft: _Draft) -> bool:
    return draft.set_attribute("NoiseLevel", "loud")


def _vague_cheap(_: re.Match[str], draft: _Draft) -> bool:
    return draft.set_price_max(VAGUE_CHEAP_PRICE_MAX)


def _dollar_signs(match: re.Match[str], draft: _Draft) -> bool:
    return draft.set_price_max(min(len(match.group("dollars")), 4))


def _expensive(match: re.Match[str], draft: _Draft) -> bool:
    """ "expensive" is recognized and cannot be encoded.

    ``ParsedQuery`` has ``price_max`` but no ``price_min``, so a *floor* on price has
    nowhere to live. Reported as a contract gap rather than worked around locally: a
    local hack (say, price_max=4, which filters nothing) would look like a working filter
    in the ablation while doing nothing at all.
    """
    return draft.note_unencodable(match.group(0))


def _category(match: re.Match[str], draft: _Draft) -> bool:
    return draft.add_category(CATEGORY_LEXICON[match.group("phrase").lower()])


# Order matters twice over: the first rule to claim a span owns it, and more specific
# patterns must therefore come first ("not loud" before "loud", "wine bar" before "bar").
_RULES: tuple[_Rule, ...] = (
    # ── hours
    _Rule("open_now", _compile(r"\bopen\s+(?:now|right\s+now|currently)\b"), _open_now),
    _Rule(
        "open_until",
        _compile(
            r"\b(?:open|opens|close|closes|serving|serves)\s+"
            r"(?:up\s+)?(?:past|after|til|till|until|through|to)\s+" + _TIME
        ),
        _open_until,
    ),
    _Rule("open_at", _compile(r"\b(?:open|opens)\s+at\s+" + _TIME), _open_at),
    _Rule(
        "late_night",
        _compile(
            r"\b(?:late[-\s]?night|open\s+late|open\s+(?:til|till|until)\s+late"
            r"|stay(?:s|ing)?\s+open\s+late)\b"
        ),
        _late_night,
    ),
    # Lower priority than the "open ..." forms: catches "ramen after midnight", where the
    # verb is implied.
    _Rule("after_time", _compile(r"\b(?:past|after|til|till|until)\s+" + _TIME), _open_until),
    # ── noise
    _Rule(
        "not_loud",
        _compile(
            r"\b(?:not|isn'?t|aren'?t|nothing|nowhere)\s+"
            r"(?:too\s+|very\s+|super\s+|that\s+)?(?:loud|noisy|rowdy)\b"
        ),
        _quiet,
    ),
    _Rule("quiet", _compile(r"\b(?:quiet|peaceful)\b"), _quiet),
    _Rule("loud", _compile(r"\b(?:loud|noisy)\b"), _loud),
    # ── amenities
    _Rule("wifi", _compile(r"\b(?:free\s+)?wi[-\s]?fi\b"), _wifi),
    _Rule(
        "outdoor_seating",
        _compile(
            r"\b(?:outdoor|outside)\s+(?:seating|seats|table|tables|dining|area)\b"
            r"|\bpatio\b|\bterrace\b|\bsidewalk\s+seating\b|\bal\s+fresco\b"
            r"|\bsit(?:ting)?\s+outside\b|\boutdoors\b"
        ),
        _outdoor,
    ),
    # ── price
    _Rule(
        "cheap",
        _compile(
            r"\b(?:cheap|cheaper|affordable|inexpensive|budget(?:[-\s]friendly)?"
            r"|low[-\s]priced|not\s+expensive|won'?t\s+break\s+the\s+bank)\b"
        ),
        _vague_cheap,
    ),
    # Dollar tiers. The lookarounds stop it matching a literal amount like "$10".
    _Rule("price_tier", _compile(r"(?<![$\w])(?P<dollars>\$+)(?![$\w])"), _dollar_signs),
    _Rule(
        "expensive",
        _compile(
            r"\b(?:expensive|pricey|pricy|upscale|fancy|high[-\s]end|splurge|fine\s+dining)\b"
        ),
        _expensive,
    ),
    # ── categories (never stripped — see `parse_rules`)
    _Rule("category", _CATEGORY_PATTERN, _category, strip=False),
)


# ─────────────────────────────────────────────────────────────────────────────
# Stripping constraint phrases out of the embeddable text
# ─────────────────────────────────────────────────────────────────────────────

_HOLE = "\x00"

# Words that only connected the constraint to the rest of the sentence. They are dropped
# only when they end up next to a removed phrase, so a "the" in the middle of an untouched
# sentence survives.
_CONNECTORS = frozenset(
    {
        "with",
        "and",
        "&",
        "that",
        "that's",
        "thats",
        "which",
        "is",
        "are",
        "open",
        "a",
        "an",
        "the",
        "in",
        "at",
        "for",
        "but",
        "also",
        "plus",
        "or",
    }
)


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def strip_phrases(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Remove the given spans and tidy up the wreckage.

    Naively deleting "open past 10pm" from "a café with wifi that's open past 10pm" leaves
    "a café with that's", whose stray connectors are noise for BM25 and dilute the
    embedding. So each removed phrase becomes a hole, connector words and punctuation
    touching a hole are dropped too, and the pass repeats until nothing more falls away.
    """
    spans = _merge_spans(spans)
    if not spans:
        return _tidy(text)

    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        pieces.append(f" {_HOLE} ")
        cursor = end
    pieces.append(text[cursor:])

    tokens = "".join(pieces).split()
    while True:
        dropped = False
        for i, token in enumerate(tokens):
            if token == _HOLE:
                continue
            next_to_hole = (i > 0 and tokens[i - 1] == _HOLE) or (
                i + 1 < len(tokens) and tokens[i + 1] == _HOLE
            )
            if not next_to_hole:
                continue
            bare = token.strip(".,;:!?\"'").lower()
            if bare in _CONNECTORS or bare == "":
                tokens[i] = _HOLE  # becomes a hole itself, so its neighbours get a turn
                dropped = True
        if not dropped:
            break

    return _tidy(" ".join(t for t in tokens if t != _HOLE))


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip(" ,;:-")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_rules(query: str, *, today: DayOfWeek | str | None = None) -> ParsedQuery:
    """Parse a query with regex rules only.

    Args:
        query: what the user typed, unmodified.
        today: the weekday to attach to an hours constraint, e.g. ``"Friday"``. A parameter
            and not a clock read, so tests are deterministic and the timezone decision stays
            with the caller. When omitted, hour phrases are recognized but not encoded, and
            they stay in ``semantic_text``.

    Returns:
        A ``ParsedQuery`` with ``source=ParseSource.RULES`` and a measured
        ``parse_latency_ms``. An unparseable query yields ``semantic_text == query`` and
        zero hard constraints — never an exception. A search that returns ten decent
        guesses beats a 500.
    """
    started = time.perf_counter()

    draft = _Draft(today=normalize_day(today) if today is not None else None)
    claimed: list[tuple[int, int]] = []
    to_strip: list[tuple[int, int]] = []

    for rule in _RULES:
        for match in rule.pattern.finditer(query):
            if _overlaps(match.span(), claimed):
                continue
            encoded = rule.apply(match, draft)
            # Claimed either way, so a later, broader rule cannot re-read the same words.
            claimed.append(match.span())
            if encoded and rule.strip:
                to_strip.append(match.span())

    # Why strip at all: `semantic_text` is what BM25 and the dense encoder score against
    # business documents. Leaving "open past 10pm" in it does three harmful things.
    #   1. It double-counts. The hours filter already enforces that constraint exactly;
    #      scoring it a second time as text similarity over-weights it against the rest of
    #      the request, and the ablation can no longer attribute the effect to one stage.
    #   2. It rewards talking over being. A document containing a review that says "open
    #      late!" scores higher than one whose structured hours actually satisfy the
    #      request — the opposite of what the filter is for.
    #   3. It dilutes. BM25 spends term weight on "open", "past", "10pm", which appear all
    #      over the corpus, and the discriminative word ("café") gets a smaller share.
    # Category words are the deliberate exception: they are NOT stripped, because the
    # category filter is coarse (Yelp category lists are noisy and incomplete) and the word
    # still carries meaning the category cannot — "ramen" is far more specific than
    # "Japanese". Stripping it would also leave "cheap ramen" with nothing to embed.
    semantic_text = strip_phrases(query, to_strip)
    if not semantic_text:
        # Everything was a constraint. Embedding an empty string returns arbitrary nearest
        # neighbours, so fall back to the raw query and accept the double-count: a weaker
        # ranking beats a meaningless one.
        semantic_text = query.strip()

    open_at = (
        OpenAt(day=draft.today, minute_of_day=draft.open_minute)
        if draft.today is not None and draft.open_minute is not None
        else None
    )

    return ParsedQuery(
        raw=query,
        semantic_text=semantic_text,
        open_at=open_at,
        attributes=dict(draft.attributes),
        price_max=draft.price_max,
        categories=tuple(draft.categories),
        geo=None,  # no gazetteer in the rule parser; geo resolution is the caller's job
        source=ParseSource.RULES,
        parse_latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _overlaps(span: tuple[int, int], claimed: Iterable[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in claimed)
