"""Build the one text document per business that every text model reads.

Why this small file matters more than its size suggests
------------------------------------------------------
BM25, the dense embedder, and the cross-encoder all score a query against this
text and nothing else. The cross-encoder's window is `rerank.max_length` tokens
(512), so anything sitting in the tail of a long document is simply not read by
the most accurate stage in the cascade. Document design therefore puts a
ceiling on achievable quality *upstream of every model choice* — a better
reranker cannot recover information that never entered its window.

Five decisions follow, each argued at the place it happens:

1. Attributes become English phrases ("free WiFi, quiet, outdoor seating, $$"),
   never key-value dumps ("WiFi=free NoiseLevel=quiet").
2. Negative attributes are left out entirely, because BM25 cannot see a "no".
3. Structured signal (name, categories, attribute phrases) goes FIRST; user
   prose (tips, review snippets) goes LAST, because the tail is what gets cut.
4. Hours are deliberately absent from the text — they are an arithmetic filter.
5. Every cap is config, so document design can be ablated like any other
   hyperparameter, and output is byte-stable for a fixed input so that indexes
   keyed by `Config.fingerprint()` are actually reproducible.

Division of labour with `concierge.data.attributes`
--------------------------------------------------
That module owns attribute *parsing and normalization* — resolving
`"u'free'"` to `"free"`, flattening `Ambience` into dotted keys, canonicalizing
spellings, and dropping keys with no data. This module owns attribute
*phrasing*, because how an attribute reads in a document is a rendering choice
we measure (see `BusinessDocument`'s docstring), not a property of the data. We
import its key spellings rather than restating them, so "RestaurantsPriceRange2"
and the `Ambience.casual` dotted form have exactly one definition in the repo.

`build_document` takes an optional `phrase_fn`, so an alternative phrasing can
be swapped in and ablated against `attribute_phrases` without editing this file.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from concierge.config import Config
from concierge.data.attributes import ATTRIBUTE_PATH_SEP, PRICE_ATTRIBUTE, PRICE_RANGE
from concierge.types import Business, BusinessDocument

__all__ = ["PhraseFn", "attribute_phrases", "build_document", "build_documents"]

# A function that turns normalized attributes into readable English phrases.
PhraseFn = Callable[[Mapping[str, Any]], Sequence[str]]

# Section prefixes. They cost a handful of tokens and earn it: they tell the
# cross-encoder that the trailing text is user prose rather than more
# attributes, which otherwise reads as one long run-on claim about the place.
_TIPS_PREFIX = "Tips: "
_REVIEWS_PREFIX = "Reviews: "

# Boundary characters used when shortening text. Sentence boundaries are
# preferred over clause boundaries, and clause boundaries over a bare word
# boundary, so a truncated snippet still reads as language rather than as a
# fragment that stops mid-thought.
_SENTENCE_MARKS = (".", "!", "?")
_CLAUSE_MARKS = (";", ",", ":")


# ─────────────────────────────────────────────────────────────────────────────
# Attribute phrasing
# ─────────────────────────────────────────────────────────────────────────────
#
# THE CENTRAL ARGUMENT OF THIS FILE.
#
# Every model that reads this document was pretrained on natural language, so
# the closer the text sits to that pretraining distribution, the better the
# representation it gets. "free WiFi, quiet, outdoor seating, $$" is a phrase
# shape an embedding model has seen thousands of variants of in web text;
# "WiFi=free NoiseLevel=quiet OutdoorSeating=True" is a shape it has mostly
# seen in code and logs, and it tokenizes badly — "OutdoorSeating" fragments
# into subwords, "=" carries no meaning, and "True" says nothing about a venue.
#
# It also matters for the lexical arm, which is the half people forget. BM25
# matches terms, and users type "quiet cafe with wifi", never
# "NoiseLevel=quiet". Phrasing puts the user's own vocabulary in the document,
# so both retrieval arms improve for the same edit.
#
# One rule is less obvious and worth defending: NEGATIVE attributes are
# omitted. BM25 has no notion of negation, so "no outdoor seating" is a strong
# lexical match for "outdoor seating", and bi-encoder embeddings handle
# negation poorly too. Writing the negative in would actively promote the
# wrong places. Nothing is lost by omitting it: a "with outdoor seating"
# request is enforced arithmetically against `Business.attributes` by the
# Stage 2 filter, not by hoping the text happens to say so.

# Boolean attributes: the phrase to use when the value is True. False
# contributes nothing (see the negation rule above).
_BOOL_PHRASES: dict[str, str] = {
    "OutdoorSeating": "outdoor seating",
    "GoodForKids": "kid friendly",
    "DogsAllowed": "dogs allowed",
    "BikeParking": "bike parking",
    "WheelchairAccessible": "wheelchair accessible",
    "HappyHour": "happy hour",
    "Caters": "catering",
    "CoatCheck": "coat check",
    "DriveThru": "drive through",
    "GoodForDancing": "good for dancing",
    "BusinessAcceptsCreditCards": "accepts credit cards",
    "ByAppointmentOnly": "by appointment only",
    "HasTV": "TV",
    "RestaurantsDelivery": "delivery",
    "RestaurantsTakeOut": "takeout",
    "RestaurantsReservations": "takes reservations",
    "RestaurantsGoodForGroups": "good for groups",
    "RestaurantsTableService": "table service",
    "RestaurantsCounterService": "counter service",
}

# Enumerated attributes, keyed by (attribute, canonical value). The canonical
# spellings come from `ATTRIBUTE_VOCABULARY` in the attributes module — that is
# the contract, and a value spelled differently here would silently never
# match. Negating values (WiFi="no", Alcohol="none", Smoking="no") are
# deliberately absent.
_ENUM_PHRASES: dict[tuple[str, str], str] = {
    ("WiFi", "free"): "free WiFi",
    ("WiFi", "paid"): "paid WiFi",
    ("NoiseLevel", "quiet"): "quiet",
    ("NoiseLevel", "average"): "moderate noise",
    ("NoiseLevel", "loud"): "loud",
    ("NoiseLevel", "very_loud"): "very loud",
    ("Alcohol", "beer_and_wine"): "beer and wine",
    ("Alcohol", "full_bar"): "full bar",
    ("RestaurantsAttire", "casual"): "casual dress",
    ("RestaurantsAttire", "dressy"): "dressy",
    ("RestaurantsAttire", "formal"): "formal dress",
    ("Smoking", "outdoor"): "outdoor smoking area",
    ("Smoking", "yes"): "smoking allowed",
    ("BYOBCorkage", "yes_free"): "BYOB",
    ("BYOBCorkage", "yes_corkage"): "BYOB with corkage fee",
}

# How a flattened nested attribute reads. `Ambience.casual` becomes "casual",
# `BusinessParking.street` becomes "street parking" — the leaf alone would be
# ambiguous ("street" is not an amenity).
_NESTED_TEMPLATES: dict[str, str] = {
    "Ambience": "{leaf}",
    "BusinessParking": "{leaf} parking",
    "GoodForMeal": "good for {leaf}",
    "Music": "{leaf} music",
    "DietaryRestrictions": "{leaf} options",
    "HairSpecializesIn": "specializes in {leaf}",
}

# `BestNights.friday` would put weekday names in the document, where they are a
# trap: "open friday night" is an hours constraint, and a lexical match on
# "friday" from an unrelated attribute is exactly the false positive the hours
# filter exists to prevent.
_SKIP_NESTED_PARENTS = frozenset({"BestNights"})

# Phrase order. Earliest phrases are the ones certain to survive the
# cross-encoder's window, so the attributes people actually search for go
# first. Everything unlisted sorts after these, alphabetically by key — a total
# order, so the document does not depend on the key order of the raw record.
_PHRASE_PRIORITY: tuple[str, ...] = (
    "WiFi",
    "NoiseLevel",
    "OutdoorSeating",
    "Ambience",
    "Alcohol",
    "GoodForMeal",
    "RestaurantsAttire",
    "GoodForKids",
    "DogsAllowed",
    "HappyHour",
    "RestaurantsReservations",
    "RestaurantsTakeOut",
    "RestaurantsDelivery",
    "BusinessParking",
    "BikeParking",
    "WheelchairAccessible",
)

# "RestaurantsTakeOut" -> "take out": Yelp prefixes many keys with
# "Restaurants", which adds a word nobody searches for.
_KEY_PREFIX_NOISE = ("Restaurants",)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _deslug(value: str) -> str:
    """Turn a canonical slug into words: "very_loud" -> "very loud"."""
    return value.replace("_", " ").strip()


def _decamel(key: str) -> str:
    """Split a camel-case key into words, minus the noisy key prefixes.

    "GoodForKids" -> "good for kids"; "RestaurantsTakeOut" -> "take out".
    """
    for prefix in _KEY_PREFIX_NOISE:
        if key.startswith(prefix) and len(key) > len(prefix):
            key = key[len(prefix) :]
    return _CAMEL_RE.sub(" ", key).replace("_", " ").lower().strip()


def _priority(key: str) -> int:
    """Sort position for a key, by its first path segment."""
    head = key.split(ATTRIBUTE_PATH_SEP, 1)[0]
    for i, name in enumerate(_PHRASE_PRIORITY):
        if head == name:
            return i
    return len(_PHRASE_PRIORITY)


def _nested_phrase(parent: str, leaf: str, value: Any) -> str | None:
    """Phrase one flattened nested attribute, e.g. ("Ambience", "casual")."""
    if parent in _SKIP_NESTED_PARENTS:
        return None
    # Nested attributes are booleans in this dataset; a False leaf is a negative
    # and is omitted for the reason given at the top of this section.
    if value is not True:
        return None
    template = _NESTED_TEMPLATES.get(parent)
    leaf_text = _deslug(_decamel(leaf))
    if template is None:
        # An unknown nested parent still phrases readably: "good for groups"
        # style guessing would be inventing meaning, so we name both parts.
        return f"{leaf_text} {_decamel(parent)}".strip()
    return template.format(leaf=leaf_text)


# Values that negate their attribute. Looked up by value then key so that
# Alcohol="none" (a real value: serves no alcohol) is omitted while a *missing*
# Alcohol key is simply absent — the two must not be conflated, but in a
# document neither one should produce text.
_NEGATING_VALUES_BY_KEY: dict[str, tuple[str, ...]] = {
    "no": ("WiFi", "Smoking", "BYOBCorkage"),
    "none": ("Alcohol",),
}


def _phrase_for(key: str, value: Any) -> str | None:
    """Phrase one normalized attribute, or None if it should not appear."""
    if ATTRIBUTE_PATH_SEP in key:
        parent, _, leaf = key.partition(ATTRIBUTE_PATH_SEP)
        return _nested_phrase(parent, leaf, value)

    if isinstance(value, bool):
        if not value:
            return None  # negation rule
        # An unmapped boolean still reads as English via the key name, which is
        # better than dropping real signal ("AcceptsInsurance" becomes
        # "accepts insurance"). Yelp keeps adding attributes; the table above is
        # for the ones where de-camel-casing reads badly.
        return _BOOL_PHRASES.get(key) or _decamel(key) or None

    if isinstance(value, str):
        phrase = _ENUM_PHRASES.get((key, value))
        if phrase is not None:
            return phrase
        if key in _NEGATING_VALUES_BY_KEY.get(value, ()):  # e.g. WiFi="no"
            return None
        # Out-of-table value: name the attribute as well as the value, because
        # a bare "21plus" or "full_bar" would be unreadable on its own. Verbose
        # beats wrong, and the common attributes are all in the table above.
        return f"{_deslug(value)} {_decamel(key)}".strip() or None

    # Numbers other than price carry no phrasing we can defend (e.g.
    # "AgesAllowed" is a string, not a number) — left out rather than guessed.
    return None


def _price_phrase(value: Any) -> str | None:
    """Render a price level 1-4 as $ … $$$$, or None when there is no data.

    Dollar signs rather than the number because a user types "cheap", "$$" or
    "pricey"; "RestaurantsPriceRange2: 2" matches nothing a human would write.
    Accepts an int (what `parse_attributes` produces), a numeric string, or an
    already-rendered run of dollar signs, so a hand-built fixture or an
    alternative phraser cannot silently lose the price.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if _is_price_phrase(stripped) and len(stripped) <= max(PRICE_RANGE):
            return stripped
        value = stripped
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    return "$" * level if level in PRICE_RANGE else None


def _is_price_phrase(phrase: str) -> bool:
    return bool(phrase) and set(phrase) == {"$"}


def attribute_phrases(attributes: Mapping[str, Any]) -> list[str]:
    """Render normalized attributes as readable English phrases.

    Expects the output of `concierge.data.attributes.parse_attributes`: a flat
    dict whose nested keys are dotted and whose values are canonical, with
    missing attributes absent rather than None.

    Returns phrases in display order, price last — a bare "$$" in the middle of
    a run of amenity words reads like a typo, and at the end it reads like a
    price tag.
    """
    phrases: list[str] = []
    for key in sorted(attributes, key=lambda k: (_priority(k), k)):
        if key == PRICE_ATTRIBUTE:
            continue
        phrase = _phrase_for(key, attributes[key])
        if phrase:
            phrases.append(phrase)
    price = _price_phrase(attributes.get(PRICE_ATTRIBUTE))
    if price:
        phrases.append(price)
    return phrases


def _business_phrases(business: Business, phrase_fn: PhraseFn) -> list[str]:
    """Phrases for one business: deduplicated, whitespace-clean, price last.

    Done here rather than inside `attribute_phrases` so that these invariants
    hold for an injected alternative phraser too — the document layout should
    not change shape just because the phrasing is being ablated.
    """
    attributes = business.attributes or {}
    raw_phrases = phrase_fn(attributes) if attributes else ()

    phrases: list[str] = []
    price_from_phraser: str | None = None
    seen: set[str] = set()
    for phrase in raw_phrases:
        cleaned = _collapse_whitespace(str(phrase))
        if not cleaned or cleaned in seen:
            # Two attributes can legitimately phrase to the same words (an
            # Ambience leaf and a top-level key, say); say it once.
            continue
        seen.add(cleaned)
        if _is_price_phrase(cleaned):
            price_from_phraser = price_from_phraser or cleaned
            continue
        phrases.append(cleaned)

    price = _price_phrase(attributes.get(PRICE_ATTRIBUTE)) or price_from_phraser
    if price:
        phrases.append(price)
    return phrases


# ─────────────────────────────────────────────────────────────────────────────
# Selecting which reviews and tips to include
# ─────────────────────────────────────────────────────────────────────────────
#
# There is no `Review` or `Tip` type in `concierge.types` — the pipeline never
# needs one, since reviews exist only as document material and as sort keys.
# So these take raw dataset mappings and read only the keys the dataset
# guarantees: "text", "useful" / "compliment_count", "date", "business_id".


def _collapse_whitespace(text: str) -> str:
    """Normalize all whitespace runs to single spaces.

    Review text contains newlines, which would break the line structure of the
    document (and make the char budget harder to reason about) if passed
    through untouched. Doing it once here also means two inputs that differ
    only in whitespace produce the same document — part of the byte-stability
    guarantee.
    """
    return " ".join(str(text).split())


def _as_sentence(text: str) -> str:
    """Give a fragment terminal punctuation so snippets concatenate readably.

    Also gives the truncator a sentence boundary to find later, which is the
    difference between cutting at "...the back wall." and "...the back".
    """
    if text and text[-1] not in _SENTENCE_MARKS:
        return text + "."
    return text


def _field_text(record: Mapping[str, Any]) -> str:
    return _collapse_whitespace(record.get("text") or "")


def _field_date(record: Mapping[str, Any]) -> str:
    """Date as a string, compared lexicographically on purpose.

    Yelp dates are "YYYY-MM-DD HH:MM:SS", a format in which string order *is*
    chronological order. Comparing strings avoids parsing thousands of dates we
    only ever use as a sort key, and a malformed date degrades to sorting last
    instead of raising mid-build.
    """
    value = record.get("date")
    return value if isinstance(value, str) else ""


def _field_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _for_business(
    records: Iterable[Mapping[str, Any]] | None, business_id: str
) -> list[Mapping[str, Any]]:
    """Keep only records belonging to this business.

    The caller normally passes an already-grouped list, so this is a guard, not
    a filter: a grouping bug upstream that mixed reviews between businesses
    would produce documents that are subtly, invisibly wrong — the worst kind
    of bug in a retrieval system, because the scores still look plausible.
    """
    if records is None:
        return []
    return [r for r in records if r.get("business_id", business_id) == business_id]


def _rank_by_useful_then_recent(
    records: Sequence[Mapping[str, Any]], useful_key: str
) -> list[Mapping[str, Any]]:
    """Highest `useful` first, newest first within a tie.

    Two stable sorts, least significant key first — the standard way to get a
    multi-key sort with mixed directions without inventing a negated string.
    Text is the final tiebreak purely so the order is total: two reviews with
    the same useful count and timestamp must still order identically on every
    run, or the index artifact stops being reproducible.

    Usefulness before recency because a highly-upvoted review is the one that
    explains *why* people go there ("open until 2am which saved me after a late
    shift"), which is exactly the signal a vibe or constraint query needs,
    whereas the newest review is often a one-line complaint about parking.
    """
    by_recency = sorted(records, key=lambda r: (_field_date(r), _field_text(r)), reverse=True)
    return sorted(by_recency, key=lambda r: -_field_int(r, useful_key))


def _rank_by_recent(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda r: (_field_date(r), _field_text(r)), reverse=True)


def _rank_randomly(
    records: Sequence[Mapping[str, Any]], seed: int, business_id: str
) -> list[Mapping[str, Any]]:
    """Shuffle, but reproducibly.

    The "random" mode exists only as an ablation baseline for the selection
    rule, so it still has to be reproducible: the seed is derived from the
    config seed *and* the business id, so every business gets a different
    permutation while the whole corpus rebuilds identically. The pool is sorted
    into a canonical order first, because shuffling a list whose input order
    varies would not be reproducible no matter what the seed is.
    """
    canonical = sorted(records, key=lambda r: (_field_date(r), _field_text(r)))
    rng = random.Random(f"{seed}:{business_id}")
    rng.shuffle(canonical)
    return canonical


def _select_reviews(
    reviews: Sequence[Mapping[str, Any]], cfg: Config, business_id: str
) -> list[Mapping[str, Any]]:
    """Pick at most `documents.max_reviews` reviews by the configured rule."""
    mode = cfg.documents.review_selection
    if mode == "useful_then_recent":
        ordered = _rank_by_useful_then_recent(reviews, "useful")
    elif mode == "recent":
        ordered = _rank_by_recent(reviews)
    elif mode == "random":
        ordered = _rank_randomly(reviews, cfg.seed, business_id)
    else:  # pragma: no cover - pydantic Literal already rejects other values
        raise ValueError(f"unknown documents.review_selection: {mode!r}")
    return ordered[: max(0, cfg.documents.max_reviews)]


def _select_tips(tips: Sequence[Mapping[str, Any]], cfg: Config) -> list[Mapping[str, Any]]:
    """Pick at most `documents.max_tips` tips.

    Tips have no selection knob of their own in the config: they are short,
    there are few per business, and `compliment_count` is the exact analogue of
    a review's `useful`, so the same usefulness-then-recency rule applies
    without needing to be configurable. If tip selection ever looks worth
    ablating, it gets its own config key rather than a hidden default here.
    """
    ordered = _rank_by_useful_then_recent(tips, "compliment_count")
    return ordered[: max(0, cfg.documents.max_tips)]


# ─────────────────────────────────────────────────────────────────────────────
# Truncation
# ─────────────────────────────────────────────────────────────────────────────


def _last_boundary(window: str, marks: Sequence[str], keep_mark: bool) -> int:
    """Index just past the last usable boundary in `window`, or -1.

    A mark counts as a boundary only when it ends the window or is followed by
    whitespace, so decimals and "p.m." do not read as sentence ends.
    """
    for i in range(len(window) - 1, -1, -1):
        if window[i] in marks and (i + 1 == len(window) or window[i + 1].isspace()):
            return i + 1 if keep_mark else i
    return -1


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Shorten `text` to at most `limit` chars, never mid-word.

    Preference order: sentence boundary, then clause boundary, then word
    boundary. Returns the text and whether anything was cut. A single word
    longer than the limit yields an empty string rather than a mangled
    fragment — half a word is noise for BM25 and for a tokenizer alike.
    """
    if limit <= 0:
        return "", bool(text)
    if len(text) <= limit:
        return text, False

    window = text[:limit]
    for marks, keep in ((_SENTENCE_MARKS, True), (_CLAUSE_MARKS, False)):
        cut = _last_boundary(window, marks, keep)
        if cut > 0:
            return window[:cut].rstrip(), True

    space = window.rfind(" ")
    if space > 0:
        return window[:space].rstrip(), True
    return "", True


# ─────────────────────────────────────────────────────────────────────────────
# Document assembly
# ─────────────────────────────────────────────────────────────────────────────


class _Budget:
    """Tracks the character cost of the document as lines are added.

    A tiny class instead of running arithmetic because the cost of a line
    depends on whether a newline joins it to the previous one, and getting that
    off by one would make `max_chars` a lie.
    """

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(0, max_chars)
        self.lines: list[str] = []
        self._used = 0

    @property
    def remaining(self) -> int:
        return self.max_chars - self._used

    def cost_of(self, line: str) -> int:
        return len(line) + (1 if self.lines else 0)

    def fits(self, line: str) -> bool:
        return self.cost_of(line) <= self.remaining

    @property
    def room_for_line(self) -> int:
        """How many characters of line *content* still fit, newline included."""
        return self.remaining - (1 if self.lines else 0)

    def add(self, line: str) -> None:
        self._used += self.cost_of(line)
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines)


def _add_or_truncate(budget: _Budget, line: str) -> bool:
    """Add `line`, shortening it at a boundary if it does not fit.

    Returns True if anything was cut, which is what sets
    `BusinessDocument.truncated`.
    """
    if budget.fits(line):
        budget.add(line)
        return False
    fitted, was_cut = _truncate(line, budget.room_for_line)
    if fitted:
        budget.add(fitted)
    return was_cut or not fitted


def _head_line(business: Business) -> str:
    """Name and categories as one short sentence.

    First position is the only position guaranteed to be read by every stage,
    so the two fields that identify the place go there. Written as a sentence
    ("Lantern Coffee. Coffee & Tea, Cafes.") rather than a header block for the
    same pretraining-distribution reason as the attribute phrases.
    """
    name = _collapse_whitespace(business.name)
    categories = ", ".join(_collapse_whitespace(c) for c in business.categories if c)
    if categories:
        return _as_sentence(f"{name}. {categories}")
    return _as_sentence(name)


def build_document(
    business: Business,
    reviews: Sequence[Mapping[str, Any]] | None = None,
    tips: Sequence[Mapping[str, Any]] | None = None,
    cfg: Config | None = None,
    *,
    phrase_fn: PhraseFn | None = None,
) -> BusinessDocument:
    """Render one business as the text every retrieval and ranking stage reads.

    Layout, in order, newline separated:

        <name>. <categories>.
        <attribute phrases>, <price>.
        Tips: <tip> <tip>
        Reviews: <snippet> <snippet>

    The order is the whole point. A 512-token cross-encoder reads the head and
    truncates the tail, so the structured signal — the part that answers
    "quiet", "wifi", "$$", "ramen" — must come first and the user prose last.
    Inverting this would mean the most accurate stage in the cascade scores
    places largely on whichever reviews happened to be sampled.

    Hours are intentionally NOT in the text. "Open past 10pm" is answered by
    comparing `HoursInterval`s, and a review that merely mentions "open late"
    must not be able to fake a satisfied hard constraint — that failure mode is
    invisible in an aggregate metric and infuriating to a user standing outside
    a closed door.

    `truncated` means content we wanted was shortened or dropped to fit
    `max_chars`. Reviews left out by `max_reviews` are *selection*, not
    truncation, and do not set the flag.
    """
    cfg = cfg if cfg is not None else _load_default_config()
    phrase_fn = phrase_fn if phrase_fn is not None else attribute_phrases

    budget = _Budget(cfg.documents.max_chars)
    truncated = False

    # ── 1. Head: name + categories. Mandatory; shortened only if it alone
    # exceeds max_chars, which would mean a pathological config.
    truncated |= _add_or_truncate(budget, _head_line(business))

    # ── 2. Attribute phrases, immediately after the head.
    phrases = _business_phrases(business, phrase_fn)
    if phrases:
        truncated |= _add_or_truncate(budget, _as_sentence(", ".join(phrases)))

    # ── 3. Tips. Short and unusually high-signal ("Outlets along the back
    # wall"), so they get the budget ahead of reviews. Whole tips only: half a
    # tip is not worth the tokens.
    selected_tips = _select_tips(_for_business(tips, business.business_id), cfg)
    tip_texts: list[str] = []
    for tip in selected_tips:
        text = _as_sentence(_field_text(tip))
        if not text:
            continue
        candidate = [*tip_texts, text]
        if budget.fits(_TIPS_PREFIX + " ".join(candidate)):
            tip_texts = candidate
        else:
            truncated = True
            break
    if tip_texts:
        budget.add(_TIPS_PREFIX + " ".join(tip_texts))

    # ── 4. Review snippets, last, and sharing whatever budget is left.
    selected_reviews = _select_reviews(
        _for_business(reviews, business.business_id), cfg, business.business_id
    )
    snippets: list[str] = []
    if selected_reviews:
        # Each remaining review gets an equal share of what is left, recomputed
        # after every snippet so a short review hands its unused budget to the
        # next one. Without a per-review share, one 4000-character review would
        # swallow the whole window and crowd out the other three — and reviews
        # disagree with each other usefully, so breadth beats depth here.
        remaining = budget.remaining - budget.cost_of(_REVIEWS_PREFIX)
        for i, review in enumerate(selected_reviews):
            reviews_left = len(selected_reviews) - i
            join_cost = 1 if snippets else 0  # the space before this snippet
            share = max(0, remaining - join_cost) // reviews_left
            snippet, was_cut = _truncate(_field_text(review), share)
            truncated = truncated or was_cut
            if not snippet:
                # Nothing of this review survived the share (or it was empty).
                continue
            # Add terminal punctuation only if it still fits; a snippet ending
            # at a clause boundary reads fine without it.
            if len(snippet) < share:
                snippet = _as_sentence(snippet)
            snippets.append(snippet)
            remaining -= len(snippet) + join_cost
    if snippets:
        budget.add(_REVIEWS_PREFIX + " ".join(snippets))

    return BusinessDocument(
        business_id=business.business_id,
        text=budget.text(),
        n_reviews_used=len(snippets),
        n_tips_used=len(tip_texts),
        truncated=truncated,
    )


def build_documents(
    businesses: Sequence[Business],
    reviews_by_business: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    tips_by_business: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    cfg: Config | None = None,
    *,
    phrase_fn: PhraseFn | None = None,
) -> list[BusinessDocument]:
    """Render many businesses, preserving input order.

    Order is preserved (rather than returning a dict) because BM25 and FAISS
    both address documents by row number: the caller keeps one list of
    businesses and one list of documents, and row i means the same business in
    the index, in the id map, and here. A dict would make that alignment the
    caller's problem, and a misaligned id map is a bug that shows up as
    nonsense results rather than as an error.
    """
    cfg = cfg if cfg is not None else _load_default_config()
    phrase_fn = phrase_fn if phrase_fn is not None else attribute_phrases
    reviews_by_business = reviews_by_business or {}
    tips_by_business = tips_by_business or {}
    return [
        build_document(
            business,
            reviews_by_business.get(business.business_id, ()),
            tips_by_business.get(business.business_id, ()),
            cfg,
            phrase_fn=phrase_fn,
        )
        for business in businesses
    ]


def _load_default_config() -> Config:
    """Fall back to the committed config when a caller passes none.

    Imported through `load_config` rather than cached at module import so that
    a test or a sweep that edits the YAML is not silently served a stale
    config.
    """
    from concierge.config import load_config

    return load_config()
