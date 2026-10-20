"""Rule-based query parser: Stage 1 without the LLM.

Two jobs, both of which matter to the final numbers:

1. **Fallback.** The free Gemini tier rate-limits and occasionally times out. Without a
   fallback, a rate-limited request means a search with no filters at all, which is a
   much worse result than a search with the *easy* filters applied. This parser handles
   the constraints that are genuinely regular: hours (with the weekday a request names),
   WiFi, outdoor seating, noise, price, and obvious category words.
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
* **One strip rule, stated once.** Whether a phrase the parser lifted also leaves
  ``semantic_text`` is decided in exactly one place — ``strips_phrase`` below — from the
  single fact each rule declares about itself (``_Rule.writes``). No rule carries its own
  strip flag, so conformance is readable at a glance and a rule added later cannot quietly
  acquire a different policy.

Known limitation, because this parser is itself an evaluated condition: the category
lexicon below is hand-written, and some of its food words also appear in the gold set's
example queries. The disclosure, the provenance rule that bounds it, and why the entries
are kept rather than deleted are all written out at ``CATEGORY_LEXICON``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from concierge.query.schema import (
    MAX_CATEGORIES,
    SUPPORTED_ATTRIBUTES,
    WEEKDAYS,
    night_order_key,
    normalize_day,
    normalize_price_max,
    resolve_open_at_day,
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

# ── Weekdays the query itself names ──────────────────────────────────────────
#
# Why this exists at all: without it, "open until 2am on friday" still produced an hours
# constraint, but anchored on the weekday the *caller's clock* reported, so the filter
# asked about a night several days away from the one the user named. The phrase also
# survived into `semantic_text`, so retrieval scored "friday" while the filter enforced
# a different night. A wrong day is the worst kind of bug this parser can have: the hours
# filter still runs, still returns ten places, and simply returns another night's venues.
#
# Why the lexicon is not `schema._DAY_ALIASES`, the table the LLM path uses. That table is
# tuned for what a *model* writes into a JSON field, where "sa" and "su" are unambiguous
# because nothing else can appear there. This pattern runs over English prose, where "sat"
# and "sun" are ordinary words ("sat outside", "in the sun"), so they only count as
# weekdays after a word that can introduce a day. Full names and the abbreviations that
# are not English words need no introducer. The spellings below are the aliases
# `normalize_day` accepts, and the output day always comes from that one function, so
# there is a single authority on how a weekday is spelled.
_DAY_INTRODUCER = r"(?:on|this|next|every|coming|upcoming)"

# Longest first so "tuesday" is tried before "tues" and "thurs" before "thu". (Python's
# alternation backtracks, so the trailing \b would reject a short match inside a longer
# word anyway; the order is for the reader.)
_UNAMBIGUOUS_DAY_WORDS = (
    *sorted((day.lower() for day in WEEKDAYS), key=len, reverse=True),
    "tues",
    "thurs",
    "thur",
    "weds",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
)

# The introducer is optional here and part of the match, so stripping the day also removes
# the "on" that only joined it to the rest of the sentence. A trailing "s" ("open fridays
# until 2am") is matched outside the capture group, so the captured token stays an alias
# `normalize_day` knows.
_WEEKDAY_PATTERN = (
    r"\b(?:" + _DAY_INTRODUCER + r"\s+)?(?P<weekday>"
    r"" + "|".join(_UNAMBIGUOUS_DAY_WORDS) + r")s?\b"
)

# "sat" and "sun" are ordinary English words, so here the introducer is required.
_SHORT_WEEKDAY_PATTERN = r"\b" + _DAY_INTRODUCER + r"\s+(?P<weekday>sat|sun)s?\b"


# ─────────────────────────────────────────────────────────────────────────────
# Category lexicon
# ─────────────────────────────────────────────────────────────────────────────

# Phrase -> Yelp category string. Hand-built and deliberately small: a category filter is
# coarse (Yelp's category lists are noisy and incomplete), so a wrong guess here costs
# recall on a hard filter. Everything not listed falls through to retrieval, which handles
# unknown food words perfectly well on text.
#
# Matching tolerates a trailing "s"/"es", so only singular forms are listed.
#
# ── PROVENANCE AND THE RULE FOR EXTENDING IT (leakage disclosure) ────────────
# The rule parser is a scored ablation condition, so this table is a *parameter of an
# evaluated system*. Where its entries came from therefore matters as much as what they
# say.
#
# Provenance: the right-hand side is the dataset's own category taxonomy — every value is
# a category string as `business.json` spells it, the same vocabulary the schema validator
# checks an LLM parse against (`known_categories`, built from `Business.categories`). The
# left-hand side is the everyday English word for that category. The table was written
# from the taxonomy outward, before the gold test set existed.
#
# The rule, and it is mechanical: **entries are derived from the dataset's category
# taxonomy and are NEVER added by reading eval/gold_queries.csv.** Pick a category, then
# write the word people use for it. Going the other way — reading the gold queries and
# adding whatever words they happen to contain — is fitting the test set by hand. It would
# raise the rules row of the ablation on exactly the queries being scored, and no metric
# downstream can reveal it afterwards. New entries need a reason that does not mention the
# gold set (a category that is common in the metro in scope, a word the failure analysis
# shows arriving unmatched from real traffic).
#
# KNOWN LIMITATION, stated rather than hidden: "ramen", "pho", "breakfast" and "brunch"
# are in here, and eval/gold_queries.csv ships EXAMPLE rows using those exact words (as
# does CLAUDE.md section 1). The overlap is unavoidable in kind — a food-search parser
# that cannot recognize "ramen" is a worse parser, and deleting the entries to look clean
# would trade real quality for the appearance of rigor. What is avoidable is the table
# *growing* to fit the gold set, which is what the rule above forbids. The residual risk
# is that the owner writes their 150 gold queries using the shipped EXAMPLE rows as a
# template; the mitigation is to delete those rows before labeling begins.
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
# THE STRIP RULE — the single place that decides what leaves `semantic_text`
# ─────────────────────────────────────────────────────────────────────────────


class Writes(StrEnum):
    """Which ``ParsedQuery`` field a rule fills when it fires.

    Every rule declares exactly one. This is the only thing a rule says about
    stripping, because ``strips_phrase`` *derives* the strip decision from it: a rule
    cannot be given its own policy by hand, so there is nothing to forget and nothing
    to get inconsistent. ``NOTHING`` is for the rules that recognize a phrase they
    cannot encode ("open now", "expensive").
    """

    OPEN_AT = "open_at"
    ATTRIBUTES = "attributes"
    PRICE_MAX = "price_max"
    CATEGORIES = "categories"
    NOTHING = "nothing"


#: The constraints `concierge.retrieval.filters` can **reject** a business for: its check
#: returns VIOLATED when the business's data contradicts the request, and `apply_filters`
#: then drops the candidate unconditionally. Hours, attributes and price are all like this.
#:
#: A *category* is not, and that is a fact about the filters rather than a preference of
#: this module: `filters.check_categories` returns SATISFIED or UNKNOWN and never
#: VIOLATED — "none of this business's categories lexically matched the words the parser
#: emitted" is a statement about our matcher and Yelp's taxonomy, not about the business
#: (`filters.POLICY_EXEMPT_CONSTRAINTS` argues it at length, and DECISIONS.md D225 is the
#: ruling). `test_the_strip_rules_premise_holds_in_the_filters` pins that premise against
#: the real filter functions, so if a category check ever starts rejecting, the test that
#: fails is the one naming this rule.
REJECTING_CONSTRAINTS = frozenset({Writes.OPEN_AT, Writes.ATTRIBUTES, Writes.PRICE_MAX})


def strips_phrase(writes: Writes) -> bool:
    """**The strip rule.** A phrase the parser lifted out of the query is removed from
    ``semantic_text`` if and only if it became a constraint the filters can reject a
    business for.

    `semantic_text` is what BM25 and the dense encoder score against business documents.
    Leaving a *rejecting* constraint's phrase in it does three harmful things:

    1. **It double-counts.** The filter already enforces that constraint exactly, so
       scoring it a second time as text similarity over-weights it against the rest of the
       request, and the ablation can no longer attribute the effect to one stage.
    2. **It rewards talking over being.** A document containing a review that says "open
       late!" outscores one whose structured hours actually satisfy the request — the
       opposite of what the filter is for.
    3. **It dilutes.** BM25 spends term weight on "open", "past", "10pm", which occur all
       over the corpus, and the discriminative word ("café") gets a smaller share.

    All three arguments need the filter to be *enforcing*. A category constraint is not:
    a miss cannot reject, so nothing is being counted twice and the word is the more
    specific signal of the two ("ramen" says more than the category "Japanese"). So
    keeping category words is not an exception to this rule — it follows from it.

    **The alternative, and what it would have cost** (DECISIONS.md D224's option (b)):
    strip nothing, leave the query text intact, and let fusion absorb the redundancy. That
    is simpler and more predictable, and one observation on the fixture corpus does favour
    it: because documents render attributes as readable phrases, the four fixture
    documents containing the word "wifi" are the three WiFi=free businesses plus the one
    WiFi=paid business the filter refutes anyway (b001/b003/b006 and b008 in
    ``tests/fixtures/businesses.json``), so leaving the word in sometimes buys a genuine
    lexical match. "patio" behaves the same way: the two fixture documents containing that
    word (b001, whose review mentions a sunny patio, and b006, whose *name* is "Patio
    Taqueria") are exactly the two businesses with ``OutdoorSeating=True``. An earlier
    version of this paragraph asserted the reverse — one patio document against two
    satisfying businesses — and that was simply wrong; it is corrected rather than deleted
    because an unchecked figure in a docstring is a defect this project has made before.
    What eight businesses cannot show is whether the match generalizes: both hits come
    from what someone *wrote* (a review, a business name) and not from the attribute the
    filter reads, so they are failure mode 2 above landing in (b)'s favour by luck.
    Neither policy has a labelled measurement behind it; no NDCG claim is attached to
    either. It was rejected because it would withdraw the double-counting argument that
    three other modules already rest on
    (`concierge.retrieval.constraints_index` argues against (b) explicitly, the LLM
    parser's JSON contract in `concierge.query.schema` asks the model for the request
    "with every constraint ... removed", and `concierge.rerank.model.training_query_text`
    exists so training scores the same stripped text serving does), and — the deciding
    reason — because the rules arm would then embed the full query while the LLM arm
    embeds stripped text, so the parser ablation row would differ by embedded-text policy
    as well as by parse quality. That is an unattributable measurement, which is the
    failure mode this project is built to avoid. The risk this rule carries instead —
    over-stripping down to a residue too thin to retrieve on — is the one D224's
    constraint-only retriever was built to backstop, and that backstop now exists.
    """
    return writes in REJECTING_CONSTRAINTS


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
    # The weekday the *query* named, which outranks `today`. None means the query named
    # none, so the hours constraint (if any) is about tonight.
    named_day: DayOfWeek | None = None
    # Set when the query named two different weekdays. `OpenAt` holds one day, so there is
    # no honest way to encode both — see `set_named_day`.
    day_conflict: bool = False
    # Spans of the hour phrases and the weekday words, held back rather than stripped on
    # the spot. Whether they belong in `semantic_text` depends on whether an hours
    # constraint is finally built, and that is not known until every rule has run: the
    # weekday may appear after the hour, or not at all, and without a day there is no
    # constraint to strip for. `parse_rules` makes the call once, at the end.
    hours_spans: list[tuple[int, int]] = field(default_factory=list)
    day_spans: list[tuple[int, int]] = field(default_factory=list)
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

    def set_open_minute(self, minute: int | None, span: tuple[int, int]) -> bool:
        """Record a requested hour, or decline a time too ambiguous to use.

        Whether the recorded hour becomes an ``OpenAt`` is decided later, by
        ``resolve_open_at``: it needs a weekday, and the query may still name one.
        """
        if minute is None:
            return False
        # Keep the *latest* hour asked for, so "open past 10pm" and "late night" together
        # resolve to the stricter 22:00 rather than whichever matched first.
        #
        # Compared in night order, not clock order. "late night ramen open until 2am" asks
        # for both 22:00 and 02:00, and 02:00 is the stricter of the two in every sense the
        # user means — but 120 < 1320, so a plain max() would discard the hour the query
        # stated outright and silently relax the filter to 22:00. `night_order_key` shifts
        # a small-hours minute past the end of the day, which is the order people use.
        if self.open_minute is None or night_order_key(minute) > night_order_key(self.open_minute):
            self.open_minute = minute
        # Recorded even for the losing phrase: both phrases asked about hours, and the
        # filter now owns that part of the request whichever minute won.
        self.hours_spans.append(span)
        return True

    def set_named_day(self, day: DayOfWeek, span: tuple[int, int]) -> bool:
        """Record a weekday the query itself named."""
        if self.named_day is not None and day != self.named_day:
            # Two different days, and `OpenAt` holds one. Taking the first would enforce
            # half the request and silently drop the other half, so the whole hours
            # constraint is declined instead and every word stays in the embeddable text.
            # Same choice `schema.normalize_day` makes in refusing "weekend".
            self.day_conflict = True
        else:
            self.named_day = day
        self.day_spans.append(span)
        return True

    def resolve_open_at(self, *, now_minute: int | None) -> OpenAt | None:
        """The hours constraint this parse ended up with, or ``None`` to decline it.

        Declines when no hour was recorded, when the query named two days, or when there is
        no weekday at all to anchor the request — neither in the query nor from the caller.
        Guessing a weekday would filter on a day the user never mentioned; declining leaves
        the phrases in the embeddable text, so the request is weakened rather than
        falsified.
        """
        if self.open_minute is None or self.day_conflict:
            return None
        reference = self.named_day or self.today
        if reference is None:
            return None
        return OpenAt(
            # `reference` anchors the request; the calendar day the hours matcher must be
            # asked about is not always that day. "open until 2am" asked on a Friday is
            # about Friday *night*, which the dataset files under Friday as an overnight
            # interval ("17:0-2:0") — but 2am is read on Saturday's clock, so the
            # constraint is Saturday 02:00 and `is_open_at` reaches it through Friday's
            # overnight spillover. Handing it Friday 02:00 instead would silently ask about
            # Thursday night. A day the *query* named is read at face value instead; that
            # rule and the reasoning behind it live in one function, shared with the LLM
            # path, so the two parsers cannot disagree about which night a query means.
            day=resolve_open_at_day(
                reference,
                self.open_minute,
                day_was_named=self.named_day is not None,
                now_minute=now_minute,
            ),
            minute_of_day=self.open_minute,
        )

    def add_category(self, name: str) -> bool:
        if name not in self.categories and len(self.categories) < MAX_CATEGORIES:
            self.categories.append(name)
        return True

    def note_unencodable(self, phrase: str) -> bool:
        self.unencodable.append(phrase)
        # False == "nothing was lifted", so the strip rule has nothing to remove and the
        # phrase stays in the embeddable text.
        return False


@dataclass(frozen=True, slots=True)
class _Rule:
    """One pattern, what it fills, and what it does.

    ``apply`` returns whether the rule turned this phrase into a filter. A rule that
    recognizes a phrase but cannot encode it (``apply`` returns False) lifted nothing, so
    there is nothing to remove — which is what keeps unenforceable requests visible to
    retrieval.

    ``writes`` is the rule's whole say in the matter: both properties below are computed
    from it by `strips_phrase`, so the strip rule lives in one place and no rule can hold
    a private copy of it.
    """

    name: str
    pattern: re.Pattern[str]
    apply: Callable[[re.Match[str], _Draft], bool]
    writes: Writes

    @property
    def strips(self) -> bool:
        """Whether a phrase this rule *encoded* leaves ``semantic_text``."""
        return strips_phrase(self.writes)

    @property
    def strips_at_the_end(self) -> bool:
        """Whether the removal has to wait until every rule has run.

        Only the hours group. An hour phrase becomes a filter only if some weekday
        anchors it, and the weekday may appear after the hour, or not at all — so
        "did this become a filter" is unknown until the parse is over. Those rules record
        their spans on the draft and `parse_rules` applies this rule to them once, at the
        end. Nothing else needs deferring: an attribute or price phrase is a filter the
        moment it matches.
        """
        return self.writes is Writes.OPEN_AT


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
    return draft.set_open_minute(_minute_from_match(match, assume_evening=True), match.span())


def _open_at(match: re.Match[str], draft: _Draft) -> bool:
    return draft.set_open_minute(_minute_from_match(match, assume_evening=False), match.span())


def _late_night(match: re.Match[str], draft: _Draft) -> bool:
    return draft.set_open_minute(LATE_NIGHT_MINUTE, match.span())


def _weekday(match: re.Match[str], draft: _Draft) -> bool:
    # Every alternative in the two weekday patterns is an alias `normalize_day` accepts, so
    # this cannot raise; `test_every_spelling_the_weekday_rule_matches_is_a_known_day`
    # pins that rather than leaving it as an assumption. Normalizing through that function
    # instead of title-casing here keeps one spelling of weekdays in the project.
    return draft.set_named_day(normalize_day(match.group("weekday")), match.span())


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
    # ── hours (removal deferred to the end of the parse — see `_Rule.strips_at_the_end`)
    _Rule(
        "open_now",
        _compile(r"\bopen\s+(?:now|right\s+now|currently)\b"),
        _open_now,
        writes=Writes.NOTHING,
    ),
    _Rule(
        "open_until",
        _compile(
            r"\b(?:open|opens|close|closes|serving|serves)\s+"
            r"(?:up\s+)?(?:past|after|til|till|until|through|to)\s+" + _TIME
        ),
        _open_until,
        writes=Writes.OPEN_AT,
    ),
    _Rule(
        "open_at",
        _compile(r"\b(?:open|opens)\s+at\s+" + _TIME),
        _open_at,
        writes=Writes.OPEN_AT,
    ),
    _Rule(
        "late_night",
        _compile(
            r"\b(?:late[-\s]?night|open\s+late|open\s+(?:til|till|until)\s+late"
            r"|stay(?:s|ing)?\s+open\s+late)\b"
        ),
        _late_night,
        writes=Writes.OPEN_AT,
    ),
    # Lower priority than the "open ..." forms: catches "ramen after midnight", where the
    # verb is implied.
    _Rule(
        "after_time",
        _compile(r"\b(?:past|after|til|till|until)\s+" + _TIME),
        _open_until,
        writes=Writes.OPEN_AT,
    ),
    # ── the weekday an hours constraint is anchored to (part of the same constraint, so
    #    the same deferred removal)
    _Rule("weekday", _compile(_WEEKDAY_PATTERN), _weekday, writes=Writes.OPEN_AT),
    _Rule(
        "weekday_short",
        _compile(_SHORT_WEEKDAY_PATTERN),
        _weekday,
        writes=Writes.OPEN_AT,
    ),
    # ── noise
    _Rule(
        "not_loud",
        _compile(
            r"\b(?:not|isn'?t|aren'?t|nothing|nowhere)\s+"
            r"(?:too\s+|very\s+|super\s+|that\s+)?(?:loud|noisy|rowdy)\b"
        ),
        _quiet,
        writes=Writes.ATTRIBUTES,
    ),
    _Rule("quiet", _compile(r"\b(?:quiet|peaceful)\b"), _quiet, writes=Writes.ATTRIBUTES),
    _Rule("loud", _compile(r"\b(?:loud|noisy)\b"), _loud, writes=Writes.ATTRIBUTES),
    # ── amenities
    _Rule(
        "wifi",
        _compile(r"\b(?:free\s+)?wi[-\s]?fi\b"),
        _wifi,
        writes=Writes.ATTRIBUTES,
    ),
    _Rule(
        "outdoor_seating",
        _compile(
            r"\b(?:outdoor|outside)\s+(?:seating|seats|table|tables|dining|area)\b"
            r"|\bpatio\b|\bterrace\b|\bsidewalk\s+seating\b|\bal\s+fresco\b"
            r"|\bsit(?:ting)?\s+outside\b|\boutdoors\b"
        ),
        _outdoor,
        writes=Writes.ATTRIBUTES,
    ),
    # ── price
    _Rule(
        "cheap",
        _compile(
            r"\b(?:cheap|cheaper|affordable|inexpensive|budget(?:[-\s]friendly)?"
            r"|low[-\s]priced|not\s+expensive|won'?t\s+break\s+the\s+bank)\b"
        ),
        _vague_cheap,
        writes=Writes.PRICE_MAX,
    ),
    # Dollar tiers. The lookarounds stop it matching a literal amount like "$10".
    _Rule(
        "price_tier",
        _compile(r"(?<![$\w])(?P<dollars>\$+)(?![$\w])"),
        _dollar_signs,
        writes=Writes.PRICE_MAX,
    ),
    _Rule(
        "expensive",
        _compile(
            r"\b(?:expensive|pricey|pricy|upscale|fancy|high[-\s]end|splurge|fine\s+dining)\b"
        ),
        _expensive,
        writes=Writes.NOTHING,
    ),
    # ── categories (kept in the text by the strip rule: a category cannot reject)
    _Rule("category", _CATEGORY_PATTERN, _category, writes=Writes.CATEGORIES),
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


def parse_rules(
    query: str, *, today: DayOfWeek | str | None = None, now_minute: int | None = None
) -> ParsedQuery:
    """Parse a query with regex rules only.

    Args:
        query: what the user typed, unmodified.
        today: the weekday to attach to an hours constraint when the query does not name
            one, e.g. ``"Friday"``. A parameter and not a clock read, so tests are
            deterministic and the timezone decision stays with the caller. A weekday the
            query names outranks it. When omitted and the query names no day either, hour
            phrases are recognized but not encoded, and they stay in ``semantic_text``.
        now_minute: minute-of-day of the moment the query is asked, when the caller knows
            the clock. It changes nothing except which night a small-hours time belongs to
            (``schema.resolve_open_at_day``): asked at 01:00 the night is already underway,
            asked at any other hour it is the one beginning this evening. A parameter for
            the same reason ``today`` is one — this module reads no clock. It has no effect
            on a query that named its own weekday.

    Returns:
        A ``ParsedQuery`` with ``source=ParseSource.RULES`` and a measured
        ``parse_latency_ms``. An unparseable query yields ``semantic_text == query`` and
        zero hard constraints — never an exception. A search that returns ten decent
        guesses beats a 500.

        ``semantic_text`` obeys ``strips_phrase``: it holds no phrase that became a
        constraint the filters can reject a business for. It is therefore **empty** for a
        query that was nothing but such constraints ("free wifi"), which by the same
        token guarantees at least one hard constraint for the constraint-only retriever to
        work from. Callers that need a string to score with use
        ``concierge.rerank.model.query_text_for``, the one definition of that, rather than
        reading this field directly.
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
            # The strip rule (`strips_phrase`), applied the moment the phrase became a
            # filter. The hours group is the one kind that cannot be decided yet.
            if encoded and rule.strips and not rule.strips_at_the_end:
                to_strip.append(match.span())

    # The deferred half of the same rule, now that every rule has run. An hours request
    # with no day to anchor it is *declined* (`resolve_open_at`), and a declined
    # constraint lifted nothing, so its words must stay where retrieval can still take a
    # soft shot at them. When it is encoded, the hour phrases and the weekday word go
    # together: they are one constraint, and the filter enforces both halves of it.
    open_at = draft.resolve_open_at(now_minute=now_minute)
    if open_at is not None:
        to_strip.extend(draft.hours_spans)
        to_strip.extend(draft.day_spans)

    # No fallback when the residue is empty, and that is the point of the rule rather
    # than an oversight. A query like "free wifi" is all constraint, so `semantic_text`
    # comes back "" — previously it was quietly replaced by the raw query, which
    # re-inserted the very phrase the filter had just taken over. That made the parser
    # obey the strip rule for "somewhere with a patio" and break it for "free wifi", for
    # no reason a reader could state (DECISIONS.md D224's secondary observation).
    #
    # Returning "" is safe because an empty residue is not an empty *query*:
    #   * Stripping only ever removes a phrase that became a rejecting filter, so an empty
    #     residue proves the query carries at least one hard constraint — and that is
    #     exactly the state `concierge.retrieval.constraints_index` generates candidates
    #     for (D224). `test_an_empty_residue_always_carries_a_hard_constraint` pins it.
    #   * No model is ever handed the empty string. "Which text does a stage score" has one
    #     definition, `concierge.rerank.model.query_text_for`, which every stage already
    #     goes through and which falls back to `raw` when `semantic_text` is empty. So the
    #     string retrieval and the reranker receive is unchanged by this; what changed is
    #     that the fallback is now visible in one documented place instead of being baked
    #     into the field, where the API response and the eval dumps showed an un-stripped
    #     query as if the parser had decided not to strip it.
    semantic_text = strip_phrases(query, to_strip)

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
