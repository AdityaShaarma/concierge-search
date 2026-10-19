"""Stage 2, hard constraints: checking a parsed query against businesses.

Why this module exists at all
-----------------------------
Retrieval and reranking are similarity machines. Ask for "open past 10pm" and
an embedding will happily return a café that closes at 6 but whose reviews talk
about late-night studying, because the *prose* matches. Hours, price, and
amenities are arithmetic facts; they have to be checked arithmetically. This is
the part of the system that separates it from "embed the text and call it
done".

Why three-valued, and not boolean
---------------------------------
Yelp attribute coverage is sparse. "This café has no NoiseLevel field" is a
completely different statement from "this café is loud", and collapsing them is
the single most expensive mistake available here:

* Treat missing as a pass, and a query for "quiet" returns loud bars.
* Treat missing as a fail, and the answer set shrinks to the small, biased
  subset of businesses whose owners filled in the attribute form — on a sparse
  attribute that can be most of the corpus.

So every check returns SATISFIED / VIOLATED / UNKNOWN, and the two non-passing
outcomes are handled by *different* mechanisms:

* **VIOLATED removes the business, always.** The data says it fails. Showing it
  would be a wrong answer, and no ranking signal should be able to outvote that.
  Because the drop is unconditional, VIOLATED is reserved for constraints the
  data can actually *refute* — a stored value that contradicts the request. A
  check that merely failed to *establish* a match returns UNKNOWN instead; see
  `check_categories`, where that distinction is the difference between a
  taxonomy gap and a deleted correct answer.
* **UNKNOWN follows ``cfg.retrieval.missing_attribute_policy``** — ``exclude``,
  ``penalize``, or ``neutral``. All three are implemented and none is
  privileged in this code. Which is best is an open measurement (DECISIONS.md
  D007); ``cfg.retrieval.missing_attribute_penalty`` is a placeholder for a
  number that has not been measured yet, and this module must not pretend
  otherwise.

Keeping that distinction crisp is the entire point of the design.

Where the ``penalize`` policy acts — read this before reading an ablation row
-----------------------------------------------------------------------------
``penalize`` multiplies the fused Stage-2 score (``SCORE_RRF``) and nothing
else. That scope is narrower than the word "penalize" suggests, and it decides
which ablation rows the D007 three-way comparison can be read on, so it is
spelled out here. Every claim below is pinned by a named test in
``tests/test_filters.py`` rather than left as an assertion:

* **With Stage 4 on, the policy reaches the final ordering.** The discount
  changes the Stage-2 ordering and so the post-filter ``RANK_RRF`` that
  `apply_filters` recomputes, and ``RANK_RRF`` is a Stage-4 feature
  (``rrf_rank``). So the three policies hand Stage 4 three different feature
  matrices and can produce three different final top-k lists — through a
  feature the model reads, not through a tie-break.
  (`test_the_three_policies_reach_the_stage_4_feature_matrix`,
  `test_the_three_policies_produce_three_different_final_orderings`.)
* **With Stage 4 off and Stage 3 on, it does not.** `apply_filters` does not
  truncate, so the cross-encoder rescores every survivor and re-sorts by its
  own score; ``penalize`` and ``neutral`` then run over the same candidate set
  and differ only where two cross-encoder scores tie exactly.
  (`test_with_stage_4_off_penalize_and_neutral_give_the_same_order`.)
* ``exclude`` is the only policy that changes the candidate set, and so the
  only one whose effect survives every stage by construction.

So the experiment D007 promises is real, on two kinds of row: the
retrieval-only rows, where ``SCORE_RRF`` *is* the final ranking, and the
full-pipeline row, where ``rrf_rank`` carries the policy into Stage 4. It is
**not** interpretable on a reranked row with Stage 4 off — a small
penalize-vs-neutral gap there says how little of the Stage-2 ordering survives
reranking, not whether missing attribute data matters. How *large* the Stage-4
effect is depends on how much the trained model weights ``rrf_rank``, which is
unmeasured; nothing here claims a size.

Stage 4 is also handed the same underlying fact directly, as the
``has_unknown_constraint_data`` feature. So if the discount should be *learned*
rather than hand-set, that feature is where it belongs and not this multiplier
— which is why D007's experiment is the retrieval-only triple plus, separately,
a Stage-4 run with and without that feature.

Which constraints the policy governs
------------------------------------
Not all of them: see `POLICY_EXEMPT_CONSTRAINTS`. A `check_categories` miss is
UNKNOWN too, but it is a gap in *our* lexical matcher rather than a gap in the
business's data, and routing it through a knob named
``missing_attribute_policy`` made that knob partly measure Yelp's taxonomy. It
is counted separately instead.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from concierge.config import Config
from concierge.data.hours import is_open_at
from concierge.types import (
    RANK_RRF,
    SCORE_RRF,
    Business,
    Candidate,
    ConstraintMatch,
    ConstraintStatus,
    GeoConstraint,
    OpenAt,
    ParsedQuery,
)

log = logging.getLogger(__name__)

# Constraint names. They appear in the "why this result" UI and are used as
# Stage 4 feature names, so they are constants rather than inline strings.
CONSTRAINT_OPEN_AT = "open_at"
CONSTRAINT_PRICE_MAX = "price_max"
CONSTRAINT_CATEGORY = "category"
CONSTRAINT_GEO = "geo"

# The Yelp attribute that holds price. `ParsedQuery.price_max` is the parsed
# form of a request against this field.
PRICE_ATTRIBUTE = "RestaurantsPriceRange2"

#: Constraints whose UNKNOWN verdict does **not** route through
#: ``cfg.retrieval.missing_attribute_policy``.
#:
#: Only the category check is exempt, and the reason is that it is answering a
#: different question from every other check here. "This business has no
#: NoiseLevel field" is a fact about the *dataset's coverage of the business*.
#: "None of this business's categories lexically matched the words the parser
#: emitted" is a fact about *our matcher and Yelp's taxonomy* — the business may
#: be a perfectly ordinary ramen shop tagged only "Japanese, Noodles"
#: (`check_categories` argues this at length). Both arrive as UNKNOWN, and
#: feeding both to one knob had two costs:
#:
#: * It made the knob partly a measurement of taxonomy gaps, so the D007
#:   comparison would not have measured what D007 names. A business with
#:   complete attribute data and no missing field at all could be penalized
#:   purely for failing a lexical category match.
#: * Under ``exclude`` it deleted exactly the candidates the dense retriever is
#:   there to rescue, which is the failure mode `check_categories` returns
#:   UNKNOWN specifically to avoid.
#:
#: Exempt does not mean ignored. The category verdict is still attached to the
#: candidate, still shown in "why this result", still a Stage-4 feature
#: (``category_match``), and still counted here — in
#: `FilterReport.unknown_by_constraint`, which is what lets the failure analysis
#: report the taxonomy gap as its own number instead of hiding it inside
#: ``n_penalized``.
#:
#: Geo is deliberately *not* exempt: "this business has no coordinates" is a
#: coverage fact like a missing attribute, so the policy is the right place to
#: decide what happens to it (`check_geo`).
POLICY_EXEMPT_CONSTRAINTS = frozenset({CONSTRAINT_CATEGORY})

# Reasons a business is dropped before/beyond constraint checking. Counted in
# `FilterReport` so the failure analysis can say *why* a query returned little.
#
# The two constraint-driven reasons are *prefixes*, not whole reasons: the
# recorded string is ``"<prefix>:<constraint name>"`` (see `drop_reason`). A
# single undifferentiated "violated_hard_constraint" cannot answer the one table
# that is buildable before any labels exist — "for constraint queries, what share
# of retrieved candidates did the hours check remove, and what share did the
# missing-data policy remove?" — because hours, price, an attribute and geo all
# collapsed into one counter. `FilterReport.drop_counts_by_kind` recovers the
# coarse view for callers that only want the four kinds.
DROP_UNKNOWN_BUSINESS = "unknown_business"
DROP_PERMANENTLY_CLOSED = "permanently_closed"
DROP_VIOLATED = "violated_hard_constraint"
DROP_MISSING_DATA = "missing_data_policy_exclude"

#: Separator between a drop reason's kind and the constraint that caused it.
DROP_REASON_SEPARATOR = ":"


def drop_reason(kind: str, constraint_name: str) -> str:
    """``"violated_hard_constraint:open_at"`` — one spelling, used by both sides.

    A function rather than an f-string at each site so the readers
    (`reason_kind`, `reason_constraint`) cannot drift from the writer.
    """
    return f"{kind}{DROP_REASON_SEPARATOR}{constraint_name}"


def reason_kind(reason: str) -> str:
    """The kind half of a recorded reason; the whole string when it has no half."""
    return reason.split(DROP_REASON_SEPARATOR, 1)[0]


def reason_constraint(reason: str) -> str | None:
    """The constraint half of a recorded reason, or None for the non-constraint kinds."""
    parts = reason.split(DROP_REASON_SEPARATOR, 1)
    return parts[1] if len(parts) == 2 else None


# Strings that mean "no value" once they reach us. `Business.attributes` is
# supposed to be normalized by the data layer, so in principle none of these can
# appear — this is defence in depth against the one variant that fails
# *silently*: the literal string "None" is truthy in Python, so a leaked
# OutdoorSeating of "None" would otherwise be read as "yes, outdoor seating"
# and quietly produce a wrong, confident answer. Fixture b008 pins this.
_MISSING_STRINGS = frozenset({"", "none", "null", "nan", "na", "n/a"})

# Boolean recognition is deliberately *strict*: only Python bools and the
# dataset's own "True"/"False" spellings. It is tempting to also accept
# "yes"/"no"/"1"/"0", and that is a bug — "no" is a real value in the WiFi
# vocabulary (free/paid/no). Coercing it would turn a perfectly comparable
# string pair ("WiFi: no" requested against "WiFi: no" stored) into a boolean
# against a vocabulary word, i.e. an UNKNOWN. Fixture b002 (WiFi "no", asked for
# "free") pins this: it must be VIOLATED, not UNKNOWN.
_TRUE_STRINGS = frozenset({"true"})
_FALSE_STRINGS = frozenset({"false"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Dropped when comparing categories so that "Coffee & Tea", "coffee and tea"
# and "coffee-tea" all collapse to the same thing. Keeping it to this one word
# avoids inventing a stopword list for a taxonomy that does not need one.
_CATEGORY_FILLER = frozenset({"and"})


# ─────────────────────────────────────────────────────────────────────────────
# Value normalization
# ─────────────────────────────────────────────────────────────────────────────


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS


def _as_bool(value: Any) -> bool | None:
    """Coerce an attribute value to a boolean, or None if it is not boolean-ish."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    return None


def _normalize_category(name: str) -> str:
    """Lowercase, strip every non-alphanumeric character, drop filler words.

    Case- and separator-insensitive by construction: "Coffee & Tea",
    "coffee-tea" and "COFFEE TEA" all become "coffeetea".
    """
    tokens = [t for t in _NON_ALNUM.split(name.lower()) if t and t not in _CATEGORY_FILLER]
    return "".join(tokens)


def _category_tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in _NON_ALNUM.split(name.lower()) if t and t not in _CATEGORY_FILLER)


# ─────────────────────────────────────────────────────────────────────────────
# Geo
# ─────────────────────────────────────────────────────────────────────────────

_EARTH_RADIUS_KM = 6371.0088  # mean radius (IUGG), the usual choice for haversine


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres.

    Haversine rather than a flat-earth approximation because it is three lines
    and removes a whole category of "why is this wrong at high latitude"
    question; rather than Vincenty/geodesic because its sub-metre accuracy is
    irrelevant next to the precision of a "near downtown" radius.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks. Each returns exactly one ConstraintMatch.
# ─────────────────────────────────────────────────────────────────────────────


def check_open_at(business: Business, open_at: OpenAt) -> ConstraintMatch:
    """Must-be-open check, delegated entirely to `concierge.data.hours.is_open_at`.

    All the hard parts — overnight intervals, the previous day's spillover,
    Yelp's "0:0-0:0" for 24 hours — live in that module and are tested there.
    Re-deriving any of it here would mean two implementations of the same
    calendar arithmetic that can disagree, which is exactly how "open past 10pm"
    queries break.

    `is_open_at` already answers three-valued, returning UNKNOWN when a business
    has no hours data at all. That matters: Yelp hours are missing often enough
    that treating absence as "closed" would silently delete a large slice of the
    corpus from every time-constrained query, and time-constrained queries are a
    whole `query_type` in the gold set.

    `is_open_at` raises `ValueError` for an unrecognized day or an out-of-range
    minute, and that is deliberately allowed to propagate rather than being
    mapped to UNKNOWN: it means the Stage 1 parser produced an `OpenAt` it should
    have rejected, and swallowing it here would hide a broken parser behind
    plausible-looking results.
    """
    status = is_open_at(business.hours, open_at.day, open_at.minute_of_day)
    return ConstraintMatch(
        name=CONSTRAINT_OPEN_AT,
        requested=open_at,
        # No hours data means there is nothing to report as the actual value;
        # "unknown" belongs in `status`, not disguised as a value.
        actual=None if status is ConstraintStatus.UNKNOWN else status.value,
        status=status,
    )


def check_price_max(business: Business, price_max: int) -> ConstraintMatch:
    """ "Cheap" means ``RestaurantsPriceRange2 <= price_max`` (1-4, i.e. $-$$$$).

    An inequality, not equality: someone asking for cheap ramen is not annoyed
    by cheaper ramen.
    """
    raw = business.attributes.get(PRICE_ATTRIBUTE)
    if _is_missing(raw):
        return ConstraintMatch(CONSTRAINT_PRICE_MAX, price_max, None, ConstraintStatus.UNKNOWN)
    try:
        actual = int(str(raw).strip())
    except (TypeError, ValueError):
        # Unparseable is not the same as absent, but for ranking purposes both
        # mean "we do not know this business's price band".
        log.debug("unparseable %s=%r on %s", PRICE_ATTRIBUTE, raw, business.business_id)
        return ConstraintMatch(CONSTRAINT_PRICE_MAX, price_max, raw, ConstraintStatus.UNKNOWN)

    status = ConstraintStatus.SATISFIED if actual <= price_max else ConstraintStatus.VIOLATED
    return ConstraintMatch(CONSTRAINT_PRICE_MAX, price_max, actual, status)


def _compare_attribute(requested: Any, actual: Any) -> bool | None:
    """Compare one requested attribute value against the stored one.

    Returns None for "the data does not say", which the caller turns into
    UNKNOWN. Two shapes have to be handled, and both occur in the dataset:

    * booleans, stored variously as ``True`` or ``"True"`` (OutdoorSeating);
    * short strings (``WiFi`` = free/paid/no, ``NoiseLevel`` = quiet/average/
      loud/very_loud).

    Nested attributes are deliberately *not* handled here, and this function is
    not where they could be. ``concierge.data.attributes`` flattens every nested
    blob to dotted leaf keys ("Ambience.casual") and keeps no parent key, so
    `Business.attributes` never holds a dict; and
    ``concierge.query.schema.SUPPORTED_ATTRIBUTES`` lists no dotted key, so a
    `ParsedQuery` can never request one. Ambience, BusinessParking, GoodForMeal
    and Music are therefore rendered into the business document and matched
    *semantically*, by BM25 and the embedder — not arithmetically, as hard
    constraints. That is the better home for a vibe ("good for a first date" is
    a judgement, not a field lookup), but it has a consequence worth being
    blunt about: ambience is **not** filterable in this system, and nothing in
    this module should read as though it were.
    """
    if _is_missing(actual):
        return None

    if isinstance(actual, Mapping) or isinstance(requested, Mapping):
        # Unreachable through the data layer and the query schema (see above), so
        # this is a contract breach rather than a data quirk — hence a raise and
        # not a quiet UNKNOWN. The string fallback below would otherwise compare
        # a dict's repr against a vocabulary word, get False, and drop the
        # business as VIOLATED: a confident wrong answer with no trace.
        raise TypeError(
            "attribute values must be flat by the time they reach the filters, "
            f"got requested={requested!r} actual={actual!r}. Nested attributes "
            "are flattened to dotted keys by concierge.data.attributes."
        )

    requested_bool = _as_bool(requested)
    actual_bool = _as_bool(actual)
    if requested_bool is not None and actual_bool is not None:
        return requested_bool == actual_bool
    if requested_bool is not None or actual_bool is not None:
        # One side is boolean and the other is not (e.g. requested WiFi=True vs
        # stored "free"). Comparing them would be guesswork; the parser is
        # responsible for emitting values in the dataset's own vocabulary.
        log.debug("attribute shape mismatch: requested=%r actual=%r", requested, actual)
        return None

    return str(requested).strip().lower() == str(actual).strip().lower()


def check_attribute(business: Business, key: str, requested: Any) -> ConstraintMatch:
    """Check one requested attribute.

    Note which decision is *not* made here: whether an attribute should be a
    hard constraint at all. "Quiet" is arguably a preference rather than a
    requirement, and enforcing it as hard removes every business with
    ``NoiseLevel == "average"``. That call belongs to the Stage 1 parser, which
    decides what lands in `ParsedQuery.attributes`; this function only enforces
    what it is given, so the two concerns stay separately measurable.
    """
    actual = business.attributes.get(key)
    verdict = _compare_attribute(requested, actual)
    if verdict is None:
        return ConstraintMatch(
            key, requested, None if _is_missing(actual) else actual, ConstraintStatus.UNKNOWN
        )
    status = ConstraintStatus.SATISFIED if verdict else ConstraintStatus.VIOLATED
    return ConstraintMatch(key, requested, actual, status)


def check_categories(business: Business, categories: Sequence[str]) -> ConstraintMatch:
    """Any-of matching against the business's category list.

    Any-of, not all-of: a parser asked about "ramen" may emit
    ``("Ramen", "Noodles", "Japanese")`` as near-synonyms, and requiring all of
    them would reject the obvious answer. Matching is case- and
    separator-insensitive, and also succeeds when the requested category's words
    are a subset of a business category's words — so "coffee" matches
    "Coffee & Tea", which is what an LLM parser will actually emit, since it
    does not know Yelp's 1,300-entry taxonomy verbatim.

    Known limitation, worth stating rather than hiding: this is lexical. It will
    not match "noodles" to "Ramen", or "pub" to "Bars", and Yelp's own category
    lists are incomplete — a ramen shop tagged only "Japanese, Noodles" is an
    ordinary thing to find in the data. Closing that gap is the dense
    retriever's job, not this filter's.

    Which is why a miss here is UNKNOWN and never VIOLATED. The distinction is
    load-bearing, so it is argued rather than asserted:

    * A miss means "this lexical matcher could not establish a match", not "the
      data refutes the request". That is precisely what UNKNOWN means in this
      module's three-valued scheme, and an incomplete category list is a
      missing-data problem like any other.
    * VIOLATED is dropped unconditionally (`apply_filters`, step 3). Returning
      it for a lexical miss would delete the correct answer from the candidate
      list before any ranker saw it — and would foreclose the mitigation named
      two paragraphs up, since the dense retriever cannot rescue a candidate
      that is no longer in the list. The failure mode is the nastiest kind:
      ten plausible results come back and the right one is simply absent.
    * An UNKNOWN from *this* check is exempt from the missing-data policy
      (`POLICY_EXEMPT_CONSTRAINTS`), so it neither drops nor discounts the
      candidate. A lexical miss is a gap in this matcher, not in the business's
      data, and a knob named ``missing_attribute_policy`` must not be measuring
      Yelp's taxonomy. The miss is still counted, in
      `FilterReport.unknown_by_constraint` under `CONSTRAINT_CATEGORY`, so the
      failure analysis can report how often the lexicon missed as its own
      number.

    So a category miss is a *signal*, not a filter: it reaches the ranking only
    through the Stage-4 ``category_match`` feature (NaN on a miss — "no signal",
    not "wrong kind of place") and through BM25 and the embedder, which are the
    components that can actually close the gap. The cost of that choice, stated
    plainly: a query whose category words miss the taxonomy gets no help here at
    all, and recovering the right answer is entirely up to retrieval and the
    reranker. Whether that is cheaper than deleting the candidate is measurable
    once labels exist (DECISIONS.md D007).
    """
    if not business.categories:
        # No category data at all. Same status as a miss below, but `actual` is
        # None rather than a list, so the two are still distinguishable in the
        # "why this result" UI and in debugging.
        return ConstraintMatch(
            CONSTRAINT_CATEGORY, tuple(categories), None, ConstraintStatus.UNKNOWN
        )

    business_norm = {_normalize_category(c) for c in business.categories}
    business_token_sets = [_category_tokens(c) for c in business.categories]

    for requested in categories:
        norm = _normalize_category(requested)
        if not norm:
            continue
        if norm in business_norm:
            return ConstraintMatch(
                CONSTRAINT_CATEGORY, tuple(categories), requested, ConstraintStatus.SATISFIED
            )
        tokens = _category_tokens(requested)
        if tokens and any(tokens <= bt for bt in business_token_sets):
            return ConstraintMatch(
                CONSTRAINT_CATEGORY, tuple(categories), requested, ConstraintStatus.SATISFIED
            )

    # Nothing matched. `actual` carries the business's own categories so the
    # explanation can say what we failed to match against.
    return ConstraintMatch(
        CONSTRAINT_CATEGORY, tuple(categories), business.categories, ConstraintStatus.UNKNOWN
    )


def check_geo(business: Business, geo: GeoConstraint) -> ConstraintMatch:
    """Haversine distance against the requested radius.

    A business with no coordinates is UNKNOWN — the check never returns
    VIOLATED for missing data, so such a business is never dropped *by this
    check*. What happens to it afterwards is the missing-data policy's call,
    which is where that decision is measured; hiding an unconditional drop
    inside a geo check would take it out of the measurement.
    """
    if business.latitude is None or business.longitude is None:
        return ConstraintMatch(CONSTRAINT_GEO, geo, None, ConstraintStatus.UNKNOWN)
    distance = haversine_km(business.latitude, business.longitude, geo.latitude, geo.longitude)
    status = ConstraintStatus.SATISFIED if distance <= geo.radius_km else ConstraintStatus.VIOLATED
    return ConstraintMatch(CONSTRAINT_GEO, geo, round(distance, 3), status)


def evaluate_constraints(business: Business, parsed: ParsedQuery) -> tuple[ConstraintMatch, ...]:
    """Every hard constraint in the query, checked against one business.

    Returns a match per constraint even when it passes, because the
    "why this result" UI shows what a result satisfied, and Stage 4 uses the
    satisfied/unknown counts as features.
    """
    matches: list[ConstraintMatch] = []
    if parsed.open_at is not None:
        matches.append(check_open_at(business, parsed.open_at))
    if parsed.price_max is not None:
        matches.append(check_price_max(business, parsed.price_max))
    for key, requested in parsed.attributes.items():
        # Price arrives as `price_max` and is checked as an inequality above.
        # Checking it again here would compare it for equality and contradict
        # that, so the dedicated field wins.
        if key == PRICE_ATTRIBUTE:
            continue
        matches.append(check_attribute(business, key, requested))
    if parsed.categories:
        matches.append(check_categories(business, parsed.categories))
    if parsed.geo is not None:
        matches.append(check_geo(business, parsed.geo))
    return tuple(matches)


# ─────────────────────────────────────────────────────────────────────────────
# Applying the policy to a candidate list
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FilterReport:
    """Survivors plus an account of what was removed and why.

    The drop reasons are not debug noise: "this query returned three results"
    needs an explanation in the failure analysis, and "180 of 200 candidates
    had no NoiseLevel data" is that explanation.
    """

    kept: tuple[Candidate, ...]
    dropped: dict[str, str] = field(default_factory=dict)  # business_id -> reason
    #: How many candidates the ``penalize`` policy discounted. Zero under the
    #: other two policies, because neither discounts anything.
    n_penalized: int = 0
    #: The same count split by the constraint that was UNKNOWN. A candidate
    #: missing two attributes is counted once per attribute here but once in
    #: ``n_penalized``, so the two are not expected to sum to each other — this
    #: view answers "which check is doing the penalising", which a single
    #: integer cannot. Empty unless the policy is ``penalize``, and it never
    #: contains an exempt constraint (`POLICY_EXEMPT_CONSTRAINTS`).
    penalized_by_constraint: dict[str, int] = field(default_factory=dict)
    #: Census of UNKNOWN verdicts, by constraint, over every candidate that
    #: reached the missing-data policy — i.e. after the unconditional
    #: VIOLATED drops, which is the population the policy actually governs.
    #:
    #: Policy-independent on purpose: identical under all three policies, so it
    #: is the denominator the three runs are compared against, and it is where
    #: the exempt category check is visible. This is the counter that answers
    #: the one table that needs no relevance labels at all — "for constraint
    #: queries, what share of retrieved candidates had no hours data, and what
    #: share just missed the category lexicon?" — which ``n_penalized`` alone
    #: cannot, because it sums causes that lead to different conclusions.
    unknown_by_constraint: dict[str, int] = field(default_factory=dict)

    @property
    def drop_counts(self) -> dict[str, int]:
        """Reason string -> count, at full per-constraint resolution."""
        counts: dict[str, int] = {}
        for reason in self.dropped.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    @property
    def drop_counts_by_kind(self) -> dict[str, int]:
        """The same counts collapsed to the four ``DROP_*`` kinds.

        For callers that want "how many did the policy remove" without caring
        which constraint was unknown. Kept as a separate property rather than
        making `drop_counts` coarse, because the coarse version is the one that
        lost the information this report exists to carry.
        """
        counts: dict[str, int] = {}
        for reason in self.dropped.values():
            kind = reason_kind(reason)
            counts[kind] = counts.get(kind, 0) + 1
        return counts


def apply_filters(
    candidates: Sequence[Candidate],
    businesses: Mapping[str, Business],
    parsed: ParsedQuery,
    cfg: Config,
) -> FilterReport:
    """Apply the query's hard constraints and the missing-data policy.

    Order of operations per candidate:

    1. No `Business` record -> drop. We could neither check it nor render it.
    2. ``is_open == 0`` -> drop. The index build already excludes permanently
       closed businesses; this is a second line of defence, because showing a
       closed business is the most embarrassing possible wrong answer and the
       check costs one boolean.
    3. Any VIOLATED constraint -> drop, unconditionally. This is why each check
       returns VIOLATED only when the stored data contradicts the request; a
       check that could not decide returns UNKNOWN and lands in step 4, where
       the consequence is a measured policy rather than a deletion.
    4. Any UNKNOWN constraint that is not in `POLICY_EXEMPT_CONSTRAINTS` ->
       ``cfg.retrieval.missing_attribute_policy``: ``exclude`` drops it,
       ``penalize`` multiplies its fused score by ``missing_attribute_penalty``,
       ``neutral`` leaves it alone. The module docstring sets out exactly how
       far ``penalize`` reaches down the cascade and therefore which ablation
       rows the D007 comparison can be read on.

    Every drop and every penalty is recorded *per constraint*, not per kind. One
    undifferentiated "violated_hard_constraint" counter cannot answer the
    cheapest useful question in the failure analysis — "for constraint queries,
    what share of retrieved candidates did the hours check remove and what share
    did the missing-data policy remove?" — because hours, price, an attribute
    and geo all collapse into it. See `drop_reason` for the recorded spelling and
    `FilterReport` for the three counters.

    Note that this function does not truncate. Stage 3 scores every survivor
    (that is the architecture: score ~200, keep ~50), so filtering changes the
    candidate *set* only through the four drop paths numbered above — and only
    the fourth of them depends on the policy.

    Survivors are re-sorted and ``RANK_RRF`` is recomputed over them, so Stage 2
    hands Stage 3 a contiguous 1..N ranking. Filtering is part of Stage 2, so
    its output ranks are the post-filter ranks; leaving the pre-filter ranks in
    place would mean the "rank" feature Stage 4 sees did not correspond to any
    list that ever existed. The penalty is not recorded as a separate score
    because it is recoverable: it is a function of the attached
    `ConstraintMatch` statuses and the config.
    """
    policy = cfg.retrieval.missing_attribute_policy
    penalty = cfg.retrieval.missing_attribute_penalty

    kept: list[Candidate] = []
    dropped: dict[str, str] = {}
    n_penalized = 0
    penalized_by_constraint: dict[str, int] = {}
    unknown_by_constraint: dict[str, int] = {}

    for candidate in candidates:
        business = businesses.get(candidate.business_id)
        if business is None:
            dropped[candidate.business_id] = DROP_UNKNOWN_BUSINESS
            continue
        if not business.is_open:
            dropped[candidate.business_id] = DROP_PERMANENTLY_CLOSED
            continue

        matches = evaluate_constraints(business, parsed)
        violated = [m for m in matches if m.status is ConstraintStatus.VIOLATED]
        if violated:
            # One candidate is dropped once, so one reason is recorded, and it
            # names the *first* refuting constraint in `evaluate_constraints`'
            # fixed check order (hours, price, attributes, category, geo). That
            # keeps the reason counts summing to the number of candidates
            # dropped, which is the property `FilterReport.drop_counts` and
            # `SearchResponse.n_candidates` are read as having. The cost: a
            # candidate that fails both hours and price is attributed to hours
            # alone, so the counters are "what removed it first", not "every way
            # it failed".
            dropped[candidate.business_id] = drop_reason(DROP_VIOLATED, violated[0].name)
            continue

        unknown = [m for m in matches if m.status is ConstraintStatus.UNKNOWN]
        for match in unknown:
            # Counted before the policy branches, and for exempt constraints
            # too, so this census is identical under all three policies — which
            # is what makes it usable as the denominator when comparing them.
            unknown_by_constraint[match.name] = unknown_by_constraint.get(match.name, 0) + 1

        governed = [m for m in unknown if m.name not in POLICY_EXEMPT_CONSTRAINTS]
        scores = dict(candidate.scores)
        if governed:
            if policy == "exclude":
                dropped[candidate.business_id] = drop_reason(DROP_MISSING_DATA, governed[0].name)
                continue
            if policy == "penalize":
                current = scores.get(SCORE_RRF)
                if current is None:
                    # No fused score to discount. Writing ``0.0 * penalty`` here
                    # (the previous behaviour) would invent a fused score for a
                    # candidate fusion never scored, and the Stage 4 feature
                    # builder reads an absent score as NaN — "this stage never
                    # scored this candidate" — which is the honest value. Not
                    # reachable through `fusion.fuse`, which always writes
                    # SCORE_RRF; guarded because the penalty must not be the
                    # thing that creates one. Not counted as penalized either:
                    # nothing was discounted.
                    log.debug(
                        "no %s to penalize on %s; leaving it alone",
                        SCORE_RRF,
                        candidate.business_id,
                    )
                else:
                    # Discounts the fused Stage-2 score only. That is the
                    # Stage-2 ordering and, through the recomputed RANK_RRF
                    # below, the Stage-4 ``rrf_rank`` feature — but not the
                    # Stage-3 ordering, which is rescored from scratch. The
                    # module docstring says which ablation rows that makes the
                    # D007 comparison readable on.
                    scores[SCORE_RRF] = current * penalty
                    n_penalized += 1
                    for match in governed:
                        penalized_by_constraint[match.name] = (
                            penalized_by_constraint.get(match.name, 0) + 1
                        )
            # "neutral": the score is left exactly as fusion produced it.

        kept.append(
            Candidate(
                business_id=candidate.business_id,
                scores=scores,
                ranks=dict(candidate.ranks),
                constraints=matches,
            )
        )

    # Deterministic tie-break on business_id, for the same reason as in fusion:
    # a penalty creates exact ties (several candidates multiplied by the same
    # factor), and an unreproducible order makes an A/B difference
    # un-attributable.
    kept.sort(key=lambda c: (-c.scores.get(SCORE_RRF, 0.0), c.business_id))
    reranked = tuple(
        Candidate(
            business_id=c.business_id,
            scores=c.scores,
            ranks={**c.ranks, RANK_RRF: position},
            constraints=c.constraints,
        )
        for position, c in enumerate(kept, start=1)
    )
    return FilterReport(
        kept=reranked,
        dropped=dropped,
        n_penalized=n_penalized,
        penalized_by_constraint=penalized_by_constraint,
        unknown_by_constraint=unknown_by_constraint,
    )
