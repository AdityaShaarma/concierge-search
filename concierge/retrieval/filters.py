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

So every check returns SATISFIED / VIOLATED / UNKNOWN, and the two outcomes are
handled by *different* mechanisms:

* **VIOLATED removes the business, always.** The data says it fails. Showing it
  would be a wrong answer, and no ranking signal should be able to outvote that.
* **UNKNOWN follows ``cfg.retrieval.missing_attribute_policy``** — ``exclude``,
  ``penalize``, or ``neutral``. All three are implemented and none is
  privileged in this code. Which is best is an open measurement (DECISIONS.md
  D007); the penalty value of 0.5 is a placeholder for a number that has not
  been measured yet, and this module must not pretend otherwise.

Keeping that distinction crisp is the entire point of the design.
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

# Reasons a business is dropped before/beyond constraint checking. Counted in
# `FilterReport` so the failure analysis can say *why* a query returned little.
DROP_UNKNOWN_BUSINESS = "unknown_business"
DROP_PERMANENTLY_CLOSED = "permanently_closed"
DROP_VIOLATED = "violated_hard_constraint"
DROP_MISSING_DATA = "missing_data_policy_exclude"

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
    """"Cheap" means ``RestaurantsPriceRange2 <= price_max`` (1-4, i.e. $-$$$$).

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

    status = (
        ConstraintStatus.SATISFIED if actual <= price_max else ConstraintStatus.VIOLATED
    )
    return ConstraintMatch(CONSTRAINT_PRICE_MAX, price_max, actual, status)


def _compare_attribute(requested: Any, actual: Any) -> bool | None:
    """Compare one requested attribute value against the stored one.

    Returns None for "the data does not say", which the caller turns into
    UNKNOWN. Three shapes have to be handled, all of which occur in the dataset:

    * booleans, stored variously as ``True`` or ``"True"`` (OutdoorSeating);
    * short strings (``WiFi`` = free/paid/no, ``NoiseLevel`` = quiet/average/
      loud/very_loud);
    * nested dicts (``Ambience`` = ``{"romantic": False, "casual": True}``).
      A request may name a single sub-key ("romantic"), which must be True in
      the dict; a sub-key the dict does not mention is UNKNOWN, not False.
    """
    if _is_missing(actual):
        return None

    if isinstance(actual, dict):
        if isinstance(requested, Mapping):
            sub = [_compare_attribute(v, actual.get(k)) for k, v in requested.items()]
            if any(s is False for s in sub):
                return False
            if any(s is None for s in sub) or not sub:
                return None
            return True
        key = str(requested).strip().lower()
        for actual_key, actual_value in actual.items():
            if str(actual_key).strip().lower() == key:
                return _compare_attribute(True, actual_value)
        return None  # dict exists but says nothing about this sub-key

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
        return ConstraintMatch(key, requested, None if _is_missing(actual) else actual,
                               ConstraintStatus.UNKNOWN)
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
    not match "noodles" to "Ramen", or "pub" to "Bars". Closing that gap is the
    dense retriever's job, not this filter's.
    """
    if not business.categories:
        return ConstraintMatch(CONSTRAINT_CATEGORY, tuple(categories), None,
                               ConstraintStatus.UNKNOWN)

    business_norm = {_normalize_category(c) for c in business.categories}
    business_token_sets = [_category_tokens(c) for c in business.categories]

    for requested in categories:
        norm = _normalize_category(requested)
        if not norm:
            continue
        if norm in business_norm:
            return ConstraintMatch(CONSTRAINT_CATEGORY, tuple(categories), requested,
                                   ConstraintStatus.SATISFIED)
        tokens = _category_tokens(requested)
        if tokens and any(tokens <= bt for bt in business_token_sets):
            return ConstraintMatch(CONSTRAINT_CATEGORY, tuple(categories), requested,
                                   ConstraintStatus.SATISFIED)

    return ConstraintMatch(CONSTRAINT_CATEGORY, tuple(categories),
                           business.categories, ConstraintStatus.VIOLATED)


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
    status = (
        ConstraintStatus.SATISFIED if distance <= geo.radius_km else ConstraintStatus.VIOLATED
    )
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
    n_penalized: int = 0

    @property
    def drop_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason in self.dropped.values():
            counts[reason] = counts.get(reason, 0) + 1
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
    3. Any VIOLATED constraint -> drop, unconditionally.
    4. Any UNKNOWN constraint -> ``cfg.retrieval.missing_attribute_policy``:
       ``exclude`` drops it, ``penalize`` multiplies its fused score by
       ``missing_attribute_penalty``, ``neutral`` leaves it alone.

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

    for candidate in candidates:
        business = businesses.get(candidate.business_id)
        if business is None:
            dropped[candidate.business_id] = DROP_UNKNOWN_BUSINESS
            continue
        if not business.is_open:
            dropped[candidate.business_id] = DROP_PERMANENTLY_CLOSED
            continue

        matches = evaluate_constraints(business, parsed)
        if any(m.status is ConstraintStatus.VIOLATED for m in matches):
            dropped[candidate.business_id] = DROP_VIOLATED
            continue

        has_unknown = any(m.status is ConstraintStatus.UNKNOWN for m in matches)
        scores = dict(candidate.scores)
        if has_unknown:
            if policy == "exclude":
                dropped[candidate.business_id] = DROP_MISSING_DATA
                continue
            if policy == "penalize":
                scores[SCORE_RRF] = scores.get(SCORE_RRF, 0.0) * penalty
                n_penalized += 1
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
    return FilterReport(kept=reranked, dropped=dropped, n_penalized=n_penalized)
