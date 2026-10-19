"""Shared data contracts for the whole pipeline.

Every stage speaks in these types. They are defined in one place because the
four ranking stages are developed independently: if each stage invented its own
notion of "a business" or "a parsed query", the seams between them would be
where the bugs live.

Design rules:
  * Frozen dataclasses. Stages must not mutate each other's objects — a reranker
    that quietly rewrites a retrieval score makes the ablation untraceable.
  * Scores are carried in a dict keyed by stage name, never collapsed into one
    number too early. Stage 4 needs every upstream score as a *feature*, so
    nothing may be thrown away between stages.
  * Constraint outcomes are three-valued, not boolean. "This café has no
    NoiseLevel data" is different from "this café is loud", and conflating them
    is the single most common source of wrong results in this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Business side
# ─────────────────────────────────────────────────────────────────────────────

DayOfWeek = str  # "Monday" ... "Sunday", as spelled in the Yelp dataset


@dataclass(frozen=True, slots=True)
class HoursInterval:
    """One opening interval, in minutes from midnight.

    Yelp stores hours as strings like "7:0-20:0". Some intervals cross midnight
    ("17:0-2:0" means 5pm until 2am the *next* day), so `end_min` may be less
    than `start_min`. Callers must use `covers()` rather than comparing the
    endpoints themselves — getting this wrong silently breaks every "open late"
    query, which is a large share of the gold set.
    """

    start_min: int
    end_min: int

    @property
    def crosses_midnight(self) -> bool:
        return self.end_min <= self.start_min

    def covers(self, minute_of_day: int) -> bool:
        """True if the given minute-of-day falls inside this interval."""
        if self.crosses_midnight:
            return minute_of_day >= self.start_min or minute_of_day < self.end_min
        return self.start_min <= minute_of_day < self.end_min


@dataclass(frozen=True, slots=True)
class Business:
    """A searchable business, after parsing and cleaning.

    `attributes` holds *normalized* values only. The raw dataset stores many of
    them as stringified Python literals ("u'quiet'", "{'casual': True}"), which
    are resolved during the data build so that nothing downstream has to know
    about that quirk. A key absent from the dict means "no data", which is
    meaningfully different from a False value.
    """

    business_id: str
    name: str
    city: str
    state: str
    latitude: float | None
    longitude: float | None
    stars: float
    review_count: int
    categories: tuple[str, ...]
    attributes: dict[str, Any]
    hours: dict[DayOfWeek, tuple[HoursInterval, ...]]
    is_open: bool

    def has_attribute_data(self, key: str) -> bool:
        return key in self.attributes and self.attributes[key] is not None


@dataclass(frozen=True, slots=True)
class BusinessDocument:
    """The text a retrieval or reranking model actually sees.

    Kept separate from `Business` because the document is a *rendering* choice
    (which reviews to include, how to phrase attributes, where to truncate) and
    that choice is an experimental variable we measure, not a property of the
    business.
    """

    business_id: str
    text: str
    # Where the text came from, for debugging and for the "why this result" UI.
    n_reviews_used: int = 0
    n_tips_used: int = 0
    truncated: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Query side
# ─────────────────────────────────────────────────────────────────────────────


class ParseSource(StrEnum):
    """Where a parsed query came from. Reported in the ablation: the LLM parser
    and the rule fallback have different quality *and* different latency."""

    LLM = "llm"
    CACHE = "cache"
    RULES = "rules"


@dataclass(frozen=True, slots=True)
class OpenAt:
    """A "must be open at this time" constraint."""

    day: DayOfWeek
    minute_of_day: int


@dataclass(frozen=True, slots=True)
class GeoConstraint:
    """A location constraint, e.g. "near downtown".

    Resolved to a point plus a radius during parsing. Kept optional throughout:
    most queries have no geo component, and a business with no coordinates
    cannot satisfy or violate one.
    """

    latitude: float
    longitude: float
    radius_km: float
    label: str = ""  # the phrase the user typed, for the explanation UI


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Stage 1 output: the user's request split into soft text and hard filters.

    The split matters. `semantic_text` is what retrieval and reranking models
    embed and score; the remaining fields are *filters* that must be checked
    arithmetically. An embedding will cheerfully return a café that closes at
    6pm for "open past 10pm" because the prose sounds right.
    """

    raw: str
    semantic_text: str
    open_at: OpenAt | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    price_max: int | None = None
    categories: tuple[str, ...] = ()
    geo: GeoConstraint | None = None
    source: ParseSource = ParseSource.RULES
    parse_latency_ms: float = 0.0

    @property
    def n_hard_constraints(self) -> int:
        """Used as a LambdaMART feature and to break down the failure analysis
        by how constrained a query is."""
        return (
            (1 if self.open_at else 0)
            + len(self.attributes)
            + (1 if self.price_max is not None else 0)
            + (1 if self.geo else 0)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result side
# ─────────────────────────────────────────────────────────────────────────────


class ConstraintStatus(StrEnum):
    """Three-valued on purpose. UNKNOWN means the dataset has no data for this
    constraint on this business — we must never present that as a pass or a
    fail, in the ranking or in the UI."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConstraintMatch:
    """One constraint checked against one business. Drives both the ranking
    features and the user-facing "why this result" explanation."""

    name: str
    requested: Any
    actual: Any
    status: ConstraintStatus

    @property
    def explanation(self) -> str:
        if self.status is ConstraintStatus.SATISFIED:
            return f"{self.name}: {self.actual}"
        if self.status is ConstraintStatus.VIOLATED:
            return f"{self.name}: wanted {self.requested}, has {self.actual}"
        return f"{self.name}: no data"


# Canonical score keys. Stages write into `Candidate.scores` using these so the
# Stage 4 feature builder can rely on the names.
SCORE_BM25 = "bm25"
SCORE_DENSE = "dense"
SCORE_RRF = "rrf"
SCORE_CROSS_ENCODER = "cross_encoder"
SCORE_LTR = "ltr"

RANK_BM25 = "bm25"
RANK_DENSE = "dense"
RANK_RRF = "rrf"
RANK_CROSS_ENCODER = "cross_encoder"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A business in flight through the cascade.

    Accumulates a score and a rank per stage rather than overwriting a single
    value, because Stage 4 consumes all of them as features and the ablation
    needs to attribute improvements to specific stages.
    """

    business_id: str
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    constraints: tuple[ConstraintMatch, ...] = ()

    def with_score(self, key: str, value: float) -> Candidate:
        """Return a copy carrying an additional score (frozen type, so no
        in-place mutation)."""
        return Candidate(
            business_id=self.business_id,
            scores={**self.scores, key: value},
            ranks=dict(self.ranks),
            constraints=self.constraints,
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One of the final ten, ready to render."""

    business: Business
    rank: int
    score: float
    constraints: tuple[ConstraintMatch, ...]
    # Kept for the API response and for debugging individual queries.
    stage_scores: dict[str, float] = field(default_factory=dict)

    @property
    def satisfied(self) -> tuple[ConstraintMatch, ...]:
        return tuple(c for c in self.constraints if c.status is ConstraintStatus.SATISFIED)

    @property
    def unknown(self) -> tuple[ConstraintMatch, ...]:
        return tuple(c for c in self.constraints if c.status is ConstraintStatus.UNKNOWN)


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Per-stage wall-clock, collected on every search so p50/p95 can be
    reported per stage and not just end to end."""

    stage: str
    ms: float


@dataclass(frozen=True, slots=True)
class SearchResponse:
    parsed: ParsedQuery
    results: tuple[SearchResult, ...]
    timings: tuple[StageTiming, ...]
    total_ms: float
    n_candidates: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


class QueryType(StrEnum):
    """Every metric is broken down by these, because the stages help different
    kinds of query by very different amounts — that breakdown is a headline
    result, not a diagnostic."""

    VIBE = "vibe"
    CONSTRAINT = "constraint"
    SPECIFIC = "specific"
    MESSY = "messy"


@dataclass(frozen=True, slots=True)
class GoldQuery:
    query_id: str
    query: str
    query_type: QueryType


# Graded relevance, as labeled by the owner.
RELEVANCE_IRRELEVANT = 0
RELEVANCE_PARTIAL = 1
RELEVANCE_HIGH = 2
