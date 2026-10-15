"""The HTTP wire contract, kept deliberately separate from ``concierge.types``.

Why two sets of types at all — this is the question an interviewer will ask, so
here is the whole argument:

1. **The wire format is a public contract; the internal types are not.**
   ``concierge/types.py`` is tuned for the four ranking stages: frozen
   dataclasses, a `scores` dict keyed by stage so nothing is thrown away before
   Stage 4, three-valued `ConstraintStatus`. Those choices exist to make the
   *ablation* traceable. A browser needs none of that, and the day we add a
   fifth score key the API response should not silently change shape.

2. **Serialising internals leaks internals.** If the response were
   ``dataclasses.asdict(SearchResponse)``, then renaming a field inside the
   pipeline would be a breaking API change, and the Next.js client would be
   coupled to the reranker's score-key constants. Both sides become harder to
   change at once — the classic reason to put an adapter at a boundary.

3. **Some internal values are not JSON at all.** ``ConstraintMatch.requested``
   holds whatever the constraint was about: an ``OpenAt``, a ``GeoConstraint``,
   a tuple of categories, an int, a bool. The pipeline is right not to care;
   the HTTP layer has to. ``_jsonable`` below is that translation, in one place.

The dependency runs one way only: this module imports ``concierge.types``, and
nothing in ``concierge/`` may ever import this module. That is what keeps the
pipeline usable from the eval harness, a notebook, and a test with no HTTP
anywhere in sight.

The adapters (``from_domain``) live here rather than in ``app.py`` so that
``app.py`` stays about HTTP concerns — status codes, CORS, lifespan — and each
mapping sits next to the schema it produces.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from concierge.types import (
    ConstraintMatch,
    ConstraintStatus,
    ParsedQuery,
    SearchResponse,
    SearchResult,
)

#: Largest `limit` the API will accept. Not an arbitrary round number: the
#: cascade hands at most `cfg.candidates.rerank` (50 by default) candidates to
#: the final stage, so a larger limit could not return more results — it would
#: just quietly return fewer than asked for, which looks like a bug. Rejecting
#: it with a 422 is the honest answer.
MAX_LIMIT = 50

#: Milliseconds are rounded to microsecond precision for display. This is a
#: formatting choice, not a measurement choice: `total_ms` as measured is a
#: float with float noise, and 17 significant digits in a JSON body is noise
#: that invites false precision. Nothing is aggregated or re-derived here.
_MS_DIGITS = 6


def _jsonable(value: Any) -> Any:
    """Coerce an arbitrary internal value into something JSON can carry.

    Needed because a `ConstraintMatch` is deliberately generic: `requested` may
    be an `OpenAt` dataclass ("open Friday at 22:00"), a `GeoConstraint`, a
    tuple of candidate categories, an int price band, a bool, or a nested dict
    from Yelp's stringified attributes.

    The fallback is `str()` rather than an exception: a single odd attribute
    value must never turn a good result set into a 500. It will render as text
    in the UI, which is the correct degradation for an explanation string.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _ms(value: float) -> float:
    return round(float(value), _MS_DIGITS)


# ─────────────────────────────────────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Body of ``POST /search``.

    ``extra="forbid"`` on purpose. A client that posts ``{"q": "ramen"}`` has a
    bug; answering it with results for an empty query would hide that bug
    behind a plausible-looking empty response. A 422 naming the unexpected field
    is far cheaper to debug, and this API has exactly one client.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=400,
        description="The user's request in natural language.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=MAX_LIMIT,
        description=f"How many results to return (1-{MAX_LIMIT}).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response pieces
# ─────────────────────────────────────────────────────────────────────────────


class ConstraintView(BaseModel):
    """One hard constraint checked against one business.

    `explanation` is carried from the domain object rather than rebuilt here:
    the phrasing of "no data" versus "wanted $$, has $$$" is a product decision
    that belongs with the constraint logic, and two copies of it would drift.
    """

    name: str
    requested: Any = None
    actual: Any = None
    status: Literal["satisfied", "violated", "unknown"]
    explanation: str

    @classmethod
    def from_domain(cls, match: ConstraintMatch) -> ConstraintView:
        return cls(
            name=match.name,
            requested=_jsonable(match.requested),
            actual=_jsonable(match.actual),
            status=match.status.value,
            explanation=match.explanation,
        )


class ConstraintGroups(BaseModel):
    """The "why this result" payload, pre-split into the three buckets the UI
    renders differently.

    Split server-side rather than client-side because the three-valued
    distinction is the single easiest thing for a frontend to get wrong: an
    `unknown` rendered as a red cross tells the user a quiet café is loud. The
    API makes that mistake impossible to make by accident.
    """

    matched: list[ConstraintView] = Field(default_factory=list)
    unmatched: list[ConstraintView] = Field(default_factory=list)
    unknown: list[ConstraintView] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, matches: Sequence[ConstraintMatch]) -> ConstraintGroups:
        buckets: dict[ConstraintStatus, list[ConstraintView]] = {
            ConstraintStatus.SATISFIED: [],
            ConstraintStatus.VIOLATED: [],
            ConstraintStatus.UNKNOWN: [],
        }
        for match in matches:
            buckets[match.status].append(ConstraintView.from_domain(match))
        return cls(
            matched=buckets[ConstraintStatus.SATISFIED],
            unmatched=buckets[ConstraintStatus.VIOLATED],
            unknown=buckets[ConstraintStatus.UNKNOWN],
        )


class BusinessView(BaseModel):
    """The business fields the demo UI shows.

    A subset of `Business`, not all of it: `is_open` is always true for anything
    in the index (closed businesses are filtered out at build time), and the raw
    `hours` intervals are minutes-from-midnight integers that would need their
    own rendering contract. Added only when the UI needs them.
    """

    business_id: str
    name: str
    city: str
    state: str
    latitude: float | None = None
    longitude: float | None = None
    stars: float
    review_count: int
    categories: list[str] = Field(default_factory=list)
    #: Normalized attributes, JSON-coerced. Useful for rendering amenity badges
    #: next to a result. Absent keys mean "the dataset has no data", which the
    #: UI must not render as a negative — same rule as `ConstraintGroups`.
    attributes: dict[str, Any] = Field(default_factory=dict)


class ResultView(BaseModel):
    rank: int
    #: The score of the last stage that actually ran, i.e. the number the result
    #: was ordered by. `stage_scores` keeps every upstream score so the demo can
    #: show how a result moved through the cascade.
    score: float
    business: BusinessView
    constraints: ConstraintGroups
    stage_scores: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, result: SearchResult) -> ResultView:
        b = result.business
        return cls(
            rank=result.rank,
            score=float(result.score),
            business=BusinessView(
                business_id=b.business_id,
                name=b.name,
                city=b.city,
                state=b.state,
                latitude=b.latitude,
                longitude=b.longitude,
                stars=float(b.stars),
                review_count=int(b.review_count),
                categories=list(b.categories),
                attributes={k: _jsonable(v) for k, v in b.attributes.items()},
            ),
            constraints=ConstraintGroups.from_domain(result.constraints),
            stage_scores={k: float(v) for k, v in result.stage_scores.items()},
        )


class ParsedQueryView(BaseModel):
    """What Stage 1 understood.

    Exposed to the client for two reasons: the demo shows the user how their
    sentence was read ("open Friday after 22:00, WiFi: free"), and `source`
    makes a fallback *visible*. An LLM parse failure is not an error — the
    parser contract says it degrades to rules — but a silent degradation would
    be indistinguishable from the LLM working, which is exactly the kind of
    thing that makes a latency or quality number wrong without anyone noticing.
    """

    semantic_text: str
    open_at: dict[str, Any] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    price_max: int | None = None
    categories: list[str] = Field(default_factory=list)
    geo: dict[str, Any] | None = None
    source: Literal["llm", "cache", "rules"]
    parse_latency_ms: float
    n_hard_constraints: int

    @classmethod
    def from_domain(cls, parsed: ParsedQuery) -> ParsedQueryView:
        return cls(
            semantic_text=parsed.semantic_text,
            open_at=_jsonable(parsed.open_at) if parsed.open_at else None,
            attributes={k: _jsonable(v) for k, v in parsed.attributes.items()},
            price_max=parsed.price_max,
            categories=list(parsed.categories),
            geo=_jsonable(parsed.geo) if parsed.geo else None,
            source=parsed.source.value,
            parse_latency_ms=_ms(parsed.parse_latency_ms),
            n_hard_constraints=parsed.n_hard_constraints,
        )


class StageTimingView(BaseModel):
    stage: str
    ms: float


class TimingsView(BaseModel):
    """Per-stage wall-clock, on every response.

    A list rather than a dict because the order is the cascade order, and the
    demo draws it as a bar per stage. `total_ms` is the pipeline's own
    end-to-end measurement; it is deliberately *not* the sum of the stages,
    which would omit the glue between them.
    """

    total_ms: float
    stages: list[StageTimingView] = Field(default_factory=list)


class SearchResponseModel(BaseModel):
    """Body of a successful ``POST /search``."""

    query: str
    parsed: ParsedQueryView
    results: list[ResultView] = Field(default_factory=list)
    timings: TimingsView
    #: Candidates surviving each stage, keyed by stage name, plus `final`.
    #: Also carries the filter stage's drop reasons under a `dropped_` prefix
    #: and `penalized_missing_data` — the numbers that answer "why did this
    #: query return three results?" without a server-side debug session.
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    #: Non-fatal problems with this specific deployment, e.g. "no fine-tuned
    #: cross-encoder loaded". Repeated here, not only on /health, so a demo
    #: screenshot can never silently be of a degraded system.
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(
        cls,
        query: str,
        response: SearchResponse,
        *,
        warnings: Sequence[str] = (),
    ) -> SearchResponseModel:
        return cls(
            query=query,
            parsed=ParsedQueryView.from_domain(response.parsed),
            results=[ResultView.from_domain(r) for r in response.results],
            timings=TimingsView(
                total_ms=_ms(response.total_ms),
                stages=[StageTimingView(stage=t.stage, ms=_ms(t.ms)) for t in response.timings],
            ),
            candidate_counts=dict(response.n_candidates),
            warnings=list(warnings),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


class ArtifactStatus(BaseModel):
    """One loadable artifact and whether this process actually has it.

    This exists because the pipeline is designed to degrade rather than crash
    when a model is missing (its docstring, point 3). That is the right
    behaviour for development and the wrong behaviour to be *unaware of*: a
    deployment serving retrieval order while the README claims a fine-tuned
    reranker is the worst possible failure for this project, because it looks
    like success. /health names every artifact so the degradation is visible.
    """

    name: str
    loaded: bool
    #: Whether the *variant this process serves* actually needs it. A missing
    #: LambdaMART model matters when serving the full system and is irrelevant
    #: when serving the `hybrid_rrf` ablation row, so "missing" and "degraded"
    #: are two different questions and only this flag connects them.
    required: bool = True
    #: What is missing and what to run to fix it, or what was loaded.
    detail: str


class HealthResponse(BaseModel):
    #: ok        — every artifact loaded; this is the full system.
    #: degraded  — can answer searches, but at least one stage is missing.
    #: unavailable — no pipeline; /search will 503.
    status: Literal["ok", "degraded", "unavailable"]
    #: True when /search can answer. Readiness probes should key on this.
    ready: bool
    artifacts: list[ArtifactStatus] = Field(default_factory=list)
    #: Human-readable lines for the missing artifacts, in load order.
    warnings: list[str] = Field(default_factory=list)
    n_businesses: int | None = None
    n_documents: int | None = None
    #: Which ablation variant this process serves (default: the full system).
    variant: str | None = None
    #: Config hash, so a number in the README can be traced to a deployment.
    config_fingerprint: str | None = None
    #: Why the pipeline failed to load, when it did. Never a stack trace.
    error: str | None = None


class ErrorResponse(BaseModel):
    """Shape of every non-422 error body.

    `detail` is actionable text, not a class name: a 503 should tell the
    operator which `make` target builds the thing that is missing.
    """

    detail: str
