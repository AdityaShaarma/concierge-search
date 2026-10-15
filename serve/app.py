"""The HTTP service: one loaded pipeline, two endpoints, honest numbers.

Run it with::

    make serve                      # uvicorn serve.app:app --reload --port 8000

Four things this module is responsible for, each of which is a decision rather
than boilerplate.

**1. The pipeline is loaded exactly once, in a lifespan handler.**
`SearchPipeline.from_config` reads the BM25 index, the FAISS index, the
embedding model, two cross-encoder checkpoints and a LightGBM booster. Loading
those per request would make the p50/p95 numbers this project reports a
measurement of disk and `torch.load`, not of the cascade. A lifespan handler
(rather than the deprecated `@app.on_event("startup")`) also gives a clean
place to *record a load failure* instead of crashing the process, which is
what keeps `/health` able to explain why the service cannot answer.

**2. A degraded deployment must be visible, not silent.**
The pipeline deliberately degrades when an artifact is missing: no fine-tuned
checkpoint means the pretrained one, no LambdaMART means cross-encoder order.
That is right for development and dangerous in a demo, because a screenshot of
the degraded system looks exactly like a screenshot of the full one. So
`/health` names every artifact and whether this process has it, and every
`/search` response repeats the warnings. `required` is judged against the
variant being served: a missing LambdaMART model is a degradation of the full
system and irrelevant to the `hybrid_rrf` ablation row.

**3. The endpoints are `def`, not `async def`.**
A search is CPU-bound: a cross-encoder forward pass over ~200 documents holds
the GIL and never awaits. Declared `async def`, one such call would block the
event loop and every other in-flight request with it, and the load test's
concurrency levels would measure a queue rather than the service. Declared
`def`, Starlette runs the handler in its worker threadpool, so requests overlap
as far as the GIL and the model's own threads allow. The honest framing for an
interview: this service is thread-pool-parallel, not async-parallel, because
nothing in the cascade is I/O-bound except the optional Stage 1 LLM call.

**4. Error semantics follow the pipeline's contract, not HTTP habit.**
  * A malformed body is a 422 — pydantic's own validation, with `extra="forbid"`
    so a client typo is reported rather than silently ignored.
  * An unloadable index or model is a 503 whose detail names the `make` target
    that builds the missing thing.
  * **An LLM parse failure is not an error at all.** `concierge.query.parser`
    falls back to the rule parser by contract, so the request succeeds; the
    degradation is reported as `parsed.source == "rules"` rather than as a
    status code. Turning it into a 5xx would make the service less reliable
    than the pipeline it wraps.
  * Anything genuinely unexpected is a 500 with a short message. The traceback
    goes to the log, never into the response body.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from concierge.config import Config, load_config
from concierge.pipeline import (
    VARIANT_FULL,
    VARIANTS,
    RerankCheckpoint,
    SearchPipeline,
    Stage,
    Variant,
)
from serve.schemas import (
    ArtifactStatus,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
    SearchResponseModel,
)

log = logging.getLogger(__name__)

#: Origins allowed by CORS. The Next.js dev server, and both spellings of it:
#: `localhost` and `127.0.0.1` are *different origins* to a browser, and which
#: one you get depends on what you typed in the address bar.
#:
#: Not `allow_origins=["*"]`: a wildcard cannot be combined with credentials, it
#: hides who the client actually is, and this API has exactly one client. The
#: list is overridable by env for a deployed demo, where the origin is not
#: knowable at commit time.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

ENV_CONFIG = "CONCIERGE_CONFIG"
ENV_VARIANT = "CONCIERGE_VARIANT"
ENV_CORS_ORIGINS = "CONCIERGE_CORS_ORIGINS"

#: Factory signature. Injectable so the tests can run the real HTTP app against
#: a fake pipeline — no model downloads, no index, no network. Without this
#: seam, testing the API would mean testing the whole project.
PipelineFactory = Callable[[Config, Variant], Any]


def _default_factory(cfg: Config, variant: Variant) -> SearchPipeline:
    return SearchPipeline.from_config(cfg, variant=variant)


# ─────────────────────────────────────────────────────────────────────────────
# What this process knows about itself
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ServiceState:
    """Mutable startup result, held on `app.state`.

    A small object rather than module-level globals, because the test suite
    builds several apps in one process (loaded, degraded, unavailable) and
    module globals would leak between them.
    """

    pipeline: Any | None = None
    variant: Variant = VARIANT_FULL
    config: Config | None = None
    artifacts: tuple[ArtifactStatus, ...] = ()
    #: Set when startup failed. A short `Type: message`, never a traceback:
    #: it is returned over HTTP, and a traceback in a response body is both a
    #: leak and unreadable.
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    @property
    def warnings(self) -> tuple[str, ...]:
        """Human-readable lines for the artifacts this variant needs and lacks.

        Derived rather than stored so it cannot disagree with `artifacts`.
        """
        return tuple(a.detail for a in self.artifacts if a.required and not a.loaded)

    @property
    def status(self) -> Literal["ok", "degraded", "unavailable"]:
        if self.pipeline is None:
            return "unavailable"
        return "degraded" if self.warnings else "ok"

    def unavailable_detail(self) -> str:
        """The 503 body: what is wrong *and* what to run about it."""
        cause = self.error or "the pipeline did not load"
        return (
            f"search is unavailable: {cause}. "
            "Build the inputs it needs — `make data` then `make index` — and restart "
            "the service. `GET /health` lists which artifacts are missing."
        )


def _artifact(
    name: str,
    *,
    loaded: bool,
    required: bool,
    loaded_detail: str,
    missing_detail: str,
) -> ArtifactStatus:
    return ArtifactStatus(
        name=name,
        loaded=loaded,
        required=required,
        detail=loaded_detail if loaded else missing_detail,
    )


def inspect_artifacts(pipeline: Any, variant: Variant) -> tuple[ArtifactStatus, ...]:
    """Report what this loaded pipeline actually has, artifact by artifact.

    Reads the pipeline's attributes instead of re-checking the filesystem, and
    that distinction is the whole point: a file on disk that failed to
    deserialize would pass a path check and still not be serving anyone. The
    only trustworthy answer to "is the fine-tuned reranker loaded?" is whether
    the object is in memory.

    `getattr` with defaults, rather than attribute access, so the function is
    total: `/health` must still answer when handed something unusual (a test
    double, or a future pipeline that drops a stage), because a health endpoint
    that raises is worse than one that reports a gap.

    A free function rather than a method so it is unit-testable against a fake
    pipeline with no HTTP in the picture.
    """
    businesses: dict[str, Any] = getattr(pipeline, "businesses", {}) or {}
    documents: dict[str, Any] = getattr(pipeline, "doc_text", {}) or {}
    rerankers: dict[Any, Any] = getattr(pipeline, "rerankers", {}) or {}
    bm25_index = getattr(pipeline, "bm25_index", None)
    dense_index = getattr(pipeline, "dense_index", None)
    ltr_ranker = getattr(pipeline, "ltr_ranker", None)

    wants_rerank = variant.enabled(Stage.RERANK)
    # Which checkpoint this variant asks for. The other one is reported too (it
    # is informative) but is not required, so its absence is not a degradation.
    wanted_checkpoint = getattr(variant, "reranker", RerankCheckpoint.FINETUNED)

    return (
        _artifact(
            "businesses",
            loaded=bool(businesses),
            required=True,
            loaded_detail=f"{len(businesses)} businesses in scope",
            missing_detail="no businesses loaded; run `make data` for this config",
        ),
        _artifact(
            "documents",
            loaded=bool(documents),
            required=True,
            loaded_detail=f"{len(documents)} business documents",
            missing_detail="no business documents loaded; run `make data` for this config",
        ),
        _artifact(
            "bm25_index",
            loaded=bm25_index is not None,
            required=variant.enabled(Stage.BM25),
            loaded_detail="BM25 index loaded (lexical half of Stage 2)",
            missing_detail="no BM25 index; run `make index`",
        ),
        _artifact(
            "dense_index",
            loaded=dense_index is not None,
            required=variant.enabled(Stage.DENSE),
            loaded_detail="FAISS index loaded (dense half of Stage 2)",
            missing_detail=(
                "no FAISS index; run `make index`. Retrieval is BM25-only, so these "
                "are NOT hybrid-retrieval results"
            ),
        ),
        _artifact(
            "cross_encoder_pretrained",
            loaded=RerankCheckpoint.PRETRAINED in rerankers,
            required=wants_rerank and wanted_checkpoint is RerankCheckpoint.PRETRAINED,
            loaded_detail="pretrained cross-encoder loaded",
            missing_detail=(
                "pretrained cross-encoder not loaded (it downloads from the model hub on first use)"
            ),
        ),
        _artifact(
            "cross_encoder_finetuned",
            loaded=RerankCheckpoint.FINETUNED in rerankers,
            required=wants_rerank and wanted_checkpoint is RerankCheckpoint.FINETUNED,
            loaded_detail="fine-tuned cross-encoder loaded (Stage 3)",
            missing_detail=(
                "no fine-tuned cross-encoder; run `make train-reranker`. Stage 3 falls "
                "back to the pretrained model or is skipped, so these are NOT "
                "fine-tuned-reranker results"
            ),
        ),
        _artifact(
            "ltr_model",
            loaded=ltr_ranker is not None,
            required=variant.enabled(Stage.LTR),
            loaded_detail="LambdaMART model loaded (Stage 4)",
            missing_detail=(
                "no LambdaMART model; run `make train-ltr`. Results keep Stage 3 order, "
                "so they are NOT final-ranker results"
            ),
        ),
    )


def _resolve_variant(name: str | None) -> Variant:
    """Look up a named ablation variant, or fail loudly.

    Loudly on purpose: a typo'd variant name that silently fell back to the full
    system would mean a deployment serving something other than what the
    operator asked for — the same class of bug as a silently degraded model.
    """
    if not name:
        return VARIANT_FULL
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown variant {name!r}. Known variants: {', '.join(sorted(VARIANTS))}"
        ) from None


def _cors_origins(explicit: Sequence[str] | None) -> list[str]:
    if explicit is not None:
        return list(explicit)
    raw = os.getenv(ENV_CORS_ORIGINS, "")
    from_env = [part.strip() for part in raw.split(",") if part.strip()]
    return from_env or list(DEFAULT_CORS_ORIGINS)


# ─────────────────────────────────────────────────────────────────────────────
# The app
# ─────────────────────────────────────────────────────────────────────────────


def get_state(request: Request) -> ServiceState:
    """The per-process state, as a dependency.

    Through a dependency rather than a module global so handlers stay testable
    and so FastAPI documents it. The state object always exists — `create_app`
    makes it before the lifespan runs — which is why an app whose lifespan
    never ran answers 503 rather than raising an `AttributeError`.
    """
    state: ServiceState | None = getattr(request.app.state, "service", None)
    if state is None:  # pragma: no cover - create_app always sets it
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service state missing; the application was not started correctly",
        )
    return state


StateDep = Annotated[ServiceState, Depends(get_state)]


def create_app(
    *,
    cfg: Config | None = None,
    variant: Variant | None = None,
    pipeline_factory: PipelineFactory | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    """Build the application.

    A factory rather than a bare module-level `app` with globals, for one
    reason that matters: the tests exercise the real routing, validation and
    error handling against an injected fake pipeline. `create_app` is the seam
    that makes "the API works" testable separately from "the models work".

    Args:
        cfg: a loaded config, or None to load `$CONCIERGE_CONFIG` (default
            `configs/base.yaml`) during startup.
        variant: which ablation variant to serve, or None to read
            `$CONCIERGE_VARIANT` (default: the full system). Serving a variant
            is how the demo can show the ablation rows, not just the best one.
        pipeline_factory: `(cfg, variant) -> pipeline`. Defaults to the real
            loader.
        cors_origins: overrides `$CONCIERGE_CORS_ORIGINS` and the default
            Next.js dev origins.
    """
    factory = pipeline_factory or _default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state: ServiceState = app.state.service
        # One try around the whole of startup, including the config load: any
        # failure here must leave a service that can still *explain itself*
        # over HTTP. A process that exits on a missing index tells the operator
        # only what the container log happens to have kept.
        try:
            loaded_cfg = (
                cfg
                if cfg is not None
                else load_config(os.getenv(ENV_CONFIG) or "configs/base.yaml")
            )
            chosen = variant if variant is not None else _resolve_variant(os.getenv(ENV_VARIANT))
            pipeline = factory(loaded_cfg, chosen)
        except Exception as exc:
            log.exception("startup failed; /search will answer 503")
            state.error = f"{type(exc).__name__}: {exc}"
        else:
            state.config = loaded_cfg
            state.variant = chosen
            state.pipeline = pipeline
            state.artifacts = inspect_artifacts(pipeline, chosen)
            for line in state.warnings:
                log.warning("degraded: %s", line)
            log.info(
                "pipeline ready: variant=%s config=%s status=%s",
                chosen.name,
                loaded_cfg.fingerprint(),
                state.status,
            )
        yield
        # Nothing to tear down: the pipeline owns in-memory models and
        # memory-mapped indexes, both released when the process exits. Said
        # explicitly so the absence of cleanup reads as a decision.

    app = FastAPI(
        title="Concierge search",
        version="0.1.0",
        summary="Natural-language place search over the Yelp Open Dataset.",
        lifespan=lifespan,
    )
    app.state.service = ServiceState()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(cors_origins),
        allow_credentials=False,  # no cookies, no auth: nothing to send
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Readiness, plus which artifacts actually loaded.",
    )
    def health(state: StateDep, response: Response) -> HealthResponse:
        """Readiness and per-artifact load status.

        The status *code* is 503 when the service cannot answer searches and 200
        otherwise, including when degraded — a container probe keys on the code
        and must not restart a service that is happily serving a BM25-only
        configuration. The *body* carries the nuance a human needs.
        """
        pipeline = state.pipeline
        if not state.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status=state.status,
            ready=state.ready,
            artifacts=list(state.artifacts),
            warnings=list(state.warnings),
            n_businesses=len(getattr(pipeline, "businesses", {})) if pipeline else None,
            n_documents=len(getattr(pipeline, "doc_text", {})) if pipeline else None,
            variant=state.variant.name,
            config_fingerprint=state.config.fingerprint() if state.config else None,
            error=state.error,
        )

    @app.post(
        "/search",
        response_model=SearchResponseModel,
        responses={
            422: {"description": "Malformed request body."},
            500: {"model": ErrorResponse, "description": "Unexpected failure."},
            503: {"model": ErrorResponse, "description": "Pipeline not loaded."},
        },
        summary="Rank places for a natural-language request.",
    )
    def search(body: SearchRequest, state: StateDep) -> SearchResponseModel:
        """Run the cascade for one query.

        Synchronous `def` on purpose — see this module's docstring, point 3.

        Note what is *not* handled here: an empty result set is a 200 with zero
        results, not a 404. "Nothing is open at 3am with outdoor seating" is a
        correct answer to a well-formed question, and the `candidate_counts`
        field (with the filter stage's drop reasons) explains where the
        candidates went. A 404 would make the failure analysis read the logs
        instead of the response.
        """
        pipeline = state.pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=state.unavailable_detail(),
            )

        try:
            response = pipeline.search(body.query, variant=state.variant, k=body.limit)
        except Exception:
            # The pipeline promises not to raise on an unanswerable query, so
            # arriving here means a real bug. Log the traceback, return a short
            # message: a stack trace in the body helps an attacker and nobody else.
            log.exception("search failed for query %r", body.query)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="search failed unexpectedly; see the service log for the traceback",
            ) from None

        return SearchResponseModel.from_domain(body.query, response, warnings=state.warnings)

    return app


#: The ASGI application uvicorn imports (`uvicorn serve.app:app`). Built at
#: import time, but it loads nothing: the pipeline is built inside the lifespan,
#: so importing this module stays cheap and an import never fails over a
#: missing index.
app = create_app()
