# Container image for the search API.
#
# ── Build (from the REPO ROOT, not from serve/) ──────────────────────────────
#
#   docker build -f serve/Dockerfile -t concierge-search:dev .
#
# The build context must be the repo root because the image needs `pyproject.toml`,
# `uv.lock`, `concierge/`, `serve/` and `configs/`.
#
# ── Run ──────────────────────────────────────────────────────────────────────
#
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/data/processed:/app/data/processed:ro" \
#     -v "$PWD/artifacts:/app/artifacts:ro" \
#     -v "$PWD/.cache/huggingface:/app/.cache/huggingface" \
#     -e CONCIERGE_CONFIG=configs/base.yaml \
#     -e CONCIERGE_CORS_ORIGINS=http://localhost:3000 \
#     concierge-search:dev
#
#   Then: curl localhost:8000/health
#
# Everything the cascade loads is MOUNTED, never baked in:
#
#   /app/data/processed   parsed businesses + business documents  (`make data`)
#   /app/artifacts        BM25 index, FAISS index, fine-tuned cross-encoder,
#                         LambdaMART model                        (`make index`, `make train`)
#
# Three reasons, in order of importance:
#
#   1. **Licensing.** The processed files are derived from the Yelp Open Dataset,
#      which is licensed for personal/educational use. Baking them into an image
#      that could be pushed to a registry would be redistribution. This is the
#      one non-negotiable reason.
#   2. They are large and regenerable. Every artifact is reproducible from a
#      `make` target, keyed by the config fingerprint, so storing them in an
#      image layer buys nothing and costs a rebuild per data change.
#   3. Mounting makes a degraded deployment *visible*: with nothing mounted the
#      service still starts and `GET /health` reports `unavailable` or
#      `degraded` with the missing artifact named, which is exactly the
#      behaviour the API was designed for.
#
# ── Notes ────────────────────────────────────────────────────────────────────
#
# * `.dockerignore`: Docker reads one from the *context root* (or, with BuildKit,
#   `<dockerfile>.dockerignore`). A file at `serve/.dockerignore` is not picked
#   up on its own, so to use the one shipped next to this Dockerfile:
#       cp serve/.dockerignore serve/Dockerfile.dockerignore   # BuildKit, or
#       cp serve/.dockerignore .dockerignore                   # context root
#   Without it the build context includes `data/` and `.venv/`, which is slow
#   and would send Yelp data to the daemon.
# * Image size is **not measured**. The runtime layer carries torch and faiss,
#   which are large; no figure is quoted here because none has been taken.
# * `OMP_NUM_THREADS` is deliberately left unset. It materially changes CPU
#   inference latency, so it is a knob of the latency experiment, not a value
#   to freeze in an image: set it explicitly with `-e` when measuring.
# * One uvicorn worker by default. Each additional worker loads its own copy of
#   every model, so `--workers N` multiplies memory; scale it only with headroom
#   and re-measure, since the load-test numbers are per-deployment.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — resolve and build the virtualenv
#
# Separate from the runtime stage so neither uv itself nor any build tool ends
# up in the shipped image, and so a dependency change does not invalidate the
# source-code layers.
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Only the lock inputs, so this layer is cached until dependencies actually change.
COPY pyproject.toml uv.lock ./

# `--frozen`: fail if the lock is out of date rather than silently resolving
#   something else — the served code must be the locked dependency set.
# `--no-dev`: pytest, ruff and mypy have no business in a serving image.
# `--no-install-project`: the app is run from the source tree copied below
#   (PYTHONPATH=/app), so there is nothing to build here and no reason to
#   invalidate this layer when application code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
#
# Plain python:3.12-slim, matching the builder's interpreter version: the venv
# copied below contains absolute paths and compiled extensions tied to 3.12.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    CONCIERGE_CONFIG=configs/base.yaml

# Non-root. Fixed uid/gid rather than whatever the system picks, so a mounted
# host directory's ownership is predictable across machines.
RUN groupadd --gid 10001 concierge \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin concierge

WORKDIR /app

# Dependencies first, then source: source changes rebuild only the small layers.
COPY --from=builder /app/.venv /app/.venv
COPY concierge/ ./concierge/
COPY serve/ ./serve/
COPY configs/ ./configs/

# Mount points, created so an unmounted run fails as a *reported* missing
# artifact rather than as a permission error on a path that does not exist.
# The HF cache and the query-parse cache are writable; the rest are read-only
# at run time by convention (`:ro` in the run command above).
RUN mkdir -p /app/data/processed /app/artifacts /app/.cache/huggingface /app/.llm_cache \
    && chown -R concierge:concierge /app/.cache /app/.llm_cache /app/artifacts

USER concierge

EXPOSE 8000

# `/health` answers 503 until the pipeline is loaded, which `urlopen` raises on —
# so this probe reports "unhealthy" exactly while the service cannot serve, and
# "healthy" for a degraded-but-serving deployment. Long start period: loading
# the embedding model and the cross-encoder takes time on CPU.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"]

# Exec form, no shell: uvicorn receives SIGTERM directly and shuts down cleanly.
# `--reload` is deliberately absent — that is a `make serve` concern.
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
