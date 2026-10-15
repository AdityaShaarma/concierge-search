"""Tests for the served API, with the pipeline faked out.

What is under test here is the *HTTP contract*: status codes, the response
shape the Next.js client reads, and whether a degraded deployment is visible in
the payload. The cascade itself is tested in `tests/test_pipeline.py`.

That split is deliberate, and it is why `serve.app.create_app` takes a
`pipeline_factory`. A test that loaded the real pipeline would need the Yelp
data, a FAISS index and two trained models, so it would be skipped in CI and the
API would effectively be untested — which is how an API ends up returning
`unmatched` constraints the UI silently ignores. Here the whole file runs in
milliseconds with no models, no indexes and no network.

The fake is duck-typed against the attributes `/health` inspects
(`businesses`, `doc_text`, `bm25_index`, `dense_index`, `rerankers`,
`ltr_ranker`) and the one method `/search` calls. If the real pipeline renames
one of those, `test_health_reads_attributes_the_real_pipeline_actually_has`
fails — that test asserts the fake and the real class agree, so the fakes
cannot quietly drift away from the thing they stand in for.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from concierge.pipeline import (
    VARIANT_FULL,
    VARIANTS,
    RerankCheckpoint,
    SearchPipeline,
    Stage,
    Variant,
)
from concierge.types import (
    SCORE_CROSS_ENCODER,
    SCORE_LTR,
    SCORE_RRF,
    Business,
    ConstraintMatch,
    ConstraintStatus,
    OpenAt,
    ParsedQuery,
    ParseSource,
    SearchResponse,
    SearchResult,
    StageTiming,
)
from serve import load_test
from serve.app import create_app, inspect_artifacts
from serve.schemas import MAX_LIMIT

# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS = Business(
    business_id="b001",
    name="Quiet Bean",
    city="Metroville",
    state="XX",
    latitude=40.0,
    longitude=-80.0,
    stars=4.5,
    review_count=312,
    categories=("Coffee & Tea", "Cafes"),
    attributes={"WiFi": "free", "NoiseLevel": "quiet", "RestaurantsPriceRange2": 2},
    hours={},
    is_open=True,
)

# One constraint of each status, because the three-valued distinction is the
# whole point of the "why this result" payload: the UI must be able to tell
# "this café is loud" from "nobody recorded how loud this café is".
CONSTRAINTS = (
    ConstraintMatch(
        name="WiFi", requested="free", actual="free", status=ConstraintStatus.SATISFIED
    ),
    ConstraintMatch(name="price_max", requested=1, actual=2, status=ConstraintStatus.VIOLATED),
    # A dataclass as `requested`, to exercise the JSON coercion in schemas.py:
    # `OpenAt` is not JSON-serializable on its own.
    ConstraintMatch(
        name="open_at",
        requested=OpenAt(day="Friday", minute_of_day=22 * 60),
        actual=None,
        status=ConstraintStatus.UNKNOWN,
    ),
)

PARSED = ParsedQuery(
    raw="quiet cafe with wifi open past 10pm",
    semantic_text="quiet cafe",
    open_at=OpenAt(day="Friday", minute_of_day=22 * 60),
    attributes={"WiFi": "free", "NoiseLevel": "quiet"},
    price_max=2,
    categories=("Cafes",),
    source=ParseSource.RULES,
    parse_latency_ms=0.123456789,
)

TIMINGS = (
    StageTiming(stage="parse", ms=0.4),
    StageTiming(stage="retrieve", ms=11.25),
    StageTiming(stage="filter", ms=1.5),
    StageTiming(stage="rerank", ms=180.0),
    StageTiming(stage="ltr", ms=3.0),
    StageTiming(stage="assemble", ms=0.2),
)

COUNTS = {
    "retrieve": 200,
    "filter": 120,
    "dropped_hours": 60,
    "penalized_missing_data": 20,
    "rerank": 50,
    "ltr": 50,
    "final": 1,
}


@dataclass
class FakePipeline:
    """Stands in for `SearchPipeline`: the attributes /health reads, plus search.

    Defaults describe a fully loaded deployment; individual tests knock out
    pieces to produce the degraded cases.
    """

    businesses: dict[str, Business] = field(default_factory=lambda: {"b001": BUSINESS})
    doc_text: dict[str, str] = field(default_factory=lambda: {"b001": "Quiet Bean. Cafes."})
    bm25_index: Any = "bm25-index"
    dense_index: Any = "faiss-index"
    rerankers: dict[RerankCheckpoint, Any] = field(
        default_factory=lambda: {
            RerankCheckpoint.PRETRAINED: "pretrained",
            RerankCheckpoint.FINETUNED: "finetuned",
        }
    )
    ltr_ranker: Any = "lambdamart"
    #: Every `search()` call, so the test can assert `limit` was forwarded.
    calls: list[tuple[str, Variant | None, int | None]] = field(default_factory=list)
    raises: BaseException | None = None
    n_results: int = 1

    def search(
        self, query: str, *, variant: Variant | None = None, k: int | None = None
    ) -> SearchResponse:
        self.calls.append((query, variant, k))
        if self.raises is not None:
            raise self.raises
        results = tuple(
            SearchResult(
                business=BUSINESS,
                rank=i + 1,
                score=0.9 - i * 0.01,
                constraints=CONSTRAINTS,
                stage_scores={
                    SCORE_RRF: 0.032,
                    SCORE_CROSS_ENCODER: 7.5,
                    SCORE_LTR: 0.9 - i * 0.01,
                },
            )
            for i in range(self.n_results)
        )
        return SearchResponse(
            parsed=PARSED,
            results=results,
            timings=TIMINGS,
            total_ms=196.35,
            n_candidates=dict(COUNTS),
        )


@contextmanager
def make_client(
    config: Any,
    *,
    pipeline: FakePipeline | None = None,
    variant: Variant = VARIANT_FULL,
    factory_error: BaseException | None = None,
) -> Iterator[TestClient]:
    """A `TestClient` whose lifespan has run, so startup is exercised too.

    `factory_error` simulates the startup failure that matters most: an index or
    model that will not load. That path is why the service records the error
    instead of exiting.
    """

    def factory(cfg: Any, chosen: Variant) -> FakePipeline:
        if factory_error is not None:
            raise factory_error
        return pipeline if pipeline is not None else FakePipeline()

    app = create_app(cfg=config, variant=variant, pipeline_factory=factory)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(config: Any) -> Iterator[TestClient]:
    with make_client(config) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────


def test_health_ok_when_every_artifact_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["warnings"] == []
    assert body["error"] is None
    assert body["variant"] == VARIANT_FULL.name
    # The fingerprint is what ties a served result back to a config; its value
    # is the config's business, so this only asserts it is reported.
    assert body["config_fingerprint"]
    assert body["n_businesses"] == 1
    assert body["n_documents"] == 1
    assert all(a["loaded"] for a in body["artifacts"])


def test_health_degraded_names_the_missing_artifacts_and_their_make_targets(
    config: Any,
) -> None:
    """The headline requirement: a missing model must be visible, not silent."""
    degraded = FakePipeline(
        dense_index=None,
        rerankers={RerankCheckpoint.PRETRAINED: "pretrained"},
        ltr_ranker=None,
    )
    with make_client(config, pipeline=degraded) as client:
        body = client.get("/health").json()

    # Still serving, so 200 and ready — a probe must not restart this.
    assert body["status"] == "degraded"
    assert body["ready"] is True

    by_name = {a["name"]: a for a in body["artifacts"]}
    assert by_name["dense_index"]["loaded"] is False
    assert by_name["cross_encoder_finetuned"]["loaded"] is False
    assert by_name["ltr_model"]["loaded"] is False
    assert by_name["bm25_index"]["loaded"] is True

    joined = " ".join(body["warnings"])
    assert "make index" in joined
    assert "make train-reranker" in joined
    assert "make train-ltr" in joined
    # One warning per missing *required* artifact, and nothing else.
    assert len(body["warnings"]) == 3


def test_health_degraded_returns_200_but_unavailable_returns_503(config: Any) -> None:
    """Readiness probes key on the status code, so the codes must differ."""
    with make_client(config, factory_error=FileNotFoundError("no FAISS index")) as client:
        response = client.get("/health")
        search = client.post("/search", json={"query": "ramen"})

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["ready"] is False
    # The cause survives startup rather than being lost to the log.
    assert "FileNotFoundError" in (body["error"] or "")
    assert "no FAISS index" in (body["error"] or "")
    # ... and is not a traceback.
    assert "Traceback" not in (body["error"] or "")

    assert search.status_code == 503
    detail = search.json()["detail"]
    assert "make data" in detail and "make index" in detail
    assert "/health" in detail


def test_missing_artifact_is_not_required_when_the_variant_skips_that_stage() -> None:
    """A missing LambdaMART model is a degradation of the full system only.

    Serving the `rerank_finetuned` ablation row, Stage 4 is off by design, so
    reporting it as degraded would cry wolf on every ablation deployment.
    """
    variant = VARIANTS["rerank_finetuned"]
    assert not variant.enabled(Stage.LTR)

    artifacts = {a.name: a for a in inspect_artifacts(FakePipeline(ltr_ranker=None), variant)}
    assert artifacts["ltr_model"].loaded is False
    assert artifacts["ltr_model"].required is False

    # And the same pipeline served as the full system *is* degraded.
    full = {a.name: a for a in inspect_artifacts(FakePipeline(ltr_ranker=None), VARIANT_FULL)}
    assert full["ltr_model"].required is True


def test_health_distinguishes_the_two_cross_encoder_checkpoints() -> None:
    """Falling back to the pretrained model must not read as "reranker loaded".

    This is the degradation that most looks like success: the system still
    reranks, so results are plausible, but they are not the fine-tuned system
    the ablation table claims.
    """
    pipeline = FakePipeline(rerankers={RerankCheckpoint.PRETRAINED: "pretrained"})
    artifacts = {a.name: a for a in inspect_artifacts(pipeline, VARIANT_FULL)}

    assert artifacts["cross_encoder_pretrained"].loaded is True
    assert artifacts["cross_encoder_finetuned"].loaded is False
    assert artifacts["cross_encoder_finetuned"].required is True
    assert "NOT" in artifacts["cross_encoder_finetuned"].detail


def test_health_reads_attributes_the_real_pipeline_actually_has(config: Any) -> None:
    """Guards the fake against drift.

    `inspect_artifacts` reads the pipeline by attribute name, so if
    `SearchPipeline` renamed `rerankers`, /health would report a perfectly
    healthy service with every model missing — and every other test in this file
    would still pass, because they all use the fake. So build a *real* pipeline
    (cheap: no indexes, no models) and assert both objects carry the names.
    """
    real = SearchPipeline(config, businesses={}, documents={})
    fake = FakePipeline()
    for name in ("businesses", "doc_text", "bm25_index", "dense_index", "rerankers", "ltr_ranker"):
        assert hasattr(real, name), f"SearchPipeline no longer exposes .{name}"
        assert hasattr(fake, name), f"FakePipeline is missing .{name}"

    # And an empty real pipeline reports honestly rather than claiming health.
    statuses = {a.name: a for a in inspect_artifacts(real, VARIANT_FULL)}
    assert statuses["businesses"].loaded is False
    assert statuses["ltr_model"].loaded is False


def test_inspect_artifacts_is_total_for_an_unknown_object() -> None:
    """A health endpoint that raises is worse than one that reports a gap."""
    artifacts = inspect_artifacts(object(), VARIANT_FULL)
    assert artifacts and not any(a.loaded for a in artifacts)


# ─────────────────────────────────────────────────────────────────────────────
# /search — the happy path and the shape the UI depends on
# ─────────────────────────────────────────────────────────────────────────────


def test_search_returns_the_documented_shape(client: TestClient) -> None:
    response = client.post(
        "/search", json={"query": "quiet cafe with wifi open past 10pm", "limit": 5}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "quiet cafe with wifi open past 10pm"
    assert len(body["results"]) == 1

    result = body["results"][0]
    assert result["rank"] == 1
    assert result["business"]["business_id"] == "b001"
    assert result["business"]["name"] == "Quiet Bean"
    assert result["business"]["categories"] == ["Coffee & Tea", "Cafes"]
    assert result["business"]["stars"] == pytest.approx(4.5)
    # Upstream stage scores survive to the client, so the demo can show how a
    # result moved through the cascade.
    assert set(result["stage_scores"]) == {"rrf", "cross_encoder", "ltr"}
    # The ordering score is the last stage that ran.
    assert result["score"] == pytest.approx(result["stage_scores"]["ltr"])


def test_search_explains_each_result_in_three_buckets(client: TestClient) -> None:
    """The "why this result" payload, pre-split so the UI cannot conflate
    "violated" with "no data"."""
    body = client.post("/search", json={"query": "quiet cafe"}).json()
    constraints = body["results"][0]["constraints"]

    assert [c["name"] for c in constraints["matched"]] == ["WiFi"]
    assert [c["name"] for c in constraints["unmatched"]] == ["price_max"]
    assert [c["name"] for c in constraints["unknown"]] == ["open_at"]

    assert constraints["matched"][0]["explanation"] == "WiFi: free"
    assert constraints["unknown"][0]["explanation"] == "open_at: no data"
    # A dataclass `requested` arrived as JSON rather than breaking the response.
    assert constraints["unknown"][0]["requested"] == {"day": "Friday", "minute_of_day": 1320}


def test_search_reports_stage_timings_and_candidate_counts(client: TestClient) -> None:
    """Latency is a reported project result, so it is on every response."""
    body = client.post("/search", json={"query": "quiet cafe"}).json()

    timings = body["timings"]
    assert timings["total_ms"] == pytest.approx(196.35)
    assert [s["stage"] for s in timings["stages"]] == [
        "parse",
        "retrieve",
        "filter",
        "rerank",
        "ltr",
        "assemble",
    ]
    assert timings["stages"][3]["ms"] == pytest.approx(180.0)

    counts = body["candidate_counts"]
    assert counts["retrieve"] == 200
    assert counts["final"] == 1
    # The filter stage's drop reasons come through, which is what answers
    # "why did this query return one result?" without a server-side session.
    assert counts["dropped_hours"] == 60
    assert counts["penalized_missing_data"] == 20


def test_search_reports_how_the_query_was_parsed(client: TestClient) -> None:
    body = client.post("/search", json={"query": "quiet cafe"}).json()
    parsed = body["parsed"]

    assert parsed["semantic_text"] == "quiet cafe"
    assert parsed["open_at"] == {"day": "Friday", "minute_of_day": 1320}
    assert parsed["attributes"] == {"WiFi": "free", "NoiseLevel": "quiet"}
    assert parsed["price_max"] == 2
    assert parsed["categories"] == ["Cafes"]
    assert parsed["n_hard_constraints"] == 4


def test_llm_parse_failure_is_not_an_http_error(client: TestClient) -> None:
    """Per the parser's contract, a failed LLM call degrades to rules.

    So the request succeeds — and the degradation is *visible* as
    `parsed.source == "rules"` rather than hidden or turned into a 5xx. The fake
    returns exactly what the parser produces after a fallback.
    """
    response = client.post("/search", json={"query": "quiet cafe"})
    assert response.status_code == 200
    assert response.json()["parsed"]["source"] == "rules"


def test_search_repeats_deployment_warnings_on_every_response(config: Any) -> None:
    """So a demo screenshot can never silently be of a degraded system."""
    degraded = FakePipeline(ltr_ranker=None)
    with make_client(config, pipeline=degraded) as client:
        body = client.post("/search", json={"query": "quiet cafe"}).json()

    assert any("make train-ltr" in w for w in body["warnings"])


def test_search_forwards_the_limit_and_the_served_variant(config: Any) -> None:
    pipeline = FakePipeline()
    variant = VARIANTS["hybrid_rrf"]
    with make_client(config, pipeline=pipeline, variant=variant) as client:
        client.post("/search", json={"query": "ramen", "limit": 3})

    assert pipeline.calls == [("ramen", variant, 3)]


def test_search_defaults_the_limit(client: TestClient) -> None:
    response = client.post("/search", json={"query": "ramen"})
    assert response.status_code == 200


def test_empty_result_set_is_a_200_not_a_404(config: Any) -> None:
    """ "Nothing is open at 3am with outdoor seating" is a correct answer."""
    with make_client(config, pipeline=FakePipeline(n_results=0)) as client:
        response = client.post("/search", json={"query": "open at 3am with a patio"})

    assert response.status_code == 200
    assert response.json()["results"] == []


# ─────────────────────────────────────────────────────────────────────────────
# /search — error handling
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({}, "query is required"),
        ({"query": ""}, "an empty query is a client bug, not a search for everything"),
        ({"q": "ramen"}, "a misspelled field must be reported, not ignored"),
        ({"query": "ramen", "limit": 0}, "limit below the allowed range"),
        ({"query": "ramen", "limit": MAX_LIMIT + 1}, "more results than the cascade can produce"),
        ({"query": "ramen", "limit": "ten"}, "limit must be an integer"),
    ],
)
def test_malformed_body_is_422(client: TestClient, body: dict[str, Any], why: str) -> None:
    response = client.post("/search", json=body)
    assert response.status_code == 422, why


def test_non_json_body_is_422(client: TestClient) -> None:
    response = client.post(
        "/search", content=b"not json at all", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_unexpected_pipeline_failure_is_a_500_without_a_traceback(config: Any) -> None:
    """The pipeline promises not to raise, so this path means a real bug.

    It must still not leak a stack trace into the response body.
    """
    broken = FakePipeline(raises=RuntimeError("tokenizer exploded"))
    with make_client(config, pipeline=broken) as client:
        response = client.post("/search", json={"query": "ramen"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "Traceback" not in detail
    assert "tokenizer exploded" not in detail
    assert "log" in detail


def test_search_before_startup_is_503_not_an_attribute_error(config: Any) -> None:
    """A TestClient used without its context manager never runs the lifespan.

    The same situation as an ASGI server that failed to start the app. The
    service must answer, not raise.
    """
    app = create_app(cfg=config, pipeline_factory=lambda cfg, variant: FakePipeline())
    client = TestClient(app)  # no `with`: lifespan has not run
    response = client.post("/search", json={"query": "ramen"})
    assert response.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://127.0.0.1:3000"])
def test_cors_preflight_allows_the_next_dev_server(client: TestClient, origin: str) -> None:
    """Both spellings: a browser treats localhost and 127.0.0.1 as different
    origins, and which one the owner gets depends on what they typed."""
    response = client.options(
        "/search",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_rejects_an_unlisted_origin(config: Any) -> None:
    with make_client(config) as client:
        response = client.options(
            "/search",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert "access-control-allow-origin" not in response.headers


# ─────────────────────────────────────────────────────────────────────────────
# load_test helpers — the measurement arithmetic, with no server involved
# ─────────────────────────────────────────────────────────────────────────────


def test_percentile_is_nearest_rank() -> None:
    """A reported p95 must be a latency some request actually had."""
    samples = list(range(1, 101))  # 1..100
    assert load_test.percentile(samples, 0.50) == 50
    assert load_test.percentile(samples, 0.95) == 95
    assert load_test.percentile(samples, 0.99) == 99
    assert load_test.percentile(samples, 1.0) == 100
    # Single sample: every percentile is that sample.
    assert load_test.percentile([7.5], 0.95) == 7.5
    # Unsorted input is handled, and the result is an observed element.
    assert load_test.percentile([9.0, 1.0, 5.0], 0.5) == 5.0


def test_percentile_refuses_to_invent_a_number() -> None:
    with pytest.raises(ValueError):
        load_test.percentile([], 0.95)
    with pytest.raises(ValueError):
        load_test.percentile([1.0], 0.0)
    with pytest.raises(ValueError):
        load_test.percentile([1.0], 1.5)


def test_summarize_reports_n_and_nothing_else_when_empty() -> None:
    assert load_test.summarize([]) == {"n": 0}
    summary = load_test.summarize([10.0, 20.0, 30.0])
    assert summary["n"] == 3
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 30.0
    assert summary["mean_ms"] == pytest.approx(20.0)


def test_throughput_counts_only_successful_requests() -> None:
    """Otherwise a service that 500s instantly looks fast."""
    level = load_test.LevelResult(
        concurrency=4,
        requested=10,
        completed=5,
        errors=5,
        elapsed_s=2.0,
        client=load_test.summarize([1.0, 2.0, 3.0, 4.0, 5.0]),
        server={"n": 0},
        stages={},
        status_counts={"200": 5, "500": 5},
        error_examples=["HTTP 500: boom"],
        results_returned={"n": 5, "mean": 10.0, "min": 10.0},
    )
    assert level.throughput_rps == pytest.approx(2.5)
    payload = level.to_json()
    assert payload["errors"] == 5
    # Small sample sizes are flagged rather than dressed up as a p99.
    assert "p99" in payload["caveat"]


def test_load_queries_reads_csv_by_column_name(tmp_path: Any) -> None:
    path = tmp_path / "q.csv"
    path.write_text("query_id,query,query_type\nq1,cheap ramen,specific\nq2,quiet cafe,vibe\n")
    assert load_test.load_queries(path) == ["cheap ramen", "quiet cafe"]


def test_load_queries_rejects_a_csv_without_a_query_column(tmp_path: Any) -> None:
    path = tmp_path / "q.csv"
    path.write_text("id,text\n1,cheap ramen\n")
    with pytest.raises(ValueError, match="query"):
        load_test.load_queries(path)


def test_load_queries_reads_one_per_line(tmp_path: Any) -> None:
    path = tmp_path / "q.txt"
    path.write_text("cheap ramen\n\n  quiet cafe  \n")
    assert load_test.load_queries(path) == ["cheap ramen", "quiet cafe"]


def test_load_queries_fails_loudly_on_an_empty_file(tmp_path: Any) -> None:
    path = tmp_path / "q.txt"
    path.write_text("\n\n")
    with pytest.raises(ValueError):
        load_test.load_queries(path)


def test_write_record_writes_a_timestamped_file_and_latest(tmp_path: Any) -> None:
    record = {"timestamp": "2026-01-02T03:04:05+00:00", "levels": []}
    path = load_test.write_record(record, tmp_path)

    assert path.parent == tmp_path
    assert json.loads(path.read_text())["timestamp"] == record["timestamp"]
    # A stable path for report generation, alongside the run history.
    assert json.loads((tmp_path / "latest.json").read_text()) == record
