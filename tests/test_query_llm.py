"""Tests for Stage 1's LLM path: cache, provider clients, retry policy and fallback.

Every test here runs with **no network and no API key**. The HTTP layer is replaced with
``httpx.MockTransport`` and the environment is made hermetic, so the owner's real ``.env``
can never be read during a test run — a test that quietly used a real key would spend
quota and would then pass or fail depending on a rate limit rather than on the code.

The only credential-shaped string in this file is an obvious dummy. The one live test is
marked ``needs_api`` and skips unless a key is actually present.

The six behaviours the task hinges on, and why each matters:

* a cache hit does not call the API         — the cost and latency claim
* a timeout falls back to rules             — a rate limit must not 500 a search
* a 400 is NOT retried                      — retrying a malformed request burns quota
* a 429 IS retried                          — the one failure retrying actually fixes
* malformed JSON falls back to rules        — never raise into the caller
* prompt_version changes the cache key      — a stale cache would corrupt an experiment

Validation itself (the attribute vocabulary, day and time normalization) belongs to
``concierge.query.schema`` and is tested with that module; here we only check that this
module routes a rejection to the rule fallback instead of raising.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from concierge.query import llm, parser, schema
from concierge.query.cache import CacheKey, ParseCache, normalize_query
from concierge.types import OpenAt, ParsedQuery, ParseSource

# Obviously not a credential. Present only so build_providers() finds a non-empty value.
DUMMY_KEY = "dummy-key-for-tests-not-a-credential"

GEMINI_HOST = "generativelanguage.googleapis.com"
GROQ_HOST = "api.groq.com"

# A reply that the schema module accepts: every key is in its top-level set, every
# attribute is in its vocabulary, and open_at carries both a day and a time.
VALID_PAYLOAD: dict[str, Any] = {
    "semantic_text": "quiet place to work",
    "open_at": {"day": "Friday", "time": "22:00"},
    "attributes": {"WiFi": "free", "NoiseLevel": "quiet"},
    "price_max": 2,
    "categories": ["Coffee & Tea"],
    "geo": None,
}

THURSDAY = datetime(2026, 10, 15)  # fixed clock: the suite must not depend on today


# ─────────────────────────────────────────────────────────────────────────────────────
# Fixtures and fakes
# ─────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def hermetic_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from the real environment, keys and backoff sleeps.

    ``_ENV_LOADED`` is forced True so python-dotenv never reads the owner's ``.env``;
    without it, a test that deletes ``GROQ_API_KEY`` could be undone by the file. Backoff
    is zeroed so the retry tests take microseconds rather than seconds.
    """
    if "needs_api" in request.keywords:
        return  # the live test deliberately uses the real environment
    monkeypatch.setattr(llm, "_ENV_LOADED", True)
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_RETRY_WAIT_MULTIPLIER_S", 0.0)
    monkeypatch.setattr(llm, "_RETRY_WAIT_MAX_S", 0.0)
    monkeypatch.setattr(llm, "_RETRY_JITTER_S", 0.0)


def _with_query_parsing(config, **updates: Any):
    return config.model_copy(
        update={"query_parsing": config.query_parsing.model_copy(update=updates)}
    )


@pytest.fixture
def cfg(config, tmp_path: Path):
    """Config with the parse cache pointed at a temp file, never the repo's."""
    return _with_query_parsing(config, cache_path=tmp_path / "parse.sqlite")


@pytest.fixture
def cache(tmp_path: Path):
    with ParseCache(tmp_path / "cache.sqlite") as c:
        yield c


class Recorder:
    """A fake HTTP backend that records requests and replays scripted responses."""

    def __init__(self, *responses: Any) -> None:
        # Each item is either an httpx.Response or an exception instance to raise.
        self._script = list(responses)
        self.requests: list[httpx.Request] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # The last scripted item repeats, so "always 429" needs one entry, not five.
        item = self._script[min(len(self.requests) - 1, len(self._script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def gemini_ok(payload: dict[str, Any] | str) -> httpx.Response:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def groq_ok(payload: dict[str, Any] | str) -> httpx.Response:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def forbid_http(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
    raise AssertionError(f"no HTTP call expected, got {request.url}")


def run(query: str, cfg, cache: ParseCache, handler: Any, **kwargs: Any) -> ParsedQuery:
    """Parse one query against a stubbed transport, with a fixed clock by default."""
    client = handler.client() if isinstance(handler, Recorder) else handler
    kwargs.setdefault("now", THURSDAY)
    return parser.parse(query, cfg, http=client, cache=cache, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────────────
# Cache unit behaviour
# ─────────────────────────────────────────────────────────────────────────────────────


def test_cache_roundtrip_and_miss(cache: ParseCache) -> None:
    key = CacheKey("gemini", "gemini-2.0-flash", "v1+abc", "quiet cafe")
    assert cache.get(key) is None

    cache.put(key, VALID_PAYLOAD, llm_latency_ms=123.4, answered_by="gemini:gemini-2.0-flash")
    entry = cache.get(key)

    assert entry is not None
    assert entry.payload == VALID_PAYLOAD
    assert entry.llm_latency_ms == pytest.approx(123.4)
    assert entry.answered_by == "gemini:gemini-2.0-flash"
    assert len(cache) == 1


def test_cache_key_distinguishes_provider_model_and_prompt_version() -> None:
    base = CacheKey("gemini", "gemini-2.0-flash", "v1+abc", "quiet cafe")
    assert base.digest() != CacheKey("groq", "gemini-2.0-flash", "v1+abc", "quiet cafe").digest()
    assert base.digest() != CacheKey("gemini", "other-model", "v1+abc", "quiet cafe").digest()
    assert base.digest() != CacheKey("gemini", "gemini-2.0-flash", "v2+xyz", "quiet cafe").digest()
    assert base.digest() != CacheKey("gemini", "gemini-2.0-flash", "v1+abc", "loud cafe").digest()
    # Case and surrounding whitespace fold: the same request, not two API calls.
    assert (
        base.digest() == CacheKey("gemini", "gemini-2.0-flash", "v1+abc", " Quiet  Cafe ").digest()
    )


def test_prompt_version_change_misses_the_cache(cache: ParseCache) -> None:
    """The decision that protects experiments: an edited prompt cannot read old entries."""
    old = CacheKey("gemini", "gemini-2.0-flash", "v1+aaaaaaaaaaaa", "quiet cafe")
    new = CacheKey("gemini", "gemini-2.0-flash", "v2+bbbbbbbbbbbb", "quiet cafe")

    cache.put(old, VALID_PAYLOAD)

    assert cache.get(old) is not None
    assert cache.get(new) is None, "a new prompt version must not read old-prompt parses"


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite"
    key = CacheKey("gemini", "gemini-2.0-flash", "v1+abc", "quiet cafe")
    with ParseCache(path) as first:
        first.put(key, VALID_PAYLOAD)
    with ParseCache(path) as second:
        entry = second.get(key)
    assert entry is not None and entry.payload == VALID_PAYLOAD


def test_cache_treats_a_corrupt_payload_as_a_miss(cache: ParseCache) -> None:
    key = CacheKey("gemini", "gemini-2.0-flash", "v1+abc", "quiet cafe")
    cache.put(key, VALID_PAYLOAD)
    # Simulate an interrupted write or a hand-edited database.
    with cache._lock:
        cache._conn.execute(
            "UPDATE parse_cache SET payload = ? WHERE key = ?", ("{not json", key.digest())
        )
        cache._conn.commit()

    assert cache.get(key) is None
    assert len(cache) == 0, "a corrupt entry is deleted, not re-read forever"


def test_normalize_query_folds_case_and_whitespace() -> None:
    assert normalize_query("  Open   Past 10PM ") == "open past 10pm"


# ─────────────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────────────


def test_llm_parse_fills_parsed_query_and_measures_latency(cfg, cache: ParseCache) -> None:
    rec = Recorder(gemini_ok(VALID_PAYLOAD))

    parsed = run("quiet cafe with wifi open friday past 10pm", cfg, cache, rec)

    assert isinstance(parsed, ParsedQuery)
    assert parsed.source is ParseSource.LLM
    assert parsed.raw == "quiet cafe with wifi open friday past 10pm"
    assert parsed.semantic_text == "quiet place to work"
    assert parsed.open_at == OpenAt(day="Friday", minute_of_day=22 * 60)
    assert parsed.attributes == {"WiFi": "free", "NoiseLevel": "quiet"}
    assert parsed.price_max == 2
    assert parsed.categories == ("Coffee & Tea",)
    assert parsed.n_hard_constraints == 4
    # Latency is measured, not compared against a target: it is a reported number, and the
    # test only checks that a real wall-clock measurement happened.
    assert parsed.parse_latency_ms > 0.0
    assert rec.calls == 1


def test_request_carries_the_schema_and_deterministic_decoding(cfg, cache: ParseCache) -> None:
    """The constrained-decoding contract: an invalid value should be impossible, not just
    unlikely, and the same query must parse the same way every time."""
    rec = Recorder(gemini_ok(VALID_PAYLOAD))

    run("quiet cafe", cfg, cache, rec)

    config = rec.bodies()[0]["generationConfig"]
    assert config["temperature"] == 0.0
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] == schema.query_json_schema("openapi")


def test_api_key_is_sent_as_a_header_never_in_the_url(cfg, cache: ParseCache) -> None:
    """URLs reach logs, traces and error bodies; headers do not."""
    rec = Recorder(gemini_ok(VALID_PAYLOAD))

    run("quiet cafe", cfg, cache, rec)

    request = rec.requests[0]
    assert DUMMY_KEY not in str(request.url)
    assert request.headers["x-goog-api-key"] == DUMMY_KEY


def test_markdown_fenced_json_is_accepted(cfg, cache: ParseCache) -> None:
    rec = Recorder(gemini_ok("```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"))

    assert run("quiet cafe", cfg, cache, rec).source is ParseSource.LLM


def test_groq_response_shape_is_parsed(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", DUMMY_KEY)
    rec = Recorder(groq_ok(VALID_PAYLOAD))
    groq_cfg = _with_query_parsing(cfg, provider="groq", model="llama-3.1-8b-instant")

    parsed = run("quiet cafe", groq_cfg, cache, rec)

    assert parsed.source is ParseSource.LLM
    assert rec.hosts() == [GROQ_HOST]
    assert rec.bodies()[0]["response_format"]["type"] == "json_schema"


def test_groq_downgrades_once_when_strict_schema_is_rejected(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict json_schema support varies by Groq model. A 400 there is not retried with the
    same request; it is downgraded once, because this provider exists for availability."""
    monkeypatch.setenv("GROQ_API_KEY", DUMMY_KEY)
    rec = Recorder(
        httpx.Response(400, text="response_format not supported"), groq_ok(VALID_PAYLOAD)
    )
    groq_cfg = _with_query_parsing(cfg, provider="groq", model="llama-3.1-8b-instant")

    parsed = run("quiet cafe", groq_cfg, cache, rec)

    assert parsed.source is ParseSource.LLM
    assert rec.calls == 2
    formats = [b["response_format"]["type"] for b in rec.bodies()]
    assert formats == ["json_schema", "json_object"], "exactly one downgrade, never a loop"


# ─────────────────────────────────────────────────────────────────────────────────────
# Caching through the public entry point
# ─────────────────────────────────────────────────────────────────────────────────────


def test_cache_hit_avoids_a_second_api_call(cfg, cache: ParseCache) -> None:
    rec = Recorder(gemini_ok(VALID_PAYLOAD))
    client = rec.client()

    first = parser.parse("quiet cafe", cfg, http=client, cache=cache, now=THURSDAY)
    second = parser.parse("quiet cafe", cfg, http=client, cache=cache, now=THURSDAY)

    assert rec.calls == 1, "an identical query must never hit the API twice"
    assert first.source is ParseSource.LLM
    assert second.source is ParseSource.CACHE
    assert second.semantic_text == first.semantic_text
    assert second.attributes == first.attributes
    assert second.price_max == first.price_max


def test_case_and_whitespace_variants_reuse_the_entry(cfg, cache: ParseCache) -> None:
    rec = Recorder(gemini_ok(VALID_PAYLOAD))
    client = rec.client()

    parser.parse("quiet cafe", cfg, http=client, cache=cache, now=THURSDAY)
    cached = parser.parse("QUIET   cafe", cfg, http=client, cache=cache, now=THURSDAY)

    assert cached.source is ParseSource.CACHE
    assert rec.calls == 1


def test_only_successful_parses_are_cached(cfg, cache: ParseCache) -> None:
    """An unusable reply must not be stored: a cached payload that always fails validation
    would poison that query for as long as the cache lives."""
    rec = Recorder(gemini_ok({"semantic_text": "x", "attributes": {"Telepathy": True}}))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert parsed.source is ParseSource.RULES
    assert len(cache) == 0


def test_prompt_version_change_forces_a_new_api_call(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder(gemini_ok(VALID_PAYLOAD))
    client = rec.client()

    parser.parse("quiet cafe", cfg, http=client, cache=cache, now=THURSDAY)
    assert rec.calls == 1

    monkeypatch.setattr(llm, "prompt_version", lambda: "v99+ffffffffffff")
    parsed = parser.parse("quiet cafe", cfg, http=client, cache=cache, now=THURSDAY)

    assert rec.calls == 2, "an edited prompt must not be served from the old cache"
    assert parsed.source is ParseSource.LLM
    assert len(cache) == 2, "both versions coexist; the old one is simply never read again"


def test_cached_entry_rejected_by_the_validator_is_dropped_and_refetched(
    cfg, cache: ParseCache
) -> None:
    key = CacheKey(
        provider=cfg.query_parsing.provider,
        model=cfg.query_parsing.model,
        prompt_version=llm.prompt_version(),
        query="quiet cafe",
    )
    cache.put(key, {"semantic_text": "x", "price_max": 99})  # 99 is not a Yelp tier
    rec = Recorder(gemini_ok(VALID_PAYLOAD))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert parsed.source is ParseSource.LLM
    assert rec.calls == 1


# ─────────────────────────────────────────────────────────────────────────────────────
# Retry policy
# ─────────────────────────────────────────────────────────────────────────────────────


def test_http_400_is_not_retried(cfg, cache: ParseCache) -> None:
    """A 400 means our request is malformed; retrying it only burns free-tier quota."""
    rec = Recorder(httpx.Response(400, json={"error": {"message": "bad request"}}))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert rec.calls == 1
    assert parsed.source is ParseSource.RULES
    assert len(cache) == 0, "a failed parse is never cached"


def test_http_429_is_retried_then_succeeds(cfg, cache: ParseCache) -> None:
    rec = Recorder(
        httpx.Response(429, json={"error": "rate limited"}),
        httpx.Response(429, json={"error": "rate limited"}),
        gemini_ok(VALID_PAYLOAD),
    )
    assert cfg.query_parsing.max_retries == 2

    parsed = run("quiet cafe", cfg, cache, rec)

    assert rec.calls == 3, "max_retries=2 means three attempts in total"
    assert parsed.source is ParseSource.LLM


def test_http_5xx_is_retried(cfg, cache: ParseCache) -> None:
    rec = Recorder(httpx.Response(503, text="upstream unavailable"), gemini_ok(VALID_PAYLOAD))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert rec.calls == 2
    assert parsed.source is ParseSource.LLM


def test_exhausted_429_falls_back_to_rules(cfg, cache: ParseCache) -> None:
    rec = Recorder(httpx.Response(429, json={"error": "rate limited"}))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert rec.calls == 3
    assert parsed.source is ParseSource.RULES


def test_retry_count_follows_config(cfg, cache: ParseCache) -> None:
    rec = Recorder(httpx.Response(429, json={"error": "rate limited"}))

    run("quiet cafe", _with_query_parsing(cfg, max_retries=0), cache, rec)

    assert rec.calls == 1, "max_retries=0 must mean exactly one attempt"


# ─────────────────────────────────────────────────────────────────────────────────────
# Degradation — none of these may raise into the caller
# ─────────────────────────────────────────────────────────────────────────────────────


def test_timeout_falls_back_to_rules(cfg, cache: ParseCache) -> None:
    rec = Recorder(httpx.ReadTimeout("simulated timeout"))

    parsed = run("quiet cafe open past 10pm", cfg, cache, rec)

    assert parsed.source is ParseSource.RULES
    assert parsed.raw == "quiet cafe open past 10pm"
    assert rec.calls == 3, "timeouts are retryable"


def test_connection_error_falls_back_to_rules(cfg, cache: ParseCache) -> None:
    rec = Recorder(httpx.ConnectError("simulated DNS failure"))

    assert run("quiet cafe", cfg, cache, rec).source is ParseSource.RULES


def test_malformed_json_falls_back_to_rules(cfg, cache: ParseCache) -> None:
    rec = Recorder(gemini_ok("this is not json at all"))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert parsed.source is ParseSource.RULES
    assert rec.calls == 1, "malformed output at temperature 0 must not be retried"
    assert len(cache) == 0


def test_schema_rejection_falls_back_to_rules(cfg, cache: ParseCache) -> None:
    """An out-of-vocabulary value would match no business at all, which looks exactly like
    a ranking failure. The rule parser gets a usable answer instead."""
    rec = Recorder(gemini_ok(dict(VALID_PAYLOAD, attributes={"NoiseLevel": "pretty quiet"})))

    assert run("quiet cafe", cfg, cache, rec).source is ParseSource.RULES


def test_out_of_range_price_falls_back_rather_than_being_clamped(cfg, cache: ParseCache) -> None:
    """Clamping would invent a hard filter the user never asked for, and an invented filter
    silently removes correct results."""
    rec = Recorder(gemini_ok(dict(VALID_PAYLOAD, price_max=7)))

    assert run("cheap cafe", cfg, cache, rec).source is ParseSource.RULES


def test_half_an_hours_constraint_falls_back(cfg, cache: ParseCache) -> None:
    """A day with no time cannot be checked against an hours interval; the schema refuses
    it rather than inventing 00:00."""
    rec = Recorder(gemini_ok(dict(VALID_PAYLOAD, open_at={"day": "Friday"})))

    assert run("something on friday", cfg, cache, rec).source is ParseSource.RULES


def test_blocked_response_with_no_candidates_falls_back(cfg, cache: ParseCache) -> None:
    rec = Recorder(httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))

    assert run("quiet cafe", cfg, cache, rec).source is ParseSource.RULES


def test_missing_api_key_falls_back_without_any_http_call(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = httpx.Client(transport=httpx.MockTransport(forbid_http))

    assert run("quiet cafe", cfg, cache, client).source is ParseSource.RULES


def test_provider_rules_skips_the_api_entirely(cfg, cache: ParseCache) -> None:
    """The rules-only ablation row must be measured with no network cost at all."""
    client = httpx.Client(transport=httpx.MockTransport(forbid_http))

    parsed = run(
        "quiet cafe open past 10pm", _with_query_parsing(cfg, provider="rules"), cache, client
    )

    assert parsed.source is ParseSource.RULES
    assert len(cache) == 0, "the rules path neither reads nor writes the LLM cache"


def test_empty_query_skips_the_api(cfg, cache: ParseCache) -> None:
    client = httpx.Client(transport=httpx.MockTransport(forbid_http))

    assert run("   ", cfg, cache, client).source is ParseSource.RULES


def test_fallback_disabled_still_never_raises(cfg, cache: ParseCache) -> None:
    """With rule fallback off (the LLM-only ablation row), a failure degrades to plain
    semantic text rather than propagating an exception into the search request."""
    rec = Recorder(httpx.Response(400, text="bad request"))

    parsed = run("quiet cafe", _with_query_parsing(cfg, fallback_to_rules=False), cache, rec)

    assert parsed.semantic_text == "quiet cafe"
    assert parsed.n_hard_constraints == 0
    assert parsed.source is ParseSource.RULES


def test_rule_parser_failure_degrades_to_passthrough(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(query: str, **kwargs: Any) -> ParsedQuery:
        raise RuntimeError("simulated rule parser bug")

    monkeypatch.setattr(parser.rules, "parse_rules", boom)
    rec = Recorder(httpx.Response(400, text="bad request"))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert parsed.semantic_text == "quiet cafe"
    assert parsed.source is ParseSource.RULES


def test_fallback_latency_includes_the_failed_api_attempt(cfg, cache: ParseCache) -> None:
    """Reporting only the regex time would make the fallback path look free."""
    rec = Recorder(httpx.Response(429, json={"error": "rate limited"}))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert parsed.parse_latency_ms > 0.0


# ─────────────────────────────────────────────────────────────────────────────────────
# Provider fallback chain
# ─────────────────────────────────────────────────────────────────────────────────────


def test_groq_answers_when_gemini_is_rate_limited(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", DUMMY_KEY)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == GEMINI_HOST:
            return httpx.Response(429, json={"error": "rate limited"})
        return groq_ok(VALID_PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    parsed = run("quiet cafe", cfg, cache, client)

    assert parsed.source is ParseSource.LLM, "a free-tier 429 must not cost the user results"
    assert seen == [GEMINI_HOST] * 3 + [GROQ_HOST], "Gemini is exhausted before Groq is tried"


def test_fallback_answer_is_cached_under_the_configured_provider(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyed on what was requested, annotated with who answered — otherwise a Groq reply
    could never be reused under the Gemini config it was fetched for."""
    monkeypatch.setenv("GROQ_API_KEY", DUMMY_KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == GEMINI_HOST:
            return httpx.Response(429, json={"error": "rate limited"})
        return groq_ok(VALID_PAYLOAD)

    run("quiet cafe", cfg, cache, httpx.Client(transport=httpx.MockTransport(handler)))

    entry = cache.get(
        CacheKey("gemini", cfg.query_parsing.model, llm.prompt_version(), "quiet cafe")
    )
    assert entry is not None
    assert entry.answered_by == f"groq:{llm.DEFAULT_GROQ_MODEL}"


def test_400_does_not_trigger_the_fallback_provider(
    cfg, cache: ParseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed request fails the same way everywhere; trying Groq would only add
    latency before the rule fallback."""
    monkeypatch.setenv("GROQ_API_KEY", DUMMY_KEY)
    rec = Recorder(httpx.Response(400, text="bad request"))

    parsed = run("quiet cafe", cfg, cache, rec)

    assert rec.hosts() == [GEMINI_HOST]
    assert parsed.source is ParseSource.RULES


def test_groq_model_default_when_the_config_names_a_gemini_model(cfg) -> None:
    assert llm._groq_model(cfg) == llm.DEFAULT_GROQ_MODEL
    assert llm._groq_model(_with_query_parsing(cfg, model="llama-3.1-8b-instant")) == (
        "llama-3.1-8b-instant"
    )


def test_rules_provider_builds_no_chain(cfg) -> None:
    with pytest.raises(llm.LlmConfigError):
        llm.build_providers(_with_query_parsing(cfg, provider="rules"))


# ─────────────────────────────────────────────────────────────────────────────────────
# Resolving the day of an hours constraint
# ─────────────────────────────────────────────────────────────────────────────────────


def test_a_null_day_is_resolved_from_the_clock(cfg, cache: ParseCache) -> None:
    """ "open past 10pm" names a time but no day. The prompt asks for a null day in that
    case precisely so the day is supplied per call rather than written into the answer."""
    rec = Recorder(gemini_ok(dict(VALID_PAYLOAD, open_at={"day": None, "time": "22:00"})))

    parsed = run("open past 10pm", cfg, cache, rec)

    assert parsed.open_at == OpenAt(day="Thursday", minute_of_day=1320)


def test_cache_hit_resolves_the_day_for_the_day_it_is_read(cfg, cache: ParseCache) -> None:
    """The reason the day must stay null in the stored payload: a weekday written into the
    cache would quietly filter on the wrong day's hours from the next midnight onwards."""
    rec = Recorder(gemini_ok(dict(VALID_PAYLOAD, open_at={"day": None, "time": "22:00"})))
    client = rec.client()

    first = parser.parse("open past 10pm", cfg, http=client, cache=cache, now=THURSDAY)
    later = parser.parse(
        "open past 10pm", cfg, http=client, cache=cache, now=datetime(2026, 10, 17)
    )

    assert rec.calls == 1
    assert first.open_at is not None and first.open_at.day == "Thursday"
    assert later.source is ParseSource.CACHE
    assert later.open_at is not None and later.open_at.day == "Saturday"


def test_a_weekday_the_model_names_is_respected_not_overwritten(cfg, cache: ParseCache) -> None:
    """A day in the reply is the user's own words ("friday morning"), so the clock must not
    override it — today is Thursday in these tests and the answer must still be Friday."""
    rec = Recorder(gemini_ok(dict(VALID_PAYLOAD, open_at={"day": "Fri", "time": "9:30"})))

    parsed = run("open friday morning", cfg, cache, rec)

    assert parsed.open_at == OpenAt(day="Friday", minute_of_day=9 * 60 + 30)


def test_queries_about_another_day_bypass_the_cache(cfg, cache: ParseCache) -> None:
    """A parse of "tomorrow" is only valid on the day it was made: the model folded the
    relative word into a concrete weekday, so there is nothing left to re-resolve."""
    rec = Recorder(gemini_ok(VALID_PAYLOAD))
    client = rec.client()

    parser.parse("open tomorrow night", cfg, http=client, cache=cache, now=THURSDAY)
    parser.parse("open tomorrow night", cfg, http=client, cache=cache, now=THURSDAY)

    assert rec.calls == 2
    assert len(cache) == 0


def test_today_is_sent_to_the_model(cfg, cache: ParseCache) -> None:
    """The schema requires a concrete weekday, so the model needs to know what today is."""
    rec = Recorder(gemini_ok(VALID_PAYLOAD))

    run("open late", cfg, cache, rec)

    prompt = rec.bodies()[0]["contents"][0]["parts"][0]["text"]
    assert "Today is Thursday." in prompt


# ─────────────────────────────────────────────────────────────────────────────────────
# Prompt: rendering, versioning and the leakage rule
# ─────────────────────────────────────────────────────────────────────────────────────


def test_prompt_header_comments_are_never_sent_to_the_model() -> None:
    rendered = llm.render_prompt("quiet cafe", today="Thursday")

    assert "LEAKAGE RULE" not in rendered
    assert "prompt_version:" not in rendered
    assert "quiet cafe" in rendered
    for placeholder in (llm.QUERY_PLACEHOLDER, llm.ATTRIBUTES_PLACEHOLDER, llm.TODAY_PLACEHOLDER):
        assert placeholder not in rendered


def test_prompt_states_the_vocabulary_from_the_schema_module() -> None:
    """One table generates the prompt, the request schema and the validator, so a value the
    prompt advertises can never be a value the validator refuses."""
    rendered = llm.render_prompt("quiet cafe", today="Thursday")

    assert "NoiseLevel" in rendered
    assert "very_loud" in rendered
    assert "Ambience" not in rendered, "nested attributes are deliberately unsupported"


def test_prompt_version_tracks_the_body_and_the_vocabulary() -> None:
    body = "Convert the request.\n\n{{ATTRIBUTES}}\n{{TODAY}}\n{{QUERY}}\n"
    header = "# prompt_version: v1\n\n"
    base = llm.prompt_version_for(header + body, "- WiFi (free | paid | no)")

    assert base == llm.prompt_version_for(
        header + "# an extra repo-facing note\n\n" + body, "- WiFi (free | paid | no)"
    ), "editing a comment must not throw away a warm cache"
    assert base != llm.prompt_version_for(header + body + "One more rule.\n", "- WiFi (free)")
    assert base != llm.prompt_version_for(
        header + body, "- WiFi (free | paid | no)\n- HasTV (true | false)"
    ), "the injected vocabulary is part of the prompt the model actually sees"
    assert base != llm.prompt_version_for("# prompt_version: v2\n\n" + body, "- WiFi (free)")
    assert base.startswith("v1+")


def test_live_prompt_version_is_well_formed() -> None:
    declared, _, digest = llm.prompt_version().partition("+")
    assert declared == "v1"
    assert len(digest) == 12


def test_prompt_contains_no_gold_query() -> None:
    """Leakage guard: few-shot examples must not be gold queries or paraphrases of them.

    Exact-substring matching catches copy-paste, which is the realistic failure. It cannot
    detect a paraphrase — which is why the prompt's header mandates off-domain example
    categories, a rule a human has to keep.
    """
    gold = Path(__file__).resolve().parent.parent / "eval" / "gold_queries.csv"
    if not gold.exists():
        pytest.skip("eval/gold_queries.csv not written yet (owner-owned work)")

    prompt_text = llm.PROMPT_PATH.read_text(encoding="utf-8").casefold()
    with gold.open(newline="") as fh:
        for row in csv.DictReader(fh):
            query = (row.get("query") or "").strip().casefold()
            if len(query) < 8:  # too short to be a meaningful match
                continue
            assert query not in prompt_text, f"gold query leaked into the prompt: {query!r}"


def test_a_query_cannot_inject_a_placeholder() -> None:
    """Substituting the query last means a user typing "{{ATTRIBUTES}}" cannot expand it."""
    rendered = llm.render_prompt("{{ATTRIBUTES}} {{TODAY}}", today="Thursday")

    assert "{{ATTRIBUTES}} {{TODAY}}" in rendered


# ─────────────────────────────────────────────────────────────────────────────────────
# Live API — skipped unless a real key is present
# ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.needs_api
def test_live_gemini_parse(config, tmp_path: Path) -> None:
    """One real call, to confirm the request shape still matches the live API.

    Everything else about Stage 1 is covered without the network; this test exists because
    a stubbed transport cannot catch a renamed field or a changed endpoint.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    cfg = _with_query_parsing(config, cache_path=tmp_path / "live.sqlite")
    parsed = parser.parse("quiet coffee shop with wifi open past 10pm", cfg)

    assert parsed.source is ParseSource.LLM
    assert parsed.semantic_text
    assert parsed.parse_latency_ms > 0.0
