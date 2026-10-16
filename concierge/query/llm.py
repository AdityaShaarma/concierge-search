"""LLM clients for Stage 1 query understanding.

Gemini Flash is the primary provider (free tier, fast, good enough at this task); Groq is
the fallback, because the realistic failure on a free tier is not "the model is wrong" but
"429, come back later", and a second free provider removes that single point of failure.

This module does exactly four things:

1. Build a provider client from config, reading the API key from the environment.
2. Render the versioned prompt in ``prompts/parse_query.txt`` around a user query.
3. Send one request with a timeout, retrying only the errors retrying can fix.
4. Decode the reply to a JSON object and report the wall-clock latency it cost.

It deliberately does **not** know about caching, the rule fallback, ``ParsedQuery``, or
the attribute vocabulary. Validation belongs to :mod:`concierge.query.schema` and
composition to :mod:`concierge.query.parser`, which is the only thing the pipeline calls.
Returning an unvalidated payload keeps the boundary narrow: one module turns HTTP into
JSON, another decides whether that JSON is trustworthy.

Secrets: the key is read from the environment (``.env`` via python-dotenv) and sent in a
request **header**, never in the URL query string, because URLs end up in logs,
tracebacks and error bodies. No code path here logs, echoes, stores or returns a key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from concierge.config import Config
from concierge.query import schema

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "parse_query.txt"
QUERY_PLACEHOLDER = "{{QUERY}}"
ATTRIBUTES_PLACEHOLDER = "{{ATTRIBUTES}}"
TODAY_PLACEHOLDER = "{{TODAY}}"

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# The shared config carries a single `model` field, which holds a Gemini model name. When
# Groq stands in as the fallback for a Gemini-configured run there is no Groq model name to
# read, so this constant supplies one. Flagged in blocked_on: the config has no
# `groq_model` field, so this is not tunable from YAML.
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# Backoff parameters. Module-level so tests can zero them instead of sleeping, and so the
# numbers are visible in one place rather than buried in a decorator. These three values
# are conventional starting points, NOT tuned against measured rate-limit behaviour — the
# free tier's recovery window is unmeasured. The total worst case (about 3.5s of waiting on
# top of three timeouts) is what bounds how long a search can take before the rule parser
# answers instead, so it is worth revisiting once the latency numbers exist.
_RETRY_WAIT_MULTIPLIER_S = 0.5
_RETRY_WAIT_MAX_S = 8.0
_RETRY_JITTER_S = 0.5

_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# ─────────────────────────────────────────────────────────────────────────────────────
# Errors
#
# A hierarchy rather than one exception type, so the retry predicate and the
# provider-fallback predicate are one-liners over types instead of string matching.
# ─────────────────────────────────────────────────────────────────────────────────────


class LlmError(RuntimeError):
    """Base class for every failure of the LLM parse path."""


class LlmConfigError(LlmError):
    """No usable provider: key missing, or the configured provider is ``rules``.

    Not retryable and not worth a fallback attempt — waiting changes nothing.
    """


class LlmTimeoutError(LlmError):
    """The request exceeded ``cfg.query_parsing.timeout_s``."""


class LlmTransportError(LlmError):
    """Connection-level failure (DNS, reset, TLS). Transient, so retryable."""


class LlmHttpError(LlmError):
    """Non-200 response. ``status_code`` decides whether retrying is sane."""

    def __init__(self, status_code: int, provider: str, detail: str = "") -> None:
        super().__init__(f"{provider} returned HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.provider = provider


class LlmBadResponseError(LlmError):
    """A 200 whose body is not the JSON object we asked for.

    Deliberately *not* retryable: at temperature 0 the same request returns the same
    malformed output, so a retry spends quota to fail again. Falling through to the rule
    parser gets the user an answer instead.
    """


def _is_retryable(exc: BaseException) -> bool:
    """Retry only what retrying can fix.

    Timeouts, connection errors, 429s and 5xx are the provider's problem and usually
    transient. A 400 means *our* request is malformed; retrying it burns free-tier quota
    to earn the same rejection three times as slowly.
    """
    if isinstance(exc, (LlmTimeoutError, LlmTransportError)):
        return True
    if isinstance(exc, LlmHttpError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return False


def _retry_wait() -> wait_exponential_jitter:
    """Exponential backoff with jitter, read from module constants at call time.

    Jitter matters even with one client: the eval harness parses queries in a loop, so a
    fixed backoff would line every retry up at the same later instant and re-trigger the
    same rate limit.
    """
    # Wait before attempt n is multiplier * 2**(n-1) + uniform(0, jitter), capped at max:
    # roughly 0.5s, 1s, 2s with up to half a second of scatter.
    return wait_exponential_jitter(
        multiplier=_RETRY_WAIT_MULTIPLIER_S, max=_RETRY_WAIT_MAX_S, jitter=_RETRY_JITTER_S
    )


# ─────────────────────────────────────────────────────────────────────────────────────
# Prompt loading and versioning
# ─────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Prompt:
    declared_version: str
    body: str
    version: str  # declared version + a hash of body and vocabulary; see prompt_version_for


def _split_header(text: str) -> tuple[list[str], str]:
    """Separate the leading ``#`` comment block from the prompt body.

    The header is repo-facing documentation — the declared version and the leakage rule —
    and must never be sent: it would waste tokens and, worse, tell the model about our
    test methodology.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return lines[:i], "\n".join(lines[i:]).strip() + "\n"


def prompt_version_for(text: str, vocabulary: str = "") -> str:
    """Version of a prompt: ``<declared>+<hash of body and vocabulary>``.

    Every part of this earns its place.

    * The declared version is for humans reading DECISIONS.md and the cache table.
    * The body hash is the safety net. Forgetting to bump the declared version after
      editing the prompt is the easiest mistake in this project to make and the hardest to
      see, because the only symptom is a cache serving parses from a prompt that no longer
      exists. Hashing means the key changes whether or not a human remembers.
    * The *vocabulary* is hashed too, because the attribute list is injected from
      ``schema.describe_attributes()`` rather than written in the file. Without this, adding
      an attribute to the schema would change the prompt the model actually sees while
      leaving the version — and therefore the cache — unchanged. That is the same silent
      corruption the version exists to prevent, arriving through the other module.

    Only the body is read, not the header, so editing a repo-facing comment does not throw
    away a warm cache.
    """
    header, body = _split_header(text)
    declared = "v0"
    for line in header:
        if "prompt_version:" in line:
            declared = line.split("prompt_version:", 1)[1].strip() or declared
            break
    digest = hashlib.sha256(f"{body}\x00{vocabulary}".encode()).hexdigest()[:12]
    return f"{declared}+{digest}"


@lru_cache(maxsize=1)
def load_prompt() -> Prompt:
    """Read and cache the prompt file.

    Cached for the process for two reasons. Re-reading a file per query adds filesystem
    latency to a number we report; more importantly, the version in the cache key must
    describe the prompt that was *actually sent*. If the body were cached while the version
    was recomputed from disk, editing the prompt under a running server would store entries
    under a version whose text nobody ever sent. A prompt edit therefore needs a process
    restart — the right trade for a serving process.
    """
    text = PROMPT_PATH.read_text(encoding="utf-8")
    _, body = _split_header(text)
    version = prompt_version_for(text, schema.describe_attributes())
    return Prompt(declared_version=version.split("+", 1)[0], body=body, version=version)


def prompt_version() -> str:
    """Version of the prompt this process is using. Part of the cache key."""
    return load_prompt().version


def render_prompt(query: str, *, today: str | None = None) -> str:
    """Fill the prompt's placeholders.

    ``str.replace`` rather than ``str.format``: the prompt is full of literal JSON braces,
    and format-style substitution would need every one of them escaped — a maintenance trap
    in a file the owner will edit often.

    ``today`` is passed in rather than read from the clock here so the caller owns the
    timezone decision (the same reasoning as ``rules.parse_rules``). The model needs it
    because ``schema``'s ``open_at`` requires a concrete weekday, so "open past 10pm" has
    to resolve to some day; the caller re-resolves it afterwards, which is what keeps a
    cached parse from freezing the day it was first made.
    """
    body = load_prompt().body
    for placeholder in (QUERY_PLACEHOLDER, ATTRIBUTES_PLACEHOLDER, TODAY_PLACEHOLDER):
        if placeholder not in body:
            raise LlmConfigError(f"prompt is missing the {placeholder} placeholder")
    weekday = today or datetime.now().strftime("%A")
    return (
        body.replace(ATTRIBUTES_PLACEHOLDER, _indent(schema.describe_attributes(), 8))
        .replace(TODAY_PLACEHOLDER, weekday)
        # Last, so a query containing a placeholder string cannot inject one.
        .replace(QUERY_PLACEHOLDER, query.strip())
    )


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in text.splitlines())


# ─────────────────────────────────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────────────────────────────────

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load ``.env`` once per process.

    ``override=False`` so a real environment variable always beats the file. That is also
    what makes tests hermetic: a test sets the variable and the owner's own ``.env`` can
    never reach into it.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        log.debug("python-dotenv not installed; reading keys from the environment only")
    _ENV_LOADED = True


def _api_key(var_name: str) -> str | None:
    """Read a key from the environment. The value is never logged or returned elsewhere."""
    _ensure_env_loaded()
    value = os.environ.get(var_name, "").strip()
    return value or None


# ─────────────────────────────────────────────────────────────────────────────────────
# Providers
# ─────────────────────────────────────────────────────────────────────────────────────


class Provider(Protocol):
    """Minimal provider interface: turn a prompt into raw response text.

    ``name`` and ``model`` are declared read-only (properties, not attributes) so that the
    frozen dataclasses below satisfy the protocol: a client whose model could be reassigned
    mid-flight would make the cache's ``answered_by`` column a lie.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def request(self, http: httpx.Client, prompt: str, timeout_s: float) -> str: ...


def _post_json(
    http: httpx.Client,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_s: float,
    provider: str,
) -> dict[str, Any]:
    """One HTTP POST, with httpx failures mapped onto this module's error types.

    ``raise_for_status`` is not used because the status code itself drives the retry
    decision and a generic ``HTTPStatusError`` would force the predicate to re-derive it.
    Error bodies are truncated in the message: a short excerpt is enough to debug, and
    nothing we send contains a secret outside the (never-logged) headers.
    """
    try:
        resp = http.post(url, json=body, headers=headers, timeout=timeout_s)
    except httpx.TimeoutException as exc:
        raise LlmTimeoutError(f"{provider} timed out after {timeout_s}s") from exc
    except httpx.HTTPError as exc:
        raise LlmTransportError(f"{provider} transport error: {type(exc).__name__}") from exc

    if resp.status_code != 200:
        raise LlmHttpError(resp.status_code, provider, resp.text[:200])
    try:
        data = resp.json()
    except ValueError as exc:
        raise LlmBadResponseError(f"{provider} returned a non-JSON body") from exc
    if not isinstance(data, dict):
        raise LlmBadResponseError(f"{provider} returned {type(data).__name__}, expected an object")
    return data


@dataclass(frozen=True, slots=True)
class GeminiProvider:
    """Google AI Studio (Gemini) ``generateContent`` client."""

    model: str
    api_key: str
    name: str = "gemini"

    def request(self, http: httpx.Client, prompt: str, timeout_s: float) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                # Deterministic: the same query must yield the same filters, both for a
                # reproducible evaluation and so a cache entry is a faithful stand-in for a
                # fresh call.
                "temperature": 0.0,
                "response_mime_type": "application/json",
                # Constrained decoding, in Gemini's OpenAPI dialect. This is strictly
                # better than asking for JSON in prose: an out-of-vocabulary attribute
                # value is refused by the decoder before it can reach us, and the same
                # table generates the validator, so the two cannot drift.
                "response_schema": schema.query_json_schema("openapi"),
            },
        }
        data = _post_json(
            http,
            GEMINI_URL_TEMPLATE.format(model=self.model),
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            body,
            timeout_s,
            self.name,
        )
        return _extract_gemini_text(data)


@dataclass(frozen=True, slots=True)
class GroqProvider:
    """Groq's OpenAI-compatible chat completions client."""

    model: str
    api_key: str
    name: str = "groq"

    def request(self, http: httpx.Client, prompt: str, timeout_s: float) -> str:
        try:
            return self._request(http, prompt, timeout_s, strict_schema=True)
        except LlmHttpError as exc:
            # Strict json_schema support varies by Groq model, and a 400 here means this
            # model does not accept it. This is NOT a retry of the same request — it is one
            # different, weaker request, tried once, because this provider exists precisely
            # to keep working when the primary is rate-limited. Client-side validation
            # catches anything the looser mode lets through.
            if exc.status_code != 400:
                raise
            log.warning(
                "groq rejected strict json_schema for model %s; retrying once in json_object mode",
                self.model,
            )
            return self._request(http, prompt, timeout_s, strict_schema=False)

    def _request(
        self, http: httpx.Client, prompt: str, timeout_s: float, *, strict_schema: bool
    ) -> str:
        response_format: dict[str, Any] = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "parsed_query",
                    "strict": True,
                    "schema": schema.query_json_schema("jsonschema"),
                },
            }
            if strict_schema
            # Groq's json_object mode requires the word "json" in the prompt, which the
            # prompt satisfies by describing the JSON object it wants.
            else {"type": "json_object"}
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": response_format,
        }
        data = _post_json(
            http,
            GROQ_URL,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            body,
            timeout_s,
            self.name,
        )
        return _extract_openai_text(data)


def _extract_gemini_text(data: dict[str, Any]) -> str:
    """Pull the model text out of a ``generateContent`` response.

    A 200 with no candidate is a real case (safety block, ``MAX_TOKENS``), so the reason is
    surfaced in the error: otherwise this failure is indistinguishable in the logs from
    malformed JSON, and the two have completely different fixes.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise LlmBadResponseError(f"gemini returned no candidates ({reason})")
    first = candidates[0] or {}
    parts = ((first.get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise LlmBadResponseError(
            f"gemini returned empty text (finishReason={first.get('finishReason')})"
        )
    return text


def _extract_openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise LlmBadResponseError("provider returned no choices")
    text = ((choices[0] or {}).get("message") or {}).get("content") or ""
    if not text.strip():
        raise LlmBadResponseError("provider returned an empty message")
    return text.strip()


def build_providers(cfg: Config) -> list[Provider]:
    """Provider chain for this config: the configured one first, the other as backup.

    The fallback is a product decision, not belt-and-braces: on a free tier the expected
    failure is a 429, and a demo that dies when the daily Gemini quota runs out is worse
    than one that finishes on Groq a few hundred milliseconds later. A provider with no key
    is skipped rather than attempted.
    """
    choice = cfg.query_parsing.provider
    if choice == "rules":
        raise LlmConfigError("provider is 'rules'; the LLM path must not be attempted")

    gemini_key = _api_key("GEMINI_API_KEY")
    groq_key = _api_key("GROQ_API_KEY")

    chain: list[Provider] = []
    for name in ("gemini", "groq") if choice == "gemini" else ("groq", "gemini"):
        if name == "gemini" and gemini_key:
            chain.append(GeminiProvider(model=_gemini_model(cfg), api_key=gemini_key))
        elif name == "groq" and groq_key:
            chain.append(GroqProvider(model=_groq_model(cfg), api_key=groq_key))

    if not chain:
        # Variable names only — never the values.
        raise LlmConfigError("neither GEMINI_API_KEY nor GROQ_API_KEY is set")
    return chain


def _gemini_model(cfg: Config) -> str:
    """The configured model, unless it is plainly not a Gemini model name."""
    model = cfg.query_parsing.model
    return model if "gemini" in model.lower() else "gemini-2.0-flash"


def _groq_model(cfg: Config) -> str:
    """The configured model if it is a Groq model, else the module default.

    Needed because the config has a single ``model`` field: sending a Gemini model name to
    Groq would be a guaranteed 400 on the one path that exists to survive a 429.
    """
    model = cfg.query_parsing.model
    return DEFAULT_GROQ_MODEL if "gemini" in model.lower() else model


@lru_cache(maxsize=4)
def _shared_client(timeout_s: float) -> httpx.Client:
    """Process-wide httpx client, keyed by timeout.

    Reused because a fresh client pays DNS resolution and a TLS handshake on every query,
    and parse latency is one of the numbers this project reports. ``httpx.Client`` is
    thread-safe, so the FastAPI service can share it.
    """
    return httpx.Client(timeout=timeout_s)


# ─────────────────────────────────────────────────────────────────────────────────────
# The one public call
# ─────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LlmParseResult:
    """A decoded (but not yet validated) reply, plus what it cost and who produced it.

    ``payload`` is raw JSON rather than a ``ParsedQuery`` on purpose. Validation needs the
    attribute vocabulary, which lives in ``concierge.query.schema``; and the cache stores
    payloads, so a bug in the payload-to-``ParsedQuery`` mapping can be fixed later without
    re-spending API quota on queries already parsed.
    """

    payload: dict[str, Any]
    provider: str
    model: str
    prompt_version: str
    latency_ms: float
    attempts: int


def _decode_json(text: str) -> dict[str, Any]:
    """Decode model text into a JSON object.

    The fence strip is defensive: models wrap JSON in ```json fences often enough that
    failing on it would discard otherwise perfect replies over a formatting quirk.
    """
    match = _CODE_FENCE.match(text)
    cleaned = match.group(1) if match else text.strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LlmBadResponseError(f"model output was not JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise LlmBadResponseError(f"model output was {type(obj).__name__}, expected an object")
    return obj


def parse_with_llm(
    query: str,
    cfg: Config,
    *,
    http: httpx.Client | None = None,
    providers: list[Provider] | None = None,
    today: str | None = None,
) -> LlmParseResult:
    """Parse one query with the LLM. Raises ``LlmError`` on every failure path.

    Raising rather than falling back is deliberate: the caller
    (:func:`concierge.query.parser.parse`) owns the fallback policy and must know that a
    failure happened so it can set ``ParseSource`` honestly. A rule-parsed result returned
    silently from here would make the ablation unable to tell the two parsers apart.

    Latency is wall clock around the whole attempt — retries and provider fallback included
    — because that is what a user actually waits for.
    """
    chain = build_providers(cfg) if providers is None else list(providers)
    if not chain:
        raise LlmConfigError("no provider available")

    prompt = render_prompt(query, today=today)
    timeout_s = cfg.query_parsing.timeout_s
    client = http if http is not None else _shared_client(timeout_s)
    # max_retries counts retries, so the attempt budget is one larger.
    max_attempts = max(1, cfg.query_parsing.max_retries + 1)

    start = time.perf_counter()
    attempts = 0  # total HTTP attempts across retries and providers, for debugging
    last_error: LlmError | None = None

    for index, provider in enumerate(chain):
        retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=_retry_wait(),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        try:
            # The policy depends on cfg.max_retries, i.e. runtime data, so tenacity is used
            # imperatively rather than as a decorator.
            text = retrying(provider.request, client, prompt, timeout_s)
            payload = _decode_json(text)
            attempts += int(retrying.statistics.get("attempt_number", 1))
        except LlmError as exc:
            attempts += int(retrying.statistics.get("attempt_number", 1))
            last_error = exc
            # Move to the fallback only for failures meaning "this provider is
            # unavailable". A 400 or a malformed reply would fail the same way there, so
            # trying again elsewhere only adds latency before the rule fallback.
            if index == len(chain) - 1 or not _is_retryable(exc):
                raise
            log.warning(
                "provider %s unavailable (%s); trying fallback provider %s",
                provider.name,
                type(exc).__name__,
                chain[index + 1].name,
            )
            continue

        return LlmParseResult(
            payload=payload,
            provider=provider.name,
            model=provider.model,
            prompt_version=prompt_version(),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            attempts=attempts,
        )

    # Unreachable: the loop either returns or raises. Kept so there is no implicit None.
    raise last_error or LlmConfigError("no provider available")
