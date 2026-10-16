"""Stage 1 public entry point: ``parse(query, cfg) -> ParsedQuery``.

The pipeline calls only this function. It composes the three parse paths in a fixed order
and records which one answered, because the ablation reports LLM parsing as its own row
and that row means nothing if a result cannot be attributed to a parser.

    cache  →  LLM  →  rules

Three rules govern this module, each a product decision rather than a style preference:

1. **A failure never reaches the caller.** A search that 500s because Gemini rate-limited
   is a worse product than a search answered with rule parsing: the user typed "quiet cafe
   with wifi" and deserves results. Every exception below is caught and turned into a
   rule-parsed (or, failing that, pass-through) ``ParsedQuery``.
2. **``provider: rules`` touches no network.** That setting is how the ablation measures
   the rule parser alone, so a stray API call would contaminate its quality *and* latency
   numbers.
3. **``ParseSource`` always tells the truth about where the answer came from**, including a
   cache hit, whose latency story differs from a live call by orders of magnitude.

Division of labour with the two modules either side:

* :mod:`concierge.query.llm` turns HTTP into a JSON payload. It does not validate.
* :mod:`concierge.query.schema` owns the vocabulary and turns a payload into a
  ``ParsedQuery``, or refuses it. Nothing here re-implements any of that: a second copy of
  the attribute vocabulary is exactly the drift D012 warns about.
* :mod:`concierge.query.rules` owns the regex fallback.

This file only decides which parser runs, resolves "today", and gates the cache.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from concierge.config import REPO_ROOT, Config
from concierge.query import llm, rules, schema
from concierge.query.cache import CacheEntry, CacheKey, ParseCache
from concierge.types import ParsedQuery, ParseSource

log = logging.getLogger(__name__)

# Words that make a parse valid only for the day it was made.
#
# The prompt asks for `open_at.day: null` whenever the request does not name a weekday, so
# "open late" and "open now" are resolved against the clock on every read and a cached
# answer never goes stale. "tomorrow" is the exception: the model resolves it to a concrete
# weekday using the date we gave it, and that weekday is wrong from the next midnight
# onwards. Those queries therefore skip the cache entirely — rare enough that the lost hit
# costs nothing, and the alternative is an answer that is silently a day off.
_DAY_SHIFTED = re.compile(r"\b(tomorrow|yesterday|day after|day before)\b", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────────────


def parse(
    query: str,
    cfg: Config,
    *,
    http: httpx.Client | None = None,
    cache: ParseCache | None = None,
    now: datetime | None = None,
    geo_resolver: schema.GeoResolver | None = None,
    known_categories: frozenset[str] | None = None,
) -> ParsedQuery:
    """Turn a user query into a ``ParsedQuery``. Never raises.

    Args:
        query: exactly what the user typed, typos included.
        cfg: loaded config; ``cfg.query_parsing`` drives every choice here.
        http: injected HTTP client. Tests pass one backed by ``httpx.MockTransport``, which
            is what makes this whole module exercisable with no network and no API key.
        cache: injected cache, for tests and for a deliberately cold run.
        now: injected clock, used to resolve a day for an hours constraint. A parameter
            rather than a bare ``datetime.now()`` so the behaviour is testable instead of
            depending on the day the suite happens to run.
        geo_resolver: passed straight to the schema validator, which turns a location
            label into coordinates. Stage 1 has no gazetteer of its own; when the pipeline
            has one (DECISIONS D003) it passes it here and geo constraints start resolving
            with no change to this file.
        known_categories: the index's real category vocabulary, when the caller has it
            loaded. Lets the validator drop hallucinated categories.
    """
    started = time.perf_counter()
    qp = cfg.query_parsing

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    today = (now or datetime.now()).strftime("%A")

    # Rule-only mode and the empty query must not touch the API: the first is the ablation
    # baseline, the second has nothing to parse.
    if qp.provider == "rules" or not query.strip():
        return _rule_parse(query, today, elapsed_ms())

    cache_obj, key = _cache_inputs(query, cfg, cache)

    if cache_obj is not None and key is not None:
        parsed = _from_cache(
            cache_obj, key, query, today, elapsed_ms, geo_resolver, known_categories
        )
        if parsed is not None:
            return parsed

    try:
        result = llm.parse_with_llm(query, cfg, http=http, today=today)
        # Canonicalized first, because that form is JSON-only and is what goes into the
        # cache; the ParsedQuery is then built from the same dict, so what we store and
        # what we return cannot disagree.
        canonical = schema.validate_parse_payload(result.payload)
        parsed = _validate(
            canonical,
            query,
            today,
            ParseSource.LLM,
            elapsed_ms(),
            geo_resolver,
            known_categories,
        )
    except Exception as exc:
        # Covers LlmConfigError (no key), timeouts, 4xx/5xx, malformed JSON and schema
        # rejections. The log records the error type and message, never a key.
        log.warning(
            "LLM parse failed (%s: %s); falling back to %s",
            type(exc).__name__,
            exc,
            "rules" if qp.fallback_to_rules else "pass-through",
        )
        return _fallback(query, cfg, today, elapsed_ms())

    # Written only after validation succeeded, so an unusable reply is never stored. A
    # cached payload that always fails validation would poison that query forever.
    if cache_obj is not None and key is not None:
        _store(cache_obj, key, canonical, result)

    return parsed


# ─────────────────────────────────────────────────────────────────────────────────────
# Validation and the day-of-week question
# ─────────────────────────────────────────────────────────────────────────────────────


def _validate(
    payload: dict[str, Any] | Any,
    query: str,
    today: str,
    source: ParseSource,
    latency_ms: float,
    geo_resolver: schema.GeoResolver | None,
    known_categories: frozenset[str] | None,
) -> ParsedQuery:
    """Validate a payload into a ``ParsedQuery``, or raise ``SchemaValidationError``.

    ``today`` is passed down rather than resolved inside the schema module, which is
    deliberately clock-free. It is what turns a payload whose ``open_at.day`` is null —
    the shape the prompt asks for whenever the user named no weekday, as in "open past
    10pm" — into a concrete day. Resolving it on every call, cache hits included, is what
    stops a stored parse from filtering on the weekday it was first made.
    """
    return schema.validate_parsed_query(
        payload,
        raw_query=query,
        source=source,
        parse_latency_ms=latency_ms,
        today=today,
        geo_resolver=geo_resolver,
        known_categories=known_categories,
    )


def _is_day_shifted(query: str) -> bool:
    """True when the query refers to a day other than today ("tomorrow").

    Such a parse is only valid on the day it was made — the day cannot be re-resolved from
    the clock, because the model folded "tomorrow" into a concrete weekday — so these
    queries bypass the cache entirely. Rare enough that the lost cache hit does not matter,
    and the alternative is an answer that is silently a day stale.
    """
    return bool(_DAY_SHIFTED.search(query))


# ─────────────────────────────────────────────────────────────────────────────────────
# Cache plumbing
# ─────────────────────────────────────────────────────────────────────────────────────


def _cache_inputs(
    query: str, cfg: Config, cache: ParseCache | None
) -> tuple[ParseCache | None, CacheKey | None]:
    """Resolve the cache object and key, or ``(None, None)`` when caching must be skipped.

    The key's provider and model come from the *config*, not from whichever provider ends
    up answering, so an entry stays reusable for the configuration that asked for it. Which
    provider actually answered is recorded in the row instead.
    """
    if _is_day_shifted(query):
        log.debug("skipping the parse cache: %r refers to another day", query)
        return None, None
    try:
        cache_obj = cache if cache is not None else _cache_for(cfg.query_parsing.cache_path)
        key = CacheKey(
            provider=cfg.query_parsing.provider,
            model=cfg.query_parsing.model,
            prompt_version=llm.prompt_version(),
            query=query,
        )
    except Exception as exc:
        # An unwritable cache directory or an unreadable prompt file must not take the
        # search down with it. Caching is lost for this call, and the log says so.
        log.warning("parse cache unavailable (%s: %s)", type(exc).__name__, exc)
        return None, None
    return cache_obj, key


def _from_cache(
    cache_obj: ParseCache,
    key: CacheKey,
    query: str,
    today: str,
    elapsed_ms: Callable[[], float],
    geo_resolver: schema.GeoResolver | None,
    known_categories: frozenset[str] | None,
) -> ParsedQuery | None:
    """Return a validated parse from the cache, or None to fall through to the LLM."""
    try:
        entry: CacheEntry | None = cache_obj.get(key)
    except Exception as exc:
        log.warning("parse cache read failed (%s: %s)", type(exc).__name__, exc)
        return None
    if entry is None:
        return None
    try:
        # Re-validated rather than trusted: the attribute vocabulary may have changed since
        # the entry was written, and the day has to be resolved against today's clock.
        return _validate(
            entry.payload,
            query,
            today,
            ParseSource.CACHE,
            elapsed_ms(),
            geo_resolver,
            known_categories,
        )
    except Exception as exc:
        # A stored payload the validator now rejects means the vocabulary changed or we had
        # a bug, not that the search should fail. Drop it so the next call re-asks.
        log.warning("cached parse is no longer valid (%s); dropping the entry", exc)
        cache_obj.delete(key)
        return None


def _store(
    cache_obj: ParseCache,
    key: CacheKey,
    payload: dict[str, Any],
    result: llm.LlmParseResult,
) -> None:
    try:
        cache_obj.put(
            key,
            payload,
            llm_latency_ms=result.latency_ms,
            # Keyed on the configured provider but recording who answered: when Gemini is
            # rate-limited and Groq replies, the entry must still be reusable under the
            # Gemini config, and we must still be able to tell afterwards which model
            # produced it.
            answered_by=f"{result.provider}:{result.model}",
        )
    except Exception as exc:
        # A failed cache write costs quota later; failing the search costs the user now.
        log.warning("could not write parse cache (%s: %s)", type(exc).__name__, exc)


@lru_cache(maxsize=4)
def _cache_for(path: Path) -> ParseCache:
    """One cache instance per path for the life of the process.

    Opening SQLite per query would add avoidable latency to a reported number;
    ``ParseCache`` is internally locked, so sharing one instance across threads is safe.
    """
    resolved = path if path.is_absolute() else REPO_ROOT / path
    return ParseCache(resolved)


# ─────────────────────────────────────────────────────────────────────────────────────
# Fallbacks
# ─────────────────────────────────────────────────────────────────────────────────────


def _fallback(query: str, cfg: Config, today: str, latency_ms: float) -> ParsedQuery:
    if cfg.query_parsing.fallback_to_rules:
        return _rule_parse(query, today, latency_ms)
    return _passthrough(query, latency_ms)


def _rule_parse(query: str, today: str, latency_ms: float) -> ParsedQuery:
    """Run the rule parser, degrading to pass-through if it somehow raises.

    ``today`` is handed over because ``rules.parse_rules`` is deliberately clock-free:
    without it, "open late" is recognized but not encoded as a filter.

    Latency is overwritten with the time measured here, not the rule parser's own: after a
    failed API attempt, what the user waited for includes that attempt. Reporting only the
    regex time would make the fallback path look free, which is exactly the kind of
    flattering-but-wrong number this project must not produce.
    """
    try:
        parsed = rules.parse_rules(query, today=today)
    except Exception as exc:
        log.warning("rule parser raised (%s: %s); using pass-through", type(exc).__name__, exc)
        return _passthrough(query, latency_ms)
    return replace(parsed, source=ParseSource.RULES, parse_latency_ms=latency_ms)


def _passthrough(query: str, latency_ms: float) -> ParsedQuery:
    """The query as pure semantic text, with no hard constraints.

    Used when rule fallback is switched off (the LLM-only ablation row) and the API fails,
    and as a last resort if the rule parser itself raises. Honest degradation: retrieval
    still runs, no constraint is claimed, and ``ParseSource.RULES`` marks it as "not an LLM
    parse" — the only one of the three values that is true here.
    """
    return ParsedQuery(
        raw=query,
        semantic_text=query.strip(),
        source=ParseSource.RULES,
        parse_latency_ms=latency_ms,
    )
