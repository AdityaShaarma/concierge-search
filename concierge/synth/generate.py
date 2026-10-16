"""Synthetic training queries: the input to both models we train ourselves.

The cross-encoder (Stage 3) and LambdaMART (Stage 4) need labeled
``(query, business)`` pairs, and we have no query log — this is a cold-start
search engine. So we invert the problem: show a language model one business
document and ask what a person would type to find *that* place. The source
business becomes the positive label for free, which is the only reason a few
thousand training pairs are obtainable at zero cost.

Three things in this file are load-bearing and each is pinned by a test:

1. **The leakage guard** (``dedupe_against_gold``). The gold test set is the only
   honest measurement in this project. If a synthetic training query is a
   paraphrase of a gold query, the cross-encoder has effectively seen the test
   set and every downstream number is inflated in a way *no metric can reveal
   afterwards*. So dedup is exact **and** embedding-based, it fails loudly when
   the gold file is absent rather than quietly passing everything through, and
   the number it removes is a required README figure. It is also *on the training
   path by construction*: the guard's output file (``queries_deduped.jsonl``) is
   the only query file the later stages read, so no amount of running stages by
   hand can route around it.
2. **Resumability** (``GenerationStore``). The free Gemini tier rate-limits
   mid-run. A run that lost its completed batches on a 429 would make a few
   thousand queries unaffordable in practice, so every batch's raw response is
   appended to disk the moment it arrives and a rerun replays it without
   spending a call.
3. **Confirmation before spending quota** (``require_confirmation``). CLAUDE.md
   §10 says to state the expected number of API calls and ask first. That is
   enforced here in code, not left to whoever writes the CLI: the call count is
   computed from the plan, cached batches are excluded from it (they cost
   nothing), and a non-interactive process refuses by default.

Everything that touches the network or a sentence-transformer is injected
(``LlmJsonClient``, ``EmbedFn``), so the whole module is unit-testable with
stubs and the test suite never needs an API key or a model download.

This is stage 1 of the four-stage pipeline (``make synth``):

    generate  ->  negatives  ->  judge  ->  splits

Each stage reads only the previous stage's artifact out of
``artifacts/synth/<config fingerprint>/`` and each has its own ``__main__``, so
every guard in the package runs on the real path rather than only in tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from concierge.config import REPO_ROOT, Config
from concierge.query.llm import prompt_version_for
from concierge.types import BusinessDocument, GoldQuery, QueryType

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_queries.txt"

N_QUERIES_PLACEHOLDER = "{{N_QUERIES}}"
QUERY_TYPES_PLACEHOLDER = "{{QUERY_TYPES}}"
BUSINESSES_PLACEHOLDER = "{{BUSINESSES}}"

# What each query type means, injected into the prompt rather than written in it.
# Same reasoning as the attribute vocabulary in concierge/query/llm.py: the enum in
# concierge.types is the contract, and a prompt file that drifted from it would ask
# for a type the parser then rejects, which looks like a model failure and is not.
QUERY_TYPE_DESCRIPTIONS: dict[QueryType, str] = {
    QueryType.VIBE: (
        "atmosphere or occasion, no hard facts — what the place feels like "
        '(e.g. "somewhere calm to work through a long afternoon")'
    ),
    QueryType.CONSTRAINT: (
        "one or more checkable requirements: opening time, price, an amenity "
        '(e.g. "open after 11pm with somewhere to sit outside")'
    ),
    QueryType.SPECIFIC: (
        'a named dish, product or service the place actually offers (e.g. "birria tacos")'
    ),
    QueryType.MESSY: (
        "how people really type: a typo, missing words, lowercase, or two intents at "
        'once (e.g. "cheap brekky place thats open erly sun")'
    ),
}

# Generation temperature. NOT 0.0, unlike query parsing. At temperature 0 the model
# returns its single most likely query per type, which across a thousand cafes is the
# same handful of generic phrasings — training data with no variety teaches the
# cross-encoder to match a template rather than a meaning. Reproducibility is preserved
# a different way: every raw response is stored on disk (GenerationStore), so a rerun
# replays the exact bytes the dataset was built from instead of re-sampling the model.
GENERATION_TEMPERATURE = 0.7

# Transport-level retries inside the default client. Small on purpose: this is a batch
# job, not a user-facing request, and the resumable store makes "stop and rerun later"
# a cheap recovery. A 429 is NOT retried at all — see GeminiJsonClient.complete_json.
_MAX_TRANSPORT_RETRIES = 2
_RETRY_BASE_SLEEP_S = 1.0

_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class SynthError(RuntimeError):
    """Base class for every failure in the synthetic-data pipeline."""


class SynthConfigError(SynthError):
    """Missing key, missing prompt placeholder — nothing a retry can fix."""


class SynthRateLimitError(SynthError):
    """The provider returned 429.

    Handled by stopping the run cleanly rather than backing off. On a free tier a
    429 usually means the *daily* quota is gone, so sleeping would burn wall-clock
    for nothing; the completed batches are already on disk and a rerun tomorrow
    picks up exactly where this one stopped.
    """


class SynthTransportError(SynthError):
    """Timeout, connection reset or 5xx. Transient, so retried a couple of times."""


class SynthBadResponseError(SynthError):
    """A 200 whose body is not the JSON object we asked for."""


class QuotaNotConfirmedError(SynthError):
    """A batch job that would spend API quota was not confirmed.

    Raised instead of silently proceeding, because the owner's free tier is a
    shared finite resource across every phase of this project.
    """


class GoldSetMissingError(SynthError):
    """``eval/gold_queries.csv`` is absent, so the leakage guard cannot run.

    A hard error rather than a warning. "No gold file, so nothing to deduplicate
    against" would hand back every synthetic query untouched and look like a
    successful dedup in the logs — the exact failure the guard exists to prevent.
    """


class SynthArtifactMissingError(SynthError):
    """A pipeline stage was run before the stage that produces its input.

    Each stage reads exactly one artifact — the previous stage's output — and
    refuses to fall back to anything else. That is what makes the gold leakage
    guard unbypassable: the only machine-readable query file the later stages
    read is the *deduplicated* one, so running ``make synth-negatives`` on its own
    cannot quietly train on queries the guard never saw. The error names the make
    target to run instead of guessing at a substitute input.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization and stable ids
# ─────────────────────────────────────────────────────────────────────────────


def normalize_query_text(text: str) -> str:
    """Canonical form used for exact-duplicate matching and for query ids.

    Case, surrounding punctuation and runs of whitespace are folded because
    "Quiet cafe with WiFi?" and "quiet cafe with wifi" are the same query for
    leakage purposes. Nothing *inside* the string is touched: word order and
    internal punctuation change meaning, and the gold set deliberately contains
    messy queries whose typos are the point of the ``messy`` category.
    """
    return _EDGE_PUNCT.sub("", _WHITESPACE.sub(" ", text.strip())).casefold()


def synthetic_query_id(business_id: str, query: str) -> str:
    """Deterministic id: ``sq-<hash of (business, normalized query)>``.

    Content-addressed rather than a counter, so rerunning generation over a
    superset of businesses does not renumber existing queries. That matters
    because the split manifest (``concierge.synth.splits``) is keyed by query id:
    with counters, adding businesses would reshuffle which split a query lands
    in, and a query could move from the cross-encoder's training split into
    LambdaMART's — silent leakage from a bookkeeping choice.

    **The limit of content addressing, and where it is handled.** The id is a
    hash of the *exact* normalized text, so two paraphrases of one query get
    unrelated ids — and therefore, if splits were hashed from the id alone,
    possibly different splits. Splitting is done over near-duplicate *clusters*
    (``cluster_near_duplicates``) precisely because of this; an id is an identity,
    not a measure of similarity, and nothing here pretends otherwise.
    """
    digest = hashlib.sha256(
        json.dumps([business_id, normalize_query_text(query)], ensure_ascii=False).encode()
    ).hexdigest()
    return f"sq-{digest[:12]}"


# ─────────────────────────────────────────────────────────────────────────────
# The dataset record
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SyntheticQuery:
    """One generated query plus the business that is its positive label.

    ``prompt_version``, ``provider`` and ``model`` travel with every row so a
    training set can always be traced to the prompt and model that produced it.
    Without them, a prompt edit halfway through generation would leave a dataset
    that is a silent mixture of two generators.
    """

    query_id: str
    query: str
    query_type: QueryType
    business_id: str
    prompt_version: str = ""
    provider: str = ""
    model: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type.value,
            "business_id": self.business_id,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> SyntheticQuery:
        return cls(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            query_type=QueryType(row["query_type"]),
            business_id=str(row["business_id"]),
            prompt_version=str(row.get("prompt_version", "")),
            provider=str(row.get("provider", "")),
            model=str(row.get("model", "")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────


def _strip_header(text: str) -> str:
    """Drop the leading ``#`` metadata block; return the prompt body.

    Mirrors the convention in ``concierge/query/prompts/parse_query.txt``: the
    header documents the leakage rule for whoever edits the file and must never
    be sent to a model — it would waste tokens and, worse, describe our test
    methodology to the thing being tested.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip() + "\n"


def render_query_types() -> str:
    """The query-type vocabulary as the model sees it (also hashed into the version)."""
    return "\n".join(
        f"- {qtype.value}: {description}" for qtype, description in QUERY_TYPE_DESCRIPTIONS.items()
    )


def prompt_text() -> str:
    """Raw prompt file contents, header included (used for versioning and tests)."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def generation_prompt_version() -> str:
    """``<declared version>+<hash of body and query-type vocabulary>``.

    Reuses ``concierge.query.llm.prompt_version_for`` so both prompts in the repo
    are versioned identically. The vocabulary is hashed in because the type
    descriptions are injected from ``QUERY_TYPE_DESCRIPTIONS``: editing them
    changes what the model is asked for while leaving the file untouched, and the
    version has to move with it or the response cache serves answers to a question
    nobody asks any more.
    """
    return prompt_version_for(prompt_text(), render_query_types())


def render_generation_prompt(
    documents: Sequence[BusinessDocument],
    queries_per_business: int,
    *,
    max_chars: int | None = None,
) -> str:
    """Fill the prompt with one batch of business documents.

    ``str.replace`` rather than ``str.format`` because the prompt contains literal
    JSON braces, and format-style substitution would require escaping every one of
    them in a file the owner is expected to edit.
    """
    body = _strip_header(prompt_text())
    for placeholder in (N_QUERIES_PLACEHOLDER, QUERY_TYPES_PLACEHOLDER, BUSINESSES_PLACEHOLDER):
        if placeholder not in body:
            raise SynthConfigError(f"generate_queries.txt is missing {placeholder}")
    blocks = []
    for doc in documents:
        text = doc.text if max_chars is None else doc.text[:max_chars]
        blocks.append(f"--- BUSINESS {doc.business_id} ---\n{text.strip()}")
    return (
        body.replace(N_QUERIES_PLACEHOLDER, str(queries_per_business))
        .replace(QUERY_TYPES_PLACEHOLDER, render_query_types())
        # Documents last, so a review quoting "{{N_QUERIES}}" cannot inject a placeholder.
        .replace(BUSINESSES_PLACEHOLDER, "\n\n".join(blocks))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quota confirmation (CLAUDE.md §10)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QuotaPlan:
    """What a batch job is about to spend, in the units the free tier bills in.

    Shared by generation and by the judge so both ask the same question the same
    way. ``n_api_calls`` counts only the batches that are *not* already cached —
    quoting the total number of batches would overstate the cost of a resumed run
    and train the owner to wave the confirmation through.
    """

    job: str
    n_items: int
    items_label: str
    n_batches: int
    n_cached_batches: int
    n_api_calls: int
    batch_size: int
    note: str = ""

    def describe(self) -> str:
        lines = [
            f"{self.job}:",
            f"  {self.n_items} {self.items_label} in {self.n_batches} batches "
            f"of up to {self.batch_size}",
            f"  {self.n_cached_batches} batches already cached (no API call)",
            f"  → {self.n_api_calls} API call(s) will be made",
        ]
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


ConfirmFn = Callable[[QuotaPlan], bool]


def confirm_on_stdin(plan: QuotaPlan) -> bool:
    """Interactive confirmation; refuses by default when there is no terminal.

    The non-interactive refusal is the important half. A cron job or a notebook
    cell that inherited this default must not be able to spend the day's quota
    because nobody was watching — such a caller has to pass ``assume_yes=True``
    explicitly, which is then a visible decision in the call site.
    """
    print(plan.describe(), file=sys.stderr)
    if not sys.stdin.isatty():
        print(
            "refusing to spend API quota without confirmation (no terminal attached); "
            "pass assume_yes=True to proceed deliberately",
            file=sys.stderr,
        )
        return False
    answer = input("type 'yes' to spend this quota: ").strip().casefold()
    return answer == "yes"


def require_confirmation(
    plan: QuotaPlan,
    confirm: ConfirmFn | None = None,
    *,
    assume_yes: bool = False,
) -> None:
    """Gate a quota-spending job. Raises ``QuotaNotConfirmedError`` if declined.

    A plan with zero API calls is never asked about: a fully cached rerun spends
    nothing, and prompting for it would be noise.
    """
    if plan.n_api_calls == 0:
        log.info("%s: every batch is cached, no API calls needed", plan.job)
        return
    if assume_yes:
        log.warning("%s: proceeding with %d API call(s) (assume_yes)", plan.job, plan.n_api_calls)
        return
    decide = confirm_on_stdin if confirm is None else confirm
    if not decide(plan):
        raise QuotaNotConfirmedError(
            f"{plan.job} not confirmed; {plan.n_api_calls} API call(s) were not made"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Resumable storage
# ─────────────────────────────────────────────────────────────────────────────


class JsonlCache:
    """Append-only JSONL keyed by a stable id. Resumability, in 40 lines.

    Append-only rather than rewrite-in-place for one reason: a process killed
    mid-write (rate limit, Ctrl-C, laptop sleep) can at worst leave one truncated
    last line, which ``load`` skips. Rewriting a whole file can lose everything.
    Each ``append`` flushes and fsyncs, so "the batch came back" and "the batch is
    on disk" are the same event — anything less and a 429 one batch later would
    throw away a call that was already paid for.

    This is also the only writer in the synthetic-data package; the judge imports
    it rather than opening files itself, which is what lets
    ``concierge.synth.judge`` be structurally incapable of writing to disk.
    """

    def __init__(self, path: str | Path, key_field: str = "id") -> None:
        self.path = Path(path)
        self.key_field = key_field

    def load(self) -> dict[str, dict[str, Any]]:
        """Records by key, later entries winning. Unreadable lines are skipped."""
        if not self.path.exists():
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Expected at most once, on the final line, after a hard kill.
                    log.warning("skipping unreadable line %d of %s", lineno, self.path)
                    continue
                key = record.get(self.key_field)
                if isinstance(key, str):
                    out[key] = record
        return out

    def append(self, records: Iterable[Mapping[str, Any]]) -> int:
        """Append records, flushing to disk before returning. Returns the count."""
        rows = [dict(r) for r in records]
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact layout
# ─────────────────────────────────────────────────────────────────────────────
#
# One directory per config fingerprint, and one filename per stage, named here
# rather than in each stage's CLI. Four separate programs read and write these
# files (`make synth-generate|synth-negatives|synth-judge|synth-splits`), and a
# filename spelled slightly differently in two of them would mean a stage
# silently reading last week's data — the failure the fingerprinted directory
# exists to prevent in the first place.

#: Every accepted query, as generated. NOT an input to training: it has not been
#: through the gold leakage guard. Only ``make synth-generate`` reads it.
QUERIES_FILENAME = "queries.jsonl"

#: The leakage guard's output and **the only query file later stages read**.
QUERIES_DEDUPED_FILENAME = "queries_deduped.jsonl"

#: The same rows as CSV, purely so the owner can skim them in a spreadsheet.
QUERIES_DEDUPED_CSV_FILENAME = "queries_deduped.csv"

#: The removal counts. A required README number, so it is a file and not a log line.
DEDUP_REPORT_FILENAME = "dedup_report.json"


def synth_dir(cfg: Config, root: Path | None = None) -> Path:
    """``artifacts/synth/<config fingerprint>/`` — where every stage reads and writes.

    Fingerprinted for the same reason the indexes are: a config change must not be
    answerable with data built under the previous one. ``root`` is injectable so
    tests (and a run that builds a dataset outside the repo tree) can redirect the
    whole pipeline without touching the repo's artifacts.
    """
    base = REPO_ROOT if root is None else Path(root)
    return base / "artifacts" / "synth" / cfg.fingerprint()


@dataclass(frozen=True, slots=True)
class GenerationStore:
    """Where a generation run keeps its raw responses and its accepted queries.

    Two files, deliberately in two different places:

    * ``responses`` lives under ``.llm_cache/`` — it is *spent money*. ``make
      clean`` wipes ``artifacts/`` and must never be able to delete the record of
      API calls that have already been paid for.
    * ``queries`` lives under ``artifacts/`` — it is derived data, rebuildable
      from the responses with no API call at all.
    """

    responses: JsonlCache
    queries: JsonlCache

    @classmethod
    def from_config(cls, cfg: Config, root: Path | None = None) -> GenerationStore:
        base = REPO_ROOT if root is None else Path(root)
        return cls(
            responses=JsonlCache(base / ".llm_cache" / "synth_generate.jsonl", "batch_key"),
            queries=JsonlCache(synth_dir(cfg, root) / QUERIES_FILENAME, "query_id"),
        )

    def cached_batch_keys(self) -> set[str]:
        return set(self.responses.load())

    def load_queries(self) -> list[SyntheticQuery]:
        """Every accepted query seen so far, deduplicated by id."""
        return [SyntheticQuery.from_json(r) for r in self.queries.load().values()]


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────


class LlmJsonClient(Protocol):
    """Prompt in, JSON object out. The only network surface in this package.

    Narrower than ``concierge.query.llm.Provider`` on purpose: that protocol's
    implementations hard-code the *query-parsing* response schema in the request
    body, so they cannot be reused to ask for anything else. Reported in
    blocked_on.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete_json(self, prompt: str) -> dict[str, Any]: ...


def _api_key(var_name: str) -> str | None:
    """Read a key from the environment, loading ``.env`` once. Never logged."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)  # a real env var always beats the file
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass
    return os.environ.get(var_name, "").strip() or None


def _decode_json(text: str) -> dict[str, Any]:
    """Decode model text to a JSON object, tolerating ```json fences."""
    match = _CODE_FENCE.match(text)
    cleaned = match.group(1) if match else text.strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SynthBadResponseError(f"model output was not JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise SynthBadResponseError(f"model output was {type(obj).__name__}, expected an object")
    return obj


@dataclass(frozen=True)
class GeminiJsonClient:
    """Gemini ``generateContent`` with a caller-supplied response schema.

    Constrained decoding (``response_schema``) rather than asking for JSON in
    prose: an out-of-vocabulary query type is then refused by the decoder before
    it reaches us, which is strictly cheaper than detecting it afterwards and
    re-spending a call.

    The key is sent in a header, never in the URL — URLs end up in logs and
    tracebacks. No code path here logs, returns or stores the key.
    """

    api_key: str
    model: str = "gemini-2.0-flash"
    response_schema: Mapping[str, Any] | None = None
    temperature: float = GENERATION_TEMPERATURE
    timeout_s: float = 60.0
    name: str = "gemini"

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        response_schema: Mapping[str, Any] | None = None,
        temperature: float = GENERATION_TEMPERATURE,
    ) -> GeminiJsonClient:
        key = _api_key("GEMINI_API_KEY")
        if not key:
            raise SynthConfigError("GEMINI_API_KEY is not set")  # variable name only
        model = cfg.query_parsing.model
        return cls(
            api_key=key,
            model=model if "gemini" in model.lower() else "gemini-2.0-flash",
            response_schema=response_schema,
            temperature=temperature,
            # Deliberately not cfg.query_parsing.timeout_s (4s): that budget is a
            # *user* waiting for a search. A batch of eight documents is a slow
            # offline call and a 4s cap would fail it every time.
            timeout_s=60.0,
        )

    def complete_json(self, prompt: str) -> dict[str, Any]:
        import httpx

        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
        }
        if self.response_schema is not None:
            generation_config["response_schema"] = dict(self.response_schema)
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        url = GEMINI_URL_TEMPLATE.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        last: SynthError | None = None
        for attempt in range(_MAX_TRANSPORT_RETRIES + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as http:
                    resp = http.post(url, json=body, headers=headers)
            except httpx.TimeoutException as exc:
                last = SynthTransportError(f"gemini timed out after {self.timeout_s}s")
                last.__cause__ = exc
            except httpx.HTTPError as exc:
                last = SynthTransportError(f"gemini transport error: {type(exc).__name__}")
                last.__cause__ = exc
            else:
                if resp.status_code == 429:
                    # Not retried: on a free tier this is usually the daily quota, so
                    # backing off burns wall-clock without changing the answer. The
                    # caller stops the run and keeps everything already on disk.
                    raise SynthRateLimitError(f"gemini rate-limited: {resp.text[:200]}")
                if resp.status_code != 200:
                    detail = resp.text[:200]
                    if 500 <= resp.status_code < 600:
                        last = SynthTransportError(f"gemini HTTP {resp.status_code}: {detail}")
                    else:
                        raise SynthBadResponseError(f"gemini HTTP {resp.status_code}: {detail}")
                else:
                    return _decode_json(_extract_gemini_text(resp.json()))
            if attempt < _MAX_TRANSPORT_RETRIES:
                time.sleep(_RETRY_BASE_SLEEP_S * (2**attempt))
        raise last or SynthTransportError("gemini request failed")


def _extract_gemini_text(data: Any) -> str:
    """Pull the text part out of a ``generateContent`` response."""
    if not isinstance(data, dict):
        raise SynthBadResponseError("gemini returned a non-object body")
    candidates = data.get("candidates") or []
    if not candidates:
        # Usually a safety block. Surfaced rather than swallowed: a whole batch
        # silently vanishing would look like a model that produces no queries for
        # certain businesses, which is a very confusing thing to debug later.
        raise SynthBadResponseError(f"gemini returned no candidates: {str(data)[:200]}")
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
    if not text.strip():
        raise SynthBadResponseError("gemini returned an empty text part")
    return text


def generation_response_schema() -> dict[str, Any]:
    """OpenAPI-dialect schema for the generation reply (constrained decoding)."""
    return {
        "type": "object",
        "properties": {
            "businesses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "business_id": {"type": "string"},
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [q.value for q in QueryType],
                                    },
                                },
                                "required": ["query", "type"],
                            },
                        },
                    },
                    "required": ["business_id", "queries"],
                },
            }
        },
        "required": ["businesses"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batching and response parsing
# ─────────────────────────────────────────────────────────────────────────────


def plan_batches(
    documents: Sequence[BusinessDocument],
    batch_size: int,
) -> list[tuple[BusinessDocument, ...]]:
    """Split documents into batches of ``batch_size``, in the order given.

    Batching is what makes a few thousand queries affordable: the free tier limits
    *requests per minute*, not tokens, so eight documents in one call costs an
    eighth of eight calls. Input order is preserved rather than sorted, so a
    caller that sampled businesses with a seed keeps its sampling intact.
    """
    if batch_size < 1:
        raise SynthConfigError(f"batch_size must be >= 1, got {batch_size}")
    return [tuple(documents[i : i + batch_size]) for i in range(0, len(documents), batch_size)]


def batch_key(
    batch: Sequence[BusinessDocument],
    prompt_version: str,
    queries_per_business: int,
) -> str:
    """Cache key for one batch: its business ids, the prompt version and the ask.

    The prompt version is in the key for the same reason it is in the query-parse
    cache key: an edited prompt is a different generator, and serving its
    predecessor's answers would make the training set a silent mixture of two.
    """
    payload = json.dumps(
        {
            "business_ids": [d.business_id for d in batch],
            "prompt_version": prompt_version,
            "queries_per_business": queries_per_business,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ParsedBatch:
    queries: tuple[SyntheticQuery, ...]
    n_rejected: int
    rejection_reasons: tuple[str, ...]


def parse_generation_response(
    payload: Mapping[str, Any],
    batch: Sequence[BusinessDocument],
    *,
    prompt_version: str = "",
    provider: str = "",
    model: str = "",
) -> ParsedBatch:
    """Turn one reply into ``SyntheticQuery`` rows, rejecting what we cannot trust.

    The strict check is ``business_id`` membership in the batch. A model that
    invents or mistypes an id would attach queries to the wrong positive, and a
    wrong positive is worse than a missing one: the cross-encoder would be trained
    to score an unrelated business as the right answer. Those rows are dropped and
    counted, never remapped by position — "it was probably the second one" is a
    guess, and a guess here corrupts labels.
    """
    allowed = {doc.business_id for doc in batch}
    queries: list[SyntheticQuery] = []
    reasons: list[str] = []
    seen: set[str] = set()

    entries = payload.get("businesses")
    if not isinstance(entries, list):
        return ParsedBatch((), 0, ("reply had no 'businesses' array",))

    for entry in entries:
        if not isinstance(entry, dict):
            reasons.append("entry was not an object")
            continue
        business_id = str(entry.get("business_id", "")).strip()
        if business_id not in allowed:
            reasons.append(f"unknown business_id {business_id!r}")
            continue
        raw_queries = entry.get("queries")
        if not isinstance(raw_queries, list):
            reasons.append(f"{business_id}: 'queries' was not an array")
            continue
        for item in raw_queries:
            if not isinstance(item, dict):
                reasons.append(f"{business_id}: query entry was not an object")
                continue
            text = str(item.get("query", "")).strip()
            raw_type = str(item.get("type", "")).strip().casefold()
            if not text:
                reasons.append(f"{business_id}: empty query")
                continue
            try:
                qtype = QueryType(raw_type)
            except ValueError:
                reasons.append(f"{business_id}: unknown query type {raw_type!r}")
                continue
            query_id = synthetic_query_id(business_id, text)
            if query_id in seen:
                # Same business, same text twice in one reply. Not an error worth
                # reporting as a rejection of content, but it must not be emitted
                # twice or the positive would be duplicated in training.
                continue
            seen.add(query_id)
            queries.append(
                SyntheticQuery(
                    query_id=query_id,
                    query=text,
                    query_type=qtype,
                    business_id=business_id,
                    prompt_version=prompt_version,
                    provider=provider,
                    model=model,
                )
            )
    return ParsedBatch(tuple(queries), len(reasons), tuple(reasons))


# ─────────────────────────────────────────────────────────────────────────────
# The run
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GenerationRun:
    """What a generation pass produced and what it cost, with nothing smoothed over.

    ``stopped_early`` plus ``stop_reason`` are part of the result rather than a log
    line: a run that was rate-limited after 40 of 150 batches produced a *usable
    but partial* dataset, and training on it without knowing that would be
    indistinguishable from training on the whole thing.
    """

    queries: tuple[SyntheticQuery, ...]
    n_api_calls: int
    n_batches_cached: int
    n_batches_failed: int
    n_rejected_entries: int
    stopped_early: bool = False
    stop_reason: str = ""
    prompt_version: str = ""

    @property
    def type_coverage(self) -> dict[str, int]:
        """How many queries of each type — the prompt asks for all four, so a
        missing type means the ask is not working and the data is unbalanced."""
        counts = Counter(q.query_type.value for q in self.queries)
        return {qtype.value: counts.get(qtype.value, 0) for qtype in QueryType}

    def summary(self) -> str:
        parts = [
            f"{len(self.queries)} queries over {len({q.business_id for q in self.queries})} "
            "businesses",
            f"{self.n_api_calls} API call(s), {self.n_batches_cached} batch(es) from cache",
            f"type coverage: {self.type_coverage}",
        ]
        if self.n_batches_failed:
            parts.append(f"{self.n_batches_failed} batch(es) failed")
        if self.n_rejected_entries:
            parts.append(f"{self.n_rejected_entries} rejected entr(ies)")
        if self.stopped_early:
            parts.append(f"STOPPED EARLY: {self.stop_reason}")
        return "; ".join(parts)


def generate_queries(
    documents: Sequence[BusinessDocument],
    cfg: Config,
    *,
    client: LlmJsonClient | None = None,
    store: GenerationStore | None = None,
    confirm: ConfirmFn | None = None,
    assume_yes: bool = False,
    on_batch: Callable[[int, int], None] | None = None,
) -> GenerationRun:
    """Generate ``cfg.synth.queries_per_business`` queries per business document.

    Order of operations matters and is the whole design:

    1. plan the batches and count how many are *not* already cached;
    2. ask before spending anything (raises if declined);
    3. per batch: serve from cache if possible, else call the API, **write the raw
       response to disk immediately**, parse it, append the accepted queries;
    4. on a rate limit: stop, keep everything, and say so in the result.

    Deduplication against the gold set is deliberately *not* done here — it is a
    separate step (``dedupe_against_gold``) so that the expensive, quota-spending
    half can be rerun or resumed without re-embedding, and so the removal count is
    produced by one auditable function.
    """
    synth = cfg.synth
    version = generation_prompt_version()
    the_store = GenerationStore.from_config(cfg) if store is None else store
    batches = plan_batches(documents, synth.batch_size)
    cached = the_store.cached_batch_keys()
    keys = [batch_key(b, version, synth.queries_per_business) for b in batches]
    n_cached = sum(1 for k in keys if k in cached)

    plan = QuotaPlan(
        job="synthetic query generation",
        n_items=len(documents),
        items_label="business documents",
        n_batches=len(batches),
        n_cached_batches=n_cached,
        n_api_calls=len(batches) - n_cached,
        batch_size=synth.batch_size,
        note=(
            f"asking for {synth.queries_per_business} queries per business "
            f"(~{len(documents) * synth.queries_per_business} queries before dedup)"
        ),
    )
    require_confirmation(plan, confirm, assume_yes=assume_yes)

    the_client = client
    if the_client is None and plan.n_api_calls > 0:
        the_client = GeminiJsonClient.from_config(cfg, response_schema=generation_response_schema())

    responses = the_store.responses.load()
    # Query ids already on disk. Replaying cached batches re-derives the same rows,
    # and appending them again would grow the JSONL on every rerun — harmless
    # (``load`` keys by id) but it would make the file's line count meaningless.
    already_stored = set(the_store.queries.load())
    collected: list[SyntheticQuery] = []
    n_api = n_failed = n_rejected = 0
    stopped_early = False
    stop_reason = ""

    for index, (batch, key) in enumerate(zip(batches, keys, strict=True), start=1):
        if on_batch is not None:
            on_batch(index, len(batches))
        record = responses.get(key)
        provider = str(record.get("provider", "")) if record else ""
        model = str(record.get("model", "")) if record else ""

        if record is None:
            assert the_client is not None  # plan.n_api_calls > 0 guaranteed a client
            try:
                payload = the_client.complete_json(
                    render_generation_prompt(
                        batch,
                        synth.queries_per_business,
                        max_chars=cfg.documents.max_chars,
                    )
                )
            except SynthRateLimitError as exc:
                stopped_early = True
                stop_reason = f"rate limited after {n_api} API call(s): {exc}"
                log.warning("%s; completed work is already on disk", stop_reason)
                break
            except SynthError as exc:
                # One bad batch must not end a run that has paid for the others.
                n_failed += 1
                log.warning("batch %d/%d failed (%s)", index, len(batches), exc)
                continue
            n_api += 1
            provider, model = the_client.name, the_client.model
            the_store.responses.append(
                [
                    {
                        "batch_key": key,
                        "payload": payload,
                        "provider": provider,
                        "model": model,
                        "prompt_version": version,
                        "business_ids": [d.business_id for d in batch],
                    }
                ]
            )
        else:
            payload = record.get("payload") or {}

        parsed = parse_generation_response(
            payload, batch, prompt_version=version, provider=provider, model=model
        )
        n_rejected += parsed.n_rejected
        for reason in parsed.rejection_reasons:
            log.warning("batch %d/%d rejected an entry: %s", index, len(batches), reason)
        fresh = [q for q in parsed.queries if q.query_id not in already_stored]
        if fresh:
            # Appended per batch, not at the end of the run: a rate limit one batch
            # later must not be able to lose a response already paid for.
            the_store.queries.append([q.to_json() for q in fresh])
            already_stored.update(q.query_id for q in fresh)
        collected.extend(parsed.queries)

    # Deduplicated by id: a rerun re-parses cached batches, and the same query must
    # not appear twice just because the store was replayed.
    unique = {q.query_id: q for q in collected}
    return GenerationRun(
        queries=tuple(unique.values()),
        n_api_calls=n_api,
        n_batches_cached=n_cached,
        n_batches_failed=n_failed,
        n_rejected_entries=n_rejected,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        prompt_version=version,
    )


# ─────────────────────────────────────────────────────────────────────────────
# THE LEAKAGE GUARD
# ─────────────────────────────────────────────────────────────────────────────


def load_gold_queries(path: str | Path) -> tuple[GoldQuery, ...]:
    """Read ``eval/gold_queries.csv``. Read-only; this module never writes it.

    Raises ``GoldSetMissingError`` when the file is absent. Returning an empty
    tuple instead would let the dedup step report "0 removed" and look like it
    worked, which is precisely the failure mode the guard exists to prevent.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise GoldSetMissingError(
            f"gold query file not found: {p}. The leakage guard cannot run without it. "
            "The owner writes these queries (CLAUDE.md §6); do not generate them."
        )
    out: list[GoldQuery] = []
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            query = (row.get("query") or "").strip()
            if not query:
                continue
            raw_type = (row.get("query_type") or "").strip().casefold()
            try:
                qtype = QueryType(raw_type)
            except ValueError as exc:
                raise SynthError(
                    f"gold query {row.get('query_id')!r} has unknown query_type "
                    f"{raw_type!r}; expected one of {[q.value for q in QueryType]}"
                ) from exc
            out.append(
                GoldQuery(
                    query_id=(row.get("query_id") or "").strip(),
                    query=query,
                    query_type=qtype,
                )
            )
    return tuple(out)


EmbedFn = Callable[[Sequence[str]], Any]
"""Texts -> one vector per text (anything ``numpy.asarray`` accepts)."""


def default_embed_fn(cfg: Config) -> EmbedFn:
    """Embed with the same model Stage 2 retrieves with, treating both sides as queries.

    Both sides get bge's *query* prefix because both sides are queries: we are
    asking "do these two requests mean the same thing", not "does this request
    match this document". Using the asymmetric passage encoding on one side would
    compare vectors from two different regions of the space and make the cosine
    threshold meaningless.

    Imported lazily — nothing in the test suite should pay for torch.
    """

    def embed(texts: Sequence[str]) -> Any:
        from concierge.retrieval.dense import DenseEmbedder

        return DenseEmbedder.from_config(cfg).embed_queries(list(texts))

    return embed


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """One removed query and what it collided with. Kept for the README evidence."""

    query_id: str
    query: str
    matched_id: str
    matched_query: str
    kind: str  # exact_gold | near_gold | exact_internal | near_internal
    cosine: float | None = None  # None for an exact match: no embedding was needed


@dataclass(frozen=True, slots=True)
class DedupReport:
    """The leakage-guard numbers. ``n_removed_gold`` is a required README figure."""

    n_input: int
    n_kept: int
    n_gold: int
    threshold: float
    n_removed_exact_gold: int
    n_removed_near_gold: int
    n_removed_exact_internal: int
    n_removed_near_internal: int
    removed: tuple[DuplicatePair, ...]

    @property
    def n_removed_gold(self) -> int:
        return self.n_removed_exact_gold + self.n_removed_near_gold

    @property
    def n_removed_internal(self) -> int:
        return self.n_removed_exact_internal + self.n_removed_near_internal

    def summary(self) -> str:
        return (
            f"gold dedup: {self.n_input} synthetic queries in, {self.n_kept} kept. "
            f"Removed {self.n_removed_gold} as gold duplicates "
            f"({self.n_removed_exact_gold} exact, {self.n_removed_near_gold} with cosine "
            f">= {self.threshold} against {self.n_gold} gold queries) and "
            f"{self.n_removed_internal} as duplicates within the synthetic set "
            f"({self.n_removed_exact_internal} exact, {self.n_removed_near_internal} near)."
        )


def _normalized_matrix(embed: EmbedFn, texts: Sequence[str]) -> Any:
    """Embed and L2-normalize, so a dot product is a cosine."""
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    matrix = np.asarray(embed(list(texts)), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise SynthError(
            f"embed function returned shape {getattr(matrix, 'shape', None)} for "
            f"{len(texts)} texts; expected ({len(texts)}, dim)"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leaving it unscaled makes its cosine 0.0
    # against everything, which is wrong for that row but harmlessly so — far
    # better than a NaN, which would compare False and silently skip the guard.
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def dedupe_against_gold(
    queries: Sequence[SyntheticQuery],
    gold: Sequence[GoldQuery],
    embed: EmbedFn,
    *,
    threshold: float = 0.92,
    dedupe_within: bool = True,
    chunk_size: int = 256,
) -> tuple[list[SyntheticQuery], DedupReport]:
    """Remove synthetic queries that duplicate a gold query (or each other).

    **Why both checks.** Exact matching catches the obvious case and costs
    nothing. It is also nearly useless on its own: "quiet cafe with wifi open
    late" and "a quiet coffee shop with wifi that's open late" are the same test
    item for every purpose that matters and share no normalized form. The
    embedding check is what actually protects the test set; the exact check just
    makes the cheap half of the answer free and auditable.

    **Why a threshold and not a classifier.** 0.92 cosine
    (``cfg.synth.gold_dedup_cosine_threshold``) is a configured, reported constant,
    not a tuned one — tuning it would need labeled near-duplicate pairs we do not
    have. It is deliberately strict: the cost of discarding a usable training
    query is one row out of thousands, and the cost of keeping a leaked one is
    that every number in the README is wrong. When in doubt, drop it.

    **Determinism.** Queries are processed sorted by id, so which member of a
    near-duplicate pair survives does not depend on dict ordering or on the order
    the batches came back in.

    Returns the kept queries and the report. Raises if ``gold`` is empty: an empty
    gold set means the guard did nothing, and that has to be impossible to miss.
    """
    import numpy as np

    if not 0.0 < threshold <= 1.0:
        raise SynthConfigError(f"threshold must be in (0, 1], got {threshold}")
    if not gold:
        raise GoldSetMissingError(
            "gold query list is empty, so the leakage guard would remove nothing. "
            "Refusing to pretend a dedup happened."
        )

    ordered = sorted(queries, key=lambda q: q.query_id)
    removed: list[DuplicatePair] = []
    n_exact_gold = n_near_gold = n_exact_internal = n_near_internal = 0

    # ── pass 1: exact matches against gold ──────────────────────────────────
    gold_by_norm: dict[str, GoldQuery] = {}
    for g in gold:
        gold_by_norm.setdefault(normalize_query_text(g.query), g)

    survivors: list[SyntheticQuery] = []
    for q in ordered:
        hit = gold_by_norm.get(normalize_query_text(q.query))
        if hit is not None:
            n_exact_gold += 1
            removed.append(
                DuplicatePair(
                    query_id=q.query_id,
                    query=q.query,
                    matched_id=hit.query_id,
                    matched_query=hit.query,
                    kind="exact_gold",
                )
            )
            continue
        survivors.append(q)

    # ── pass 2: near matches against gold ───────────────────────────────────
    if survivors:
        gold_matrix = _normalized_matrix(embed, [g.query for g in gold])
        kept: list[SyntheticQuery] = []
        for start in range(0, len(survivors), chunk_size):
            chunk = survivors[start : start + chunk_size]
            sims = _normalized_matrix(embed, [q.query for q in chunk]) @ gold_matrix.T
            for row, q in enumerate(chunk):
                best = int(np.argmax(sims[row]))
                score = float(sims[row][best])
                if score >= threshold:
                    n_near_gold += 1
                    removed.append(
                        DuplicatePair(
                            query_id=q.query_id,
                            query=q.query,
                            matched_id=gold[best].query_id,
                            matched_query=gold[best].query,
                            kind="near_gold",
                            cosine=score,
                        )
                    )
                    continue
                kept.append(q)
        survivors = kept

    # ── pass 3: duplicates inside the synthetic set ─────────────────────────
    # Why this matters even though it is not leakage: duplicate training queries
    # reweight the loss towards whatever they say, and the same text landing in two
    # different splits would mean the cross-encoder was tested on its own training
    # item. Exact first (free), then embedding, keeping the lowest id of each group.
    if dedupe_within and survivors:
        by_norm: dict[str, SyntheticQuery] = {}
        deduped: list[SyntheticQuery] = []
        for q in survivors:
            norm = normalize_query_text(q.query)
            first = by_norm.get(norm)
            if first is not None:
                n_exact_internal += 1
                removed.append(
                    DuplicatePair(
                        query_id=q.query_id,
                        query=q.query,
                        matched_id=first.query_id,
                        matched_query=first.query,
                        kind="exact_internal",
                    )
                )
                continue
            by_norm[norm] = q
            deduped.append(q)

        matrix = _normalized_matrix(embed, [q.query for q in deduped])
        kept_rows: list[int] = []
        for i, q in enumerate(deduped):
            if kept_rows:
                sims = matrix[i] @ matrix[kept_rows].T
                best_local = int(np.argmax(sims))
                score = float(sims[best_local])
                if score >= threshold:
                    other = deduped[kept_rows[best_local]]
                    n_near_internal += 1
                    removed.append(
                        DuplicatePair(
                            query_id=q.query_id,
                            query=q.query,
                            matched_id=other.query_id,
                            matched_query=other.query,
                            kind="near_internal",
                            cosine=score,
                        )
                    )
                    continue
            kept_rows.append(i)
        survivors = [deduped[i] for i in kept_rows]

    report = DedupReport(
        n_input=len(queries),
        n_kept=len(survivors),
        n_gold=len(gold),
        threshold=threshold,
        n_removed_exact_gold=n_exact_gold,
        n_removed_near_gold=n_near_gold,
        n_removed_exact_internal=n_exact_internal,
        n_removed_near_internal=n_near_internal,
        removed=tuple(removed),
    )
    log.info("%s", report.summary())
    return survivors, report


def dedupe_against_gold_from_config(
    queries: Sequence[SyntheticQuery],
    cfg: Config,
    *,
    embed: EmbedFn | None = None,
    gold_path: str | Path | None = None,
) -> tuple[list[SyntheticQuery], DedupReport]:
    """``dedupe_against_gold`` with the gold path, threshold and embedder from config.

    A thin wrapper so the guard's own function stays free of the config object and
    testable with literals, while no caller can accidentally run it with a
    different threshold than the one the README quotes.
    """
    gold = load_gold_queries(cfg.evaluation.gold_queries if gold_path is None else gold_path)
    return dedupe_against_gold(
        queries,
        gold,
        default_embed_fn(cfg) if embed is None else embed,
        threshold=cfg.synth.gold_dedup_cosine_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEAR-DUPLICATE CLUSTERING — what may not span a split boundary
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ClusterReport:
    """Which queries must stay on the same side of the split boundary.

    ``cluster_of`` maps every query id to its cluster's key (the lowest query id
    in the cluster, so the key does not depend on input order). It is fed straight
    to ``concierge.synth.splits`` as the grouping, which is what makes the
    out-of-fold guarantee hold for *paraphrases* and not only for identical ids.
    """

    cluster_of: Mapping[str, str]
    threshold: float
    n_queries: int
    n_clusters: int
    n_multi_query_clusters: int
    largest_cluster: int
    merged_same_business: bool

    def summary(self) -> str:
        return (
            f"near-duplicate clustering: {self.n_queries} queries in {self.n_clusters} "
            f"cluster(s) at cosine >= {self.threshold}"
            + (" (queries from one business merged)" if self.merged_same_business else "")
            + f"; {self.n_multi_query_clusters} cluster(s) hold more than one query, "
            f"largest holds {self.largest_cluster}"
        )


def cluster_near_duplicates(
    queries: Sequence[SyntheticQuery],
    embed: EmbedFn,
    *,
    threshold: float,
    merge_same_business: bool = True,
    chunk_size: int = 256,
) -> ClusterReport:
    """Group queries that must not be split apart: paraphrases of one another.

    **The leak this closes.** A query id is ``sha256(business_id, normalized
    text)``, and a split is a hash of that id. Two *paraphrases* — "quiet cafe
    with wifi" and "a quiet coffee shop with wifi" — normalize differently, so
    they get unrelated ids and unrelated splits. One can therefore land in
    ``crossencoder_train`` and the other in ``ltr_train``, which means the
    cross-encoder is fine-tuned on a query it then scores while LambdaMART is
    being trained. Every id-level out-of-fold check still passes, because at the
    level of ids nothing is wrong — and the cross-encoder feature is inflated
    exactly where LambdaMART's labels are. Grouping by cluster instead of by id
    is what makes that impossible rather than merely unlikely.

    **How the clusters are formed.** Union-find over three kinds of link:

    1. identical normalized text (free, and catches the same query generated for
       two different businesses);
    2. cosine >= ``threshold`` on the embeddings — the same measure and the same
       configured threshold the gold guard uses, because "near-duplicate" has to
       mean one thing in this project, and a second, lower threshold would be a
       number nobody has measured;
    3. same source business, when ``merge_same_business`` (the default). Queries
       written from one business document are the paraphrase risk at its most
       concentrated — four queries about the same place, from one prompt — and
       they also carry the *document*-level leak described under
       ``concierge.synth.splits.GroupBy``. Merging them costs only coarser split
       sizes and removes both risks at once. Pass ``False`` for clustering by
       query text alone.

    Union-find gives transitive closure (A~B, B~C puts A, B and C together even
    when A and C fall under the threshold), which is the conservative direction:
    an over-merged cluster shrinks the usable split sizes, an under-merged one
    leaks.

    ``embed`` is called even for a single query so a misshaped embedder fails here
    rather than at the split boundary.
    """
    import numpy as np

    if not 0.0 < threshold <= 1.0:
        raise SynthConfigError(f"threshold must be in (0, 1], got {threshold}")

    ordered = sorted(queries, key=lambda q: q.query_id)
    ids = [q.query_id for q in ordered]
    parent = {qid: qid for qid in ids}

    def find(qid: str) -> str:
        root = qid
        while parent[root] != root:
            root = parent[root]
        while parent[qid] != root:  # path compression
            parent[qid], qid = root, parent[qid]
        return root

    def union(a: str, b: str) -> None:
        # The smaller id always becomes the root, so a cluster's key is its lowest
        # member id no matter what order the links were discovered in.
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        low, high = sorted((root_a, root_b))
        parent[high] = low

    by_text: dict[str, str] = {}
    by_business: dict[str, str] = {}
    for q in ordered:
        norm = normalize_query_text(q.query)
        first_text = by_text.setdefault(norm, q.query_id)
        if first_text != q.query_id:
            union(first_text, q.query_id)
        if merge_same_business:
            first_business = by_business.setdefault(q.business_id, q.query_id)
            if first_business != q.query_id:
                union(first_business, q.query_id)

    if ordered:
        matrix = _normalized_matrix(embed, [q.query for q in ordered])
        for start in range(0, len(ordered), chunk_size):
            stop = min(start + chunk_size, len(ordered))
            # Only the lower triangle: every pair is considered exactly once, and
            # a query is never compared with itself (cosine 1.0, a false link).
            sims = matrix[start:stop] @ matrix[:stop].T
            for row in range(stop - start):
                i = start + row
                for j in np.nonzero(sims[row][:i] >= threshold)[0]:
                    union(ids[int(j)], ids[i])

    cluster_of = {qid: find(qid) for qid in ids}
    sizes = Counter(cluster_of.values())
    report = ClusterReport(
        cluster_of=cluster_of,
        threshold=threshold,
        n_queries=len(ids),
        n_clusters=len(sizes),
        n_multi_query_clusters=sum(1 for size in sizes.values() if size > 1),
        largest_cluster=max(sizes.values(), default=0),
        merged_same_business=merge_same_business,
    )
    log.info("%s", report.summary())
    return report


def cluster_near_duplicates_from_config(
    queries: Sequence[SyntheticQuery],
    cfg: Config,
    *,
    embed: EmbedFn | None = None,
    merge_same_business: bool = True,
) -> ClusterReport:
    """``cluster_near_duplicates`` with the configured threshold and embedder.

    Same reason as ``dedupe_against_gold_from_config``: no caller can run the
    clustering at a threshold other than the one the README quotes.
    """
    return cluster_near_duplicates(
        queries,
        default_embed_fn(cfg) if embed is None else embed,
        threshold=cfg.synth.gold_dedup_cosine_threshold,
        merge_same_business=merge_same_business,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sampling and persistence helpers
# ─────────────────────────────────────────────────────────────────────────────


def sample_documents(
    documents: Sequence[BusinessDocument],
    n: int,
    seed: int,
) -> list[BusinessDocument]:
    """Pick ``n`` documents reproducibly (sorted first, so input order is irrelevant).

    Sorting before sampling is what makes the sample a function of the seed alone.
    Without it, the same seed over the same businesses in a different order gives a
    different training set, and "fixed seed" would be decoration.
    """
    pool = sorted(documents, key=lambda d: d.business_id)
    if n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def write_queries_csv(queries: Sequence[SyntheticQuery], path: str | Path) -> Path:
    """Write the synthetic set as CSV, for eyeballing it in a spreadsheet.

    ``write_queries_jsonl`` produces the machine-readable artifact the pipeline
    consumes; this is for the owner to read a few hundred rows and sanity-check
    that the generator is producing queries a person would actually type. Nothing
    downstream reads this file, so a stale copy cannot contaminate a training run.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "query", "query_type", "business_id"])
        for q in sorted(queries, key=lambda q: q.query_id):
            writer.writerow([q.query_id, q.query, q.query_type.value, q.business_id])
    return p


def write_json_artifact(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write one stage's report as pretty JSON. The package's only JSON writer.

    Centralised for the same reason ``JsonlCache`` is the only JSONL writer:
    ``concierge.synth.judge`` must remain structurally incapable of writing a
    label file (see its module docstring and the AST test that enforces it), so
    its CLI delegates every write here instead of opening a file itself.

    Pretty-printed and key-sorted because these files are read by a human while
    writing the README, and diffed between runs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote %s", p)
    return p


def dedup_report_json(report: DedupReport) -> dict[str, Any]:
    """The leakage guard's numbers, in the shape written to ``dedup_report.json``.

    ``n_removed_gold`` is a required README figure, so the removed queries travel
    with the counts: a number in a README that cannot be traced back to the rows
    it came from is a number the owner cannot defend.
    """
    return {
        "summary": report.summary(),
        "n_input": report.n_input,
        "n_kept": report.n_kept,
        "n_gold": report.n_gold,
        "cosine_threshold": report.threshold,
        "n_removed_gold": report.n_removed_gold,
        "n_removed_exact_gold": report.n_removed_exact_gold,
        "n_removed_near_gold": report.n_removed_near_gold,
        "n_removed_internal": report.n_removed_internal,
        "n_removed_exact_internal": report.n_removed_exact_internal,
        "n_removed_near_internal": report.n_removed_near_internal,
        "removed": [
            {
                "query_id": pair.query_id,
                "query": pair.query,
                "matched_id": pair.matched_id,
                "matched_query": pair.matched_query,
                "kind": pair.kind,
                "cosine": pair.cosine,
            }
            for pair in report.removed
        ],
    }


def write_queries_jsonl(queries: Sequence[SyntheticQuery], path: str | Path) -> Path:
    """Write the deduplicated query set — the machine-readable training input.

    This file, not ``queries.jsonl``, is what hard-negative mining reads, which is
    the whole point: the un-deduplicated set is never on a training path, so the
    gold leakage guard cannot be skipped by running a later stage directly.

    Rewritten rather than appended (unlike ``JsonlCache``) because it is a
    *filtered view* of the store: appending would leave rows the guard removed on
    a previous run sitting in the file that feeds training.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for q in sorted(queries, key=lambda q: q.query_id):
            fh.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")
    log.info("wrote %d deduplicated queries to %s", len(queries), p)
    return p


def read_queries_jsonl(path: str | Path) -> list[SyntheticQuery]:
    """Read the deduplicated query set, or say which make target produces it."""
    p = Path(path)
    if not p.exists():
        raise SynthArtifactMissingError(
            f"no deduplicated synthetic queries at {p}. Run `make synth-generate` "
            "first: it generates the queries and runs the gold leakage guard, and "
            "this is the only query file the later stages are allowed to read."
        )
    out: list[SyntheticQuery] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(SyntheticQuery.from_json(json.loads(line)))
    if not out:
        raise SynthArtifactMissingError(
            f"{p} is empty: the leakage guard removed every query, or generation "
            "produced none. Nothing can be trained on this."
        )
    return out


def default_documents_path(cfg: Config) -> Path:
    """Where ``make data`` leaves the business documents.

    Resolved here rather than imported from ``concierge.retrieval.build_index``
    so that nothing in this package pulls in FAISS or torch at import time — the
    synthetic-data stage must stay runnable (and testable) without them.
    """
    processed = cfg.data.processed_dir
    if not processed.is_absolute():
        processed = REPO_ROOT / processed
    for name in ("documents.parquet", "documents.jsonl"):
        candidate = processed / name
        if candidate.exists():
            return candidate
    raise SynthArtifactMissingError(
        f"no business documents in {processed} (looked for documents.parquet and "
        "documents.jsonl). Run `make data` first, or pass --documents."
    )


def read_documents(path: str | Path) -> list[BusinessDocument]:
    """Read business documents from Parquet (what ``make data`` writes) or JSONL."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return read_documents_jsonl(p)
    if p.suffix == ".parquet":
        import polars as pl

        return [
            BusinessDocument(
                business_id=str(row["business_id"]),
                text=str(row["text"]),
                n_reviews_used=int(row.get("n_reviews_used", 0) or 0),
                n_tips_used=int(row.get("n_tips_used", 0) or 0),
                truncated=bool(row.get("truncated", False)),
            )
            for row in pl.read_parquet(p).to_dicts()
        ]
    raise SynthConfigError(f"unsupported documents format {p.suffix!r}: use .parquet or .jsonl")


def documents_text_map(documents: Sequence[BusinessDocument]) -> dict[str, str]:
    """``{business_id: text}`` — the shape the judge and the trainers consume."""
    return {d.business_id: d.text for d in documents}


def read_documents_jsonl(path: str | Path) -> list[BusinessDocument]:
    """Read ``{"business_id": ..., "text": ...}`` rows into ``BusinessDocument``.

    The CLI takes documents from a file rather than rebuilding them so that this
    module has no dependency on the document builder's API or on the Yelp download
    being present. Extra keys are ignored on purpose, so the document artifact can
    grow fields without breaking generation.
    """
    out: list[BusinessDocument] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                BusinessDocument(
                    business_id=str(row["business_id"]),
                    text=str(row["text"]),
                    n_reviews_used=int(row.get("n_reviews_used", 0)),
                    n_tips_used=int(row.get("n_tips_used", 0)),
                    truncated=bool(row.get("truncated", False)),
                )
            )
    return out


def main(
    argv: Sequence[str] | None = None,
    *,
    client: LlmJsonClient | None = None,
    embed: EmbedFn | None = None,
) -> int:
    """Stage 1 of 4: generate synthetic queries, then run the gold leakage guard.

        python -m concierge.synth.generate --config configs/base.yaml

    Writes, into ``artifacts/synth/<config fingerprint>/``:

    * ``queries.jsonl`` — everything generated (resumable store; not a training input);
    * ``queries_deduped.jsonl`` — the guard's output, and **the only query file the
      rest of the pipeline reads**;
    * ``queries_deduped.csv`` — the same rows, for skimming in a spreadsheet;
    * ``dedup_report.json`` — how many queries the guard removed, which is a
      required README number.

    Quota: the plan is printed and confirmed before any call is made, and batches
    already in the response cache are excluded from the count, so a resumed run
    asks about what it will actually spend (CLAUDE.md §10).

    ``client`` and ``embed`` are injection seams, not flags: they let the tests
    exercise this CLI end to end with a stub model and a hashed embedder. A CLI
    that could only be run with an API key and a 100 MB download would not be
    tested, and an untested entry point is how a pipeline ends up with guards
    nothing ever runs.
    """
    import argparse

    from concierge.config import load_config

    parser = argparse.ArgumentParser(description="Generate synthetic training queries.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--documents",
        default=None,
        help=(
            "business documents as .parquet or .jsonl "
            "(default: documents.parquet/.jsonl in the processed data dir)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample this many documents (default: cfg.synth.businesses_sampled)",
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="skip the quota confirmation prompt (states the call count regardless)",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help=(
            "generate only, writing no training input. The gold leakage guard is "
            "NOT optional: nothing downstream reads a file this mode produces."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="write under this root instead of the repo (tests, or a scratch dataset)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    root = None if args.artifacts_root is None else Path(args.artifacts_root)
    out_dir = synth_dir(cfg, root)

    documents = read_documents(
        default_documents_path(cfg) if args.documents is None else args.documents
    )
    n = cfg.synth.businesses_sampled if args.limit is None else args.limit
    sampled = sample_documents(documents, n, cfg.seed)

    run = generate_queries(
        sampled,
        cfg,
        client=client,
        store=GenerationStore.from_config(cfg, root),
        assume_yes=args.assume_yes,
    )
    print(run.summary())
    if run.stopped_early:
        # Not an error: the completed batches are on disk and a rerun resumes. But
        # the dataset is partial, and training on a partial dataset unknowingly is
        # indistinguishable from training on the whole one.
        print("Run again when quota resets to finish the remaining batches.")

    if args.skip_dedup:
        print(
            "WARNING: gold dedup skipped, so NO training input was written. "
            "Rerun without --skip-dedup (the cached batches cost nothing) before "
            "`make synth-negatives`."
        )
        return 0

    kept, report = dedupe_against_gold_from_config(run.queries, cfg, embed=embed)
    print(report.summary())
    write_json_artifact(dedup_report_json(report), out_dir / DEDUP_REPORT_FILENAME)
    write_queries_csv(kept, out_dir / QUERIES_DEDUPED_CSV_FILENAME)
    out = write_queries_jsonl(kept, out_dir / QUERIES_DEDUPED_FILENAME)
    print(f"wrote {len(kept)} deduplicated queries to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
