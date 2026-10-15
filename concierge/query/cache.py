"""On-disk cache for LLM query parses.

Why this exists at all:

* **Cost.** The Gemini free tier is a hard quota. Evaluation reruns the same ~150 gold
  queries on every ablation row; without a cache a single `make eval` would spend the
  day's quota re-parsing queries whose answers cannot have changed.
* **Latency.** Parse latency is a *reported number* in this project. A cached parse costs
  a local SQLite lookup instead of a network round trip, and the ablation reports the two
  separately (``ParseSource.CACHE`` vs ``ParseSource.LLM``) rather than averaging a
  warm-cache run into an LLM-latency claim.
* **Reproducibility.** Two eval runs over the same queries, config and prompt see the
  same parses, so a change in a metric is a change in the system and not LLM sampling
  noise.

Why the key is ``(provider, model, prompt_version, query)``:

A cache entry is only valid for the exact parser that produced it. Switching providers or
models is a different parser; so is editing the prompt. ``prompt_version`` is the most
dangerous of the four, because editing a prompt file leaves no other trace: a stale cache
would quietly serve old-prompt parses alongside new-prompt ones inside a single
experiment, and the resulting metric would describe neither parser. Including the prompt
version in the key makes that impossible — an edited prompt simply misses.

SQLite rather than a JSON file on disk: concurrent access. The FastAPI service is
multi-threaded and `make eval` may parse queries in parallel; SQLite gives atomic writes
and a real primary key for free, with no extra dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

# `payload` holds the validated parse JSON, not a rendered ParsedQuery: the mapping from
# payload to ParsedQuery is our own code and may change (bug fix, new field), and we do
# not want a mapping fix to require re-spending API quota. `answered_by` records which
# provider actually served the response, which can differ from the key's provider when the
# primary was rate-limited and the fallback answered (see D-notes in concierge/query/llm.py).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS parse_cache (
    key             TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    query_norm      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    llm_latency_ms  REAL,
    answered_by     TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS parse_cache_prompt_version ON parse_cache (prompt_version);
"""


def normalize_query(query: str) -> str:
    """Canonical form of a user query for cache-key purposes.

    Case and whitespace are folded because "Open Past 10PM" and "open past 10pm" are the
    same request to a language model, and paying twice for them is pure waste. Nothing
    else is touched: punctuation and word order can change meaning, and the gold set
    deliberately contains messy queries whose typos are the point.

    Consequence worth knowing: two queries that differ only in case share one entry, so
    the stored parse was produced from whichever spelling arrived first. The parse is
    case-independent, so this is a non-issue in practice — but it is the kind of detail
    that becomes confusing in a debugging session if it is undocumented.
    """
    return _WHITESPACE.sub(" ", query.strip()).casefold()


@dataclass(frozen=True, slots=True)
class CacheKey:
    """The four things that identify a parse. Change any one and the answer may change."""

    provider: str
    model: str
    prompt_version: str
    query: str

    @property
    def query_norm(self) -> str:
        return normalize_query(self.query)

    def digest(self) -> str:
        """Stable hash of the key fields.

        JSON-encodes the fields as a list before hashing so the boundaries between them
        are unambiguous: with a plain string join, a model name containing the separator
        could collide with a different (provider, model) pair. Cheap insurance against a
        bug that would be nearly impossible to diagnose from the symptoms.
        """
        canonical = json.dumps(
            [self.provider, self.model, self.prompt_version, self.query_norm],
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A stored parse plus the metadata needed to report on it honestly."""

    payload: dict[str, Any]
    llm_latency_ms: float | None
    answered_by: str | None
    created_at: str


class ParseCache:
    """SQLite-backed store of validated parse payloads.

    Thread-safe: one connection is shared and guarded by a lock. A connection per call
    would be simpler, but opening SQLite on every query adds latency to a number we
    report, and a connection per thread would leak handles in the server.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = 5.0) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is safe only because every statement below runs under
        # self._lock; the lock is what makes it safe, not the flag.
        self._conn = sqlite3.connect(self.path, timeout=timeout_s, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets the eval harness read while another thread writes instead of
            # blocking on a reader/writer lock. Not available for in-memory databases.
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── reads ────────────────────────────────────────────────────────────────────

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Return the stored parse for this key, or None on a miss.

        A row whose payload will not decode is treated as a miss and deleted: a corrupt
        entry (interrupted write, hand-edited database) should cost one API call, not
        poison every future run of that query.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, llm_latency_ms, answered_by, created_at "
                "FROM parse_cache WHERE key = ?",
                (key.digest(),),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            log.warning("discarding undecodable cache entry for provider=%s", key.provider)
            self.delete(key)
            return None
        if not isinstance(payload, dict):
            self.delete(key)
            return None
        return CacheEntry(
            payload=payload,
            llm_latency_ms=row["llm_latency_ms"],
            answered_by=row["answered_by"],
            created_at=row["created_at"],
        )

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM parse_cache").fetchone()[0])

    # ── writes ───────────────────────────────────────────────────────────────────

    def put(
        self,
        key: CacheKey,
        payload: dict[str, Any],
        *,
        llm_latency_ms: float | None = None,
        answered_by: str | None = None,
    ) -> None:
        """Store (or replace) the parse for this key."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO parse_cache "
                "(key, provider, model, prompt_version, query_norm, payload, "
                " llm_latency_ms, answered_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key.digest(),
                    key.provider,
                    key.model,
                    key.prompt_version,
                    key.query_norm,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    llm_latency_ms,
                    answered_by,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()

    def delete(self, key: CacheKey) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM parse_cache WHERE key = ?", (key.digest(),))
            self._conn.commit()

    def clear(self) -> None:
        """Drop every entry. Used by tests and by `make` targets that want a cold run."""
        with self._lock:
            self._conn.execute("DELETE FROM parse_cache")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── context manager ──────────────────────────────────────────────────────────

    def __enter__(self) -> ParseCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
