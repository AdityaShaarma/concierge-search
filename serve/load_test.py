"""Concurrent load test for the search API: p50/p95/p99 and throughput.

    uv run python -m serve.load_test \
        --url http://127.0.0.1:8000 \
        --queries path/to/queries.txt \
        --concurrency 1,4,8,16 \
        --requests 200

Writes a JSON record under `artifacts/load_test/` and prints the same numbers.
It measures; it never asserts a target. **No latency figure appears anywhere in
this file** — not in the help text, not in a docstring, not as a default — so
that no number in this repository can be mistaken for a measurement that was
never taken.

Design notes worth defending in an interview:

**Why asyncio and httpx rather than a load-testing tool.** The project's
constraint is free tools and a reviewable repo; `locust` or `k6` would add a
dependency and a config language to explain for one table. `httpx` is already a
dependency (the LLM client uses it) and asyncio makes "N requests in flight"
literal. The cost is that the client shares a GIL with its own measurement
loop, so at high concurrency some of what is reported as latency is client-side
scheduling. That is a real limitation and it is recorded in the output rather
than argued away.

**Why client-observed *and* server-reported latency.** Each response carries the
pipeline's own `timings.total_ms` and its per-stage breakdown. Reporting both
separates two different quantities: what a user waits (HTTP, queueing, the
threadpool, serialization) and what the cascade costs (the stage table). A
single end-to-end number cannot tell you which one to go fix, and the per-stage
p95 is the thing the quality-vs-latency curve is built from.

**Why nearest-rank percentiles.** `percentile` returns an observed sample, not
an interpolation between two of them. For a latency report that is the honest
choice: "95% of requests finished within X" should name a time some request
actually took. It also keeps this module free of numpy.

**Why warmup requests are excluded.** The first requests to a fresh process pay
one-time costs (lazy imports inside the stages, the first tokenizer allocation,
a cold query-parse cache). Including them would inflate the tail and understate
steady-state throughput; excluding them silently would be dishonest, so the
count is a flag and it is recorded in the output.

**Why the query list is a required argument.** There is no default query file: a
latency measurement is only interpretable together with the workload that
produced it, so the workload has to be named explicitly. Note that pointing it
at `eval/gold_queries.csv` is a *measurement* use of those queries — no labels
are read and nothing is trained — but it does warm the query-parse cache, which
would make a later run of the same queries faster. Use a separate list if that
matters.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import httpx

from concierge.config import REPO_ROOT

DEFAULT_OUT_DIR = Path("artifacts/load_test")
#: Below this many timed requests per level, the nearest-rank p99 *is* the
#: maximum observed sample. Reported as a caveat rather than hidden, because a
#: "p99" computed from a handful of requests is a max wearing a suit.
P99_MIN_SAMPLES = 100


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested in tests/test_serve.py)
# ─────────────────────────────────────────────────────────────────────────────


def percentile(samples: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest observed sample at or above `q`.

    `q` is a fraction in (0, 1]. Returns an element of `samples`, never an
    interpolation — see the module docstring. Raises on an empty input rather
    than returning 0.0, because a zero latency in a report would be read as a
    measurement.
    """
    if not samples:
        raise ValueError("percentile of no samples is undefined")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(samples)
    rank = math.ceil(q * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    """Latency summary for one set of samples, in milliseconds."""
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "max_ms": max(samples),
    }


def load_queries(path: str | Path) -> list[str]:
    """Read the workload from a `.csv` (a `query` column) or a text file.

    CSV support exists so the gold query file can be used directly; the `query`
    column is required by name rather than by position so a column reorder
    cannot silently load query *ids* as queries.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"query file not found: {p}")

    if p.suffix.lower() == ".csv":
        with p.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "query" not in reader.fieldnames:
                raise ValueError(f"{p} has no 'query' column; found {reader.fieldnames}")
            queries = [row["query"].strip() for row in reader if (row.get("query") or "").strip()]
    else:
        queries = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    if not queries:
        raise ValueError(f"{p} contains no queries")
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Sample:
    """One request's outcome.

    `client_ms` is wall-clock around the HTTP call. `server_total_ms` and
    `stage_ms` are the pipeline's own numbers, lifted from the response body, so
    the two views can be compared rather than conflated.
    """

    client_ms: float
    status: int
    ok: bool
    server_total_ms: float | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    n_results: int | None = None


@dataclass(slots=True)
class LevelResult:
    concurrency: int
    requested: int
    completed: int
    errors: int
    elapsed_s: float
    client: dict[str, float | int]
    server: dict[str, float | int]
    stages: dict[str, dict[str, float | int]]
    status_counts: dict[str, int]
    error_examples: list[str]
    results_returned: dict[str, float | int]

    @property
    def throughput_rps(self) -> float:
        """Successful requests per second over the level's wall-clock.

        Successful only: counting failures would make a service that 500s
        instantly look fast, which is the classic way to produce a flattering
        load-test number.
        """
        return self.completed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def to_json(self) -> dict[str, Any]:
        payload = {
            "concurrency": self.concurrency,
            "requested": self.requested,
            "completed": self.completed,
            "errors": self.errors,
            "elapsed_s": self.elapsed_s,
            "throughput_rps": self.throughput_rps,
            "client_latency": self.client,
            "server_reported_latency": self.server,
            "server_stage_latency": self.stages,
            "status_counts": self.status_counts,
            "results_returned": self.results_returned,
        }
        if self.error_examples:
            payload["error_examples"] = self.error_examples
        if 0 < self.completed < P99_MIN_SAMPLES:
            payload["caveat"] = (
                f"only {self.completed} timed requests: the nearest-rank p99 equals the "
                f"maximum observed sample at this sample size"
            )
        return payload


async def _one_request(client: httpx.AsyncClient, path: str, query: str, limit: int) -> Sample:
    started = time.perf_counter()
    try:
        response = await client.post(path, json={"query": query, "limit": limit})
    except Exception as exc:
        # A transport error is a data point, not a crash: a service that drops
        # connections under load must show up in the table as errors.
        return Sample(
            client_ms=(time.perf_counter() - started) * 1000.0,
            status=0,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    client_ms = (time.perf_counter() - started) * 1000.0

    if response.status_code != 200:
        return Sample(
            client_ms=client_ms,
            status=response.status_code,
            ok=False,
            error=f"HTTP {response.status_code}: {response.text[:200]}",
        )

    body = response.json()
    timings = body.get("timings") or {}
    return Sample(
        client_ms=client_ms,
        status=response.status_code,
        ok=True,
        server_total_ms=timings.get("total_ms"),
        stage_ms={s["stage"]: float(s["ms"]) for s in timings.get("stages", [])},
        n_results=len(body.get("results") or []),
    )


async def _run_level(
    base_url: str,
    queries: Sequence[str],
    *,
    concurrency: int,
    n_requests: int,
    limit: int,
    timeout_s: float,
    warmup: int,
    offset: int,
) -> LevelResult:
    """Fire `n_requests` with `concurrency` in flight, and time them.

    The workload is a deterministic cycle through the query list, so every
    concurrency level sees the same mix of queries in the same order — without
    that, a difference between two levels could be a difference in workload.
    `offset` advances the cycle for the warmup requests so the warmup and the
    timed run are not the same queries hitting the same parse cache.

    A fresh client per level, with its connection pool sized to the level, so
    pooled connections (and the cache state they imply) do not leak between
    levels.
    """
    plan = list(islice(cycle(queries), offset, offset + n_requests))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits) as client:
        # Warmup: dispatched at the same concurrency, results discarded.
        if warmup:
            warm_plan = list(islice(cycle(queries), 0, warmup))
            await _drain(client, warm_plan, concurrency, limit)

        started = time.perf_counter()
        samples = await _drain(client, plan, concurrency, limit)
        elapsed = time.perf_counter() - started

    ok = [s for s in samples if s.ok]
    failed = [s for s in samples if not s.ok]

    status_counts: dict[str, int] = {}
    for s in samples:
        key = str(s.status) if s.status else "transport_error"
        status_counts[key] = status_counts.get(key, 0) + 1

    server_samples = [s.server_total_ms for s in ok if s.server_total_ms is not None]
    stage_names = sorted({name for s in ok for name in s.stage_ms})
    stages = {
        name: summarize([s.stage_ms[name] for s in ok if name in s.stage_ms])
        for name in stage_names
    }
    counts = [float(s.n_results) for s in ok if s.n_results is not None]

    return LevelResult(
        concurrency=concurrency,
        requested=n_requests,
        completed=len(ok),
        errors=len(failed),
        elapsed_s=elapsed,
        client=summarize([s.client_ms for s in ok]),
        server=summarize(server_samples),
        stages=stages,
        status_counts=status_counts,
        error_examples=[e for e in (s.error for s in failed[:3]) if e],
        # Not a latency number, but it is the cheapest available check that the
        # service was doing real work: a level that got fast because it started
        # returning empty result sets is a broken deployment, not a win.
        results_returned=(
            {"n": len(counts), "mean": statistics.fmean(counts), "min": min(counts)}
            if counts
            else {"n": 0}
        ),
    )


async def _drain(
    client: httpx.AsyncClient, plan: Sequence[str], concurrency: int, limit: int
) -> list[Sample]:
    """Run `plan` with at most `concurrency` requests in flight.

    A queue with N consumer tasks rather than `asyncio.gather` over everything:
    gather would dispatch all requests at once and measure the server's accept
    backlog instead of its behaviour at a fixed concurrency.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()
    for query in plan:
        queue.put_nowait(query)

    samples: list[Sample] = []

    async def worker() -> None:
        while True:
            try:
                query = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            samples.append(await _one_request(client, "/search", query, limit))

    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    return samples


async def _fetch_health(base_url: str, timeout_s: float) -> dict[str, Any]:
    """Record what was under test.

    A latency number measured against a degraded deployment (no fine-tuned
    reranker, no LambdaMART) is a latency number for a *different system*. The
    health snapshot goes into the JSON so a result can never be read as
    belonging to the full system when it does not.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
            response = await client.get("/health")
            return {"http_status": response.status_code, "body": response.json()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}"


def print_report(record: dict[str, Any]) -> None:
    """Print exactly what was measured, with no derived claims."""
    health = record.get("health", {})
    body = health.get("body") or {}
    print(f"\nload test  {record['url']}  ({record['timestamp']})")
    print(f"workload   {record['queries_file']}  ({record['n_queries']} queries)")
    print(
        "service    "
        f"status={body.get('status', 'unknown')} variant={body.get('variant', 'unknown')} "
        f"config={body.get('config_fingerprint', 'unknown')}"
    )
    for line in body.get("warnings") or []:
        print(f"  ! degraded: {line}")

    header = (
        f"\n{'conc':>5} {'ok':>6} {'err':>5} {'rps':>8} "
        f"{'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   (client-observed ms)"
    )
    print(header)
    print("-" * (len(header) - 1))
    for level in record["levels"]:
        c = level["client_latency"]
        print(
            f"{level['concurrency']:>5} {level['completed']:>6} {level['errors']:>5} "
            f"{level['throughput_rps']:>8.1f} "
            f"{_fmt(c.get('p50_ms')):>8} {_fmt(c.get('p95_ms')):>8} "
            f"{_fmt(c.get('p99_ms')):>8} {_fmt(c.get('max_ms')):>8}"
        )
        if level.get("caveat"):
            print(f"      ! {level['caveat']}")
        for example in level.get("error_examples", []):
            print(f"      ! {example}")

    print("\nserver-reported pipeline time (ms), per concurrency level:")
    for level in record["levels"]:
        s = level["server_reported_latency"]
        if not s.get("n"):
            continue
        print(
            f"  conc={level['concurrency']:<4} total p50={_fmt(s.get('p50_ms'))} "
            f"p95={_fmt(s.get('p95_ms'))}"
        )
        for stage, stats in level["server_stage_latency"].items():
            print(
                f"      {stage:<10} p50={_fmt(stats.get('p50_ms')):>8} "
                f"p95={_fmt(stats.get('p95_ms')):>8}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _parse_concurrency(raw: str) -> list[int]:
    levels = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError(f"concurrency must be >= 1, got {value}")
        levels.append(value)
    if not levels:
        raise ValueError("no concurrency levels given")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serve.load_test",
        description="Measure p50/p95/p99 latency and throughput of the search API.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base URL of the service")
    parser.add_argument(
        "--queries",
        required=True,
        help="file of queries: .csv with a 'query' column, or one query per line",
    )
    parser.add_argument(
        "--concurrency",
        default="1,4,8,16",
        help="comma-separated concurrency levels to measure in order",
    )
    parser.add_argument(
        "--requests", type=int, default=200, help="timed requests per concurrency level"
    )
    parser.add_argument("--limit", type=int, default=10, help="results per request")
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed requests before each level (excluded from all statistics)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="per-request HTTP timeout, seconds"
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="directory for the JSON record",
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_queries(args.queries)
    levels = _parse_concurrency(args.concurrency)
    base_url = args.url.rstrip("/")

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": base_url,
        "queries_file": str(args.queries),
        "n_queries": len(queries),
        "requests_per_level": args.requests,
        "warmup_per_level": args.warmup,
        "limit": args.limit,
        "percentile_method": "nearest-rank (an observed sample, not an interpolation)",
        "client": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": (
                "single-process asyncio client; at high concurrency part of the "
                "client-observed latency is client-side scheduling"
            ),
        },
        "health": await _fetch_health(base_url, args.timeout),
        "levels": [],
    }

    results: list[LevelResult] = []
    for i, concurrency in enumerate(levels):
        print(f"… level {i + 1}/{len(levels)}: concurrency={concurrency}", file=sys.stderr)
        level = await _run_level(
            base_url,
            queries,
            concurrency=concurrency,
            n_requests=args.requests,
            limit=args.limit,
            timeout_s=args.timeout,
            warmup=args.warmup,
            # Offset the cycle per level so levels do not all start on query 0
            # after their warmup; the plan stays deterministic either way.
            offset=args.warmup,
        )
        results.append(level)
        record["levels"].append(level.to_json())

    if all(level.completed == 0 for level in results):
        record["measurement_valid"] = False
        record["measurement_note"] = (
            "every request failed: there are no latency numbers in this record"
        )
    return record


def write_record(record: dict[str, Any], out_dir: str | Path) -> Path:
    """Write the JSON record, plus a stable `latest.json` alongside it.

    Timestamped so a history of runs accumulates (a latency claim should be
    traceable to a run), and `latest.json` so the README generation step has a
    fixed path to read.
    """
    directory = Path(out_dir)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)

    stamp = record["timestamp"].replace(":", "").replace("-", "")
    path = directory / f"load_test_{stamp}.json"
    payload = json.dumps(record, indent=2, sort_keys=False)
    path.write_text(payload + "\n")
    (directory / "latest.json").write_text(payload + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = asyncio.run(run(args))
    print_report(record)
    path = write_record(record, args.out_dir)
    print(f"\nwrote {path}")
    # A run in which nothing succeeded must not exit 0: `make`-driven reporting
    # would otherwise carry on and publish a record with no numbers in it.
    return 0 if record.get("measurement_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
