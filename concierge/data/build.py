"""Phase 1 step 3, the data build: raw Yelp JSON -> processed parquet.

    python -m concierge.data.build --config configs/base.yaml

What it produces, in `cfg.data.processed_dir`:

* `businesses.parquet` — one row per searchable business, with attributes and
  hours already parsed. This is what the hard-constraint filters read.
* `documents.parquet` — one text document per business, in the exact shape
  `concierge.retrieval.build_index` expects, so `make data && make index` works
  with nothing in between.
* `build_report.json` — what was counted, and how long each phase took.

Why parquet rather than keeping the JSON around: every later stage (document
builder, BM25, FAISS, reranker training, eval) reads this data repeatedly, and
re-parsing stringified attribute literals and hour strings on every run is both
slow and an invitation to two code paths disagreeing about what a value means.
Parsing happens once, here.

Two things are deliberately *not* done here:

* **No metro guessing.** `cfg.require_metro()` raises until the owner has picked
  a scope at the Phase 1 checkpoint. Building the wrong scope silently would
  invalidate every relevance label collected against it.
* **No estimates in the report.** Counts are counted, seconds are measured.
  Where a quantity is not measured (how many businesses the raw file holds in
  total, for instance — this build never reads the out-of-scope rows), the
  report omits it rather than approximating it.

Idempotent: a re-run recomputes everything and atomically replaces the outputs.
Document selection is deterministic for all three `review_selection` modes
(`random` is a hash of the review id, not a random draw), so two runs of the
same config produce byte-identical documents.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.table import Table

from concierge.config import REPO_ROOT, Config, load_config
from concierge.data.attributes import coverage, parse_attributes
from concierge.data.documents import build_documents
from concierge.data.hours import has_hours_data, parse_hours
from concierge.data.load import (
    MissingDatasetError,
    load_businesses,
    load_reviews_for,
    load_tips_for,
    parse_categories,
)
from concierge.types import Business, BusinessDocument, HoursInterval

BUSINESSES_FILE: Final = "businesses.parquet"
DOCUMENTS_FILE: Final = "documents.parquet"
REPORT_FILE: Final = "build_report.json"

#: How many attribute keys the printed coverage table shows. The full mapping
#: goes into the JSON report; the terminal gets the head of it.
_COVERAGE_ROWS: Final = 20


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────


def resolve_processed_dir(cfg: Config, root: Path | None = None) -> Path:
    """Absolute processed-data directory.

    Relative config paths resolve against the repo root — the same rule
    `load_config` and `concierge.data.load.raw_path` use — so a `make` target
    behaves identically whatever directory it was invoked from.
    """
    base = cfg.data.processed_dir
    if not base.is_absolute():
        base = (root if root is not None else REPO_ROOT) / base
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Record  ->  Business
# ─────────────────────────────────────────────────────────────────────────────


def business_from_record(record: Mapping[str, Any]) -> Business:
    """Build a `Business` from one raw `business.json` record.

    This is the single place raw records become the shared type, which is why
    it delegates all three awkward fields rather than handling them inline:
    `parse_attributes` owns the stringified-literal quirk, `parse_hours` owns
    the overnight-interval quirk, and `parse_categories` owns the
    comma-separated string. A second implementation of any of them would
    eventually disagree with the tests that pin them.
    """
    return Business(
        business_id=record["business_id"],
        name=record.get("name") or "",
        city=record.get("city") or "",
        state=record.get("state") or "",
        latitude=_as_float(record.get("latitude")),
        longitude=_as_float(record.get("longitude")),
        stars=float(record.get("stars") or 0.0),
        review_count=int(record.get("review_count") or 0),
        categories=parse_categories(record.get("categories")),
        attributes=parse_attributes(record.get("attributes")),
        hours=parse_hours(record.get("hours")),
        is_open=bool(record.get("is_open")),
    )


def _as_float(value: Any) -> float | None:
    """Coordinates stay None when absent — a 0.0 here would be a point in the
    Gulf of Guinea that every "near me" query would measure distance against."""
    if value is None or value == "":
        return None
    return float(value)


# ─────────────────────────────────────────────────────────────────────────────
# Business  <->  parquet
# ─────────────────────────────────────────────────────────────────────────────
#
# `attributes` and `hours` are stored as JSON strings rather than as nested
# parquet types. Attributes are an open-ended, heterogeneous mapping (bools,
# vocabulary strings, ints) whose key set differs per business, which a parquet
# struct cannot express without inventing a column per key; hours are a mapping
# of day to a variable-length list of intervals. One JSON string per field keeps
# the schema fixed and the round-trip exact, and these two columns are never
# filtered on in SQL — they are read back into Python objects immediately.


def _business_to_row(business: Business) -> dict[str, Any]:
    hours = {
        day: [[interval.start_min, interval.end_min] for interval in intervals]
        for day, intervals in business.hours.items()
    }
    return {
        "business_id": business.business_id,
        "name": business.name,
        "city": business.city,
        "state": business.state,
        "latitude": business.latitude,
        "longitude": business.longitude,
        "stars": business.stars,
        "review_count": business.review_count,
        "categories": list(business.categories),
        # sort_keys so a rebuild of unchanged data produces identical bytes.
        # No `default=` fallback: an attribute value that is not JSON-native
        # means the normalizer produced something unexpected, and that must
        # fail the build loudly rather than be silently stringified.
        "attributes_json": json.dumps(business.attributes, sort_keys=True),
        "hours_json": json.dumps(hours, sort_keys=True),
        "is_open": business.is_open,
    }


def _business_from_row(row: Mapping[str, Any]) -> Business:
    hours_raw: dict[str, list[list[int]]] = json.loads(row["hours_json"])
    return Business(
        business_id=row["business_id"],
        name=row["name"],
        city=row["city"],
        state=row["state"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        stars=float(row["stars"]),
        review_count=int(row["review_count"]),
        categories=tuple(row["categories"] or ()),
        attributes=json.loads(row["attributes_json"]),
        hours={
            day: tuple(HoursInterval(int(start), int(end)) for start, end in intervals)
            for day, intervals in hours_raw.items()
        },
        is_open=bool(row["is_open"]),
    )


def _business_schema() -> dict[str, Any]:
    """Explicit parquet schema.

    Spelled out rather than inferred because inference on a column that happens
    to be all-null (every business in a tiny scope missing coordinates, say)
    produces a Null-typed column, and the next run with one coordinate present
    would write a different schema for the same data.
    """
    import polars as pl

    return {
        "business_id": pl.Utf8,
        "name": pl.Utf8,
        "city": pl.Utf8,
        "state": pl.Utf8,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "stars": pl.Float64,
        "review_count": pl.Int64,
        "categories": pl.List(pl.Utf8),
        "attributes_json": pl.Utf8,
        "hours_json": pl.Utf8,
        "is_open": pl.Boolean,
    }


def _document_schema() -> dict[str, Any]:
    import polars as pl

    return {
        "business_id": pl.Utf8,
        "text": pl.Utf8,
        "n_reviews_used": pl.Int64,
        "n_tips_used": pl.Int64,
        "truncated": pl.Boolean,
    }


def _write_parquet(rows: Sequence[Mapping[str, Any]], schema: dict[str, Any], path: Path) -> Path:
    """Write rows to `path` atomically.

    Via a temporary file and `replace` so an interrupted build leaves the
    previous good parquet in place: a half-written index input is worse than a
    stale one, because it fails much later and much less obviously.
    """
    import polars as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(list(rows), schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)
    return path


def write_businesses(businesses: Sequence[Business], directory: Path) -> Path:
    if not businesses:
        raise ValueError("refusing to write an empty businesses.parquet")
    rows = [_business_to_row(b) for b in businesses]
    return _write_parquet(rows, _business_schema(), Path(directory) / BUSINESSES_FILE)


def write_documents(documents: Sequence[BusinessDocument], directory: Path) -> Path:
    if not documents:
        raise ValueError("refusing to write an empty documents.parquet")
    rows = [
        {
            "business_id": d.business_id,
            "text": d.text,
            "n_reviews_used": d.n_reviews_used,
            "n_tips_used": d.n_tips_used,
            "truncated": d.truncated,
        }
        for d in documents
    ]
    return _write_parquet(rows, _document_schema(), Path(directory) / DOCUMENTS_FILE)


def read_businesses(directory: Path) -> list[Business]:
    """Read `businesses.parquet` back into the shared type.

    Converted into dataclasses rather than handed out as a dataframe so a
    renamed column fails here, once, instead of as an attribute error deep
    inside a search request.
    """
    import polars as pl

    path = Path(directory) / BUSINESSES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no {BUSINESSES_FILE} in {directory}. Run `make data` for this config first."
        )
    return [_business_from_row(row) for row in pl.read_parquet(path).to_dicts()]


def read_documents(directory: Path) -> list[BusinessDocument]:
    """Read `documents.parquet` back into `BusinessDocument`s, in file order.

    Order matters: BM25 and FAISS address documents by row number, so the list
    order is the index's id mapping.
    """
    import polars as pl

    path = Path(directory) / DOCUMENTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no {DOCUMENTS_FILE} in {directory}. Run `make data` for this config first."
        )
    return [
        BusinessDocument(
            business_id=row["business_id"],
            text=row["text"],
            n_reviews_used=int(row["n_reviews_used"] or 0),
            n_tips_used=int(row["n_tips_used"] or 0),
            truncated=bool(row["truncated"]),
        )
        for row in pl.read_parquet(path).to_dicts()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# The build
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildSummary:
    """Everything this build measured. Nothing here is estimated.

    Written to `build_report.json` next to the parquet so any later number can
    be traced to the build that produced its inputs.
    """

    metro: str
    config_fingerprint: str
    n_businesses: int
    n_documents: int
    n_with_hours_data: int
    n_with_any_attributes: int
    n_review_rows_attached: int
    n_tip_rows_attached: int
    n_documents_truncated: int
    mean_document_chars: float
    seconds: dict[str, float]
    attribute_coverage: dict[str, float] = field(default_factory=dict)
    #: Set when `--limit` was used. Present so a sampled run can never be
    #: mistaken for a full one when the report is read later.
    limit_applied: int | None = None
    outputs: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())


def build(
    cfg: Config,
    *,
    limit: int | None = None,
    processed_dir: Path | None = None,
) -> BuildSummary:
    """Run the whole build and return what it measured.

    Separate from `main` so tests and notebooks can drive it without argv, and
    so the CLI layer only does argument parsing, printing and exit codes.
    """
    metro = cfg.require_metro()
    seconds: dict[str, float] = {}
    out_dir = resolve_processed_dir(cfg) if processed_dir is None else Path(processed_dir)

    with _phase(seconds, "load_businesses"):
        records = load_businesses(cfg, metro)
        if limit is not None:
            records = records[:limit]
        businesses = [business_from_record(r) for r in records]

    if not businesses:
        raise ValueError(
            f"no businesses found for metro={metro!r}. Check the metro name against "
            "`make explore` output — an empty scope is never a successful build."
        )

    business_ids = [b.business_id for b in businesses]

    with _phase(seconds, "load_reviews"):
        # Capped inside DuckDB at the document builder's own limit: anything
        # beyond `max_reviews` per business would be loaded and then thrown
        # away, and review.json is the 5 GB file.
        review_rows = load_reviews_for(
            cfg,
            business_ids,
            max_per_business=cfg.documents.max_reviews,
            order=cfg.documents.review_selection,
        )
        reviews_by_business = _group_by_business(review_rows)

    with _phase(seconds, "load_tips"):
        tip_rows = load_tips_for(cfg, business_ids, max_per_business=cfg.documents.max_tips)
        tips_by_business = _group_by_business(tip_rows)

    with _phase(seconds, "build_documents"):
        documents = build_documents(businesses, reviews_by_business, tips_by_business, cfg)

    with _phase(seconds, "write"):
        outputs = [
            str(write_businesses(businesses, out_dir)),
            str(write_documents(documents, out_dir)),
        ]

    summary = BuildSummary(
        metro=metro,
        config_fingerprint=cfg.fingerprint(),
        n_businesses=len(businesses),
        n_documents=len(documents),
        # Counted from the *raw* records: the parsed form cannot tell "no hours
        # field" from "hours we failed to parse", and the first is a dataset
        # fact while the second would be our bug.
        n_with_hours_data=sum(1 for r in records if has_hours_data(r.get("hours"))),
        n_with_any_attributes=sum(1 for b in businesses if b.attributes),
        n_review_rows_attached=len(review_rows),
        n_tip_rows_attached=len(tip_rows),
        n_documents_truncated=sum(1 for d in documents if d.truncated),
        mean_document_chars=sum(len(d.text) for d in documents) / len(documents),
        seconds=seconds,
        attribute_coverage=coverage(businesses),
        limit_applied=limit,
        outputs=outputs,
    )
    _write_report(summary, out_dir)
    return summary


def _group_by_business(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group review/tip rows by business, preserving the loader's order.

    Order is preserved because the loader already applied the configured
    selection order (useful-then-recent, recent, or a deterministic hash), and
    the document builder treats its input as pre-ranked.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["business_id"]].append(row)
    return dict(grouped)


@dataclass
class _Phase:
    """Tiny context manager recording wall-clock per build phase."""

    into: dict[str, float]
    name: str
    started: float = 0.0

    def __enter__(self) -> _Phase:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.into[self.name] = time.perf_counter() - self.started


def _phase(into: dict[str, float], name: str) -> _Phase:
    return _Phase(into, name)


def _write_report(summary: BuildSummary, directory: Path) -> Path:
    path = Path(directory) / REPORT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def render(summary: BuildSummary, console: Console) -> None:
    """Print what was built. Counts and seconds only — see the module docstring."""
    if summary.limit_applied is not None:
        console.print(
            f"[bold yellow]--limit {summary.limit_applied} was applied: this is a "
            f"sampled build, not the full metro.[/bold yellow]"
        )

    console.print(
        f"\n[bold]metro[/bold] {summary.metro}   [bold]config[/bold] {summary.config_fingerprint}"
    )

    counts = Table(title="counts", title_justify="left")
    counts.add_column("quantity")
    counts.add_column("value", justify="right")
    counts.add_row("businesses in scope (open only)", f"{summary.n_businesses:,}")
    counts.add_row("documents written", f"{summary.n_documents:,}")
    counts.add_row(
        "with any hours data",
        f"{summary.n_with_hours_data:,} ({_pct(summary.n_with_hours_data, summary.n_businesses)})",
    )
    counts.add_row(
        "with any attributes",
        f"{summary.n_with_any_attributes:,} "
        f"({_pct(summary.n_with_any_attributes, summary.n_businesses)})",
    )
    counts.add_row("review rows attached", f"{summary.n_review_rows_attached:,}")
    counts.add_row("tip rows attached", f"{summary.n_tip_rows_attached:,}")
    counts.add_row(
        "documents truncated to max_chars",
        f"{summary.n_documents_truncated:,} "
        f"({_pct(summary.n_documents_truncated, summary.n_documents)})",
    )
    counts.add_row("mean document length (chars)", f"{summary.mean_document_chars:,.0f}")
    console.print(counts)

    cov = Table(
        title=f"attribute coverage (top {_COVERAGE_ROWS} of {len(summary.attribute_coverage)})",
        title_justify="left",
    )
    cov.add_column("attribute")
    cov.add_column("share of businesses", justify="right")
    for key, fraction in list(summary.attribute_coverage.items())[:_COVERAGE_ROWS]:
        cov.add_row(key, f"{fraction * 100:.1f}%")
    console.print(cov)

    timing = Table(title="wall clock", title_justify="left")
    timing.add_column("phase")
    timing.add_column("seconds", justify="right")
    for name, value in summary.seconds.items():
        timing.add_row(name, f"{value:.2f}")
    timing.add_row("[bold]total[/bold]", f"[bold]{summary.total_seconds:.2f}[/bold]")
    console.print(timing)

    for path in summary.outputs:
        console.print(f"wrote {path}")
    console.print(f"wrote {Path(summary.outputs[0]).parent / REPORT_FILE}")


def _pct(part: int, whole: int) -> str:
    """Percentage, or an explicit marker when there is nothing to divide by."""
    if whole == 0:
        return "n/a"
    return f"{part / whole * 100:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m concierge.data.build",
        description="Parse raw Yelp JSON into the processed parquet the indexes read.",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build from the first N businesses only, for a fast smoke run. "
        "The report records that a limit was applied.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override cfg.data.processed_dir (useful for a scratch build).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    console = Console()
    cfg = load_config(args.config)

    try:
        summary = build(
            cfg,
            limit=args.limit,
            processed_dir=Path(args.out_dir) if args.out_dir else None,
        )
    except MissingDatasetError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2
    except ValueError as exc:
        # Covers the unset metro (`require_metro`) and an empty scope. Both are
        # the owner's decision to fix, so they get a message rather than a
        # traceback.
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    render(summary, console)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
