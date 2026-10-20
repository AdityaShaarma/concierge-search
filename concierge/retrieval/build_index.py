"""Build the Stage 2 indexes: `python -m concierge.retrieval.build_index`.

One command builds both halves of the hybrid retriever (BM25 and FAISS) from the
same processed documents, in the same process. That matters more than it looks:
if the two indexes were built from different document sets, Reciprocal Rank
Fusion would still happily fuse their rankings and the resulting system would be
subtly broken in a way no test on either half could catch.

What this prints is deliberately limited to things it actually measured —
document count, wall-clock seconds, bytes on disk. No throughput estimates, no
projected query latency: retrieval latency is measured by the eval harness under
realistic conditions, not guessed at here.
"""

from __future__ import annotations

import argparse
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path

from concierge.config import REPO_ROOT, Config, load_config
from concierge.retrieval.dense import build_and_save as build_dense
from concierge.types import BusinessDocument

# Where `concierge.data` is expected to write the business documents. Parquet
# first (typed, compact, what the data stage writes); JSONL accepted as a
# fallback because it is the obvious hand-made format for a small debug corpus.
_DOC_FILENAMES = ("documents.parquet", "documents.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Loading documents
# ─────────────────────────────────────────────────────────────────────────────


def find_documents_file(cfg: Config) -> Path:
    """Locate the processed business documents, or explain what is missing."""
    processed = cfg.data.processed_dir
    if not processed.is_absolute():
        processed = REPO_ROOT / processed
    for name in _DOC_FILENAMES:
        candidate = processed / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no business documents found in {processed} (looked for "
        f"{', '.join(_DOC_FILENAMES)}). Run `make data` first."
    )


def load_documents(path: Path) -> list[BusinessDocument]:
    """Read documents into the shared `BusinessDocument` type.

    Converted into the dataclass rather than passed around as a dataframe so that
    a missing or renamed column fails here, once, instead of as an attribute
    error deep inside an embedding loop twenty minutes into a build.
    """
    if path.suffix == ".parquet":
        import polars as pl

        rows = pl.read_parquet(path).to_dicts()
    elif path.suffix == ".jsonl":
        import json

        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:  # pragma: no cover - guarded by find_documents_file
        raise ValueError(f"unsupported documents format: {path.suffix}")

    documents: list[BusinessDocument] = []
    for row in rows:
        try:
            documents.append(
                BusinessDocument(
                    business_id=row["business_id"],
                    text=row["text"],
                    n_reviews_used=int(row.get("n_reviews_used", 0) or 0),
                    n_tips_used=int(row.get("n_tips_used", 0) or 0),
                    truncated=bool(row.get("truncated", False)),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"{path} is missing required column {exc} (need business_id and text)"
            ) from exc
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# BM25 half
# ─────────────────────────────────────────────────────────────────────────────


def _build_bm25(documents: list[BusinessDocument], cfg: Config) -> tuple[int, Path]:
    """Build and save the BM25 index. Returns (document count, directory).

    Deliberately `build_index` + `save` rather than the module's `load_or_build`:
    `make index` means "build the index now". Silently reusing a cached one would
    make the reported build time meaningless and could leave a stale index in
    place after a change to the document builder.
    """
    from concierge.retrieval import bm25 as bm25_module

    index = bm25_module.build_index(documents, cfg)
    directory = index.save(bm25_module.index_dir(cfg))
    return index.n_docs, Path(directory)


# ─────────────────────────────────────────────────────────────────────────────
# Preflight: the faiss + torch OpenMP collision
# ─────────────────────────────────────────────────────────────────────────────


def duplicate_libomp_paths() -> list[Path]:
    """Return the distinct bundled `libomp.dylib` copies found in site-packages.

    Why this check exists. On macOS, the `faiss-cpu` and `torch` wheels each
    bundle their own copy of LLVM's OpenMP runtime. This build needs both
    libraries in one process (torch to embed, faiss to index), and the second
    runtime to initialize kills the interpreter with `OMP: Error #15` and
    `Abort trap: 6` — a C-level abort with no Python traceback, which is a
    miserable thing to debug from scratch.

    Measured on this machine: the abort happens whichever order the two are
    imported, and the documented `KMP_DUPLICATE_LIB_OK=TRUE` escape hatch
    replaced the abort with a deadlock (a trivial workload hung for over ten
    minutes). Pointing faiss's bundled copy at torch's — one runtime in the
    process — fixed it. See DECISIONS.md.

    Warn rather than fail: this is a heuristic on file layout, other platforms
    bundle differently, and a false positive must not block a build that works.
    """
    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidates = [
        site_packages / "faiss" / ".dylibs" / "libomp.dylib",
        site_packages / "torch" / "lib" / "libomp.dylib",
    ]
    # A symlink means someone already pointed one at the other, which is the fix.
    real = {c.resolve() for c in candidates if c.exists()}
    return sorted(real) if len(real) > 1 else []


def warn_on_duplicate_libomp() -> None:
    """Print the diagnosis up front, before a build that may take a long time."""
    copies = duplicate_libomp_paths()
    if not copies:
        return
    print(
        "WARNING: two separate OpenMP runtimes are installed:\n"
        + "\n".join(f"  {p}" for p in copies)
        + "\n  faiss and torch must share one process here. If this build dies with\n"
        "  'OMP: Error #15' / 'Abort trap: 6', point faiss's copy at torch's:\n"
        "    ln -sf <torch copy> <faiss copy>\n"
        "  Do NOT set KMP_DUPLICATE_LIB_OK=TRUE — measured on this machine it\n"
        "  deadlocks instead of aborting, which is strictly harder to diagnose.",
        file=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def directory_size_bytes(path: Path) -> int:
    """Total size of every file under `path`. Index size is a real cost: it
    bounds what can be loaded into a free-tier container at serve time."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_bytes(size: int) -> str:
    """Human-readable size. Binary units, labeled as such, so the printed number
    matches what `du -h` reports."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class BuildReport:
    """What was built, measured rather than assumed."""

    stage: str
    n_documents: int
    seconds: float
    directory: Path
    size_bytes: int

    def line(self) -> str:
        return (
            f"{self.stage:<6} {self.n_documents:>8} docs  "
            f"{self.seconds:>8.2f}s  {_format_bytes(self.size_bytes):>10}  "
            f"{self.directory}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# The document-truncation measurement
# ─────────────────────────────────────────────────────────────────────────────

#: The probe query the truncation measurement tokenizes against.
#:
#: The document budget is ``max_length`` minus the query's tokens minus specials,
#: so the number depends on the query. A short query would flatter it. This is a
#: deliberately ordinary one of the length the gold set actually contains, and it
#: is a constant so the reported number is reproducible rather than depending on
#: whatever string a caller passed.
TRUNCATION_PROBE_QUERY = "quiet cafe with free wifi open past 10pm near downtown"


def report_truncation(documents: list[BusinessDocument], cfg: Config) -> str | None:
    """Measure how much of each document fits in the cross-encoder's window.

    Why this runs here, inside ``make index``, and not behind a flag someone has
    to remember. ``concierge.rerank.model`` has measured this since it was
    written, and nothing called it: ``grep -rn truncation_report`` found hits only
    in that module and its tests — no make target, no training run, no build
    report, no eval path. Meanwhile the README described it as measured
    infrastructure ("model.py also measures how much of each document actually
    fits") and derived the document builder's head/tail ordering from it.

    ``concierge/pipeline.py`` states the rule this violated: "a measurement that
    only happens when someone remembers to ask for it is a measurement that will
    be missing from the README". Document truncation is a ceiling on achievable
    quality upstream of every model choice, so it belongs in the build report
    beside the document count.

    Tokenization only — no forward passes — so the cost is seconds over a whole
    metro. Returns ``None``, with a warning, if the tokenizer cannot be loaded:
    the indexes are the deliverable of this command and a missing diagnostic must
    not cost them.
    """
    try:
        from concierge.rerank.model import CrossEncoderReranker

        reranker = CrossEncoderReranker.from_config(cfg)
        report = reranker.truncation_report_for_documents(TRUNCATION_PROBE_QUERY, documents)
    except Exception as exc:  # broad: a diagnostic may never fail a build
        print(
            f"truncation measurement skipped ({type(exc).__name__}: {exc}). "
            "The indexes are unaffected; rerun once the cross-encoder tokenizer "
            "is available to get the number.",
            file=sys.stderr,
        )
        return None
    line = report.summary()
    # The comparison the number exists to make. `documents.max_chars` and
    # `rerank.max_length` are two caps in different units on the same text, and
    # which one binds is a tokenizer fact, not something to reason about.
    binding = (
        f"documents.max_chars={cfg.documents.max_chars} binds first"
        if report.fraction_truncated < 0.5
        else f"rerank.max_length={cfg.rerank.max_length} binds first"
    )
    return f"{line}; probe query {TRUNCATION_PROBE_QUERY!r}; {binding}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m concierge.retrieval.build_index",
        description="Build the BM25 and FAISS indexes over processed business documents.",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--only",
        choices=("both", "bm25", "dense"),
        default="both",
        help="Build one half only. Useful when iterating on the embedder.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Suppress the embedding progress line."
    )
    parser.add_argument(
        "--no-truncation-report",
        action="store_true",
        help=(
            "skip the document-truncation measurement (it loads the cross-encoder "
            "tokenizer and tokenizes every document; no forward passes)"
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # Refuses to run before the owner has picked a metro at the Phase 1
    # checkpoint. Indexing the wrong scope is expensive: every relevance label
    # collected against it has to be thrown away.
    metro = cfg.require_metro()
    warn_on_duplicate_libomp()

    documents_path = find_documents_file(cfg)
    documents = load_documents(documents_path)
    print(
        f"metro={metro}  config={cfg.fingerprint()}\n"
        f"{len(documents)} documents from {documents_path}",
        file=sys.stderr,
    )

    reports: list[BuildReport] = []

    if args.only in ("both", "bm25"):
        started = time.perf_counter()
        n_bm25, bm25_dir = _build_bm25(documents, cfg)
        reports.append(
            BuildReport(
                stage="bm25",
                n_documents=n_bm25,
                seconds=time.perf_counter() - started,
                directory=bm25_dir,
                size_bytes=directory_size_bytes(bm25_dir),
            )
        )

    if args.only in ("both", "dense"):
        started = time.perf_counter()
        index, dense_dir = build_dense(documents, cfg, progress=not args.no_progress)
        reports.append(
            BuildReport(
                stage="dense",
                n_documents=index.n_documents,
                seconds=time.perf_counter() - started,
                directory=dense_dir,
                size_bytes=directory_size_bytes(dense_dir),
            )
        )

    print("\nstage   documents      build        size  path")
    for report in reports:
        print(report.line())

    if not args.no_truncation_report:
        truncation = report_truncation(documents, cfg)
        if truncation is not None:
            print(f"\n{truncation}")
    # Both halves must have indexed the same documents, or RRF would fuse two
    # rankings over different corpora and no test on either half would notice.
    counts = {r.n_documents for r in reports}
    if len(counts) > 1:
        raise RuntimeError(f"index halves disagree on document count: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
