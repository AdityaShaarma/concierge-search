"""Run the API over the test fixtures, so the UI is demonstrable without the dataset.

WHY THIS EXISTS
---------------
The real service needs `make data` and `make index`, which need the Yelp Open
Dataset. That is a correct dependency for the real system and a bad one for
development: it means nobody can look at the frontend, check a layout, or
confirm the API contract until a multi-gigabyte download has finished.

This serves the eight fixture businesses through the REAL FastAPI app and the
REAL pipeline — only the corpus is small. Nothing here is a mock: the same
parsers, documents, BM25 index, filters and constraint evaluation run.

It is BM25-only by default because a dense index would download the embedding
model. The service announces that itself: every response carries the degradation
warnings saying these are not hybrid-retrieval results, so a screenshot taken
from this server can never be mistaken for a measurement of the full system.

    uv run python scripts/demo_server.py [--port 8000] [--dense]
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

import uvicorn

from concierge.config import load_config
from concierge.data import documents as docs_mod
from concierge.data.attributes import parse_attributes
from concierge.data.hours import parse_hours
from concierge.pipeline import VARIANTS, SearchPipeline
from concierge.retrieval.bm25 import build_index
from concierge.types import Business

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _ndjson(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


def load_fixture_corpus(cfg):
    """Parse the fixtures through the real data layer."""
    reviews: dict[str, list[dict]] = collections.defaultdict(list)
    for row in _ndjson("reviews.json"):
        reviews[row["business_id"]].append(row)
    tips: dict[str, list[dict]] = collections.defaultdict(list)
    for row in _ndjson("tips.json"):
        tips[row["business_id"]].append(row)

    businesses = {}
    for raw in _ndjson("businesses.json"):
        if raw["is_open"] == 0:  # closed businesses never enter a searchable index
            continue
        businesses[raw["business_id"]] = Business(
            business_id=raw["business_id"],
            name=raw["name"],
            city=raw["city"],
            state=raw["state"],
            latitude=raw.get("latitude"),
            longitude=raw.get("longitude"),
            stars=raw["stars"],
            review_count=raw["review_count"],
            categories=tuple(
                c.strip() for c in (raw.get("categories") or "").split(",") if c.strip()
            ),
            attributes=parse_attributes(raw.get("attributes")),
            hours=parse_hours(raw.get("hours")),
            is_open=True,
        )
    documents = docs_mod.build_documents(list(businesses.values()), reviews, tips, cfg=cfg)
    return businesses, documents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--dense",
        action="store_true",
        help="also build a dense index (downloads the embedding model on first run)",
    )
    args = ap.parse_args()

    cfg = load_config("configs/base.yaml")
    businesses, documents = load_fixture_corpus(cfg)
    bm25 = build_index(documents, cfg=cfg)

    dense = None
    variant = VARIANTS["bm25"]
    if args.dense:
        from concierge.retrieval.dense import DenseEmbedder, DenseIndex

        dense = DenseIndex.build(documents, DenseEmbedder.from_config(cfg), cfg)
        variant = VARIANTS["hybrid_rrf"]

    # A fixed instant so "open late" resolves deterministically: a demo whose
    # results change with the wall clock is impossible to screenshot honestly.
    now = datetime(2026, 10, 16, 20, 0)

    def factory(loaded_cfg, _variant):
        return SearchPipeline(
            loaded_cfg,
            businesses=businesses,
            documents=documents,
            bm25_index=bm25,
            dense_index=dense,
            variant=variant,
            now=now,
        )

    from serve.app import create_app

    print(
        f"demo corpus: {len(businesses)} businesses, variant={variant.name}\n"
        f"  NOTE: fixture data, not the Yelp Open Dataset. Results here are a\n"
        f"  functional demonstration and are NOT a measurement of anything.\n"
        f"  http://{args.host}:{args.port}"
    )
    uvicorn.run(create_app(pipeline_factory=factory), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
