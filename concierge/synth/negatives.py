"""Hard-negative mining: the part that decides whether fine-tuning does anything.

A cross-encoder learns from contrast. For each synthetic query we already have a
positive for free (the business the query was written from); the question is what
to put beside it as a negative.

**Why hard negatives rather than random ones.** A random business — a tyre shop
for "quiet spot to read" — is separable on vocabulary alone. The model can score
it correctly using features it already had before fine-tuning, so the gradient is
near zero and we have spent a training step learning nothing. Worse, a model
trained on easy negatives is never asked the question that matters at serving
time: at serving time the reranker only ever sees the *top 200 of hybrid
retrieval*, i.e. documents that already look right. Its entire job is to separate
the genuinely right one from a shortlist of plausible ones, so that is the
distribution it has to be trained on. Hard negatives — other businesses that the
hybrid retriever ranked highly for this very query — are drawn from exactly that
distribution, which puts them near the decision boundary where the gradient is.

**The cost, stated honestly.** Mining from the top of the retrieval list
guarantees some "negatives" are actually relevant (other quiet cafés with wifi do
exist). Training on those teaches the model something false, and the harder the
negatives the more of them there are. That is what
:mod:`concierge.synth.judge` is for, and why hard-negative mining and
false-negative filtering are two halves of one decision rather than two features.

``sample_random_negatives`` is here so "hard beats random" can be *measured* on
the gold set instead of asserted from a paragraph like this one.

The retriever is injected (``RetrieveFn``), so every test runs against a tiny
deterministic stub — no index, no FAISS, no model download.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concierge.config import Config
from concierge.synth.generate import SyntheticQuery
from concierge.types import SCORE_RRF, Candidate, QueryType

log = logging.getLogger(__name__)

#: ``(query_text, top_k) -> [(business_id, score)]``, best first.
#:
#: Exactly the shape ``bm25.search`` and ``DenseIndex.search`` already return, so
#: the hybrid pipeline can be passed straight in and so a stub is three lines.
RetrieveFn = Callable[[str, int], Sequence[tuple[str, float]]]

# How much deeper than `hard_negatives_per_query` to retrieve. Oversampling is not
# a safety margin, it is a requirement: the LLM judge removes some negatives as
# false, and the positive itself usually occupies one of the top slots. Mining
# exactly N would leave the *most confidently retrieved* queries — the ones whose
# top results are most plausible and therefore most likely to be judged relevant —
# with the fewest negatives, which is precisely backwards.
DEFAULT_POOL_MULTIPLIER = 3


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HardNegative:
    """One negative, with the retrieval evidence that made it hard.

    ``retrieval_rank`` is kept because it is the only honest measure of "how hard"
    this negative was, and the judge uses it to decide which negatives are worth
    spending a grading call on (false negatives concentrate at the top).
    ``source`` distinguishes mined negatives from randomly sampled ones so the two
    training sets can never be mixed up in an ablation.
    """

    business_id: str
    retrieval_rank: int
    retrieval_score: float
    source: str = "hybrid"

    def to_json(self) -> dict[str, Any]:
        return {
            "business_id": self.business_id,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> HardNegative:
        return cls(
            business_id=str(row["business_id"]),
            retrieval_rank=int(row["retrieval_rank"]),
            retrieval_score=float(row["retrieval_score"]),
            source=str(row.get("source", "hybrid")),
        )


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One query with its positive and its negatives — a training group.

    A "group" in the LambdaMART sense too: Stage 4 needs queries as groups, and
    keeping the example as one object means the group boundary cannot drift
    between the two models' training sets.

    ``positive_retrieval_rank is None`` means the retriever never surfaced the
    source business for its own query. That is a real and interesting event (it
    bounds Stage 2 recall), so it is recorded rather than quietly dropped — the
    example is still trainable, because the positive label does not depend on
    retrieval finding it.
    """

    query_id: str
    query: str
    query_type: QueryType
    positive_business_id: str
    negatives: tuple[HardNegative, ...] = ()
    positive_retrieval_rank: int | None = None

    @property
    def n_negatives(self) -> int:
        return len(self.negatives)

    def without_negatives(self, business_ids: Iterable[str]) -> TrainingExample:
        """Copy with some negatives dropped (used by the false-negative filter).

        Frozen type, so filtering returns a new example rather than mutating one;
        the judge must not be able to edit a mined dataset in place.
        """
        drop = set(business_ids)
        return TrainingExample(
            query_id=self.query_id,
            query=self.query,
            query_type=self.query_type,
            positive_business_id=self.positive_business_id,
            negatives=tuple(n for n in self.negatives if n.business_id not in drop),
            positive_retrieval_rank=self.positive_retrieval_rank,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type.value,
            "positive_business_id": self.positive_business_id,
            "negatives": [n.to_json() for n in self.negatives],
            "positive_retrieval_rank": self.positive_retrieval_rank,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> TrainingExample:
        rank = row.get("positive_retrieval_rank")
        return cls(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            query_type=QueryType(row["query_type"]),
            positive_business_id=str(row["positive_business_id"]),
            negatives=tuple(HardNegative.from_json(n) for n in row.get("negatives", [])),
            positive_retrieval_rank=None if rank is None else int(rank),
        )


@dataclass(frozen=True, slots=True)
class MiningReport:
    """What mining produced, including the parts that look bad.

    ``n_positive_missed`` is the one to read first: it is the share of synthetic
    queries whose own source business was not in the retriever's pool, which is a
    direct (if synthetic) read on Stage 2 recall and an early warning that the
    documents or the retriever are weak.
    """

    n_queries: int
    n_requested_per_query: int
    pool_depth: int
    n_negatives_total: int
    n_queries_without_negatives: int
    n_positive_missed: int
    n_duplicate_business_ids_skipped: int

    @property
    def mean_negatives(self) -> float:
        return self.n_negatives_total / self.n_queries if self.n_queries else 0.0

    def summary(self) -> str:
        return (
            f"mined {self.n_negatives_total} hard negatives for {self.n_queries} queries "
            f"(mean {self.mean_negatives:.2f}, asked for {self.n_requested_per_query}, "
            f"pool depth {self.pool_depth}); "
            f"{self.n_queries_without_negatives} query/queries got none; "
            f"the retriever missed the source business for {self.n_positive_missed} "
            "query/queries"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Adapters
# ─────────────────────────────────────────────────────────────────────────────


def candidates_to_results(candidates: Sequence[Candidate]) -> list[tuple[str, float]]:
    """``[Candidate] -> [(business_id, fused score)]``, so ``fuse()`` output can be
    passed to a ``RetrieveFn``-shaped interface.

    Reads ``SCORE_RRF``, which is the slot the fused Stage 2 score always lands in
    (whichever fusion method produced it — see ``concierge.retrieval.fusion.fuse``).
    """
    return [(c.business_id, float(c.scores.get(SCORE_RRF, 0.0))) for c in candidates]


# ─────────────────────────────────────────────────────────────────────────────
# Mining
# ─────────────────────────────────────────────────────────────────────────────


def mine_for_query(
    query: SyntheticQuery,
    retrieve: RetrieveFn,
    n_negatives: int,
    pool_depth: int,
    *,
    exclude: Iterable[str] = (),
) -> tuple[TrainingExample, int]:
    """Mine negatives for one query. Returns the example and a duplicate count.

    The positive is removed from the pool rather than relied upon being absent:
    the source business *should* rank highly for a query written from its own
    document, and including it as a negative would be a direct label contradiction
    — the worst possible training row.
    """
    excluded = set(exclude) | {query.business_id}
    results = list(retrieve(query.query, pool_depth))
    negatives: list[HardNegative] = []
    seen: set[str] = set()
    positive_rank: int | None = None
    n_duplicates = 0

    for rank, (business_id, score) in enumerate(results, start=1):
        if business_id == query.business_id:
            positive_rank = rank if positive_rank is None else positive_rank
            continue
        if business_id in excluded:
            continue
        if business_id in seen:
            # A retriever that emits the same id twice (a fusion bug, or two
            # metros in one index) would otherwise contribute the same negative
            # twice and double its weight in the loss.
            n_duplicates += 1
            continue
        seen.add(business_id)
        negatives.append(
            HardNegative(
                business_id=business_id,
                retrieval_rank=rank,
                retrieval_score=float(score),
                source="hybrid",
            )
        )
        if len(negatives) >= n_negatives:
            break

    example = TrainingExample(
        query_id=query.query_id,
        query=query.query,
        query_type=query.query_type,
        positive_business_id=query.business_id,
        negatives=tuple(negatives),
        positive_retrieval_rank=positive_rank,
    )
    return example, n_duplicates


def mine_hard_negatives(
    queries: Sequence[SyntheticQuery],
    cfg: Config,
    retrieve: RetrieveFn,
    *,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
    exclude: Iterable[str] = (),
) -> tuple[list[TrainingExample], MiningReport]:
    """Mine ``cfg.synth.hard_negatives_per_query`` negatives for every query.

    ``exclude`` is for businesses that must never be a negative for a structural
    reason — e.g. anything outside the indexed metro, or a business already in the
    gold label pool. It is not a quality filter; relevance is the judge's job.

    The pool is capped at ``cfg.candidates.retrieve`` because that is the depth the
    served system actually produces: mining from deeper than Stage 2 ever returns
    would train the reranker on documents it will never be shown.
    """
    per_query = cfg.synth.hard_negatives_per_query
    pool_depth = min(max(per_query * max(1, pool_multiplier), per_query), cfg.candidates.retrieve)
    examples: list[TrainingExample] = []
    n_total = n_empty = n_missed = n_dupes = 0

    for query in queries:
        example, duplicates = mine_for_query(
            query, retrieve, per_query, pool_depth, exclude=exclude
        )
        n_dupes += duplicates
        n_total += example.n_negatives
        if example.n_negatives == 0:
            n_empty += 1
        if example.positive_retrieval_rank is None:
            n_missed += 1
        examples.append(example)

    report = MiningReport(
        n_queries=len(queries),
        n_requested_per_query=per_query,
        pool_depth=pool_depth,
        n_negatives_total=n_total,
        n_queries_without_negatives=n_empty,
        n_positive_missed=n_missed,
        n_duplicate_business_ids_skipped=n_dupes,
    )
    log.info("%s", report.summary())
    return examples, report


def sample_random_negatives(
    queries: Sequence[SyntheticQuery],
    business_ids: Sequence[str],
    cfg: Config,
    *,
    seed: int | None = None,
) -> list[TrainingExample]:
    """The control condition: negatives drawn uniformly from the whole corpus.

    This exists to make the hard-negative claim falsifiable. Train one
    cross-encoder on these and one on mined negatives, evaluate both on the gold
    set, and the difference is the measured value of hard-negative mining — the
    number that belongs in the README instead of a citation.

    Seeded per query (``seed:query_id``) rather than from one shared generator, so
    a query's negatives do not change when the query list is filtered or reordered.
    """
    pool = sorted(set(business_ids))
    base_seed = cfg.seed if seed is None else seed
    n = cfg.synth.hard_negatives_per_query
    out: list[TrainingExample] = []

    for query in queries:
        rng = random.Random(f"{base_seed}:{query.query_id}")
        choices = [b for b in pool if b != query.business_id]
        picked = rng.sample(choices, min(n, len(choices)))
        out.append(
            TrainingExample(
                query_id=query.query_id,
                query=query.query,
                query_type=query.query_type,
                positive_business_id=query.business_id,
                # Rank 0 and score 0.0: these were never retrieved, and faking a
                # rank would let a random negative masquerade as a mined one in
                # any downstream analysis that reads the rank.
                negatives=tuple(
                    HardNegative(
                        business_id=b, retrieval_rank=0, retrieval_score=0.0, source="random"
                    )
                    for b in picked
                ),
                positive_retrieval_rank=None,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def write_examples_jsonl(examples: Sequence[TrainingExample], path: str | Path) -> Path:
    """Write examples as JSONL, sorted by query id so the file is diffable."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for example in sorted(examples, key=lambda e: e.query_id):
            fh.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")
    return p


def read_examples_jsonl(path: str | Path) -> list[TrainingExample]:
    """Read examples back. Blank lines are skipped; anything else must parse."""
    out: list[TrainingExample] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(TrainingExample.from_json(json.loads(line)))
    return out


def report_over_examples(
    examples: Sequence[TrainingExample],
    report: MiningReport,
) -> MiningReport:
    """Re-state a mining report over a full example set, after a resumed run.

    ``mine_hard_negatives`` only sees the queries it was asked to mine, so on a
    resumed run its report describes the newly mined ones and silently omits
    everything already on disk — while ``mining_report.json`` is read as a
    description of the file beside it, and the negative count in it is a README
    figure. Recounting over the merged set keeps the reported numbers equal to the
    artifact they claim to describe.

    Two fields are carried over rather than recounted: ``pool_depth`` is derived
    from the config, not from the examples, and
    ``n_duplicate_business_ids_skipped`` is only observable while retrieving, so
    it necessarily covers this run's mining alone — an earlier run counted its own
    in the report it wrote.
    """
    return MiningReport(
        n_queries=len(examples),
        n_requested_per_query=report.n_requested_per_query,
        pool_depth=report.pool_depth,
        n_negatives_total=sum(e.n_negatives for e in examples),
        n_queries_without_negatives=sum(1 for e in examples if e.n_negatives == 0),
        n_positive_missed=sum(1 for e in examples if e.positive_retrieval_rank is None),
        n_duplicate_business_ids_skipped=report.n_duplicate_business_ids_skipped,
    )


def mining_report_json(report: MiningReport) -> dict[str, Any]:
    """The mining numbers, in the shape written to ``mining_report.json``.

    ``n_positive_missed`` is reported because it is a measurement, not a defect
    to hide: it is how often Stage 2 failed to retrieve the business a query was
    written from, i.e. a synthetic read on retrieval recall.
    """
    return {
        "summary": report.summary(),
        "n_queries": report.n_queries,
        "n_requested_per_query": report.n_requested_per_query,
        "pool_depth": report.pool_depth,
        "n_negatives_total": report.n_negatives_total,
        "mean_negatives_per_query": report.mean_negatives,
        "n_queries_without_negatives": report.n_queries_without_negatives,
        "n_positive_missed": report.n_positive_missed,
        "n_duplicate_business_ids_skipped": report.n_duplicate_business_ids_skipped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI (stage 2 of 4)
# ─────────────────────────────────────────────────────────────────────────────

#: Mined negatives, before the judge has removed the false ones.
EXAMPLES_MINED_FILENAME = "examples_mined.jsonl"
MINING_REPORT_FILENAME = "mining_report.json"


def hybrid_retrieve_fn(cfg: Config) -> RetrieveFn:
    """The real Stage 2 retriever (BM25 + dense, fused), as a ``RetrieveFn``.

    Mining has to use the *same* retriever the service runs, because the whole
    argument for hard negatives is that they come from the distribution the
    reranker will actually be shown. Negatives mined from BM25 alone would train
    the reranker to separate the positive from lexical near-misses, which is not
    the shortlist it sees at serving time.

    A missing index is a hard error naming ``make index``: falling back to the
    lexical half would quietly produce a different, weaker training set under the
    same filename, and nothing downstream could tell which one it got.
    """
    from concierge.retrieval import fusion
    from concierge.retrieval.bm25 import BM25Index, index_dir
    from concierge.retrieval.dense import dense_index_dir, load_dense_index
    from concierge.types import SCORE_BM25, SCORE_DENSE

    bm25_dir = index_dir(cfg)
    if not bm25_dir.exists():
        raise FileNotFoundError(
            f"no BM25 index at {bm25_dir}. Run `make index` for this config before "
            "mining hard negatives."
        )
    if not dense_index_dir(cfg).exists():
        raise FileNotFoundError(
            f"no dense index at {dense_index_dir(cfg)}. Run `make index` for this "
            "config: negatives must be mined from the hybrid retriever the service "
            "serves, not from BM25 alone."
        )
    bm25 = BM25Index.load(bm25_dir)
    dense = load_dense_index(cfg)

    def retrieve(query_text: str, top_k: int) -> list[tuple[str, float]]:
        systems = {
            SCORE_BM25: list(bm25.search(query_text, top_k)),
            SCORE_DENSE: list(dense.search(query_text, top_k)),
        }
        return candidates_to_results(fusion.fuse(systems, cfg, top_k=top_k))

    return retrieve


def main(argv: Sequence[str] | None = None, *, retrieve: RetrieveFn | None = None) -> int:
    """Stage 2 of 4: mine hard negatives for the deduplicated synthetic queries.

        python -m concierge.synth.negatives --config configs/base.yaml

    Reads ``queries_deduped.jsonl`` — the gold leakage guard's output, and the
    only query file this stage will read — and writes ``examples_mined.jsonl``
    plus ``mining_report.json`` beside it.

    **Resumable.** Mining spends no API quota but it does run two index searches
    per query, which is minutes over a few thousand queries. Examples already in
    ``examples_mined.jsonl`` for queries still in the input are reused, so an
    interrupted run continues instead of restarting. ``--refresh`` re-mines
    everything, which is what to use after rebuilding the indexes.

    The quota plan is still printed (zero calls), both because CLAUDE.md §10 asks
    every batch job to state its cost up front and because "this stage costs no
    quota" is worth seeing rather than assuming.

    ``retrieve`` is an injection seam so the tests can exercise this entry point
    against a deterministic stub retriever instead of a built index.
    """
    import argparse

    from concierge.config import load_config
    from concierge.synth.generate import (
        QUERIES_DEDUPED_FILENAME,
        QuotaPlan,
        read_queries_jsonl,
        require_confirmation,
        synth_dir,
        write_json_artifact,
    )

    parser = argparse.ArgumentParser(description="Mine hard negatives for synthetic queries.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-mine every query instead of reusing examples already on disk",
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="read and write under this root instead of the repo (tests, scratch runs)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    root = None if args.artifacts_root is None else Path(args.artifacts_root)
    out_dir = synth_dir(cfg, root)

    queries = read_queries_jsonl(out_dir / QUERIES_DEDUPED_FILENAME)
    mined_path = out_dir / EXAMPLES_MINED_FILENAME

    done: dict[str, TrainingExample] = {}
    if mined_path.exists() and not args.refresh:
        wanted = {q.query_id for q in queries}
        done = {e.query_id: e for e in read_examples_jsonl(mined_path) if e.query_id in wanted}
    todo = [q for q in queries if q.query_id not in done]

    require_confirmation(
        QuotaPlan(
            job="hard-negative mining",
            n_items=len(todo),
            items_label="queries to mine (local indexes only)",
            n_batches=0,
            n_cached_batches=0,
            n_api_calls=0,
            batch_size=1,
            note=f"{len(done)} query/queries already mined on disk; 0 API calls",
        )
    )
    print(f"mining {len(todo)} query/queries ({len(done)} reused from {mined_path.name})")

    examples, report = mine_hard_negatives(
        todo, cfg, hybrid_retrieve_fn(cfg) if retrieve is None else retrieve
    )
    merged = [*done.values(), *examples]
    write_examples_jsonl(merged, mined_path)
    # Recounted over the merged set, not just this run's: see report_over_examples.
    merged_report = report_over_examples(merged, report)
    write_json_artifact(mining_report_json(merged_report), out_dir / MINING_REPORT_FILENAME)
    print(merged_report.summary())
    print(f"wrote {len(merged)} training example(s) to {mined_path}")
    return 0


def pairs_for_training(
    example: TrainingExample,
    documents: Mapping[str, str],
) -> list[tuple[str, str, int]]:
    """Flatten one example into ``(query, document, label)`` rows, positive first.

    Label 1 for the positive and 0 for each negative — binary, because that is
    what a cross-encoder trained with a classification or ranking loss consumes.
    The graded 0/1/2 scale belongs to the gold labels and the judge; mapping a
    judge grade onto a training label is the *reranker trainer's* decision, not
    this module's.

    A business with no document is skipped rather than given an empty string: an
    empty document is a training row that teaches "this query matches nothing",
    which is a lesson about our data pipeline, not about relevance.
    """
    rows: list[tuple[str, str, int]] = []
    positive_doc = documents.get(example.positive_business_id)
    if positive_doc:
        rows.append((example.query, positive_doc, 1))
    else:
        log.warning(
            "no document for positive %s (query %s); example skipped",
            example.positive_business_id,
            example.query_id,
        )
        return []
    for negative in example.negatives:
        doc = documents.get(negative.business_id)
        if doc:
            rows.append((example.query, doc, 0))
    return rows


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
