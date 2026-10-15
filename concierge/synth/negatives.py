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
