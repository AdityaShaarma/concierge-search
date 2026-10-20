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

**WHICH QUERY TEXT THE MINING RETRIEVES ON.** Not the raw synthetic query. The
same string Stage 2 retrieves on at serving time, which is the
constraint-stripped ``semantic_text``: the served pipeline calls
``query_text_for(parsed)`` (``concierge/pipeline.py``, ``_retrieve``), and
:func:`concierge.rerank.model.training_query_text` is the one function that
derives it from a raw query. Given "quiet cafe with wifi open past 10pm", the
rule parser lifts "quiet" into a NoiseLevel filter, "with wifi" into a WiFi
filter and "open past 10pm" into an hours filter, and what reaches the index is
the remainder — "cafe". Those constraints are enforced arithmetically by the
Stage 2 filters rather than by text matching, and asking a text model to re-judge
them double-counts them.

That matters here and not only in the trainer. Mining on the raw string draws
negatives from a candidate pool the served system never produces: the extra
constraint words act as a free relevance signal, so the top of the raw-query list
is cleaner than the top of the list the reranker will really be handed, and the
mined negatives are therefore systematically *easier* than the real ones. The
fine-tune then optimises a shortlist it never sees. Nothing errors, no validation
number moves, and the fine-tuned-vs-pretrained ablation delta stops measuring
fine-tuning alone — which is the one number this whole module exists to make
honest. The query text was fixed on the *training* side first; this is the mining
side of the same bug, and it is the same function on both sides so they cannot
drift apart again.

``TrainingExample.query`` keeps the **raw** query, deliberately: every downstream
consumer re-derives what it needs from it (the reranker trainer derives the
scored text, the LLM judge wants the full request including the constraints when
it decides whether a negative is secretly relevant). Storing the derived text
would double-apply the derivation for one consumer and silently withhold the
constraints from the other.

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

# ONE definition of "the text a stage scores", imported rather than re-derived.
# Mining has to agree with it *exactly*, and the only way to guarantee that is to
# call it: composing `query_text_for(parse_rules(...))` here instead would be a
# second definition of the same string, which is precisely the defect this import
# closes.
#
# It comes from `concierge.rerank.model`, not from the trainer. It was defined in
# the trainer when this import was first written, which pointed `synth` at
# `concierge.rerank.train` — and that module imports `concierge.synth.splits`, so
# the two packages referenced each other, resolving only because
# `concierge/synth/__init__.py` is empty and `splits` imports `negatives` inside
# functions rather than at module scope. The helper now lives beside
# `query_text_for` in `concierge.rerank.model`, which imports nothing from
# `concierge.synth`, so this edge is gone rather than merely surviving.
from concierge.rerank.model import training_query_text

# `build_pairs`/`SynthExample` are a genuine dependency on the trainer: it owns
# what a training pair *is* (label values, the missing-document policy), and
# `pairs_for_training` delegates to it rather than flattening examples a second
# time. Imported here rather than inside the function so that a future cycle
# fails loudly at import time, not mid-run inside a mining job.
from concierge.rerank.train import SynthExample, build_pairs
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

    **``verdict`` is the negative's label provenance, and it is why this record
    exists rather than a bare id.** ``concierge.synth.judge`` grades only the
    ``top_n`` hardest negatives of each example, so the negatives that survive
    into ``examples_filtered.jsonl`` are a *mixture*: some were put to the LLM
    judge and kept because it did not call them relevant, and the rest were never
    looked at. Downstream, both become label-0 rows in Stage 4's training file
    (``concierge.ltr.build_examples``, DECISIONS.md D227), and without this field
    nothing could tell the two apart — the evidence behind a 0 would be
    unrecoverable from the artifact. ``verdict`` carries the judge's grade, not
    just a flag, because *which* grade matters: a negative the judge graded 0
    ("irrelevant") is stronger evidence than one it graded 1 ("partially
    relevant"), and the filter keeps both (see the threshold in
    ``concierge.synth.judge``).

    ``judged`` is a property rather than a stored field on purpose. Two fields
    that must agree is an invariant something will eventually break; here
    "ungraded" has exactly one spelling, ``verdict is None``, and a record cannot
    claim to have been judged without carrying the grade that proves it.
    """

    business_id: str
    retrieval_rank: int
    retrieval_score: float
    source: str = "hybrid"
    #: The LLM judge's grade for this (query, business) pair, or None when this
    #: negative was never graded. The default is None because mining happens
    #: before any grading: a freshly mined negative has no verdict by definition.
    verdict: int | None = None

    @property
    def judged(self) -> bool:
        """Whether an LLM verdict stands behind this negative's label."""
        return self.verdict is not None

    def with_verdict(self, grade: int) -> HardNegative:
        """Copy carrying the judge's grade. Frozen type, so this is not in place."""
        return HardNegative(
            business_id=self.business_id,
            retrieval_rank=self.retrieval_rank,
            retrieval_score=self.retrieval_score,
            source=self.source,
            verdict=int(grade),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "business_id": self.business_id,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "source": self.source,
            # Written even when None. An explicit null is the record saying "never
            # graded", which is a different statement from a key that is absent
            # because the file predates the field — and `from_json` has to treat
            # the two the same way, so the artifact should not be ambiguous too.
            "verdict": self.verdict,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> HardNegative:
        """Read one negative, including from a file written before ``verdict``.

        A missing or null ``verdict`` reads as *not judged*, which is the
        conservative direction: an older ``examples_filtered.jsonl`` carries no
        grading provenance at all, and the one thing this must not do is let those
        negatives pass as judged. The consequence is visible rather than silent —
        ``concierge.ltr.build_examples`` reports how many label-0 rows rest on a
        verdict, so a run over an old file reports zero and says why.
        """
        verdict = row.get("verdict")
        return cls(
            business_id=str(row["business_id"]),
            retrieval_rank=int(row["retrieval_rank"]),
            retrieval_score=float(row["retrieval_score"]),
            source=str(row.get("source", "hybrid")),
            verdict=None if verdict is None else int(verdict),
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

    def with_verdicts(self, grades: Mapping[str, int]) -> TrainingExample:
        """Copy whose negatives carry the judge's grade, where there is one.

        The other half of what the false-negative filter does: it *removes* the
        negatives the judge called relevant (``without_negatives``) and *records
        the verdict* on the ones it kept, so a surviving negative's label can be
        traced to an actual grading call rather than to the absence of one
        (``HardNegative.verdict``).

        Negatives absent from ``grades`` are returned untouched — ungraded, which
        is what they are. An id in ``grades`` that is not a negative of this
        example is ignored rather than refused: the judge batches pairs across
        examples, and making every caller pre-split the mapping per example would
        move that filtering into the caller for no gain.
        """
        if not grades:
            return self
        return TrainingExample(
            query_id=self.query_id,
            query=self.query,
            query_type=self.query_type,
            positive_business_id=self.positive_business_id,
            negatives=tuple(
                n.with_verdict(grades[n.business_id]) if n.business_id in grades else n
                for n in self.negatives
            ),
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

    The retriever is handed :func:`training_query_text`, not ``query.query`` — the
    string the served Stage 2 retrieves on, for the reason in the module
    docstring. One consequence worth naming: ``positive_retrieval_rank`` is
    therefore a read on recall *as served*, which is the recall figure that means
    something, rather than recall for a string no user request ever becomes.
    """
    excluded = set(exclude) | {query.business_id}
    mining_text = training_query_text(query.query)
    results = list(retrieve(mining_text, pool_depth))
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
    would train the reranker on documents it will never be shown. Matching the
    served *depth* is only half of matching the served distribution; matching the
    served query *text* is the other half, and :func:`mine_for_query` does it.
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

    ``query_text`` arrives already derived — :func:`mine_for_query` strips the hard
    constraints before calling, exactly as the served pipeline does. This function
    must not parse it again: a second parse of an already-stripped string is a
    second definition of the scored text, and the two would be free to disagree.
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

    A **convenience shape** over :func:`concierge.rerank.train.build_pairs`, not a
    second pair builder. It used to be one, and it emitted ``example.query`` — the
    raw synthetic query — long after the trainer had switched to the
    constraint-stripped text serving scores. Nothing called it outside a test, so
    nothing caught the divergence; a dead function carrying a known skew is a
    loaded gun for whoever calls it next. It now delegates, so the query text, the
    label values and the missing-document policy all have exactly one definition,
    and ``tests/test_synth.py`` asserts the two agree row for row.

    What the delegate decides, stated so it is not a surprise: label 1.0 for the
    positive and 0.0 for each negative (binary, matching the head
    ``ms-marco-MiniLM`` was pretrained with); a business whose id is absent from
    ``documents`` is dropped and counted, because a text model cannot score a
    document that does not exist and substituting an empty string silently
    reweights the dataset. Mapping a graded 0/1/2 judge verdict onto a training
    label stays the trainer's decision, not this module's.

    One behaviour this *changed*, recorded rather than hidden: the old local
    implementation tested document text for truthiness, so a present-but-empty
    document counted as missing; ``build_pairs`` tests for absence, so an empty
    string is a (useless) pair. Whether an empty document is missing data or bad
    data is the trainer's call to make, and delegating means it is made in one
    place — but the two answers are not identical and the document builder is
    where an empty document should be prevented in the first place.

    Returns ``[]`` when the positive has no document. That is this function's
    per-example contract; ``build_pairs`` is a batch API and treats a batch with
    nothing trainable in it as a configuration error (``ValueError``), which is the
    right call for a whole training file and the wrong one for a single example.
    Hence the one pre-check below — a check, not a reimplementation.
    """
    if documents.get(example.positive_business_id) is None:
        log.warning(
            "no document for positive %s (query %s); example skipped",
            example.positive_business_id,
            example.query_id,
        )
        return []

    pairs, _ = build_pairs(
        [
            SynthExample(
                query=example.query,
                positive_business_id=example.positive_business_id,
                negative_business_ids=tuple(n.business_id for n in example.negatives),
                query_id=example.query_id,
            )
        ],
        dict(documents),
    )
    # int(label) rather than the float the trainer carries: this function's callers
    # (and its tests) read 0/1, and the two are the same value. Keeping the shape
    # is what lets the delegation be invisible to them.
    return [(pair.query, pair.document, int(pair.label)) for pair in pairs]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
