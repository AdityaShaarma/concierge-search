"""Stage 3 inference: the cross-encoder reranker.

WHY A CROSS-ENCODER — the justification for the whole cascade
─────────────────────────────────────────────────────────────
Stage 2 retrieves with a *bi-encoder*: the query and every business document are
embedded independently, and relevance is the dot product of two vectors that
never met. That independence is exactly what makes it fast — document vectors are
computed once, offline, and a query only has to be compared against them.

A cross-encoder gives up that independence. It concatenates the query and the
document into one sequence, ``[CLS] query [SEP] document [SEP]``, and runs them
through the transformer *together*, so every query token can attend to every
document token and vice versa. The model can therefore ask questions a dot
product cannot: does *this* phrase in the document actually answer *that* word in
the query, or does it merely sit in the same topic?

That mechanism is the standard motivation for cross-encoder reranking in the
retrieval literature, where the reported pattern is that cross-encoders rank a
shortlist better than bi-encoder similarity does. It is *not* a result of this
project, and it is not assumed here: whether it holds on our queries and our
documents is exactly what ablation rows 3 and 4 (cross-encoder, pretrained and
fine-tuned) measure against row 2 (hybrid retrieval alone) when ``make eval``
runs. Until those rows have numbers in them, the paragraph above is the argument
for *trying* a cross-encoder, not evidence that it earned its place. (The same
standard applies to cost, below.)

That same joint attention is also why a cross-encoder cannot be precomputed —
and this part is mechanical rather than empirical. The forward pass depends on
the query, so nothing can be cached offline and the cost is one transformer
pass *per (query, document) pair*. Scoring every business in the corpus per query is
therefore out of the question, while scoring a shortlist is affordable — how
affordable is a measurement this project owes (``eval/run_eval.py`` reports p95
per stage, and the quality-vs-latency curve sweeps ``candidates.rerank``), not
something to assert here. Hence the cascade: cheap-and-wide first,
expensive-and-narrow second. Every candidate count in ``configs/base.yaml``
exists because of this asymmetry.

Three practical consequences, all handled below:

1. **Truncation is a silent quality cap.** ``cfg.rerank.max_length`` (512 tokens)
   covers the query *and* the document. If business documents are longer than the
   budget, the tail is dropped before the model ever sees it — no error, no
   warning, just a lower ceiling on achievable quality. ``truncation_report()``
   measures exactly how much document text survives, so this is a number we can
   look at rather than a hope.

2. **Which query text to score.** We score ``ParsedQuery.semantic_text``, not the
   raw user string. Hard constraints ("open past 10pm", "under $20") are enforced
   arithmetically by the Stage 2 filters; leaving them in the text asks the model
   to judge something it cannot verify from prose and double-counts constraints
   that have already been applied. ``query_text_for()`` is a single named
   function so the eval harness can ablate that choice instead of arguing about it.

3. **Which weights are loaded.** The ablation table reports pretrained *and*
   fine-tuned cross-encoder rows, so a reranker that silently fell back to the
   base checkpoint would corrupt a headline result. The loaded source is recorded
   on ``weights_source`` and must be printed by anything that reports a number.

The scorer is injected behind a Protocol (``PairScorer``) so the whole reranking
contract — ordering, tie behaviour, top_k, score bookkeeping — is unit-tested with
a tiny deterministic fake and the test suite never downloads a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol, runtime_checkable

from concierge.config import REPO_ROOT, Config
from concierge.types import (
    RANK_CROSS_ENCODER,
    SCORE_CROSS_ENCODER,
    BusinessDocument,
    Candidate,
    ParsedQuery,
)

# Filenames inside the artifact directory. train.py writes them; this module and
# onnx_export.py read them, so they live here as the single definition.
CHECKPOINT_DIRNAME = "model"
RUN_METADATA_FILENAME = "run.json"

DocumentSource = Mapping[str, str] | Mapping[str, BusinessDocument] | Sequence[BusinessDocument]


# ─────────────────────────────────────────────────────────────────────────────
# Artifact locations
# ─────────────────────────────────────────────────────────────────────────────


def rerank_dir(cfg: Config, root: Path | None = None) -> Path:
    """Artifact directory for this exact config: ``artifacts/rerank/<fingerprint>``.

    Same convention as the retrieval indexes (``artifacts/dense/<fingerprint>``)
    so a result can always be traced back to the settings that produced it, and so
    changing ``cfg.rerank`` cannot quietly reuse a checkpoint trained under
    different hyperparameters.
    """
    base = root if root is not None else REPO_ROOT / "artifacts"
    return Path(base) / "rerank" / cfg.fingerprint()


def resolve_weights(cfg: Config, model_path: str | Path | None = None) -> tuple[str, str]:
    """Decide which weights to load, and label them.

    Returns ``(model_name_or_path, source_label)``. The label is carried all the
    way to whoever reports a metric, because "cross-encoder (pretrained)" and
    "cross-encoder (fine-tuned)" are two different rows of the ablation table and
    confusing them would be the worst kind of reporting error: one that looks like
    a result.
    """
    if model_path is not None:
        return str(model_path), f"explicit path {model_path}"
    checkpoint = rerank_dir(cfg) / CHECKPOINT_DIRNAME
    if (checkpoint / "config.json").exists():
        return str(checkpoint), "fine-tuned checkpoint"
    # No fine-tune for this config yet. Falling back to the pretrained
    # ms-marco checkpoint is the right behaviour (the pipeline still works, and
    # it is a legitimate ablation row), but it must be visible.
    return cfg.rerank.base_model, "pretrained base_model (no fine-tuned checkpoint found)"


def load_run_metadata(cfg: Config) -> dict[str, Any] | None:
    """Read the training-run metadata for this config, if a run exists."""
    path = rerank_dir(cfg) / RUN_METADATA_FILENAME
    if not path.exists():
        return None
    metadata: dict[str, Any] = json.loads(path.read_text())
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# The scorer boundary
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class PairScorer(Protocol):
    """Scores (query, document) pairs jointly.

    Three methods, not one, because truncation has to be *measurable*. The token
    budget and the token count of a text are tokenizer facts, so only the scorer
    can answer them; the reranker turns them into the diagnostic. A test fake
    implements all three in a few lines with whitespace tokens.
    """

    @property
    def max_length(self) -> int:
        """Total token budget for query + document + special tokens."""
        ...

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One relevance score per (query, document) pair, higher is better."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens this text costs, excluding special tokens."""
        ...

    def document_token_budget(self, query_text: str) -> int:
        """Tokens left for the document once the query and specials are paid for."""
        ...


class TransformersCrossEncoderScorer:
    """The real scorer: a HuggingFace sequence-classification model on CPU.

    Plain ``transformers`` rather than ``sentence_transformers.CrossEncoder`` for
    the same reason the training loop uses it (see ``train.py``): train and serve
    then share one model class, so a checkpoint saved by our trainer is loaded
    here by exactly the code path it was saved from. One fewer place for a
    train/serve mismatch to hide.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        # Imported lazily: `import transformers` pulls in torch, which costs
        # seconds of startup. Only a real scoring run should pay that, and the
        # test suite must never pay it at all.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self._model.eval()
        # CPU is the required serving path: the laptop that serves this has no
        # GPU. `device` stays settable so an offline batch job (e.g. generating
        # LambdaMART features over thousands of pairs) can use mps/cuda.
        self._device = torch.device(device)
        self._model.to(self._device)
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
        self.model_name_or_path = str(model_name_or_path)
        # Number of labels decides how a logit becomes a score; see _reduce().
        self._n_labels = int(self._model.config.num_labels)

    @property
    def max_length(self) -> int:
        return self._max_length

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text))

    def document_token_budget(self, query_text: str) -> int:
        specials = self._tokenizer.num_special_tokens_to_add(pair=True)
        return max(0, self._max_length - specials - self.count_tokens(query_text))

    def _reduce(self, logits: Any) -> list[float]:
        """Collapse the model head's output to one score per pair.

        ``ms-marco-MiniLM-L-6-v2`` has a single-logit head, so the logit *is* the
        relevance score. Some other cross-encoders ship a two-class head; for
        those the relevant-class logit is the score. Anything else we refuse to
        guess at, because a wrong reduction produces a plausible ranking that is
        quietly backwards.
        """
        if self._n_labels == 1:
            return [float(x) for x in logits[:, 0].tolist()]
        if self._n_labels == 2:
            return [float(x) for x in logits[:, 1].tolist()]
        raise ValueError(
            f"cross-encoder head has {self._n_labels} labels; this module only knows "
            "how to score 1-logit (regression-style) and 2-class heads"
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Batched joint scoring. Truncation is ``only_second``: see below."""
        if not pairs:
            return []
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(pairs), self._batch_size):
                batch = list(pairs[start : start + self._batch_size])
                encoded = self._tokenizer(
                    [query for query, _ in batch],
                    [document for _, document in batch],
                    padding=True,
                    truncation="only_second",
                    # "only_second" truncates the document, never the query.
                    # The default ("longest_first") would start eating the query
                    # once a document is long — losing part of the user's request
                    # is far worse than losing the tail of a review snippet.
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits
                scores.extend(self._reduce(logits.detach().cpu()))
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Truncation diagnostic
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TruncationStat:
    """How much of one document actually reached the model."""

    business_id: str
    document_tokens: int
    tokens_kept: int

    @property
    def truncated(self) -> bool:
        return self.tokens_kept < self.document_tokens

    @property
    def kept_fraction(self) -> float:
        """Share of the document's tokens the model saw.

        An empty document is reported as 1.0 (nothing was lost) rather than as a
        division by zero; it is a document-builder problem, not a truncation one.
        """
        if self.document_tokens == 0:
            return 1.0
        return self.tokens_kept / self.document_tokens


@dataclass(frozen=True, slots=True)
class TruncationReport:
    """Aggregate answer to "how much document text survives ``max_length``?".

    This exists because the alternative is invisible failure. ``documents.max_chars``
    is 1800 and ``rerank.max_length`` is 512 *tokens*; whether 1800 characters fit
    in 512 tokens depends on the tokenizer and on the text, and guessing is not a
    measurement. Call this on a real document sample and read the answer.
    """

    max_length: int
    query_tokens: int
    document_token_budget: int
    stats: tuple[TruncationStat, ...]

    @property
    def n_documents(self) -> int:
        return len(self.stats)

    @property
    def n_truncated(self) -> int:
        return sum(1 for s in self.stats if s.truncated)

    @property
    def fraction_truncated(self) -> float:
        return self.n_truncated / self.n_documents if self.stats else 0.0

    @property
    def mean_kept_fraction(self) -> float:
        return mean(s.kept_fraction for s in self.stats) if self.stats else 1.0

    @property
    def median_kept_fraction(self) -> float:
        return median(s.kept_fraction for s in self.stats) if self.stats else 1.0

    @property
    def min_kept_fraction(self) -> float:
        return min((s.kept_fraction for s in self.stats), default=1.0)

    def worst(self, n: int = 5) -> tuple[TruncationStat, ...]:
        """The documents that lost the most, for eyeballing what got cut."""
        return tuple(sorted(self.stats, key=lambda s: s.kept_fraction)[:n])

    def summary(self) -> str:
        """One-line, number-only summary. No verdict: whether a given amount of
        truncation costs quality is an eval question, not something to assert
        here."""
        return (
            f"truncation: {self.n_truncated}/{self.n_documents} documents clipped "
            f"({self.fraction_truncated:.1%}); document budget "
            f"{self.document_token_budget} tokens of max_length {self.max_length} "
            f"(query used {self.query_tokens}); kept fraction "
            f"mean {self.mean_kept_fraction:.2f} / median {self.median_kept_fraction:.2f} "
            f"/ min {self.min_kept_fraction:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Query and document normalization
# ─────────────────────────────────────────────────────────────────────────────


def query_text_for(parsed: ParsedQuery) -> str:
    """The string the cross-encoder scores for this query.

    ``semantic_text`` by design (module docstring, point 2). We fall back to
    ``raw`` only when the parser produced nothing usable — scoring every candidate
    against an empty string would make the whole stage a no-op that still costs
    full latency, which is worse than scoring slightly redundant text.
    """
    text = parsed.semantic_text.strip()
    return text if text else parsed.raw.strip()


def _document_texts(documents: DocumentSource) -> dict[str, str]:
    """Accept any of the three shapes callers naturally have, return id -> text."""
    if isinstance(documents, Mapping):
        return {
            business_id: (value if isinstance(value, str) else value.text)
            for business_id, value in documents.items()
        }
    return {doc.business_id: doc.text for doc in documents}


# ─────────────────────────────────────────────────────────────────────────────
# The reranker
# ─────────────────────────────────────────────────────────────────────────────


class CrossEncoderReranker:
    """Scores a Stage 2 shortlist jointly and reorders it.

    Holds no state that changes between calls: ``score`` and ``rerank`` are pure
    functions of their arguments. Diagnostics are a separate explicit call so that
    serving never pays for tokenization twice without being asked.
    """

    def __init__(self, scorer: PairScorer, *, weights_source: str = "injected scorer") -> None:
        self._scorer = scorer
        self.weights_source = weights_source

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        scorer: PairScorer | None = None,
        model_path: str | Path | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
    ) -> CrossEncoderReranker:
        """Build from config, loading a model only if no scorer is injected.

        ``batch_size`` defaults to ``cfg.rerank.batch_size``, which is really a
        *training* batch size (bounded by backprop memory). Inference has no
        gradients to store, so a serving deployment can safely raise it; it is a
        parameter rather than a second config key because the right value is a
        property of the machine, not of the experiment.
        """
        if scorer is not None:
            return cls(scorer, weights_source="injected scorer")
        name_or_path, source = resolve_weights(cfg, model_path)
        real = TransformersCrossEncoderScorer(
            name_or_path,
            max_length=cfg.rerank.max_length,
            batch_size=batch_size if batch_size is not None else cfg.rerank.batch_size,
            device=device,
        )
        return cls(real, weights_source=source)

    @property
    def max_length(self) -> int:
        return self._scorer.max_length

    # ── scoring ──────────────────────────────────────────────────────────────

    def score(self, query_text: str, documents: Sequence[str]) -> list[float]:
        """Score one query against many documents, in the order given."""
        pairs = [(query_text, document) for document in documents]
        scores = self._scorer.score_pairs(pairs)
        if len(scores) != len(pairs):
            # A scorer that silently returns a short list would shift every score
            # onto the wrong business — the kind of bug that still produces ten
            # plausible results.
            raise ValueError(f"scorer returned {len(scores)} scores for {len(pairs)} pairs")
        return [float(s) for s in scores]

    def truncation_report(self, query_text: str, documents: Sequence[str]) -> TruncationReport:
        """Measure how much of each document fits inside ``max_length``.

        Tokenization only — no forward passes — so this is cheap enough to run
        over a few thousand documents during the data build or the eval, which is
        where it belongs.
        """
        budget = self._scorer.document_token_budget(query_text)
        stats = []
        for index, document in enumerate(documents):
            length = self._scorer.count_tokens(document)
            stats.append(
                TruncationStat(
                    business_id=str(index),
                    document_tokens=length,
                    tokens_kept=min(length, budget),
                )
            )
        return TruncationReport(
            max_length=self._scorer.max_length,
            query_tokens=self._scorer.count_tokens(query_text),
            document_token_budget=budget,
            stats=tuple(stats),
        )

    def truncation_report_for_documents(
        self, query_text: str, documents: DocumentSource
    ) -> TruncationReport:
        """Same diagnostic, keyed by real business ids so ``worst()`` is usable."""
        texts = _document_texts(documents)
        report = self.truncation_report(query_text, list(texts.values()))
        stats = tuple(
            TruncationStat(
                business_id=business_id,
                document_tokens=stat.document_tokens,
                tokens_kept=stat.tokens_kept,
            )
            for business_id, stat in zip(texts.keys(), report.stats, strict=True)
        )
        return TruncationReport(
            max_length=report.max_length,
            query_tokens=report.query_tokens,
            document_token_budget=report.document_token_budget,
            stats=stats,
        )

    # ── reranking ────────────────────────────────────────────────────────────

    def rerank(
        self,
        parsed_query: ParsedQuery,
        candidates: Sequence[Candidate],
        documents: DocumentSource,
        top_k: int,
        *,
        on_missing_document: str = "error",
    ) -> tuple[Candidate, ...]:
        """Score every candidate jointly with its document and return the best ``top_k``.

        Contract, all of it pinned by tests in ``tests/test_rerank.py``:

        * Upstream scores and ranks are **preserved**. Stage 4 consumes BM25,
          dense and RRF scores as features, so this stage may only *add*
          ``SCORE_CROSS_ENCODER`` and ``RANK_CROSS_ENCODER``, never overwrite.
        * Ordering is a **stable** sort by descending cross-encoder score. Ties
          therefore keep the upstream (RRF) order, which is the only
          defensible tiebreak: it is a real signal, whereas dict or hash order is
          noise that would make the eval irreproducible.
        * ``RANK_CROSS_ENCODER`` is 1-based, matching the convention in
          ``fusion.py``.
        * ``top_k`` clamps the output; asking for more than exists is not an
          error, it just returns everything.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not candidates:
            return ()

        texts = _document_texts(documents)
        query_text = query_text_for(parsed_query)

        missing = [c.business_id for c in candidates if c.business_id not in texts]
        if missing and on_missing_document == "error":
            # The index and the document store are built from the same file, so a
            # candidate with no document means those two have drifted apart. That
            # is a build bug: fail now rather than serve a shortlist that quietly
            # lost members.
            raise KeyError(
                f"{len(missing)} candidate(s) have no document text, e.g. {missing[:3]}. "
                "The retrieval index and the document store disagree — rebuild both."
            )
        if missing and on_missing_document not in {"error", "rank_last"}:
            raise ValueError(
                f"on_missing_document must be 'error' or 'rank_last', got {on_missing_document!r}"
            )

        scorable = [c for c in candidates if c.business_id in texts]
        unscorable = [c for c in candidates if c.business_id not in texts]
        scores = self.score(query_text, [texts[c.business_id] for c in scorable])

        # Stable sort: `sorted` preserves input order among equal keys, and the
        # input order here is the upstream ranking.
        ordered = sorted(zip(scorable, scores, strict=True), key=lambda pair: -pair[1])
        # Unscorable candidates (only reachable with on_missing_document="rank_last")
        # go last and carry no cross-encoder score at all — a missing score is
        # honest, a zero would be a made-up measurement that Stage 4 would learn from.
        tail: list[tuple[Candidate, float | None]] = [(c, None) for c in unscorable]

        reranked: list[Candidate] = []
        for position, (candidate, score) in enumerate(list(ordered) + tail, start=1):
            if position > top_k:
                break
            reranked.append(_with_cross_encoder(candidate, score, position))
        return tuple(reranked)


def _with_cross_encoder(candidate: Candidate, score: float | None, rank: int) -> Candidate:
    """Copy a candidate, adding this stage's score and rank.

    Built by hand rather than via ``Candidate.with_score`` because that helper
    carries scores only, and Stage 4 uses the cross-encoder *rank* as a feature
    too.
    """
    scores = dict(candidate.scores)
    if score is not None:
        scores[SCORE_CROSS_ENCODER] = float(score)
    return Candidate(
        business_id=candidate.business_id,
        scores=scores,
        ranks={**candidate.ranks, RANK_CROSS_ENCODER: rank},
        constraints=candidate.constraints,
    )
