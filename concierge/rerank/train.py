"""Fine-tune the Stage 3 cross-encoder.

    python -m concierge.rerank.train --config configs/base.yaml

WHY PLAIN ``transformers`` AND NOT ``sentence_transformers.CrossEncoder``
────────────────────────────────────────────────────────────────────────
Both were considered. ``CrossEncoder.fit()`` is three lines, but those three
lines hide the loss function, the batching, the scheduler and the evaluator
inside a Trainer whose defaults move between releases. The owner has to defend
this training loop in an interview, and "the library did it" is not a defence.
An explicit loop is maybe forty lines longer and every one of them is visible:
labels, forward, loss, backward, clip, step, schedule, validate.

It also removes a train/serve seam. ``model.py`` serves with
``AutoModelForSequenceClassification``, so training the *same* class means a
checkpoint is loaded at serving time by the code path that saved it.

THE OBJECTIVE
─────────────
Pointwise binary cross-entropy over a single logit: the positive business for a
query gets label 1, each mined hard negative gets label 0. This matches the head
``cross-encoder/ms-marco-MiniLM-L-6-v2`` already has — it was pretrained exactly
this way on MS MARCO — so fine-tuning continues its objective rather than
bolting a new one onto pretrained weights.

The alternative worth naming: a *listwise* objective (softmax cross-entropy over
one positive and its negatives in the same batch, as in ``MultipleNegativesRanking``
or LambdaRank-style losses). Listwise losses optimise the ordering directly,
which is what NDCG measures, and often win. They also require every group to sit
intact in one batch and make the loss harder to reason about. Pointwise BCE first;
switching is a contained change (``_loss_for_batch``) and should be judged by the
eval harness, not by assertion.

LEAKAGE
───────
Gold queries are test-only. If one reached this trainer the reported NDCG@10
would be inflated and the entire evaluation would be worthless — and nothing
about the number would look wrong. So the gold file is loaded here and any exact
(normalized) collision with a training query is a hard crash, not a warning. This
is the second line of defence; ``concierge/synth`` does the near-duplicate
embedding dedup. Defence in depth, because this failure is silent and fatal.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from concierge.config import REPO_ROOT, Config, load_config
from concierge.rerank.model import CHECKPOINT_DIRNAME, RUN_METADATA_FILENAME, rerank_dir

# Splits are produced by `concierge/synth` and read from disk here. They are NEVER
# re-derived: the LambdaMART agent's out-of-fold guarantee ("the cross-encoder
# feature comes from queries the cross-encoder never saw") only holds if both
# trainers read the same saved split. Re-splitting with the same seed would
# *look* safe and break the moment either side's data changed.
TRAIN_SPLIT = "crossencoder_train"
VAL_SPLIT = "crossencoder_val"


# Where to look for the synthetic data, in order. Config-fingerprinted directory
# first (artifacts are versioned by config), then the plain processed directory.
def _search_roots(cfg: Config) -> list[Path]:
    processed = cfg.data.processed_dir
    if not processed.is_absolute():
        processed = REPO_ROOT / processed
    return [
        REPO_ROOT / "artifacts" / "synth" / cfg.fingerprint(),
        REPO_ROOT / "artifacts" / "synth",
        processed / "synth",
        processed,
    ]


# How many real training steps to time before extrapolating a total runtime.
# Measured, then extrapolated — never a guessed constant.
_PROBE_STEPS = 20


class GoldLeakageError(RuntimeError):
    """Raised when a gold (test) query appears in training data."""


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SynthExample:
    """One query with its known-good business and its mined hard negatives.

    The contract with ``concierge/synth``: JSONL, one object per line, keys
    ``query``, ``positive_business_id``, ``negative_business_ids`` (and an
    optional ``query_id``).
    """

    query: str
    positive_business_id: str
    negative_business_ids: tuple[str, ...]
    query_id: str | None = None


def parse_example(row: dict[str, Any]) -> SynthExample:
    """Validate one JSONL row. Missing keys fail here, naming what was found.

    Strict on purpose: a renamed key that defaulted to empty would train on
    positives with no negatives and still report a falling loss.
    """
    try:
        query = str(row["query"]).strip()
        positive = str(row["positive_business_id"])
        negatives = tuple(str(n) for n in row.get("negative_business_ids") or ())
    except KeyError as exc:
        raise KeyError(
            f"synthetic example missing {exc.args[0]!r}; row has keys {sorted(row)}. "
            "Expected {'query', 'positive_business_id', 'negative_business_ids'}."
        ) from exc
    if not query:
        raise ValueError(f"synthetic example has an empty query: {row}")
    query_id = row.get("query_id")
    return SynthExample(
        query=query,
        positive_business_id=positive,
        negative_business_ids=negatives,
        query_id=str(query_id) if query_id is not None else None,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_split_file(split: str, cfg: Config, roots: Sequence[Path] | None = None) -> Path:
    """Locate one split's JSONL, or explain every place that was searched.

    Two layouts are accepted because the synth stage may reasonably write either:
    one file per split (``crossencoder_train.jsonl``), or a split subdirectory
    (``splits/crossencoder_train.jsonl``). Nothing is inferred beyond that — a
    trainer that guessed which file was the training split could silently train
    on the LambdaMART split and destroy the out-of-fold guarantee.
    """
    candidates: list[Path] = []
    for root in roots if roots is not None else _search_roots(cfg):
        candidates += [
            root / f"{split}.jsonl",
            root / "splits" / f"{split}.jsonl",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no synthetic data file for split {split!r}. Searched:\n  {searched}\n"
        "Generate it with the synth stage first; this trainer never re-splits."
    )


def load_split(split: str, cfg: Config, roots: Sequence[Path] | None = None) -> list[SynthExample]:
    path = find_split_file(split, cfg, roots)
    examples = [parse_example(row) for row in read_jsonl(path)]
    if not examples:
        raise ValueError(f"split {split!r} at {path} is empty")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Leakage guard
# ─────────────────────────────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Casefold and collapse whitespace before comparing two queries.

    Byte-exact comparison is a weaker guard than it looks: "Quiet cafe  with
    wifi" and "quiet cafe with wifi" are the same test query and byte equality
    would wave one of them through. Normalizing first makes the check strictly
    stronger while still being an *exact* match, which is what we want here —
    fuzzy near-duplicate removal is the synth stage's job and uses embeddings.
    """
    return _WHITESPACE.sub(" ", text.strip()).casefold()


def load_gold_queries(path: Path) -> list[str]:
    """Read the owner's gold query strings from ``eval/gold_queries.csv``."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "query" not in reader.fieldnames:
            raise ValueError(
                f"{path} has no 'query' column (found {reader.fieldnames}); "
                "cannot verify that training data is free of test queries"
            )
        return [row["query"] for row in reader if (row.get("query") or "").strip()]


@dataclass(frozen=True, slots=True)
class LeakageCheck:
    """Recorded in the run metadata: a run with no evidence of this check having
    passed should not be trusted."""

    gold_queries_path: str
    n_gold_queries: int
    n_training_queries: int
    n_collisions: int


def assert_no_gold_leakage(
    examples: Iterable[SynthExample], gold_path: Path, *, split: str = TRAIN_SPLIT
) -> LeakageCheck:
    """Hard-fail if any training query is a gold query.

    Deliberately offers no override flag. Every escape hatch here is a future
    inflated metric, and a loud crash costs the owner ten minutes while a quiet
    leak costs them the credibility of the whole project.

    A missing gold file is also a failure, not a skip: "we could not check" and
    "we checked and it was clean" must never look the same in the metadata.
    """
    if not gold_path.exists():
        raise FileNotFoundError(
            f"gold query file not found at {gold_path}. Training is refused: without it "
            "there is no way to prove the training data contains no test queries."
        )
    gold = {normalize_query(q) for q in load_gold_queries(gold_path)}
    training = [e.query for e in examples]
    collisions = sorted({q for q in training if normalize_query(q) in gold})
    if collisions:
        raise GoldLeakageError(
            f"{len(collisions)} query/queries in split {split!r} also appear in the gold "
            f"test set, e.g. {collisions[:3]}. Training would inflate every reported "
            "metric. Fix the synth stage's gold dedup; do not bypass this check."
        )
    return LeakageCheck(
        gold_queries_path=str(gold_path),
        n_gold_queries=len(gold),
        n_training_queries=len(training),
        n_collisions=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pair building
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LabeledPair:
    query: str
    document: str
    label: float  # 1.0 positive, 0.0 hard negative


@dataclass
class PairBuildStats:
    """Counts, so a shrinking dataset is visible instead of mysterious."""

    n_examples: int = 0
    n_examples_dropped_no_positive_doc: int = 0
    n_negatives_dropped_no_doc: int = 0
    n_positive_pairs: int = 0
    n_negative_pairs: int = 0

    @property
    def n_pairs(self) -> int:
        return self.n_positive_pairs + self.n_negative_pairs


def build_pairs(
    examples: Sequence[SynthExample], documents: dict[str, str]
) -> tuple[list[LabeledPair], PairBuildStats]:
    """Turn examples into labeled (query, document) pairs.

    A business id with no document cannot be scored by a text model, so it is
    dropped and counted rather than substituted with an empty string — training
    on empty documents teaches the model that blank text is irrelevant, which is
    true but useless, and it silently reweights the dataset.
    """
    pairs: list[LabeledPair] = []
    stats = PairBuildStats(n_examples=len(examples))
    for example in examples:
        positive_text = documents.get(example.positive_business_id)
        if positive_text is None:
            stats.n_examples_dropped_no_positive_doc += 1
            continue
        pairs.append(LabeledPair(example.query, positive_text, 1.0))
        stats.n_positive_pairs += 1
        for negative_id in example.negative_business_ids:
            negative_text = documents.get(negative_id)
            if negative_text is None:
                stats.n_negatives_dropped_no_doc += 1
                continue
            pairs.append(LabeledPair(example.query, negative_text, 0.0))
            stats.n_negative_pairs += 1
    if not pairs:
        raise ValueError(
            "no trainable pairs: none of the synthetic business ids have documents. "
            "Was the document build run for this metro/config?"
        )
    return pairs, stats


def load_document_texts(cfg: Config) -> dict[str, str]:
    """business_id -> document text, from the processed document file.

    Reuses the retrieval stage's loader rather than re-reading the file, so the
    reranker trains on byte-identical text to what retrieval indexes. If those
    two ever diverge, the reranker is being fine-tuned on documents it will never
    see at serving time.
    """
    from concierge.retrieval.build_index import find_documents_file, load_documents

    return {doc.business_id: doc.text for doc in load_documents(find_documents_file(cfg))}


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────


def detect_device(requested: str | None = None) -> str:
    """Pick cuda > mps > cpu, or honour an explicit request.

    mps (Apple Silicon) is tried before cpu because it is usually several times
    faster for a 6-layer model, but it is kept overridable: a few transformer ops
    still fall back or misbehave on mps depending on the torch build, and when
    that happens the fix is ``--device cpu``, not a code change.
    """
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Reproducibility is a stated project claim."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# The training loop
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpochRecord:
    epoch: int
    mean_train_loss: float
    val_loss: float | None
    val_mrr: float | None
    val_top1_rate: float | None
    seconds: float


@dataclass
class RunMetadata:
    """Everything needed to defend, reproduce, or distrust this run.

    No field is optional-by-convenience: a run whose metadata lacks the leakage
    check or the device is a run whose numbers cannot be used.
    """

    base_model: str
    config_fingerprint: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    max_length: int
    train_split: str
    val_split: str
    n_train_examples: int
    n_val_examples: int
    pair_stats: dict[str, Any]
    leakage_check: dict[str, Any]
    loss_curve: list[dict[str, Any]] = field(default_factory=list)
    measured_seconds_per_step: float | None = None
    estimated_total_seconds: float | None = None
    wall_clock_seconds: float | None = None
    checkpoint_dir: str | None = None
    # True when --limit-examples truncated the data: the checkpoint exists, but
    # nothing about it may be reported.
    is_smoke_run: bool = False
    notes: str = (
        "Loss values here are training diagnostics only. Retrieval quality "
        "(NDCG@10 on the gold set) is measured by eval/run_eval.py, never inferred "
        "from this curve."
    )


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _encode(
    tokenizer: Any, pairs: Sequence[tuple[str, str]], max_length: int, device: Any
) -> dict[str, Any]:
    """Tokenize a batch of pairs with the same truncation rule inference uses.

    ``truncation="only_second"`` keeps the whole query and clips the document.
    Train and serve MUST agree on this: training on full queries and serving
    truncated ones (or vice versa) is a distribution shift the model never sees
    in validation.
    """
    encoded = tokenizer(
        [q for q, _ in pairs],
        [d for _, d in pairs],
        padding=True,
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _loss_for_batch(
    model: Any, tokenizer: Any, batch: Sequence[LabeledPair], cfg: Config, device: Any
) -> Any:
    """Forward pass + loss for one batch.

    Single-logit head -> binary cross-entropy on the logit (the ms-marco head's
    own objective). Two-class head -> standard cross-entropy. Anything else is
    refused rather than guessed at, because a wrong head reduction trains
    perfectly happily towards the wrong thing.
    """
    import torch

    encoded = _encode(
        tokenizer, [(p.query, p.document) for p in batch], cfg.rerank.max_length, device
    )
    logits = model(**encoded).logits
    n_labels = int(model.config.num_labels)
    if n_labels == 1:
        targets = torch.tensor([p.label for p in batch], dtype=torch.float32, device=device)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)
    if n_labels == 2:
        targets = torch.tensor([int(p.label) for p in batch], dtype=torch.long, device=device)
        return torch.nn.functional.cross_entropy(logits, targets)
    raise ValueError(f"unsupported head with {n_labels} labels")


def _score_pairs(
    model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]], cfg: Config, device: Any
) -> list[float]:
    """Inference over pairs using the in-memory model.

    A local helper rather than ``model.CrossEncoderReranker`` because that loads
    weights from a path, and validation has to score the model as it currently
    stands in memory — mid-training, before any checkpoint exists.
    """
    import torch

    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in _batches(list(pairs), cfg.rerank.batch_size):
            logits = model(**_encode(tokenizer, batch, cfg.rerank.max_length, device)).logits
            column = 0 if int(model.config.num_labels) == 1 else 1
            scores.extend(float(x) for x in logits[:, column].detach().cpu().tolist())
    model.train()
    return scores


def validate(
    model: Any,
    tokenizer: Any,
    examples: Sequence[SynthExample],
    documents: dict[str, str],
    cfg: Config,
    device: Any,
) -> tuple[float, float, float]:
    """Validation loss plus two ranking diagnostics on ``crossencoder_val``.

    Returns ``(mean_loss, mrr, top1_rate)``.

    MRR is computed *within each query's own group* (its positive against its own
    mined negatives). That is a weaker, easier task than ranking the positive
    against the whole corpus, so these numbers are NOT comparable to the gold-set
    NDCG@10 the eval harness reports and must never be quoted as retrieval
    quality. They are here to answer one question during training: is the model
    learning to prefer the positive at all?
    """
    import torch

    val_pairs, _ = build_pairs(examples, documents)
    losses: list[float] = []
    for batch in _batches(val_pairs, cfg.rerank.batch_size):
        with torch.no_grad():
            model.eval()
            losses.append(float(_loss_for_batch(model, tokenizer, batch, cfg, device)))
            model.train()

    reciprocal_ranks: list[float] = []
    for example in examples:
        group_ids = [example.positive_business_id, *example.negative_business_ids]
        texts = [documents[i] for i in group_ids if i in documents]
        if example.positive_business_id not in documents or len(texts) < 2:
            # Nothing to rank: a group of one is trivially rank 1 and would
            # inflate MRR towards 1.0 for free.
            continue
        scores = _score_pairs(model, tokenizer, [(example.query, t) for t in texts], cfg, device)
        positive_score = scores[0]
        # Ties count against us (>=): if the model cannot separate the positive
        # from a negative, pretending it did would flatter the number.
        better = sum(1 for s in scores[1:] if s >= positive_score)
        reciprocal_ranks.append(1.0 / (1 + better))

    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else float("nan")
    top1 = (
        sum(1 for rr in reciprocal_ranks if rr == 1.0) / len(reciprocal_ranks)
        if reciprocal_ranks
        else float("nan")
    )
    return mean_loss, mrr, top1


def train(
    cfg: Config,
    *,
    device: str | None = None,
    limit_examples: int | None = None,
    out_dir: Path | None = None,
    save: bool = True,
    gold_path: Path | None = None,
    data_roots: Sequence[Path] | None = None,
    documents: dict[str, str] | None = None,
) -> RunMetadata:
    """Fine-tune the cross-encoder and return the run metadata.

    Order of operations matters: splits are loaded and the leakage check runs
    *before* a single weight is touched, so a leaky dataset costs seconds rather
    than an hour of GPU time and a retracted number.

    RUNNING ON COLAB (free T4) instead of the laptop: pip install the project,
    upload or mount ``data/processed`` (documents) and the synth split JSONLs plus
    ``eval/gold_queries.csv``, then run this module unchanged — ``detect_device``
    picks up cuda on its own. Only two things are worth changing there:
    ``rerank.batch_size`` (a T4 fits far more than 16) and copying
    ``artifacts/rerank/<fingerprint>/`` back down afterwards.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    started = time.perf_counter()
    set_seed(cfg.seed)

    train_examples = load_split(TRAIN_SPLIT, cfg, data_roots)
    val_examples = load_split(VAL_SPLIT, cfg, data_roots)
    if limit_examples is not None:
        # Smoke-test path. Deliberately truncates rather than subsamples so the
        # subset is reproducible without another seed.
        train_examples = train_examples[:limit_examples]
        val_examples = val_examples[: max(1, limit_examples // 4)]

    gold = gold_path if gold_path is not None else _gold_path(cfg)
    leakage = assert_no_gold_leakage(train_examples, gold, split=TRAIN_SPLIT)
    # The val split tunes decisions, so a gold query in there is leakage too.
    assert_no_gold_leakage(val_examples, gold, split=VAL_SPLIT)

    # `documents` is injectable so a smoke test (or a Colab run with the document
    # file in a different place) can supply the text directly instead of
    # depending on the processed-data layout. Nothing else about the run changes.
    texts = documents if documents is not None else load_document_texts(cfg)
    pairs, pair_stats = build_pairs(train_examples, texts)

    resolved_device = detect_device(device)
    torch_device = torch.device(resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.rerank.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.rerank.base_model)
    model.to(torch_device)
    model.train()

    steps_per_epoch = max(1, (len(pairs) + cfg.rerank.batch_size - 1) // cfg.rerank.batch_size)
    total_steps = steps_per_epoch * cfg.rerank.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rerank.lr)
    # Linear warmup then linear decay: the standard BERT fine-tuning schedule.
    # Warmup exists because a large first step on pretrained weights can destroy
    # them before the optimizer's moment estimates have settled.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.rerank.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    meta = RunMetadata(
        base_model=cfg.rerank.base_model,
        config_fingerprint=cfg.fingerprint(),
        seed=cfg.seed,
        device=resolved_device,
        epochs=cfg.rerank.epochs,
        batch_size=cfg.rerank.batch_size,
        learning_rate=cfg.rerank.lr,
        warmup_ratio=cfg.rerank.warmup_ratio,
        max_length=cfg.rerank.max_length,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        n_train_examples=len(train_examples),
        n_val_examples=len(val_examples),
        pair_stats=asdict(pair_stats),
        leakage_check=asdict(leakage),
        is_smoke_run=limit_examples is not None,
    )

    print(
        f"device={resolved_device}  pairs={pair_stats.n_pairs} "
        f"({pair_stats.n_positive_pairs} positive / {pair_stats.n_negative_pairs} negative)  "
        f"steps/epoch={steps_per_epoch}  epochs={cfg.rerank.epochs}\n"
        f"leakage check: {leakage.n_training_queries} training queries vs "
        f"{leakage.n_gold_queries} gold queries, {leakage.n_collisions} collisions",
        file=sys.stderr,
    )

    rng = random.Random(cfg.seed)
    step = 0
    probe_started = time.perf_counter()

    for epoch in range(1, cfg.rerank.epochs + 1):
        epoch_started = time.perf_counter()
        order = list(pairs)
        # Shuffle pairs (not examples): consecutive pairs from one query are
        # highly correlated, and a batch of nine negatives for the same query
        # gives a noisier gradient than a mixed batch.
        rng.shuffle(order)
        epoch_losses: list[float] = []

        for batch in _batches(order, cfg.rerank.batch_size):
            loss = _loss_for_batch(model, tokenizer, batch, cfg, torch_device)
            loss.backward()
            # Gradient clipping at 1.0: the usual BERT fine-tuning guard against
            # one pathological batch (e.g. a 512-token document of boilerplate)
            # taking a huge step.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_losses.append(float(loss.detach().cpu()))
            step += 1

            if step == _PROBE_STEPS:
                # Extrapolate from MEASURED step time. The point is to tell the
                # owner within a minute whether this run is a coffee break or a
                # reason to move to Colab — not to publish a benchmark.
                per_step = (time.perf_counter() - probe_started) / _PROBE_STEPS
                meta.measured_seconds_per_step = per_step
                meta.estimated_total_seconds = per_step * total_steps
                print(
                    f"  measured {per_step * 1000:.0f} ms/step over {_PROBE_STEPS} steps on "
                    f"{resolved_device} -> extrapolated "
                    f"{per_step * total_steps / 60:.1f} min for {total_steps} steps "
                    "(extrapolation, not a completed measurement)",
                    file=sys.stderr,
                )

        val_loss, val_mrr, val_top1 = validate(
            model, tokenizer, val_examples, texts, cfg, torch_device
        )
        record = EpochRecord(
            epoch=epoch,
            mean_train_loss=sum(epoch_losses) / len(epoch_losses),
            val_loss=val_loss,
            val_mrr=val_mrr,
            val_top1_rate=val_top1,
            seconds=time.perf_counter() - epoch_started,
        )
        meta.loss_curve.append(asdict(record))
        print(
            f"  epoch {epoch}: train_loss={record.mean_train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_group_mrr={val_mrr:.4f} "
            f"val_top1={val_top1:.3f} ({record.seconds:.1f}s)",
            file=sys.stderr,
        )

    meta.wall_clock_seconds = time.perf_counter() - started

    if save:
        # A smoke run's checkpoint goes to a side directory. `resolve_weights`
        # looks for `<artifact dir>/model`, so a checkpoint trained on fifty
        # pairs would otherwise be picked up silently by the serving path and the
        # eval — a fine-tuned-looking row produced by a run that trained on
        # nothing.
        default_dir = rerank_dir(cfg) / "smoke" if limit_examples is not None else rerank_dir(cfg)
        directory = out_dir if out_dir is not None else default_dir
        checkpoint = Path(directory) / CHECKPOINT_DIRNAME
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        # The tokenizer is saved alongside the weights because the two are a
        # unit: serving with a different vocabulary silently mis-tokenizes every
        # input.
        tokenizer.save_pretrained(checkpoint)
        meta.checkpoint_dir = str(checkpoint)
        (Path(directory) / RUN_METADATA_FILENAME).write_text(json.dumps(asdict(meta), indent=2))
        print(f"✓ checkpoint {checkpoint}", file=sys.stderr)

    return meta


def _gold_path(cfg: Config) -> Path:
    path = cfg.evaluation.gold_queries
    return path if path.is_absolute() else REPO_ROOT / path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the Stage 3 cross-encoder.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--device",
        default=None,
        help="cpu | mps | cuda. Default: best available (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="Train on the first N examples only — a smoke test, not a run to report.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Train without writing a checkpoint."
    )
    parser.add_argument("--out-dir", default=None, help="Override the artifact directory.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    meta = train(
        cfg,
        device=args.device,
        limit_examples=args.limit_examples,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        save=not args.no_save,
    )
    if args.limit_examples is not None:
        print(
            "NOTE: --limit-examples was set, so this checkpoint is a smoke test and its "
            "numbers must not be reported.",
            file=sys.stderr,
        )
    print(f"wall clock {meta.wall_clock_seconds:.1f}s on {meta.device}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
