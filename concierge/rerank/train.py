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

WHICH QUERY TEXT IS SCORED
──────────────────────────
The same string serving scores. ``model.query_text_for(parsed)`` defines that —
it is ``ParsedQuery.semantic_text``, the request with the hard constraints
stripped out, because those are enforced arithmetically by the Stage 2 filters
and asking a text model to re-judge them double-counts them.

So training cannot be fed the raw synthetic query: it would fine-tune on
"quiet cafe with wifi open past 10pm" and then be served "quiet cafe". Nothing
errors, no validation number moves, and the fine-tuned-vs-pretrained ablation
delta stops being a measurement of fine-tuning — it measures fine-tuning plus a
distribution shift. :func:`training_query_text` is therefore the one function
that derives the scored string here, and ``tests/test_rerank.py`` asserts it
equals what the serving path computes for the same query.

LEAKAGE
───────
Gold queries are test-only. If one reached this trainer the reported NDCG@10
would be inflated and the entire evaluation would be worthless — and nothing
about the number would look wrong. So the gold file is loaded here and any exact
(normalized) collision with a training query is a hard crash, not a warning. This
is the second line of defence; ``concierge/synth`` does the near-duplicate
embedding dedup. Defence in depth, because this failure is silent and fatal.

The out-of-fold guarantee Stage 4 depends on ("LambdaMART's cross-encoder
feature comes from queries the cross-encoder never saw") needs this trainer to
prove *which* queries it fitted. It does that by loading the fingerprinted split
manifest through ``concierge.synth.splits`` — checksum, partition, seed and
fractions all verified — and requiring every training row's ``query_id`` to be
assigned to the split being loaded. The manifest's checksum is then recorded in
the run metadata, which is what lets ``concierge.ltr.train`` check that both
models were trained under the same split rather than hoping they were. Recorded
beside it is whether that manifest carried a config fingerprint at all, so
"tied to a configuration" and "there was nothing to tie it to" stay
distinguishable in the artifact instead of both reading as a clean check.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from concierge.config import REPO_ROOT, Config, load_config
from concierge.data.hours import DAYS
from concierge.query.rules import parse_rules
from concierge.rerank.model import (
    CHECKPOINT_DIRNAME,
    RUN_METADATA_FILENAME,
    query_text_for,
    rerank_dir,
)
from concierge.synth import splits as synth_splits

# Splits are produced by `concierge/synth` and read from disk here. They are NEVER
# re-derived: the LambdaMART agent's out-of-fold guarantee ("the cross-encoder
# feature comes from queries the cross-encoder never saw") only holds if both
# trainers read the same saved split. Re-splitting with the same seed would
# *look* safe and break the moment either side's data changed.
#
# The names come from `concierge.synth.splits`, which owns the split vocabulary
# and the manifest format. Re-spelling them here is how a renamed split turns
# into a check that silently looks at an empty set.
TRAIN_SPLIT, VAL_SPLIT = synth_splits.CROSSENCODER_SPLITS


# Where to look for the synthetic data. ONE root: the config-fingerprinted
# artifacts directory, which is also where `splits.splits_path` puts the split
# manifest. The earlier version fell back to the unfingerprinted
# `artifacts/synth`, then to `data/processed/synth`, then to `data/processed` —
# and that fallback chain was a leak with a search path. The manifest is
# fingerprint-keyed, so after any config change the fingerprinted directory is
# empty and the fallback would quietly load a `crossencoder_train.jsonl` built
# under the PREVIOUS split: the cross-encoder then fine-tunes on queries the
# current manifest assigns to LambdaMART, and every out-of-fold check still
# passes because each manifest is internally consistent. Failing to find the
# file is recoverable in a minute; training on the wrong split is not
# detectable at all.
def _search_roots(cfg: Config) -> list[Path]:
    # The directory is taken from the writer, not re-spelled here.
    # `splits.split_example_path` is the synthetic stage's own answer to "where
    # does crossencoder_train.jsonl go"; its parent is therefore the only root
    # worth searching. Spelling `artifacts/synth/<fingerprint>` out a second
    # time in this module was a quiet way for the two sides to drift: move the
    # synth directory on the writing side and this reader keeps searching the
    # old place, which is a FileNotFoundError on a clean tree but a hit on a
    # leftover file from the previous layout on a tree that has run before —
    # exactly the stale-split failure the comment above is about.
    return [synth_splits.split_example_path(TRAIN_SPLIT, cfg).parent]


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
    ``query``, ``positive_business_id``, and the negatives in either of the two
    shapes :func:`parse_example` documents (plus an optional ``query_id``).
    """

    query: str
    positive_business_id: str
    negative_business_ids: tuple[str, ...]
    query_id: str | None = None


# The two spellings of "the negatives for this query" that exist in this repo.
# `negatives` is what `concierge.synth.negatives.write_examples_jsonl` actually
# writes (a list of objects, each carrying the retrieval rank and score that
# mined it); `negative_business_ids` is the flat form. Both are read because the
# trainer only needs the ids, but neither may be *assumed*: see parse_example.
_NEGATIVES_KEYS = ("negatives", "negative_business_ids")


def _negative_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Pull business ids out of whichever negatives encoding the row uses.

    Objects are accepted as well as bare strings because the miner records *why*
    a negative was chosen, and throwing that file at this trainer must not
    quietly yield zero negatives — which is exactly what happened before this
    function existed: the mined file stores ``negatives``, the trainer read only
    ``negative_business_ids``, and `.get(...) or ()` turned the mismatch into a
    silent empty list. A cross-encoder trained on positives alone still reports a
    falling loss, so nothing downstream would have flagged it.
    """
    for key in _NEGATIVES_KEYS:
        if key not in row:
            continue
        ids: list[str] = []
        for item in row[key] or ():
            if isinstance(item, Mapping):
                try:
                    ids.append(str(item["business_id"]))
                except KeyError as exc:
                    raise KeyError(
                        f"negative object missing 'business_id'; has keys {sorted(item)}"
                    ) from exc
            else:
                ids.append(str(item))
        return tuple(ids)
    raise KeyError(
        f"synthetic example has no negatives: row has keys {sorted(row)}, expected one of "
        f"{list(_NEGATIVES_KEYS)}. Refusing to default to zero negatives — a cross-encoder "
        "trained on positives only still shows a falling loss, so this would not surface later."
    )


def parse_example(row: Mapping[str, Any]) -> SynthExample:
    """Validate one JSONL row. Missing keys fail here, naming what was found.

    Strict on purpose: a renamed key that defaulted to empty would train on
    positives with no negatives and still report a falling loss. An *explicitly*
    empty negatives list is allowed through — the miner records queries whose
    retrieval pool yielded nothing, and dropping them here would hide that.
    """
    try:
        query = str(row["query"]).strip()
        positive = str(row["positive_business_id"])
    except KeyError as exc:
        raise KeyError(
            f"synthetic example missing {exc.args[0]!r}; row has keys {sorted(row)}. "
            "Expected {'query', 'positive_business_id', 'negative_business_ids'}."
        ) from exc
    negatives = _negative_ids(row)
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


def load_verified_manifest(
    cfg: Config, path: str | Path | None = None
) -> tuple[synth_splits.SplitManifest, Path]:
    """Load the split manifest and verify it, or refuse to train.

    Every check is ``concierge.synth.splits``': the checksum (so a hand-edited
    manifest is an error), the partition, the out-of-fold property, and the
    seed and fractions against this config. They are *called*, not copied — the
    repo must have exactly one implementation of this guarantee, and until now
    this module did not call it at all, while a comment claimed the splits were
    "read from disk here".

    The config-fingerprint comparison is the one line not inside
    ``require_disjoint_splits``, because it is the caller's concern: the default
    path carries the fingerprint, but an explicitly supplied manifest does not.
    ``concierge.ltr.train.load_verified_manifest`` makes the same comparison for
    the same reason.

    A manifest whose ``config_fingerprint`` field is *empty* predates that field
    (or was written by hand) and so cannot be tied to any configuration. It is
    accepted — the seed and the fractions are still verified against this config
    — but it is said out loud, and ``train`` records what was found in the run
    metadata. Until this warning existed, "checked and clean" and "nothing to
    check" looked identical here: the fine-tune's metadata recorded a checksum
    that ``concierge.ltr.train`` then matched happily, so both artifacts agreed
    with each other while neither was tied to a configuration.
    """
    resolved = Path(path) if path is not None else synth_splits.splits_path(cfg)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    manifest = synth_splits.require_disjoint_splits(cfg, path=resolved)
    if manifest.config_fingerprint and manifest.config_fingerprint != cfg.fingerprint():
        raise synth_splits.SplitLeakageError(
            f"split manifest at {resolved} was written for config fingerprint "
            f"{manifest.config_fingerprint}, not this run's {cfg.fingerprint()}. "
            "Regenerate the synthetic split rather than fine-tuning against a "
            "split built for another configuration."
        )
    if not manifest.config_fingerprint:
        # stderr rather than `logging`, which this module does not configure;
        # the warning has to reach the person watching the fine-tune.
        print(
            f"WARNING: split manifest at {resolved} records no config fingerprint, "
            "so it cannot be tied to this configuration beyond its seed and "
            "fractions. The run metadata records this.",
            file=sys.stderr,
        )
    return manifest, resolved


def assert_rows_in_split(
    examples: Sequence[SynthExample],
    manifest: synth_splits.SplitManifest,
    split: str,
    path: Path,
) -> None:
    """Every row must be a query the manifest assigns to ``split``.

    This is what ties the queries the cross-encoder *actually fitted* to the
    manifest that Stage 4's out-of-fold guarantee is asserted against. Without
    it that guarantee rested on a filename: a file called
    ``crossencoder_train.jsonl`` was assumed to hold the crossencoder_train
    split, and a stale or hand-assembled file would be fine-tuned on in
    silence.

    A row with no ``query_id`` is refused rather than skipped. An id is the only
    thing that can be checked against the manifest, so a file without ids is a
    file whose split cannot be shown — which is exactly the state this check
    exists to end.
    """
    missing_ids = sum(1 for e in examples if not e.query_id)
    if missing_ids:
        raise synth_splits.SplitLeakageError(
            f"{missing_ids} of {len(examples)} rows in {path} have no 'query_id', so "
            f"they cannot be checked against the split manifest. Regenerate the "
            "synthetic data with query ids; without them nothing can show which "
            "queries this model was fine-tuned on."
        )
    wrong: list[tuple[str, str]] = []
    unassigned: list[str] = []
    for example in examples:
        assigned = manifest.assignment.get(str(example.query_id))
        if assigned is None:
            unassigned.append(str(example.query_id))
        elif assigned != split:
            wrong.append((str(example.query_id), assigned))
    if unassigned:
        raise synth_splits.SplitLeakageError(
            f"{len(unassigned)} query/queries in {path} are not in the split manifest "
            f"(e.g. {sorted(unassigned)[:3]}). Regenerate the manifest after adding "
            "queries rather than fine-tuning on unassigned data."
        )
    if wrong:
        raise synth_splits.SplitLeakageError(
            f"{len(wrong)} query/queries in {path} are assigned to another split by the "
            f"manifest (e.g. {sorted(wrong)[:3]} as (query_id, manifest split)), but "
            f"this file is being loaded as {split!r}. The cross-encoder would be "
            "fine-tuned on queries LambdaMART treats as unseen."
        )


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


def load_split(
    split: str,
    cfg: Config,
    roots: Sequence[Path] | None = None,
    *,
    manifest: synth_splits.SplitManifest | None,
) -> list[SynthExample]:
    """Read one split's JSONL and check every row against the manifest.

    ``manifest`` is keyword-only and has no default on purpose. Passing ``None``
    skips the membership check, which is right for a test that is inspecting a
    file's shape and wrong for anything that trains — and making the caller
    write ``manifest=None`` means that choice is visible at the call site
    instead of being the quiet default. :func:`train` never passes ``None``.
    """
    path = find_split_file(split, cfg, roots)
    examples = [parse_example(row) for row in read_jsonl(path)]
    if not examples:
        raise ValueError(f"split {split!r} at {path} is empty")
    if manifest is not None:
        assert_rows_in_split(examples, manifest, split, path)
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Leakage guard
# ─────────────────────────────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Casefold, collapse whitespace, and drop trailing punctuation before
    comparing two queries.

    Byte-exact comparison is a weaker guard than it looks: "Quiet cafe  with
    wifi" and "quiet cafe with wifi" are the same test query and byte equality
    would wave one of them through. Normalizing first makes the check strictly
    stronger while still being an *exact* match, which is what we want here —
    fuzzy near-duplicate removal is the synth stage's job and uses embeddings.

    Trailing ``.?!`` are stripped for the same reason case is folded: "good pho?"
    and "good pho" are one query, not two. This must stay identical to
    ``concierge.ltr.train._normalize_query`` — when the two drifted, the same
    gold query was caught before LambdaMART training but waved through before an
    hour of cross-encoder fine-tuning, and the *weaker* of two guards is the one
    that decides. ``tests/test_rerank.py`` pins them to each other.
    """
    return _WHITESPACE.sub(" ", text.strip()).casefold().rstrip(" .?!")


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
    query: str  # the text the model scores — see training_query_text
    document: str
    label: float  # 1.0 positive, 0.0 hard negative


# The weekday handed to the rule parser when deriving training text.
#
# Why a weekday is needed at all: `parse_rules` declines to encode an hours
# phrase when it is given no day (an `OpenAt` needs one, and guessing a day
# would filter on a day the user never mentioned), and a phrase it does not
# encode is a phrase it does not strip — so with no day, "open past 10pm" stays
# in the text and the skew this function exists to remove is still there.
#
# Why *this* day is immaterial: the day lands only in `ParsedQuery.open_at`,
# which the cross-encoder never sees. It does not change `semantic_text`, which
# is the only part used here. `test_training_text_does_not_depend_on_the_day`
# asserts that invariance over every weekday, so this is a pinned property and
# not an assumption. Taken from the dataset's own day list rather than written
# out, so there is one spelling of weekdays in the project.
_PARSE_DAY_FOR_TRAINING_TEXT = DAYS[0]


@cache
def training_query_text(query: str) -> str:
    """The exact string the cross-encoder is trained to score for ``query``.

    Serving scores ``query_text_for(parsed)``; so does training, which is the
    entire point of this function. Previously training used the raw synthetic
    query while serving used the constraint-stripped ``semantic_text``, so the
    fine-tune optimised one input distribution and the eval measured another.
    That costs measured quality with nothing to show for it, and it
    contaminates the one ablation the reranker exists to produce: the
    fine-tuned row would differ from the pretrained row by fine-tuning *and* by
    a change of input text.

    The parse is the **rule** parser, not the LLM one, for two reasons: it is
    deterministic (the same synthetic query yields the same training text on
    every run and on Colab), and it needs no API key, so training stays
    runnable and testable offline. It is also the parser that serves when the
    LLM path fails, so this text is one the serving system really produces.
    The residual, stated rather than hidden: when the LLM parser is the one
    serving, its ``semantic_text`` can differ from the rule parser's on the same
    query. That gap is far narrower than raw-vs-stripped and it is measurable —
    the eval harness ablates the parser — whereas the gap this closes was not
    visible anywhere.

    Cached because training re-derives the text for every epoch's validation
    pass and the function is a pure function of the string.
    """
    return query_text_for(parse_rules(query, today=_PARSE_DAY_FOR_TRAINING_TEXT))


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

    The query side of each pair is :func:`training_query_text`, not the raw
    synthetic query: it must be the string serving scores (module docstring).
    """
    pairs: list[LabeledPair] = []
    stats = PairBuildStats(n_examples=len(examples))
    for example in examples:
        positive_text = documents.get(example.positive_business_id)
        if positive_text is None:
            stats.n_examples_dropped_no_positive_doc += 1
            continue
        scored_query = training_query_text(example.query)
        pairs.append(LabeledPair(scored_query, positive_text, 1.0))
        stats.n_positive_pairs += 1
        for negative_id in example.negative_business_ids:
            negative_text = documents.get(negative_id)
            if negative_text is None:
                stats.n_negatives_dropped_no_doc += 1
                continue
            pairs.append(LabeledPair(scored_query, negative_text, 0.0))
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

    mps (Apple Silicon) is tried before cpu on the expectation that it is faster
    for a 6-layer model. That expectation is UNMEASURED here — nothing in this
    repo has timed mps against cpu. What would settle it is already recorded:
    every run writes ``device`` into its metadata, and any run that gets past
    the timing probe (``_PROBE_STEPS`` steps) writes
    ``measured_seconds_per_step`` beside it, so two full runs (``--device mps``
    and ``--device cpu``) turn this ordering into a measured fact rather than an
    expectation. The ordering is kept overridable for a
    second reason that does not depend on speed at all: a few transformer ops
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
    # Which split manifest this fine-tune was made under, by path and by
    # checksum. The checksum is the load-bearing field: `concierge.ltr.train`
    # compares it against the manifest it verifies, which is what turns "both
    # models used the same split" from an assumption into a check. A path alone
    # could not do that — the file at a path can be regenerated.
    # None rather than a placeholder: "no manifest was recorded" and "this
    # manifest was recorded" must not look alike, and a default seed of 0 would
    # read as a real seed in an artifact.
    split_manifest: str | None = None
    split_manifest_checksum: str | None = None
    split_manifest_seed: int | None = None
    # The config fingerprint the verified manifest *itself* carried, which is a
    # different fact from `config_fingerprint` above (this run's config). Three
    # distinguishable values, deliberately:
    #
    #   * a fingerprint string — the manifest was tied to a configuration, and
    #     it equalled this run's (a mismatch raises in
    #     `load_verified_manifest`, so no other string can be saved here);
    #   * ``""`` — the manifest recorded no fingerprint, so this fine-tune is
    #     tied to a split but not to a configuration;
    #   * ``None`` — the field was absent from the metadata file, i.e. the file
    #     predates this field. "Not recorded" and "recorded as untied" must not
    #     collapse into one value, because `concierge.ltr.train` reasons about
    #     them differently.
    split_manifest_config_fingerprint: str | None = None
    # The text derivation used for the query side of every pair, named so an
    # artifact records which one produced it if this ever becomes an ablated
    # choice.
    query_text_source: str = "rules_parse.semantic_text"
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
        # Same text derivation as training and as serving: a validation
        # diagnostic computed on a different input than the model was fitted on
        # would not answer the one question it is here to answer.
        scored_query = training_query_text(example.query)
        scores = _score_pairs(model, tokenizer, [(scored_query, t) for t in texts], cfg, device)
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
    splits_manifest: Path | None = None,
) -> RunMetadata:
    """Fine-tune the cross-encoder and return the run metadata.

    Order of operations matters: the split manifest is verified, the splits are
    loaded and checked against it, and the gold leakage check runs — all
    *before* a single weight is touched, so a leaky or mismatched dataset costs
    seconds rather than an hour of GPU time and a retracted number.

    ``splits_manifest`` defaults to the fingerprinted path
    ``concierge.synth.splits`` writes. It is a parameter for the same reason
    ``data_roots`` is: a Colab run may hold the manifest somewhere else. It is
    verified identically wherever it comes from.

    RUNNING ON COLAB (free T4) instead of the laptop: pip install the project,
    upload or mount ``data/processed`` (documents) and the synth split JSONLs plus
    ``eval/gold_queries.csv``, then run this module unchanged — ``detect_device``
    picks up cuda on its own. Only two things are worth changing there:
    ``rerank.batch_size`` (16 is sized for a laptop CPU; raise it on a T4 until
    it OOMs, since what fits depends on ``rerank.max_length`` and the checkpoint
    and has not been profiled here) and copying
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

    # The manifest first: it decides whether these files may be trained on at
    # all, so it is verified before they are read.
    manifest, manifest_path = load_verified_manifest(cfg, splits_manifest)
    train_examples = load_split(TRAIN_SPLIT, cfg, data_roots, manifest=manifest)
    val_examples = load_split(VAL_SPLIT, cfg, data_roots, manifest=manifest)
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
        split_manifest=str(manifest_path),
        split_manifest_checksum=manifest.checksum,
        split_manifest_seed=manifest.seed,
        # Recorded verbatim, empty string included: see the field's comment.
        split_manifest_config_fingerprint=manifest.config_fingerprint,
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
    parser.add_argument(
        "--splits-manifest",
        default=None,
        help=(
            "Split manifest written by the synthetic-data stage. Defaults to "
            "artifacts/synth/<config fingerprint>/splits.json. Verified either way, "
            "including that it was written for this config's fingerprint."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    meta = train(
        cfg,
        device=args.device,
        limit_examples=args.limit_examples,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        save=not args.no_save,
        splits_manifest=Path(args.splits_manifest) if args.splits_manifest else None,
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
