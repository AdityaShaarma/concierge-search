"""Stage 4: train the LambdaMART final ranker.

    python -m concierge.ltr.train --config configs/base.yaml

What this stage is for
----------------------
Stages 2 and 3 answer "how well does this text match the request". They say
nothing about whether the place is any good, whether it is open, or whether it
is nearby. LambdaMART's job is to blend relevance with those quality and
constraint signals, learned from labels rather than from a hand-written formula
with hand-set weights.

LambdaMART in one paragraph (the interview answer)
--------------------------------------------------
It is gradient-boosted decision trees trained on a *pairwise* objective with a
*listwise* weighting. For every pair of candidates within the same query where
one is more relevant than the other, the gradient pushes them apart; each pair's
gradient is scaled by how much swapping those two documents would change NDCG.
So the model spends its capacity where the metric is sensitive — the top of the
list — instead of treating a swap at rank 90 as being as important as one at
rank 1. "Queries as groups" is how the trainer knows which rows may be compared:
pairs are only formed inside a group.

The two bugs this file is built to prevent
------------------------------------------
1. **An in-fold cross-encoder feature.** The cross-encoder score is one of the
   strongest features. If it is computed for queries the cross-encoder was
   fine-tuned on, it is inflated — the model has partly memorised those
   (query, business) pairs. LambdaMART then learns to trust that feature more
   than it deserves, and at serving time, on queries nobody trained on, the
   feature is weaker than the learned weight assumes. Offline numbers look
   good; served quality is worse; nothing errors.

   How that is prevented here, in three steps that each have to hold:

   a. :func:`load_verified_manifest` loads the *fingerprinted* manifest through
      ``concierge.synth.splits.require_disjoint_splits`` — the one
      implementation of this guarantee — so the checksum, the partition, the
      seed and the configured fractions are all verified before any data is
      read. A missing manifest, an edited one, an unknown split name or one
      produced under a different configuration all stop the run.
   b. :func:`select_split_rows` chooses the training rows **by manifest
      membership**, not by the ``split`` string each JSONL row carries. Because
      the manifest's assignment is a mapping from query id to exactly one
      split, "this row is in both a cross-encoder split and a LambdaMART split"
      becomes unrepresentable rather than merely unlikely. A row whose own
      label disagrees with the manifest is the realistic shape of this leak (a
      stale examples file), and it is reported, not silently honoured.
   c. :func:`assert_run_followed_the_manifest` re-checks the ids actually
      loaded against the manifest, and :func:`crossencoder_run_check` compares
      the manifest the *cross-encoder* recorded training against with the one
      used here. Step (a) proves the plan was sound; (b) and (c) prove the run
      followed it.

2. **A wrong group array.** LightGBM takes groups as *sizes of consecutive
   runs* — ``[3, 2]`` means rows 0-2 are query A and rows 3-4 are query B. It
   does not take query ids and cannot detect that rows are interleaved. Get it
   wrong and the trainer happily forms pairs across different queries, which is
   a meaningless objective: it trains, it converges, and the model is silently
   wrong. :func:`group_sizes` therefore *requires* contiguity and raises rather
   than guessing.

Both are tested directly in ``tests/test_ltr.py``.

What this stage reads, and what does not exist yet
--------------------------------------------------
Two files, both in the config-fingerprinted synth directory
``artifacts/synth/<config fingerprint>/``:

* ``splits.json`` — the checksummed split manifest, written by
  ``concierge.synth.splits`` (:func:`default_manifest_path`);
* ``ltr_examples.jsonl`` — candidate-level feature rows, one per
  (query, retrieved business) pair (:func:`default_examples_path`).

The second file has no producer in this repository. The synthetic-data stage
writes the *query*-level ``ltr_train.jsonl`` and ``ltr_val.jsonl`` and says so;
turning those into feature rows requires running retrieval and the *fine-tuned*
cross-encoder over each query's candidates, which is only possible after Stage 3
has trained. So ``make train-ltr`` currently stops at a ``FileNotFoundError``
from :func:`load_examples` that states the contract. That is deliberate — the
alternative to failing closed is a hand-assembled file whose cross-encoder
scores nobody can account for — but it is a gap, not a design: the stage between
the fine-tune and this trainer still has to be written.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concierge.config import REPO_ROOT, Config, load_config
from concierge.ltr.features import (
    FEATURE_NAMES,
    NON_TEXT_FEATURE_NAMES,
    FeatureMatrix,
    ablated_feature_names,
    feature_matrix,
)
from concierge.synth import splits as synth_splits
from concierge.types import (
    Business,
    Candidate,
    ConstraintMatch,
    ConstraintStatus,
    GeoConstraint,
    OpenAt,
    ParsedQuery,
    ParseSource,
)

log = logging.getLogger(__name__)

# Split names come from `concierge.synth.splits`, which owns the vocabulary, the
# manifest format and the disjointness guarantee. Re-spelling the strings here
# would let the two modules drift: a trainer asking for "crossencoder_train"
# while the writer had renamed it would look at an empty set, and a leakage
# check that looks at nothing always passes. Unpacking the writer's own tuples
# means a rename there is an ImportError here, not a silent vacuous check.
SPLIT_CE_TRAIN, SPLIT_CE_VAL = synth_splits.CROSSENCODER_SPLITS
SPLIT_LTR_TRAIN, SPLIT_LTR_VAL = synth_splits.LTR_SPLITS
ALL_SPLITS = synth_splits.SPLIT_NAMES

# The filename this stage's feature rows are read from. It lives beside the
# split files inside the config-fingerprinted synth directory (see
# :func:`default_examples_path`), not under ``data/processed``: these rows are
# derived from a specific manifest and a specific fine-tuned checkpoint, so a
# config change must not be answerable with the previous configuration's rows.
EXAMPLES_FILENAME = "ltr_examples.jsonl"
DEFAULT_OUT_ROOT = Path("artifacts/ltr")

# NDCG cutoff used for early stopping. Same cutoff as the primary reported
# metric, so the checkpoint is chosen by the thing the project is judged on.
EARLY_STOPPING_NDCG_K = 10


class LeakageError(synth_splits.SplitError):
    """Raised when training data overlaps data it must be independent of.

    A distinct exception type so the tests can assert *why* training refused,
    and so a leakage failure can never be caught by a generic ``except
    ValueError`` somewhere up the stack.

    Subclasses ``concierge.synth.splits.SplitError`` so that one ``except``
    clause covers every way this guarantee can fail — a corrupt manifest
    (``SplitManifestCorruptError``), a manifest that does not partition
    (``SplitLeakageError``) and the row-level failures raised here. They are one
    guarantee with one correct response: stop, because nothing measured
    downstream of them could be reported.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Split manifest — loaded, never re-derived, and never re-implemented
# ─────────────────────────────────────────────────────────────────────────────


def default_manifest_path(cfg: Config) -> Path:
    """Where the split manifest lives for this configuration.

    ``concierge.synth.splits.splits_path`` is the authority: the writer and the
    readers must agree on one path, and the path is keyed by the config
    fingerprint so a config change cannot silently answer with a split built for
    a different configuration. This module used to default to
    ``data/processed/synth_splits.json``, which nothing in the repo ever writes.
    """
    return synth_splits.splits_path(cfg)


def default_examples_path(cfg: Config, root: Path | None = None) -> Path:
    """Where this stage's feature rows live: ``<synth dir>/ltr_examples.jsonl``.

    The directory is taken from the writer rather than spelled out again here.
    ``concierge.synth.splits.split_example_path`` is that module's own answer to
    "where does ``ltr_train.jsonl`` go", so its parent is the fingerprinted
    synth directory by construction: relocate the synth artifacts on the writing
    side and this reader moves with them, instead of going on looking in a
    directory nothing writes any more.

    Why not ``data/processed/ltr_examples.jsonl``, which this module used to
    default to? Two reasons, and the second is the serious one:

    1. ``data/processed`` is not keyed by the config fingerprint, so one stale
       file answers every configuration. The rows carry a cross-encoder score
       per candidate, and that score is only meaningful for the checkpoint and
       the split it was produced under — which the fingerprint is exactly the
       key for.
    2. Nothing in the repository ever wrote that path, so the default pointed at
       a file that could only arrive by hand. The leakage checks below still run
       on such a file — they check the manifest *assignment* of each row — but
       no check can tie a hand-written ``cross_encoder`` score to any
       checkpoint. Defaulting into the fingerprinted directory does not fix
       that; what it fixes is reach — rows built under one configuration can no
       longer answer a run under another, because the directory name changes
       with the config.

    **The producer is not in the repository yet.** ``concierge.synth.splits``
    writes ``ltr_train.jsonl`` / ``ltr_val.jsonl`` — query-level rows — and says
    so explicitly; turning those into the candidate-level rows
    :func:`load_examples` reads needs retrieval plus the *fine-tuned*
    cross-encoder over each query's candidate list, so it can only run after
    Stage 3. Until that stage exists this function names the file it must write,
    and :func:`load_examples` fails with that contract in the message. It fails
    closed: no model, no number, nothing to retract.
    """
    return synth_splits.split_example_path(SPLIT_LTR_TRAIN, cfg, root).parent / EXAMPLES_FILENAME


def load_verified_manifest(
    cfg: Config, path: str | Path | None = None
) -> tuple[synth_splits.SplitManifest, Path]:
    """Load the split manifest and verify everything about it. Returns it and its path.

    Every check lives in ``concierge.synth.splits`` and is called, not copied:
    :func:`concierge.synth.splits.require_disjoint_splits` verifies the
    checksum, that the four splits partition the queries, that no query is in
    both a cross-encoder and a LambdaMART split, and that the manifest's seed
    and fractions still match the config. This module previously shipped its own
    reader that skipped the checksum and the fractions, which meant the function
    documented as "what the trainers call" was called only by tests — two
    implementations of one guarantee, with the weaker one wired in.

    The config fingerprint is checked here rather than there because it is this
    caller's concern: ``--splits`` and any hand-placed file can point at a
    manifest from another configuration, and the fingerprinted default path is
    the only thing that otherwise prevents it. An empty fingerprint field means
    the manifest predates that field; it is reported rather than guessed about.

    Nothing here is caught: a split that cannot be verified means this run
    cannot produce a reportable number, so refusing to start is the outcome.
    """
    resolved = _resolve(path if path is not None else default_manifest_path(cfg))
    manifest = synth_splits.require_disjoint_splits(cfg, path=resolved)
    if manifest.config_fingerprint and manifest.config_fingerprint != cfg.fingerprint():
        raise LeakageError(
            f"split manifest at {resolved} was written for config fingerprint "
            f"{manifest.config_fingerprint} but this run's config fingerprint is "
            f"{cfg.fingerprint()}. The cross-encoder was split under a different "
            "configuration; regenerate the synthetic split instead of reusing it."
        )
    if not manifest.config_fingerprint:
        log.warning(
            "split manifest at %s records no config fingerprint, so it cannot be "
            "tied to this configuration beyond its seed and fractions",
            resolved,
        )
    log.info("split manifest verified: %s", manifest.summary())
    return manifest, resolved


def select_split_rows(
    examples: Sequence[LtrExample],
    manifest: synth_splits.SplitManifest,
    split: str,
    *,
    enforce: bool = True,
) -> list[LtrExample]:
    """The rows of ``examples`` the *manifest* assigns to ``split``.

    Selection is by manifest membership, not by the ``split`` string each JSONL
    record carries, and that is the whole point. The manifest assigns each query
    id to exactly one split, so selecting from it makes "this query is in both a
    cross-encoder split and a LambdaMART split" unrepresentable instead of
    merely checked. Selecting on the row's own label left the manifest
    decorative: a stale ``ltr_examples.jsonl`` whose rows say ``ltr_train``
    while the current manifest says ``crossencoder_train`` would train
    LambdaMART on in-fold cross-encoder scores with every leakage check
    reporting clean.

    Two failures are therefore hard, not logged:

    * a row whose query id the manifest does not mention at all — it would be
      training data nobody assigned, and :func:`concierge.synth.splits.split_of`
      refuses that for the same reason;
    * a row whose own ``split`` disagrees with the manifest — the row and the
      manifest describe different experiments, and there is no way to tell from
      here which one the cross-encoder was fine-tuned under.

    ``enforce=False`` (``cfg.ltr.enforce_oof_crossencoder_feature: false``)
    downgrades the *disagreement* to a loud log, for the deliberate experiment
    that measures what the inflation is worth. It does not change which rows are
    selected — the manifest still decides that — and the artifact's metadata
    records that the run was made in that mode.
    """
    # Raises on an unknown split name instead of returning an empty selection,
    # which is the failure mode this whole mechanism exists to avoid: a typo'd
    # name makes every check pass by looking at nothing.
    if split not in synth_splits.SPLIT_NAMES:
        raise LeakageError(f"unknown split {split!r}; expected one of {synth_splits.SPLIT_NAMES}")

    unassigned = sorted({e.query_id for e in examples if e.query_id not in manifest.assignment})
    if unassigned:
        raise LeakageError(
            f"{len(unassigned)} training query/queries are not in the split manifest "
            f"(e.g. {unassigned[:3]}). Regenerate the manifest after adding queries "
            "rather than training on unassigned data."
        )

    disagreements = sorted(
        {
            (e.query_id, e.split, manifest.assignment[e.query_id])
            for e in examples
            if e.split and e.split != manifest.assignment[e.query_id]
        }
    )
    if disagreements:
        message = (
            f"{len(disagreements)} example row(s) disagree with the split manifest "
            f"(e.g. {disagreements[:3]} as (query_id, row split, manifest split)). "
            "The examples file and the manifest describe different splits, so the "
            "cross-encoder feature cannot be shown to be out of fold. Regenerate "
            "the examples from the current manifest."
        )
        if enforce:
            raise LeakageError(message)
        log.error("OUT-OF-FOLD CHECK DISABLED: %s", message)

    return [e for e in examples if manifest.assignment[e.query_id] == split]


def assert_run_followed_the_manifest(
    examples: Sequence[LtrExample], manifest: synth_splits.SplitManifest
) -> None:
    """Check the ids this run actually loaded against the manifest.

    Delegates to :func:`concierge.synth.splits.assert_models_trained_on_disjoint_queries`,
    which is documented as "the one worth calling from ``concierge.ltr.train``"
    and was, until now, called only from the tests. Verifying the manifest proves
    the plan was sound; this proves the rows in hand follow it. It is cheap and
    it is the assertion the artifact's metadata implicitly claims.
    """
    synth_splits.assert_models_trained_on_disjoint_queries(
        manifest.ids(*synth_splits.CROSSENCODER_SPLITS),
        [e.query_id for e in examples],
        manifest,
    )


def crossencoder_run_check(cfg: Config, manifest: synth_splits.SplitManifest) -> dict[str, Any]:
    """Compare the manifest the cross-encoder trained under with this one.

    The remaining hole after everything above: both trainers can verify the
    manifest at the fingerprinted path and still disagree, because that path can
    be *rewritten* — regenerate the synthetic queries under the same config and
    the manifest at that path is a different split with a different checksum.
    The cross-encoder's run metadata records the checksum it trained under, so
    the two can be compared for the price of reading one JSON file.

    No fine-tuned run metadata is not a failure, and saying so is not a dodge:
    without a fine-tuned checkpoint for this config the cross-encoder feature
    was produced by the pretrained model, which was fitted on no synthetic query
    at all and therefore cannot be in fold. What would be dishonest is leaving
    that unrecorded, so the returned dict always says which case this run was.

    The fine-tune also records which config fingerprint its manifest carried
    (``split_manifest_config_fingerprint``), and that is cross-checked here,
    because the checksum does not cover it:
    ``concierge.synth.splits.compute_checksum`` hashes the assignment, seed,
    fractions and grouping, so two manifests can share a checksum — be the same
    split — while having been written under different configurations. This run
    has already verified that *its* manifest's fingerprint equals
    ``cfg.fingerprint()``, so a different fingerprint recorded by the fine-tune
    means the checkpoint supplying the ``cross_encoder_score`` column was
    produced under other settings (another base checkpoint, another
    ``max_length``, another retrieval config). The split is sound; the feature's
    provenance is not, and that is the same objection
    :func:`load_verified_manifest` raises about a manifest from another
    configuration. So it is refused.

    Two weaker outcomes are reported rather than refused:

    * an empty string — the manifest carried no fingerprint at all. This run's
      own :func:`load_verified_manifest` has already warned about that very
      manifest; refusing twice for one cause only invites a flag that disables
      both.
    * a missing key — the metadata predates the field. "Old artifact" and
      "untied artifact" are different problems and the returned dict names which
      one this was, so the LambdaMART artifact records it either way.
    """
    from concierge.rerank.model import RUN_METADATA_FILENAME, rerank_dir

    path = rerank_dir(cfg) / RUN_METADATA_FILENAME
    if not path.exists():
        log.info(
            "no cross-encoder run metadata at %s: the cross_encoder_score feature "
            "comes from the pretrained model, which saw no synthetic query",
            path,
        )
        return {"checked": False, "reason": f"no fine-tuned run metadata at {path}"}

    recorded = json.loads(path.read_text())
    checksum = recorded.get("split_manifest_checksum")
    if not checksum:
        raise LeakageError(
            f"the cross-encoder run metadata at {path} records no split manifest "
            "checksum, so there is no way to show it was fine-tuned on the same "
            "split this run assumes. Re-run the cross-encoder fine-tune."
        )
    if checksum != manifest.checksum:
        raise LeakageError(
            f"the cross-encoder was fine-tuned under split manifest {checksum[:12]} "
            f"but this run verified {manifest.checksum[:12]}. The manifest was "
            "regenerated after the fine-tune, so queries LambdaMART treats as "
            "unseen may be queries the cross-encoder fitted. Retrain the "
            "cross-encoder or regenerate the examples."
        )
    if recorded.get("is_smoke_run"):
        raise LeakageError(
            f"the cross-encoder run at {path} is marked a smoke run "
            "(--limit-examples), so its checkpoint trained on a truncated subset "
            "and nothing measured from it may be reported. Re-run the fine-tune "
            "in full before training LambdaMART on its scores."
        )
    # Sentinel rather than None, so "key absent" is distinguishable from "key
    # present and empty" — the two mean different things (see the docstring).
    _ABSENT = object()
    recorded_fingerprint = recorded.get("split_manifest_config_fingerprint", _ABSENT)
    if recorded_fingerprint is _ABSENT:
        fingerprint_tie = "not_recorded"
        log.warning(
            "the cross-encoder run metadata at %s predates the "
            "split_manifest_config_fingerprint field, so this run cannot confirm "
            "the fine-tune observed the same config fingerprint in the manifest it "
            "shares with this run",
            path,
        )
    elif not recorded_fingerprint:
        fingerprint_tie = "manifest_recorded_none"
        log.warning(
            "the cross-encoder at %s was fine-tuned under a split manifest that "
            "recorded no config fingerprint, so neither model is tied to a "
            "configuration beyond the manifest's seed and fractions",
            path,
        )
    elif recorded_fingerprint != manifest.config_fingerprint:
        # LeakageError for the same reason load_verified_manifest uses it on a
        # manifest from another configuration: the type is what every caller
        # treats as "stop, nothing downstream of this is reportable". The
        # out-of-fold property is intact here (same checksum, same partition);
        # what is broken is the feature's provenance.
        raise LeakageError(
            f"the cross-encoder run metadata at {path} says it was fine-tuned "
            f"under config fingerprint {recorded_fingerprint}, but this run's is "
            f"{manifest.config_fingerprint or '(none)'}. The split partition "
            f"matches (checksum {checksum[:12]}; the checksum does not cover the "
            "fingerprint), so the out-of-fold guarantee holds — but the "
            "checkpoint producing the cross_encoder_score feature was trained "
            "under settings other than the ones this run will record. Re-run the "
            "fine-tune under this config."
        )
    else:
        fingerprint_tie = "matched"
    return {
        "checked": True,
        "run_metadata": str(path),
        "split_manifest_checksum": checksum,
        "config_fingerprint_tie": fingerprint_tie,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gold leakage
# ─────────────────────────────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")


def _normalize_query(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    Exact-match-after-normalization only. Near-duplicate detection by embedding
    similarity is the synthetic stage's job (``cfg.synth.gold_dedup_cosine_threshold``)
    because it needs the embedding model; this check is the cheap backstop that
    runs unconditionally at training time.
    """
    return _WHITESPACE.sub(" ", text.strip().lower()).rstrip(" .?!")


def load_gold_query_texts(path: str | Path) -> tuple[set[str], set[str]]:
    """``(normalized query texts, query ids)`` from the gold CSV."""
    p = _resolve(path)
    texts: set[str] = set()
    ids: set[str] = set()
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            query = (row.get("query") or "").strip()
            if query:
                texts.add(_normalize_query(query))
            query_id = (row.get("query_id") or "").strip()
            if query_id:
                ids.add(query_id)
    return texts, ids


def assert_no_gold_leakage(
    examples: Sequence[LtrExample],
    gold_path: str | Path,
    *,
    allow_missing_gold: bool = False,
) -> None:
    """Fail if any training query is a gold query, by id or by text.

    The gold set is test-only (CLAUDE.md section 6). A gold query that leaks
    into LambdaMART training does not just inflate one feature — it corrupts the
    headline NDCG@10 the whole project reports, and the corruption is invisible
    because the number moves in the direction you were hoping for.

    **Fails closed when the gold file is missing.** A leakage check that passes
    because it had nothing to compare against is strictly worse than no check:
    it prints reassurance it has not earned. ``allow_missing_gold=True`` is the
    explicit, logged override for the period before the owner has written the
    gold queries.
    """
    p = _resolve(gold_path)
    if not p.exists():
        if allow_missing_gold:
            log.warning(
                "GOLD LEAKAGE CHECK SKIPPED: %s does not exist yet. This model's "
                "metadata records that the check did not run.",
                p,
            )
            return
        raise LeakageError(
            f"gold query file not found: {p}. Refusing to train without being "
            "able to check for gold leakage; pass --allow-missing-gold only if "
            "the gold set genuinely does not exist yet."
        )

    gold_texts, gold_ids = load_gold_query_texts(p)
    hit_ids = sorted({e.query_id for e in examples} & gold_ids)
    hit_texts = sorted(
        {e.query_text for e in examples if _normalize_query(e.query_text) in gold_texts}
    )
    if hit_ids or hit_texts:
        raise LeakageError(
            "gold queries found in LambdaMART training data — "
            f"{len(hit_ids)} by query_id (e.g. {hit_ids[:3]}), "
            f"{len(hit_texts)} by text (e.g. {hit_texts[:3]}). The gold set is "
            "test-only; every reported metric would be invalid."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training examples
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LtrExample:
    """One labeled (query, candidate) pair, carrying the contract objects.

    The example holds a ``ParsedQuery``, a ``Candidate`` and a ``Business``
    rather than a precomputed feature vector. That is the whole anti-skew
    design: the training job runs the *same* ``build_feature_row`` the serving
    path runs, over the same types, so there is no offline feature table that
    could be computed differently from the online one.
    """

    query_id: str
    query_text: str
    label: int
    parsed: ParsedQuery
    candidate: Candidate
    business: Business
    # Which split the file that produced this row *claims* it belongs to. The
    # manifest, not this field, decides which rows train the model
    # (:func:`select_split_rows`); the field is kept because comparing the two
    # is how a stale examples file is caught, and that comparison is now made
    # on every run. Empty means the writer recorded no split, which is accepted
    # — the manifest still assigns the row — and leaves nothing to compare.
    split: str = ""


def load_examples(path: str | Path) -> list[LtrExample]:
    """Read every JSONL training example in the file.

    Deliberately does **not** filter by split. Which rows belong to which split
    is the manifest's decision, applied by :func:`select_split_rows`; a filter
    here (on each row's own ``split`` string) is what made the manifest
    decorative, so there is now exactly one way to choose training rows.

    Expected record shape (written by the synthetic-data stage):

    .. code-block:: json

        {"query_id": "s0001", "split": "ltr_train", "label": 2,
         "query": {"raw": "quiet cafe with wifi", "semantic_text": "quiet cafe",
                   "attributes": {"WiFi": "free"}, "categories": ["Coffee & Tea"],
                   "open_at": {"day": "Friday", "minute_of_day": 1320},
                   "price_max": 2,
                   "geo": {"latitude": 1.0, "longitude": 2.0, "radius_km": 3.0}},
         "business": {"business_id": "b001", "stars": 4.5, "review_count": 120,
                      "categories": ["Coffee & Tea"], "latitude": 1.0,
                      "longitude": 2.0},
         "scores": {"bm25": 8.1, "dense": 0.74, "rrf": 0.031,
                    "cross_encoder": 3.2},
         "ranks": {"rrf": 4},
         "constraints": [{"name": "WiFi", "status": "satisfied"}]}

    Constraints arrive already evaluated because the constraint checker ran
    upstream, on the real candidate list. Re-checking them here would require
    the full hours/attribute data in every record and could disagree with what
    the system actually decided.
    """
    p = _resolve(path)
    if not p.exists():
        # `make ltr-examples` (concierge.ltr.build_examples) writes this file.
        # It is deliberately NOT part of `make synth`: the rows carry a
        # cross-encoder score per candidate, so they can only be built after the
        # reranker is fine-tuned. Pointing the owner at `make synth` for a file
        # `make synth` does not write would cost an afternoon.
        raise FileNotFoundError(
            f"LambdaMART training examples not found: {p}.\n"
            "These are candidate-level feature rows: one per (query, retrieved "
            "business) pair, carrying the Stage 2 scores and ranks, the "
            "fine-tuned cross-encoder score, the evaluated constraints and a "
            "relevance label (see this function's docstring for the exact "
            "shape).\n"
            "No stage in this repository writes them yet. The synthetic-data "
            "stage writes the QUERY-level files (ltr_train.jsonl, "
            "ltr_val.jsonl) and the verified split manifest into the same "
            "directory; producing these rows needs retrieval and the fine-tuned "
            "cross-encoder run over each of those queries, so it can only "
            "happen after the Stage 3 fine-tune. Hand-assembling the file "
            "instead bypasses nothing — every leakage check in this module "
            "still runs on whatever it contains — but the cross-encoder scores "
            "in it would be unexplainable, so build it with code."
        )
    examples: list[LtrExample] = []
    with p.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(example_from_json(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{p}:{line_no}: bad training example: {exc}") from exc
    return examples


def example_from_json(record: Mapping[str, Any]) -> LtrExample:
    """Materialize the contract objects from one JSONL record."""
    query = dict(record.get("query") or {})
    business = dict(record.get("business") or {})
    parsed = _parsed_from_json(query)
    return LtrExample(
        query_id=str(record["query_id"]),
        query_text=parsed.raw,
        label=int(record["label"]),
        parsed=parsed,
        candidate=Candidate(
            business_id=str(business.get("business_id", record.get("business_id", ""))),
            scores={str(k): float(v) for k, v in (record.get("scores") or {}).items()},
            ranks={str(k): int(v) for k, v in (record.get("ranks") or {}).items()},
            constraints=_constraints_from_json(record.get("constraints") or ()),
        ),
        business=_business_from_json(business),
        split=str(record.get("split", "")),
    )


def _parsed_from_json(query: Mapping[str, Any]) -> ParsedQuery:
    raw = str(query.get("raw", ""))
    open_at = query.get("open_at")
    geo = query.get("geo")
    return ParsedQuery(
        raw=raw,
        semantic_text=str(query.get("semantic_text", raw)),
        open_at=(
            OpenAt(day=str(open_at["day"]), minute_of_day=int(open_at["minute_of_day"]))
            if open_at
            else None
        ),
        attributes=dict(query.get("attributes") or {}),
        price_max=(int(query["price_max"]) if query.get("price_max") is not None else None),
        categories=tuple(str(c) for c in (query.get("categories") or ())),
        geo=(
            GeoConstraint(
                latitude=float(geo["latitude"]),
                longitude=float(geo["longitude"]),
                radius_km=float(geo["radius_km"]),
                label=str(geo.get("label", "")),
            )
            if geo
            else None
        ),
        # Synthetic training queries are generated, not parsed by the live LLM
        # path, so the source is recorded as RULES rather than claiming an LLM
        # parse happened.
        source=ParseSource.RULES,
    )


def _business_from_json(business: Mapping[str, Any]) -> Business:
    """Only the fields Stage 4 features actually read are required.

    Hours and attributes default to empty: the constraint outcomes are carried
    on the candidate (see :func:`load_examples`), so the feature builder never
    touches them, and demanding them in every training record would bloat the
    file for no gain.
    """
    return Business(
        business_id=str(business.get("business_id", "")),
        name=str(business.get("name", "")),
        city=str(business.get("city", "")),
        state=str(business.get("state", "")),
        latitude=_opt_float(business.get("latitude")),
        longitude=_opt_float(business.get("longitude")),
        stars=float(business.get("stars", 0.0)),
        review_count=int(business.get("review_count", 0)),
        categories=tuple(str(c) for c in (business.get("categories") or ())),
        attributes=dict(business.get("attributes") or {}),
        hours={},
        is_open=bool(business.get("is_open", True)),
    )


def _constraints_from_json(items: Sequence[Mapping[str, Any]]) -> tuple[ConstraintMatch, ...]:
    return tuple(
        ConstraintMatch(
            name=str(item["name"]),
            requested=item.get("requested"),
            actual=item.get("actual"),
            status=ConstraintStatus(str(item["status"])),
        )
        for item in items
    )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


# ─────────────────────────────────────────────────────────────────────────────
# Groups
# ─────────────────────────────────────────────────────────────────────────────


def group_sizes(query_ids: Sequence[str]) -> np.ndarray:
    """Run-length encoding of ``query_ids``, requiring contiguity.

    LightGBM's ``group`` is a list of consecutive run lengths summing to the
    number of rows; it never sees the query ids. So ``["a","b","a"]`` has no
    valid group array — the only honest response is to refuse, because the
    alternative (treating it as ``[1,1,1]``, or as ``[2,1]``) trains a model on
    pairs that mean nothing and reports a plausible NDCG for it.

    Raises ``ValueError`` on an empty input too: an empty group array and an
    empty dataset are different kinds of nothing, and LightGBM's error for the
    latter is not informative.
    """
    if not query_ids:
        raise ValueError("cannot build groups from zero rows")

    sizes: list[int] = []
    seen: set[str] = set()
    current = query_ids[0]
    count = 0
    for query_id in query_ids:
        if query_id == current:
            count += 1
            continue
        sizes.append(count)
        seen.add(current)
        if query_id in seen:
            raise ValueError(
                f"query id {query_id!r} appears in more than one block — rows must "
                "be grouped contiguously by query before building the group array. "
                "Call sort_by_query() first."
            )
        current = query_id
        count = 1
    sizes.append(count)

    group = np.asarray(sizes, dtype=np.int32)
    if int(group.sum()) != len(query_ids):
        # Cannot happen with the loop above; asserted because a group array that
        # does not sum to the row count is the exact off-by-one that LightGBM
        # accepts silently in some versions.
        raise ValueError(f"group sizes sum to {group.sum()}, expected {len(query_ids)}")
    return group


def sort_by_query(examples: Sequence[LtrExample]) -> list[LtrExample]:
    """Group rows by query without reordering queries or rows within a query.

    Stable, and ordered by each query's *first appearance* rather than
    alphabetically, so the row order of the input file is preserved as far as
    contiguity allows. Determinism matters here: the group array, and hence the
    pairs LambdaMART forms, must be identical across runs for a fixed seed to
    mean anything.
    """
    first_seen: dict[str, int] = {}
    for example in examples:
        first_seen.setdefault(example.query_id, len(first_seen))
    return sorted(examples, key=lambda e: first_seen[e.query_id])


@dataclass(frozen=True, slots=True)
class RankingData:
    """Features, labels and groups, guaranteed mutually consistent."""

    features: FeatureMatrix
    labels: np.ndarray
    group: np.ndarray
    query_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n = self.features.n_rows
        if len(self.labels) != n:
            raise ValueError(f"{len(self.labels)} labels for {n} feature rows")
        if len(self.query_ids) != n:
            raise ValueError(f"{len(self.query_ids)} query ids for {n} feature rows")
        if int(self.group.sum()) != n:
            raise ValueError(f"groups sum to {self.group.sum()} but there are {n} rows")

    @property
    def n_queries(self) -> int:
        return len(self.group)


def build_ranking_data(
    examples: Sequence[LtrExample],
    names: Sequence[str] = FEATURE_NAMES,
) -> RankingData:
    """Sort into contiguous query blocks, then build features, labels, groups."""
    ordered = sort_by_query(examples)
    query_ids = tuple(e.query_id for e in ordered)
    matrix = feature_matrix(((e.parsed, e.candidate, e.business) for e in ordered), names=names)
    labels = np.asarray([e.label for e in ordered], dtype=np.float64)
    return RankingData(matrix, labels, group_sizes(query_ids), query_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TrainedRanker:
    """A trained booster plus everything needed to interpret and reproduce it."""

    booster: Any  # lightgbm.Booster, typed loosely to keep the import local
    feature_names: tuple[str, ...]
    best_iteration: int
    val_ndcg_at_10: float | None
    n_train_rows: int
    n_train_queries: int
    n_val_rows: int
    n_val_queries: int

    def importances(self, importance_type: str = "gain") -> dict[str, float]:
        """Named feature importances, largest first.

        ``gain`` rather than ``split`` by default: split count rewards features
        that are merely high-cardinality (every continuous score can be split
        on endlessly), while gain measures how much the objective actually
        improved. Both are saved in the metadata so the difference can be shown.
        """
        values = self.booster.feature_importance(importance_type=importance_type)
        pairs = zip(self.feature_names, (float(v) for v in values), strict=True)
        return dict(sorted(pairs, key=lambda kv: kv[1], reverse=True))


def lgb_params(cfg: Config, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """LightGBM parameters from the config, plus the determinism settings.

    ``deterministic`` and ``force_row_wise`` are set because reproducibility is
    a stated project promise: without them LightGBM may choose a different
    histogram construction path depending on thread count and data layout, and
    the model changes run to run even with a fixed seed. The cost is some speed
    on this dataset's scale, which does not matter here.
    """
    params: dict[str, Any] = {
        "objective": cfg.ltr.objective,
        "metric": "ndcg",
        "eval_at": [EARLY_STOPPING_NDCG_K],
        "num_leaves": cfg.ltr.num_leaves,
        "learning_rate": cfg.ltr.learning_rate,
        "seed": cfg.seed,
        # All four derived seeds are pinned explicitly rather than relying on
        # `seed` to seed them, which is version-dependent behaviour.
        "bagging_seed": cfg.seed,
        "feature_fraction_seed": cfg.seed,
        "data_random_seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,  # thread count can change float summation order
        "verbosity": -1,
    }
    if extra:
        params.update(extra)
    return params


def train_lambdamart(
    train: RankingData,
    val: RankingData | None,
    cfg: Config,
    extra_params: Mapping[str, Any] | None = None,
) -> TrainedRanker:
    """Fit the ranker, early-stopping on validation NDCG@10.

    ``extra_params`` exists for tiny synthetic datasets in the tests, which need
    ``min_data_in_leaf`` lowered below LightGBM's default or no tree ever
    splits. Production runs pass nothing and get exactly the config's values.
    """
    import lightgbm as lgb

    if val is not None and val.features.names != train.features.names:
        # Would train on one column order and early-stop on another. LightGBM
        # does not check this for raw numpy input, so it has to be checked here.
        raise ValueError(
            f"train features {train.features.names} differ from "
            f"validation features {val.features.names}"
        )

    params = lgb_params(cfg, extra_params)
    train_set = lgb.Dataset(
        train.features.values,
        label=train.labels,
        group=train.group,
        feature_name=list(train.features.names),
        free_raw_data=False,
    )
    # The training set is evaluated too (so the metadata can show the train/val
    # gap), and it MUST be named "training". LightGBM's early-stopping callback
    # skips the training data by *name*, comparing against
    # `Booster._train_data_name`, whose default is exactly "training". Name it
    # anything else — "train", say — and the callback treats it as a validation
    # set: train NDCG rises monotonically, so at the final round the callback
    # reports that entry's best iteration, i.e. the last one. Early stopping
    # then silently has no effect and the saved model is the overfitted final
    # model. This one string is the difference.
    valid_sets = [train_set]
    valid_names = ["training"]
    callbacks: list[Callable[..., Any]] = [lgb.log_evaluation(period=0)]
    if val is not None:
        valid_sets.append(
            lgb.Dataset(
                val.features.values,
                label=val.labels,
                group=val.group,
                feature_name=list(val.features.names),
                reference=train_set,
                free_raw_data=False,
            )
        )
        valid_names.append("val")
        callbacks.append(
            lgb.early_stopping(cfg.ltr.early_stopping_rounds, first_metric_only=True, verbose=False)
        )

    evals: dict[str, dict[str, list[float]]] = {}
    callbacks.append(lgb.record_evaluation(evals))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.ltr.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    val_ndcg = None
    if val is not None:
        history = evals.get("val", {}).get(f"ndcg@{EARLY_STOPPING_NDCG_K}", [])
        if history:
            # `best_iteration` is 1-based when early stopping fired, 0/-1 when it
            # did not; index defensively rather than trusting one convention.
            index = booster.best_iteration - 1 if booster.best_iteration > 0 else len(history) - 1
            val_ndcg = float(history[max(0, min(index, len(history) - 1))])

    return TrainedRanker(
        booster=booster,
        feature_names=train.features.names,
        best_iteration=int(booster.best_iteration or booster.current_iteration()),
        val_ndcg_at_10=val_ndcg,
        n_train_rows=train.features.n_rows,
        n_train_queries=train.n_queries,
        n_val_rows=0 if val is None else val.features.n_rows,
        n_val_queries=0 if val is None else val.n_queries,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Artifacts
# ─────────────────────────────────────────────────────────────────────────────


def save_ranker(
    ranker: TrainedRanker,
    out_dir: str | Path,
    cfg: Config,
    *,
    manifest: synth_splits.SplitManifest | None = None,
    manifest_path: str | Path | None = None,
    crossencoder_run: Mapping[str, Any] | None = None,
    ablated_non_text: bool = False,
    ablated_constraints: bool = False,
    gold_check_ran: bool = True,
) -> Path:
    """Write model, feature names and metadata; return the directory.

    Feature names are saved twice on purpose: inside the LightGBM model file
    (so ``booster.feature_name()`` is right) and in a standalone JSON the
    serving path can assert against before it builds a single row. The
    duplication is cheap insurance against the failure this cannot detect at
    runtime — columns in the wrong order produce numbers, not errors.

    The metadata contains only measured facts: sizes, the best iteration, the
    validation NDCG actually recorded, and importances read off the model. No
    field is a placeholder; where something did not happen (no validation split,
    gold check skipped) the JSON says so explicitly.
    """
    directory = _resolve(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / "model.txt"
    ranker.booster.save_model(str(model_path), num_iteration=ranker.best_iteration)
    (directory / "feature_names.json").write_text(
        json.dumps(list(ranker.feature_names), indent=2) + "\n"
    )

    metadata: dict[str, Any] = {
        "config_fingerprint": cfg.fingerprint(),
        "seed": cfg.seed,
        "objective": cfg.ltr.objective,
        "feature_names": list(ranker.feature_names),
        "ablated_non_text_features": ablated_non_text,
        # Recorded so a model file can never be mistaken for the full-feature run.
        "ablated_constraint_features": ablated_constraints,
        "non_text_features": list(NON_TEXT_FEATURE_NAMES),
        "best_iteration": ranker.best_iteration,
        "n_estimators_max": cfg.ltr.n_estimators,
        "early_stopping_rounds": cfg.ltr.early_stopping_rounds,
        f"val_ndcg_at_{EARLY_STOPPING_NDCG_K}": ranker.val_ndcg_at_10,
        "n_train_rows": ranker.n_train_rows,
        "n_train_queries": ranker.n_train_queries,
        "n_val_rows": ranker.n_val_rows,
        "n_val_queries": ranker.n_val_queries,
        "importance_gain": ranker.importances("gain"),
        "importance_split": ranker.importances("split"),
        "enforce_oof_crossencoder_feature": cfg.ltr.enforce_oof_crossencoder_feature,
        "gold_leakage_check_ran": gold_check_ran,
        "split_manifest": None if manifest_path is None else str(manifest_path),
        "split_sizes": None if manifest is None else manifest.counts,
        "split_seed": None if manifest is None else manifest.seed,
        # The checksum is what makes "trained under this split" a checkable
        # claim rather than a sentence: it covers the assignment, the seed, the
        # fractions and the grouping, so a later reader can prove the artifact
        # and the cross-encoder's artifact refer to the same split.
        "split_manifest_checksum": None if manifest is None else manifest.checksum,
        "split_group_by": None if manifest is None else manifest.group_by,
        # Whether the cross-encoder's own run metadata was found and agreed
        # with this split, and when it was not, why. "Not checked" and "checked
        # and clean" must never look the same in an artifact.
        "crossencoder_run_check": dict(crossencoder_run) if crossencoder_run else None,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def default_out_dir(cfg: Config, ablated_non_text: bool, ablated_constraints: bool = False) -> Path:
    """``artifacts/ltr/<config fingerprint>[-textonly|-noconstraints]``.

    Keyed by the config hash (D008) so a model can always be traced to the
    settings that produced it, and suffixed per ablation so the runs cannot
    overwrite each other — which would make the ablation table unreproducible in
    the most confusing possible way.
    """
    suffix = ""
    if ablated_non_text:
        suffix = "-textonly"
    elif ablated_constraints:
        suffix = "-noconstraints"
    return DEFAULT_OUT_ROOT / f"{cfg.fingerprint()}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# The run, and the CLI
# ─────────────────────────────────────────────────────────────────────────────


def run(
    cfg: Config,
    examples_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    ablate_non_text: bool = False,
    ablate_constraints: bool = False,
    out_dir: str | Path | None = None,
    allow_missing_gold: bool = False,
) -> Path:
    """Full training run: checks, data, fit, save. Returns the artifact dir.

    Order matters. Every leakage check runs **before** any training, so a
    contaminated run costs seconds rather than producing a model somebody might
    later evaluate by accident.

    ``examples_path`` and ``manifest_path`` both default to ``None``, meaning
    "the fingerprinted paths this config implies"
    (:func:`default_examples_path`, :func:`default_manifest_path`). The two
    defaults resolve into the *same* directory, which is the point: a
    feature-row file and the manifest its rows were split under belong to one
    configuration, and separating them is how a stale examples file gets paired
    with a fresh manifest. Passing either explicitly is supported for tests and
    for artifacts kept outside the repo, and both are verified exactly as hard —
    the manifest against this config's fingerprint, the rows against the
    manifest.
    """
    manifest, resolved_manifest = load_verified_manifest(cfg, manifest_path)
    if examples_path is None:
        examples_path = default_examples_path(cfg)
    enforce = cfg.ltr.enforce_oof_crossencoder_feature

    # Every row, then the manifest decides which ones train. Rows are never
    # selected by the `split` string they carry — see select_split_rows.
    examples = load_examples(examples_path)
    train_examples = select_split_rows(examples, manifest, SPLIT_LTR_TRAIN, enforce=enforce)
    val_examples = select_split_rows(examples, manifest, SPLIT_LTR_VAL, enforce=enforce)
    if not train_examples:
        raise ValueError(
            f"no examples the manifest assigns to {SPLIT_LTR_TRAIN!r} in {examples_path}. "
            "LambdaMART trains on the ltr_train split only."
        )

    # The run-followed-the-plan check, over the ids actually in hand, plus the
    # cross-artifact check that the cross-encoder was fine-tuned under this very
    # split. Both before a single boosting round.
    assert_run_followed_the_manifest([*train_examples, *val_examples], manifest)
    crossencoder_check = crossencoder_run_check(cfg, manifest)

    assert_no_gold_leakage(
        [*train_examples, *val_examples],
        cfg.evaluation.gold_queries,
        allow_missing_gold=allow_missing_gold,
    )

    names = ablated_feature_names(ablate_non_text, drop_constraints=ablate_constraints)
    train_data = build_ranking_data(train_examples, names)
    val_data = build_ranking_data(val_examples, names) if val_examples else None
    if val_data is None:
        log.warning(
            "no %s examples: training without early stopping, for the full %d rounds",
            SPLIT_LTR_VAL,
            cfg.ltr.n_estimators,
        )

    ranker = train_lambdamart(train_data, val_data, cfg)
    target = (
        default_out_dir(cfg, ablate_non_text, ablate_constraints) if out_dir is None else out_dir
    )
    directory = save_ranker(
        ranker,
        target,
        cfg,
        manifest=manifest,
        manifest_path=resolved_manifest,
        crossencoder_run=crossencoder_check,
        ablated_non_text=ablate_non_text,
        ablated_constraints=ablate_constraints,
        gold_check_ran=_resolve(cfg.evaluation.gold_queries).exists(),
    )
    log.info(
        "trained on %d rows / %d queries; best_iteration=%d; artifacts in %s",
        ranker.n_train_rows,
        ranker.n_train_queries,
        ranker.best_iteration,
        directory,
    )
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the LambdaMART final ranker.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--examples",
        default=None,
        help=(
            "JSONL of labeled (query, candidate) pairs with stage scores. "
            "Defaults to artifacts/synth/<config fingerprint>/ltr_examples.jsonl, "
            "beside the split manifest these rows must agree with. No stage in "
            "this repository writes that file yet; see default_examples_path."
        ),
    )
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Split manifest written by the synthetic-data stage. Defaults to "
            "artifacts/synth/<config fingerprint>/splits.json, which is where that "
            "stage writes it. A manifest given here is verified exactly as hard, "
            "including that it was written for this config's fingerprint."
        ),
    )
    parser.add_argument(
        "--ablate-non-text",
        action="store_true",
        help=(
            "Train on the stage-score features only, dropping rating, "
            "popularity, constraint and distance features. This is the "
            "measurement behind 'do ratings and popularity matter?'."
        ),
    )
    parser.add_argument(
        "--ablate-constraints",
        action="store_true",
        help=(
            "Train without the hard-constraint block only, keeping rating, "
            "popularity and distance. This is the measurement behind 'does "
            "constraint arithmetic matter, as distinct from similarity and from "
            "quality signals?' — which --ablate-non-text cannot answer, because "
            "it drops all three blocks at once. See DECISIONS.md D225."
        ),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--allow-missing-gold",
        action="store_true",
        help="Train even though eval/gold_queries.csv does not exist yet (logged).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    run(
        cfg,
        # None for either path means "the fingerprinted path this config
        # implies"; run() resolves both, so there is exactly one place that
        # knows each default and the CLI cannot drift from the library call.
        args.examples,
        args.splits,
        ablate_non_text=args.ablate_non_text,
        ablate_constraints=args.ablate_constraints,
        out_dir=args.out_dir,
        allow_missing_gold=args.allow_missing_gold,
    )
    return 0


def _resolve(path: str | Path) -> Path:
    """Relative paths are relative to the repo root, not the shell's cwd.

    Config paths like ``eval/qrels.csv`` are repo-relative; resolving them
    against the working directory would make `make train-ltr` behave
    differently depending on where it was run from.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
