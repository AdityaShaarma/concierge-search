"""The stage between the fine-tune and Stage 4: candidate-level feature rows.

    python -m concierge.ltr.build_examples --config configs/base.yaml

What was missing, and why it had to be its own stage
----------------------------------------------------
``make synth-splits`` writes *query*-level files — ``ltr_train.jsonl`` and
``ltr_val.jsonl``, one row per synthetic query, each carrying its source business
and its judged hard negatives (``concierge.synth.negatives.TrainingExample``).
``concierge.ltr.train`` reads something else entirely: *candidate*-level rows,
one per (query, retrieved business) pair, carrying the Stage 2 scores and ranks,
the fine-tuned cross-encoder score, the evaluated constraints and a relevance
label (``concierge.ltr.train.load_examples``). Nothing turned the first into the
second, so Stage 4 could not be trained at all.

It cannot be folded into either neighbour. Every row needs a cross-encoder score,
so it can only be produced *after* Stage 3 is fine-tuned; and the fine-tune reads
a different split, so it cannot produce these rows as a side effect. Hence a
separate module, run between ``make train-reranker`` and ``make train-ltr``.

What it does, per query: run the real cascade — hybrid retrieval, the
constraint-only backfill, the hard filters, the fine-tuned cross-encoder — and
emit one JSONL row per *labeled* surviving candidate.

The four properties this module exists to hold
----------------------------------------------
**1. The cross-encoder feature is out of fold.** This is the whole reason the
stage is separate rather than a loop inside the trainer. The cross-encoder score
is one of Stage 4's strongest features; computed for queries the cross-encoder
was fine-tuned on, it is inflated, LambdaMART learns to over-trust it, and served
quality drops while the training numbers look fine. Nothing errors. So
:func:`assert_crossencoder_is_out_of_fold` checks four facts before a single
model is loaded, and **hard-fails on all of them** — there is no warn-and-continue
path here, because a warning on a batch job is a line of scrollback:

  a. the manifest is loaded through ``concierge.synth.splits.require_disjoint_splits``
     (via ``concierge.ltr.train.load_verified_manifest``), which verifies the
     checksum, the partition, the cross-encoder/LambdaMART disjointness and the
     seed and fractions against this config — the one implementation of that
     guarantee in the repository, called and not copied;
  b. every query read from ``<split>.jsonl`` must be assigned to that split *by
     the manifest* (``concierge.synth.splits.split_of``), so a stale per-split
     file cannot smuggle a cross-encoder query into the LambdaMART folds;
  c. the ids actually being built are checked against the manifest's plan with
     ``concierge.synth.splits.assert_models_trained_on_disjoint_queries``;
  d. the checkpoint supplying the score must be the one fine-tuned under *this*
     manifest, which is ``concierge.ltr.train.crossencoder_run_check`` — the same
     evidence Stage 4 will later demand, deliberately reused rather than
     re-derived. This module additionally refuses the case that function reports
     rather than raises on (no fine-tuned run metadata at all): for the trainer
     that means "the feature came from the pretrained model, which saw no
     synthetic query"; for the *producer* it means there is no fine-tuned
     checkpoint to score with, which is a missing prerequisite.

**2. No train/serve skew.** This module never computes a feature, and it no
longer restates a single stage. :class:`Cascade` runs
``concierge.pipeline.SearchPipeline.candidates`` — the same method ``search()``
itself calls — so the stage order here *is* the served stage order rather than a
copy of it that has to be kept in step. It then writes ``Candidate.scores`` and
``Candidate.ranks`` out under their own canonical keys
(``concierge.types.SCORE_*`` / ``RANK_*``) and lets
``concierge.ltr.features.build_feature_row`` do the arithmetic at training time
exactly as it does at request time.

This module used to restate the order of retrieve/filter/rerank, because the
pipeline exposed no candidate-level entry point: ``search()`` returns rendered
``SearchResult``s, which carry the per-stage scores but have dropped the
per-stage *ranks* that are two of Stage 4's features. A test pinned the
restatement against a real ``SearchPipeline``, which caught a disagreement but
could not prevent one — see DECISIONS.md D228. ``SearchPipeline.candidates``
closed that gap, and the test became
``test_the_producer_and_search_agree_on_the_cascade``: cheaper, and strong enough
that no copy of the order survives to drift.

**3. Gold leakage.** These queries are synthetic and were deduplicated against
the gold set two stages upstream, but the output is still checked with
``concierge.ltr.train.assert_no_gold_leakage`` — the same function the trainer
calls, run here on the rows about to be written, so a leak is caught by the stage
that would have introduced it.

**4. Determinism.** Same inputs, byte-identical file. This stage draws no random
numbers at all, so determinism is a property of ordering and tie-breaks rather
than of a seed:

* splits are visited in ``concierge.synth.splits.LTR_SPLITS`` order and queries
  within a split in sorted ``query_id`` order;
* candidates within a query come out in cascade order, which is deterministic by
  construction: ``fusion.fuse`` breaks fused-score ties on ``business_id``, and
  ``CrossEncoderReranker.rerank`` is a *stable* sort, so cross-encoder ties keep
  fused order;
* rows are written with ``sort_keys=True``, so the key order in the file does not
  depend on dict insertion order;
* ``ParsedQuery.parse_latency_ms`` is deliberately **not** serialized. It is a
  measured wall-clock number, it would differ on every run, and no Stage 4
  feature reads it — serializing it would have made the file non-reproducible for
  nothing.

Each query's rows are written as one contiguous block, which is what
``concierge.ltr.train.group_sizes`` requires of the group array.

Everything is injectable: :class:`Cascade` wraps a ``SearchPipeline``, whose own
parts (retrievers, reranker, business records, document texts) are all
constructor arguments, so the tests run the whole stage against
``tests/fixtures/`` with fake retrievers and a fake pair scorer — no dataset, no
checkpoint, no API key.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from concierge.config import REPO_ROOT, Config, load_config
from concierge.ltr import train as ltr_train
from concierge.pipeline import (
    STAGE_FILTER,
    STAGE_RETRIEVE,
    VARIANT_FULL,
    RerankCheckpoint,
    SearchPipeline,
    Stage,
    StageRecorder,
    Variant,
)
from concierge.rerank.model import training_parse
from concierge.synth import splits as synth_splits
from concierge.types import (
    RELEVANCE_HIGH,
    RELEVANCE_IRRELEVANT,
    Business,
    Candidate,
    ConstraintMatch,
    GeoConstraint,
    OpenAt,
    ParsedQuery,
)

if TYPE_CHECKING:
    # Type-only: `concierge.synth.negatives` pulls in the mining and generation
    # machinery, and this stage needs only its reader, which `run` imports at
    # call time (the same arrangement `concierge.synth.splits` uses for the same
    # module). Importing it here for the annotation keeps the folds typed without
    # that cost at import time.
    from concierge.synth.negatives import HardNegative, TrainingExample

log = logging.getLogger(__name__)

#: The two folds this stage builds rows for, taken from the module that owns the
#: split vocabulary so a rename there is an ImportError here rather than a stage
#: that silently builds nothing.
SPLIT_LTR_TRAIN, SPLIT_LTR_VAL = synth_splits.LTR_SPLITS

#: Counts and skip reasons for the run, written beside the examples file.
REPORT_FILENAME = "ltr_examples_report.json"

# ── Labels ───────────────────────────────────────────────────────────────────
#
# Graded relevance, on the same 0/1/2 scale the owner labels the gold set with
# (``concierge.types.RELEVANCE_*``), so a training label and a test label mean
# the same thing. Only two of the three grades are used, because only two kinds
# of candidate carry evidence:
#
#   * the query's **source business** — the synthetic query was generated *from*
#     that business's document, so it satisfies the request by construction.
#     Graded HIGH (2).
#   * a **hard negative that survived the judge** — mined from the top of hybrid
#     retrieval for this query, then put to the LLM judge by
#     ``concierge.synth.judge``, which removed the ones it graded relevant.
#     What is left is the evidence this project has for "retrieved and not
#     relevant". Graded IRRELEVANT (0).
#
# A candidate that is neither is **not emitted at all**. See
# :func:`rows_for_query` for that decision and the alternative.
LABEL_POSITIVE = RELEVANCE_HIGH
LABEL_HARD_NEGATIVE = RELEVANCE_IRRELEVANT


class BuildError(RuntimeError):
    """A prerequisite is missing, or an input is self-contradictory.

    Distinct from ``concierge.ltr.train.LeakageError`` on purpose: a leakage
    failure means a number could not be reported honestly, while this means the
    stage was run too early or against the wrong artifacts. The tests assert on
    which one happened, and the two have different fixes.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────


def default_output_path(cfg: Config, root: Path | None = None) -> Path:
    """Where this stage writes: ``<synth dir>/ltr_examples.jsonl``.

    Delegated to ``concierge.ltr.train.default_examples_path`` — the *reader's*
    own answer to "where do the feature rows live" — rather than spelled out
    again. A producer and a consumer that each know the path independently is how
    a stage ends up writing a file nothing reads, which is the state this module
    was written to end.
    """
    return ltr_train.default_examples_path(cfg, root)


def query_level_path(split: str, cfg: Config, root: Path | None = None) -> Path:
    """One LambdaMART fold's query-level file, or a message naming the fix.

    ``concierge.synth.splits.split_example_path`` is the authority on the
    filename; this adds only the existence check and the command to run, because
    "no such file" with no next step is the error this project's CLIs exist not
    to emit.
    """
    path = synth_splits.split_example_path(split, cfg, root)
    if not path.exists():
        raise BuildError(
            f"no query-level file for split {split!r} at {path}.\n"
            "Run `make synth` (or just `make synth-splits` if the earlier stages "
            "have already run) for this config: that stage writes one JSONL per "
            "split, and this stage turns the LambdaMART folds into the "
            "candidate-level rows Stage 4 trains on."
        )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# The out-of-fold guarantee
# ─────────────────────────────────────────────────────────────────────────────


def assert_queries_match_the_manifest(
    query_ids: Sequence[str],
    manifest: synth_splits.SplitManifest,
    split: str,
    path: Path,
) -> None:
    """Every query in ``path`` must be assigned to ``split`` by the manifest.

    The manifest decides, never the filename. A per-split file left over from an
    earlier manifest is the realistic shape of this leak: it is still called
    ``ltr_train.jsonl``, it still parses, and the queries in it may now belong to
    a cross-encoder split. Then the rows this stage builds would carry
    cross-encoder scores for queries the cross-encoder was fitted on, and every
    later check would report clean, because at the level of ids nothing is wrong.

    ``concierge.synth.splits.split_of`` is called rather than
    ``manifest.assignment.get``, because it already refuses an unassigned query
    for exactly this reason and there must be one implementation of that refusal.
    """
    wrong: list[tuple[str, str]] = []
    for query_id in query_ids:
        # Raises SplitLeakageError when the manifest does not mention the query.
        assigned = synth_splits.split_of(manifest, query_id)
        if assigned != split:
            wrong.append((query_id, assigned))
    if wrong:
        raise ltr_train.LeakageError(
            f"{len(wrong)} query/queries in {path} are assigned to another split by "
            f"the manifest (e.g. {sorted(wrong)[:3]} as (query_id, manifest split)), "
            f"but this file is being read as {split!r}. The feature rows would carry "
            "cross-encoder scores for queries the cross-encoder may have been "
            "fine-tuned on. Regenerate the per-split files with `make synth-splits`."
        )


def assert_crossencoder_is_out_of_fold(
    cfg: Config,
    manifest: synth_splits.SplitManifest,
    query_ids: Sequence[str],
) -> dict[str, Any]:
    """Prove the checkpoint about to score these queries never trained on them.

    Returns the evidence, which is recorded in the run's report so that "checked
    and clean" and "not checked" can never look the same in an artifact.

    Three facts, all of them refusals:

    1. the ids being built are disjoint from the cross-encoder's splits, checked
       against the manifest's own plan by
       ``concierge.synth.splits.assert_models_trained_on_disjoint_queries``;
    2. the fine-tuned checkpoint recorded the *same* manifest checksum this run
       verified, and is not a ``--limit-examples`` smoke run — both of which are
       ``concierge.ltr.train.crossencoder_run_check``, called rather than
       reimplemented so this stage and the trainer agree on what the evidence is;
    3. that run metadata **exists**. The trainer treats its absence as "the
       feature came from the pretrained model, which was fitted on no synthetic
       query and therefore cannot be in fold" — true for a reader, useless for a
       writer: this stage's entire job is to attach the *fine-tuned* score, so a
       missing fine-tune is a missing prerequisite, not a benign case.
    """
    synth_splits.assert_models_trained_on_disjoint_queries(
        manifest.ids(*synth_splits.CROSSENCODER_SPLITS),
        query_ids,
        manifest,
    )
    # Raises LeakageError on a checksum mismatch, a missing checksum, a smoke
    # run, or a checkpoint fine-tuned under another config fingerprint.
    evidence = ltr_train.crossencoder_run_check(cfg, manifest)
    if not evidence.get("checked"):
        from concierge.rerank.model import RUN_METADATA_FILENAME, rerank_dir

        raise BuildError(
            f"no fine-tuned cross-encoder run metadata at "
            f"{rerank_dir(cfg) / RUN_METADATA_FILENAME} "
            f"({evidence.get('reason', 'reason not recorded')}).\n"
            "Every row this stage writes carries a fine-tuned cross-encoder score "
            "as a Stage 4 feature, so there is nothing to build until Stage 3 has "
            "been fine-tuned for this config. Run `make train-reranker`."
        )
    return evidence


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: the parse
# ─────────────────────────────────────────────────────────────────────────────


def parse_for_training(query: str) -> ParsedQuery:
    """The ``ParsedQuery`` a training row is built against.

    ``concierge.rerank.model.training_parse`` — *the* parse the fine-tuned
    checkpoint's training text came from, called rather than re-derived. Three
    properties follow from that and all three matter here:

    * the string scored by the cross-encoder in this stage is byte-identical to
      the string it was fine-tuned on, because both are ``query_text_for`` of
      this same parse. A different parse would reopen exactly the skew
      ``training_query_text`` was written to close, on the Stage 4 side;
    * the weekday is fixed, so "open past 10pm" resolves to the same ``OpenAt``
      on every run and the hours filter keeps the same businesses. A clock read
      would make this file non-reproducible;
    * it is the **rule** parser, not the LLM one: deterministic, no API key
      (so the stage is runnable and testable offline), and the parser that serves
      when the LLM path fails, so the filters here are ones the served system
      really applies.

    The residual, stated rather than hidden: when the LLM parser is the one
    serving, its ``semantic_text`` and its constraints can differ from the rule
    parser's on the same query, so Stage 4 is trained on rule-parsed rows and may
    serve LLM-parsed ones. That gap is measurable — the eval harness ablates the
    parser (``concierge.pipeline.ABLATION``) — and closing it by calling the LLM
    over a few thousand synthetic queries would spend quota to make the stage
    non-reproducible.

    Kept as a named function rather than inlined at the call site: it is where
    this stage's Stage 1 decision is documented, and it is the seam the tests
    assert on.
    """
    return training_parse(query)


# ─────────────────────────────────────────────────────────────────────────────
# Stages 2 and 3: the cascade
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """One query's candidates, with the counts the report needs.

    The two counts are measurements, not diagnostics: ``n_retrieved`` minus
    ``n_after_filter`` is how much the hard constraints removed from this query's
    pool, and a fold where that is most of the pool is the reason a later
    LambdaMART run has few rows.
    """

    candidates: tuple[Candidate, ...]
    n_retrieved: int
    n_after_filter: int


#: The configuration these rows are built under: every stage the served system
#: runs up to and including Stage 3.
#:
#: Two flags are off, for different reasons. ``Stage.LTR`` because these rows are
#: Stage 4's *training input*: running Stage 4 over them would feed the model its
#: own prediction as a feature. ``Stage.QUERY_LLM`` because Stage 1 happens in
#: :func:`parse_for_training`, with the rule parser, for the reasons given there —
#: the flag is not read by ``SearchPipeline.candidates`` at all, and leaving it on
#: would claim a parser this stage does not use.
CASCADE_VARIANT: Final[Variant] = Variant(
    name="ltr_examples_cascade",
    stages=VARIANT_FULL.stages & ~Stage.LTR & ~Stage.QUERY_LLM,
    reranker=RerankCheckpoint.FINETUNED,
)


@dataclass(frozen=True, slots=True)
class Cascade:
    """Stages 2 and 3 for one parsed query, returning ``Candidate`` objects.

    A thin wrapper over ``SearchPipeline.candidates`` — the method ``search()``
    itself calls — so this stage does not restate the cascade order and cannot
    drift from it. What the wrapper adds is the two things a producer needs and a
    request does not: a loader that *refuses* missing artifacts instead of
    degrading (:meth:`from_config`), and the per-stage counts
    (:class:`CascadeResult`).

    It holds a whole pipeline rather than the individual indexes because the
    pipeline is where the stage order, the candidate widths and the missing-data
    policy already live; taking them apart here is what created the duplication
    this class used to carry (DECISIONS.md D228).

    Note what is *not* needed: a clock. ``SearchPipeline``'s injected ``now`` is
    read only by Stage 1, which this stage does for itself
    (:func:`parse_for_training`), so the rows cannot depend on the day the
    producer ran.

    The pipeline is a constructor argument, which is what makes this stage
    testable with no dataset and no checkpoint; :meth:`from_config` is the real
    loader.
    """

    pipeline: SearchPipeline
    #: Which stages to run. A field so a diagnostic run can build rows under
    #: another configuration; the default is the one Stage 4 is trained on.
    variant: Variant = CASCADE_VARIANT

    @property
    def cfg(self) -> Config:
        return self.pipeline.cfg

    @property
    def businesses(self) -> Mapping[str, Business]:
        """The business records, read from the pipeline so there is one copy."""
        return self.pipeline.businesses

    @property
    def documents(self) -> Mapping[str, str]:
        """``business_id -> document text``: what the cross-encoder reads."""
        return self.pipeline.doc_text

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        artifacts_root: Path | None = None,
        processed_dir: Path | None = None,
    ) -> Cascade:
        """Load the real indexes, documents and fine-tuned checkpoint, or refuse.

        Nothing degrades here, unlike ``SearchPipeline.from_config``. That class
        has to be runnable at every point in the project's own construction, so a
        missing artifact warns and the cascade shortens. This stage is the
        opposite case: a shortened cascade would still write a plausible file,
        and the rows in it would carry a BM25-only fused score, or a *pretrained*
        cross-encoder score, under a filename Stage 4 reads as the fine-tuned
        system. So each missing artifact is an error naming the target that
        builds it.

        The checkpoint is loaded **by explicit path** rather than through
        ``resolve_weights``, whose documented fallback hands back the pretrained
        model when no fine-tune exists. Here that fallback would silently relabel
        the pretrained model's scores as the fine-tuned feature.
        """
        from concierge.data.build import read_businesses, read_documents, resolve_processed_dir
        from concierge.rerank.model import (
            CHECKPOINT_DIRNAME,
            CrossEncoderReranker,
            rerank_dir,
        )
        from concierge.retrieval.bm25 import BM25Index, index_dir
        from concierge.retrieval.dense import dense_index_dir, load_dense_index

        processed = resolve_processed_dir(cfg) if processed_dir is None else Path(processed_dir)
        businesses = {b.business_id: b for b in read_businesses(processed)}
        documents = {d.business_id: d.text for d in read_documents(processed)}

        bm25_dir = index_dir(cfg, artifacts_root)
        if not bm25_dir.exists():
            raise BuildError(
                f"no BM25 index at {bm25_dir}. Run `make index` for this config: "
                "these rows must carry the lexical score the served system produces."
            )
        dense_dir = dense_index_dir(cfg, artifacts_root)
        if not dense_dir.exists():
            raise BuildError(
                f"no dense index at {dense_dir}. Run `make index` for this config: "
                "rows built from BM25 alone would give Stage 4 a NaN dense_score on "
                "every candidate, which is not the system it will serve."
            )

        checkpoint = rerank_dir(cfg, artifacts_root) / CHECKPOINT_DIRNAME
        if not (checkpoint / "config.json").exists():
            raise BuildError(
                f"no fine-tuned cross-encoder checkpoint at {checkpoint}. "
                "Run `make train-reranker` for this config; the cross_encoder_score "
                "feature must come from the fine-tuned model, never the pretrained one."
            )

        # The checkpoint is registered under the FINETUNED key only. The pipeline
        # falls back pretrained -> none when a wanted checkpoint is absent, and
        # that fallback must not be reachable here: `CASCADE_VARIANT` asks for
        # the fine-tuned model, the key is present, and nothing else is.
        return cls(
            pipeline=SearchPipeline(
                cfg,
                businesses=businesses,
                documents=documents,
                bm25_index=BM25Index.load(bm25_dir),
                dense_index=load_dense_index(cfg, directory=dense_dir),
                reranker={
                    RerankCheckpoint.FINETUNED: CrossEncoderReranker.from_config(
                        cfg, model_path=checkpoint
                    )
                },
                ltr_ranker=None,
                variant=CASCADE_VARIANT,
            )
        )

    def run(self, parsed: ParsedQuery) -> CascadeResult:
        """Retrieve, backfill, filter, rerank — via the serving pipeline itself.

        Nothing about the order is decided here: ``SearchPipeline.candidates``
        owns it, and the counts below are the pipeline's own per-stage counts
        rather than a second tally of the same lists.
        """
        record = StageRecorder()
        candidates = self.pipeline.candidates(parsed, variant=self.variant, record=record)
        return CascadeResult(
            candidates=candidates,
            n_retrieved=record.counts[STAGE_RETRIEVE],
            n_after_filter=record.counts[STAGE_FILTER],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Labeling one query's candidates
# ─────────────────────────────────────────────────────────────────────────────

#: Why a query contributed no rows. Recorded per query rather than summed, so the
#: report can say which queries were dropped and a reader can go and look at one.
SKIP_POSITIVE_ABSENT = "positive_not_in_candidates"
SKIP_NO_NEGATIVE = "no_labeled_negative_in_candidates"


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """What one query produced, including the case where it produced nothing.

    The two negative counts split this query's label-0 rows by what their label
    rests on: an actual LLM verdict (``HardNegative.verdict``), or the absence of
    one. See :class:`BuildReport` for why that split is reported.
    """

    query_id: str
    rows: tuple[dict[str, Any], ...]
    n_candidates: int
    n_unlabeled: int
    n_negative_rows_judged: int = 0
    n_negative_rows_unjudged: int = 0
    skipped: str = ""


def rows_for_query(
    *,
    query_id: str,
    split: str,
    parsed: ParsedQuery,
    positive_business_id: str,
    negatives: Sequence[HardNegative],
    candidates: Sequence[Candidate],
    businesses: Mapping[str, Business],
) -> QueryOutcome:
    """One JSONL record per *labeled* candidate, in cascade order.

    **The labeling decision, which is a real judgement call.** Three kinds of
    candidate reach this function:

    * the query's source business — labeled :data:`LABEL_POSITIVE` (2);
    * a hard negative that survived ``concierge.synth.judge`` — labeled
      :data:`LABEL_HARD_NEGATIVE` (0);
    * everything else the cascade surfaced — **not emitted**.

    The third is the decision. The alternative, and the one most LTR pipelines
    take, is to label every other candidate 0 and emit the whole shortlist. It
    was rejected here:

    * Those candidates are, by construction, the ones hybrid retrieval and the
      cross-encoder ranked *highest* for this query, which is exactly the
      population most likely to contain genuinely relevant businesses. Other
      quiet cafés with wifi do exist — that is the premise of
      ``concierge.synth.judge``, which spends LLM quota removing mined negatives
      it grades relevant. Labeling the unjudged remainder 0 would reintroduce the
      same false negatives at a larger scale and with no filter at all.
    * Under ``lambdarank`` the damage is concentrated where the metric is most
      sensitive. Each pair's gradient is scaled by how much swapping those two
      documents would change NDCG, so a mislabeled candidate near the top of the
      list is weighted most heavily of all. The noise would land precisely on the
      rows that decide the model.
    * A third grade (1, "partially relevant") for the unjudged remainder was
      rejected faster: nothing measured it, and inventing a label is the one
      thing this project may not do.

    **What that costs, stated rather than glossed.** Each group is then a
    positive against a handful of judged-irrelevant negatives, which is a
    narrower and *easier* list than the ~50 the served system ranks: the
    plausible-but-unjudged middle of the shortlist is absent from training.
    LambdaMART therefore learns from a biased sample of the serving
    distribution. The count of candidates dropped this way is reported per run
    (``n_unlabeled``) so the size of that gap is visible, and the way to close it
    is to judge deeper — a measurable change in ``concierge.synth.judge``'s
    ``top_n`` with a known API cost — not to guess here.

    **Two kinds of query are skipped entirely**, and both are counted:

    * the positive is not among the survivors (the retriever missed it, a hard
      filter dropped it, or the reranker cut it). Emitting the positive anyway
      with no stage scores would be worse than dropping the query: at serving
      time a candidate with no scores never reaches Stage 4 at all, so such a row
      teaches the model that missing scores mean relevance — a shortcut that
      would show up as excellent training NDCG and nothing else.
    * no labeled negative survives. A group with a single label value yields no
      pairs for ``lambdarank`` to learn from, and on the validation fold it is
      worse than useless: a one-row group scores a perfect NDCG by definition, so
      keeping them would inflate the early-stopping metric.

    **The label-0 evidence is split, not assumed.** ``negatives`` arrives as the
    mined records, not as bare ids, so each emitted label-0 row can be attributed
    to either an actual LLM verdict or to never having been graded
    (``HardNegative.judged``; the judge grades only the hardest few per query).
    The two counts travel on the :class:`QueryOutcome` and are summed into the
    run's report. This does not change any label — D227 settled that — it makes
    the evidence behind a 0 recoverable from the artifact instead of needing
    someone to remember the judge's ``top_n``.
    """
    negative_ids = {n.business_id for n in negatives}
    judged_ids = {n.business_id for n in negatives if n.judged}
    if positive_business_id in negative_ids:
        # A mined example cannot have its own positive as a negative; if one
        # does, the upstream artifact is self-contradictory and guessing which
        # label it meant would pick the label silently.
        raise BuildError(
            f"query {query_id!r} lists its positive {positive_business_id!r} among its "
            "hard negatives. Regenerate the synthetic examples (`make synth`); this "
            "stage will not choose a label for a contradictory row."
        )

    rows: list[dict[str, Any]] = []
    n_unlabeled = 0
    positive_seen = False
    negatives_seen = 0
    n_judged = 0

    for candidate in candidates:
        if candidate.business_id == positive_business_id:
            label = LABEL_POSITIVE
            positive_seen = True
        elif candidate.business_id in negative_ids:
            label = LABEL_HARD_NEGATIVE
            negatives_seen += 1
            if candidate.business_id in judged_ids:
                n_judged += 1
        else:
            n_unlabeled += 1
            continue

        business = businesses.get(candidate.business_id)
        if business is None:
            # `apply_filters` drops candidates with no business record, so this
            # means the index and the business table have drifted apart — the
            # same build bug `CrossEncoderReranker.rerank` refuses a missing
            # document for, and it cannot be papered over: the feature row needs
            # the rating, the review count and the coordinates.
            raise BuildError(
                f"candidate {candidate.business_id!r} (query {query_id!r}) has no "
                "business record. The retrieval index and the processed businesses "
                "disagree — rebuild both with `make data` and `make index`."
            )
        rows.append(
            record_for_candidate(
                query_id=query_id,
                split=split,
                label=label,
                parsed=parsed,
                candidate=candidate,
                business=business,
            )
        )

    skipped = ""
    if not positive_seen:
        skipped = SKIP_POSITIVE_ABSENT
    elif negatives_seen == 0:
        skipped = SKIP_NO_NEGATIVE
    if skipped:
        # The negative counts are zero here, not the counts of the rows that were
        # built and thrown away: nothing from a skipped query reaches the file, so
        # counting them would make the report's totals describe rows that do not
        # exist. The query itself is named in `skipped_queries`.
        return QueryOutcome(
            query_id=query_id,
            rows=(),
            n_candidates=len(candidates),
            n_unlabeled=n_unlabeled,
            skipped=skipped,
        )
    return QueryOutcome(
        query_id=query_id,
        rows=tuple(rows),
        n_candidates=len(candidates),
        n_unlabeled=n_unlabeled,
        n_negative_rows_judged=n_judged,
        n_negative_rows_unjudged=negatives_seen - n_judged,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialization — the exact shape `load_examples` parses
# ─────────────────────────────────────────────────────────────────────────────


def record_for_candidate(
    *,
    query_id: str,
    split: str,
    label: int,
    parsed: ParsedQuery,
    candidate: Candidate,
    business: Business,
) -> dict[str, Any]:
    """One JSONL record, in the shape ``ltr_train.load_examples`` documents.

    ``Candidate.scores`` and ``Candidate.ranks`` are written out **as they are**,
    under the canonical keys the stages set them with
    (``concierge.types.SCORE_*`` / ``RANK_*``). No key is renamed and no value is
    recomputed: that is the anti-skew contract — the reader rebuilds a
    ``Candidate`` from these dicts and ``build_feature_row`` reads the same keys
    it reads at request time, so an offline and an online feature vector for the
    same candidate are the same vector.

    Only the business fields the reader reads are written. ``hours`` and
    ``attributes`` are deliberately omitted: ``train._business_from_json``
    documents that it defaults them to empty because the constraint *outcomes*
    travel on the candidate, so no feature touches them, and carrying every
    business's attribute dict would multiply the file size for nothing.

    ``ParsedQuery.parse_latency_ms`` and ``source`` are not written either — the
    first is a wall-clock measurement that would break byte-reproducibility, the
    second is fixed to ``RULES`` by the reader, which is what this stage uses.
    """
    return {
        "query_id": query_id,
        "split": split,
        "label": int(label),
        "query": {
            "raw": parsed.raw,
            "semantic_text": parsed.semantic_text,
            "open_at": _open_at_json(parsed.open_at),
            "attributes": dict(parsed.attributes),
            "price_max": parsed.price_max,
            "categories": list(parsed.categories),
            "geo": _geo_json(parsed.geo),
        },
        "business": {
            "business_id": business.business_id,
            # Not read by any feature; carried so the file can be read by a human
            # during failure analysis without a join back to the parquet.
            "name": business.name,
            "stars": business.stars,
            "review_count": business.review_count,
            "categories": list(business.categories),
            "latitude": business.latitude,
            "longitude": business.longitude,
        },
        "scores": dict(candidate.scores),
        "ranks": dict(candidate.ranks),
        "constraints": [_constraint_json(match) for match in candidate.constraints],
    }


def _open_at_json(open_at: OpenAt | None) -> dict[str, Any] | None:
    if open_at is None:
        return None
    return {"day": open_at.day, "minute_of_day": open_at.minute_of_day}


def _geo_json(geo: GeoConstraint | None) -> dict[str, Any] | None:
    if geo is None:
        return None
    return {
        "latitude": geo.latitude,
        "longitude": geo.longitude,
        "radius_km": geo.radius_km,
        "label": geo.label,
    }


def _constraint_json(match: ConstraintMatch) -> dict[str, Any]:
    """One evaluated constraint.

    ``name`` and ``status`` are the load-bearing fields — they are the only two
    any Stage 4 feature reads (``features._constraint_features`` and
    ``features._category_match``). ``requested`` and ``actual`` are carried for
    human inspection and for the "why this result" explanation, and they hold
    whatever the checker put there: an ``OpenAt``, a ``GeoConstraint``, a tuple of
    categories, an int, a bool. :func:`_jsonable` renders those, lossily where it
    must, which is safe precisely because no feature reads them — a rendering
    choice here cannot change a feature value.
    """
    return {
        "name": match.name,
        "requested": _jsonable(match.requested),
        "actual": _jsonable(match.actual),
        "status": match.status.value,
    }


def _jsonable(value: Any) -> Any:
    """Render a constraint's requested/actual value as JSON.

    Dataclasses become objects, tuples become lists, primitives pass through, and
    anything else becomes its ``repr``. The fallback is a last resort rather than
    a convenience: it keeps the file writable whatever a future constraint check
    stores, and the field it applies to is read by people, not by models.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, OpenAt):
        return _open_at_json(value)
    if isinstance(value, GeoConstraint):
        return _geo_json(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    return repr(value)


def write_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the records as JSONL, one per line, keys sorted.

    ``sort_keys=True`` is what makes the file byte-identical across runs: without
    it the key order follows dict insertion order, which is stable today and is
    not a property worth depending on. ``ensure_ascii=False`` keeps business
    names readable, matching ``concierge.synth.negatives.write_examples_jsonl``.

    Row order is *not* sorted here. It is the order the caller built — split
    order, then query id, then cascade order within a query — because Stage 4
    requires each query's rows to be one contiguous block
    (``concierge.ltr.train.group_sizes``) and because the order inside a query is
    the served ranking, which is worth preserving in the artifact.
    """
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# The run
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BuildReport:
    """Measured counts for one run. Every field is counted, none is estimated.

    Two fields are easy to over-read, so what they do *not* say:

    * ``n_unlabeled_candidates_dropped`` counts candidates that carried no label
      evidence. It does **not** include the labeled rows of a query that was
      skipped entirely — those queries are named in ``skipped_queries``, which is
      a list of ids rather than a count precisely so that a reader can go and
      look at one instead of reconciling two totals.
    * ``gold_leakage_check_ran`` is whether the gold file existed. The check
      itself cannot have failed and left this report behind: it raises.

    **The two label-0 fields are the evidence check D227 was missing.** Every
    label-0 row comes from a mined hard negative that survived
    ``concierge.synth.judge``, but the judge grades only the hardest few
    negatives of each query, so "survived" means one of two different things:
    *graded, and not called relevant*, or *never looked at*.
    ``n_negative_rows_with_judge_verdict`` and ``n_negative_rows_unjudged`` say
    how many rows rest on each. The labeling policy is unchanged; what changes is
    that a reader can now see how much of it is backed by a verdict without
    guessing at the judge's ``top_n``.

    A run over an ``examples_filtered.jsonl`` written before negatives carried
    verdicts reports every label-0 row as unjudged. That is the honest reading of
    such a file — it records no grading provenance at all — and the fix is to
    rerun ``make synth-judge``, whose cached verdicts make that nearly free.
    """

    examples_path: str
    manifest_path: str
    manifest_checksum: str
    crossencoder_run_check: dict[str, Any]
    gold_leakage_check_ran: bool
    n_rows: int
    n_queries_emitted: int
    n_queries_read: dict[str, int]
    n_rows_by_split: dict[str, int]
    n_rows_by_label: dict[str, int]
    n_candidates_scored: int
    n_unlabeled_candidates_dropped: int
    n_negative_rows_with_judge_verdict: int
    n_negative_rows_unjudged: int
    skipped_queries: dict[str, list[str]]

    def summary(self) -> str:
        return (
            f"{self.n_rows} candidate row(s) for {self.n_queries_emitted} query/queries "
            f"({self.n_rows_by_split}); {self.n_unlabeled_candidates_dropped} unlabeled "
            f"candidate(s) of {self.n_candidates_scored} scored were not emitted; "
            f"{self.n_negative_rows_with_judge_verdict} label-{LABEL_HARD_NEGATIVE} row(s) "
            f"rest on an LLM verdict and {self.n_negative_rows_unjudged} on mining alone; "
            f"{sum(len(v) for v in self.skipped_queries.values())} query/queries "
            f"contributed no rows ({ {k: len(v) for k, v in self.skipped_queries.items()} })"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "examples": self.examples_path,
            "split_manifest": self.manifest_path,
            "split_manifest_checksum": self.manifest_checksum,
            "crossencoder_run_check": self.crossencoder_run_check,
            "gold_leakage_check_ran": self.gold_leakage_check_ran,
            "n_rows": self.n_rows,
            "n_queries_emitted": self.n_queries_emitted,
            "n_queries_read": self.n_queries_read,
            "n_rows_by_split": self.n_rows_by_split,
            "n_rows_by_label": self.n_rows_by_label,
            "n_candidates_scored": self.n_candidates_scored,
            "n_unlabeled_candidates_dropped": self.n_unlabeled_candidates_dropped,
            "n_negative_rows_with_judge_verdict": self.n_negative_rows_with_judge_verdict,
            "n_negative_rows_unjudged": self.n_negative_rows_unjudged,
            "skipped_queries": self.skipped_queries,
            "labels": {
                "positive": LABEL_POSITIVE,
                "hard_negative": LABEL_HARD_NEGATIVE,
                # Said here rather than only in the docstring: a hard negative is
                # label 0 whether or not the judge graded it, and the two counts
                # above are how much of that label rests on a verdict.
                "hard_negative_evidence": (
                    "label 0 = a mined hard negative that survived the LLM judge; "
                    "see n_negative_rows_with_judge_verdict vs n_negative_rows_unjudged "
                    "for how many were actually graded (the judge grades only the "
                    "hardest few per query)"
                ),
                "unjudged_candidates": "not emitted (see rows_for_query)",
            },
        }


def run(
    cfg: Config,
    *,
    cascade: Cascade | None = None,
    manifest_path: str | Path | None = None,
    out_path: str | Path | None = None,
    artifacts_root: Path | None = None,
    allow_missing_gold: bool = False,
) -> tuple[Path, BuildReport]:
    """Build the candidate-level rows for both LambdaMART folds.

    Order is deliberate and matches ``concierge.ltr.train.run``: every leakage
    check runs **before** any model is loaded or any candidate is scored, so a
    contaminated run costs seconds instead of producing a file somebody later
    trains on. The gold check is the exception — it runs on the rows that were
    built, because that is the thing it has to be true of — but it runs before
    anything is written, so a leak leaves no artifact behind.

    ``cascade`` is the injection seam: ``None`` means load the real indexes,
    documents and fine-tuned checkpoint (:meth:`Cascade.from_config`).
    """
    from concierge.synth.negatives import read_examples_jsonl

    manifest, resolved_manifest = ltr_train.load_verified_manifest(cfg, manifest_path)

    # ── read the query-level folds, and hold them to the manifest ────────────
    folds: dict[str, list[TrainingExample]] = {}
    paths: dict[str, Path] = {}
    for split in (SPLIT_LTR_TRAIN, SPLIT_LTR_VAL):
        path = query_level_path(split, cfg, artifacts_root)
        paths[split] = path
        # Sorted by query id so the output order is a property of the data and
        # not of how the file happened to be written.
        folds[split] = sorted(read_examples_jsonl(path), key=lambda e: e.query_id)
        assert_queries_match_the_manifest([e.query_id for e in folds[split]], manifest, split, path)

    all_ids = [e.query_id for examples in folds.values() for e in examples]
    duplicates = sorted({qid for qid in all_ids if all_ids.count(qid) > 1})
    if duplicates:
        # A query appearing twice would have its rows emitted twice, which
        # silently doubles its weight in the lambdarank objective and puts two
        # rows for the same business inside one group. The cross-fold case
        # cannot arrive here (the manifest assigns one split per query, so
        # `assert_queries_match_the_manifest` rejects it); this is the
        # within-fold case, which only a corrupt or hand-edited file produces.
        raise BuildError(
            f"{len(duplicates)} query id(s) appear more than once in the LambdaMART "
            f"folds (e.g. {duplicates[:3]}). Regenerate them with `make synth-splits`; "
            "duplicated rows would weight those queries twice."
        )
    if not all_ids:
        raise BuildError(
            f"the LambdaMART folds at {paths[SPLIT_LTR_TRAIN]} and "
            f"{paths[SPLIT_LTR_VAL]} contain no queries. Nothing to build; "
            "regenerate the synthetic data with `make synth`."
        )
    crossencoder_evidence = assert_crossencoder_is_out_of_fold(cfg, manifest, all_ids)

    # ── say what will happen, then do it ─────────────────────────────────────
    target = _resolve(
        out_path if out_path is not None else default_output_path(cfg, artifacts_root)
    )
    print(_plan(cfg, folds, paths, resolved_manifest, target, crossencoder_evidence))

    if cascade is None:
        cascade = Cascade.from_config(cfg, artifacts_root=artifacts_root)

    rows: list[dict[str, Any]] = []
    outcomes: list[QueryOutcome] = []
    for split in (SPLIT_LTR_TRAIN, SPLIT_LTR_VAL):
        for example in folds[split]:
            parsed = parse_for_training(example.query)
            result = cascade.run(parsed)
            outcome = rows_for_query(
                query_id=example.query_id,
                split=split,
                parsed=parsed,
                positive_business_id=example.positive_business_id,
                # The whole records, not just their ids: each one carries whether
                # the judge graded it, which is what the report's label-0
                # evidence split is counted from.
                negatives=example.negatives,
                candidates=result.candidates,
                businesses=cascade.businesses,
            )
            outcomes.append(outcome)
            rows.extend(outcome.rows)

    if not rows:
        raise BuildError(
            "no labeled candidate rows were produced. Every query either lost its "
            "positive before Stage 4 or had no surviving judged negative — see "
            "rows_for_query. Check that `make index` and `make synth` ran for this "
            "config before building these rows."
        )

    # ── hold the output to the reader's own checks, before writing it ────────
    # Parsing the records back through `train.example_from_json` is not a
    # formality: it is what proves this writer and `train.load_examples` agree,
    # on every run rather than only in the test suite, and it is what lets the
    # trainer's own leakage functions (which are typed on `LtrExample`) be called
    # here rather than re-implemented.
    parsed_examples = [ltr_train.example_from_json(row) for row in rows]
    # enforce=True unconditionally: this stage *writes* the split field, so a
    # disagreement with the manifest is a bug in this module, never an
    # experiment. `cfg.ltr.enforce_oof_crossencoder_feature` exists to let the
    # trainer measure what the inflation is worth on an existing file; it has no
    # meaning for the producer.
    selected = {
        split: ltr_train.select_split_rows(parsed_examples, manifest, split, enforce=True)
        for split in (SPLIT_LTR_TRAIN, SPLIT_LTR_VAL)
    }
    ltr_train.assert_run_followed_the_manifest(parsed_examples, manifest)
    ltr_train.assert_no_gold_leakage(
        parsed_examples, cfg.evaluation.gold_queries, allow_missing_gold=allow_missing_gold
    )

    report = BuildReport(
        examples_path=str(target),
        manifest_path=str(resolved_manifest),
        manifest_checksum=manifest.checksum,
        crossencoder_run_check=crossencoder_evidence,
        gold_leakage_check_ran=_resolve(cfg.evaluation.gold_queries).exists(),
        n_rows=len(rows),
        n_queries_emitted=sum(1 for o in outcomes if o.rows),
        n_queries_read={split: len(folds[split]) for split in folds},
        n_rows_by_split={split: len(found) for split, found in selected.items()},
        n_rows_by_label={
            str(LABEL_POSITIVE): sum(1 for e in parsed_examples if e.label == LABEL_POSITIVE),
            str(LABEL_HARD_NEGATIVE): sum(
                1 for e in parsed_examples if e.label == LABEL_HARD_NEGATIVE
            ),
        },
        n_candidates_scored=sum(o.n_candidates for o in outcomes),
        n_unlabeled_candidates_dropped=sum(o.n_unlabeled for o in outcomes),
        n_negative_rows_with_judge_verdict=sum(o.n_negative_rows_judged for o in outcomes),
        n_negative_rows_unjudged=sum(o.n_negative_rows_unjudged for o in outcomes),
        skipped_queries={
            reason: [o.query_id for o in outcomes if o.skipped == reason]
            for reason in (SKIP_POSITIVE_ABSENT, SKIP_NO_NEGATIVE)
        },
    )

    write_rows(rows, target)
    (target.parent / REPORT_FILENAME).write_text(
        json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.info("wrote %d row(s) to %s", len(rows), target)
    return target, report


def _plan(
    cfg: Config,
    folds: Mapping[str, Sequence[TrainingExample]],
    paths: Mapping[str, Path],
    manifest_path: Path,
    target: Path,
    crossencoder_evidence: Mapping[str, Any],
) -> str:
    """What this run will do, printed before it starts.

    Every number here is a count of something already on disk. No throughput or
    duration is quoted: this stage runs two index searches and one cross-encoder
    pass per query, and how long that takes is a property of the machine, which
    this module has not measured.
    """
    lines = [
        "build LambdaMART feature rows:",
        f"  split manifest {manifest_path} (checksum {crossencoder_evidence.get('split_manifest_checksum', '')[:12]})",
        f"  cross-encoder checked against it: {crossencoder_evidence.get('run_metadata', 'n/a')}",
    ]
    for split in (SPLIT_LTR_TRAIN, SPLIT_LTR_VAL):
        lines.append(f"  {len(folds[split])} query/queries from {paths[split]}")
    lines += [
        f"  per query: retrieve {cfg.candidates.retrieve}, filter, rerank to "
        f"{cfg.candidates.rerank} with the fine-tuned cross-encoder",
        f"  emitting one row per labeled candidate -> {target}",
        "  0 API calls; local indexes and the local checkpoint only",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the candidate-level feature rows Stage 4 trains on, from the "
            "query-level LambdaMART folds and the fine-tuned cross-encoder."
        )
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Split manifest written by the synthetic-data stage. Defaults to "
            "artifacts/synth/<config fingerprint>/splits.json. A manifest given "
            "here is verified exactly as hard, including that it was written for "
            "this config's fingerprint."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to write the rows. Defaults to ltr_examples.jsonl beside the "
            "manifest, which is the path concierge.ltr.train reads."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="read and write under this root instead of the repo (tests, scratch runs)",
    )
    parser.add_argument(
        "--allow-missing-gold",
        action="store_true",
        help="build even though eval/gold_queries.csv does not exist yet (logged).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    target, report = run(
        cfg,
        manifest_path=args.splits,
        out_path=args.out,
        artifacts_root=None if args.artifacts_root is None else Path(args.artifacts_root),
        allow_missing_gold=args.allow_missing_gold,
    )
    print(report.summary())
    print(f"wrote {target}")
    print("Next: `make train-ltr`, which verifies this file against the same manifest.")
    return 0


def _resolve(path: str | Path) -> Path:
    """Relative paths are relative to the repo root, not the shell's cwd.

    Same rule as ``concierge.ltr.train._resolve``, for the same reason: a `make`
    target must behave identically whatever directory it was invoked from.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
