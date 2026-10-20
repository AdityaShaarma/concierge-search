"""Tests for the stage that turns query-level folds into Stage 4 feature rows.

Four of these are not ordinary unit tests but checks on claims the module makes
about itself, and they are why this file exists at all:

* ``test_rows_round_trip_through_the_real_loader`` and
  ``test_features_are_identical_offline_and_from_the_file`` — the writer's output
  is parsed by the *real* ``concierge.ltr.train.load_examples`` and the feature
  row rebuilt from the file is compared against the one built from the in-memory
  objects. A writer whose shape only *looks* right is the whole reason this file
  was asked for.
* ``test_refuses_*`` — each plants one specific way the cross-encoder score could
  be in fold (a stale per-split file, a regenerated manifest, a smoke-run
  checkpoint, no fine-tune at all) and asserts the stage refuses. A guard that
  cannot be made to fail is not evidence of anything.
* ``test_the_producer_and_search_agree_on_the_cascade`` — runs one
  ``SearchPipeline`` both ways, through ``search()`` and through
  :class:`concierge.ltr.build_examples.Cascade`, and asserts the per-stage scores
  and the order agree. The stage used to restate the cascade order and this test
  compared two implementations of it; both now call
  ``SearchPipeline.candidates``, so what is checked is that no second path has
  reappeared (DECISIONS.md D228).
* ``test_the_parse_is_the_crossencoders_own_training_parse`` — the stage must use
  ``concierge.rerank.model.training_parse`` itself, not a parse that happens to
  match it, so the text it scores and the day it filters on are the checkpoint's.
* ``test_the_report_splits_label0_rows_by_whether_the_judge_graded_them`` — a
  label-0 row rests on an LLM verdict or on mining alone, and the run has to say
  which, including for a filtered file written before negatives recorded it.

Everything runs against injected fakes: no dataset, no checkpoint, no API key.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from concierge.config import Config
from concierge.ltr import build_examples as build
from concierge.ltr import train as ltr_train
from concierge.ltr.features import FEATURE_NAMES, build_feature_row
from concierge.pipeline import SearchPipeline
from concierge.rerank.model import CrossEncoderReranker, PairScorer, training_parse
from concierge.synth import splits as synth_splits
from concierge.synth.negatives import HardNegative, TrainingExample, write_examples_jsonl
from concierge.types import (
    RANK_CROSS_ENCODER,
    RANK_RRF,
    SCORE_BM25,
    SCORE_CROSS_ENCODER,
    SCORE_DENSE,
    SCORE_LTR,
    SCORE_RRF,
    Business,
    ConstraintStatus,
    QueryType,
)

SPLIT_CE_TRAIN, SPLIT_CE_VAL = synth_splits.CROSSENCODER_SPLITS
SPLIT_LTR_TRAIN, SPLIT_LTR_VAL = synth_splits.LTR_SPLITS

#: A query the rule parser leaves alone: no attributes, no hours, no category, so
#: the constraint-only backfill does not fire and `evaluate_constraints` returns
#: nothing. Used wherever a test is about the row shape rather than constraints.
PLAIN_QUERY = "somewhere to read"
#: A query with two attribute constraints, for the constraint-serialization test.
CONSTRAINED_QUERY = "quiet cafe with wifi"


# ─────────────────────────────────────────────────────────────────────────────
# Fakes. Deliberately tiny and deterministic.
# ─────────────────────────────────────────────────────────────────────────────


class FakeRetriever:
    """Returns a fixed ranking, truncated to ``top_k``.

    Satisfies ``concierge.pipeline.Retriever``, which is the same shape
    ``BM25Index.search`` and ``DenseIndex.search`` return, so the cascade under
    test is the real one with only its inputs replaced.
    """

    def __init__(self, ranking: Sequence[tuple[str, float]]) -> None:
        self.ranking = list(ranking)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        return self.ranking[:top_k]


class IdScorer:
    """Scores a document by a table keyed on its first whitespace token.

    Every fixture document starts with its own business id, so a test can state
    the cross-encoder's ordering directly instead of reasoning about text.
    """

    def __init__(self, table: Mapping[str, float], max_length: int = 64) -> None:
        self._table = dict(table)
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        return self._max_length

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self._table[document.split()[0]] for _, document in pairs]

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def document_token_budget(self, query_text: str) -> int:
        return max(0, self._max_length - 3 - self.count_tokens(query_text))


def test_the_fake_scorer_satisfies_the_protocol() -> None:
    """If the fake drifts from `PairScorer` the cascade tests prove nothing."""
    assert isinstance(IdScorer({}), PairScorer)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus and cascade builders
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_IDS = ("b1", "b2", "b3", "b4", "b5", "b6")


def make_business(business_id: str, *, stars: float = 4.0, review_count: int = 50) -> Business:
    return Business(
        business_id=business_id,
        name=f"Place {business_id}",
        city="Metroville",
        state="XX",
        latitude=36.17,
        longitude=-115.14,
        stars=stars,
        review_count=review_count,
        categories=("Coffee & Tea",),
        attributes={},
        hours={},
        is_open=True,
    )


@pytest.fixture
def businesses() -> dict[str, Business]:
    return {
        bid: make_business(bid, stars=3.0 + index * 0.25, review_count=10 * (index + 1))
        for index, bid in enumerate(BUSINESS_IDS)
    }


@pytest.fixture
def documents() -> dict[str, str]:
    # First token is the business id: see IdScorer.
    return {bid: f"{bid} a calm room with books and coffee" for bid in BUSINESS_IDS}


def make_pipeline(
    cfg: Config,
    businesses: Mapping[str, Business],
    documents: Mapping[str, str],
    *,
    bm25: Sequence[tuple[str, float]] | None = None,
    dense: Sequence[tuple[str, float]] | None = None,
    cross_encoder: Mapping[str, float] | None = None,
) -> SearchPipeline:
    """A real `SearchPipeline` with fake retrievers and a fake pair scorer.

    The reranker is the *real* ``CrossEncoderReranker`` wrapping a fake scorer,
    not a fake reranker: the ordering, the stable tie-break and the score/rank
    bookkeeping are part of what this stage must reproduce, so they are exercised
    rather than stubbed. Stage 4 is left out entirely — `Cascade` must stop before
    it, and a pipeline with no LambdaMART cannot accidentally run one.
    """
    default_bm25 = [(bid, 10.0 - index) for index, bid in enumerate(BUSINESS_IDS)]
    default_dense = [(bid, 0.9 - 0.05 * index) for index, bid in enumerate(BUSINESS_IDS)]
    table = (
        {bid: float(len(BUSINESS_IDS) - index) for index, bid in enumerate(BUSINESS_IDS)}
        if cross_encoder is None
        else dict(cross_encoder)
    )
    return SearchPipeline(
        cfg,
        businesses=dict(businesses),
        documents=dict(documents),
        bm25_index=FakeRetriever(default_bm25 if bm25 is None else bm25),
        dense_index=FakeRetriever(default_dense if dense is None else dense),
        reranker=CrossEncoderReranker(IdScorer(table)),
        ltr_ranker=None,
    )


def make_cascade(
    cfg: Config,
    businesses: Mapping[str, Business],
    documents: Mapping[str, str],
    *,
    bm25: Sequence[tuple[str, float]] | None = None,
    dense: Sequence[tuple[str, float]] | None = None,
    cross_encoder: Mapping[str, float] | None = None,
) -> build.Cascade:
    """The producer's cascade, wrapping the pipeline `make_pipeline` builds."""
    return build.Cascade(
        make_pipeline(
            cfg, businesses, documents, bm25=bm25, dense=dense, cross_encoder=cross_encoder
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# On-disk inputs: the split manifest, the per-split folds, the run metadata
# ─────────────────────────────────────────────────────────────────────────────


def write_manifest(
    path: Path,
    cfg: Config,
    by_split: Mapping[str, Sequence[str]],
    *,
    config_fingerprint: str | None = None,
) -> Path:
    """Write a manifest the real reader accepts.

    Built through ``concierge.synth.splits`` rather than as hand-written JSON so
    the checksum, the seed and the fractions agree with the config — a manifest
    that failed verification for an unrelated reason would make every test below
    pass for the wrong reason.
    """
    assignment = {qid: split for split, ids in by_split.items() for qid in ids}
    fractions = synth_splits.fractions_from_config(cfg)
    group_by = "query"
    manifest = synth_splits.SplitManifest(
        seed=cfg.seed,
        group_by=group_by,
        fractions=fractions,
        assignment=assignment,
        checksum=synth_splits.compute_checksum(assignment, cfg.seed, fractions, group_by),
        n_groups=len(assignment),
        config_fingerprint=(
            # The synth key, matching what `synth.splits.assign_splits` records
            # and what both trainers now compare against (artifacts are keyed on
            # the sections they depend on, not the whole config — see
            # `Config.fingerprint_for` / DECISIONS.md D237).
            cfg.fingerprint_for(*synth_splits.SYNTH_SECTIONS)
            if config_fingerprint is None
            else config_fingerprint
        ),
    )
    return synth_splits.save_manifest(manifest, path)


def make_negatives(
    *business_ids: str,
    verdicts: Mapping[str, int] | None = None,
) -> tuple[HardNegative, ...]:
    """Mined hard negatives, carrying a judge grade only where one is given.

    No verdict is the default because that is what mining produces, and what an
    ``examples_filtered.jsonl`` written before negatives carried verdicts says
    about every negative in it.
    """
    graded = {} if verdicts is None else dict(verdicts)
    return tuple(
        HardNegative(
            business_id=bid,
            retrieval_rank=index + 2,
            retrieval_score=0.01,
            verdict=graded.get(bid),
        )
        for index, bid in enumerate(business_ids)
    )


def make_example(
    query_id: str,
    query: str = PLAIN_QUERY,
    *,
    positive: str = "b1",
    negatives: Sequence[str] = ("b2", "b3"),
    verdicts: Mapping[str, int] | None = None,
) -> TrainingExample:
    return TrainingExample(
        query_id=query_id,
        query=query,
        query_type=QueryType.VIBE,
        positive_business_id=positive,
        negatives=make_negatives(*negatives, verdicts=verdicts),
        positive_retrieval_rank=1,
    )


def write_fold(split: str, examples: Sequence[TrainingExample], cfg: Config, root: Path) -> Path:
    """Write one query-level fold with the writer the synth stage uses."""
    return write_examples_jsonl(list(examples), synth_splits.split_example_path(split, cfg, root))


def write_rerank_run(directory: Path, payload: Mapping[str, object]) -> Path:
    from concierge.rerank.model import RUN_METADATA_FILENAME

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_METADATA_FILENAME
    path.write_text(json.dumps(dict(payload)))
    return path


@pytest.fixture
def gold_file(tmp_path: Path) -> Path:
    """A gold set that shares no query with the folds.

    Written rather than reusing ``eval/gold_queries.csv`` so the leakage check
    genuinely runs (it fails closed on a missing file) and so the test does not
    change meaning when the owner adds a gold query.
    """
    path = tmp_path / "gold_queries.csv"
    path.write_text("query_id,query,query_type\ng001,best deep dish pizza,specific\n")
    return path


@pytest.fixture
def cfg(config: Config, gold_file: Path) -> Config:
    """The fixture config, pointed at a tmp gold set.

    One config object per test, built once: the artifact paths are keyed by its
    fingerprint, so re-deriving it would move the directories mid-test.
    """
    return config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"gold_queries": gold_file})}
    )


@dataclass(frozen=True)
class World:
    """The on-disk inputs one test runs against.

    A typed object rather than a dict because every test touches it: as a dict
    of ``object`` each access needed a cast to keep mypy honest, and a wall of
    casts is exactly the noise that stops a reader checking what a test does.
    """

    root: Path
    manifest_path: Path
    manifest: synth_splits.SplitManifest
    rerank_root: Path


@pytest.fixture
def world(cfg: Config, tmp_path: Path, monkeypatch) -> World:
    """A complete, valid set of inputs: manifest, folds, fine-tuned run metadata.

    Each test below breaks exactly one thing in this, which is the only way a
    refusal test proves the refusal and not a setup error.
    """
    from concierge.rerank import model as rerank_model

    rerank_root = tmp_path / "rerank"
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: rerank_root)

    manifest_path = write_manifest(
        tmp_path / "splits.json",
        cfg,
        {
            SPLIT_CE_TRAIN: ["s001", "s002"],
            SPLIT_CE_VAL: ["s003"],
            SPLIT_LTR_TRAIN: ["s100", "s101"],
            SPLIT_LTR_VAL: ["s200"],
        },
    )
    write_fold(SPLIT_LTR_TRAIN, [make_example("s100"), make_example("s101")], cfg, tmp_path)
    write_fold(SPLIT_LTR_VAL, [make_example("s200")], cfg, tmp_path)

    manifest = synth_splits.load_manifest(manifest_path)
    # Both provenance fields, so the happy path is the fully-tied case
    # (``config_fingerprint_tie == "matched"``) rather than the weaker one the
    # trainer only warns about.
    write_rerank_run(
        rerank_root,
        {
            "split_manifest_checksum": manifest.checksum,
            "split_manifest_config_fingerprint": manifest.config_fingerprint,
        },
    )
    return World(
        root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        rerank_root=rerank_root,
    )


def do_run(
    cfg: Config,
    world: World,
    cascade: build.Cascade,
) -> tuple[Path, build.BuildReport]:
    return build.run(
        cfg,
        cascade=cascade,
        manifest_path=world.manifest_path,
        artifacts_root=world.root,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The round trip — the assertion this file exists for
# ─────────────────────────────────────────────────────────────────────────────


def test_rows_round_trip_through_the_real_loader(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The written file must parse with `train.load_examples`, not a copy of it."""
    cascade = make_cascade(cfg, businesses, documents)

    path, report = do_run(cfg, world, cascade)

    assert path == ltr_train.default_examples_path(cfg, world.root)
    examples = ltr_train.load_examples(path)
    assert len(examples) == report.n_rows
    assert {e.query_id for e in examples} == {"s100", "s101", "s200"}
    assert {e.split for e in examples} == {SPLIT_LTR_TRAIN, SPLIT_LTR_VAL}
    for example in examples:
        assert example.query_text == PLAIN_QUERY
        assert example.parsed.semantic_text == PLAIN_QUERY
        # Every row carries the four stage signals Stage 4 reads by name.
        assert SCORE_BM25 in example.candidate.scores
        assert SCORE_DENSE in example.candidate.scores
        assert SCORE_RRF in example.candidate.scores
        assert SCORE_CROSS_ENCODER in example.candidate.scores
        assert RANK_RRF in example.candidate.ranks
        assert RANK_CROSS_ENCODER in example.candidate.ranks
        assert example.business.business_id == example.candidate.business_id


def test_features_are_identical_offline_and_from_the_file(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The anti-skew claim, checked rather than argued.

    Builds the cascade's candidates directly, computes `build_feature_row` over
    the in-memory objects, then compares against the row rebuilt from the JSONL
    the stage wrote. If serialization dropped or renamed a score key, a feature
    silently becomes NaN — which produces a trained model, not an error.
    """
    cascade = make_cascade(cfg, businesses, documents)
    path, _ = do_run(cfg, world, cascade)

    parsed = build.parse_for_training(PLAIN_QUERY)
    in_memory = {
        candidate.business_id: build_feature_row(
            parsed, candidate, businesses[candidate.business_id]
        )
        for candidate in cascade.run(parsed).candidates
    }

    from_file = {
        e.candidate.business_id: build_feature_row(e.parsed, e.candidate, e.business)
        for e in ltr_train.load_examples(path)
        if e.query_id == "s100"
    }

    assert from_file, "the query produced no rows, so nothing was compared"
    for business_id, row in from_file.items():
        expected = in_memory[business_id]
        for name in FEATURE_NAMES:
            got, want = row[name], expected[name]
            if math.isnan(want):
                assert math.isnan(got), f"{name} for {business_id}: {got} is not NaN"
            else:
                assert got == pytest.approx(want), f"{name} for {business_id}"


def test_the_rows_train_a_ranker_without_further_massaging(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """`build_ranking_data` requires contiguous query blocks; assert it accepts ours."""
    path, report = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    examples = ltr_train.load_examples(path)
    data = ltr_train.build_ranking_data(examples)

    assert data.features.n_rows == report.n_rows
    assert data.n_queries == report.n_queries_emitted
    # Contiguity is what `group_sizes` refuses to guess at: the writer must have
    # emitted each query's rows as one block, and this proves it did.
    assert int(data.group.sum()) == report.n_rows
    assert ltr_train.group_sizes(list(data.query_ids)).tolist() == data.group.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Labeling
# ─────────────────────────────────────────────────────────────────────────────


def test_the_positive_is_graded_high_and_judged_negatives_zero(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    path, _ = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    labels = {
        e.candidate.business_id: e.label
        for e in ltr_train.load_examples(path)
        if e.query_id == "s100"
    }

    assert labels == {
        "b1": build.LABEL_POSITIVE,
        "b2": build.LABEL_HARD_NEGATIVE,
        "b3": build.LABEL_HARD_NEGATIVE,
    }


def test_unlabeled_candidates_are_not_emitted_but_are_counted(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The labeling decision, pinned.

    b4-b6 are retrieved and reranked but neither the positive nor judged
    negatives, so they carry no label evidence and are left out. The count is
    reported, because the size of that gap is the cost of the choice.
    """
    path, report = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    emitted = {e.candidate.business_id for e in ltr_train.load_examples(path)}

    assert emitted == {"b1", "b2", "b3"}
    # Three queries, six candidates each, three of them labeled.
    assert report.n_candidates_scored == 18
    assert report.n_unlabeled_candidates_dropped == 9
    assert report.n_rows_by_label == {
        str(build.LABEL_POSITIVE): 3,
        str(build.LABEL_HARD_NEGATIVE): 6,
    }


def test_a_query_whose_positive_did_not_survive_contributes_no_rows(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    """Emitting a scoreless positive would teach "missing scores mean relevant".

    s100's positive is a business the retriever never surfaces, so that query
    drops out while the other two are unaffected — which is also the shape of the
    event ``TrainingExample.positive_retrieval_rank is None`` already records
    upstream.
    """
    write_fold(
        SPLIT_LTR_TRAIN,
        [make_example("s100", positive="b_absent"), make_example("s101")],
        cfg,
        tmp_path,
    )

    path, report = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    assert report.skipped_queries[build.SKIP_POSITIVE_ABSENT] == ["s100"]
    assert {e.query_id for e in ltr_train.load_examples(path)} == {"s101", "s200"}
    assert report.n_queries_read == {SPLIT_LTR_TRAIN: 2, SPLIT_LTR_VAL: 1}


def test_a_run_in_which_no_query_keeps_its_positive_writes_nothing(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """An empty training file is not a successful run; it is a missing prerequisite."""
    without_b1 = [(bid, 9.0) for bid in BUSINESS_IDS if bid != "b1"]
    cascade = make_cascade(cfg, businesses, documents, bm25=without_b1, dense=without_b1)

    with pytest.raises(build.BuildError, match="no labeled candidate rows"):
        do_run(cfg, world, cascade)

    assert not ltr_train.default_examples_path(cfg, world.root).exists()


def test_a_query_with_no_surviving_negative_contributes_no_rows(
    cfg: Config,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """A group with one label value yields no pairs, and scores a perfect NDCG."""
    only_positive = [("b1", 9.0), ("b4", 8.0)]
    cascade = make_cascade(cfg, businesses, documents, bm25=only_positive, dense=only_positive)

    outcome = build.rows_for_query(
        query_id="s100",
        split=SPLIT_LTR_TRAIN,
        parsed=build.parse_for_training(PLAIN_QUERY),
        positive_business_id="b1",
        negatives=make_negatives("b2", "b3"),
        candidates=cascade.run(build.parse_for_training(PLAIN_QUERY)).candidates,
        businesses=businesses,
    )

    assert outcome.rows == ()
    assert outcome.skipped == build.SKIP_NO_NEGATIVE


def test_a_positive_listed_as_its_own_negative_is_refused(
    cfg: Config,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """A contradictory upstream row must not have its label chosen silently."""
    cascade = make_cascade(cfg, businesses, documents)
    parsed = build.parse_for_training(PLAIN_QUERY)

    with pytest.raises(build.BuildError, match="among its"):
        build.rows_for_query(
            query_id="s100",
            split=SPLIT_LTR_TRAIN,
            parsed=parsed,
            positive_business_id="b1",
            negatives=make_negatives("b1", "b2"),
            candidates=cascade.run(parsed).candidates,
            businesses=businesses,
        )


# ─────────────────────────────────────────────────────────────────────────────
# What the label-0 rows rest on (DECISIONS.md D227/D228)
# ─────────────────────────────────────────────────────────────────────────────


def _negative_rows(path: Path) -> list[tuple[str, str]]:
    """``(query_id, business_id)`` of every label-0 row in the written file."""
    return [
        (e.query_id, e.candidate.business_id)
        for e in ltr_train.load_examples(path)
        if e.label == build.LABEL_HARD_NEGATIVE
    ]


def test_the_report_splits_label0_rows_by_whether_the_judge_graded_them(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    """A label-0 row rests either on an LLM verdict or on nothing but mining.

    The judge grades only each query's hardest few negatives, so the filtered
    file is a mixture and the rows built from it are too. The labeling policy is
    D227's and is not in question here; what is asserted is that the run says how
    much of it is backed by a verdict, which it could not before negatives carried
    one (D228).
    """
    graded = {("s100", "b2"), ("s101", "b2")}
    write_fold(
        SPLIT_LTR_TRAIN,
        [
            make_example("s100", verdicts={"b2": 0}),
            # Grade 1 ("partially relevant") is kept as a negative by the judge's
            # threshold, so it is a verdict too — a weaker one, which is exactly
            # why the report counts verdicts rather than asserting they are 0s.
            make_example("s101", verdicts={"b2": 1}),
        ],
        cfg,
        tmp_path,
    )
    write_fold(SPLIT_LTR_VAL, [make_example("s200")], cfg, tmp_path)

    path, report = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    rows = _negative_rows(path)
    judged = [row for row in rows if row in graded]
    assert judged, "the fixture produced no graded negative row, so nothing is proven"
    assert len(judged) < len(rows), "and no ungraded one either"
    assert report.n_negative_rows_with_judge_verdict == len(judged)
    assert report.n_negative_rows_unjudged == len(rows) - len(judged)
    written = json.loads((path.parent / build.REPORT_FILENAME).read_text())
    assert written["n_negative_rows_with_judge_verdict"] == len(judged)
    assert written["n_negative_rows_unjudged"] == len(rows) - len(judged)


def test_negatives_from_a_file_without_verdicts_are_reported_as_unjudged(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """An ``examples_filtered.jsonl`` written before the field must not over-claim.

    `world`'s folds carry no verdicts, which is what such a file parses as. The
    run must succeed — these files are usable, the labels are unchanged — and must
    attribute every label-0 row to mining alone rather than to a grading call
    that left no record.
    """
    path, report = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    rows = _negative_rows(path)
    assert rows, "no label-0 row was written, so nothing is proven"
    assert report.n_negative_rows_with_judge_verdict == 0
    assert report.n_negative_rows_unjudged == len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Out of fold — the reason this stage is separate
# ─────────────────────────────────────────────────────────────────────────────


def test_refuses_when_the_crossencoder_was_never_fine_tuned(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The trainer tolerates a missing fine-tune; the producer cannot.

    For a reader, "no run metadata" means the score came from the pretrained
    model, which saw no synthetic query. For the writer it means there is no
    fine-tuned checkpoint to attach a score from at all.
    """
    from concierge.rerank.model import RUN_METADATA_FILENAME

    (world.rerank_root / RUN_METADATA_FILENAME).unlink()

    with pytest.raises(build.BuildError, match="make train-reranker"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_refuses_a_checkpoint_fine_tuned_under_another_manifest(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The manifest at a path can be regenerated; the checksum is what pins it."""
    write_rerank_run(world.rerank_root, {"split_manifest_checksum": "0" * 64})

    with pytest.raises(ltr_train.LeakageError, match="fine-tuned under split manifest"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_refuses_a_smoke_run_checkpoint(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """`--limit-examples` trains on a truncated subset: nothing from it is reportable."""
    write_rerank_run(
        world.rerank_root,
        {
            "split_manifest_checksum": world.manifest.checksum,
            "is_smoke_run": True,
        },
    )

    with pytest.raises(ltr_train.LeakageError, match="smoke run"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_refuses_a_checkpoint_fine_tuned_under_another_configuration(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """Same split, other settings: the partition holds, the provenance does not.

    The checksum covers the assignment, the seed, the fractions and the grouping
    — not the config fingerprint — so two manifests can be the same split while
    having been written under different configurations. A checkpoint from the
    other one was trained with another base model or another ``max_length``, and
    its scores are not the feature this run would claim to have recorded.
    """
    write_rerank_run(
        world.rerank_root,
        {
            "split_manifest_checksum": world.manifest.checksum,
            "split_manifest_config_fingerprint": "deadbeef00",
        },
    )

    with pytest.raises(ltr_train.LeakageError, match="config fingerprint"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_refuses_a_fold_holding_a_crossencoder_query(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    """The realistic leak: a stale ``ltr_train.jsonl`` from an earlier manifest.

    The filename still says ltr_train and the file still parses. Only the
    manifest knows that s001 trained the cross-encoder.
    """
    write_fold(SPLIT_LTR_TRAIN, [make_example("s001")], cfg, tmp_path)

    with pytest.raises(ltr_train.LeakageError, match="assigned to another split"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_refuses_a_fold_query_the_manifest_never_assigned(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    write_fold(SPLIT_LTR_VAL, [make_example("s999")], cfg, tmp_path)

    with pytest.raises(synth_splits.SplitLeakageError, match="not in the split manifest"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_refuses_a_manifest_from_another_configuration(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    """Delegated to the trainer's loader, and asserted so the delegation is real."""
    write_manifest(
        world.manifest_path,
        cfg,
        {SPLIT_LTR_TRAIN: ["s100", "s101"], SPLIT_LTR_VAL: ["s200"]},
        config_fingerprint="deadbeef00",
    )

    with pytest.raises(ltr_train.LeakageError, match="config fingerprint"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_the_evidence_is_recorded_in_the_report(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """ "Checked and clean" and "not checked" must never look the same."""
    path, report = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    assert report.crossencoder_run_check["checked"] is True
    assert report.crossencoder_run_check["split_manifest_checksum"] == world.manifest.checksum
    assert report.manifest_checksum == world.manifest.checksum
    assert report.gold_leakage_check_ran is True

    written = json.loads((path.parent / build.REPORT_FILENAME).read_text())
    assert written["crossencoder_run_check"]["checked"] is True
    assert written["n_rows"] == report.n_rows


# ─────────────────────────────────────────────────────────────────────────────
# Gold leakage
# ─────────────────────────────────────────────────────────────────────────────


def test_a_gold_query_in_a_fold_is_refused_before_anything_is_written(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    gold_file: Path,
) -> None:
    """The gold set is test-only; a leak would corrupt the headline metric."""
    gold_file.write_text(f"query_id,query,query_type\ng001,{PLAIN_QUERY},vibe\n")

    with pytest.raises(ltr_train.LeakageError, match="gold queries found"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))

    assert not ltr_train.default_examples_path(cfg, world.root).exists()


def test_a_missing_gold_file_refuses_unless_the_override_is_given(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    gold_file: Path,
) -> None:
    """A check that passes because it had nothing to compare against is worse than none."""
    gold_file.unlink()
    cascade = make_cascade(cfg, businesses, documents)

    with pytest.raises(ltr_train.LeakageError, match="gold query file not found"):
        do_run(cfg, world, cascade)

    path, _ = build.run(
        cfg,
        cascade=cascade,
        manifest_path=world.manifest_path,
        artifacts_root=world.root,
        allow_missing_gold=True,
    )
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_two_runs_produce_a_byte_identical_file(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """No seed is involved: determinism is ordering plus deterministic tie-breaks.

    Includes the trap this module avoids by omission — `ParsedQuery.parse_latency_ms`
    is a wall-clock measurement, and serializing it would make this test flap.
    """
    path, _ = do_run(cfg, world, make_cascade(cfg, businesses, documents))
    first = path.read_bytes()

    path_again, _ = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    assert path_again == path
    assert path_again.read_bytes() == first


def test_rows_are_ordered_by_split_then_query_then_cascade_rank(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """Row order is the served ranking, and query blocks are contiguous."""
    # b3 is made the cross-encoder's favourite, so the emitted order is not the
    # retrieval order and the assertion is about the reranker, not the input.
    table = {"b1": 5.0, "b2": 4.0, "b3": 9.0, "b4": 1.0, "b5": 0.5, "b6": 0.25}
    path, _ = do_run(cfg, world, make_cascade(cfg, businesses, documents, cross_encoder=table))

    examples = ltr_train.load_examples(path)

    assert [e.query_id for e in examples] == ["s100"] * 3 + ["s101"] * 3 + ["s200"] * 3
    assert [e.candidate.business_id for e in examples[:3]] == ["b3", "b1", "b2"]


# ─────────────────────────────────────────────────────────────────────────────
# Parity with the serving path
# ─────────────────────────────────────────────────────────────────────────────


def test_the_producer_and_search_agree_on_the_cascade(
    cfg: Config,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The producer and `search()` must run the same stages in the same order.

    They do by construction — both go through
    `SearchPipeline.candidates` — and this asserts the construction rather than
    the arrangement it replaced. This stage used to restate the order of
    retrieve/filter/rerank and the old version of this test compared the two
    implementations candidate by candidate; it could catch a disagreement but
    could not prevent one (DECISIONS.md D228). Now one pipeline is run both ways:
    through `search()` with Stage 4 off, and through `Cascade`. A future change
    that gave the producer a path of its own would fail here.
    """
    pipeline = make_pipeline(cfg, businesses, documents)
    cascade = build.Cascade(pipeline)

    response = pipeline.search(PLAIN_QUERY, variant=cascade.variant, k=len(BUSINESS_IDS))
    result = cascade.run(build.parse_for_training(PLAIN_QUERY))
    mine = {c.business_id: c for c in result.candidates}

    assert response.results, "the pipeline returned nothing, so nothing was compared"
    assert [r.business.business_id for r in response.results] == list(mine)
    for served in response.results:
        assert served.stage_scores == pytest.approx(mine[served.business.business_id].scores)
    # The counts are the pipeline's own, so they describe the same run.
    assert result.n_after_filter <= result.n_retrieved
    assert len(result.candidates) <= result.n_after_filter


def test_the_cascade_does_not_run_stage_four(
    cfg: Config,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """The rows this stage writes are Stage 4's training input.

    `CASCADE_VARIANT` has `Stage.LTR` off, and the pipeline this stage builds has
    no LambdaMART at all. If either changed, every row would carry a feature
    derived from the model the rows are used to fit.
    """
    from concierge.pipeline import Stage

    assert not build.CASCADE_VARIANT.enabled(Stage.LTR)
    cascade = make_cascade(cfg, businesses, documents)

    assert cascade.pipeline.ltr_ranker is None
    for candidate in cascade.run(build.parse_for_training(PLAIN_QUERY)).candidates:
        assert SCORE_LTR not in candidate.scores


def test_the_parse_is_the_crossencoders_own_training_parse(
    cfg: Config,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """`parse_for_training` must *be* `rerank.model.training_parse`, not match it.

    The scored text has to be the string the checkpoint was fine-tuned on, and
    the weekday has to be the one that text was derived with — otherwise this
    stage filters on a different day than it scores for. That used to be a
    private constant copied into this module and pinned by comparing the two
    texts (DECISIONS.md D228); delegating leaves nothing to drift, so the
    assertion is the delegation itself, plus the property it buys.
    """
    from concierge.rerank.model import query_text_for, training_query_text

    for query in (PLAIN_QUERY, CONSTRAINED_QUERY, "cheap tacos open past 10pm on a friday"):
        # `parse_latency_ms` is dropped from the comparison: it is a wall-clock
        # measurement taken per call, not part of the parse — which is why
        # `record_for_candidate` does not serialize it either.
        assert replace(build.parse_for_training(query), parse_latency_ms=0.0) == replace(
            training_parse(query), parse_latency_ms=0.0
        )
        assert query_text_for(build.parse_for_training(query)) == training_query_text(query)


# ─────────────────────────────────────────────────────────────────────────────
# Constraints, and the shape of what is written
# ─────────────────────────────────────────────────────────────────────────────


def test_constraints_are_written_as_the_checker_evaluated_them(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    """Constraint outcomes travel on the candidate; they are never re-derived.

    The fixture businesses carry no attributes at all, so a query asking for a
    quiet room with wifi must come back UNKNOWN on both — not VIOLATED, which is
    the distinction three of Stage 4's features are built on.
    """
    write_fold(
        SPLIT_LTR_TRAIN,
        [make_example("s100", CONSTRAINED_QUERY), make_example("s101", CONSTRAINED_QUERY)],
        cfg,
        tmp_path,
    )
    write_fold(SPLIT_LTR_VAL, [make_example("s200", CONSTRAINED_QUERY)], cfg, tmp_path)

    path, _ = do_run(cfg, world, make_cascade(cfg, businesses, documents))

    rows = [e for e in ltr_train.load_examples(path) if e.query_id == "s100"]
    assert rows, "the constrained query produced no rows"
    statuses = {match.name: match.status for match in rows[0].candidate.constraints}
    assert statuses["NoiseLevel"] is ConstraintStatus.UNKNOWN
    assert statuses["WiFi"] is ConstraintStatus.UNKNOWN
    assert rows[0].parsed.attributes == {"NoiseLevel": "quiet", "WiFi": "free"}


def test_the_record_omits_what_the_reader_defaults_and_what_would_flap(
    cfg: Config,
    businesses: dict[str, Business],
    documents: dict[str, str],
) -> None:
    """Hours, attributes and the parse latency are deliberately not written.

    `train._business_from_json` documents that it defaults hours and attributes
    to empty because the constraint outcomes travel on the candidate; and
    `parse_latency_ms` is a wall-clock measurement no feature reads, so writing it
    would cost byte-reproducibility for nothing.
    """
    parsed = build.parse_for_training(PLAIN_QUERY)
    cascade = make_cascade(cfg, businesses, documents)
    candidate = cascade.run(parsed).candidates[0]

    record = build.record_for_candidate(
        query_id="s100",
        split=SPLIT_LTR_TRAIN,
        label=build.LABEL_POSITIVE,
        parsed=parsed,
        candidate=candidate,
        business=businesses[candidate.business_id],
    )

    assert "hours" not in record["business"]
    assert "attributes" not in record["business"]
    assert "parse_latency_ms" not in record["query"]
    # And it still round-trips: the reader supplies the defaults.
    example = ltr_train.example_from_json(record)
    assert example.business.hours == {}
    assert example.parsed.parse_latency_ms == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Failing closed on missing inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_a_missing_fold_names_the_make_target(cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(build.BuildError, match="make synth"):
        build.query_level_path(SPLIT_LTR_TRAIN, cfg, tmp_path)


def test_empty_folds_are_an_error_not_an_empty_file(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    write_fold(SPLIT_LTR_TRAIN, [], cfg, tmp_path)
    write_fold(SPLIT_LTR_VAL, [], cfg, tmp_path)

    with pytest.raises(build.BuildError, match="no queries"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_a_duplicated_query_is_refused(
    cfg: Config,
    world: World,
    businesses: dict[str, Business],
    documents: dict[str, str],
    tmp_path: Path,
) -> None:
    """A query emitted twice would weight itself twice in the objective.

    `write_examples_jsonl` would not normally produce this; a hand-edited or
    half-regenerated fold would, and the result trains without complaint.
    """
    write_fold(
        SPLIT_LTR_TRAIN,
        [make_example("s100"), make_example("s100"), make_example("s101")],
        cfg,
        tmp_path,
    )

    with pytest.raises(build.BuildError, match="more than once"):
        do_run(cfg, world, make_cascade(cfg, businesses, documents))


def test_the_default_output_path_is_the_one_the_trainer_reads(cfg: Config, tmp_path: Path) -> None:
    """One path, known in one place: the reader's.

    A producer that knew the path independently is how a stage ends up writing a
    file nothing reads, which is the gap this module closes.
    """
    assert build.default_output_path(cfg, tmp_path) == ltr_train.default_examples_path(
        cfg, tmp_path
    )
    assert build.default_output_path(cfg, tmp_path).name == ltr_train.EXAMPLES_FILENAME
    assert (
        build.default_output_path(cfg, tmp_path).parent
        == synth_splits.split_example_path(SPLIT_LTR_TRAIN, cfg, tmp_path).parent
    )


@pytest.mark.needs_data
def test_cascade_from_config_loads_the_real_artifacts() -> None:
    """The one test that needs a built dataset, built indexes and a fine-tune.

    Skipped unless all three exist, following the convention in
    ``tests/test_load.py``. It exists to catch what only the real artifacts can
    show: that `Cascade.from_config` reads the directories the other stages
    actually write.
    """
    from concierge.config import load_config
    from concierge.data.build import resolve_processed_dir
    from concierge.rerank.model import CHECKPOINT_DIRNAME, rerank_dir
    from concierge.retrieval.bm25 import index_dir
    from concierge.retrieval.dense import dense_index_dir

    real = load_config("configs/base.yaml")
    if not (resolve_processed_dir(real) / "businesses.parquet").exists():
        pytest.skip("no processed dataset for this config (run `make data`)")
    if not index_dir(real).exists() or not dense_index_dir(real).exists():
        pytest.skip("no retrieval indexes for this config (run `make index`)")
    if not (rerank_dir(real) / CHECKPOINT_DIRNAME / "config.json").exists():
        pytest.skip("no fine-tuned cross-encoder for this config (run `make train-reranker`)")

    cascade = build.Cascade.from_config(real)

    assert cascade.businesses and cascade.documents
