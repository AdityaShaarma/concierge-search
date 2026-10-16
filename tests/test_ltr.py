"""Tests for Stage 4: the feature builder and the LambdaMART training loop.

Three things here are not ordinary unit tests but checks on claims the project
makes about itself, and they are the reason this file is long:

* ``test_out_of_fold_*`` — plants an overlapping query in the split manifest and
  asserts training refuses. Without this, the cross-encoder feature could be
  computed on queries the cross-encoder was fine-tuned on, which inflates it
  and makes LambdaMART over-trust it.
* ``test_group_sizes_*`` — LightGBM takes group *sizes*, never query ids, so a
  wrong group array trains a model on cross-query pairs and reports a plausible
  NDCG for nonsense.
* ``test_recovers_monotone_label`` — trains on a problem whose right answer is
  known by construction (label is a monotone function of one feature) and
  asserts the model finds it. That is how the training loop gets verified
  end to end without any real data.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from concierge.config import Config
from concierge.ltr import features as feat
from concierge.ltr import train as ltr_train
from concierge.ltr.features import (
    FEATURE_NAMES,
    NON_TEXT_FEATURE_NAMES,
    TEXT_FEATURE_NAMES,
    FeatureMatrix,
    ablated_feature_names,
    build_feature_row,
    feature_matrix,
    feature_vector,
)
from concierge.ltr.train import (
    SPLIT_CE_TRAIN,
    SPLIT_CE_VAL,
    SPLIT_LTR_TRAIN,
    SPLIT_LTR_VAL,
    LeakageError,
    LtrExample,
    assert_no_gold_leakage,
    assert_out_of_fold,
    build_ranking_data,
    example_from_json,
    group_sizes,
    load_split_manifest,
    sort_by_query,
    train_lambdamart,
)
from concierge.retrieval.filters import CONSTRAINT_CATEGORY, CONSTRAINT_GEO, haversine_km
from concierge.types import (
    RANK_RRF,
    SCORE_BM25,
    SCORE_CROSS_ENCODER,
    SCORE_DENSE,
    Business,
    Candidate,
    ConstraintMatch,
    ConstraintStatus,
    GeoConstraint,
    ParsedQuery,
)

# ─────────────────────────────────────────────────────────────────────────────
# Builders. Deliberately explicit rather than factory-fancy: a test that fails
# should be readable without chasing helpers.
# ─────────────────────────────────────────────────────────────────────────────


def make_business(
    business_id: str = "b001",
    stars: float = 4.0,
    review_count: int = 100,
    categories: tuple[str, ...] = ("Coffee & Tea",),
    latitude: float | None = 36.17,
    longitude: float | None = -115.14,
) -> Business:
    return Business(
        business_id=business_id,
        name="Test Cafe",
        city="Metroville",
        state="XX",
        latitude=latitude,
        longitude=longitude,
        stars=stars,
        review_count=review_count,
        categories=categories,
        attributes={},
        hours={},
        is_open=True,
    )


def make_query(
    categories: tuple[str, ...] = (),
    geo: GeoConstraint | None = None,
    attributes: dict | None = None,
) -> ParsedQuery:
    return ParsedQuery(
        raw="quiet cafe with wifi",
        semantic_text="quiet cafe with wifi",
        attributes=attributes or {},
        categories=categories,
        geo=geo,
    )


def make_candidate(
    business_id: str = "b001",
    scores: dict | None = None,
    ranks: dict | None = None,
    constraints: tuple[ConstraintMatch, ...] = (),
) -> Candidate:
    return Candidate(
        business_id=business_id,
        scores={SCORE_BM25: 8.0, SCORE_DENSE: 0.7, SCORE_CROSS_ENCODER: 3.0}
        if scores is None
        else scores,
        ranks={RANK_RRF: 5} if ranks is None else ranks,
        constraints=constraints,
    )


def satisfied(name: str) -> ConstraintMatch:
    return ConstraintMatch(name, True, True, ConstraintStatus.SATISFIED)


def violated(name: str) -> ConstraintMatch:
    return ConstraintMatch(name, True, False, ConstraintStatus.VIOLATED)


def unknown(name: str) -> ConstraintMatch:
    return ConstraintMatch(name, True, None, ConstraintStatus.UNKNOWN)


# ─────────────────────────────────────────────────────────────────────────────
# features.py — schema and name alignment
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_row_covers_exactly_the_declared_schema() -> None:
    """A feature that exists but is not in FEATURE_NAMES would never be used;
    a name with no value would KeyError at matrix build. Both are silent-ish,
    so pin the schema."""
    row = build_feature_row(make_query(), make_candidate(), make_business())
    assert set(row) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_text_and_non_text_sets_partition_the_schema() -> None:
    assert set(TEXT_FEATURE_NAMES) | set(NON_TEXT_FEATURE_NAMES) == set(FEATURE_NAMES)
    assert not set(TEXT_FEATURE_NAMES) & set(NON_TEXT_FEATURE_NAMES)


def test_matrix_shape_and_column_alignment() -> None:
    """The failure this guards against: columns in a different order than the
    names claim, which produces numbers rather than an error."""
    triples = [
        (make_query(), make_candidate(), make_business(stars=4.5, review_count=10)),
        (make_query(), make_candidate(), make_business(stars=2.0, review_count=0)),
        (make_query(), make_candidate(), make_business(stars=5.0, review_count=999)),
    ]
    matrix = feature_matrix(triples)

    assert matrix.values.shape == (3, len(FEATURE_NAMES))
    assert matrix.names == FEATURE_NAMES
    assert matrix.column(feat.F_STARS).tolist() == [4.5, 2.0, 5.0]
    assert matrix.column(feat.F_LOG_REVIEWS)[1] == pytest.approx(0.0)  # log1p(0)
    assert matrix.column(feat.F_LOG_REVIEWS)[2] == pytest.approx(math.log1p(999))

    # feature_vector must agree with the matrix row for row.
    vector = feature_vector(*triples[0])
    np.testing.assert_allclose(vector, matrix.values[0], equal_nan=True)


def test_empty_input_still_has_the_right_width() -> None:
    """A zero-row matrix with zero columns would blow up at predict time with a
    shape error far from the cause."""
    matrix = feature_matrix([])
    assert matrix.values.shape == (0, len(FEATURE_NAMES))


def test_matrix_rejects_name_value_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        FeatureMatrix(("a", "b"), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="duplicate"):
        FeatureMatrix(("a", "a"), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="2-D"):
        FeatureMatrix(("a",), np.zeros(3))


# ─────────────────────────────────────────────────────────────────────────────
# features.py — missing values
# ─────────────────────────────────────────────────────────────────────────────


def test_absent_stage_scores_are_nan_not_zero() -> None:
    """A candidate the dense retriever never returned has *no* dense score. 0.0
    would place it among the weakly-similar documents, a claim the data never
    made; NaN lets LightGBM learn a direction for "absent"."""
    candidate = make_candidate(scores={SCORE_BM25: 8.0}, ranks={})
    row = build_feature_row(make_query(), candidate, make_business())

    assert row[feat.F_BM25] == 8.0
    assert math.isnan(row[feat.F_DENSE])
    assert math.isnan(row[feat.F_CROSS_ENCODER])
    assert math.isnan(row[feat.F_RRF_RANK])
    # And nothing silently became a sentinel.
    assert row[feat.F_DENSE] != 0.0 or math.isnan(row[feat.F_DENSE])


def test_present_zero_score_stays_zero() -> None:
    """The mirror image: a real measured 0.0 must not be confused with missing.
    If this ever returned NaN, genuine evidence would be thrown away."""
    candidate = make_candidate(scores={SCORE_BM25: 0.0, SCORE_DENSE: 0.0}, ranks={RANK_RRF: 1})
    row = build_feature_row(make_query(), candidate, make_business())
    assert row[feat.F_BM25] == 0.0
    assert row[feat.F_DENSE] == 0.0
    assert row[feat.F_RRF_RANK] == 1.0


def test_constraint_counts_three_valued() -> None:
    constraints = (satisfied("WiFi"), violated("NoiseLevel"), unknown("OutdoorSeating"))
    row = build_feature_row(make_query(), make_candidate(constraints=constraints), make_business())

    assert row[feat.F_N_SATISFIED] == 1.0
    assert row[feat.F_FRAC_SATISFIED] == pytest.approx(1 / 3)
    assert row[feat.F_HAS_UNKNOWN] == 1.0


def test_unconstrained_query_has_nan_fraction_but_defined_count() -> None:
    """0/0 is undefined, and 0.0 would read as "satisfied none of the request"
    for a query that asked for nothing. The count and the indicator *are*
    defined at zero, and that asymmetry is intentional."""
    row = build_feature_row(make_query(), make_candidate(constraints=()), make_business())
    assert math.isnan(row[feat.F_FRAC_SATISFIED])
    assert row[feat.F_N_SATISFIED] == 0.0
    assert row[feat.F_HAS_UNKNOWN] == 0.0


def test_all_constraints_satisfied_gives_fraction_one() -> None:
    constraints = (satisfied("WiFi"), satisfied("NoiseLevel"))
    row = build_feature_row(make_query(), make_candidate(constraints=constraints), make_business())
    assert row[feat.F_FRAC_SATISFIED] == pytest.approx(1.0)
    assert row[feat.F_HAS_UNKNOWN] == 0.0


def test_geo_constraint_excluded_from_counts() -> None:
    """Distance already carries the geo signal as a magnitude; counting the
    thresholded version too would feed the same evidence twice."""
    geo = GeoConstraint(latitude=36.17, longitude=-115.14, radius_km=2.0)
    constraints = (
        satisfied("WiFi"),
        ConstraintMatch(CONSTRAINT_GEO, geo, 0.5, ConstraintStatus.SATISFIED),
    )
    row = build_feature_row(
        make_query(geo=geo), make_candidate(constraints=constraints), make_business()
    )
    assert row[feat.F_N_SATISFIED] == 1.0
    assert row[feat.F_FRAC_SATISFIED] == pytest.approx(1.0)  # 1 of 1 counted constraint


@pytest.mark.parametrize(
    ("match", "expected"),
    [
        (ConstraintMatch(CONSTRAINT_CATEGORY, ("cafe",), "cafe", ConstraintStatus.SATISFIED), 1.0),
        (
            ConstraintMatch(CONSTRAINT_CATEGORY, ("cafe",), ("Bars",), ConstraintStatus.VIOLATED),
            0.0,
        ),
        (ConstraintMatch(CONSTRAINT_CATEGORY, ("cafe",), None, ConstraintStatus.UNKNOWN), None),
    ],
)
def test_category_match_is_three_valued(match: ConstraintMatch, expected: float | None) -> None:
    row = build_feature_row(
        make_query(categories=("cafe",)),
        make_candidate(constraints=(match,)),
        make_business(),
    )
    value = row[feat.F_CATEGORY_MATCH]
    if expected is None:
        assert math.isnan(value), "no category data must be NaN, not a 0 'wrong kind of place'"
    else:
        assert value == expected


def test_category_match_nan_when_query_asked_for_no_category() -> None:
    row = build_feature_row(make_query(categories=()), make_candidate(), make_business())
    assert math.isnan(row[feat.F_CATEGORY_MATCH])


def test_category_requested_but_unchecked_is_an_error() -> None:
    """Features built from pre-filter candidates would differ from the ones the
    serving path builds. Fail rather than return a quiet NaN."""
    with pytest.raises(ValueError, match="no category constraint"):
        build_feature_row(
            make_query(categories=("cafe",)), make_candidate(constraints=()), make_business()
        )


def test_distance_feature_matches_haversine_and_is_nan_without_geo() -> None:
    geo = GeoConstraint(latitude=36.1699, longitude=-115.1398, radius_km=3.0)
    business = make_business(latitude=36.1, longitude=-115.2)

    row = build_feature_row(make_query(geo=geo), make_candidate(), business)
    expected = haversine_km(36.1, -115.2, 36.1699, -115.1398)
    assert row[feat.F_DISTANCE_KM] == pytest.approx(expected)

    assert math.isnan(
        build_feature_row(make_query(), make_candidate(), business)[feat.F_DISTANCE_KM]
    )
    no_coords = make_business(latitude=None, longitude=None)
    row_missing = build_feature_row(make_query(geo=geo), make_candidate(), no_coords)
    assert math.isnan(row_missing[feat.F_DISTANCE_KM]), "0 km would mean 'at the anchor'"


# ─────────────────────────────────────────────────────────────────────────────
# features.py — ablation
# ─────────────────────────────────────────────────────────────────────────────


def test_ablation_keeps_only_text_features() -> None:
    assert ablated_feature_names(False) == FEATURE_NAMES
    assert ablated_feature_names(True) == TEXT_FEATURE_NAMES
    assert feat.F_STARS not in ablated_feature_names(True)


def test_select_drops_columns_rather_than_masking_them() -> None:
    matrix = feature_matrix([(make_query(), make_candidate(), make_business(stars=4.5))] * 2)
    ablated = matrix.select(TEXT_FEATURE_NAMES)
    assert ablated.names == TEXT_FEATURE_NAMES
    assert ablated.values.shape == (2, len(TEXT_FEATURE_NAMES))
    with pytest.raises(KeyError):
        ablated.column(feat.F_STARS)


# ─────────────────────────────────────────────────────────────────────────────
# train.py — groups
# ─────────────────────────────────────────────────────────────────────────────


def test_group_sizes_hand_built_three_query_example() -> None:
    """Hand-built so the expected answer is obvious by eye: three queries with
    3, 1 and 2 candidates must become [3, 1, 2]."""
    query_ids = ["qa", "qa", "qa", "qb", "qc", "qc"]
    group = group_sizes(query_ids)

    assert group.tolist() == [3, 1, 2]
    assert int(group.sum()) == len(query_ids), "group array must cover every row exactly once"
    assert len(group) == 3


def test_group_sizes_single_query_and_singletons() -> None:
    assert group_sizes(["q"] * 4).tolist() == [4]
    assert group_sizes(["a", "b", "c"]).tolist() == [1, 1, 1]


def test_group_sizes_refuses_interleaved_rows() -> None:
    """["a","b","a"] has no valid group array. Guessing [1,1,1] would form
    pairs across different queries — a meaningless objective that still
    trains."""
    with pytest.raises(ValueError, match="more than one block"):
        group_sizes(["a", "b", "a"])


def test_group_sizes_refuses_empty_input() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        group_sizes([])


def test_sort_by_query_makes_blocks_contiguous_in_first_seen_order() -> None:
    examples = [
        make_example("qb", label=1),
        make_example("qa", label=2),
        make_example("qb", label=0),
        make_example("qa", label=1),
    ]
    ordered = sort_by_query(examples)

    assert [e.query_id for e in ordered] == ["qb", "qb", "qa", "qa"]
    # Within a query, the original row order survives — needed for runs to be
    # byte-identical under a fixed seed.
    assert [e.label for e in ordered] == [1, 0, 2, 1]
    assert group_sizes([e.query_id for e in ordered]).tolist() == [2, 2]


def test_ranking_data_is_internally_consistent() -> None:
    examples = [make_example("qa"), make_example("qa"), make_example("qb")]
    data = build_ranking_data(examples)

    assert data.features.n_rows == 3
    assert len(data.labels) == 3
    assert data.n_queries == 2
    assert int(data.group.sum()) == 3
    assert data.query_ids == ("qa", "qa", "qb")


def test_ranking_data_rejects_inconsistent_pieces() -> None:
    """The dataclass is the last line of defence before LightGBM, which accepts
    a mismatched group array far too willingly."""
    matrix = feature_matrix([(make_query(), make_candidate(), make_business())] * 3)
    with pytest.raises(ValueError, match="groups sum"):
        ltr_train.RankingData(matrix, np.zeros(3), np.asarray([2], dtype=np.int32), ("a",) * 3)
    with pytest.raises(ValueError, match="labels"):
        ltr_train.RankingData(matrix, np.zeros(2), np.asarray([3], dtype=np.int32), ("a",) * 3)


def make_example(
    query_id: str = "q1",
    label: int = 1,
    stars: float = 4.0,
    split: str = SPLIT_LTR_TRAIN,
    bm25: float = 5.0,
    query_text: str = "quiet cafe with wifi",
) -> LtrExample:
    return LtrExample(
        query_id=query_id,
        query_text=query_text,
        label=label,
        parsed=ParsedQuery(raw=query_text, semantic_text=query_text),
        candidate=make_candidate(scores={SCORE_BM25: bm25, SCORE_DENSE: 0.5}, ranks={RANK_RRF: 3}),
        business=make_business(stars=stars),
        split=split,
    )


# ─────────────────────────────────────────────────────────────────────────────
# train.py — the out-of-fold cross-encoder guarantee
# ─────────────────────────────────────────────────────────────────────────────


def write_manifest(path: Path, splits: dict[str, list[str]], seed: int = 42) -> Path:
    path.write_text(json.dumps({"seed": seed, "splits": splits}))
    return path


def test_clean_split_manifest_passes_the_out_of_fold_check(tmp_path: Path) -> None:
    manifest = load_split_manifest(
        write_manifest(
            tmp_path / "splits.json",
            {
                SPLIT_CE_TRAIN: ["q1", "q2"],
                SPLIT_CE_VAL: ["q3"],
                SPLIT_LTR_TRAIN: ["q4", "q5"],
                SPLIT_LTR_VAL: ["q6"],
            },
        )
    )
    assert_out_of_fold(manifest)  # must not raise
    assert manifest.sizes == {
        SPLIT_CE_TRAIN: 2,
        SPLIT_CE_VAL: 1,
        SPLIT_LTR_TRAIN: 2,
        SPLIT_LTR_VAL: 1,
    }
    assert manifest.seed == 42


def test_planted_overlap_with_crossencoder_train_hard_fails(tmp_path: Path) -> None:
    """THE test for this stage. "q2" is in both the cross-encoder's training
    split and LambdaMART's, so its cross-encoder score is partly memorised and
    the feature is inflated. Training must refuse, loudly."""
    manifest = load_split_manifest(
        write_manifest(
            tmp_path / "splits.json",
            {
                SPLIT_CE_TRAIN: ["q1", "q2"],
                SPLIT_CE_VAL: ["q3"],
                SPLIT_LTR_TRAIN: ["q2", "q4"],  # ← planted leak
                SPLIT_LTR_VAL: ["q6"],
            },
        )
    )
    with pytest.raises(LeakageError) as excinfo:
        assert_out_of_fold(manifest)

    message = str(excinfo.value)
    assert SPLIT_LTR_TRAIN in message
    assert "q2" in message
    assert "over-trust" in message, "the error must explain the consequence, not just the fact"


def test_planted_overlap_with_crossencoder_val_also_fails(tmp_path: Path) -> None:
    """Stricter than the brief requires, on purpose: early stopping *selected a
    checkpoint* using the cross-encoder val split, so scores there are
    optimistic too — same bias, smaller size."""
    manifest = load_split_manifest(
        write_manifest(
            tmp_path / "splits.json",
            {SPLIT_CE_TRAIN: ["q1"], SPLIT_CE_VAL: ["q7"], SPLIT_LTR_TRAIN: ["q7"]},
        )
    )
    with pytest.raises(LeakageError):
        assert_out_of_fold(manifest)


def test_overlap_in_ltr_val_also_fails(tmp_path: Path) -> None:
    """ltr_val chooses the boosting iteration, so contamination there biases the
    model that ships."""
    manifest = load_split_manifest(
        write_manifest(
            tmp_path / "splits.json",
            {SPLIT_CE_TRAIN: ["q1"], SPLIT_LTR_TRAIN: ["q2"], SPLIT_LTR_VAL: ["q1"]},
        )
    )
    with pytest.raises(LeakageError):
        assert_out_of_fold(manifest)


def test_out_of_fold_check_can_be_disabled_but_screams(tmp_path: Path, caplog) -> None:
    """The escape hatch exists to *measure* the inflation, not for convenience,
    so it must leave a record."""
    manifest = load_split_manifest(
        write_manifest(
            tmp_path / "splits.json",
            {SPLIT_CE_TRAIN: ["q1"], SPLIT_LTR_TRAIN: ["q1"]},
        )
    )
    with caplog.at_level("ERROR"):
        assert_out_of_fold(manifest, enforce=False)
    assert "OUT-OF-FOLD CHECK DISABLED" in caplog.text


def test_manifest_accepts_id_to_split_form(tmp_path: Path) -> None:
    path = tmp_path / "flat.json"
    path.write_text(json.dumps({"q1": SPLIT_CE_TRAIN, "q2": SPLIT_LTR_TRAIN, "seed": 11}))
    manifest = load_split_manifest(path)
    assert manifest.ids(SPLIT_CE_TRAIN) == frozenset({"q1"})
    assert manifest.ids(SPLIT_LTR_TRAIN) == frozenset({"q2"})
    assert manifest.seed == 11
    assert_out_of_fold(manifest)


def test_reads_the_manifest_the_synth_stage_actually_writes(tmp_path: Path) -> None:
    """Cross-module seam test: write with the real writer, read with this reader.

    The other manifest tests hand-build JSON, so all of them passed while this
    reader could not parse the only manifest that ever exists on disk — the
    writer stores the assignment under ``assignment`` next to its checksum and
    fraction bookkeeping, and the reader was treating those extra keys as split
    names. That made `make train-ltr` fail at startup and, worse, meant the
    out-of-fold guarantee was never exercised against a real artifact.
    """
    from concierge.config import load_config
    from concierge.synth import splits as synth_splits

    cfg = load_config("configs/base.yaml")
    items = [(f"q{i:03d}", f"q{i:03d}") for i in range(40)]
    written = synth_splits.assign_splits(items, cfg)
    path = synth_splits.save_manifest(written, tmp_path / "synth_splits.json")

    manifest = load_split_manifest(path)

    assert manifest.seed == cfg.seed
    assert manifest.sizes == written.counts
    for name in (SPLIT_CE_TRAIN, SPLIT_CE_VAL, SPLIT_LTR_TRAIN, SPLIT_LTR_VAL):
        assert manifest.ids(name) == frozenset(written.ids(name)), name
    # The whole point of reading it: the guarantee must be checkable on it.
    assert_out_of_fold(manifest)


def test_missing_manifest_is_an_error_not_a_pass(tmp_path: Path) -> None:
    """A safety check that passes because its input is absent is worse than no
    check: it reads as reassurance it has not earned."""
    with pytest.raises(FileNotFoundError, match="split manifest not found"):
        load_split_manifest(tmp_path / "nope.json")


# ─────────────────────────────────────────────────────────────────────────────
# train.py — gold leakage
# ─────────────────────────────────────────────────────────────────────────────


def write_gold(path: Path, rows: list[tuple[str, str]]) -> Path:
    lines = ["query_id,query,query_type"]
    lines += [f"{qid},{query},constraint" for qid, query in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_gold_query_text_in_training_data_fails(tmp_path: Path) -> None:
    """Caught by normalized text even when the id differs — a synthetic query
    that happens to equal a gold query is still leakage."""
    gold = write_gold(tmp_path / "gold.csv", [("g001", "cheap ramen with outdoor seating")])
    examples = [make_example("s001", query_text="Cheap Ramen with outdoor seating.")]
    with pytest.raises(LeakageError, match="gold queries found"):
        assert_no_gold_leakage(examples, gold)


def test_gold_query_id_in_training_data_fails(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.csv", [("g001", "some gold query")])
    examples = [make_example("g001", query_text="a completely different text")]
    with pytest.raises(LeakageError, match="by query_id"):
        assert_no_gold_leakage(examples, gold)


def test_clean_training_data_passes_gold_check(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.csv", [("g001", "cheap ramen downtown")])
    assert_no_gold_leakage([make_example("s001", query_text="quiet cafe with wifi")], gold)


def test_missing_gold_file_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "gold_queries.csv"
    with pytest.raises(LeakageError, match="Refusing to train"):
        assert_no_gold_leakage([make_example()], missing)
    # ...unless explicitly overridden for the period before the gold set exists.
    assert_no_gold_leakage([make_example()], missing, allow_missing_gold=True)


# ─────────────────────────────────────────────────────────────────────────────
# train.py — the training loop itself
# ─────────────────────────────────────────────────────────────────────────────

# LightGBM's defaults assume thousands of rows. On a toy problem no tree would
# ever split, so these loosen the leaf/bin minimums. Production runs pass
# nothing and use the config's values untouched.
TINY_DATA_PARAMS = {
    "min_data_in_leaf": 2,
    "min_sum_hessian_in_leaf": 1e-4,
    "min_data_in_bin": 1,
    "max_bin": 63,
    "feature_pre_filter": False,
}


def tiny_config(config: Config) -> Config:
    """The real config with the round count cut, so tests stay fast without
    changing any modelling parameter the project reports."""
    return config.model_copy(
        update={
            "ltr": config.ltr.model_copy(
                update={"n_estimators": 60, "early_stopping_rounds": 20, "num_leaves": 7}
            )
        }
    )


def monotone_dataset(n_queries: int, seed: int) -> list[LtrExample]:
    """A ranking problem whose right answer is known by construction.

    The label is a monotone function of ``stars`` alone; every other feature is
    noise drawn independently of the label. A correctly wired LambdaMART must
    therefore (a) rank by stars within each query and (b) put its importance on
    the stars feature. If the group array were wrong, or the labels misaligned
    with the rows, neither would hold — which is exactly what makes this a test
    of the training *plumbing* and not of LightGBM.
    """
    rng = random.Random(seed)
    examples: list[LtrExample] = []
    for q in range(n_queries):
        query_id = f"s{q:03d}"
        for stars in (1.0, 2.0, 3.0, 4.0, 5.0):
            # Label: 0 for poor, 1 for middling, 2 for good. Monotone in stars.
            label = 0 if stars <= 2.0 else (1 if stars <= 3.0 else 2)
            examples.append(
                LtrExample(
                    query_id=query_id,
                    query_text=f"synthetic query {q}",
                    label=label,
                    parsed=ParsedQuery(raw=f"synthetic query {q}", semantic_text="x"),
                    candidate=Candidate(
                        business_id=f"b{q}_{int(stars)}",
                        # Noise, independent of the label: the model must not be
                        # able to reach the right answer through these.
                        scores={
                            SCORE_BM25: rng.uniform(0, 10),
                            SCORE_DENSE: rng.uniform(0, 1),
                            SCORE_CROSS_ENCODER: rng.uniform(-5, 5),
                        },
                        ranks={RANK_RRF: rng.randint(1, 50)},
                        constraints=(),
                    ),
                    business=make_business(
                        business_id=f"b{q}_{int(stars)}",
                        stars=stars,
                        review_count=rng.randint(1, 500),
                    ),
                    split=SPLIT_LTR_TRAIN,
                )
            )
    return examples


def test_recovers_monotone_label(config: Config) -> None:
    cfg = tiny_config(config)
    train_data = build_ranking_data(monotone_dataset(40, seed=1))
    val_data = build_ranking_data(monotone_dataset(10, seed=2))

    ranker = train_lambdamart(train_data, val_data, cfg, extra_params=TINY_DATA_PARAMS)

    # (a) Within every query, ranking by the model must put the labels in
    # non-increasing order — i.e. a perfect ranking.
    #
    # The assertion is on *labels*, not on stars, and that distinction is the
    # point: the label buckets ties (stars 4 and 5 are both label 2), so the
    # data contains no evidence for an order within a bucket. Demanding that
    # the model also sort stars 5 above stars 4 would be demanding it invent
    # information, and a test that insists on invented information is a test
    # that rewards overfitting.
    held_out = build_ranking_data(monotone_dataset(5, seed=99))
    predictions = ranker.booster.predict(held_out.features.values)
    stars = held_out.features.column(feat.F_STARS)
    start = 0
    for size in held_out.group:
        block_pred = predictions[start : start + size]
        block_labels = held_out.labels[start : start + size]
        block_stars = stars[start : start + size]
        order_by_pred = np.argsort(-block_pred, kind="stable")
        ranked_labels = block_labels[order_by_pred].tolist()
        assert ranked_labels == sorted(ranked_labels, reverse=True), (
            "the model did not recover the known ordering — the training loop, "
            "the group array, or the label alignment is wrong"
        )
        # And the ordering really is driven by the informative feature: every
        # high-star candidate outranks every low-star one.
        best = block_pred[block_stars >= 4.0].min()
        worst = block_pred[block_stars <= 2.0].max()
        assert best > worst
        start += size

    # (b) The informative feature must dominate the importance table. This is
    # the check that the feature *names* are attached to the right columns.
    importances = ranker.importances("gain")
    assert max(importances, key=importances.get) == feat.F_STARS
    assert importances[feat.F_STARS] > 0

    # Early stopping produced a usable checkpoint and a real recorded metric.
    assert 1 <= ranker.best_iteration <= cfg.ltr.n_estimators
    assert ranker.val_ndcg_at_10 is not None
    assert 0.0 <= ranker.val_ndcg_at_10 <= 1.0
    assert ranker.n_train_queries == 40
    assert ranker.n_train_rows == 200


def test_training_is_deterministic_under_a_fixed_seed(config: Config) -> None:
    """Reproducibility is a stated project promise, and the seed is only
    meaningful if the whole path (row order, groups, LightGBM internals) is
    pinned. Byte-identical predictions, not approximately equal."""
    cfg = tiny_config(config)
    train_data = build_ranking_data(monotone_dataset(20, seed=1))
    val_data = build_ranking_data(monotone_dataset(6, seed=2))

    first = train_lambdamart(train_data, val_data, cfg, extra_params=TINY_DATA_PARAMS)
    second = train_lambdamart(train_data, val_data, cfg, extra_params=TINY_DATA_PARAMS)

    np.testing.assert_array_equal(
        first.booster.predict(val_data.features.values),
        second.booster.predict(val_data.features.values),
    )
    assert first.best_iteration == second.best_iteration
    assert first.val_ndcg_at_10 == second.val_ndcg_at_10


def test_ablated_model_cannot_see_the_non_text_features(config: Config) -> None:
    """The ablation has to actually remove the columns, or the "do ratings
    matter?" measurement is comparing a model against itself."""
    cfg = tiny_config(config)
    names = ablated_feature_names(True)
    train_data = build_ranking_data(monotone_dataset(20, seed=1), names)
    ranker = train_lambdamart(train_data, None, cfg, extra_params=TINY_DATA_PARAMS)

    assert ranker.feature_names == TEXT_FEATURE_NAMES
    assert feat.F_STARS not in ranker.importances()
    assert ranker.booster.feature_name() == list(TEXT_FEATURE_NAMES)
    assert ranker.val_ndcg_at_10 is None, "no validation split means no validation metric"


def test_train_val_feature_mismatch_is_caught(config: Config) -> None:
    cfg = tiny_config(config)
    full = build_ranking_data(monotone_dataset(6, seed=1))
    text_only = build_ranking_data(monotone_dataset(6, seed=2), TEXT_FEATURE_NAMES)
    with pytest.raises(ValueError, match="differ from"):
        train_lambdamart(full, text_only, cfg, extra_params=TINY_DATA_PARAMS)


def test_lgb_params_pin_every_seed(config: Config) -> None:
    params = ltr_train.lgb_params(config)
    assert params["objective"] == config.ltr.objective == "lambdarank"
    assert params["eval_at"] == [10]
    for key in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        assert params[key] == config.seed
    assert params["deterministic"] is True


# ─────────────────────────────────────────────────────────────────────────────
# train.py — record parsing and the end-to-end run
# ─────────────────────────────────────────────────────────────────────────────


def test_example_from_json_materializes_the_contract_types() -> None:
    """The training job reconstructs ParsedQuery / Candidate / Business and then
    calls the *same* feature function serving calls. That is the structural
    defence against train/serve feature skew, so the parsing is pinned."""
    record = {
        "query_id": "s001",
        "split": SPLIT_LTR_TRAIN,
        "label": 2,
        "query": {
            "raw": "quiet cafe open late",
            "semantic_text": "quiet cafe",
            "attributes": {"WiFi": "free"},
            "categories": ["Coffee & Tea"],
            "open_at": {"day": "Friday", "minute_of_day": 1320},
            "price_max": 2,
            "geo": {"latitude": 36.1, "longitude": -115.2, "radius_km": 3.0},
        },
        "business": {
            "business_id": "b001",
            "stars": 4.5,
            "review_count": 120,
            "categories": ["Coffee & Tea"],
            "latitude": 36.11,
            "longitude": -115.21,
        },
        "scores": {"bm25": 8.1, "dense": 0.74, "cross_encoder": 3.2},
        "ranks": {"rrf": 4},
        "constraints": [
            {"name": "WiFi", "requested": "free", "actual": "free", "status": "satisfied"},
            {
                "name": CONSTRAINT_CATEGORY,
                "requested": ["Coffee & Tea"],
                "actual": "Coffee & Tea",
                "status": "satisfied",
            },
        ],
    }
    example = example_from_json(record)

    assert example.query_id == "s001"
    assert example.split == SPLIT_LTR_TRAIN
    assert example.label == 2
    assert example.parsed.open_at is not None and example.parsed.open_at.minute_of_day == 1320
    assert example.parsed.n_hard_constraints == 4  # open_at + WiFi + price + geo
    assert example.business.stars == 4.5
    assert example.candidate.ranks["rrf"] == 4

    row = build_feature_row(example.parsed, example.candidate, example.business)
    assert row[feat.F_CATEGORY_MATCH] == 1.0
    assert row[feat.F_FRAC_SATISFIED] == pytest.approx(1.0)
    assert not math.isnan(row[feat.F_DISTANCE_KM])


def test_bad_record_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text('{"query_id": "s1", "label": 1, "query": {}, "business": {}}\n{"nope": 1}\n')
    with pytest.raises(ValueError, match=r"examples.jsonl:2"):
        ltr_train.load_examples(path)


def write_examples_jsonl(path: Path, examples: list[LtrExample]) -> Path:
    """Serialize examples back to the JSONL shape the trainer reads, so the
    end-to-end test exercises the real loader rather than bypassing it."""
    lines = []
    for e in examples:
        lines.append(
            json.dumps(
                {
                    "query_id": e.query_id,
                    "split": e.split,
                    "label": e.label,
                    "query": {"raw": e.query_text, "semantic_text": e.parsed.semantic_text},
                    "business": {
                        "business_id": e.business.business_id,
                        "stars": e.business.stars,
                        "review_count": e.business.review_count,
                        "categories": list(e.business.categories),
                        "latitude": e.business.latitude,
                        "longitude": e.business.longitude,
                    },
                    "scores": e.candidate.scores,
                    "ranks": e.candidate.ranks,
                    "constraints": [],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def with_gold(cfg: Config, gold_path: Path) -> Config:
    """Point the config's gold-query path at a test file.

    Needed so these tests do not depend on whether the owner has written
    ``eval/gold_queries.csv`` yet: a test whose outcome flips when an unrelated
    repo file appears is a test that will be "fixed" by deleting the assertion.
    """
    return cfg.model_copy(
        update={"evaluation": cfg.evaluation.model_copy(update={"gold_queries": gold_path})}
    )


def test_run_writes_model_names_and_measured_metadata(tmp_path: Path, config: Config) -> None:
    gold = write_gold(tmp_path / "gold.csv", [("g001", "a gold query nobody trains on")])
    cfg = with_gold(tiny_config(config), gold)
    train_examples = monotone_dataset(20, seed=1)
    val_examples = [replace(e, split=SPLIT_LTR_VAL) for e in monotone_dataset(6, seed=2)]
    examples_path = write_examples_jsonl(
        tmp_path / "examples.jsonl", [*train_examples, *val_examples]
    )
    manifest_path = write_manifest(
        tmp_path / "splits.json",
        {
            SPLIT_CE_TRAIN: ["ce000"],
            SPLIT_CE_VAL: ["ce001"],
            SPLIT_LTR_TRAIN: sorted({e.query_id for e in train_examples}),
            SPLIT_LTR_VAL: sorted({e.query_id for e in val_examples}),
        },
    )
    out_dir = tmp_path / "artifact"

    directory = ltr_train.run(cfg, examples_path, manifest_path, out_dir=out_dir)

    assert (directory / "model.txt").exists()
    assert json.loads((directory / "feature_names.json").read_text()) == list(FEATURE_NAMES)

    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["config_fingerprint"] == cfg.fingerprint()
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["n_train_rows"] == 100
    assert metadata["n_train_queries"] == 20
    assert metadata["n_val_queries"] == 6
    assert metadata["ablated_non_text_features"] is False
    assert metadata["enforce_oof_crossencoder_feature"] is True
    assert metadata["gold_leakage_check_ran"] is True
    assert metadata["split_sizes"][SPLIT_LTR_TRAIN] == 20
    assert set(metadata["importance_gain"]) == set(FEATURE_NAMES)
    assert 1 <= metadata["best_iteration"] <= cfg.ltr.n_estimators


def test_metadata_records_a_skipped_gold_check_rather_than_implying_it_passed(
    tmp_path: Path, config: Config
) -> None:
    """Honesty requirement: before the owner has written the gold set, the
    artifact must say the check did not run. A metadata file that is silent on
    this would let a later reader assume the model was verified."""
    cfg = with_gold(tiny_config(config), tmp_path / "does_not_exist.csv")
    examples = monotone_dataset(10, seed=1)
    examples_path = write_examples_jsonl(tmp_path / "examples.jsonl", examples)
    manifest_path = write_manifest(
        tmp_path / "splits.json",
        {SPLIT_CE_TRAIN: ["ce000"], SPLIT_LTR_TRAIN: sorted({e.query_id for e in examples})},
    )

    directory = ltr_train.run(
        cfg,
        examples_path,
        manifest_path,
        out_dir=tmp_path / "artifact",
        allow_missing_gold=True,
    )
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["gold_leakage_check_ran"] is False
    assert metadata[f"val_ndcg_at_{ltr_train.EARLY_STOPPING_NDCG_K}"] is None


def test_run_refuses_a_leaky_manifest_before_training(tmp_path: Path, config: Config) -> None:
    """Order matters: the check must fire before any model exists, so a
    contaminated run cannot leave an artifact somebody later evaluates."""
    cfg = tiny_config(config)
    examples = monotone_dataset(5, seed=1)
    examples_path = write_examples_jsonl(tmp_path / "examples.jsonl", examples)
    leaky = write_manifest(
        tmp_path / "splits.json",
        {
            SPLIT_CE_TRAIN: [examples[0].query_id],
            SPLIT_LTR_TRAIN: sorted({e.query_id for e in examples}),
        },
    )
    out_dir = tmp_path / "artifact"

    with pytest.raises(LeakageError):
        ltr_train.run(cfg, examples_path, leaky, out_dir=out_dir, allow_missing_gold=True)
    assert not out_dir.exists(), "no artifact may be written for a leaky run"


def test_run_with_ablation_writes_a_separate_artifact(tmp_path: Path, config: Config) -> None:
    gold = write_gold(tmp_path / "gold.csv", [("g001", "a gold query nobody trains on")])
    cfg = with_gold(tiny_config(config), gold)
    examples = monotone_dataset(10, seed=1)
    examples_path = write_examples_jsonl(tmp_path / "examples.jsonl", examples)
    manifest_path = write_manifest(
        tmp_path / "splits.json",
        {SPLIT_CE_TRAIN: ["ce000"], SPLIT_LTR_TRAIN: sorted({e.query_id for e in examples})},
    )

    full = ltr_train.default_out_dir(cfg, ablated_non_text=False)
    ablated = ltr_train.default_out_dir(cfg, ablated_non_text=True)
    assert full != ablated, "the two runs must not overwrite each other"
    assert ablated.name.endswith("-textonly")

    directory = ltr_train.run(
        cfg,
        examples_path,
        manifest_path,
        ablate_non_text=True,
        out_dir=tmp_path / "ablated",
    )
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["ablated_non_text_features"] is True
    assert metadata["feature_names"] == list(TEXT_FEATURE_NAMES)
    assert metadata["non_text_features"] == list(NON_TEXT_FEATURE_NAMES)


def test_run_requires_ltr_train_rows(tmp_path: Path, config: Config) -> None:
    cfg = tiny_config(config)
    val_only = [replace(e, split=SPLIT_LTR_VAL) for e in monotone_dataset(3, seed=1)]
    examples_path = write_examples_jsonl(tmp_path / "examples.jsonl", val_only)
    manifest_path = write_manifest(tmp_path / "splits.json", {SPLIT_LTR_VAL: ["s000"]})
    with pytest.raises(ValueError, match="no examples with split"):
        ltr_train.run(
            cfg, examples_path, manifest_path, out_dir=tmp_path / "a", allow_missing_gold=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# train.py -> pipeline.py — the trained-artifact seam
# ─────────────────────────────────────────────────────────────────────────────


def test_a_saved_ranker_loads_into_the_serving_adapter(tmp_path: Path, config: Config) -> None:
    """Train for real, save, then load through the *serving* adapter.

    Every other Stage 4 serving test substitutes a fake booster, so the one
    thing this seam exists to guarantee — that `save_ranker`'s two copies of the
    column order survive a LightGBM round trip and satisfy
    `pipeline.check_feature_order` — was never exercised against a real model
    file. A permutation here yields plausible predictions and no error.
    """
    from concierge.pipeline import LightGbmRanker

    cfg = tiny_config(config)
    ranker = train_lambdamart(
        build_ranking_data(monotone_dataset(40, seed=1)),
        build_ranking_data(monotone_dataset(10, seed=2)),
        cfg,
        extra_params=TINY_DATA_PARAMS,
    )
    directory = ltr_train.save_ranker(ranker, tmp_path / "ltr", cfg)

    served = LightGbmRanker.load(cfg, directory)

    assert served.feature_names == tuple(FEATURE_NAMES)
    assert list(served.booster.feature_name()) == list(FEATURE_NAMES)

    # And the loaded model still ranks the known-answer problem correctly, which
    # proves the saved file is the trained model and not an empty booster.
    held_out = monotone_dataset(3, seed=99)
    parsed = held_out[0].parsed
    businesses = {e.business.business_id: e.business for e in held_out}
    candidates = [e.candidate for e in held_out if e.query_id == held_out[0].query_id]
    scores = served.score(parsed, candidates, businesses)

    assert len(scores) == len(candidates)
    by_stars = sorted(
        zip(scores, (businesses[c.business_id].stars for c in candidates), strict=True),
        key=lambda pair: -pair[0],
    )
    assert by_stars[0][1] >= by_stars[-1][1], "the reloaded model lost its ordering"


def test_default_manifest_path_points_where_the_synth_stage_writes(
    config: Config, tmp_path: Path
) -> None:
    """`make train-ltr` passes no --splits, so this default is the only path it
    ever uses. It pointed at data/processed/synth_splits.json, which nothing in
    the repo writes — the synth stage writes
    artifacts/synth/<fingerprint>/splits.json — so the command could only fail.
    """
    from concierge.synth.splits import splits_path

    assert ltr_train.default_manifest_path(config) == splits_path(config)


def test_default_manifest_path_still_honours_a_legacy_file(
    config: Config, monkeypatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "synth_splits.json"
    legacy.write_text(json.dumps({"seed": 1, "splits": {SPLIT_LTR_TRAIN: ["q1"]}}))
    monkeypatch.setattr(ltr_train, "DEFAULT_MANIFEST_PATH", legacy)
    assert ltr_train.default_manifest_path(config) == legacy
