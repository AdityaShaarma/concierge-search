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
    assert_run_followed_the_manifest,
    build_ranking_data,
    crossencoder_run_check,
    example_from_json,
    group_sizes,
    load_verified_manifest,
    select_split_rows,
    sort_by_query,
    train_lambdamart,
)
from concierge.retrieval.filters import (
    CONSTRAINT_CATEGORY,
    CONSTRAINT_GEO,
    check_categories,
    haversine_km,
)
from concierge.synth import splits as synth_splits
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


def test_category_constraint_excluded_from_counts() -> None:
    """`category_match` already carries this verdict; counting it as well would
    feed one piece of evidence to the model twice — and would do worse than
    that to `has_unknown_constraint_data`.

    `filters.check_categories` is lexical and Yelp's category lists are
    incomplete, so it answers UNKNOWN (never VIOLATED) when it cannot establish
    a match. Counted, that UNKNOWN sets `has_unknown_constraint_data` on
    essentially every candidate of every query naming a kind of place, and the
    feature stops meaning "this business's attribute data is missing" — which is
    the thing it exists to let the model discount. The business below has
    complete attribute data and one satisfied amenity check; the only thing the
    system does not know about it is a category word, and that must not be
    reported as missing data.
    """
    constraints = (
        satisfied("WiFi"),
        ConstraintMatch(CONSTRAINT_CATEGORY, ("Ramen",), ("Japanese",), ConstraintStatus.UNKNOWN),
    )
    row = build_feature_row(
        make_query(categories=("Ramen",)),
        make_candidate(constraints=constraints),
        make_business(categories=("Japanese", "Noodles")),
    )

    assert row[feat.F_HAS_UNKNOWN] == 0.0, "a category lexicon miss is not missing attribute data"
    assert row[feat.F_N_SATISFIED] == 1.0, "only the amenity check is counted"
    assert row[feat.F_FRAC_SATISFIED] == pytest.approx(1.0)  # 1 of 1 counted constraint
    # The category evidence is not lost, it is carried by its own feature.
    assert math.isnan(row[feat.F_CATEGORY_MATCH])


def test_a_category_only_query_has_no_counted_constraints() -> None:
    """The consequence of the exclusion above, asserted rather than left implicit.

    A query that named a kind of place and nothing else imposes no arithmetic
    constraint, so there is no fraction to satisfy: 0/0, the same answer as a
    wholly unconstrained query. 0.0 would read as "satisfied none of the
    request" about a request the candidate in fact matched.
    """
    match = ConstraintMatch(CONSTRAINT_CATEGORY, ("cafe",), "cafe", ConstraintStatus.SATISFIED)
    row = build_feature_row(
        make_query(categories=("cafe",)), make_candidate(constraints=(match,)), make_business()
    )

    assert row[feat.F_N_SATISFIED] == 0.0
    assert math.isnan(row[feat.F_FRAC_SATISFIED])
    assert row[feat.F_HAS_UNKNOWN] == 0.0
    assert row[feat.F_CATEGORY_MATCH] == 1.0, "the category signal still reaches the model"


def test_counted_constraints_are_the_schema_minus_the_two_with_own_features() -> None:
    """Pins the exclusion set, so adding a constraint kind is a deliberate act.

    Both exclusions exist for the same reason — the constraint has a dedicated
    feature of its own — and a third one added without that justification would
    quietly remove evidence from the count features.
    """
    excluded = feat._CONSTRAINTS_EXCLUDED_FROM_COUNTS
    assert excluded == frozenset({CONSTRAINT_GEO, CONSTRAINT_CATEGORY})


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


def test_category_match_is_computable_without_the_filter_stage() -> None:
    """The `no_filters` ablation runs Stage 4 with no constraint checks attached.

    This used to raise, which made every ablation query containing a category
    word score zero and overstated what the hard filters are worth — an
    ablation row that measured a crash while reading as a result. The feature
    must be computable from the business's own categories instead, and it must
    agree with the filter stage's verdict, since both call check_categories.
    """
    business = make_business(categories=("Coffee & Tea",))

    hit = build_feature_row(
        make_query(categories=("Coffee",)), make_candidate(constraints=()), business
    )
    assert hit[feat.F_CATEGORY_MATCH] == 1.0
    assert hit[feat.F_CATEGORY_MATCH] == _category_feature_from_the_filter_stage(
        business, ("Coffee",)
    )

    # A lexical miss is NaN, not 0.0, and that is `filters.check_categories`'
    # documented three-valued reading rather than a shortcut here: Yelp's
    # category lists are incomplete, so "this matcher could not establish a
    # match" is not "the data refutes the request". The assertion is written
    # against the filter stage's own verdict so the two cannot drift apart.
    miss = build_feature_row(
        make_query(categories=("Ramen",)), make_candidate(constraints=()), business
    )
    assert math.isnan(miss[feat.F_CATEGORY_MATCH])
    assert math.isnan(_category_feature_from_the_filter_stage(business, ("Ramen",)))

    # No category data on the business is the same "no signal" answer, reached
    # by a different route — UNKNOWN with no `actual` list at all.
    no_data = build_feature_row(
        make_query(categories=("Ramen",)),
        make_candidate(constraints=()),
        make_business(categories=()),
    )
    assert math.isnan(no_data[feat.F_CATEGORY_MATCH])


def _category_feature_from_the_filter_stage(
    business: Business, categories: tuple[str, ...]
) -> float:
    """What the feature would be if the filter stage HAD attached its check.

    Written out in the test rather than reusing the production translation, so
    the assertion above compares two independently derived values.
    """
    status = check_categories(business, categories).status
    return {ConstraintStatus.SATISFIED: 1.0, ConstraintStatus.VIOLATED: 0.0}.get(
        status, float("nan")
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
#
# Every test below plants ONE specific failure in an otherwise valid manifest
# and asserts the trainer raises. They are written that way because the audit
# of the previous version found the opposite: a guard that passed when its
# input was missing or misspelled. A check that cannot be made to fail is not
# evidence of anything, so each of these starts from a manifest the real reader
# accepts and breaks exactly one thing.
#
# Note the shape of the manifest: one query id maps to one split name, so "this
# query is in two splits" cannot be written down at all. The leak therefore
# arrives in one of the forms tested here — a stale examples file, an edited
# manifest, a manifest from another configuration, or a cross-encoder fine-tuned
# under a different manifest than the one LambdaMART verifies.


def assignment_of(by_split: dict[str, list[str]]) -> dict[str, str]:
    """``{split: [query_id]}`` flipped into the manifest's ``{query_id: split}``."""
    return {query_id: split for split, ids in by_split.items() for query_id in ids}


def write_manifest(
    path: Path,
    cfg: Config,
    by_split: dict[str, list[str]],
    *,
    seed: int | None = None,
    fractions: dict[str, float] | None = None,
    config_fingerprint: str | None = None,
) -> Path:
    """Write a manifest the real reader accepts, unless a keyword breaks it.

    Checksum, seed, fractions and config fingerprint all default to agreeing
    with ``cfg``, because that is what ``concierge.synth.splits`` verifies now
    that the trainer reads the manifest through it. Hand-written JSON that
    skipped those fields used to be accepted, which is how a manifest edited to
    clear an out-of-fold failure would have trained a model.
    """
    assignment = assignment_of(by_split)
    seed = cfg.seed if seed is None else seed
    fractions = synth_splits.fractions_from_config(cfg) if fractions is None else fractions
    group_by = "query"
    manifest = synth_splits.SplitManifest(
        seed=seed,
        group_by=group_by,
        fractions=fractions,
        assignment=assignment,
        checksum=synth_splits.compute_checksum(assignment, seed, fractions, group_by),
        n_groups=len(assignment),
        config_fingerprint=(
            cfg.fingerprint() if config_fingerprint is None else config_fingerprint
        ),
    )
    return synth_splits.save_manifest(manifest, path)


def test_a_verified_manifest_is_loaded_and_summarised(config: Config, tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "splits.json",
        config,
        {
            SPLIT_CE_TRAIN: ["q1", "q2"],
            SPLIT_CE_VAL: ["q3"],
            SPLIT_LTR_TRAIN: ["q4", "q5"],
            SPLIT_LTR_VAL: ["q6"],
        },
    )

    manifest, resolved = load_verified_manifest(config, path)

    assert resolved == path
    assert manifest.counts == {
        SPLIT_CE_TRAIN: 2,
        SPLIT_CE_VAL: 1,
        SPLIT_LTR_TRAIN: 2,
        SPLIT_LTR_VAL: 1,
    }
    assert manifest.seed == config.seed
    # The checksum is what lets the cross-encoder's artifact and this one be
    # shown to refer to the same split, so it must survive the round trip.
    assert manifest.checksum == synth_splits.load_manifest(path).checksum


def test_rows_are_selected_by_the_manifest_not_by_their_own_split_label(
    config: Config, tmp_path: Path
) -> None:
    """The row's ``split`` string is not permitted to decide anything.

    This is the leak the previous version shipped: training rows were filtered
    on the string each JSONL record carried, so a stale examples file whose rows
    said ``ltr_train`` while the manifest said ``crossencoder_train`` trained
    LambdaMART on in-fold cross-encoder scores with every check reporting clean.
    """
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(
            tmp_path / "splits.json",
            config,
            {SPLIT_CE_TRAIN: ["q1"], SPLIT_LTR_TRAIN: ["q2"], SPLIT_LTR_VAL: ["q3"]},
        ),
    )
    # The row claims ltr_train; the manifest says this query trained the
    # cross-encoder. One of the two is stale and from here there is no way to
    # know which, so the only safe answer is to refuse.
    stale = [make_example("q1", split=SPLIT_LTR_TRAIN)]

    with pytest.raises(LeakageError, match="disagree with the split manifest"):
        select_split_rows(stale, manifest, SPLIT_LTR_TRAIN)


def test_the_manifest_decides_even_when_the_disagreement_is_downgraded(
    config: Config, tmp_path: Path, caplog
) -> None:
    """``enforce=False`` silences the refusal, not the manifest.

    The escape hatch exists to *measure* what the inflation is worth. It must
    not change which rows are selected, or the measurement would be of a
    different dataset than the one the manifest describes.
    """
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(
            tmp_path / "splits.json",
            config,
            {SPLIT_CE_TRAIN: ["q1"], SPLIT_LTR_TRAIN: ["q2"], SPLIT_LTR_VAL: ["q3"]},
        ),
    )
    rows = [make_example("q1", split=SPLIT_LTR_TRAIN), make_example("q2", split=SPLIT_LTR_TRAIN)]

    with caplog.at_level("ERROR"):
        selected = select_split_rows(rows, manifest, SPLIT_LTR_TRAIN, enforce=False)

    assert "OUT-OF-FOLD CHECK DISABLED" in caplog.text
    assert [e.query_id for e in selected] == ["q2"], "the manifest still chose the rows"


def test_a_row_the_manifest_never_assigned_is_refused(config: Config, tmp_path: Path) -> None:
    """Training data nobody assigned is not out-of-fold anything."""
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}),
    )
    with pytest.raises(LeakageError, match="not in the split manifest"):
        select_split_rows([make_example("q_unknown")], manifest, SPLIT_LTR_TRAIN)


def test_an_unknown_split_name_raises_instead_of_selecting_nothing(
    config: Config, tmp_path: Path
) -> None:
    """THE vacuous-check test. A typo must not silently select zero rows.

    The previous version only logged a warning for an unrecognised split name,
    which left the leakage check comparing against an empty set — and a check
    that looks at nothing always passes. The misspelling here is one keystroke
    from the real name, which is exactly how it would arrive.
    """
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}),
    )
    with pytest.raises(LeakageError, match="unknown split"):
        select_split_rows([make_example("q1")], manifest, SPLIT_LTR_TRAIN + "x")


def test_a_manifest_naming_a_split_that_does_not_exist_is_refused(
    config: Config, tmp_path: Path
) -> None:
    """The same typo, one level up: inside the manifest's own assignment."""
    path = write_manifest(
        tmp_path / "splits.json", config, {SPLIT_CE_TRAIN + "x": ["q1"], SPLIT_LTR_TRAIN: ["q2"]}
    )
    with pytest.raises(synth_splits.SplitLeakageError, match="unknown split name"):
        load_verified_manifest(config, path)


def test_a_hand_edited_manifest_is_refused(config: Config, tmp_path: Path) -> None:
    """The edit that would clear an out-of-fold failure is the one to catch.

    Moving a query from the cross-encoder's split to LambdaMART's in the file is
    all it takes to make a leaky dataset look clean, and it leaves the rest of
    the manifest perfectly consistent. Only the checksum notices — which is why
    the trainer now reads the manifest through
    ``concierge.synth.splits.require_disjoint_splits`` rather than its own
    loader, which verified no checksum at all.
    """
    path = write_manifest(
        tmp_path / "splits.json",
        config,
        {SPLIT_CE_TRAIN: ["q1"], SPLIT_LTR_TRAIN: ["q2"]},
    )
    edited = json.loads(path.read_text())
    edited["assignment"]["q1"] = SPLIT_LTR_TRAIN  # ← the hand edit, checksum untouched
    path.write_text(json.dumps(edited))

    with pytest.raises(synth_splits.SplitManifestCorruptError, match="checksum"):
        load_verified_manifest(config, path)


def test_a_missing_manifest_is_an_error_not_a_pass(config: Config, tmp_path: Path) -> None:
    """A safety check that passes because its input is absent is worse than no
    check: it reads as reassurance it has not earned."""
    with pytest.raises(synth_splits.SplitManifestCorruptError, match="no split manifest"):
        load_verified_manifest(config, tmp_path / "nope.json")


def test_a_manifest_built_under_another_seed_is_refused(config: Config, tmp_path: Path) -> None:
    """Same query ids under a different seed are a different split."""
    path = write_manifest(
        tmp_path / "splits.json",
        config,
        {SPLIT_LTR_TRAIN: ["q1"]},
        seed=config.seed + 1,
    )
    with pytest.raises(synth_splits.SplitLeakageError, match="seed"):
        load_verified_manifest(config, path)


def test_a_manifest_built_under_other_fractions_is_refused(config: Config, tmp_path: Path) -> None:
    fractions = synth_splits.fractions_from_config(config)
    moved = dict(fractions)
    # Move weight from one split to another, keeping the sum, so the only thing
    # wrong with this manifest is that it is not the configured split.
    shift = fractions[SPLIT_LTR_VAL] / 2
    moved[SPLIT_LTR_VAL] = fractions[SPLIT_LTR_VAL] - shift
    moved[SPLIT_LTR_TRAIN] = fractions[SPLIT_LTR_TRAIN] + shift
    path = write_manifest(
        tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}, fractions=moved
    )
    with pytest.raises(synth_splits.SplitLeakageError, match="fractions"):
        load_verified_manifest(config, path)


def test_a_manifest_from_another_configuration_is_refused(config: Config, tmp_path: Path) -> None:
    """The fingerprint check, which is this caller's job rather than splits'.

    The default manifest path carries the fingerprint, so a config change
    normally shows up as a missing file. An explicitly passed manifest — the
    ``--splits`` flag, or a file placed by hand — does not, and a split built
    under another configuration is a different split.
    """
    path = write_manifest(
        tmp_path / "splits.json",
        config,
        {SPLIT_LTR_TRAIN: ["q1"]},
        config_fingerprint="0" * 10,
    )
    with pytest.raises(LeakageError, match="fingerprint"):
        load_verified_manifest(config, path)


def test_run_followed_the_manifest_check_catches_a_regenerated_manifest(
    config: Config, tmp_path: Path
) -> None:
    """The ids actually loaded, checked against the manifest's plan.

    Verifying the manifest proves the plan was sound; this proves the rows in
    hand follow it. It delegates to
    ``splits.assert_models_trained_on_disjoint_queries``, which was documented
    as "the one worth calling from concierge.ltr.train" and was called only by
    tests until now.
    """
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(
            tmp_path / "splits.json",
            config,
            {SPLIT_CE_TRAIN: ["q1"], SPLIT_LTR_TRAIN: ["q2"], SPLIT_LTR_VAL: ["q3"]},
        ),
    )

    assert_run_followed_the_manifest([make_example("q2"), make_example("q3")], manifest)

    with pytest.raises(synth_splits.SplitLeakageError):
        assert_run_followed_the_manifest([make_example("q2"), make_example("q1")], manifest)


def test_reads_the_manifest_the_synth_stage_actually_writes(config: Config, tmp_path: Path) -> None:
    """Cross-module seam test: write with the real writer, read with the trainer.

    The hand-built manifests above all go through the same reader, but only this
    one proves the trainer can read the file the synthetic stage produces for a
    real set of queries — the artifact that will actually exist on disk.
    """
    items = [(f"q{i:03d}", f"q{i:03d}") for i in range(40)]
    written = synth_splits.assign_splits(items, config)
    path = synth_splits.save_manifest(written, tmp_path / "splits.json")

    manifest, resolved = load_verified_manifest(config, path)

    assert resolved == path
    assert manifest.checksum == written.checksum
    assert manifest.counts == written.counts
    # And the guarantee holds on it, by construction rather than by assumption.
    synth_splits.assert_out_of_fold(manifest)


def test_crossencoder_run_check_reports_a_pretrained_feature_rather_than_failing(
    config: Config, tmp_path: Path, monkeypatch
) -> None:
    """No fine-tuned checkpoint is not a leak, and the metadata must say which.

    Without a fine-tune for this config the cross-encoder score comes from the
    pretrained model, which saw no synthetic query and so cannot be in fold.
    What would be dishonest is recording that as a passed check.
    """
    from concierge.rerank import model as rerank_model

    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path)
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}),
    )

    result = crossencoder_run_check(config, manifest)

    assert result["checked"] is False
    assert "no fine-tuned run metadata" in result["reason"]


def _write_rerank_run_metadata(directory: Path, payload: dict) -> Path:
    from concierge.rerank.model import RUN_METADATA_FILENAME

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_METADATA_FILENAME
    path.write_text(json.dumps(payload))
    return path


def test_crossencoder_run_check_passes_when_both_models_used_one_split(
    config: Config, tmp_path: Path, monkeypatch
) -> None:
    from concierge.rerank import model as rerank_model

    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path / "rerank")
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}),
    )
    _write_rerank_run_metadata(tmp_path / "rerank", {"split_manifest_checksum": manifest.checksum})

    result = crossencoder_run_check(config, manifest)

    assert result["checked"] is True
    assert result["split_manifest_checksum"] == manifest.checksum


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # A fine-tune that recorded no checksum cannot be tied to any split.
        ({}, "records no split manifest checksum"),
        # The manifest was regenerated between the two trainings: both are
        # internally consistent and the pair is not.
        ({"split_manifest_checksum": "f" * 64}, "was fine-tuned under split manifest"),
    ],
)
def test_crossencoder_run_check_refuses_a_fine_tune_from_another_split(
    config: Config, tmp_path: Path, monkeypatch, payload: dict, expected: str
) -> None:
    from concierge.rerank import model as rerank_model

    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path / "rerank")
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}),
    )
    _write_rerank_run_metadata(tmp_path / "rerank", payload)

    with pytest.raises(LeakageError, match=expected):
        crossencoder_run_check(config, manifest)


def test_crossencoder_run_check_refuses_a_smoke_run_checkpoint(
    config: Config, tmp_path: Path, monkeypatch
) -> None:
    """``--limit-examples`` truncates the training set, so its scores are not the
    feature the ablation table claims to measure."""
    from concierge.rerank import model as rerank_model

    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path / "rerank")
    manifest, _ = load_verified_manifest(
        config,
        write_manifest(tmp_path / "splits.json", config, {SPLIT_LTR_TRAIN: ["q1"]}),
    )
    _write_rerank_run_metadata(
        tmp_path / "rerank",
        {"split_manifest_checksum": manifest.checksum, "is_smoke_run": True},
    )

    with pytest.raises(LeakageError, match="smoke run"):
        crossencoder_run_check(config, manifest)


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


def monotone_dataset(n_queries: int, seed: int, id_prefix: str = "s") -> list[LtrExample]:
    """A ranking problem whose right answer is known by construction.

    The label is a monotone function of ``stars`` alone; every other feature is
    noise drawn independently of the label. A correctly wired LambdaMART must
    therefore (a) rank by stars within each query and (b) put its importance on
    the stars feature. If the group array were wrong, or the labels misaligned
    with the rows, neither would hold — which is exactly what makes this a test
    of the training *plumbing* and not of LightGBM.

    ``id_prefix`` keeps two datasets' query ids apart. Without it a train and a
    validation set built from this function share ids ``s000...``, and a split
    manifest cannot describe that: one id belongs to exactly one split. The
    tests that go through the manifest therefore give each set its own prefix,
    which is also what the real synthetic stage produces.
    """
    rng = random.Random(seed)
    examples: list[LtrExample] = []
    for q in range(n_queries):
        query_id = f"{id_prefix}{q:03d}"
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
    val_examples = [
        replace(e, split=SPLIT_LTR_VAL) for e in monotone_dataset(6, seed=2, id_prefix="v")
    ]
    examples_path = write_examples_jsonl(
        tmp_path / "examples.jsonl", [*train_examples, *val_examples]
    )
    manifest_path = write_manifest(
        tmp_path / "splits.json",
        cfg,
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
    # The split the run was verified against, recorded by checksum so a reader
    # can prove this artifact and the cross-encoder's refer to the same split.
    assert metadata["split_manifest"] == str(manifest_path)
    assert metadata["split_manifest_checksum"] == synth_splits.load_manifest(manifest_path).checksum
    assert metadata["split_seed"] == cfg.seed
    # No fine-tuned cross-encoder exists for this throwaway config, so the
    # feature came from the pretrained model. "Not checked" and "checked and
    # clean" must not look the same in an artifact.
    assert metadata["crossencoder_run_check"]["checked"] is False


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
        cfg,
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


def test_run_refuses_a_stale_examples_file_before_training(tmp_path: Path, config: Config) -> None:
    """The realistic leak, end to end, through ``run()``.

    The examples file says these rows are LambdaMART training data; the manifest
    says their queries fine-tuned the cross-encoder. Training on them would mean
    LambdaMART learning from in-fold cross-encoder scores — the one failure this
    stage is built to prevent — and nothing downstream would show it. The check
    must fire before any model exists, so a contaminated run cannot leave an
    artifact somebody later evaluates.
    """
    cfg = tiny_config(config)
    examples = monotone_dataset(5, seed=1)
    examples_path = write_examples_jsonl(tmp_path / "examples.jsonl", examples)
    ids = sorted({e.query_id for e in examples})
    stale = write_manifest(
        tmp_path / "splits.json",
        cfg,
        # Every row claims ltr_train (monotone_dataset's default); the manifest
        # assigns the first query to the cross-encoder instead.
        {SPLIT_CE_TRAIN: ids[:1], SPLIT_LTR_TRAIN: ids[1:]},
    )
    out_dir = tmp_path / "artifact"

    with pytest.raises(LeakageError, match="disagree with the split manifest"):
        ltr_train.run(cfg, examples_path, stale, out_dir=out_dir, allow_missing_gold=True)
    assert not out_dir.exists(), "no artifact may be written for a leaky run"


def test_run_with_ablation_writes_a_separate_artifact(tmp_path: Path, config: Config) -> None:
    gold = write_gold(tmp_path / "gold.csv", [("g001", "a gold query nobody trains on")])
    cfg = with_gold(tiny_config(config), gold)
    examples = monotone_dataset(10, seed=1)
    examples_path = write_examples_jsonl(tmp_path / "examples.jsonl", examples)
    manifest_path = write_manifest(
        tmp_path / "splits.json",
        cfg,
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
    manifest_path = write_manifest(
        tmp_path / "splits.json",
        cfg,
        {SPLIT_LTR_VAL: sorted({e.query_id for e in val_only})},
    )
    with pytest.raises(ValueError, match="no examples the manifest assigns to"):
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
    assert ltr_train.default_manifest_path(config) == synth_splits.splits_path(config)
    # And it carries the config fingerprint, so a config change cannot be
    # answered with the split built for the previous one.
    assert config.fingerprint() in str(ltr_train.default_manifest_path(config))
