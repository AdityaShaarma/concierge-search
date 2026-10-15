"""Stage 4 features: turning one (query, candidate, business) triple into numbers.

Why this file is a pure function, and why that matters more than it looks
------------------------------------------------------------------------
The classic way a production ranking system goes quietly wrong is **training /
serving feature skew**: the offline training job computes a feature one way
(say, from a precomputed table, with 0 standing in for a missing score) and the
online service computes it another way (live, with NaN standing in for the same
thing). Nothing crashes. Offline metrics look fine. Served quality is worse than
the numbers claim, and the gap is invisible because the two code paths are never
compared.

The structural fix used here: there is exactly **one** function that produces
features, :func:`build_feature_row`, and it is a pure function of the three
contract objects (``ParsedQuery``, ``Candidate``, ``Business``). The training
job and the serving path both call it. There is no second implementation to
drift from, no hidden state, no config-dependent branch inside it. If a feature
is wrong, it is wrong identically in both places — which is a bug you can find,
rather than a discrepancy you cannot see.

Why missing values are NaN and not -1 (or 0)
-------------------------------------------
LightGBM handles NaN natively: during training it learns, per split, which side
of the threshold missing values should go to, and it ignores them when choosing
the threshold itself. NaN therefore means "this signal does not exist for this
row".

A sentinel like ``-1`` or ``0`` means something completely different, because a
decision tree only ever asks ``feature <= t``. A sentinel is a real, ordered
value sitting at one end of the axis, so:

* ``0.0`` for a missing dense score places that candidate among the genuinely
  *weakly similar* documents, which is a factual claim the data never made —
  the dense retriever simply never returned this candidate.
* ``-1`` for a missing star rating places an unrated business below the worst
  rated one, inventing a ranking between "bad" and "unknown".
* Worse, sentinels interact: a split at ``stars <= 0.5`` lumps missing together
  with 0-star businesses forever, and no amount of extra trees can separate them
  again because the information was destroyed before the model saw it.

So: NaN for genuinely-absent signals, real numbers for real observations, and
never a magic number in between.

Why only these features
-----------------------
The list is CLAUDE.md's starting set (stage scores, quality, popularity,
constraint counts, category match) plus distance, which DECISIONS.md D003 put in
scope. Features get added **with evidence** — an importance plot plus an
ablation — not because they sounded plausible. Keeping the set small also keeps
the importance chart readable, which is the point of having one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from concierge.retrieval.filters import CONSTRAINT_CATEGORY, CONSTRAINT_GEO, haversine_km
from concierge.types import (
    RANK_RRF,
    SCORE_BM25,
    SCORE_CROSS_ENCODER,
    SCORE_DENSE,
    Business,
    Candidate,
    ConstraintStatus,
    ParsedQuery,
)

# ─────────────────────────────────────────────────────────────────────────────
# The feature schema
# ─────────────────────────────────────────────────────────────────────────────

# Feature names are part of the model artifact, not a debugging nicety. A
# LightGBM importance plot labelled f0/f1/f2 is unusable in an interview — "the
# third column matters most" is not an answer. Saving names alongside the model
# also means a future serving process can assert the column order it builds
# matches the order the model was trained on, instead of silently feeding
# `stars` into the slot where `bm25_score` belongs.

F_BM25 = "bm25_score"
F_DENSE = "dense_score"
F_RRF_RANK = "rrf_rank"
F_CROSS_ENCODER = "cross_encoder_score"
F_STARS = "stars"
F_LOG_REVIEWS = "log1p_review_count"
F_N_SATISFIED = "n_constraints_satisfied"
F_FRAC_SATISFIED = "frac_constraints_satisfied"
F_HAS_UNKNOWN = "has_unknown_constraint_data"
F_CATEGORY_MATCH = "category_match"
F_DISTANCE_KM = "distance_km"

#: Canonical column order. Everything downstream (matrix building, the model
#: artifact, the serving path) derives its order from this tuple alone.
FEATURE_NAMES: tuple[str, ...] = (
    F_BM25,
    F_DENSE,
    F_RRF_RANK,
    F_CROSS_ENCODER,
    F_STARS,
    F_LOG_REVIEWS,
    F_N_SATISFIED,
    F_FRAC_SATISFIED,
    F_HAS_UNKNOWN,
    F_CATEGORY_MATCH,
    F_DISTANCE_KM,
)

#: Features that are a function of *text similarity* — the outputs of stages
#: 2 and 3. These are what a pure "semantic" system already has.
TEXT_FEATURE_NAMES: tuple[str, ...] = (F_BM25, F_DENSE, F_RRF_RANK, F_CROSS_ENCODER)

#: Everything else: quality, popularity, hard-constraint satisfaction, geography.
#: The "do ratings and popularity actually matter?" ablation drops exactly this
#: set, so the question is answered by measurement instead of assertion.
NON_TEXT_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if name not in TEXT_FEATURE_NAMES
)

# Constraint names that are *not* generic hard constraints for counting
# purposes. Geo is excluded from the satisfied-count features because its
# magnitude is already carried by `distance_km`; counting it as well would feed
# the same signal twice, once thresholded at the query's radius and once raw.
_CONSTRAINTS_EXCLUDED_FROM_COUNTS = frozenset({CONSTRAINT_GEO})

NAN = float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# One row
# ─────────────────────────────────────────────────────────────────────────────


def build_feature_row(
    parsed: ParsedQuery,
    candidate: Candidate,
    business: Business,
) -> dict[str, float]:
    """Features for one (query, candidate, business) triple, keyed by name.

    Pure: same inputs, same output, no I/O, no config, no globals. That is what
    makes the training job and the serving path provably identical (see the
    module docstring on feature skew).

    Returns a dict rather than an array so that a wrong value is debuggable by
    name; :func:`feature_vector` imposes the column order.
    """
    row: dict[str, float] = {
        # ── Stage 2 and 3 signals. Absent means "this stage never scored this
        # candidate" (e.g. it was retrieved by BM25 only, or it was cut before
        # reranking), which is not the same as a low score — hence NaN.
        F_BM25: _score(candidate, SCORE_BM25),
        F_DENSE: _score(candidate, SCORE_DENSE),
        F_RRF_RANK: _rank(candidate, RANK_RRF),
        F_CROSS_ENCODER: _score(candidate, SCORE_CROSS_ENCODER),
        # ── Quality and popularity. Always present in the dataset, so a real
        # number always exists here; no missing-value policy needed.
        F_STARS: float(business.stars),
        # log1p, not log: review_count is 0 for a brand-new business and log(0)
        # is -inf. log1p(0) = 0 and keeps the transform monotone over the whole
        # range. The transform itself is for interpretability of the importance
        # plot and partial-dependence shape, not for the trees — a tree is
        # invariant to any monotone transform of a feature.
        F_LOG_REVIEWS: math.log1p(max(0, business.review_count)),
    }
    row.update(_constraint_features(candidate))
    row[F_CATEGORY_MATCH] = _category_match(parsed, candidate)
    row[F_DISTANCE_KM] = _distance_km(parsed, business)
    return row


def feature_vector(
    parsed: ParsedQuery,
    candidate: Candidate,
    business: Business,
    names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """One row as a float64 array in ``names`` order."""
    row = build_feature_row(parsed, candidate, business)
    return np.asarray([row[name] for name in names], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Individual features
# ─────────────────────────────────────────────────────────────────────────────


def _score(candidate: Candidate, key: str) -> float:
    value = candidate.scores.get(key)
    return NAN if value is None else float(value)


def _rank(candidate: Candidate, key: str) -> float:
    value = candidate.ranks.get(key)
    return NAN if value is None else float(value)


def _constraint_features(candidate: Candidate) -> dict[str, float]:
    """Counts over the constraint checks the system already performed.

    Deliberately *reads* ``candidate.constraints`` instead of re-deriving the
    outcomes. The constraint checker (``concierge.retrieval.filters``) is the
    single authority on whether a business satisfies "open past 10pm"; a second
    implementation here could disagree with it, and then the ranker's features
    would describe a system different from the one that produced the results and
    the "why this result" explanation.

    The three-valued status (D006) is why there are three features and not one:

    * ``n_constraints_satisfied`` — how much of the request this candidate
      provably meets, in absolute terms.
    * ``frac_constraints_satisfied`` — the same, normalized by how constrained
      the query is. 2-of-2 and 2-of-6 are very different answers, and the count
      alone cannot tell them apart across queries. LambdaMART is trained with
      queries as groups, so it compares rows *within* a query — but the trees
      are shared across queries, and a threshold on a raw count means different
      things for a 2-constraint and a 6-constraint query.
    * ``has_unknown_constraint_data`` — whether any check came back UNKNOWN.
      This is the feature that lets the model learn how much to discount
      "we don't know" versus "we checked and it passes", which is a decision
      D007 currently makes with a hand-set penalty of 0.5. Giving the model the
      indicator means it can learn that discount from labels instead.
    """
    counted = [
        match
        for match in candidate.constraints
        if match.name not in _CONSTRAINTS_EXCLUDED_FROM_COUNTS
    ]
    n_total = len(counted)
    n_satisfied = sum(1 for m in counted if m.status is ConstraintStatus.SATISFIED)
    n_unknown = sum(1 for m in counted if m.status is ConstraintStatus.UNKNOWN)

    return {
        # A count is well defined even at zero: an unconstrained query really
        # does have zero satisfied constraints, for every candidate alike. It
        # carries no information in that case, which is fine — it is constant
        # within the group, so it cannot affect the within-query ordering.
        F_N_SATISFIED: float(n_satisfied),
        # A fraction is *not* defined at zero: 0/0. Writing 0.0 would claim
        # "satisfied none of the request", which is false for a query that asked
        # for nothing, and would make unconstrained queries look maximally
        # unsatisfied next to constrained ones.
        F_FRAC_SATISFIED: NAN if n_total == 0 else n_satisfied / n_total,
        # Defined at zero (nothing was checked, so nothing was missing), unlike
        # the fraction above. Stated explicitly because the asymmetry between
        # these two features is deliberate, not an oversight.
        F_HAS_UNKNOWN: 1.0 if n_unknown > 0 else 0.0,
    }


def _category_match(parsed: ParsedQuery, candidate: Candidate) -> float:
    """1.0 satisfied, 0.0 violated, NaN when undefined.

    Undefined covers two genuinely different situations that both mean "no
    signal": the query named no category at all, and the business has no
    category data (UNKNOWN). Encoding either as 0.0 would tell the model
    "this candidate is the wrong kind of place", which the data never said.
    """
    for match in candidate.constraints:
        if match.name != CONSTRAINT_CATEGORY:
            continue
        if match.status is ConstraintStatus.SATISFIED:
            return 1.0
        if match.status is ConstraintStatus.VIOLATED:
            return 0.0
        return NAN
    # No category constraint was checked. Guard against the one inconsistent
    # case: the query *did* request categories but the constraint list does not
    # contain the check. That means the caller passed a candidate from before
    # filtering, and quietly returning NaN would hide it.
    if parsed.categories:
        raise ValueError(
            "query requested categories but the candidate carries no category "
            f"constraint (business_id={candidate.business_id!r}). Build features "
            "from post-filter candidates so train and serve see the same checks."
        )
    return NAN


def _distance_km(parsed: ParsedQuery, business: Business) -> float:
    """Great-circle km from the query's geo anchor, or NaN.

    Recomputed from coordinates rather than read off the geo ``ConstraintMatch``
    (which stores a rounded distance for display) so the feature is exactly a
    function of the inputs. Haversine is imported from the filter module so
    there is one distance implementation in the project, not two that could
    disagree at the radius boundary.

    NaN when the query has no geo constraint (most queries) or when the business
    has no coordinates. Zero would mean "at the anchor point", i.e. the best
    possible value — the worst possible encoding for missing data here.
    """
    geo = parsed.geo
    if geo is None or business.latitude is None or business.longitude is None:
        return NAN
    return haversine_km(business.latitude, business.longitude, geo.latitude, geo.longitude)


# ─────────────────────────────────────────────────────────────────────────────
# A matrix of rows, with its names attached
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A 2-D float matrix that cannot be separated from its column names.

    The names travel *with* the values because the two going out of sync is the
    failure this type exists to prevent: a model trained on columns in one order
    and served columns in another produces plausible-looking nonsense and no
    error. Carrying them together means the mismatch is caught at construction.
    """

    names: tuple[str, ...]
    values: np.ndarray  # shape (n_rows, n_features), float64

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {self.values.shape}")
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"column count {self.values.shape[1]} does not match "
                f"{len(self.names)} feature names"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate feature names: {self.names}")

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        """One named column, for assertions and for debugging a single feature."""
        try:
            index = self.names.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"no such feature: {name!r}") from exc
        return self.values[:, index]

    def select(self, names: Sequence[str]) -> FeatureMatrix:
        """A new matrix with only ``names``, in the order given.

        Used by the ablation. Columns are *dropped*, not NaN-filled: a dropped
        column is provably unusable by the model, whereas an all-NaN column
        leaves the feature in the model's schema and makes the ablated model's
        importance table misleading.
        """
        indices = [self.names.index(name) for name in names]
        return FeatureMatrix(tuple(names), self.values[:, indices])


def feature_matrix(
    triples: Iterable[tuple[ParsedQuery, Candidate, Business]],
    names: Sequence[str] = FEATURE_NAMES,
) -> FeatureMatrix:
    """Build a named matrix, one row per triple, in ``names`` order."""
    names = tuple(names)
    rows = [
        [build_feature_row(parsed, candidate, business)[name] for name in names]
        for parsed, candidate, business in triples
    ]
    values = (
        np.asarray(rows, dtype=np.float64) if rows else np.empty((0, len(names)), dtype=np.float64)
    )
    return FeatureMatrix(names, values)


def ablated_feature_names(drop_non_text: bool) -> tuple[str, ...]:
    """Feature set for a training run, with or without the non-text block.

    One function so the ablation cannot accidentally be run with a different
    feature set than the one it claims to be testing.
    """
    if not drop_non_text:
        return FEATURE_NAMES
    return TEXT_FEATURE_NAMES
