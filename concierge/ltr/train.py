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
   good; served quality is worse; nothing errors. :func:`assert_out_of_fold`
   reads the split manifest and hard-fails on any overlap.

2. **A wrong group array.** LightGBM takes groups as *sizes of consecutive
   runs* — ``[3, 2]`` means rows 0-2 are query A and rows 3-4 are query B. It
   does not take query ids and cannot detect that rows are interleaved. Get it
   wrong and the trainer happily forms pairs across different queries, which is
   a meaningless objective: it trains, it converges, and the model is silently
   wrong. :func:`group_sizes` therefore *requires* contiguity and raises rather
   than guessing.

Both are tested directly in ``tests/test_ltr.py``.
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

# Split names, as they appear in `cfg.synth.splits` and in the manifest the
# synthetic-data stage writes. Spelled as constants because a typo in a split
# name would make a leakage check pass by looking at nothing.
SPLIT_CE_TRAIN = "crossencoder_train"
SPLIT_CE_VAL = "crossencoder_val"
SPLIT_LTR_TRAIN = "ltr_train"
SPLIT_LTR_VAL = "ltr_val"
ALL_SPLITS = (SPLIT_CE_TRAIN, SPLIT_CE_VAL, SPLIT_LTR_TRAIN, SPLIT_LTR_VAL)

DEFAULT_EXAMPLES_PATH = Path("data/processed/ltr_examples.jsonl")
DEFAULT_MANIFEST_PATH = Path("data/processed/synth_splits.json")
DEFAULT_OUT_ROOT = Path("artifacts/ltr")

# NDCG cutoff used for early stopping. Same cutoff as the primary reported
# metric, so the checkpoint is chosen by the thing the project is judged on.
EARLY_STOPPING_NDCG_K = 10


class LeakageError(RuntimeError):
    """Raised when training data overlaps data it must be independent of.

    A distinct exception type so the tests can assert *why* training refused,
    and so a leakage failure can never be caught by a generic ``except
    ValueError`` somewhere up the stack.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Split manifest
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Which synthetic query ids belong to which split.

    Written by the synthetic-data stage (``concierge/synth``), consumed here.
    This module treats it as read-only input and never writes it: the split must
    be decided once, with a fixed seed, by the code that generated the queries —
    a trainer that could re-split its own data could re-split its way out of a
    leakage failure.
    """

    splits: dict[str, frozenset[str]]
    seed: int | None = None
    source_path: Path | None = None

    def ids(self, split: str) -> frozenset[str]:
        return self.splits.get(split, frozenset())

    @property
    def sizes(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in sorted(self.splits.items())}


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Read the manifest, accepting either of its two sane encodings.

    Two shapes are supported because the writer lives in another module and
    either is a reasonable thing for it to emit:

    * ``{"seed": 42, "splits": {"ltr_train": ["q1", ...], ...}}``
    * ``{"q1": "ltr_train", "q2": "crossencoder_train", ...}`` (id -> split)

    Anything else is an error rather than a best-effort guess. A manifest this
    module misreads would make the leakage check vacuous, and a vacuous safety
    check is worse than none at all because it reads as reassurance.
    """
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"split manifest not found: {p}. It is written by the synthetic-data "
            "stage; training cannot verify out-of-fold cross-encoder features "
            "without it."
        )
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"split manifest must be a JSON object, got {type(raw).__name__}")

    seed = raw.get("seed") if isinstance(raw.get("seed"), int) else None
    body = raw.get("splits", raw)
    if not isinstance(body, dict):
        raise ValueError("split manifest 'splits' must be an object")

    splits: dict[str, set[str]] = {}
    if all(isinstance(v, str) for v in body.values()):
        # id -> split form.
        for query_id, split in body.items():
            if query_id == "seed":
                continue
            splits.setdefault(str(split), set()).add(str(query_id))
    else:
        # split -> [ids] form.
        for split, ids in body.items():
            if split == "seed":
                continue
            if not isinstance(ids, list):
                raise ValueError(f"split {split!r} must map to a list of query ids")
            splits[str(split)] = {str(i) for i in ids}

    unknown = set(splits) - set(ALL_SPLITS)
    if unknown:
        # Not fatal: the manifest may legitimately carry extra splits later. But
        # it is logged, because an unrecognised split name is also what a typo
        # looks like, and a typo'd `crossencoder_train` would make the
        # out-of-fold check compare against an empty set.
        log.warning("split manifest has unrecognised split names: %s", sorted(unknown))

    return SplitManifest(
        splits={name: frozenset(ids) for name, ids in splits.items()},
        seed=seed,
        source_path=p,
    )


def assert_out_of_fold(manifest: SplitManifest, *, enforce: bool = True) -> None:
    """Fail unless every LambdaMART query is unseen by the cross-encoder.

    Checked against **both** cross-encoder splits, not just its training split:

    * ``crossencoder_train`` — the model fitted on these. Overlap means the
      cross-encoder feature is partly memorised, which is the inflation this
      whole mechanism exists to prevent.
    * ``crossencoder_val`` — the model never fitted on these, but early stopping
      *selected a checkpoint* using them, so its scores on them are optimistic
      too. Smaller effect, same direction. Including it is the strict reading,
      and strict is the right default for a check whose failure mode is a
      silently better-looking number.

    ``ltr_val`` is checked as well as ``ltr_train``: the validation split chooses
    the boosting iteration, so contamination there also biases the model that
    gets shipped.

    With ``enforce=False`` (``cfg.ltr.enforce_oof_crossencoder_feature: false``)
    the overlap is logged loudly instead of raised. That escape hatch exists for
    a deliberate experiment — *measuring* how much the inflation is worth — not
    for convenience, and the metadata records which mode produced the model.
    """
    ce_seen = manifest.ids(SPLIT_CE_TRAIN) | manifest.ids(SPLIT_CE_VAL)
    problems: dict[str, list[str]] = {}
    for ltr_split in (SPLIT_LTR_TRAIN, SPLIT_LTR_VAL):
        overlap = sorted(manifest.ids(ltr_split) & ce_seen)
        if overlap:
            problems[ltr_split] = overlap

    if not problems:
        return

    detail = "; ".join(
        f"{split}: {len(ids)} overlapping query ids (e.g. {ids[:3]})"
        for split, ids in problems.items()
    )
    message = (
        "cross-encoder feature would not be out-of-fold — "
        f"{detail}. The cross-encoder score for these queries is inflated, so "
        "LambdaMART would learn to over-trust it and served quality would "
        "degrade without any offline signal. Re-run the synthetic split."
    )
    if enforce:
        raise LeakageError(message)
    log.error("OUT-OF-FOLD CHECK DISABLED: %s", message)


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
    # Which synthetic split this row belongs to. Carried on the row as well as
    # in the manifest so a filtered load cannot disagree with the manifest
    # silently: both are reported in the metadata and can be cross-checked.
    split: str = ""


def load_examples(path: str | Path, split: str | None = None) -> list[LtrExample]:
    """Read JSONL training examples, optionally filtered to one split.

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
        raise FileNotFoundError(
            f"LambdaMART training examples not found: {p}. They are produced by "
            "the synthetic-data stage over the ltr_train split."
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
    if split is not None:
        examples = [e for e in examples if e.split == split]
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
    manifest: SplitManifest | None = None,
    ablated_non_text: bool = False,
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
        "split_manifest": None if manifest is None else str(manifest.source_path),
        "split_sizes": None if manifest is None else manifest.sizes,
        "split_seed": None if manifest is None else manifest.seed,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def default_out_dir(cfg: Config, ablated_non_text: bool) -> Path:
    """``artifacts/ltr/<config fingerprint>[-textonly]``.

    Keyed by the config hash (D008) so a model can always be traced to the
    settings that produced it, and suffixed for the ablation so the two runs
    cannot overwrite each other — which would make the ablation table
    unreproducible in the most confusing possible way.
    """
    suffix = "-textonly" if ablated_non_text else ""
    return DEFAULT_OUT_ROOT / f"{cfg.fingerprint()}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def run(
    cfg: Config,
    examples_path: str | Path = DEFAULT_EXAMPLES_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    ablate_non_text: bool = False,
    out_dir: str | Path | None = None,
    allow_missing_gold: bool = False,
) -> Path:
    """Full training run: checks, data, fit, save. Returns the artifact dir.

    Order matters. Both leakage checks run **before** any training, so a
    contaminated run costs seconds rather than producing a model somebody might
    later evaluate by accident.
    """
    manifest = load_split_manifest(manifest_path)
    assert_out_of_fold(manifest, enforce=cfg.ltr.enforce_oof_crossencoder_feature)

    train_examples = load_examples(examples_path, split=SPLIT_LTR_TRAIN)
    val_examples = load_examples(examples_path, split=SPLIT_LTR_VAL)
    if not train_examples:
        raise ValueError(
            f"no examples with split={SPLIT_LTR_TRAIN!r} in {examples_path}. "
            "LambdaMART trains on the ltr_train split only."
        )

    assert_no_gold_leakage(
        [*train_examples, *val_examples],
        cfg.evaluation.gold_queries,
        allow_missing_gold=allow_missing_gold,
    )

    names = ablated_feature_names(ablate_non_text)
    train_data = build_ranking_data(train_examples, names)
    val_data = build_ranking_data(val_examples, names) if val_examples else None
    if val_data is None:
        log.warning(
            "no %s examples: training without early stopping, for the full %d rounds",
            SPLIT_LTR_VAL,
            cfg.ltr.n_estimators,
        )

    ranker = train_lambdamart(train_data, val_data, cfg)
    target = default_out_dir(cfg, ablate_non_text) if out_dir is None else out_dir
    directory = save_ranker(
        ranker,
        target,
        cfg,
        manifest=manifest,
        ablated_non_text=ablate_non_text,
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
        default=str(DEFAULT_EXAMPLES_PATH),
        help="JSONL of labeled (query, candidate) pairs with stage scores.",
    )
    parser.add_argument(
        "--splits",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Split manifest written by the synthetic-data stage.",
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
        args.examples,
        args.splits,
        ablate_non_text=args.ablate_non_text,
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
