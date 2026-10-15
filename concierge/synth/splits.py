"""Train/val splits for the two models we train, and the proof that they are disjoint.

Why this file exists at all, and why it is more than four slices of a shuffled
list: **LambdaMART's most powerful feature is the cross-encoder's score.** If
LambdaMART is trained on queries the cross-encoder was fine-tuned on, that feature
is inflated exactly where the labels are — the cross-encoder has memorised those
queries — so LambdaMART learns "trust the cross-encoder completely", and at
serving time, on queries nobody has seen, that lesson is wrong. The model would
look excellent in training and mediocre on the gold set, with no single line of
code to blame.

``cfg.ltr.enforce_oof_crossencoder_feature`` says this must be enforced in code.
``verify_splits`` and ``assert_out_of_fold`` are that enforcement: the trainers
call them and fail loudly, which is what turns the out-of-fold guarantee from a
comment into a property of the system.

Three design choices worth defending in an interview:

1. **Hash-based assignment, not a seeded shuffle.** Each query's split is
   ``hash(seed, group_key)`` bucketed by the configured fractions, so a query's
   split depends only on its id — not on how many other queries exist. Generating
   more synthetic data later therefore cannot move an existing query from the
   cross-encoder's training split into LambdaMART's, which is exactly the leakage
   a reshuffle would introduce and nothing downstream would notice. The cost is
   that the split sizes are only approximately the configured fractions (binomial
   scatter around them), so the manifest records the realised counts rather than
   the nominal ones.
2. **A manifest on disk with a checksum.** The split has to be the same next week,
   on Colab, after a config edit. The checksum means a corrupted or hand-edited
   manifest is an error rather than a quietly different experiment.
3. **The grouping unit is configurable, and ``query`` is only the default.** See
   ``GroupBy`` below: splitting by query is what the project specifies, but
   splitting by *source business* is strictly stronger, and the residual risk of
   the default is stated rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from concierge.config import REPO_ROOT, Config

log = logging.getLogger(__name__)

#: Split names, in the fixed order the fraction bands are laid out in.
SPLIT_NAMES: tuple[str, ...] = (
    "crossencoder_train",
    "crossencoder_val",
    "ltr_train",
    "ltr_val",
)

#: Which splits belong to which model. The out-of-fold check is defined over these.
CROSSENCODER_SPLITS: tuple[str, ...] = ("crossencoder_train", "crossencoder_val")
LTR_SPLITS: tuple[str, ...] = ("ltr_train", "ltr_val")

MANIFEST_VERSION = 1

GroupBy = Literal["query", "business"]
"""What may not span splits.

``"query"`` — the project's specification, and the default. A query lands in
exactly one split.

``"business"`` — stronger. All four queries generated from one business document
land in the same split. Worth knowing why: with ``"query"`` grouping, the
cross-encoder can be fine-tuned on "quiet spot to read" written from business X
and then be asked, during LambdaMART training, to score business X for a different
query. It has seen X's document and its score for X is mildly optimistic. That is
a *document*-level leak rather than a query-level one, and it is smaller than the
query-level leak this module prevents — but it is real, it is unmeasured, and the
option to eliminate it is one argument away.
"""


class _HasQueryId(Protocol):
    """Anything carrying a query id — a ``SyntheticQuery`` or a ``TrainingExample``."""

    @property
    def query_id(self) -> str: ...


class _HasQueryAndBusiness(_HasQueryId, Protocol):
    """Duck type for ``SyntheticQuery`` without importing it.

    Structural typing keeps this module free of the generation machinery, so the
    trainers can verify a manifest without importing an LLM client.
    """

    @property
    def business_id(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class SplitError(RuntimeError):
    """Base class for split failures."""


class SplitManifestCorruptError(SplitError):
    """The manifest on disk does not match its own checksum, or is unreadable."""


class SplitLeakageError(SplitError):
    """Two splits share a query (or the manifest disagrees with the config).

    Raised by the verification the trainers call. If this is ever raised, nothing
    downstream of it may be reported: the numbers would describe a model evaluated
    partly on its own training data.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Assignment
# ─────────────────────────────────────────────────────────────────────────────


def bucket(group_key: str, seed: int) -> float:
    """Deterministic value in [0, 1) for a group, from its id and the seed.

    SHA-256 rather than Python's ``hash()``: ``hash()`` on a string is salted per
    process (PYTHONHASHSEED), so the same query would land in different splits in
    different runs. That is the single most dangerous bug this module could have —
    it would leak quietly, differently every time, and tests passing once would
    prove nothing.

    The first 16 hex digits give 64 bits of resolution, which is far finer than
    the split boundaries need.
    """
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).hexdigest()[:16]
    return int(digest, 16) / float(1 << 64)


def fractions_from_config(cfg: Config) -> dict[str, float]:
    """The configured split fractions, in ``SPLIT_NAMES`` order.

    Read off the pydantic model rather than a dict so a renamed field is an
    AttributeError at startup instead of a silently missing split.
    """
    splits = cfg.synth.splits
    return {
        "crossencoder_train": splits.crossencoder_train,
        "crossencoder_val": splits.crossencoder_val,
        "ltr_train": splits.ltr_train,
        "ltr_val": splits.ltr_val,
    }


def split_for_bucket(value: float, fractions: Mapping[str, float]) -> str:
    """Which split a bucket value falls into, walking the bands in fixed order."""
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += fractions[name]
        if value < cumulative:
            return name
    # Floating-point slack at the very top of the range (fractions sum to 1.0 only
    # to within 1e-6, which pydantic already enforces). The last band absorbs it.
    return SPLIT_NAMES[-1]


def items_from_queries(
    queries: Iterable[_HasQueryAndBusiness],
    group_by: GroupBy = "query",
) -> list[tuple[str, str]]:
    """``[(query_id, group_key)]`` from synthetic queries, per the grouping choice."""
    if group_by == "query":
        return [(q.query_id, q.query_id) for q in queries]
    if group_by == "business":
        return [(q.query_id, q.business_id) for q in queries]
    raise SplitError(f"unknown group_by {group_by!r}; use 'query' or 'business'")


# ─────────────────────────────────────────────────────────────────────────────
# The manifest
# ─────────────────────────────────────────────────────────────────────────────


def compute_checksum(
    assignment: Mapping[str, str],
    seed: int,
    fractions: Mapping[str, float],
    group_by: str,
) -> str:
    """Checksum over everything that defines the split.

    The seed, fractions and grouping are hashed in alongside the assignment so
    that a manifest produced under a different config cannot pass verification by
    accident: the same query ids under a different seed are a *different split*,
    even if the file happens to look similar.
    """
    payload = json.dumps(
        {
            "version": MANIFEST_VERSION,
            "seed": seed,
            "group_by": group_by,
            "fractions": {name: round(fractions[name], 12) for name in SPLIT_NAMES},
            "assignment": sorted(assignment.items()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """The split, saved to disk, with everything needed to re-derive and check it."""

    seed: int
    group_by: str
    fractions: Mapping[str, float]
    assignment: Mapping[str, str]  # query_id -> split name
    checksum: str
    n_groups: int
    config_fingerprint: str = ""
    version: int = MANIFEST_VERSION

    # ── reads ────────────────────────────────────────────────────────────────

    @property
    def counts(self) -> dict[str, int]:
        """Realised counts per split (not the nominal fractions — see the module
        docstring on hash-based assignment)."""
        out = {name: 0 for name in SPLIT_NAMES}
        for split in self.assignment.values():
            out[split] = out.get(split, 0) + 1
        return out

    @property
    def n_queries(self) -> int:
        return len(self.assignment)

    def ids(self, *split_names: str) -> tuple[str, ...]:
        """Query ids in the named split(s), sorted for reproducible iteration."""
        for name in split_names:
            if name not in SPLIT_NAMES:
                raise SplitError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        wanted = set(split_names)
        return tuple(sorted(qid for qid, s in self.assignment.items() if s in wanted))

    def realised_fractions(self) -> dict[str, float]:
        total = self.n_queries
        return {name: (count / total if total else 0.0) for name, count in self.counts.items()}

    def summary(self) -> str:
        realised = {k: round(v, 4) for k, v in self.realised_fractions().items()}
        return (
            f"{self.n_queries} queries in {self.n_groups} group(s) by {self.group_by}, "
            f"seed {self.seed}; counts {self.counts}; realised fractions {realised}; "
            f"checksum {self.checksum[:12]}"
        )

    # ── serialization ────────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "group_by": self.group_by,
            "fractions": dict(self.fractions),
            "config_fingerprint": self.config_fingerprint,
            "n_groups": self.n_groups,
            "counts": self.counts,
            "checksum": self.checksum,
            # Sorted so the file is diffable and two equal splits are byte-equal.
            "assignment": dict(sorted(self.assignment.items())),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SplitManifest:
        try:
            assignment = {str(k): str(v) for k, v in dict(data["assignment"]).items()}
            fractions = {str(k): float(v) for k, v in dict(data["fractions"]).items()}
            manifest = cls(
                seed=int(data["seed"]),
                group_by=str(data["group_by"]),
                fractions=fractions,
                assignment=assignment,
                checksum=str(data["checksum"]),
                n_groups=int(data["n_groups"]),
                config_fingerprint=str(data.get("config_fingerprint", "")),
                version=int(data.get("version", MANIFEST_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitManifestCorruptError(f"unreadable split manifest: {exc}") from exc
        expected = compute_checksum(
            manifest.assignment, manifest.seed, manifest.fractions, manifest.group_by
        )
        if expected != manifest.checksum:
            raise SplitManifestCorruptError(
                "split manifest checksum mismatch: the file was edited or truncated "
                f"(stored {manifest.checksum[:12]}, recomputed {expected[:12]}). "
                "Regenerate it rather than trusting it."
            )
        return manifest


def assign_splits(
    items: Sequence[tuple[str, str]],
    cfg: Config,
    *,
    group_by: GroupBy = "query",
) -> SplitManifest:
    """Assign ``(query_id, group_key)`` pairs to the four splits.

    A group is assigned once and every query in it inherits that split, which is
    what the word "group" buys: with ``group_by="business"`` all of a business's
    queries move together.

    A query id appearing twice with *different* group keys is an error, not a
    last-one-wins: it would mean the same query belongs to two groups, and the
    disjointness guarantee would be meaningless.
    """
    fractions = fractions_from_config(cfg)
    group_of: dict[str, str] = {}
    for query_id, group_key in items:
        existing = group_of.get(query_id)
        if existing is not None and existing != group_key:
            raise SplitError(
                f"query {query_id!r} was given two group keys ({existing!r}, {group_key!r})"
            )
        group_of[query_id] = group_key

    split_of_group = {
        key: split_for_bucket(bucket(key, cfg.seed), fractions)
        for key in sorted(set(group_of.values()))
    }
    assignment = {qid: split_of_group[key] for qid, key in sorted(group_of.items())}

    manifest = SplitManifest(
        seed=cfg.seed,
        group_by=group_by,
        fractions=fractions,
        assignment=assignment,
        checksum=compute_checksum(assignment, cfg.seed, fractions, group_by),
        n_groups=len(split_of_group),
        config_fingerprint=cfg.fingerprint(),
    )
    log.info("split: %s", manifest.summary())
    return manifest


def assign_splits_for_queries(
    queries: Sequence[_HasQueryAndBusiness],
    cfg: Config,
    *,
    group_by: GroupBy = "query",
) -> SplitManifest:
    """``assign_splits`` straight from ``SyntheticQuery`` objects."""
    return assign_splits(items_from_queries(queries, group_by), cfg, group_by=group_by)


# ─────────────────────────────────────────────────────────────────────────────
# Disk
# ─────────────────────────────────────────────────────────────────────────────


def splits_path(cfg: Config, root: Path | None = None) -> Path:
    """Where the manifest lives: ``artifacts/synth/<config fingerprint>/splits.json``.

    Keyed by the config fingerprint so a config change cannot silently reuse an old
    split. It sits under ``artifacts/`` (which ``make clean`` wipes) because it is
    *derivable*: the same query ids and seed regenerate it byte for byte, and the
    checksum proves it.
    """
    base = REPO_ROOT if root is None else Path(root)
    return base / "artifacts" / "synth" / cfg.fingerprint() / "splits.json"


def save_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    """Write the manifest as pretty JSON (it is meant to be read by a human too)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n", "utf-8")
    log.info("wrote split manifest to %s", p)
    return p


def load_manifest(path: str | Path) -> SplitManifest:
    """Read and checksum-verify a manifest."""
    p = Path(path)
    if not p.exists():
        raise SplitManifestCorruptError(
            f"no split manifest at {p}. Build it with concierge.synth.splits before "
            "training, so both models can be checked against the same split."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitManifestCorruptError(f"{p} is not valid JSON: {exc}") from exc
    return SplitManifest.from_json(data)


def build_and_save(
    queries: Sequence[_HasQueryAndBusiness],
    cfg: Config,
    *,
    group_by: GroupBy = "query",
    root: Path | None = None,
) -> tuple[SplitManifest, Path]:
    """Assign, verify and persist in one call — the normal entry point."""
    manifest = assign_splits_for_queries(queries, cfg, group_by=group_by)
    verify_splits(manifest, cfg=cfg)
    return manifest, save_manifest(manifest, splits_path(cfg, root))


# ─────────────────────────────────────────────────────────────────────────────
# Verification — what the trainers call
# ─────────────────────────────────────────────────────────────────────────────


def assert_no_overlap(crossencoder_ids: Iterable[str], ltr_ids: Iterable[str]) -> None:
    """The out-of-fold check itself, over two collections of query ids.

    Separated from ``assert_out_of_fold`` so the *failure* path is testable. A
    single manifest cannot represent one query in two splits (the assignment is a
    dict), but the leak this guards against does not arrive that way in practice:
    it arrives as a cross-encoder trained from one manifest and a LambdaMART
    trained from another. Taking two id collections is therefore the shape that
    matches the real risk, and the only shape a test can break on purpose.
    """
    overlap = set(crossencoder_ids) & set(ltr_ids)
    if overlap:
        sample = sorted(overlap)[:5]
        raise SplitLeakageError(
            f"{len(overlap)} query/queries are in both a cross-encoder split and a "
            f"LambdaMART split (e.g. {sample}). The cross-encoder score would be an "
            "inflated feature for LambdaMART; refusing to train."
        )


def assert_out_of_fold(manifest: SplitManifest) -> None:
    """Assert no query is in both a cross-encoder split and a LambdaMART split.

    This is the function that makes the out-of-fold cross-encoder feature real.
    ``cfg.ltr.enforce_oof_crossencoder_feature`` is a promise; this is the check.
    """
    assert_no_overlap(manifest.ids(*CROSSENCODER_SPLITS), manifest.ids(*LTR_SPLITS))


def assert_models_trained_on_disjoint_queries(
    crossencoder_query_ids: Iterable[str],
    ltr_query_ids: Iterable[str],
    manifest: SplitManifest,
) -> None:
    """Check the ids a trainer *actually used* against the manifest.

    The stronger version of the guarantee, and the one worth calling from
    ``concierge.ltr.train``: verifying the manifest proves the plan was sound,
    while this proves the run followed it. A model trained before a regenerated
    manifest is exactly how an out-of-fold promise quietly stops being true.
    """
    ce = set(crossencoder_query_ids)
    ltr = set(ltr_query_ids)
    assert_no_overlap(ce, ltr)
    expected_ce = set(manifest.ids(*CROSSENCODER_SPLITS))
    expected_ltr = set(manifest.ids(*LTR_SPLITS))
    stray_ce = ce - expected_ce
    stray_ltr = ltr - expected_ltr
    if stray_ce or stray_ltr:
        raise SplitLeakageError(
            f"{len(stray_ce)} cross-encoder and {len(stray_ltr)} LambdaMART training "
            "query/queries are not in the splits this manifest assigns them to; the "
            "manifest was regenerated after training. Retrain or regenerate."
        )


def verify_splits(manifest: SplitManifest, *, cfg: Config | None = None) -> SplitManifest:
    """Every invariant a trainer needs, checked in one place.

    Checks, in order of how badly each one would mislead a reported number:

    1. the four splits are disjoint and cover every query exactly once;
    2. cross-encoder splits and LambdaMART splits share no query (out-of-fold);
    3. the checksum matches the assignment (the manifest was not edited);
    4. if a config is given, its seed and fractions match the manifest's — so a
       config change cannot be answered with a stale split.

    Returns the manifest so it can be used inline:
    ``ids = verify_splits(load_manifest(p), cfg=cfg).ids("crossencoder_train")``.
    """
    unknown = {s for s in manifest.assignment.values() if s not in SPLIT_NAMES}
    if unknown:
        raise SplitLeakageError(f"manifest contains unknown split name(s): {sorted(unknown)}")

    # Disjointness and coverage. A dict assignment makes a query-in-two-splits
    # literally unrepresentable, which is the point of this data shape — so what is
    # actually verified here is that the sets derived FROM it still partition, which
    # is what every caller downstream relies on.
    buckets = {name: set(manifest.ids(name)) for name in SPLIT_NAMES}
    total = sum(len(v) for v in buckets.values())
    if total != manifest.n_queries:
        raise SplitLeakageError(
            f"splits hold {total} ids but the manifest has {manifest.n_queries} queries"
        )
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1 :]:
            shared = buckets[a] & buckets[b]
            if shared:
                raise SplitLeakageError(
                    f"splits {a} and {b} share {len(shared)} query/queries "
                    f"(e.g. {sorted(shared)[:5]})"
                )

    assert_out_of_fold(manifest)

    expected = compute_checksum(
        manifest.assignment, manifest.seed, manifest.fractions, manifest.group_by
    )
    if expected != manifest.checksum:
        raise SplitManifestCorruptError(
            f"checksum mismatch (stored {manifest.checksum[:12]}, recomputed {expected[:12]})"
        )

    if cfg is not None:
        if manifest.seed != cfg.seed:
            raise SplitLeakageError(
                f"manifest seed {manifest.seed} != config seed {cfg.seed}; regenerate "
                "the split instead of reusing one from another configuration"
            )
        configured = fractions_from_config(cfg)
        drift = {
            name
            for name in SPLIT_NAMES
            if abs(manifest.fractions.get(name, -1.0) - configured[name]) > 1e-9
        }
        if drift:
            raise SplitLeakageError(
                f"manifest fractions differ from the config for {sorted(drift)}; "
                "regenerate the split"
            )
    return manifest


def require_disjoint_splits(
    cfg: Config,
    *,
    path: str | Path | None = None,
    root: Path | None = None,
) -> SplitManifest:
    """Load and fully verify the manifest. **This is what the trainers call.**

    ``concierge.rerank.train`` and ``concierge.ltr.train`` should both call this
    before touching any data, and must not catch what it raises: a split that
    cannot be verified means the training run cannot produce a reportable number,
    so failing is the correct outcome.
    """
    manifest = load_manifest(splits_path(cfg, root) if path is None else path)
    return verify_splits(manifest, cfg=cfg)


def split_of(manifest: SplitManifest, query_id: str) -> str:
    """Which split a query is in. Raises if it is in none — an unassigned query
    must never be silently treated as training data."""
    split = manifest.assignment.get(query_id)
    if split is None:
        raise SplitLeakageError(
            f"query {query_id!r} is not in the split manifest; regenerate the manifest "
            "after adding queries rather than training on unassigned data"
        )
    return split


def partition(
    manifest: SplitManifest,
    items: Iterable[_HasQueryId],
) -> dict[str, list[_HasQueryId]]:
    """Group objects carrying a ``query_id`` by their split, preserving input order."""
    out: dict[str, list[_HasQueryId]] = {name: [] for name in SPLIT_NAMES}
    for item in items:
        out[split_of(manifest, item.query_id)].append(item)
    return out
