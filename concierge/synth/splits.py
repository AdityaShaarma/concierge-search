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
   split depends only on its group — not on how many other queries exist.
   Generating more synthetic data later therefore cannot move an existing query
   from the cross-encoder's training split into LambdaMART's, which is exactly the
   leakage a reshuffle would introduce and nothing downstream would notice. The
   cost is that the split sizes are only approximately the configured fractions
   (binomial scatter around them), so the manifest records the realised counts
   rather than the nominal ones.
2. **A manifest on disk with a checksum.** The split has to be the same next week,
   on Colab, after a config edit. The checksum means a corrupted or hand-edited
   manifest is an error rather than a quietly different experiment.
3. **The group, not the query, is the unit.** See ``GroupBy`` below. Splitting by
   query id is what the project specifies literally, and it is not enough: a query
   id is a hash of its exact text, so two paraphrases of one query get unrelated
   ids and could be assigned to different splits — the cross-encoder fine-tuned on
   one, LambdaMART trained on the other, every id-level check passing. So
   ``make synth-splits`` groups by *near-duplicate cluster* (``--group-by
   cluster``), which also keeps a business's own queries together and therefore
   closes the document-level leak at the same time.

Stage 4 of 4 in ``make synth``, and the last one before training: ``main`` below
writes the manifest plus one training file per split, which is what the two
trainers read.
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

GroupBy = Literal["query", "business", "cluster"]
"""What may not span splits.

``"query"`` — the project's literal specification. A query lands in exactly one
split. **Not sufficient on its own, and not what the pipeline uses:** a query id
is a hash of its exact normalized text, so two *paraphrases* of one query get
unrelated ids and can be assigned to different splits. The cross-encoder is then
fine-tuned on one of them and scores the other while LambdaMART trains on it. No
id-level check can see that, because at the level of ids nothing is wrong.

``"business"`` — stronger. All of the queries generated from one business document
land in the same split. Worth knowing why: with ``"query"`` grouping, the
cross-encoder can be fine-tuned on "quiet spot to read" written from business X
and then be asked, during LambdaMART training, to score business X for a different
query. It has seen X's document and its score for X is mildly optimistic. That is
a *document*-level leak rather than a query-level one, and it is smaller than the
query-level leak this module prevents — but it is real.

``"cluster"`` — what ``make synth-splits`` uses. The caller supplies a
``cluster_of`` mapping (built by ``concierge.synth.generate.cluster_near_duplicates``,
which links identical text, embedding near-duplicates above the configured cosine
threshold, and — by default — queries sharing a source business). Every member of
a cluster lands in the same split, so paraphrases cannot straddle the boundary and
the document-level leak above is closed at the same time. The cost is coarser
split sizes: groups are no longer single queries, so the realised fractions
scatter further from the configured ones, which is why the manifest records the
realised counts.
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
    *,
    cluster_of: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """``[(query_id, group_key)]`` from synthetic queries, per the grouping choice.

    ``cluster_of`` is required for (and only used by) ``group_by="cluster"``: it
    maps each query id to its near-duplicate cluster's key. A query missing from
    it is an error rather than a cluster of its own — a query that silently
    becomes its own group is exactly the paraphrase that then straddles the split
    boundary, which is the leak cluster grouping exists to close.
    """
    if group_by == "query":
        return [(q.query_id, q.query_id) for q in queries]
    if group_by == "business":
        return [(q.query_id, q.business_id) for q in queries]
    if group_by == "cluster":
        if cluster_of is None:
            raise SplitError(
                "group_by='cluster' needs cluster_of; build it with "
                "concierge.synth.generate.cluster_near_duplicates"
            )
        items: list[tuple[str, str]] = []
        for q in queries:
            key = cluster_of.get(q.query_id)
            if key is None:
                raise SplitError(
                    f"query {q.query_id!r} is not in the cluster map; recompute the "
                    "clusters over exactly the queries being split"
                )
            items.append((q.query_id, key))
        return items
    raise SplitError(f"unknown group_by {group_by!r}; use 'query', 'business' or 'cluster'")


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
    cluster_of: Mapping[str, str] | None = None,
) -> SplitManifest:
    """``assign_splits`` straight from ``SyntheticQuery`` objects."""
    return assign_splits(
        items_from_queries(queries, group_by, cluster_of=cluster_of), cfg, group_by=group_by
    )


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
    cluster_of: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> tuple[SplitManifest, Path]:
    """Assign, verify and persist in one call — the normal entry point."""
    manifest = assign_splits_for_queries(queries, cfg, group_by=group_by, cluster_of=cluster_of)
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI (stage 4 of 4)
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_REPORT_FILENAME = "split_report.json"


def split_example_path(split: str, cfg: Config, root: Path | None = None) -> Path:
    """Where one split's training examples live: ``<synth dir>/<split>.jsonl``.

    This filename is a contract with the trainers, not a local choice:
    ``concierge.rerank.train.find_split_file`` looks for exactly
    ``crossencoder_train.jsonl`` in the config-fingerprinted synth directory. A
    different spelling here means a trainer that cannot find its data — or, worse,
    finds an older file somewhere else on its search path.
    """
    from concierge.synth.generate import synth_dir

    if split not in SPLIT_NAMES:
        raise SplitError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    return synth_dir(cfg, root) / f"{split}.jsonl"


def write_split_example_files(
    examples: Sequence[_HasQueryId],
    manifest: SplitManifest,
    cfg: Config,
    root: Path | None = None,
) -> dict[str, Path]:
    """Write one JSONL per split, from one mined-and-filtered example file.

    Splitting *here*, from the manifest, rather than letting each trainer filter a
    shared file by a ``split`` string on the row, is deliberate: the row and the
    manifest could disagree, and then the weaker of the two decides what a model
    trains on. With one file per split there is a single source of truth, and an
    example whose query is not in the manifest raises instead of landing in a
    default bucket (see ``split_of``).

    Imports the writer from ``concierge.synth.negatives`` inside the function so
    this module stays importable by the trainers without pulling in the mining
    machinery.
    """
    from concierge.synth.negatives import TrainingExample, write_examples_jsonl

    grouped = partition(manifest, examples)
    written: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        rows = [e for e in grouped[split] if isinstance(e, TrainingExample)]
        if len(rows) != len(grouped[split]):  # pragma: no cover - defensive
            raise SplitError(f"split {split!r} holds objects that are not TrainingExamples")
        path = split_example_path(split, cfg, root)
        write_examples_jsonl(rows, path)
        written[split] = path
    return written


def main(argv: Sequence[str] | None = None, *, embed: Any = None) -> int:
    """Stage 4 of 4: assign the splits and write one training file per split.

        python -m concierge.synth.splits --config configs/base.yaml

    Reads ``queries_deduped.jsonl`` (the leakage guard's output) and
    ``examples_filtered.jsonl`` (the judge's output), and writes, into
    ``artifacts/synth/<config fingerprint>/``:

    * ``splits.json`` — the checksummed manifest both trainers verify against;
    * ``crossencoder_train.jsonl``, ``crossencoder_val.jsonl``, ``ltr_train.jsonl``,
      ``ltr_val.jsonl`` — the per-split training files;
    * ``split_report.json`` — realised counts, fractions and cluster statistics.

    Grouping defaults to ``cluster`` (see ``GroupBy``): near-duplicate queries are
    kept on the same side of the boundary, which is what makes the out-of-fold
    cross-encoder feature true for paraphrases and not just for identical ids.
    ``--group-by business`` is the fallback that needs no embedding model;
    ``--group-by query`` is the weakest and exists only for comparison.

    Spends no API quota (the plan is printed anyway, as every stage does) and is
    idempotent: the same queries, seed and config regenerate a byte-identical
    manifest, which the checksum proves.

    **What this stage does not produce:** ``ltr_examples.jsonl``, the feature-level
    file ``concierge.ltr.train`` consumes. Those rows need a cross-encoder score
    per candidate, so they can only be built *after* the reranker is fine-tuned on
    ``crossencoder_train.jsonl``; building them is the LambdaMART stage's job and
    it reads ``ltr_train.jsonl`` and this manifest to do it.
    """
    import argparse
    import logging as logging_module

    from concierge.config import load_config
    from concierge.synth.generate import (
        QuotaPlan,
        cluster_near_duplicates_from_config,
        read_queries_jsonl,
        require_confirmation,
        synth_dir,
        write_json_artifact,
    )
    from concierge.synth.judge import EXAMPLES_FILTERED_FILENAME
    from concierge.synth.negatives import read_examples_jsonl

    parser = argparse.ArgumentParser(description="Assign synthetic train/val splits.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--group-by",
        choices=("cluster", "business", "query"),
        default="cluster",
        help="what may not span splits; 'cluster' groups near-duplicate queries",
    )
    parser.add_argument(
        "--no-merge-business",
        action="store_true",
        help=(
            "with --group-by cluster, link queries only by text similarity, not by "
            "their source business (weaker: leaves the document-level leak open)"
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="read and write under this root instead of the repo (tests, scratch runs)",
    )
    args = parser.parse_args(argv)

    logging_module.basicConfig(level=logging_module.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    root = None if args.artifacts_root is None else Path(args.artifacts_root)
    out_dir = synth_dir(cfg, root)

    from concierge.synth.generate import QUERIES_DEDUPED_FILENAME

    queries = read_queries_jsonl(out_dir / QUERIES_DEDUPED_FILENAME)
    examples_path = out_dir / EXAMPLES_FILTERED_FILENAME
    if not examples_path.exists():
        raise SplitError(
            f"no filtered examples at {examples_path}. Run `make synth-judge` first: "
            "the splits are written over the examples the judge cleaned, so the two "
            "cannot disagree about which negatives a model saw."
        )
    examples = read_examples_jsonl(examples_path)

    require_confirmation(
        QuotaPlan(
            job="split assignment",
            n_items=len(queries),
            items_label="queries to assign",
            n_batches=0,
            n_cached_batches=0,
            n_api_calls=0,
            batch_size=1,
            note="0 API calls; deterministic from the seed and the query ids",
        )
    )

    cluster_report = None
    cluster_of = None
    if args.group_by == "cluster":
        cluster_report = cluster_near_duplicates_from_config(
            queries,
            cfg,
            embed=embed,
            merge_same_business=not args.no_merge_business,
        )
        cluster_of = cluster_report.cluster_of
        print(cluster_report.summary())

    manifest, manifest_path = build_and_save(
        queries, cfg, group_by=args.group_by, cluster_of=cluster_of, root=root
    )
    print(manifest.summary())

    written = write_split_example_files(examples, manifest, cfg, root)
    counts = {split: len(read_examples_jsonl(path)) for split, path in written.items()}
    for split, n in counts.items():
        if n == 0:
            # Both trainers refuse an empty split, which is correct — but saying so
            # here points at the cause (too few queries, or every example filtered
            # out) rather than at a FileNotFoundError an hour later.
            log.warning("split %s has no training examples", split)

    write_json_artifact(
        {
            "summary": manifest.summary(),
            "manifest": str(manifest_path),
            "group_by": manifest.group_by,
            "seed": manifest.seed,
            "config_fingerprint": manifest.config_fingerprint,
            "n_queries": manifest.n_queries,
            "n_groups": manifest.n_groups,
            "query_counts": manifest.counts,
            "realised_fractions": manifest.realised_fractions(),
            "configured_fractions": dict(manifest.fractions),
            "example_counts": counts,
            "example_files": {split: str(path) for split, path in written.items()},
            "clustering": (
                None
                if cluster_report is None
                else {
                    "summary": cluster_report.summary(),
                    "cosine_threshold": cluster_report.threshold,
                    "merged_same_business": cluster_report.merged_same_business,
                    "n_clusters": cluster_report.n_clusters,
                    "n_multi_query_clusters": cluster_report.n_multi_query_clusters,
                    "largest_cluster": cluster_report.largest_cluster,
                }
            ),
        },
        out_dir / SPLIT_REPORT_FILENAME,
    )
    print(f"wrote {manifest_path} and {len(written)} per-split example file(s)")
    print(
        "Next: `make train-reranker` (reads crossencoder_train.jsonl), then build "
        "the LambdaMART feature rows from ltr_train.jsonl with the fine-tuned "
        "cross-encoder, then `make train-ltr`."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
