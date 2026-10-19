"""Train/val splits for the two models we train, and the proof that they are disjoint.

Why this file exists at all, and why it is more than four slices of a shuffled
list: LambdaMART's cross-encoder score is expected to be one of its most powerful
features (``make train-ltr`` reports the measured importances, which is where
that expectation stops being a hypothesis). If LambdaMART is trained on queries
the cross-encoder was fine-tuned on, that feature is inflated exactly where the
labels are — the cross-encoder has memorised those queries — so LambdaMART learns
"trust the cross-encoder completely", and at serving time, on queries nobody has
seen, that lesson is wrong. The model would look excellent in training and
mediocre on the gold set, with no single line of code to blame. Note that the
argument does not need the feature to be *the* strongest one: it needs the
feature to be inflated on the training queries, which it is by construction.

``cfg.ltr.enforce_oof_crossencoder_feature`` says this must be enforced in code.
``verify_splits`` and ``assert_out_of_fold`` are that enforcement: the trainers
call them and fail loudly, which is what turns the out-of-fold guarantee from a
comment into a property of the system.

Four design choices worth defending in an interview:

1. **Hash-based assignment, not a seeded shuffle.** Each query's split is
   ``hash(seed, group_key)`` bucketed by the configured fractions, so a query's
   split depends only on its group key — not on how many other queries exist and
   not on their order. For a **fixed query set** that makes the assignment stable
   and reproducible, which a reshuffle would not be.

   It does *not* make the assignment stable under data growth, and the honest
   statement of the limit matters more than the property: under the default
   ``--group-by cluster`` the hashed key is the cluster key, which is the lowest
   query id in the near-duplicate cluster (see
   ``concierge.synth.generate.cluster_near_duplicates``). Generating more
   synthetic data can admit a member with a lower id, or link two clusters into
   one, and either changes the key, the bucket and therefore the split of queries
   that were already assigned. Growing the dataset is consequently a *retrain*,
   not an append — and it is caught rather than silent: ``concierge.ltr.train``
   compares the cross-encoder artifact's recorded manifest checksum against the
   current manifest and refuses to train on a mismatch. ``--group-by query``
   restores per-query stability and gives up the paraphrase guarantee; that trade
   is the subject of design choice 3.

   The cost of hashing in either mode is that the split sizes are only
   approximately the configured fractions (binomial scatter around them), so the
   manifest records the realised counts rather than the nominal ones.
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
   closes the document-level leak at the same time. Which groupings are acceptable
   is a *config* setting (``cfg.synth.min_split_grouping``) and not a CLI default:
   a flag that quietly reopens the paraphrase straddle is exactly how a guarantee
   stops holding, so the CLI refuses a weaker grouping than the config allows and
   the manifest records the minimum that applied.
4. **The manifest carries the leakage guard's receipt.** ``make synth-generate``
   deduplicates the synthetic queries against the gold set; this stage copies that
   step's counts, cosine threshold, gold-file hash and report hash into the
   manifest, inside the checksummed payload. Without it, "the training rows were
   deduplicated against gold" is a property of the library — something a step did
   once — rather than a property of the artifact a trainer opens, and a
   hand-assembled query file would train and report clean. ``verify_splits`` holds
   a manifest to every fact it carries, and ``verify_splits_for_training`` refuses
   a manifest that carries none.

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

#: Bumped from 1 when the manifest grew the two provenance fields below
#: (``min_grouping`` and ``gold_dedup``). The version is hashed into the checksum,
#: so a v1 file is reported as corrupt rather than read with the new fields
#: defaulted — which is the correct outcome: a manifest written before the guard's
#: receipt existed cannot supply one, and regenerating it costs no API calls.
MANIFEST_VERSION = 2

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
the document-level leak above is closed at the same time.

Two costs, both real:

* coarser split sizes — groups are no longer single queries, so the realised
  fractions scatter further from the configured ones, which is why the manifest
  records the realised counts;
* **cluster membership is not monotonic under data growth.** The cluster key is
  the lowest member id, so adding synthetic queries can introduce a lower id or
  merge two clusters, changing the key and therefore the split of queries that
  were already assigned. Under ``"query"`` a query's split never moves; under
  ``"cluster"`` it can, so growing the dataset means re-running the cross-encoder
  fine-tune. That is caught, not silent — ``concierge.ltr.train`` refuses a
  cross-encoder artifact whose recorded manifest checksum no longer matches — but
  the cost lands as a forced retrain, and it is the price paid for the paraphrase
  guarantee.
"""

#: How strong each grouping is, so "weaker than configured" is a comparison and
#: not a chain of ``if``s. ``cluster`` implies ``business`` (the same-business
#: merge is on by default) which implies ``query``; a larger number is a stronger
#: guarantee about what may not span splits.
GROUPING_STRENGTH: dict[str, int] = {"query": 0, "business": 1, "cluster": 2}


def grouping_is_at_least(group_by: str, minimum: str) -> bool:
    """Is ``group_by`` at least as strong a grouping as ``minimum``?"""
    for name in (group_by, minimum):
        if name not in GROUPING_STRENGTH:
            raise SplitError(
                f"unknown grouping {name!r}; expected one of {sorted(GROUPING_STRENGTH)}"
            )
    return GROUPING_STRENGTH[group_by] >= GROUPING_STRENGTH[minimum]


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


@dataclass(frozen=True, slots=True)
class GoldDedupRecord:
    """The gold leakage guard's receipt, as a fact about *this* split.

    ``make synth-generate`` deduplicates the synthetic queries against the gold
    test set and writes ``dedup_report.json``. That file on its own proves only
    that the step ran *at some point*, in some directory, at some threshold —
    nothing ties it to the rows a trainer is about to learn from. Copying these
    fields into the manifest, inside the checksummed payload, is what turns the
    guard from a step in a Makefile into a checkable property of the artifact:
    ``verify_splits`` compares the recorded threshold against the running config
    and the recorded kept-count against the number of queries actually assigned,
    so a hand-assembled or stale query file cannot train and report clean.

    ``gold_sha256`` and ``report_sha256`` answer the two questions counts cannot:
    *which* gold set, and whether the report these numbers were copied from is
    still the one on disk.

    ``generation_complete`` is ``None`` when the report predates the field or was
    written without a generation run (a test). ``False`` means the run was
    rate-limited or lost batches, so the dataset is usable but partial — recorded
    rather than warned about once on stdout.
    """

    cosine_threshold: float
    n_input: int
    n_kept: int
    n_gold: int
    n_removed_gold: int
    n_removed_internal: int
    gold_sha256: str
    report_sha256: str
    generation_complete: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "cosine_threshold": self.cosine_threshold,
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_gold": self.n_gold,
            "n_removed_gold": self.n_removed_gold,
            "n_removed_internal": self.n_removed_internal,
            "gold_sha256": self.gold_sha256,
            "report_sha256": self.report_sha256,
            "generation_complete": self.generation_complete,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GoldDedupRecord:
        """Read the record back out of a manifest."""
        try:
            complete = data.get("generation_complete")
            return cls(
                cosine_threshold=float(data["cosine_threshold"]),
                n_input=int(data["n_input"]),
                n_kept=int(data["n_kept"]),
                n_gold=int(data["n_gold"]),
                n_removed_gold=int(data["n_removed_gold"]),
                n_removed_internal=int(data["n_removed_internal"]),
                gold_sha256=str(data["gold_sha256"]),
                report_sha256=str(data["report_sha256"]),
                generation_complete=None if complete is None else bool(complete),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitManifestCorruptError(f"unreadable gold_dedup record: {exc}") from exc

    @classmethod
    def from_dedup_report(cls, data: Mapping[str, Any], *, report_sha256: str) -> GoldDedupRecord:
        """Read the record out of ``dedup_report.json``.

        Fails closed on a report that cannot answer the questions the record
        exists to answer — a missing threshold, or a report that does not say
        which gold file it ran against. Regenerating costs no API calls (the
        responses are cached), so refusing is cheap and guessing is not.
        """
        try:
            gold_sha = str(data["gold_sha256"]).strip()
            generation = data.get("generation") or {}
            if not isinstance(generation, Mapping):
                raise TypeError("'generation' is not an object")
            complete = generation.get("complete")
            record = cls(
                cosine_threshold=float(data["cosine_threshold"]),
                n_input=int(data["n_input"]),
                n_kept=int(data["n_kept"]),
                n_gold=int(data["n_gold"]),
                n_removed_gold=int(data["n_removed_gold"]),
                n_removed_internal=int(data["n_removed_internal"]),
                gold_sha256=gold_sha,
                report_sha256=report_sha256,
                generation_complete=None if complete is None else bool(complete),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitError(
                f"dedup report is missing or malformed field(s) ({exc}); regenerate it "
                "with `make synth-generate` rather than splitting rows whose leakage "
                "guard cannot be described"
            ) from exc
        if not record.gold_sha256:
            raise SplitError(
                "dedup report does not record gold_sha256, so it cannot say which gold "
                "set the training queries were deduplicated against. Rerun "
                "`make synth-generate` (cached batches cost nothing)."
            )
        return record

    def summary(self) -> str:
        partial = "" if self.generation_complete is not False else "; PARTIAL generation run"
        return (
            f"gold dedup: {self.n_input} in, {self.n_kept} kept, "
            f"{self.n_removed_gold} removed as gold duplicates and "
            f"{self.n_removed_internal} as internal duplicates at cosine >= "
            f"{self.cosine_threshold} against {self.n_gold} gold queries "
            f"(gold sha256 {self.gold_sha256[:12]}){partial}"
        )


def compute_checksum(
    assignment: Mapping[str, str],
    seed: int,
    fractions: Mapping[str, float],
    group_by: str,
    *,
    min_grouping: str = "",
    gold_dedup: GoldDedupRecord | None = None,
) -> str:
    """Checksum over everything that defines the split.

    The seed, fractions and grouping are hashed in alongside the assignment so
    that a manifest produced under a different config cannot pass verification by
    accident: the same query ids under a different seed are a *different split*,
    even if the file happens to look similar.

    The two provenance fields are hashed in as well, and unconditionally — the
    keys are present in the payload even when empty. Hashing them only when set
    would mean that *deleting* the gold-dedup record from a manifest produced a
    file that still passed its own checksum, which is precisely the edit the
    record exists to make detectable.
    """
    payload = json.dumps(
        {
            "version": MANIFEST_VERSION,
            "seed": seed,
            "group_by": group_by,
            "min_grouping": min_grouping,
            "gold_dedup": None if gold_dedup is None else gold_dedup.to_json(),
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
    #: The weakest grouping the *config* allowed when this manifest was built
    #: (``cfg.synth.min_split_grouping``). Recorded by the CLI, which is the one
    #: caller that enforces it; empty for a manifest built directly from the
    #: library (tests, notebooks), where nothing promised a grouping and
    #: ``verify_splits`` says so out loud rather than assuming one.
    min_grouping: str = ""
    #: The gold leakage guard's receipt for the rows in this split, or ``None``
    #: when the manifest carries no evidence that the guard ran.
    gold_dedup: GoldDedupRecord | None = None
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
        dedup = (
            "gold dedup NOT recorded"
            if self.gold_dedup is None
            else f"gold dedup recorded at cosine >= {self.gold_dedup.cosine_threshold}"
        )
        return (
            f"{self.n_queries} queries in {self.n_groups} group(s) by {self.group_by}, "
            f"seed {self.seed}; counts {self.counts}; realised fractions {realised}; "
            f"{dedup}; checksum {self.checksum[:12]}"
        )

    # ── serialization ────────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "group_by": self.group_by,
            "min_grouping": self.min_grouping,
            "gold_dedup": None if self.gold_dedup is None else self.gold_dedup.to_json(),
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
            raw_dedup = data.get("gold_dedup")
            manifest = cls(
                seed=int(data["seed"]),
                group_by=str(data["group_by"]),
                fractions=fractions,
                assignment=assignment,
                checksum=str(data["checksum"]),
                n_groups=int(data["n_groups"]),
                config_fingerprint=str(data.get("config_fingerprint", "")),
                min_grouping=str(data.get("min_grouping", "")),
                gold_dedup=(None if raw_dedup is None else GoldDedupRecord.from_json(raw_dedup)),
                version=int(data.get("version", MANIFEST_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitManifestCorruptError(f"unreadable split manifest: {exc}") from exc
        expected = compute_checksum(
            manifest.assignment,
            manifest.seed,
            manifest.fractions,
            manifest.group_by,
            min_grouping=manifest.min_grouping,
            gold_dedup=manifest.gold_dedup,
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
    min_grouping: str = "",
    gold_dedup: GoldDedupRecord | None = None,
) -> SplitManifest:
    """Assign ``(query_id, group_key)`` pairs to the four splits.

    A group is assigned once and every query in it inherits that split, which is
    what the word "group" buys: with ``group_by="business"`` all of a business's
    queries move together.

    A query id appearing twice with *different* group keys is an error, not a
    last-one-wins: it would mean the same query belongs to two groups, and the
    disjointness guarantee would be meaningless.

    ``min_grouping`` and ``gold_dedup`` are the provenance the CLI stamps in (see
    ``SplitManifest``). They default to "nothing recorded" rather than to the
    running config's values on purpose: this function cannot know whether the
    grouping it was handed was *enforced* or whether the guard ran over these
    exact rows, and writing the config's intent into the manifest as though it
    were an observation is how a manifest comes to assert something nobody
    checked. ``make synth-splits`` is the caller that knows, so it is the caller
    that fills them.
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
        checksum=compute_checksum(
            assignment,
            cfg.seed,
            fractions,
            group_by,
            min_grouping=min_grouping,
            gold_dedup=gold_dedup,
        ),
        n_groups=len(split_of_group),
        config_fingerprint=cfg.fingerprint(),
        min_grouping=min_grouping,
        gold_dedup=gold_dedup,
    )
    log.info("split: %s", manifest.summary())
    return manifest


def assign_splits_for_queries(
    queries: Sequence[_HasQueryAndBusiness],
    cfg: Config,
    *,
    group_by: GroupBy = "query",
    cluster_of: Mapping[str, str] | None = None,
    min_grouping: str = "",
    gold_dedup: GoldDedupRecord | None = None,
) -> SplitManifest:
    """``assign_splits`` straight from ``SyntheticQuery`` objects."""
    return assign_splits(
        items_from_queries(queries, group_by, cluster_of=cluster_of),
        cfg,
        group_by=group_by,
        min_grouping=min_grouping,
        gold_dedup=gold_dedup,
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
    min_grouping: str = "",
    gold_dedup: GoldDedupRecord | None = None,
) -> tuple[SplitManifest, Path]:
    """Assign, verify and persist in one call — the normal entry point."""
    manifest = assign_splits_for_queries(
        queries,
        cfg,
        group_by=group_by,
        cluster_of=cluster_of,
        min_grouping=min_grouping,
        gold_dedup=gold_dedup,
    )
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
       config change cannot be answered with a stale split;
    5. **every fact the manifest carries is held against the config**: a recorded
       grouping minimum must actually be met and must not be weaker than the
       config's, and a recorded gold-dedup receipt must match the configured
       cosine threshold and must cover exactly the queries being split.

    What (5) deliberately does *not* do is invent the facts a manifest does not
    carry. A manifest with no recorded provenance — built straight from the
    library by a test or a notebook, or predating the fields — is reported as
    unverified at WARNING level rather than assumed good or assumed bad: this
    function's job is to check claims, and there is no claim to check. The gate
    that *requires* the claims is ``verify_splits_for_training``, which is what a
    run whose numbers get reported should call; ``make synth-splits`` runs it over
    its own output and records the result in ``split_report.json``.

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
        manifest.assignment,
        manifest.seed,
        manifest.fractions,
        manifest.group_by,
        min_grouping=manifest.min_grouping,
        gold_dedup=manifest.gold_dedup,
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
        _verify_recorded_grouping(manifest, cfg)
        _verify_recorded_gold_dedup(manifest, cfg)
    return manifest


def _verify_recorded_grouping(manifest: SplitManifest, cfg: Config) -> None:
    """Hold a manifest to the grouping minimum it records, and to the config's.

    Three cases, and the middle one is the leak (leakage-8): a manifest built with
    ``--group-by query`` lets two paraphrases of one query sit on opposite sides
    of the cross-encoder/LambdaMART boundary while every id-level check passes.
    """
    required = cfg.synth.min_split_grouping
    if not manifest.min_grouping:
        log.warning(
            "split manifest records no grouping minimum, so its grouping (%s) is "
            "UNVERIFIED against synth.min_split_grouping=%s. Regenerate it with "
            "`make synth-splits` to make the grouping a recorded fact.",
            manifest.group_by,
            required,
        )
        return
    if not grouping_is_at_least(manifest.group_by, manifest.min_grouping):
        raise SplitLeakageError(
            f"manifest groups by {manifest.group_by!r} but records a minimum of "
            f"{manifest.min_grouping!r}; it contradicts itself, so neither claim can "
            "be trusted. Regenerate the split."
        )
    if not grouping_is_at_least(manifest.group_by, required):
        raise SplitLeakageError(
            f"manifest groups by {manifest.group_by!r}, weaker than the configured "
            f"minimum {required!r} (synth.min_split_grouping). Under a weaker grouping "
            "two paraphrases of one query can land on opposite sides of the "
            "cross-encoder/LambdaMART boundary with every id-level check still "
            "passing, which inflates the cross-encoder feature exactly where "
            "LambdaMART's labels are. Regenerate the split, or lower the config "
            "setting deliberately (it is part of the fingerprint, so comparison runs "
            "get their own artifact directory)."
        )


def _verify_recorded_gold_dedup(manifest: SplitManifest, cfg: Config) -> None:
    """Hold a manifest to the gold-dedup receipt it records.

    The point is the *tie*: a receipt whose kept-count does not equal the number
    of queries in the split describes some other set of rows, and a receipt at a
    different cosine threshold than the running config describes some other
    experiment. Either way the claim "these rows were deduplicated against gold at
    the configured threshold" is unsupported, and an unsupported leakage guarantee
    has to be an error rather than a footnote.
    """
    record = manifest.gold_dedup
    if record is None:
        log.warning(
            "split manifest records no gold-dedup receipt: there is no evidence in "
            "this artifact that its queries were deduplicated against the gold set. "
            "Regenerate with `make synth-generate && make synth-splits` before "
            "reporting any number trained on it."
        )
        return
    if abs(record.cosine_threshold - cfg.synth.gold_dedup_cosine_threshold) > 1e-9:
        raise SplitLeakageError(
            f"gold dedup ran at cosine {record.cosine_threshold} but the config now "
            f"says {cfg.synth.gold_dedup_cosine_threshold}; the guard that protected "
            "these rows is not the guard this configuration describes. Rerun "
            "`make synth-generate` (cached batches cost nothing) and re-split."
        )
    if record.n_kept != manifest.n_queries:
        raise SplitLeakageError(
            f"gold dedup kept {record.n_kept} queries but this manifest splits "
            f"{manifest.n_queries}; the receipt describes a different set of rows, so "
            "it proves nothing about these. Re-split from the deduplicated query file."
        )
    if record.generation_complete is False:
        log.warning(
            "the generation run behind this split was PARTIAL (rate-limited or lost "
            "batches): the dataset is usable but incomplete, and any number trained "
            "on it has to say so."
        )


def verify_splits_for_training(manifest: SplitManifest, cfg: Config) -> SplitManifest:
    """``verify_splits`` plus *presence* of the provenance a reported run needs.

    The difference is deliberate and is the whole of SB2/leakage-6: ``verify_splits``
    checks the claims a manifest makes, which leaves "makes no claims" passing.
    This gate refuses that state. Call it before training anything whose numbers
    will be reported:

    * the gold-dedup receipt must be present (and, via ``verify_splits``, must tie
      to these rows at this threshold), so the leakage guard is evidence rather
      than a remembered habit;
    * the grouping minimum must be recorded and met, so the paraphrase guarantee
      is not resting on an argparse default.

    A partial generation run is *not* refused here — it is a known-and-recorded
    property of the data, not an unverifiable one — but it is logged by
    ``verify_splits`` every time the manifest is read.
    """
    verify_splits(manifest, cfg=cfg)
    missing: list[str] = []
    if manifest.gold_dedup is None:
        missing.append(
            "gold_dedup (no evidence the gold leakage guard ran over these queries; "
            "run `make synth-generate`, then `make synth-splits`)"
        )
    if not manifest.min_grouping:
        missing.append(
            "min_grouping (no evidence the grouping was enforced against "
            "synth.min_split_grouping; rebuild the manifest with `make synth-splits`)"
        )
    if missing:
        raise SplitLeakageError(
            "this split manifest cannot support a reported number — it is missing: "
            + "; ".join(missing)
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

    It runs ``verify_splits``, which holds the manifest to every fact it records
    and logs a warning for every fact it does not. A trainer that wants the
    provenance *required* rather than reported — which is what a run feeding the
    README should want — calls ``verify_splits_for_training`` on the manifest this
    returns, and copies ``manifest.gold_dedup`` into its own run metadata so the
    model artifact carries the receipt too.
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

    * ``splits.json`` — the checksummed manifest both trainers verify against,
      carrying the gold leakage guard's receipt (see ``GoldDedupRecord``);
    * ``crossencoder_train.jsonl``, ``crossencoder_val.jsonl``, ``ltr_train.jsonl``,
      ``ltr_val.jsonl`` — the per-split training files;
    * ``split_report.json`` — realised counts, fractions, cluster statistics and
      whether the manifest passed ``verify_splits_for_training``.

    Grouping defaults to ``cluster`` (see ``GroupBy``): near-duplicate queries are
    kept on the same side of the boundary, which is what makes the out-of-fold
    cross-encoder feature true for paraphrases and not just for identical ids.
    ``--group-by business`` is the fallback that needs no embedding model and
    ``--group-by query`` is the weakest, and **both are refused unless
    ``synth.min_split_grouping`` in the config allows them**: a comparison run has
    to be a config change (which is part of the fingerprint, so it gets its own
    artifact directory) rather than a flag that silently reopens the paraphrase
    straddle on the next run.

    Spends no API quota (the plan is printed anyway, as every stage does) and is
    idempotent: the same queries, seed and config regenerate a byte-identical
    manifest, which the checksum proves.

    **What this stage does not produce:** ``ltr_examples.jsonl``, the feature-level
    file ``concierge.ltr.train`` consumes. Those rows need a cross-encoder score
    per candidate, so they can only be built *after* the reranker is fine-tuned on
    ``crossencoder_train.jsonl``, from ``ltr_train.jsonl`` plus this manifest. **No
    module in the repository writes them yet** — that stage is a gap, not a
    delegation, and ``make train-ltr`` fails closed on the missing file rather
    than training on rows whose cross-encoder scores nothing can account for
    (see ``concierge.ltr.train.default_examples_path``).
    """
    import argparse
    import logging as logging_module

    from concierge.config import load_config
    from concierge.synth.generate import (
        DEDUP_REPORT_FILENAME,
        QUERIES_DEDUPED_FILENAME,
        QuotaPlan,
        cluster_near_duplicates_from_config,
        file_sha256,
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
        "--dedup-report",
        default=None,
        help=(
            "path to the gold leakage guard's report (default: dedup_report.json in "
            "the synth directory). Its counts are recorded in the manifest."
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

    # The grouping is enforced against the config before anything is read, so a
    # weakening flag fails in milliseconds instead of after the clustering pass.
    # This is the hole that made the leakage-8 fix depend on an argparse default:
    # `--group-by query` used to rebuild the manifest in exactly the mode the fix
    # removed, and every downstream check still reported clean because at the
    # level of ids nothing is wrong.
    required_grouping = cfg.synth.min_split_grouping
    if not grouping_is_at_least(args.group_by, required_grouping):
        raise SplitError(
            f"--group-by {args.group_by} is weaker than synth.min_split_grouping="
            f"{required_grouping} in {args.config}. A weaker grouping lets two "
            "paraphrases of one query straddle the cross-encoder/LambdaMART boundary "
            "(see GroupBy). To run that comparison, lower the config setting: it is "
            "part of the config fingerprint, so the weaker run gets its own artifact "
            "directory and the manifest records which minimum applied."
        )
    if args.no_merge_business and required_grouping in ("cluster", "business"):
        # Same argument one level down. The same-business merge is the only reason
        # "cluster" implies "business" in GROUPING_STRENGTH: without it, a
        # business's own queries can be split apart again, so the manifest would
        # record a minimum it does not actually meet. Dropping the merge is
        # therefore a weakening too, and belongs in the config like the others.
        raise SplitError(
            "--no-merge-business drops the same-business merge, so a business's own "
            "queries can land in different splits — weaker than "
            f"synth.min_split_grouping={required_grouping}. Set it to 'query' "
            "deliberately if text-only clustering is the comparison you want."
        )

    queries = read_queries_jsonl(out_dir / QUERIES_DEDUPED_FILENAME)
    examples_path = out_dir / EXAMPLES_FILTERED_FILENAME
    if not examples_path.exists():
        raise SplitError(
            f"no filtered examples at {examples_path}. Run `make synth-judge` first: "
            "the splits are written over the examples the judge cleaned, so the two "
            "cannot disagree about which negatives a model saw."
        )
    examples = read_examples_jsonl(examples_path)

    # ── the gold leakage guard's receipt ─────────────────────────────────────
    # Read here and recorded in the manifest, which is the whole of SB2: the guard
    # runs two stages earlier, and without this the only evidence it ran over
    # *these* rows is that somebody remembers typing `make synth-generate`.
    #
    # A missing report is a loud warning rather than a hard error because a
    # hand-built query file is a legitimate testing path, and failing here would
    # make the library untestable without a full generation run. What makes the
    # warning more than a comment is that the absence travels too: the manifest
    # records `gold_dedup: null`, `verify_splits` warns on every read, and
    # `verify_splits_for_training` — run below and reported in split_report.json —
    # refuses it outright.
    dedup_path = (
        out_dir / DEDUP_REPORT_FILENAME if args.dedup_report is None else Path(args.dedup_report)
    )
    gold_dedup: GoldDedupRecord | None = None
    if dedup_path.exists():
        dedup_data = json.loads(dedup_path.read_text(encoding="utf-8"))
        gold_dedup = GoldDedupRecord.from_dedup_report(
            dedup_data, report_sha256=file_sha256(dedup_path)
        )
        # The one cross-check the counts cannot make: a query the guard removed
        # must not be back in the file being split. That is what a hand-edited or
        # half-regenerated queries_deduped.jsonl looks like from here, and it is
        # the only failure mode in which the receipt would describe a clean dedup
        # over rows that include a gold duplicate.
        removed_ids = {
            str(row.get("query_id"))
            for row in (dedup_data.get("removed") or [])
            if isinstance(row, Mapping)
        }
        resurrected = sorted({q.query_id for q in queries} & removed_ids)
        if resurrected:
            raise SplitLeakageError(
                f"{len(resurrected)} query/queries the gold dedup removed are back in "
                f"{out_dir / QUERIES_DEDUPED_FILENAME} (e.g. {resurrected[:5]}). The "
                "query file and the dedup report disagree; regenerate both with "
                "`make synth-generate`."
            )
        print(gold_dedup.summary())
    else:
        log.error(
            "no gold-dedup report at %s: this split will record NO evidence that its "
            "queries were deduplicated against the gold set, and nothing trained on "
            "it can be reported. Run `make synth-generate` first.",
            dedup_path,
        )

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
        queries,
        cfg,
        group_by=args.group_by,
        cluster_of=cluster_of,
        root=root,
        min_grouping=required_grouping,
        gold_dedup=gold_dedup,
    )
    print(manifest.summary())

    # The stage checks its own output against the gate the trainers need, so "this
    # manifest cannot support a reported number" is known now rather than an hour
    # into a fine-tune. Not fatal: the per-split files are still worth writing for
    # an exploratory run, and the reason is recorded in split_report.json.
    training_gate: dict[str, Any] = {"ok": True, "reason": ""}
    try:
        verify_splits_for_training(manifest, cfg)
    except SplitError as exc:
        training_gate = {"ok": False, "reason": str(exc)}
        log.error("this split is NOT fit for a reported training run: %s", exc)

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
            "min_grouping": manifest.min_grouping,
            "gold_dedup": None if manifest.gold_dedup is None else manifest.gold_dedup.to_json(),
            "dedup_report": str(dedup_path) if gold_dedup is not None else None,
            "fit_for_reported_training": training_gate,
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
    if training_gate["ok"]:
        print("this split carries the provenance a reported training run needs")
    else:
        print(f"WARNING: not fit for a reported training run — {training_gate['reason']}")
    print(
        "Next: `make train-reranker` (reads crossencoder_train.jsonl), then build "
        "the LambdaMART feature rows from ltr_train.jsonl with the fine-tuned "
        "cross-encoder, then `make train-ltr`."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
