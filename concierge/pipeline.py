"""End-to-end search: the four stages wired into one callable.

    pipeline = SearchPipeline.from_config(load_config())
    response = pipeline.search("quiet cafe with wifi open past 10pm")

Three things this module exists to get right.

**1. Load once, search many times.**
Every expensive object — the BM25 index, the FAISS index, the embedding model,
the cross-encoder, the LambdaMART booster — is loaded in `from_config` and held
on the instance. A per-request load would dominate the latency numbers that are
a headline result of this project, and the FastAPI service would be unusable.
Everything the pipeline needs can also be *injected*, which is what makes the
whole cascade testable in milliseconds with no model downloads (`tests/test_pipeline.py`).

**2. Every stage is independently disableable.**
The project's main deliverable is an ablation table with one row per stage, and
each row is this same pipeline run in a different configuration. That is why the
toggle is a `Stage` flag inside a frozen `Variant` object, and not a handful of
boolean keyword arguments:

* the six table rows are *data* (`ABLATION`), defined once, in this module, so
  the table cannot drift from the code that produced it;
* a `Variant` is hashable and printable, so a result row can carry the exact
  variant that produced it;
* flags compose, so "the full system minus hard filters" is an expression
  rather than a new code path.

The one thing that is *not* a flag is which cross-encoder checkpoint to use
(pretrained vs. fine-tuned). Those are two values of one choice, not two
independent switches — as separate booleans, "both on" would be a meaningless
state the type system invited. So it is an enum field read only when
`Stage.RERANK` is on.

**3. A missing model degrades, it does not crash.**
This pipeline has to be runnable at every point in the project's own
construction: before the cross-encoder is fine-tuned, before LambdaMART is
trained, before the dense index is built. A missing artifact therefore produces
a warning and a shorter cascade (fine-tuned -> pretrained -> no reranking;
LambdaMART -> cross-encoder order), never an exception. The warning names the
artifact, because a silent degradation would show up only as a mysteriously
weak ablation row.

**4. One definition of the cascade order.**
`search()` answers a request, so it returns rendered `SearchResult`s: the
business record, the final score, the matched constraints. Offline stages want
something else — the *candidates*, with every per-stage score **and rank** still
attached, before Stage 4 touches them (`concierge.ltr.build_examples` builds
LambdaMART's training rows from exactly that, and two of its features are
`rrf_rank` and the cross-encoder rank, which a `SearchResult` has dropped).
`candidates()` is that entry point, and `search()` calls it rather than running
the stages itself. The alternative — an offline producer that restates the order
of retrieve/filter/rerank — is a duplication that drifts silently: someone
reorders the stages in one place, the producer keeps building rows the old way,
nothing errors, and Stage 4 is trained on a feature distribution the served
system no longer produces (DECISIONS.md D228).

Latency is measured on every search, not behind a debug flag, for the same
reason: p50/p95 per stage is a reported result, and a measurement that only
happens when someone remembers to ask for it is a measurement that will be
missing from the README.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from concierge.config import Config
from concierge.query import parser as query_parser

# `query_text_for` is the single definition of "which text does a stage score" —
# semantic_text, falling back to the raw query. It lives in the rerank module
# because that is where the choice is ablated; importing it (rather than
# restating the rule) keeps retrieval and reranking scoring the same string.
from concierge.rerank.model import query_text_for
from concierge.retrieval import bm25 as bm25_mod
from concierge.retrieval import constraints_index as constraints_mod
from concierge.retrieval import fusion
from concierge.retrieval.filters import apply_filters, evaluate_constraints
from concierge.types import (
    SCORE_BM25,
    SCORE_CROSS_ENCODER,
    SCORE_DENSE,
    SCORE_LTR,
    SCORE_RRF,
    Business,
    BusinessDocument,
    Candidate,
    ParsedQuery,
    SearchResponse,
    SearchResult,
    StageTiming,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage names, used as timing and candidate-count keys
# ─────────────────────────────────────────────────────────────────────────────

STAGE_PARSE: Final = "parse"
STAGE_RETRIEVE: Final = "retrieve"
STAGE_FILTER: Final = "filter"
STAGE_RERANK: Final = "rerank"
STAGE_LTR: Final = "ltr"
STAGE_ASSEMBLE: Final = "assemble"

#: Order of the cascade. `SearchResponse.n_candidates` carries one entry per
#: name here (plus `final`), and those counts must be non-increasing — a later
#: stage handing on more candidates than it received would be a bug.
CASCADE: Final[tuple[str, ...]] = (
    STAGE_RETRIEVE,
    STAGE_FILTER,
    STAGE_RERANK,
    STAGE_LTR,
)
COUNT_FINAL: Final = "final"

#: Prefix for the filter stage's drop reasons in `n_candidates`. The failure
#: analysis needs to answer "why did this query return three results?", and
#: `FilterReport` already counts the reasons; carrying them out through the
#: existing `n_candidates` dict keeps `SearchResponse` (a frozen shared
#: contract) unchanged.
DROPPED_PREFIX: Final = "dropped_"
COUNT_PENALIZED: Final = "penalized_missing_data"


# ─────────────────────────────────────────────────────────────────────────────
# The stage-toggle mechanism
# ─────────────────────────────────────────────────────────────────────────────


class Stage(Flag):
    """The independently switchable parts of the cascade.

    A flag enum rather than booleans so a configuration is one value that can be
    stored, compared, logged and composed (`VARIANT_FULL.stages & ~Stage.FILTERS`).
    """

    NONE = 0
    #: Stage 1 via the LLM. Off means the rule parser, with no network call at
    #: all — that is what makes the "+LLM query parsing" ablation row honest
    #: about both quality *and* latency.
    QUERY_LLM = auto()
    #: Stage 2, lexical half.
    BM25 = auto()
    #: Stage 2, dense half. BM25 alone is the first ablation row; both together
    #: are fused with RRF.
    DENSE = auto()
    #: Stage 2, hard constraints (hours/price/attributes/category/geo) plus the
    #: missing-data policy. Separately switchable so the contribution of exact
    #: constraint checking over pure similarity can be measured.
    FILTERS = auto()
    #: Stage 3, the cross-encoder. Which checkpoint is `Variant.reranker`.
    RERANK = auto()
    #: Stage 4, LambdaMART.
    LTR = auto()


class RerankCheckpoint(str, Enum):  # noqa: UP042 - matches ParseSource in types.py
    """Which cross-encoder weights Stage 3 should use.

    Two values of one choice — the ablation reports the pretrained model and our
    fine-tuned model as separate rows, and exactly one of them runs.
    """

    PRETRAINED = "pretrained"
    FINETUNED = "finetuned"


@dataclass(frozen=True, slots=True)
class Variant:
    """One system configuration: a named point in the ablation.

    `reranker` is read only when `Stage.RERANK` is enabled; it is ignored
    otherwise rather than validated against the flag, so that
    `variant.stages | Stage.RERANK` is always a legal expression.
    """

    name: str
    stages: Stage
    reranker: RerankCheckpoint = RerankCheckpoint.FINETUNED

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a variant needs a name; it labels an ablation row")
        if not self.stages & (Stage.BM25 | Stage.DENSE):
            # With neither retriever there is no candidate set for anything
            # downstream to rank. That is a mistake in an experiment definition,
            # and it should fail when the variant is written, not produce a
            # plausible-looking row of zeros.
            raise ValueError(
                f"variant {self.name!r} enables no retriever; "
                "at least one of Stage.BM25 / Stage.DENSE is required"
            )

    def enabled(self, stage: Stage) -> bool:
        return stage in self.stages


_BM25_ONLY = Stage.BM25 | Stage.FILTERS
_HYBRID = _BM25_ONLY | Stage.DENSE
_RERANKED = _HYBRID | Stage.RERANK
_PARSED = _RERANKED | Stage.QUERY_LLM
_FULL = _PARSED | Stage.LTR

#: **The ablation table, defined once.** `eval/run_eval.py` iterates this; the
#: README table has one row per entry, in this order. Each row adds exactly one
#: thing to the row above it, which is what lets a difference between two
#: adjacent rows be attributed to that one thing.
#:
#: Note what is *not* varied here: `Stage.FILTERS` is on in every row. Row 5
#: measures LLM query parsing against the rule parser, not against having no
#: constraint handling at all — rows 1-4 still apply the constraints the rule
#: parser finds. The filters-vs-no-filters comparison is `VARIANTS["no_filters"]`,
#: a separate experiment rather than a row of the main table.
ABLATION: Final[tuple[Variant, ...]] = (
    Variant(name="bm25", stages=_BM25_ONLY),
    Variant(name="hybrid_rrf", stages=_HYBRID),
    Variant(name="rerank_pretrained", stages=_RERANKED, reranker=RerankCheckpoint.PRETRAINED),
    Variant(name="rerank_finetuned", stages=_RERANKED, reranker=RerankCheckpoint.FINETUNED),
    Variant(name="llm_parse", stages=_PARSED, reranker=RerankCheckpoint.FINETUNED),
    Variant(name="full", stages=_FULL, reranker=RerankCheckpoint.FINETUNED),
)

#: The production configuration: everything on.
VARIANT_FULL: Final[Variant] = ABLATION[-1]

#: Named variants, including diagnostics that are not ablation rows.
VARIANTS: Final[dict[str, Variant]] = {
    **{v.name: v for v in ABLATION},
    # Full system minus hard constraint checking: how much of the quality comes
    # from arithmetic rather than similarity.
    "no_filters": Variant(name="no_filters", stages=_FULL & ~Stage.FILTERS),
    # Dense half alone, as the counterpart to the `bm25` row.
    "dense_only": Variant(name="dense_only", stages=Stage.DENSE | Stage.FILTERS),
}


# ─────────────────────────────────────────────────────────────────────────────
# What the pipeline needs from Stages 3 and 4
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Retriever(Protocol):
    """Stage 2's surface. `BM25Index` and `DenseIndex` both satisfy it.

    That the two halves of retrieval are interchangeable is not an accident of
    this protocol — it is what lets Reciprocal Rank Fusion treat them alike,
    and it is asserted by both of their own test suites.
    """

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Up to `top_k` `(business_id, score)` pairs, best first."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Stage 3's surface, matching `concierge.rerank.model.CrossEncoderReranker`.

    A protocol rather than a hard dependency on the class so that a test (or a
    sweep over reranker sizes) can inject a substitute. The pipeline delegates
    the *whole* Stage 3 contract — scoring, the stable sort, writing
    `SCORE_CROSS_ENCODER` and `RANK_CROSS_ENCODER`, truncating to `top_k` — to
    this object rather than reimplementing any of it here: two orderings of the
    same stage would eventually disagree, and the eval would be measuring
    whichever one it happened to call.
    """

    def rerank(
        self,
        parsed_query: ParsedQuery,
        candidates: Sequence[Candidate],
        documents: Mapping[str, str],
        top_k: int,
    ) -> tuple[Candidate, ...]:
        """Score every candidate jointly with its document; return the best `top_k`."""
        ...


@runtime_checkable
class LtrRanker(Protocol):
    """Stage 4's inference surface.

    It takes whole `Candidate` objects (not just text) because LambdaMART's
    features are the upstream stage scores, the business's rating and
    popularity, and the constraint-match counts — all of which are already on
    the candidate and the business record. Feature *construction* lives in
    `concierge.ltr`, deliberately not here: the pipeline must not know the
    feature list, or changing it would mean editing two files.
    """

    def score(
        self,
        parsed: ParsedQuery,
        candidates: Sequence[Candidate],
        businesses: Mapping[str, Business],
    ) -> Sequence[float]:
        """One score per candidate, in input order. Higher is better."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────


class StageRecorder:
    """Collects a `StageTiming` per stage and the per-stage candidate counts.

    Deliberately boring: the pipeline must never be in a state where timings
    are half-collected, so the context manager records in a `finally` and a
    stage that raises still reports the time it burned.

    Public, because `candidates()` is public and its caller may want the same
    measurements `search()` reports — `concierge.ltr.build_examples` records how
    many candidates the hard filters removed from each training query's pool.
    Handing the measurement sink in, rather than returning a second object from
    `candidates()`, keeps that method's return type the candidate list itself.
    """

    def __init__(self) -> None:
        self.timings: list[StageTiming] = []
        self.counts: dict[str, int] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        # Refuse a second measurement of the same stage rather than appending it.
        #
        # One recorder measures ONE cascade run. The two fields disagree about
        # what reuse means — `timings` is a list and appends, `counts` is a dict
        # keyed by stage and overwrites — so a caller that reused a recorder
        # across queries would collect every stage several times while keeping
        # only the last query's counts. Nothing would raise; the per-stage p50
        # and p95 computed from those timings would simply be wrong, and wrong
        # in the direction of looking plausible.
        #
        # Latency is a reported result of this project, so the cheapest honest
        # contract is the one that cannot be used incorrectly: make reuse an
        # error instead of documenting that it is a mistake.
        if any(existing.stage == stage for existing in self.timings):
            raise RuntimeError(
                f"StageRecorder already measured {stage!r}. A recorder measures one "
                "cascade run; reusing it across queries mixes their timings while "
                "keeping only the last query's counts. Create a new recorder per query."
            )
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings.append(
                StageTiming(stage=stage, ms=(time.perf_counter() - started) * 1000.0)
            )


# ─────────────────────────────────────────────────────────────────────────────
# The pipeline
# ─────────────────────────────────────────────────────────────────────────────


class SearchPipeline:
    """The four stages, held in memory, run per query.

    Not a frozen dataclass (unlike the rest of the project's types) because this
    object owns loaded models and a one-shot warning ledger; it is a long-lived
    service component, not a value passed between stages.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        businesses: Mapping[str, Business],
        documents: Sequence[BusinessDocument] | Mapping[str, str],
        bm25_index: Retriever | None = None,
        dense_index: Retriever | None = None,
        reranker: Reranker | Mapping[RerankCheckpoint, Reranker] | None = None,
        ltr_ranker: LtrRanker | None = None,
        constraint_index: constraints_mod.ConstraintIndex | None = None,
        variant: Variant = VARIANT_FULL,
        now: datetime | None = None,
    ) -> None:
        """Assemble a pipeline from already-loaded parts.

        Args:
            cfg: the loaded config. `cfg.candidates` sets the cascade widths.
            businesses: `business_id -> Business`, needed by the filters (hours,
                price, attributes) and to render results.
            documents: the indexed documents, or just `business_id -> text`.
                Only the text is kept: it is what the cross-encoder reads.
            bm25_index: anything with `.search(text, top_k) -> [(id, score)]`.
            dense_index: same signature. The two retrievers are interchangeable
                by construction, which is what lets RRF treat them alike.
            reranker: one reranker (used for whichever checkpoint a variant
                asks for, which is what a test wants) or a mapping from
                checkpoint to reranker, which is what serving wants.
            ltr_ranker: Stage 4, or None to stop after Stage 3.
            constraint_index: Stage 2's constraint-only path. None means build
                one from `businesses`, which is what every caller wants; the
                parameter exists so a test can inject a substitute.
            variant: the default configuration; `search()` can override it.
            now: injected clock for resolving "open late" against a weekday.
                A parameter so a test does not depend on the day it runs.
        """
        self.cfg = cfg
        self.businesses = dict(businesses)
        self.doc_text = _as_text_map(documents)
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.rerankers = _as_reranker_map(reranker)
        self.ltr_ranker = ltr_ranker
        self.variant = variant
        self._now = now

        # Stage 2's constraint-only path (`concierge.retrieval.constraints_index`).
        # Built here rather than in `from_config` because it needs nothing but
        # the business records this constructor already holds, and because every
        # construction path — serving, eval, tests — must have it: a query whose
        # `semantic_text` is a generic residue retrieves nothing, and without
        # this the correctly-parsed constraints then have no candidate list to
        # filter (DECISIONS.md D224). Injectable so a test can substitute one.
        self.constraint_index = (
            constraints_mod.ConstraintIndex.build(self.businesses.values())
            if constraint_index is None
            else constraint_index
        )

        # Exact-string vocabulary, as the dataset spells it: the schema
        # validator drops categories it cannot find here, which is how a
        # hallucinated category is prevented from emptying a result set.
        self._known_categories = frozenset(
            category for business in self.businesses.values() for category in business.categories
        )

        # `provider: rules` is how the rule parser is run with no network call.
        # Built once here rather than per query: a pydantic model_copy per
        # request would be measured as parse latency and is pure waste.
        self._rules_cfg = cfg.model_copy(
            update={"query_parsing": cfg.query_parsing.model_copy(update={"provider": "rules"})}
        )
        self._warned: set[str] = set()

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        variant: Variant = VARIANT_FULL,
        encoder: object | None = None,
        artifacts_root: Path | None = None,
        processed_dir: Path | None = None,
        load_models: bool = True,
        now: datetime | None = None,
    ) -> SearchPipeline:
        """Load everything this config points at, once.

        Missing artifacts are warnings, not errors (module docstring, point 3),
        so this is callable at any point in the project's construction.

        Args:
            encoder: injected text encoder for the dense index. None means load
                the real embedding model named in the config — seconds of
                startup, which is why the API loads it exactly once.
            load_models: skip the Stage 3/4 model loads, for a retrieval-only run.
        """
        from concierge.data.build import read_businesses, read_documents, resolve_processed_dir

        processed = resolve_processed_dir(cfg) if processed_dir is None else Path(processed_dir)
        businesses = read_businesses(processed)
        documents = read_documents(processed)

        # BM25 is cheap to rebuild from the documents we just loaded, so a
        # missing index is repaired rather than reported. The dense index is
        # not cheap (it needs the embedding model over every document), so a
        # missing one degrades the variant instead — see below.
        bm25_index = bm25_mod.load_or_build(documents, cfg, root=artifacts_root)

        dense_index = _load_dense_index(cfg, encoder=encoder, root=artifacts_root)
        rerankers = _load_rerankers(cfg) if load_models else {}
        ltr_ranker = _load_ltr_ranker(cfg) if load_models else None

        return cls(
            cfg,
            businesses={b.business_id: b for b in businesses},
            documents=documents,
            bm25_index=bm25_index,
            dense_index=dense_index,
            reranker=rerankers,
            ltr_ranker=ltr_ranker,
            variant=variant,
            now=now,
        )

    # ── the search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        variant: Variant | None = None,
        k: int | None = None,
    ) -> SearchResponse:
        """Run one search. Returns results, per-stage timings and candidate counts.

        Never raises on an empty or unanswerable query: a query whose
        constraints nothing satisfies returns zero results with the timings and
        counts that explain why, which is exactly what the failure analysis
        reads.
        """
        chosen = self.variant if variant is None else variant
        limit = self.cfg.candidates.final if k is None else k
        rec = StageRecorder()
        started = time.perf_counter()

        with rec.measure(STAGE_PARSE):
            parsed = self._parse(query, chosen)

        if not self._has_searchable_text(parsed):
            # Nothing to retrieve on. Zero is the honest count here, and the
            # caller gets a well-formed response rather than an exception.
            # Checked here as well as inside `candidates()` so this case costs
            # *no* stage timings at all: a response whose only timing is the
            # parse is how the failure analysis recognises an unanswerable
            # query. Both call sites ask the same predicate, so there is still
            # one definition of "there is nothing to search for".
            for stage in CASCADE:
                rec.counts[stage] = 0
            rec.counts[COUNT_FINAL] = 0
            return self._response(parsed, (), rec, started)

        # Stages 2 and 3, in the one place their order is defined.
        candidates = list(self.candidates(parsed, variant=chosen, record=rec))

        with rec.measure(STAGE_LTR):
            candidates = self._rank_ltr(parsed, candidates, chosen)
        rec.counts[STAGE_LTR] = len(candidates)

        with rec.measure(STAGE_ASSEMBLE):
            results = self._assemble(candidates, limit)
        rec.counts[COUNT_FINAL] = len(results)

        return self._response(parsed, results, rec, started)

    def candidates(
        self,
        parsed: ParsedQuery,
        *,
        variant: Variant | None = None,
        limit: int | None = None,
        record: StageRecorder | None = None,
    ) -> tuple[Candidate, ...]:
        """Run the cascade up to and including Stage 3; return the candidates.

        This is the pipeline's candidate-level entry point, and the only place
        the order of retrieve -> filter -> rerank is written down. `search()`
        calls it and then adds Stage 4 and result assembly on top, so a caller
        here gets exactly the stages the served system runs, with every
        per-stage score *and rank* still attached (module docstring, point 4).

        **Stage 4 is deliberately not run, and nothing is rendered.** The caller
        this method exists for is `concierge.ltr.build_examples`, which builds
        the rows Stage 4 is *trained* on: running Stage 4 over them would feed
        the model its own output as a feature, and rendering them would drop the
        ranks it needs. `Stage.LTR` in `variant` is therefore not read here;
        every other flag means exactly what it means in `search()`.

        Args:
            parsed: an already-parsed query. Stage 1 is the caller's, because
                the offline stages parse with the rule parser deliberately
                (`concierge.ltr.build_examples.parse_for_training`) and because
                a `ParsedQuery` is the only input Stages 2 and 3 need.
            variant: which stages to run; the instance default if None.
            limit: keep only this many candidates. None (the default, and what
                `search()` passes) returns everything Stage 3 handed on —
                `search()` truncates after Stage 4, not before it, since Stage 4
                is allowed to promote a candidate Stage 3 ranked last. Applied
                after the counts are recorded, so `record` keeps reporting what
                each stage itself produced rather than what this caller asked to
                see.
            record: where to put the per-stage timings and candidate counts. A
                fresh `StageRecorder` is used when None, i.e. the measurements
                are taken either way and simply discarded.

        Returns an empty tuple — never raises — when the parse left nothing to
        search for, which is the same honest answer `search()` gives.
        """
        chosen = self.variant if variant is None else variant
        rec = StageRecorder() if record is None else record
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0 or None, got {limit}")

        if not self._has_searchable_text(parsed):
            for stage in (STAGE_RETRIEVE, STAGE_FILTER, STAGE_RERANK):
                rec.counts[stage] = 0
            return ()

        with rec.measure(STAGE_RETRIEVE):
            candidates = self._retrieve(parsed, chosen)
        rec.counts[STAGE_RETRIEVE] = len(candidates)

        with rec.measure(STAGE_FILTER):
            candidates = self._filter(parsed, candidates, chosen, rec)
        rec.counts[STAGE_FILTER] = len(candidates)

        with rec.measure(STAGE_RERANK):
            candidates = self._rerank(parsed, candidates, chosen)
        rec.counts[STAGE_RERANK] = len(candidates)

        return tuple(candidates if limit is None else candidates[:limit])

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def _parse(self, query: str, variant: Variant) -> ParsedQuery:
        """Parse with the LLM, or with the rules, depending on the variant.

        With `Stage.QUERY_LLM` off we hand the parser a config whose provider is
        `rules`, rather than calling the rule parser directly. That keeps one
        entry point (`query.parser.parse`) responsible for producing a
        `ParsedQuery` and for stamping `ParseSource`, so the ablation's parser
        attribution comes from the module that owns it.
        """
        cfg = self.cfg if variant.enabled(Stage.QUERY_LLM) else self._rules_cfg
        return query_parser.parse(
            query,
            cfg,
            now=self._now,
            known_categories=self._known_categories or None,
        )

    @staticmethod
    def _has_searchable_text(parsed: ParsedQuery) -> bool:
        return bool(query_text_for(parsed))

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def _retrieve(self, parsed: ParsedQuery, variant: Variant) -> list[Candidate]:
        """Run the enabled retrievers and fuse them.

        A single enabled retriever still goes through `fuse`, which then simply
        preserves its ranking. Same code path, same `Candidate` shape, same
        deterministic tie-break — an ablation row must not differ from the row
        above it in plumbing as well as in substance.
        """
        width = self.cfg.candidates.retrieve
        systems: dict[str, list[tuple[str, float]]] = {}
        text = query_text_for(parsed)

        if variant.enabled(Stage.BM25):
            if self.bm25_index is None:
                self._warn_once("bm25-missing", "no BM25 index loaded; skipping the lexical half")
            else:
                systems[SCORE_BM25] = list(self.bm25_index.search(text, width))

        if variant.enabled(Stage.DENSE):
            if self.dense_index is None:
                self._warn_once(
                    "dense-missing",
                    "no dense index loaded; running lexical-only. Build it with "
                    "`make index` — until then this variant is not the hybrid system.",
                )
            else:
                systems[SCORE_DENSE] = list(self.dense_index.search(text, width))

        if not systems:
            return []
        candidates = fusion.fuse(systems, self.cfg, top_k=width)

        # Stage 2's third path. It only fires when the enabled retrievers
        # under-delivered *and* the query has hard constraints, and its
        # candidates are scored strictly below every fused one, so a query with
        # real lexical or semantic evidence is unaffected. Gated on
        # `Stage.FILTERS` because a no-filters row that received
        # constraint-driven candidates would be measuring constraint handling
        # under another name.
        if variant.enabled(Stage.FILTERS):
            candidates = constraints_mod.backfill(
                candidates, index=self.constraint_index, parsed=parsed, cfg=self.cfg
            )
        return candidates

    def _filter(
        self,
        parsed: ParsedQuery,
        candidates: list[Candidate],
        variant: Variant,
        rec: StageRecorder,
    ) -> list[Candidate]:
        """Evaluate the hard constraints, then apply the removal policy.

        **The constraint checks always run here; `Stage.FILTERS` controls only
        what is done with them.** That split matters more than it looks, and the
        reasoning is worth following because getting it wrong is silent:

        * Stage 4 reads three of its features off `Candidate.constraints`
          (`n_constraints_satisfied`, `frac_constraints_satisfied`,
          `has_unknown_constraint_data`). So the checks have to have happened
          *before* `_rank_ltr`, not after it. Attaching them during result
          assembly — which is where they used to be attached in a no-filters
          run — left LambdaMART reading an empty constraint list: on every
          query of that variant, those three features took the same value for
          every candidate. LightGBM cannot split on a constant column, so all
          three would have vanished from the feature-importance chart — one of
          the project's listed deliverables. The honest-looking
          conclusion ("constraint features turned out unimportant") would have
          been drawn from features that were never populated.
        * It is also a train/serve parity requirement. `concierge.ltr.train`
          builds its rows from recorded `ConstraintMatch`es; serving must hand
          the model the same thing, from the same checker. One switched-off
          stage silently blanking three columns is exactly the feature skew
          `concierge.ltr.features` is organised to prevent.
        * And the "why this result" UI needs the checks in every variant, which
          is what the old arrangement was for. It still gets them — earlier.

        Why *here* and not somewhere cheaper or earlier: constraint evaluation
        is per-candidate arithmetic over an already-loaded `Business` record, so
        its cost is O(candidates Stage 2 produced) — at most
        `cfg.candidates.retrieve` of them, not the whole corpus. Running it
        before retrieval would mean checking every
        business in the metro on every query, which is the one placement that
        would actually be expensive. And it has to be before Stage 3/4, so the
        end of Stage 2 is the only slot left. `apply_filters` already pays this
        exact cost at this exact point when filters are on, so the no-filters
        variant now pays it too and the two are latency-comparable.

        What `Stage.FILTERS` off therefore means, stated precisely because the
        variant's name is broader than its effect: no candidate is *dropped* for
        a violated constraint or missing data, and no fused score is penalised
        (`_retrieve` gates the constraint-only candidate path on the same flag,
        for its own reasons, stated there). The *features* remain populated in
        both arms, because Stage 4
        was trained with them populated and feeding it values it never saw in
        training would make the comparison un-interpretable rather than
        informative. The "how much does Stage 4 get from constraint arithmetic
        rather than text similarity" question is a feature ablation instead —
        `concierge.ltr.features.ablated_feature_names` — which is the experiment
        that can actually answer it.
        """
        if not variant.enabled(Stage.FILTERS):
            return [self._with_constraints(parsed, candidate) for candidate in candidates]

        report = apply_filters(candidates, self.businesses, parsed, self.cfg)
        for reason, count in report.drop_counts.items():
            rec.counts[f"{DROPPED_PREFIX}{reason}"] = count
        if report.n_penalized:
            rec.counts[COUNT_PENALIZED] = report.n_penalized
        return list(report.kept)

    def _with_constraints(self, parsed: ParsedQuery, candidate: Candidate) -> Candidate:
        """A copy of `candidate` carrying this query's constraint checks.

        Calls `evaluate_constraints` — the same function `apply_filters` calls —
        rather than reimplementing any check, so the filters-on and filters-off
        paths cannot disagree about whether a business is open past 10pm.

        A candidate with no `Business` record cannot be checked (or rendered, or
        scored by Stage 4), so it is passed through untouched. It is dropped
        later, by `_assemble`; inventing an empty set of checks for it here
        would be indistinguishable from a query that asked for nothing.
        """
        business = self.businesses.get(candidate.business_id)
        if business is None:
            return candidate
        return Candidate(
            business_id=candidate.business_id,
            scores=dict(candidate.scores),
            ranks=dict(candidate.ranks),
            constraints=evaluate_constraints(business, parsed),
        )

    # ── Stage 3 ──────────────────────────────────────────────────────────────

    def _rerank(
        self, parsed: ParsedQuery, candidates: list[Candidate], variant: Variant
    ) -> list[Candidate]:
        """Hand the whole Stage 2 list to the cross-encoder, keep its top `rerank`.

        Every retrieved candidate is scored, not just the first
        `cfg.candidates.rerank` of them: that is the architecture (score ~200,
        keep ~50) and it is the only version that can *rescue* a candidate
        ranked 180th by retrieval, which is the main thing a reranker is for.

        A disabled (or unavailable) Stage 3 is a pass-through that neither
        truncates nor reorders. Truncating anyway would be harmless for the top
        10 but would misreport the candidate count, and the counts are part of
        the latency analysis.
        """
        if not variant.enabled(Stage.RERANK) or not candidates:
            return candidates

        reranker = self._resolve_reranker(variant)
        if reranker is None:
            return candidates

        return list(reranker.rerank(parsed, candidates, self.doc_text, self.cfg.candidates.rerank))

    def _resolve_reranker(self, variant: Variant) -> Reranker | None:
        """The reranker this variant asks for, or the best available substitute.

        fine-tuned -> pretrained -> none. The fallback is warned about once per
        process, not per query: a per-query warning in a load test would drown
        the log, and a silent one would make a degraded ablation row look like a
        real measurement.
        """
        wanted = variant.reranker
        reranker = self.rerankers.get(wanted)
        if reranker is not None:
            return reranker

        if wanted is RerankCheckpoint.FINETUNED:
            pretrained = self.rerankers.get(RerankCheckpoint.PRETRAINED)
            if pretrained is not None:
                self._warn_once(
                    "finetuned-missing",
                    "no fine-tuned cross-encoder checkpoint; falling back to the "
                    "pretrained model. Results are NOT the fine-tuned system.",
                )
                return pretrained

        self._warn_once(
            f"reranker-missing-{wanted.value}",
            f"no {wanted.value} cross-encoder available; skipping Stage 3 "
            "(results are retrieval order)",
        )
        return None

    # ── Stage 4 ──────────────────────────────────────────────────────────────

    def _rank_ltr(
        self, parsed: ParsedQuery, candidates: list[Candidate], variant: Variant
    ) -> list[Candidate]:
        """Final LambdaMART ordering, or cross-encoder order if it is missing."""
        if not variant.enabled(Stage.LTR) or not candidates:
            return candidates
        if self.ltr_ranker is None:
            self._warn_once(
                "ltr-missing",
                "no LambdaMART model loaded; keeping Stage 3 order. Train it with "
                "`make train-ltr`.",
            )
            return candidates

        # Features need the business record (rating, popularity, coordinates), so
        # a candidate without one cannot be scored. It keeps its place behind the
        # scored ones and carries no SCORE_LTR: a missing score is honest, a zero
        # would be an invented measurement. Only reachable with filters disabled.
        scorable = [c for c in candidates if c.business_id in self.businesses]
        unscorable = [c for c in candidates if c.business_id not in self.businesses]
        if not scorable:
            return candidates

        scores = self.ltr_ranker.score(parsed, scorable, self.businesses)
        if len(scores) != len(scorable):
            raise ValueError(
                f"LTR ranker returned {len(scores)} scores for {len(scorable)} candidates"
            )
        scored = [c.with_score(SCORE_LTR, float(s)) for c, s in zip(scorable, scores, strict=True)]
        # Stable sort on the score alone, so ties keep Stage 3's order — the same
        # tie-break rule the reranker uses, and for the same reason: upstream
        # order is a real signal where `business_id` is noise.
        scored.sort(key=lambda c: -c.scores[SCORE_LTR])
        return scored + unscorable

    # ── Results ──────────────────────────────────────────────────────────────

    def _assemble(self, candidates: Sequence[Candidate], limit: int) -> tuple[SearchResult, ...]:
        """Take the top `limit` candidates and render them for the caller.

        Renders only: the `ConstraintMatch`es shown as "why this result" are
        read off the candidate, never recomputed. Every candidate that reaches
        here has been through `_filter`, which evaluates the checks in every
        variant, so there is one evaluation per (query, candidate) and the
        explanation the user reads is provably the same data Stage 4 scored. A
        second evaluation here would be a second chance to disagree — and it
        would disagree invisibly, since nothing compares the two.
        """
        results: list[SearchResult] = []
        for candidate in candidates:
            if len(results) >= limit:
                break
            business = self.businesses.get(candidate.business_id)
            if business is None:
                # Indexed but not in the business table: unrenderable, so it
                # cannot be a result. The filter stage counts this case when it
                # runs; here it can only happen with filters disabled.
                log.debug("dropping %s from results: no business record", candidate.business_id)
                continue
            results.append(
                SearchResult(
                    business=business,
                    rank=len(results) + 1,
                    score=_final_score(candidate),
                    constraints=candidate.constraints,
                    stage_scores=dict(candidate.scores),
                )
            )
        return tuple(results)

    def _response(
        self,
        parsed: ParsedQuery,
        results: tuple[SearchResult, ...],
        rec: StageRecorder,
        started: float,
    ) -> SearchResponse:
        return SearchResponse(
            parsed=parsed,
            results=results,
            timings=tuple(rec.timings),
            total_ms=(time.perf_counter() - started) * 1000.0,
            n_candidates=dict(rec.counts),
        )

    # ── misc ─────────────────────────────────────────────────────────────────

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        log.warning(message)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _final_score(candidate: Candidate) -> float:
    """The score of the last stage that actually ran.

    Ordered deliberately: a `SearchResult.score` must be the number the result
    was ordered by, or the API would report one ranking and show another. Every
    upstream score is still available in `stage_scores`.
    """
    for key in (SCORE_LTR, SCORE_CROSS_ENCODER, SCORE_RRF):
        if key in candidate.scores:
            return candidate.scores[key]
    return 0.0  # unreachable in practice: fusion always writes SCORE_RRF.


def _as_text_map(documents: Sequence[BusinessDocument] | Mapping[str, str]) -> dict[str, str]:
    if isinstance(documents, Mapping):
        return dict(documents)
    return {doc.business_id: doc.text for doc in documents}


def _as_reranker_map(
    reranker: Reranker | Mapping[RerankCheckpoint, Reranker] | None,
) -> dict[RerankCheckpoint, Reranker]:
    """Accept one scorer or a per-checkpoint mapping.

    One scorer means "use this for whichever checkpoint is asked for", which is
    what a test with a fake scorer wants and what makes those tests short.
    """
    if reranker is None:
        return {}
    if isinstance(reranker, Mapping):
        return dict(reranker)
    return dict.fromkeys(RerankCheckpoint, reranker)


def _load_dense_index(
    cfg: Config, *, encoder: object | None, root: Path | None
) -> Retriever | None:
    """Load the FAISS index, or warn and return None.

    The import is inside the function because `concierge.retrieval.dense` pulls
    in faiss (and, with no injected encoder, sentence-transformers and torch).
    A BM25-only run should not pay that, and neither should an import of this
    module.
    """
    from concierge.retrieval.dense import dense_index_dir, load_dense_index

    directory = dense_index_dir(cfg, root) if root is not None else None
    try:
        return load_dense_index(cfg, encoder=encoder, directory=directory)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        log.warning("%s — the dense half of retrieval is disabled", exc)
        return None


def _load_rerankers(cfg: Config, *, device: str = "cpu") -> dict[RerankCheckpoint, Reranker]:
    """Load both cross-encoder checkpoints, as far as they exist.

    The two are loaded *explicitly by path* rather than through
    `resolve_weights`'s own fallback, because that fallback would hand back the
    pretrained model under the fine-tuned key — and then the ablation's two
    reranker rows would be the same model reported as two different systems.
    Here, a missing fine-tune simply means the FINETUNED key is absent and
    `_resolve_reranker` degrades loudly.
    """
    from concierge.rerank.model import CHECKPOINT_DIRNAME, CrossEncoderReranker, rerank_dir

    paths: dict[RerankCheckpoint, str | Path] = {
        RerankCheckpoint.PRETRAINED: cfg.rerank.base_model,
        RerankCheckpoint.FINETUNED: rerank_dir(cfg) / CHECKPOINT_DIRNAME,
    }
    loaded: dict[RerankCheckpoint, Reranker] = {}
    for checkpoint, path in paths.items():
        if checkpoint is RerankCheckpoint.FINETUNED and not Path(str(path), "config.json").exists():
            log.warning(
                "no fine-tuned cross-encoder at %s; train it with `make train-reranker`", path
            )
            continue
        try:
            loaded[checkpoint] = CrossEncoderReranker.from_config(
                cfg, model_path=path, device=device
            )
        except Exception as exc:  # any load failure degrades, it never crashes a search
            log.warning("could not load the %s cross-encoder: %s", checkpoint.value, exc)
    return loaded


def check_feature_order(saved_names: Sequence[str], names_in_model: Sequence[str]) -> None:
    """Fail unless the two copies of the feature order agree.

    `train.save_ranker` writes the column order twice — inside the model file
    and in `feature_names.json` — precisely so serving can check it. A
    permutation here would produce plausible predictions and no error at all,
    which makes it the most dangerous silent bug available in Stage 4.

    A module-level function rather than a method so it is testable without a
    trained booster.
    """
    if names_in_model and list(names_in_model) != list(saved_names):
        raise ValueError(
            f"feature_names.json and the booster disagree about feature order: "
            f"{list(saved_names)} vs {list(names_in_model)}. "
            "Retrain; serving cannot guess the order."
        )


class LightGbmRanker:
    """Serving adapter for the LambdaMART model `concierge.ltr.train` saved.

    It lives here rather than in `concierge.ltr` because it is wiring, not
    modelling: every feature value comes from `concierge.ltr.features`, which
    stays the single definition of each feature and so of train/serve parity.
    What this class adds is one assertion — that the saved column order is the
    order rows are built in. Permuted columns produce plausible numbers rather
    than an error, which makes it the most dangerous possible silent bug in
    Stage 4.
    """

    def __init__(self, booster: object, feature_names: Sequence[str]) -> None:
        self.booster = booster
        self.feature_names = tuple(feature_names)

    @classmethod
    def load(cls, cfg: Config, directory: Path | None = None) -> LightGbmRanker:
        import json

        import lightgbm as lgb

        from concierge.config import REPO_ROOT
        from concierge.ltr.train import default_out_dir

        if directory is None:
            directory = default_out_dir(cfg, False)
        directory = Path(directory)
        if not directory.is_absolute():
            directory = REPO_ROOT / directory

        model_path = directory / "model.txt"
        names_path = directory / "feature_names.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"no LambdaMART model at {model_path}. Run `make train-ltr` for this config."
            )
        booster = lgb.Booster(model_file=str(model_path))
        names: list[str] = json.loads(names_path.read_text())
        check_feature_order(names, list(booster.feature_name()))
        return cls(booster, names)

    def score(
        self,
        parsed: ParsedQuery,
        candidates: Sequence[Candidate],
        businesses: Mapping[str, Business],
    ) -> list[float]:
        from concierge.ltr.features import feature_matrix

        matrix = feature_matrix(
            ((parsed, c, businesses[c.business_id]) for c in candidates), self.feature_names
        )
        predictions = self.booster.predict(matrix.values)  # type: ignore[attr-defined]
        return [float(value) for value in predictions]


def _load_ltr_ranker(cfg: Config) -> LtrRanker | None:
    """Load the LambdaMART model, or warn and run without Stage 4."""
    try:
        return LightGbmRanker.load(cfg)
    except Exception as exc:  # see _load_rerankers: degrade, never crash
        log.warning("Stage 4 is off: %s", exc)
        return None
