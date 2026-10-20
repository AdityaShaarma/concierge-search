"""Typed configuration loading.

Two reasons this is a pydantic model rather than a raw dict:

1. Typos fail at startup instead of silently defaulting. A config key quietly
   ignored would mean an experiment that didn't actually test what it claimed —
   fatal for a project whose value is its measurements.
2. The config hash identifies artifacts. Every index, model, and eval report is
   written under a directory keyed by `Config.fingerprint()`, so any number in
   the README can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Strict(BaseModel):
    """Base: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    metro: str | None = None
    drop_closed: bool = True


class DocumentsConfig(_Strict):
    max_chars: int = 1800
    max_reviews: int = 4
    max_tips: int = 3
    review_selection: Literal["useful_then_recent", "random", "recent"] = "useful_then_recent"


class QueryParsingConfig(_Strict):
    provider: Literal["gemini", "groq", "rules"] = "gemini"
    model: str = "gemini-2.0-flash"
    timeout_s: float = 4.0
    max_retries: int = 2
    cache_path: Path = Path(".llm_cache/query_parse.sqlite")
    fallback_to_rules: bool = True


class BM25Config(_Strict):
    k1: float = 1.5
    b: float = 0.75


class DenseConfig(_Strict):
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    normalize: bool = True
    batch_size: int = 64


class FaissConfig(_Strict):
    index_type: Literal["IndexFlatIP", "HNSWFlat"] = "IndexFlatIP"


class FusionConfig(_Strict):
    method: Literal["rrf", "weighted_sum"] = "rrf"
    k: int = 60


class RetrievalConfig(_Strict):
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    faiss: FaissConfig = Field(default_factory=FaissConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    missing_attribute_policy: Literal["exclude", "penalize", "neutral"] = "penalize"
    missing_attribute_penalty: float = 0.5


class CandidatesConfig(_Strict):
    retrieve: int = 200
    rerank: int = 50
    final: int = 10
    sweep_retrieve: tuple[int, ...] = (50, 100, 200, 400)
    sweep_rerank: tuple[int, ...] = (10, 25, 50, 100)

    @model_validator(mode="after")
    def _ordering(self) -> CandidatesConfig:
        # A cascade that widens at a later stage is a bug, not a configuration.
        if not self.retrieve >= self.rerank >= self.final:
            raise ValueError(
                f"candidate counts must shrink down the cascade, got "
                f"retrieve={self.retrieve}, rerank={self.rerank}, final={self.final}"
            )
        return self


class RerankConfig(_Strict):
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.1


class LtrConfig(_Strict):
    objective: str = "lambdarank"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 400
    early_stopping_rounds: int = 50
    enforce_oof_crossencoder_feature: bool = True


class SynthSplits(_Strict):
    crossencoder_train: float = 0.45
    crossencoder_val: float = 0.10
    ltr_train: float = 0.35
    ltr_val: float = 0.10

    @model_validator(mode="after")
    def _sums_to_one(self) -> SynthSplits:
        total = self.crossencoder_train + self.crossencoder_val + self.ltr_train + self.ltr_val
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"synth splits must sum to 1.0, got {total}")
        return self


class SynthConfig(_Strict):
    queries_per_business: int = 4
    businesses_sampled: int = 1200
    batch_size: int = 8
    hard_negatives_per_query: int = 8
    false_negative_filter: bool = True
    gold_dedup_cosine_threshold: float = 0.92
    splits: SynthSplits = Field(default_factory=SynthSplits)
    # The weakest grouping a split manifest may use and still be trained on.
    # "cluster" is the pipeline's default and the only one that keeps paraphrases
    # of one query on the same side of the cross-encoder/LambdaMART boundary; the
    # weaker modes exist for comparison runs and must be chosen on purpose, not
    # inherited from a forgotten CLI flag. `concierge.synth.splits.verify_splits`
    # refuses a manifest weaker than this, so the out-of-fold guarantee stops
    # depending on an argparse default. See `GroupBy` in that module.
    min_split_grouping: Literal["query", "business", "cluster"] = "cluster"


class EvaluationConfig(_Strict):
    primary_metric: str = "ndcg@10"
    cutoffs: tuple[int, ...] = (5, 10)
    recall_cutoff: int = 100
    pool_depth: int = 20
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    label_subset_size: int = 50
    gold_queries: Path = Path("eval/gold_queries.csv")
    qrels: Path = Path("eval/qrels.csv")
    out_dir: Path = Path("artifacts/eval")


class ServingConfig(_Strict):
    host: str = "0.0.0.0"
    port: int = 8000
    use_onnx: bool = False


class Config(_Strict):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    query_parsing: QueryParsingConfig = Field(default_factory=QueryParsingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    ltr: LtrConfig = Field(default_factory=LtrConfig)
    synth: SynthConfig = Field(default_factory=SynthConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    # ── helpers ──────────────────────────────────────────────────────────────

    def fingerprint(self, length: int = 10) -> str:
        """Short stable hash of the **whole** config.

        Still the right key for anything that describes a whole run — the eval
        report, the data build manifest — because a reader tracing a number back
        wants every setting, including the ones that could not have changed it.

        It is the wrong key for a single artifact, and
        :meth:`fingerprint_for` exists for that. See its docstring.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:length]

    def fingerprint_for(self, *sections: str, length: int = 10) -> str:
        """Short stable hash of only the named config sections.

        ``sections`` are dotted paths into the config, e.g. ``"documents"`` or
        ``"retrieval.bm25"``. Order does not matter (the payload is a sorted
        mapping), and an unknown path is an error rather than an empty
        contribution — a typo'd section would silently produce a key that
        ignores the thing it was meant to include.

        **Why artifacts are not keyed on the whole config.** They were, and the
        consequence is easy to state and hard to defend: changing
        ``serving.port`` from 8000 to 8001 moves ``Config.fingerprint()`` from
        ``dc1110dcfa`` to ``0f7aabbb93``, so every artifact directory moves with
        it. The FAISS index, the BM25 index and both trained checkpoints are
        suddenly "missing"; the serving path *degrades* rather than refusing
        (``_load_dense_index`` catches ``FileNotFoundError`` and drops to
        lexical-only with a log warning), and ``make train-ltr`` refuses outright
        on a fingerprint mismatch. "Change the HTTP port, lose your FAISS index
        and your Colab fine-tune" is not an answer the owner should have to give.

        The tradeoff, stated because it is the whole of the decision and it cuts
        both ways:

        * a **coarse** key (the whole config) can never serve a stale artifact,
          because any change at all invalidates everything — at the cost of
          invalidating artifacts that could not possibly have changed;
        * a **fine** key is never spuriously orphaned, at the cost of being wrong
          if a section an artifact really depends on is left out of its list.

        So the lists below are deliberately *generous*: each one names every
        section that could plausibly alter the artifact, and errs towards
        including one too many. Being over-keyed costs a rebuild; being
        under-keyed serves a wrong artifact silently, which is the failure this
        project cannot afford. Recorded as DECISIONS.md D237.
        """
        dumped = self.model_dump(mode="json")
        payload: dict[str, Any] = {}
        for section in sections:
            node: Any = dumped
            for part in section.split("."):
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(
                        f"fingerprint_for: {section!r} is not a config path "
                        f"(failed at {part!r}). A typo here would key an artifact on "
                        "fewer settings than it depends on."
                    )
                node = node[part]
            payload[section] = node
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:length]

    def require_metro(self) -> str:
        """Fail loudly rather than indexing an unintended scope.

        `metro` is deliberately null in the committed config: the owner picks it
        at the Phase 1 checkpoint after seeing the exploration output.
        """
        if not self.data.metro:
            raise ValueError(
                "config.data.metro is not set. Run `make explore` and set it in "
                "configs/base.yaml before building data or indexes."
            )
        return self.data.metro


# ─────────────────────────────────────────────────────────────────────────────
# Which config sections each artifact actually depends on
#
# One list per artifact kind, used with `Config.fingerprint_for`. Defined here,
# beside the config they index into, so the question "what does this artifact
# depend on?" has one answer that every producer and every consumer reads — a
# producer and a consumer disagreeing about the key would look exactly like a
# missing artifact.
#
# `seed` and `data` are in every list: the seed reaches every sampling decision,
# and `data` decides which corpus is in scope at all (metro, drop_closed), which
# changes every artifact downstream of it.
# ─────────────────────────────────────────────────────────────────────────────

#: The BM25 index: the documents it indexes, and BM25's own two parameters.
#: Not `retrieval` whole — `retrieval.fusion` and `retrieval.filters` change
#: how results are *combined and filtered*, which happens per query and cannot
#: change the inverted index on disk.
BM25_SECTIONS = ("seed", "data", "documents", "retrieval.bm25")

#: The dense index: the documents, the embedding model and prefixes, and the
#: FAISS index type.
DENSE_SECTIONS = ("seed", "data", "documents", "retrieval.dense", "retrieval.faiss")

#: The synthetic training data (queries, negatives, judge verdicts, splits).
#: `retrieval` in full, because hard negatives are mined from the real hybrid
#: retriever and so depend on fusion and filtering as well as the indexes;
#: `candidates` because the mining depth is read from it.
SYNTH_SECTIONS = (
    "seed",
    "data",
    "documents",
    "retrieval",
    "candidates",
    "synth",
    "query_parsing.provider",
    "query_parsing.model",
)

#: The fine-tuned cross-encoder: its own hyperparameters, plus the synthetic
#: data it trains on (hence everything in SYNTH_SECTIONS).
RERANK_SECTIONS = (*SYNTH_SECTIONS, "rerank")

#: The LambdaMART model: its own hyperparameters, plus the cross-encoder whose
#: score is one of its features, plus the data behind both.
LTR_SECTIONS = (*RERANK_SECTIONS, "ltr")


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    """Load and validate a config file."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    return Config.model_validate(raw)
