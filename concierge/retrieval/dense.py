"""Stage 2, semantic half: sentence-transformers embeddings behind a FAISS index.

BM25 matches words; this matches meaning. "somewhere quiet to read" shares almost
no vocabulary with a bookstore-café's document, so the lexical half of the hybrid
cannot find it. The two are fused by rank (RRF) rather than by score, which is why
this module exposes exactly the same ``search(query_text, top_k) -> [(id, score)]``
shape as ``bm25.py``: the fusion code should not need to know which retriever it
is talking to.

Three things in here are easy to get silently wrong, so each is called out in a
comment and pinned by a test:

1. **The bge prefix asymmetry.** ``BAAI/bge-small-en-v1.5`` was trained
   asymmetrically: queries carry an instruction prefix, passages carry nothing.
   Prefixing both, or prefixing the passages instead, still produces plausible
   vectors and plausible-looking results — it just retrieves measurably worse.
   There is no exception that can catch that, so the asymmetry lives in one small
   class (``DenseEmbedder``) with a test that inspects exactly what text reached
   the model.

2. **Cosine vs. inner product.** We L2-normalize every vector and use
   ``IndexFlatIP``. For unit vectors, ``a·b == cos(a, b)`` because
   ``a·b = |a||b|cos(a,b)`` and both magnitudes are 1 — so an inner-product index
   *is* a cosine index, without FAISS needing a cosine metric. Skip the
   normalization and the same index silently ranks long documents higher, because
   their vector magnitudes inflate the dot product.

3. **The id mapping.** FAISS returns integer *row positions*, never business ids.
   The position-to-id list must be saved and loaded alongside the index. An
   off-by-one there corrupts every single result while the system keeps looking
   healthy — every query still returns ten businesses, they are just the wrong
   ten. Hence the length check on load and the round-trip test.

The embedder is injectable (``TextEncoder``) so the whole FAISS wiring can be
tested with a tiny deterministic fake instead of a 130 MB model download.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

from concierge.config import REPO_ROOT, Config
from concierge.types import BusinessDocument

# FAISS accepts float32 only. Converting once at the boundary avoids a hidden
# per-query copy (and the confusing error you get when you forget).
EMBEDDING_DTYPE = np.float32

_INDEX_FILENAME = "index.faiss"
_IDS_FILENAME = "ids.json"
_META_FILENAME = "meta.json"

# HNSW graph degree. Not in configs/base.yaml, which only exposes `index_type`;
# 32 is the common faiss default-ish value and is a starting point, not a tuned
# result. Exact search is the default precisely so no quality number this project
# reports carries an unmeasured approximation error.
_HNSW_M = 32
# Candidate-list width at query time. Larger = slower and more accurate. Also
# untuned.
_HNSW_EF_SEARCH = 64
_HNSW_EF_CONSTRUCTION = 200


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class TextEncoder(Protocol):
    """Raw text -> vectors.

    Deliberately ignorant of the query/passage distinction. Keeping the bge
    prefix rule one layer up (in `DenseEmbedder`) means a fake encoder that just
    records the strings it was handed is enough to test the rule, so the test
    suite never downloads a model.

    An encoder MAY also expose a ``model_name: str`` attribute naming the
    checkpoint it loaded, as `SentenceTransformerEncoder` does. It is not part
    of this Protocol because Protocol members cannot be optional, and requiring
    it would force every test fake to invent a model name it does not have.
    `DenseIndex.load` reads it when it is there: that is the only check that can
    catch "right width, wrong model" (see `encoder_model_name`).
    """

    @property
    def dim(self) -> int:
        """Embedding width, needed to size the FAISS index before any text exists."""
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one row per input text, shape (len(texts), dim)."""
        ...


def encoder_model_name(encoder: TextEncoder) -> str | None:
    """The checkpoint an encoder loaded, or None if it cannot say.

    One function rather than an inline `getattr` so there is a single place to
    read about why the attribute is optional: a real encoder always knows its
    model name, but an injected test fake has no honest answer, and refusing to
    load an index without one would make every test invent a fiction. None
    therefore means "unverifiable", and `DenseIndex.load` treats it as such.
    """
    name = getattr(encoder, "model_name", None)
    return name if isinstance(name, str) and name else None


class SentenceTransformerEncoder:
    """The real encoder. Loads the model named in `cfg.retrieval.dense.model`."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        # Imported lazily: `import sentence_transformers` drags in torch, which
        # costs seconds of startup. Nothing but a real index build should pay it,
        # and the test suite must never pay it at all.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        # sentence-transformers 6.x renamed get_sentence_embedding_dimension to
        # get_embedding_dimension. Checking both keeps the index buildable across
        # the version boundary, which matters because the lockfile will outlive
        # this particular minor release.
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        dim = getter()
        if dim is None:  # pragma: no cover - defensive; all bge models report it
            raise RuntimeError(f"{model_name} did not report an embedding dimension")
        self._dim = int(dim)
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # normalize_embeddings=False on purpose: normalization is our invariant
        # (see module docstring point 2) and we want it asserted in our own test
        # rather than trusted to a library flag that could change defaults.
        vectors = self._model.encode(
            list(texts),
            batch_size=max(1, len(texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=EMBEDDING_DTYPE)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding with the query/passage asymmetry
# ─────────────────────────────────────────────────────────────────────────────


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, so inner product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction. Dividing would give NaN, and a single NaN
    # row propagates into FAISS and makes an entire result list nonsense. Leaving
    # it at zero instead means it scores 0.0 against every query: still wrong for
    # that one document, but harmlessly so.
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(EMBEDDING_DTYPE, copy=False)


@dataclass(frozen=True)
class DenseEmbedder:
    """Applies the bge query/passage convention, batching, and normalization.

    This is the only place in the codebase that knows a query is spelled
    differently from a passage.
    """

    encoder: TextEncoder
    query_prefix: str = ""
    normalize: bool = True
    batch_size: int = 64

    @classmethod
    def from_config(cls, cfg: Config, encoder: TextEncoder | None = None) -> DenseEmbedder:
        """Build from config, downloading the model only if no encoder is injected."""
        dense_cfg = cfg.retrieval.dense
        return cls(
            encoder=encoder if encoder is not None else SentenceTransformerEncoder(dense_cfg.model),
            query_prefix=dense_cfg.query_prefix,
            normalize=dense_cfg.normalize,
            batch_size=dense_cfg.batch_size,
        )

    @property
    def dim(self) -> int:
        return self.encoder.dim

    def embed_passages(self, texts: Sequence[str], *, progress: bool = False) -> np.ndarray:
        """Embed business documents. NO prefix — this is the asymmetric half.

        bge's training objective paired a *prefixed* query with a *bare* passage.
        Adding the prefix here would move the document vectors into the region of
        the space the model reserves for queries, shrinking the very gap the
        retrieval relies on.
        """
        return self._encode_batched(list(texts), progress=progress, label="documents")

    def embed_queries(self, texts: Sequence[str], *, progress: bool = False) -> np.ndarray:
        """Embed user queries, WITH the instruction prefix from config."""
        prefixed = [f"{self.query_prefix}{text}" for text in texts]
        return self._encode_batched(prefixed, progress=progress, label="queries")

    def _encode_batched(self, texts: list[str], *, progress: bool, label: str) -> np.ndarray:
        """Encode in `batch_size` chunks, normalizing the result.

        Chunking here rather than inside the encoder keeps batching behaviour
        identical for the real model and the test fake, so a batching bug cannot
        hide behind "well, the fake does it differently".
        """
        if not texts:
            # An empty (0, dim) array keeps every caller's shape arithmetic valid
            # instead of making them special-case the empty corpus.
            return np.zeros((0, self.dim), dtype=EMBEDDING_DTYPE)

        chunks: list[np.ndarray] = []
        started = time.perf_counter()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors = np.asarray(self.encoder.encode(batch), dtype=EMBEDDING_DTYPE)
            if vectors.shape != (len(batch), self.dim):
                raise ValueError(
                    f"encoder returned {vectors.shape}, expected {(len(batch), self.dim)}"
                )
            chunks.append(vectors)
            if progress:
                _report_progress(len(chunks[-1]) + start, len(texts), label, started)

        matrix = np.vstack(chunks)
        return l2_normalize(matrix) if self.normalize else matrix


def _report_progress(done: int, total: int, label: str, started: float) -> None:
    """Single-line progress to stderr.

    stderr and not a progress-bar library so that `make index > log.txt` keeps a
    readable log and the test suite can capture it. Only counts and elapsed
    seconds are printed — no throughput extrapolation, because a half-finished
    rate is not a measurement.
    """
    elapsed = time.perf_counter() - started
    print(
        f"\r  embedding {label}: {done}/{total} ({elapsed:6.1f}s elapsed)",
        end="" if done < total else "\n",
        file=sys.stderr,
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The index
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DenseIndexMeta:
    """Saved next to the index so a mismatched load fails loudly.

    Loading an index built by a different embedding model is the kind of mistake
    that produces garbage rankings with no error at all — the dimensions can even
    match, since many small models are 384-dimensional. `DenseIndex.load`
    enforces `model`, `dim` and `query_prefix` against the embedder about to
    query the index; the remaining fields are provenance, recorded so an
    artifact can be traced back to the settings that built it.
    """

    model: str
    index_type: str
    dim: int
    n_documents: int
    normalize: bool
    query_prefix: str
    config_fingerprint: str


def dense_index_dir(cfg: Config, root: Path | None = None) -> Path:
    """Artifact directory for this exact config: ``artifacts/dense/<fingerprint>``.

    Keyed by `cfg.fingerprint()` so changing the model, the prefix, or the index
    type cannot quietly reuse a stale index — the project's reproducibility claim
    depends on every artifact being traceable to the settings that built it.

    Layout mirrors `bm25.index_dir` (``artifacts/bm25/<fingerprint>``) so the two
    halves of the hybrid retriever are findable the same way.
    """
    base = root if root is not None else REPO_ROOT / "artifacts"
    return Path(base) / "dense" / cfg.fingerprint()


def _make_faiss_index(dim: int, index_type: str) -> faiss.Index:
    """Create an empty index of the configured type.

    Both options use inner product, which on normalized vectors is cosine
    similarity (module docstring, point 2).
    """
    if index_type == "IndexFlatIP":
        # Exact brute-force search: every query is compared against every vector.
        # O(n) per query, but at ~10^4-10^5 businesses that is milliseconds, and
        # it means the retrieval quality numbers contain zero approximation error.
        return faiss.IndexFlatIP(dim)
    if index_type == "HNSWFlat":
        # Approximate graph search. Sub-linear queries, at the cost of a recall
        # loss that must be *measured* against the exact index before it is
        # accepted — never assumed to be negligible.
        index = faiss.IndexHNSWFlat(dim, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = _HNSW_EF_SEARCH
        return index
    raise ValueError(f"unsupported faiss index_type: {index_type!r}")


class DenseIndex:
    """A FAISS index plus the position -> business_id mapping it cannot hold itself."""

    def __init__(
        self,
        index: faiss.Index,
        business_ids: Sequence[str],
        embedder: DenseEmbedder,
        meta: DenseIndexMeta,
    ) -> None:
        if index.ntotal != len(business_ids):
            raise ValueError(
                f"id mapping has {len(business_ids)} entries but the index holds "
                f"{index.ntotal} vectors — results would be mislabeled"
            )
        self._index = index
        self._business_ids = list(business_ids)
        self._embedder = embedder
        self.meta = meta

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        documents: Sequence[BusinessDocument],
        embedder: DenseEmbedder,
        cfg: Config,
        *,
        progress: bool = False,
    ) -> DenseIndex:
        """Embed every document and load the vectors into a fresh index.

        Document order defines the id mapping; nothing downstream may reorder it.
        """
        if not documents:
            raise ValueError("cannot build a dense index over zero documents")

        business_ids = [doc.business_id for doc in documents]
        duplicates = len(business_ids) - len(set(business_ids))
        if duplicates:
            # A duplicate id means two rows claim the same business, so one of
            # them can never be retrieved and the other may be double-counted in
            # the eval. Cheap to check, miserable to debug later.
            raise ValueError(f"documents contain {duplicates} duplicate business_id(s)")

        vectors = embedder.embed_passages([doc.text for doc in documents], progress=progress)
        index = _make_faiss_index(embedder.dim, cfg.retrieval.faiss.index_type)
        # Neither IndexFlatIP nor IndexHNSWFlat needs training (no learned
        # quantizer), so `add` is the whole build. An IVF index would need
        # `train` first — worth knowing if this ever moves to IVF for scale.
        index.add(vectors)

        meta = DenseIndexMeta(
            model=cfg.retrieval.dense.model,
            index_type=cfg.retrieval.faiss.index_type,
            dim=embedder.dim,
            n_documents=len(documents),
            normalize=embedder.normalize,
            query_prefix=embedder.query_prefix,
            config_fingerprint=cfg.fingerprint(),
        )
        return cls(index, business_ids, embedder, meta)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> Path:
        """Write index, id mapping, and metadata. Returns the directory."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _INDEX_FILENAME))
        (directory / _IDS_FILENAME).write_text(json.dumps(self._business_ids))
        (directory / _META_FILENAME).write_text(json.dumps(asdict(self.meta), indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path, embedder: DenseEmbedder) -> DenseIndex:
        """Load a saved index, checking it matches the embedder about to query it."""
        index = faiss.read_index(str(directory / _INDEX_FILENAME))
        business_ids: list[str] = json.loads((directory / _IDS_FILENAME).read_text())
        meta = DenseIndexMeta(**json.loads((directory / _META_FILENAME).read_text()))

        # The index stores vectors, not the model that made them. Without these
        # checks, querying a bge index with a differently-trained encoder of the
        # same width returns confidently-ranked nonsense.
        #
        # Width first, because a mismatch there is proof of the wrong model and
        # needs no cooperation from the encoder.
        if meta.dim != embedder.dim:
            raise ValueError(
                f"index was built at dim {meta.dim} but the embedder produces "
                f"{embedder.dim} — wrong model for this index"
            )
        # Then identity, which is the check this class was built around: equal
        # width is NOT proof of the right model, because many small sentence
        # encoders are 384-dimensional. Every vector would still be valid and
        # every query would still return results — ranked in an unrelated
        # embedding space, with nothing raising. Skipped only when the encoder
        # cannot name itself (an injected fake); see `encoder_model_name`.
        name = encoder_model_name(embedder.encoder)
        if name is not None and name != meta.model:
            raise ValueError(
                f"index was built with model {meta.model!r} but the embedder "
                f"loaded {name!r} — wrong model for this index. The vectors "
                "would be compared across two unrelated embedding spaces, which "
                "produces plausible rankings and no error. Rebuild the index "
                "(`make index`) or point the config back at the original model."
            )
        if meta.query_prefix != embedder.query_prefix:
            raise ValueError(
                "query_prefix differs between the saved index and the embedder: "
                f"{meta.query_prefix!r} vs {embedder.query_prefix!r}. Queries would "
                "sit in a different region of the space than the documents."
            )
        return cls(index, business_ids, embedder, meta)

    # ── search ───────────────────────────────────────────────────────────────

    @property
    def n_documents(self) -> int:
        return len(self._business_ids)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Top-k businesses by cosine similarity, best first.

        Signature deliberately identical to the BM25 retriever's so Reciprocal
        Rank Fusion can treat the two interchangeably.
        """
        return self.search_batch([query_text], top_k)[0]

    def search_batch(self, query_texts: Sequence[str], top_k: int) -> list[list[tuple[str, float]]]:
        """Batched variant. One encoder call for many queries, which matters when
        the eval harness runs ~150 gold queries end to end."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query_texts:
            return []

        query_vectors = self._embedder.embed_queries(query_texts)
        # Asking for more neighbours than exist is legal in FAISS but pads the
        # result with id -1 and score -inf; clamping keeps those out of the loop.
        k = min(top_k, self.n_documents)
        scores, positions = self._index.search(query_vectors, k)

        results: list[list[tuple[str, float]]] = []
        for row_scores, row_positions in zip(scores, positions, strict=True):
            hits: list[tuple[str, float]] = []
            for position, score in zip(row_positions, row_scores, strict=True):
                if position < 0:
                    # FAISS's "no more neighbours" sentinel. Reachable with HNSW
                    # even when k <= ntotal, if the graph walk finds too few.
                    continue
                hits.append((self._business_ids[int(position)], float(score)))
            results.append(hits)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry points
# ─────────────────────────────────────────────────────────────────────────────


def build_and_save(
    documents: Sequence[BusinessDocument],
    cfg: Config,
    *,
    encoder: TextEncoder | None = None,
    progress: bool = True,
    out_dir: Path | None = None,
) -> tuple[DenseIndex, Path]:
    """Build the dense index and persist it under the config-keyed artifact dir."""
    embedder = DenseEmbedder.from_config(cfg, encoder=encoder)
    index = DenseIndex.build(documents, embedder, cfg, progress=progress)
    directory = out_dir if out_dir is not None else dense_index_dir(cfg)
    index.save(directory)
    return index, directory


def load_dense_index(
    cfg: Config, *, encoder: TextEncoder | None = None, directory: Path | None = None
) -> DenseIndex:
    """Load the dense index for this config, ready to serve queries."""
    target = directory if directory is not None else dense_index_dir(cfg)
    if not (target / _INDEX_FILENAME).exists():
        raise FileNotFoundError(
            f"no dense index at {target}. Run `make index` for this config first."
        )
    embedder = DenseEmbedder.from_config(cfg, encoder=encoder)
    return DenseIndex.load(target, embedder)
