"""Tests for the dense (semantic) half of Stage 2.

None of these download a model. `bge-small` is ~130 MB and takes seconds to load,
which would make the suite too slow to run on every edit — and more importantly,
a test that needs the network cannot pin a *logic* bug. So the encoder is
injected, and two tiny deterministic fakes stand in for it:

  `RecordingEncoder`  hash-based vectors; records every string it was handed, so
                      the bge query/passage prefix rule can be asserted directly
                      on the text that reached the model.
  `VocabEncoder`      counts occurrences of a fixed eight-word vocabulary, so
                      similarity is *interpretable* and a ranking assertion means
                      something ("quiet coffee" must rank the quiet café first)
                      rather than just checking plumbing.

The one test that genuinely downloads bge-small is marked `slow`.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import faiss
import numpy as np
import pytest

from concierge.config import BM25_SECTIONS, DENSE_SECTIONS, Config, FaissConfig
from concierge.retrieval import bm25, build_index
from concierge.retrieval.dense import (
    DenseEmbedder,
    DenseIndex,
    SentenceTransformerEncoder,
    _make_faiss_index,
    dense_index_dir,
    encoder_model_name,
    l2_normalize,
    load_dense_index,
)
from concierge.types import BusinessDocument

# ─────────────────────────────────────────────────────────────────────────────
# Fake encoders
# ─────────────────────────────────────────────────────────────────────────────


class RecordingEncoder:
    """Deterministic hash-based vectors, plus a log of what it was asked to encode."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.seen: list[str] = []
        self.batch_sizes: list[int] = []

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        self.seen.extend(texts)
        self.batch_sizes.append(len(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Map the first `dim` digest bytes into [-1, 1]. Arbitrary but stable,
            # which is all a wiring test needs.
            rows.append([(byte / 127.5) - 1.0 for byte in digest[: self._dim]])
        return np.asarray(rows, dtype=np.float32)


VOCAB = ("quiet", "coffee", "ramen", "loud", "wifi", "patio", "books", "deli")


class VocabEncoder:
    """Bag-of-words counts over a fixed eight-word vocabulary.

    Interpretable on purpose: two texts sharing vocabulary words get a high
    cosine similarity, so a top-1 assertion tests retrieval behaviour and not
    just that FAISS returned something. The bge prefix contains none of these
    words, so prefixing a query leaves its vector unchanged — which is exactly
    the property that lets the same fake test ranking *and* coexist with the
    prefix rule.
    """

    @property
    def dim(self) -> int:
        return len(VOCAB)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float(lowered.count(word)) for word in VOCAB])
        return np.asarray(rows, dtype=np.float32)


class NamedVocabEncoder(VocabEncoder):
    """A `VocabEncoder` that also reports which checkpoint it "loaded".

    The real `SentenceTransformerEncoder` exposes `model_name`; the other fakes
    here do not, because a fake has no honest answer. This one opts in so the
    model-identity check on load can be tested without downloading anything:
    two instances with different names, and identical widths, stand in for two
    different same-width encoders — the case the check exists for, and the one
    case no other check can catch.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class ZeroEncoder:
    """Returns the zero vector for everything. Guards the divide-by-zero path."""

    @property
    def dim(self) -> int:
        return 4

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(list(texts)), 4), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

PREFIX = "Represent this sentence for searching relevant passages: "


@pytest.fixture
def vocab_documents() -> list[BusinessDocument]:
    """A hand-written corpus whose vocabulary overlaps `VOCAB` in known ways."""
    return [
        BusinessDocument(business_id="b001", text="Lantern Coffee quiet coffee wifi"),
        BusinessDocument(business_id="b002", text="Midnight Ramen loud ramen"),
        BusinessDocument(business_id="b003", text="The Quiet Shelf quiet books wifi"),
        BusinessDocument(business_id="b006", text="Patio Taqueria patio"),
        BusinessDocument(business_id="b007", text="Always Open Deli deli"),
    ]


@pytest.fixture
def fixture_documents(raw_businesses: list[dict]) -> list[BusinessDocument]:
    """Documents derived from the real-shaped Yelp fixtures.

    Closed businesses are dropped here the way the data stage drops them, so the
    index under test holds the same population the live system would.
    """
    return [
        BusinessDocument(
            business_id=b["business_id"],
            text=f"{b['name']}. {b['categories']}",
        )
        for b in raw_businesses
        if b["is_open"] == 1
    ]


def _hnsw_config(config: Config) -> Config:
    """Same config, approximate index. Written as a copy because Config is frozen."""
    return config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(
                update={"faiss": FaissConfig(index_type="HNSWFlat")}
            )
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. The bge query/passage prefix asymmetry
# ─────────────────────────────────────────────────────────────────────────────


def test_query_prefix_applied_to_queries_only(config):
    """bge was trained with a prefixed query against a BARE passage.

    This is the module's highest-risk bug: applying the prefix to both sides, or
    to the passages instead, produces perfectly valid vectors and a system that
    just retrieves worse. Nothing raises. So assert on the literal strings the
    model received.
    """
    encoder = RecordingEncoder()
    embedder = DenseEmbedder.from_config(config, encoder=encoder)
    assert embedder.query_prefix == PREFIX, "config lost the bge prefix"

    embedder.embed_passages(["quiet cafe with wifi"])
    assert encoder.seen == ["quiet cafe with wifi"]

    encoder.seen.clear()
    embedder.embed_queries(["quiet cafe with wifi"])
    assert encoder.seen == [PREFIX + "quiet cafe with wifi"]


def test_prefix_is_not_applied_twice_to_queries(config):
    """A second prefix would be as wrong as none; guards against a helper that
    prefixes in both `embed_queries` and the encoder."""
    encoder = RecordingEncoder()
    embedder = DenseEmbedder.from_config(config, encoder=encoder)
    embedder.embed_queries(["ramen"])
    assert encoder.seen[0].count(PREFIX) == 1


def test_prefixed_query_and_bare_passage_differ_in_vector_space(config):
    """Sanity check that the prefix is not cosmetic: with a prefix-sensitive
    encoder the same text embeds differently as a query and as a passage, which
    is precisely why mixing the two conventions degrades retrieval."""
    embedder = DenseEmbedder.from_config(config, encoder=RecordingEncoder())
    as_passage = embedder.embed_passages(["quiet cafe"])
    as_query = embedder.embed_queries(["quiet cafe"])
    assert not np.allclose(as_passage, as_query)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normalization, so inner product == cosine
# ─────────────────────────────────────────────────────────────────────────────


def test_embeddings_are_unit_length(config, vocab_documents):
    """IndexFlatIP computes a dot product. It equals cosine similarity only for
    unit vectors (a·b = |a||b|cos θ). Without this, longer documents with larger
    magnitudes outrank better matches."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    vectors = embedder.embed_passages([d.text for d in vocab_documents])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_inner_product_of_identical_text_is_one(config):
    """The concrete form of the cosine equivalence: a vector against itself."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    vectors = embedder.embed_passages(["quiet coffee", "quiet coffee"])
    assert float(vectors[0] @ vectors[1]) == pytest.approx(1.0, abs=1e-6)


def test_normalize_disabled_leaves_magnitudes_alone(config):
    """`normalize: false` is a real config option; check it is actually honoured
    rather than hard-coded on, so the knob can be ablated."""
    cfg = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(
                update={"dense": config.retrieval.dense.model_copy(update={"normalize": False})}
            )
        }
    )
    embedder = DenseEmbedder.from_config(cfg, encoder=VocabEncoder())
    vectors = embedder.embed_passages(["quiet quiet quiet coffee"])
    assert np.linalg.norm(vectors[0]) > 1.0


def test_zero_vector_does_not_become_nan(config):
    """A document with no signal must score 0.0 against every query, not NaN —
    one NaN row propagates through FAISS and ruins whole result lists."""
    embedder = DenseEmbedder.from_config(config, encoder=ZeroEncoder())
    vectors = embedder.embed_passages(["anything"])
    assert not np.isnan(vectors).any()
    assert np.allclose(vectors, 0.0)


def test_l2_normalize_handles_mixed_rows():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(matrix)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Batching and progress
# ─────────────────────────────────────────────────────────────────────────────


def test_encoding_respects_configured_batch_size(config):
    """The build encodes ~10^5 documents; the batch size is the knob that keeps
    that inside laptop memory, so verify it is actually used."""
    cfg = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(
                update={"dense": config.retrieval.dense.model_copy(update={"batch_size": 3})}
            )
        }
    )
    encoder = RecordingEncoder()
    embedder = DenseEmbedder.from_config(cfg, encoder=encoder)
    embedder.embed_passages([f"doc {i}" for i in range(8)])
    assert encoder.batch_sizes == [3, 3, 2]


def test_empty_input_returns_correctly_shaped_array(config):
    """(0, dim) rather than an error, so callers need no empty-corpus branch."""
    embedder = DenseEmbedder.from_config(config, encoder=RecordingEncoder())
    vectors = embedder.embed_passages([])
    assert vectors.shape == (0, 8)


def test_progress_goes_to_stderr(config, vocab_documents, capsys):
    """Progress must not pollute stdout, which carries the build report."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    embedder.embed_passages([d.text for d in vocab_documents], progress=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "embedding documents" in captured.err


# ─────────────────────────────────────────────────────────────────────────────
# 4. Index build and search behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_search_ranks_the_semantically_closest_document_first(config, vocab_documents):
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)

    hits = index.search("quiet coffee wifi", top_k=3)
    assert hits[0][0] == "b001"  # Lantern Coffee: quiet + coffee + wifi
    # Scores must be non-increasing; fusion assumes rank order matches score order.
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_cosine_scores_in_range(config, vocab_documents):
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)
    for _, score in index.search("quiet books", top_k=5):
        assert -1.0 - 1e-5 <= score <= 1.0 + 1e-5


def test_search_signature_matches_the_bm25_retriever(config, vocab_documents):
    """`list[tuple[str, float]]`, best first. RRF consumes both retrievers through
    this one shape, so a drift here breaks fusion rather than failing a type check."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)
    hits = index.search("ramen", top_k=2)
    assert isinstance(hits, list)
    assert all(isinstance(bid, str) and isinstance(score, float) for bid, score in hits)


def test_top_k_larger_than_corpus_is_clamped(config, vocab_documents):
    """FAISS pads short result lists with id -1 and score -inf. Those must never
    be returned — a business_id of "-1" would crash lookup, and -inf would poison
    any score aggregation."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)
    hits = index.search("quiet", top_k=500)
    assert len(hits) == len(vocab_documents)
    assert all(bid in {d.business_id for d in vocab_documents} for bid, _ in hits)


def test_search_batch_matches_single_search(config, vocab_documents):
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)
    queries = ["quiet coffee", "loud ramen"]
    batched = index.search_batch(queries, top_k=3)
    assert batched == [index.search(q, top_k=3) for q in queries]


def test_search_rejects_nonpositive_top_k(config, vocab_documents):
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)
    with pytest.raises(ValueError, match="top_k"):
        index.search("quiet", top_k=0)


def test_build_rejects_empty_corpus(config):
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    with pytest.raises(ValueError, match="zero documents"):
        DenseIndex.build([], embedder, config)


def test_build_rejects_duplicate_business_ids(config):
    """Two rows claiming one business means one is unreachable and the other may
    be double-counted in the eval pool."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    docs = [
        BusinessDocument(business_id="b001", text="quiet coffee"),
        BusinessDocument(business_id="b001", text="loud ramen"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        DenseIndex.build(docs, embedder, config)


# ─────────────────────────────────────────────────────────────────────────────
# 5. The id mapping — FAISS returns positions, not business ids
# ─────────────────────────────────────────────────────────────────────────────


def test_id_mapping_preserves_document_order(config, fixture_documents):
    embedder = DenseEmbedder.from_config(config, encoder=RecordingEncoder())
    index = DenseIndex.build(fixture_documents, embedder, config)
    assert index._business_ids == [d.business_id for d in fixture_documents]
    assert index.n_documents == len(fixture_documents)


def test_id_mapping_length_mismatch_is_rejected(config, fixture_documents):
    """The off-by-one that would corrupt every result while looking healthy."""
    embedder = DenseEmbedder.from_config(config, encoder=RecordingEncoder())
    index = DenseIndex.build(fixture_documents, embedder, config)
    with pytest.raises(ValueError, match="mislabeled"):
        DenseIndex(index._index, index._business_ids[:-1], embedder, index.meta)


def test_save_load_round_trip_returns_identical_results(config, vocab_documents, tmp_path):
    """The real guard on the id mapping: results must survive a trip through disk
    unchanged, ids and scores both."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    built = DenseIndex.build(vocab_documents, embedder, config)
    before = built.search("quiet coffee wifi", top_k=5)

    built.save(tmp_path / "dense")
    reloaded = DenseIndex.load(tmp_path / "dense", embedder)
    after = reloaded.search("quiet coffee wifi", top_k=5)

    assert [bid for bid, _ in after] == [bid for bid, _ in before]
    assert np.allclose([s for _, s in after], [s for _, s in before])
    assert reloaded.n_documents == built.n_documents


def test_saved_metadata_records_what_built_the_index(config, vocab_documents, tmp_path):
    """Metadata exists so a mismatched load fails loudly instead of ranking
    nonsense; check the fields that make that possible are populated."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, config)
    index.save(tmp_path / "dense")

    reloaded = DenseIndex.load(tmp_path / "dense", embedder)
    assert reloaded.meta.model == config.retrieval.dense.model
    assert reloaded.meta.index_type == "IndexFlatIP"
    assert reloaded.meta.dim == len(VOCAB)
    assert reloaded.meta.n_documents == len(vocab_documents)
    assert reloaded.meta.query_prefix == PREFIX
    assert reloaded.meta.config_fingerprint == config.fingerprint()


def test_load_rejects_an_encoder_of_the_wrong_width(config, vocab_documents, tmp_path):
    """Many small models are 384-dimensional, so a dimension match is not proof
    of the right model — but a mismatch is proof of the wrong one."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    DenseIndex.build(vocab_documents, embedder, config).save(tmp_path / "dense")

    # VocabEncoder is 8-wide, so the stand-in has to be a different width for the
    # mismatch to be real rather than accidentally passing.
    wrong = DenseEmbedder.from_config(config, encoder=RecordingEncoder(dim=16))
    with pytest.raises(ValueError, match="wrong model"):
        DenseIndex.load(tmp_path / "dense", wrong)


def test_load_rejects_an_encoder_naming_a_different_model(config, vocab_documents, tmp_path):
    """The check a dimension match cannot make.

    Many small sentence encoders are 384-dimensional, so equal width is no proof
    of the right model: the vectors stay valid, every query still returns
    results, and they are ranked in an unrelated embedding space with nothing
    raising. This is the silent-garbage case `DenseIndexMeta` exists for, so the
    metadata has to be *enforced* and not merely recorded.
    """
    built = DenseEmbedder.from_config(
        config, encoder=NamedVocabEncoder(config.retrieval.dense.model)
    )
    DenseIndex.build(vocab_documents, built, config).save(tmp_path / "dense")

    # Same width, same prefix, different checkpoint — nothing else would catch it.
    impostor = DenseEmbedder.from_config(config, encoder=NamedVocabEncoder("some-other/encoder"))
    assert impostor.dim == built.dim
    with pytest.raises(ValueError, match="wrong model"):
        DenseIndex.load(tmp_path / "dense", impostor)


def test_load_accepts_an_encoder_naming_the_same_model(config, vocab_documents, tmp_path):
    """The check must not fire on the ordinary case, or `make index` could never
    be followed by a search."""
    embedder = DenseEmbedder.from_config(
        config, encoder=NamedVocabEncoder(config.retrieval.dense.model)
    )
    DenseIndex.build(vocab_documents, embedder, config).save(tmp_path / "dense")
    reloaded = DenseIndex.load(tmp_path / "dense", embedder)
    assert reloaded.meta.model == config.retrieval.dense.model


def test_load_skips_the_model_check_for_an_encoder_that_cannot_name_itself(
    config, vocab_documents, tmp_path
):
    """The documented escape hatch, asserted so it stays deliberate.

    An injected fake has no model name to give, and refusing to load without one
    would force every test in this file to invent a fiction. So None means
    "unverifiable" and the load proceeds on width and prefix alone — which is
    also why the real encoder's `model_name` is not optional in practice.
    """
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    assert encoder_model_name(embedder.encoder) is None
    DenseIndex.build(vocab_documents, embedder, config).save(tmp_path / "dense")
    assert DenseIndex.load(tmp_path / "dense", embedder).n_documents == len(vocab_documents)


def test_load_rejects_a_different_query_prefix(config, vocab_documents, tmp_path):
    """Querying a bge index without the prefix is the silent-degradation bug
    again, this time across a save/load boundary."""
    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    DenseIndex.build(vocab_documents, embedder, config).save(tmp_path / "dense")

    no_prefix = DenseEmbedder(encoder=VocabEncoder(), query_prefix="", normalize=True)
    with pytest.raises(ValueError, match="query_prefix"):
        DenseIndex.load(tmp_path / "dense", no_prefix)


def test_load_dense_index_explains_a_missing_index(config, tmp_path):
    with pytest.raises(FileNotFoundError, match="make index"):
        load_dense_index(config, encoder=VocabEncoder(), directory=tmp_path / "nope")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Index types and artifact paths
# ─────────────────────────────────────────────────────────────────────────────


def test_default_index_type_is_exact(config):
    """Exact first, so later quality numbers carry no unmeasured approximation
    error. HNSW is opt-in and its recall cost has to be measured."""
    assert config.retrieval.faiss.index_type == "IndexFlatIP"
    assert isinstance(_make_faiss_index(4, "IndexFlatIP"), faiss.IndexFlatIP)


def test_hnsw_index_builds_and_searches(config, vocab_documents):
    """HNSW is supported but approximate: this checks it works, NOT that it
    matches exact search. Recall against the exact index is an eval-time
    measurement and is currently unmeasured."""
    cfg = _hnsw_config(config)
    embedder = DenseEmbedder.from_config(cfg, encoder=VocabEncoder())
    index = DenseIndex.build(vocab_documents, embedder, cfg)
    hits = index.search("quiet coffee wifi", top_k=3)
    assert hits, "HNSW returned no neighbours at all"
    assert all(bid in {d.business_id for d in vocab_documents} for bid, _ in hits)
    assert index.meta.index_type == "HNSWFlat"


def test_unknown_index_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported faiss index_type"):
        _make_faiss_index(4, "IndexIVFPQ")


def test_artifact_dir_changes_with_the_config(config, tmp_path):
    """Artifacts are namespaced by config hash so a changed embedding model can
    never silently reuse the previous index."""
    other = _hnsw_config(config)
    assert dense_index_dir(config, root=tmp_path) != dense_index_dir(other, root=tmp_path)
    assert dense_index_dir(config, root=tmp_path) == tmp_path / "dense" / config.fingerprint_for(
        *DENSE_SECTIONS
    )
    # Scoped to what can change the vectors: an HTTP port cannot (DECISIONS.md D237).
    ported = config.model_copy(update={"serving": config.serving.model_copy(update={"port": 8001})})
    assert dense_index_dir(ported, root=tmp_path) == dense_index_dir(config, root=tmp_path)


def test_dense_and_bm25_artifact_layouts_match(config, tmp_path):
    """`artifacts/<half>/<fingerprint>` for both halves of the hybrid retriever,
    so an index pair is findable (and prunable) the same way."""
    dense_dir = dense_index_dir(config, root=tmp_path)
    bm25_dir = bm25.index_dir(config, root=tmp_path)
    # The *layout* matches; the two keys deliberately do not. The dense index
    # depends on the embedding model and BM25 does not, so keying them on the
    # same hash would mean changing the embedder rebuilt the BM25 index too
    # (DECISIONS.md D237).
    assert dense_dir.name == config.fingerprint_for(*DENSE_SECTIONS)
    assert bm25_dir.name == config.fingerprint_for(*BM25_SECTIONS)
    assert dense_dir.name != bm25_dir.name
    assert dense_dir.parent.parent == bm25_dir.parent.parent == tmp_path
    assert {dense_dir.parent.name, bm25_dir.parent.name} == {"dense", "bm25"}


# ─────────────────────────────────────────────────────────────────────────────
# 7. The real model — opt in with `-m slow`
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CONCIERGE_SLOW_TESTS"),
    reason="downloads BAAI/bge-small-en-v1.5; set CONCIERGE_SLOW_TESTS=1 to run",
)
def test_real_bge_small_end_to_end(config, tmp_path, capfd):
    """Downloads BAAI/bge-small-en-v1.5 and checks the convention on the real model.

    Opt-in twice over: the `slow` marker so `-m "not slow"` excludes it, and an
    env var so a plain `pytest` never reaches for the network. The fakes above
    verify the *wiring*; only the real model can confirm the prefix convention
    actually produces a sensible ranking.
    """
    # Importing torch while pytest has the file descriptors redirected aborts the
    # interpreter on this machine (torch's C extension loads during the import,
    # inside the capture). Restoring the real fds for the model load is enough;
    # see DECISIONS.md. The alternative is running the whole suite with `-s`.
    with capfd.disabled():
        encoder = SentenceTransformerEncoder(config.retrieval.dense.model)
    embedder = DenseEmbedder.from_config(config, encoder=encoder)
    assert embedder.dim == 384

    documents = [
        BusinessDocument(
            business_id="cafe",
            text="Lantern Coffee. A hushed neighbourhood cafe with free wifi and long tables.",
        ),
        BusinessDocument(
            business_id="ramen",
            text="Midnight Ramen. A packed, noisy late-night ramen counter.",
        ),
    ]
    index = DenseIndex.build(documents, embedder, config)
    assert index.search("somewhere quiet to work on my laptop", top_k=1)[0][0] == "cafe"
    assert index.search("noodles after midnight", top_k=1)[0][0] == "ramen"

    index.save(tmp_path / "dense")
    reloaded = DenseIndex.load(tmp_path / "dense", embedder)
    assert reloaded.search("somewhere quiet to work on my laptop", top_k=2) == index.search(
        "somewhere quiet to work on my laptop", top_k=2
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. The build CLI
# ─────────────────────────────────────────────────────────────────────────────
#
# These live in this file rather than a test_build_index.py because the CLI's
# only non-trivial logic is on the dense side. Nothing here touches the real
# encoder, so no test imports torch: every path exercised below runs before the
# model would be loaded.


def test_build_index_refuses_to_run_without_a_metro():
    """`make index` must stop before doing anything if the owner has not picked a
    metro yet. Indexing the wrong scope is not just wasted compute — every
    relevance label collected against it has to be thrown away."""
    with pytest.raises(ValueError, match="metro"):
        build_index.main(["--config", "configs/base.yaml"])


def test_find_documents_file_points_at_make_data(config, tmp_path):
    cfg = config.model_copy(
        update={"data": config.data.model_copy(update={"processed_dir": tmp_path})}
    )
    with pytest.raises(FileNotFoundError, match="make data"):
        build_index.find_documents_file(cfg)


def test_find_documents_file_prefers_parquet(config, tmp_path):
    (tmp_path / "documents.jsonl").write_text("")
    (tmp_path / "documents.parquet").write_bytes(b"")
    cfg = config.model_copy(
        update={"data": config.data.model_copy(update={"processed_dir": tmp_path})}
    )
    assert build_index.find_documents_file(cfg).name == "documents.parquet"


def test_load_documents_from_jsonl(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(
        '{"business_id": "b001", "text": "quiet coffee", "n_reviews_used": 2, "truncated": true}\n'
        '{"business_id": "b002", "text": "loud ramen"}\n'
    )
    documents = build_index.load_documents(path)
    assert [d.business_id for d in documents] == ["b001", "b002"]
    assert documents[0].n_reviews_used == 2
    assert documents[0].truncated is True
    # Defaults, not crashes, for the optional provenance columns.
    assert documents[1].n_tips_used == 0
    assert documents[1].truncated is False


def test_load_documents_names_the_missing_column(tmp_path):
    """Fail at load with the column name, not twenty minutes into an embedding
    loop with an AttributeError."""
    path = tmp_path / "documents.jsonl"
    path.write_text('{"business_id": "b001"}\n')
    with pytest.raises(ValueError, match="text"):
        build_index.load_documents(path)


def test_load_documents_rejects_an_empty_file(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no documents"):
        build_index.load_documents(path)


def test_directory_size_is_measured_not_estimated(tmp_path):
    """Index size on disk is a real serving constraint, so it is read from the
    filesystem rather than computed from vector counts."""
    assert build_index.directory_size_bytes(tmp_path / "absent") == 0
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_bytes(b"x" * 1000)
    (tmp_path / "b.bin").write_bytes(b"y" * 24)
    assert build_index.directory_size_bytes(tmp_path) == 1024


def test_format_bytes_uses_binary_units():
    assert build_index._format_bytes(512) == "512 B"
    assert build_index._format_bytes(1024) == "1.0 KiB"
    assert build_index._format_bytes(5 * 1024 * 1024) == "5.0 MiB"


def test_build_index_builds_and_saves_the_bm25_half(
    config, fixture_documents, tmp_path, monkeypatch
):
    """The BM25 half of `make index`, end to end through the real bm25 module.

    No torch anywhere in this path, so it runs in the default suite — which is
    the whole reason the two halves are built by separate functions.
    """
    from concierge.retrieval import bm25 as bm25_module

    monkeypatch.setattr(bm25_module, "index_dir", lambda cfg, root=None: tmp_path / "bm25")
    n_docs, directory = build_index._build_bm25(fixture_documents, config)

    assert n_docs == len(fixture_documents)
    assert directory == tmp_path / "bm25"
    assert build_index.directory_size_bytes(directory) > 0
    # Reloadable, which is the only property that makes a saved index worth having.
    assert bm25_module.BM25Index.load(directory).n_docs == len(fixture_documents)


def test_both_retrievers_expose_the_same_search_signature(
    config, vocab_documents, fixture_documents
):
    """RRF treats the two retrievers uniformly, so their return shapes must agree.

    A drift here is the kind of thing that only shows up as mysteriously bad
    fusion results, so it is asserted rather than assumed.
    """
    from concierge.retrieval import bm25 as bm25_module

    embedder = DenseEmbedder.from_config(config, encoder=VocabEncoder())
    dense = DenseIndex.build(vocab_documents, embedder, config)
    lexical = bm25_module.build_index(fixture_documents, config)

    for retriever, query in ((dense, "quiet coffee"), (lexical, "coffee")):
        hits = retriever.search(query, top_k=3)
        assert isinstance(hits, list)
        assert all(
            isinstance(pair, tuple) and isinstance(pair[0], str) and isinstance(pair[1], float)
            for pair in hits
        )


def test_duplicate_libomp_check_returns_paths_or_nothing():
    """A heuristic on file layout, so assert only its contract: distinct real
    paths, or an empty list. Whether this machine has the collision is an
    environment fact, not something the test should assert."""
    copies = build_index.duplicate_libomp_paths()
    assert isinstance(copies, list)
    assert len(copies) == len(set(copies))
    assert all(p.is_absolute() for p in copies)
