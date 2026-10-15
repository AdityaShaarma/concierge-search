"""Unit tests for the BM25 half of Stage 2.

These run against `tests/fixtures/businesses.json`, so they need neither the
5 GB Yelp download nor any API key. The documents are built here from the raw
fixture records with a deliberately minimal renderer: the real document builder
is another module's responsibility, and a lexical test should fail because
retrieval is wrong, not because someone changed how attributes are phrased.
"""

from __future__ import annotations

import json
import re

import pytest

from concierge.retrieval import bm25 as bm25_mod
from concierge.retrieval.bm25 import BM25Index, build_index, index_dir, load_or_build
from concierge.types import BusinessDocument

_LITERAL = re.compile(r"^u?'(?P<inner>.*)'$")


def _attribute_words(attributes: dict | None) -> str:
    """Render attribute values as bare words, mimicking a document builder.

    Only the string-valued attributes are rendered, and the stringified-literal
    quoting ("u'free'") is stripped, because unstripped quotes would tokenize
    into a stray "free" anyway and make the test pass for the wrong reason.
    """
    if not attributes:
        return ""
    words: list[str] = []
    for key, value in attributes.items():
        if not isinstance(value, str):
            continue
        match = _LITERAL.match(value)
        inner = match.group("inner") if match else value
        if inner in {"True", "None", "False"} or inner.startswith("{"):
            continue
        if inner.isdigit():
            continue
        words.append(f"{key} {inner}")
    return " ".join(words)


@pytest.fixture
def documents(raw_businesses: list[dict]) -> list[BusinessDocument]:
    return [
        BusinessDocument(
            business_id=b["business_id"],
            text=f"{b['name']}. {b['categories']}. {_attribute_words(b['attributes'])}",
        )
        for b in raw_businesses
    ]


@pytest.fixture
def index(documents: list[BusinessDocument], config) -> BM25Index:
    return build_index(documents, config)


# ── basic retrieval ──────────────────────────────────────────────────────────


def test_search_finds_the_obvious_document(index: BM25Index) -> None:
    """"ramen" occurs in exactly one fixture business, in both its name and its
    categories, so it is an unambiguous lexical hit."""
    results = index.search("ramen", top_k=5)
    assert results, "expected at least one lexical match for 'ramen'"
    assert results[0][0] == "b002"


def test_search_is_ordered_by_descending_score(index: BM25Index) -> None:
    results = index.search("quiet cafes free wifi", top_k=8)
    scores = [score for _bid, score in results]
    assert scores == sorted(scores, reverse=True)


def test_zero_score_documents_are_dropped(index: BM25Index) -> None:
    """A BM25 score of 0 means no query term occurred in the document at all.

    bm25s returns such rows to fill out top-k; passing them on would earn them
    a real 1/(k+rank) contribution in fusion for nothing. Only one fixture
    business mentions ramen, so asking for 8 must still return 1.
    """
    results = index.search("ramen", top_k=8)
    assert [bid for bid, _ in results] == ["b002"]
    assert all(score > 0 for _bid, score in results)


def test_top_k_larger_than_corpus_does_not_raise(index: BM25Index) -> None:
    """bm25s raises if k exceeds the corpus size; the wrapper clamps.

    This is not a hypothetical: `cfg.candidates.retrieve` is 200, and a
    single-metro scope can be smaller than that during development.
    """
    results = index.search("cafes", top_k=1000)
    assert len(results) <= index.n_docs


def test_stopword_only_query_returns_nothing(index: BM25Index) -> None:
    """"the best of it" is entirely stopwords under the English list, so there
    is no lexical evidence to offer. Returning an arbitrary top-k here would
    inject pure noise into fusion."""
    assert index.search("the best of it", top_k=5) == []


def test_empty_query_returns_nothing(index: BM25Index) -> None:
    assert index.search("", top_k=5) == []


def test_build_index_rejects_an_empty_corpus(config) -> None:
    with pytest.raises(ValueError, match="zero documents"):
        build_index([], config)


# ── configuration is actually honored ────────────────────────────────────────


def test_k1_and_b_come_from_config(documents: list[BusinessDocument], config) -> None:
    """A config key that is silently ignored means an experiment that did not
    test what it claimed to, so this asserts the values reached bm25s."""
    tweaked_bm25 = config.retrieval.bm25.model_copy(update={"k1": 0.9, "b": 0.3})
    tweaked = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(update={"bm25": tweaked_bm25}),
        }
    )
    index = build_index(documents, tweaked)
    assert index.retriever.k1 == pytest.approx(0.9)
    assert index.retriever.b == pytest.approx(0.3)
    assert index.meta["k1"] == pytest.approx(0.9)
    assert index.meta["b"] == pytest.approx(0.3)


def test_b_zero_disables_length_normalization(documents: list[BusinessDocument], config) -> None:
    """A behavioural check, not just a plumbed-through number: b controls
    document-length normalization, so b=0 must change the scores."""
    default = build_index(documents, config)
    no_norm_bm25 = config.retrieval.bm25.model_copy(update={"b": 0.0})
    no_norm = build_index(
        documents,
        config.model_copy(
            update={"retrieval": config.retrieval.model_copy(update={"bm25": no_norm_bm25})}
        ),
    )
    query = "quiet cafes free wifi"
    assert default.search(query, top_k=8) != no_norm.search(query, top_k=8)


def test_index_dir_is_keyed_by_config_fingerprint(config, tmp_path) -> None:
    """Two configs must never read each other's index — that is how a result
    gets attributed to the wrong settings."""
    other_bm25 = config.retrieval.bm25.model_copy(update={"k1": 1.9})
    other = config.model_copy(
        update={"retrieval": config.retrieval.model_copy(update={"bm25": other_bm25})}
    )
    first = index_dir(config, root=tmp_path)
    second = index_dir(other, root=tmp_path)
    assert first != second
    assert config.fingerprint() in str(first)


# ── persistence ──────────────────────────────────────────────────────────────


def test_save_and_load_round_trip(index: BM25Index, tmp_path) -> None:
    directory = index.save(tmp_path / "idx")
    loaded = BM25Index.load(directory)
    assert loaded.business_ids == index.business_ids
    assert loaded.meta == index.meta
    query = "quiet cafes free wifi"
    assert loaded.search(query, top_k=5) == index.search(query, top_k=5)


def test_load_or_build_writes_then_reuses(
    documents: list[BusinessDocument], config, tmp_path
) -> None:
    built = load_or_build(documents, config, root=tmp_path)
    directory = index_dir(config, root=tmp_path)
    assert directory.exists()

    # Second call must not rebuild. Detected by making a rebuild impossible:
    # if `build_index` were called, this would raise.
    def _explode(*_args, **_kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("load_or_build rebuilt an index that was already on disk")

    original = bm25_mod.build_index
    bm25_mod.build_index = _explode  # type: ignore[assignment]
    try:
        reused = load_or_build(documents, config, root=tmp_path)
    finally:
        bm25_mod.build_index = original  # type: ignore[assignment]
    assert reused.business_ids == built.business_ids


def test_corrupt_index_falls_back_to_rebuilding(
    documents: list[BusinessDocument], config, tmp_path
) -> None:
    """An index is derived data, so an unreadable cache must never break a run."""
    directory = index_dir(config, root=tmp_path)
    load_or_build(documents, config, root=tmp_path)
    (directory / "vocab.index.json").write_text("not json at all")
    rebuilt = load_or_build(documents, config, root=tmp_path)
    assert rebuilt.search("ramen", top_k=3)[0][0] == "b002"


def test_tokenization_mismatch_is_refused_at_query_time(
    index: BM25Index, tmp_path
) -> None:
    """An index built with a different tokenizer must not be queried.

    If the corpus was stemmed and the query is not (or vice versa), multi-token
    queries silently lose recall — a bug that produces plausible-looking but
    quietly worse results, which is the worst kind for a project whose output is
    measurements.
    """
    directory = index.save(tmp_path / "idx")
    meta = json.loads((directory / "concierge_meta.json").read_text())
    meta["stemmer"] = "some-other-stemmer"
    (directory / "concierge_meta.json").write_text(json.dumps(meta))

    loaded = BM25Index.load(directory)
    with pytest.raises(RuntimeError, match="stemmer"):
        loaded.search("ramen", top_k=3)


# ── stemming ─────────────────────────────────────────────────────────────────


def test_stemming_matches_singular_query_to_plural_document(
    documents: list[BusinessDocument], config
) -> None:
    """Skipped unless PyStemmer is installed, which it may not be — it is a C
    extension and is not a declared dependency. When it is present, a singular
    query term must reach a plural document term: "bookstore" vs the fixture's
    "Bookstores" is exactly the recall loss stemming buys back.
    """
    pytest.importorskip("Stemmer", reason="PyStemmer not installed; stemming disabled")
    index = build_index(documents, config)
    assert index.meta["stemmer"] == "snowball-english"
    assert [bid for bid, _ in index.search("bookstore", top_k=3)] == ["b003"]


def test_stemmer_name_is_recorded_in_metadata(index: BM25Index) -> None:
    """Whichever way it resolved, it must be written down — the whole mismatch
    guard depends on it."""
    assert index.meta["stemmer"] in {"none", "snowball-english"}
