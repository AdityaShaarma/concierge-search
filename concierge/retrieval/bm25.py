"""Stage 2, lexical half: BM25 over business documents.

BM25 is in the cascade alongside dense embeddings because the two fail in
opposite directions. An embedding model knows "ramen" and "noodle soup" are
related but is weak on rare literal tokens — a business name, a neighbourhood,
"birria", "horchata". BM25 is the reverse: exact-term matching with no notion of
synonymy. Fusing them (see ``fusion.py``) is cheaper than trying to fix either
one, which is why hybrid retrieval is the standard first move.

Implementation notes that matter for the interview defence:

* ``bm25s`` stores the index as a sparse CSC matrix and scores a query by
  slicing the columns for the query terms, so query cost scales with the number
  of query terms rather than the corpus size.
* The BM25 *variant* is a real choice, not a detail — see ``BM25_METHOD`` below.
* Tokenization is the single largest lever on lexical recall here — see
  ``TOKENIZER_STOPWORDS`` and ``_build_stemmer``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s

from concierge.config import REPO_ROOT, Config
from concierge.types import BusinessDocument

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenization and variant choices.
#
# These are module constants rather than config keys because `BM25Config` is a
# frozen, extra="forbid" contract owned elsewhere and carries only k1/b. They
# are recorded in the saved index metadata so an index built under different
# settings is refused at load time rather than silently reused.
# ─────────────────────────────────────────────────────────────────────────────

# bm25s' default BM25 variant is Lucene's, i.e. the one Elasticsearch/Solr ship.
# Why it is the right default here: its IDF formula is clamped so that a term
# appearing in more than half the corpus still gets a small *positive* weight,
# whereas the textbook Robertson/Sparck-Jones IDF goes negative and lets a very
# common term actively push a document down. Our documents are highly
# repetitive ("Restaurants", "Food", "wifi" appear everywhere), so negative IDF
# would be hit constantly. Keeping the Lucene variant also means a number
# measured here is comparable to the BM25 baseline a reader would get from
# Elasticsearch.
BM25_METHOD = "lucene"

# English stopwords are removed. The risk of stopword removal in this domain is
# dropping a word that carries a constraint ("no wifi", "not loud"): bm25s'
# English list contains "no" and "not". We accept that, because negation and
# every other hard constraint is handled arithmetically by `filters.py` from the
# Stage 1 parse, never by lexical overlap. BM25's job here is topical matching
# ("ramen", "bookstore", "birria"), and for that, stopwords are pure noise that
# inflates document length and therefore distorts the length normalization.
TOKENIZER_STOPWORDS = "english"

# bm25s' default: ASCII/unicode word characters, two or more. Single characters
# and bare punctuation are dropped, so "$$" does not survive — intentional,
# because price is a hard filter, not a lexical signal.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

# Snowball stemming is used *when PyStemmer is importable* and skipped
# otherwise. Why stemming at all: without it, "cafes" in a document and "cafe"
# in a query are unrelated tokens, and the same goes for
# bakery/bakeries, noodle/noodles, open/opens. That is a direct and
# measurable recall loss on short queries, where there are few terms to
# compensate. Why it is optional: PyStemmer is a C extension and is not a
# declared dependency of this project, so it cannot be assumed present.
# Because stemming changes the index contents, the resolved stemmer name is
# written into the index metadata and checked on load — an index built with
# stemming must not be served by a process that cannot stem its queries the
# same way, or every multi-token query silently loses recall.
STEMMER_LANGUAGE = "english"


def _build_stemmer() -> tuple[Callable[[list[str]], list[str]] | None, str]:
    """Return ``(stemmer_fn, name)``; ``(None, "none")`` if PyStemmer is absent.

    The name is what gets persisted, so "none" and "snowball-english" produce
    different, incompatible indexes — which is the point.
    """
    try:
        import Stemmer  # optional C extension; see the note above
    except ImportError:
        log.warning(
            "PyStemmer not installed: BM25 will index unstemmed tokens, which "
            "costs lexical recall on plural/inflected query terms. This is "
            "recorded in the index metadata."
        )
        return None, "none"
    return Stemmer.Stemmer(STEMMER_LANGUAGE).stemWords, f"snowball-{STEMMER_LANGUAGE}"


def _tokenize(texts: Sequence[str] | str, stemmer: Callable[[list[str]], list[str]] | None):
    """Tokenize to lists of token *strings* rather than vocabulary ids.

    bm25s can return integer ids keyed to a vocabulary built at tokenization
    time. We deliberately do not use that: the ids are only meaningful next to
    the exact vocabulary that produced them, so a query tokenized against a
    freshly-built vocabulary would index into the wrong columns of a loaded
    matrix. Passing strings makes bm25s resolve them through the vocabulary
    stored *with the index*, which removes that whole class of bug for a cost
    (string hashing per query term) that is irrelevant at query time.
    """
    return bm25s.tokenize(
        texts,
        lower=True,
        token_pattern=TOKEN_PATTERN,
        stopwords=TOKENIZER_STOPWORDS,
        stemmer=stemmer,
        return_ids=False,
        show_progress=False,
    )


# File names inside an index directory. `bm25s` owns the rest.
_IDS_FILE = "business_ids.json"
_META_FILE = "concierge_meta.json"


def index_dir(cfg: Config, root: Path | None = None) -> Path:
    """Where this config's BM25 index lives.

    Keyed by `Config.fingerprint()` so two experiments can never read each
    other's index, and so any number in the README traces back to the settings
    that produced it.
    """
    base = root if root is not None else REPO_ROOT / "artifacts"
    return Path(base) / "bm25" / cfg.fingerprint()


@dataclass(frozen=True, slots=True)
class BM25Index:
    """A built BM25 index plus the row->business_id mapping it needs.

    bm25s identifies documents by their row number. Keeping the id list
    alongside the matrix (and saving them together) is what makes a loaded
    index usable; storing them separately is how row-offset bugs happen.
    """

    retriever: Any  # bm25s.BM25 — typed as Any because bm25s ships no stubs.
    business_ids: tuple[str, ...]
    meta: dict[str, Any]

    @property
    def n_docs(self) -> int:
        return len(self.business_ids)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(business_id, score)`` pairs, best first.

        Documents whose score is not strictly positive are dropped. A BM25 score
        of exactly zero means *no query term occurred in the document at all*;
        bm25s still returns such rows to fill out top-k. Letting them through
        would hand them a fusion rank and therefore a real 1/(k+rank)
        contribution purely for existing, which is noise, not evidence.

        So an empty list is a legitimate answer here, and a frequent one: a
        constraint-only query such as "somewhere with a patio" leaves the
        semantic residue "somewhere", which occurs in no document. That is
        correct for *this* retriever and is not a complete answer for the
        system, because the query's correctly-parsed hard constraints then have
        no candidate list to filter. The gap is closed one level up, by
        `concierge.retrieval.constraints_index`, which generates candidates from
        the constraints themselves when retrieval under-delivers. Recorded here
        because this return is where a reader chasing an empty result page would
        start looking (DECISIONS.md D224).
        """
        stemmer, _ = self._stemmer()
        tokens = _tokenize(query_text, stemmer)
        # tokens is [[...]] — one list per query; we only ever send one.
        if not tokens or not tokens[0]:
            # Everything was a stopword or punctuation ("the best!!"). There is
            # no lexical evidence to offer; the dense half still gets its say.
            return []

        k = min(top_k, self.n_docs)
        if k <= 0:
            return []

        doc_rows, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        out: list[tuple[str, float]] = []
        for row, score in zip(doc_rows[0], scores[0], strict=True):
            if score <= 0.0:
                continue
            out.append((self.business_ids[int(row)], float(score)))
        return out

    def _stemmer(self) -> tuple[Callable[[list[str]], list[str]] | None, str]:
        """Resolve the stemmer and assert it matches the one the index was built
        with, so a query is never tokenized differently from the corpus."""
        stemmer, name = _build_stemmer()
        built_with = self.meta.get("stemmer")
        if built_with is not None and built_with != name:
            raise RuntimeError(
                f"BM25 index was built with stemmer={built_with!r} but this "
                f"environment resolves {name!r}. Queries would be tokenized "
                f"differently from the corpus. Rebuild the index, or install "
                f"PyStemmer to match."
            )
        return stemmer, name

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory), show_progress=False)
        (directory / _IDS_FILE).write_text(json.dumps(list(self.business_ids)))
        (directory / _META_FILE).write_text(json.dumps(self.meta, indent=2, sort_keys=True))
        return directory

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        directory = Path(directory)
        meta = json.loads((directory / _META_FILE).read_text())
        ids = tuple(json.loads((directory / _IDS_FILE).read_text()))
        retriever = bm25s.BM25.load(str(directory), load_corpus=False, show_progress=False)
        return cls(retriever=retriever, business_ids=ids, meta=meta)


def build_index(documents: Sequence[BusinessDocument], cfg: Config) -> BM25Index:
    """Build a BM25 index over the given business documents.

    Document order defines row order, so it is captured in `business_ids`
    rather than assumed to be stable across runs.
    """
    if not documents:
        raise ValueError("cannot build a BM25 index over zero documents")

    stemmer, stemmer_name = _build_stemmer()
    corpus_tokens = _tokenize([d.text for d in documents], stemmer)

    retriever = bm25s.BM25(k1=cfg.retrieval.bm25.k1, b=cfg.retrieval.bm25.b, method=BM25_METHOD)
    retriever.index(corpus_tokens, show_progress=False)

    meta = {
        "k1": cfg.retrieval.bm25.k1,
        "b": cfg.retrieval.bm25.b,
        "method": BM25_METHOD,
        "stopwords": TOKENIZER_STOPWORDS,
        "token_pattern": TOKEN_PATTERN,
        "stemmer": stemmer_name,
        "n_docs": len(documents),
        "config_fingerprint": cfg.fingerprint(),
    }
    return BM25Index(
        retriever=retriever,
        business_ids=tuple(d.business_id for d in documents),
        meta=meta,
    )


def load_or_build(
    documents: Sequence[BusinessDocument],
    cfg: Config,
    root: Path | None = None,
    force_rebuild: bool = False,
) -> BM25Index:
    """Load this config's index from disk, building and saving it if absent.

    A corrupt or partially-written index directory falls through to a rebuild
    rather than crashing: the index is derived data, so recomputing is always
    safe, and an unreadable cache should never be able to break a run.
    """
    directory = index_dir(cfg, root)
    if not force_rebuild and (directory / _META_FILE).exists():
        try:
            return BM25Index.load(directory)
        except Exception as exc:  # any failure to read a cache means "rebuild it"
            log.warning("could not load BM25 index at %s (%s); rebuilding", directory, exc)

    index = build_index(documents, cfg)
    index.save(directory)
    return index
