"""Tests for Stage 3: the cross-encoder reranker.

Nothing here downloads a model by default. ``ms-marco-MiniLM-L-6-v2`` is ~90 MB
and a few seconds to load, which would make the suite too slow to run on every
edit — and more importantly, a test that needs the network cannot pin a *logic*
bug. So the scorer is injected behind the ``PairScorer`` protocol and two tiny
deterministic fakes stand in for it:

  ``ScriptedScorer``  returns a score looked up per document, so an expected
                      ordering can be asserted exactly, including ties.
  ``WordScorer``      scores by shared-word overlap, so a ranking assertion means
                      something ("quiet cafe" must rank the quiet café first)
                      rather than only checking plumbing.

What is actually being pinned here:

* the stage **adds** ``cross_encoder`` score and rank and never touches upstream
  ones (Stage 4 consumes those as features);
* ties keep upstream order (a *stable* sort), so the eval is reproducible;
* ``top_k`` truncates, and a missing document fails loudly;
* the truncation diagnostic's arithmetic, since that number is the only warning
  we get before ``max_length`` silently caps quality;
* the gold-leakage assertion in ``train.py`` actually fires on a planted gold
  query — the single most consequential check in the project.

One section here is not about Stage 3 at all: "artifact layout" pins the paths
the *two* trainers resolve against the paths ``concierge.synth.splits``
actually writes. It lives in this file because it is one seam, not two — a
disagreement between the writer and either reader is the same bug, and testing
the halves in separate files is how the halves drifted in the first place.

Tests that genuinely load a model or export ONNX are marked ``slow``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from concierge.config import RERANK_SECTIONS
from concierge.data.hours import DAYS
from concierge.ltr import train as ltr_train
from concierge.query.rules import parse_rules
from concierge.rerank import model as rerank_model
from concierge.rerank import train as rerank_train
from concierge.rerank.model import (
    CrossEncoderReranker,
    PairScorer,
    TruncationReport,
    TruncationStat,
    query_text_for,
    rerank_dir,
    resolve_weights,
    training_parse,
    training_query_text,
)
from concierge.rerank.onnx_export import DEFAULT_TOLERANCE, ParityResult
from concierge.synth import splits as synth_splits
from concierge.types import (
    RANK_CROSS_ENCODER,
    RANK_RRF,
    SCORE_BM25,
    SCORE_CROSS_ENCODER,
    SCORE_DENSE,
    SCORE_RRF,
    BusinessDocument,
    Candidate,
    ParsedQuery,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fake scorers
# ─────────────────────────────────────────────────────────────────────────────


class ScriptedScorer:
    """Returns a per-document score from a table. Records every pair it saw."""

    def __init__(self, table: dict[str, float], max_length: int = 32) -> None:
        self._table = table
        self._max_length = max_length
        self.seen: list[tuple[str, str]] = []

    @property
    def max_length(self) -> int:
        return self._max_length

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.seen.extend(pairs)
        return [self._table[document] for _, document in pairs]

    def count_tokens(self, text: str) -> int:
        # Whitespace tokens: enough to exercise the truncation arithmetic without
        # a real tokenizer, and obvious to reason about in an assertion.
        return len(text.split())

    def document_token_budget(self, query_text: str) -> int:
        return max(0, self._max_length - 3 - self.count_tokens(query_text))


class WordScorer:
    """Scores by how many query words appear in the document."""

    def __init__(self, max_length: int = 64) -> None:
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        return self._max_length

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        for query, document in pairs:
            words = set(document.lower().split())
            scores.append(float(sum(1 for w in query.lower().split() if w in words)))
        return scores

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def document_token_budget(self, query_text: str) -> int:
        return max(0, self._max_length - 3 - self.count_tokens(query_text))


class ShortReturningScorer(ScriptedScorer):
    """Returns fewer scores than pairs — the bug that would misalign every score."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return super().score_pairs(pairs)[:-1]


def test_fakes_satisfy_the_protocol():
    """If the fakes drift from the Protocol the other tests prove nothing."""
    assert isinstance(ScriptedScorer({}), PairScorer)
    assert isinstance(WordScorer(), PairScorer)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _candidate(business_id: str, rrf_rank: int, rrf_score: float = 0.0) -> Candidate:
    """A Stage 2 candidate carrying the scores Stage 4 will need."""
    return Candidate(
        business_id=business_id,
        scores={SCORE_BM25: 1.0, SCORE_DENSE: 0.5, SCORE_RRF: rrf_score},
        ranks={RANK_RRF: rrf_rank},
    )


@pytest.fixture
def parsed() -> ParsedQuery:
    return ParsedQuery(raw="quiet cafe with wifi", semantic_text="quiet cafe")


@pytest.fixture
def documents() -> dict[str, str]:
    return {
        "b001": "Corner Coffee. quiet cafe with free wifi.",
        "b002": "Loud Sports Bar. loud bar with tvs.",
        "b003": "Reading Room. quiet bookstore cafe.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# score()
# ─────────────────────────────────────────────────────────────────────────────


def test_score_returns_one_score_per_document_in_order(documents):
    scorer = ScriptedScorer({documents["b001"]: 3.0, documents["b002"]: 1.0})
    reranker = CrossEncoderReranker(scorer)

    scores = reranker.score("quiet cafe", [documents["b001"], documents["b002"]])

    assert scores == [3.0, 1.0]
    # The query must be paired with each document, which is the whole point of a
    # cross-encoder: query and document go through the model together.
    assert scorer.seen == [
        ("quiet cafe", documents["b001"]),
        ("quiet cafe", documents["b002"]),
    ]


def test_score_rejects_a_scorer_that_returns_the_wrong_count(documents):
    reranker = CrossEncoderReranker(
        ShortReturningScorer({documents["b001"]: 1.0, documents["b002"]: 2.0})
    )
    with pytest.raises(ValueError, match="1 scores for 2 pairs"):
        reranker.score("quiet cafe", [documents["b001"], documents["b002"]])


def test_query_text_prefers_semantic_text_but_falls_back_to_raw():
    parsed = ParsedQuery(raw="open past 10pm quiet cafe", semantic_text="quiet cafe")
    assert query_text_for(parsed) == "quiet cafe"

    # A parser that stripped everything into hard filters leaves no semantic text;
    # scoring every candidate against "" would make the stage a no-op that still
    # costs full latency.
    stripped = ParsedQuery(raw="open past 10pm", semantic_text="   ")
    assert query_text_for(stripped) == "open past 10pm"


def test_rerank_scores_the_semantic_text_not_the_raw_query(parsed, documents):
    scorer = ScriptedScorer({text: 1.0 for text in documents.values()})
    CrossEncoderReranker(scorer).rerank(parsed, [_candidate("b001", 1)], documents, top_k=1)
    assert scorer.seen[0][0] == "quiet cafe"


# ─────────────────────────────────────────────────────────────────────────────
# rerank(): bookkeeping
# ─────────────────────────────────────────────────────────────────────────────


def test_rerank_preserves_upstream_scores_and_adds_its_own(parsed, documents):
    scorer = ScriptedScorer({documents["b001"]: 2.0, documents["b002"]: 9.0})
    candidates = [_candidate("b001", 1, rrf_score=0.03), _candidate("b002", 2, rrf_score=0.02)]

    reranked = CrossEncoderReranker(scorer).rerank(parsed, candidates, documents, top_k=10)

    best = reranked[0]
    assert best.business_id == "b002"
    # Upstream features survive untouched: Stage 4 trains on them.
    assert best.scores[SCORE_BM25] == 1.0
    assert best.scores[SCORE_DENSE] == 0.5
    assert best.scores[SCORE_RRF] == 0.02
    assert best.ranks[RANK_RRF] == 2
    # And this stage contributes both a score and a 1-based rank.
    assert best.scores[SCORE_CROSS_ENCODER] == 9.0
    assert best.ranks[RANK_CROSS_ENCODER] == 1
    assert reranked[1].ranks[RANK_CROSS_ENCODER] == 2


def test_rerank_does_not_mutate_its_inputs(parsed, documents):
    scorer = ScriptedScorer({documents["b001"]: 2.0})
    candidate = _candidate("b001", 1)
    before = dict(candidate.scores), dict(candidate.ranks)

    CrossEncoderReranker(scorer).rerank(parsed, [candidate], documents, top_k=1)

    assert (candidate.scores, candidate.ranks) == before
    assert SCORE_CROSS_ENCODER not in candidate.scores


def test_rerank_reorders_by_cross_encoder_score(parsed, documents):
    """The reranker must be able to overturn the retrieval order — otherwise the
    stage is pure cost."""
    scorer = WordScorer()
    # b002 (the loud bar) is wrongly ranked first by retrieval.
    candidates = [_candidate("b002", 1), _candidate("b003", 2), _candidate("b001", 3)]

    reranked = CrossEncoderReranker(scorer).rerank(parsed, candidates, documents, top_k=3)

    assert [c.business_id for c in reranked] == ["b001", "b003", "b002"]
    assert reranked[0].scores[SCORE_CROSS_ENCODER] == 2.0  # "quiet" + "cafe"
    assert reranked[-1].scores[SCORE_CROSS_ENCODER] == 0.0


def test_rerank_is_a_stable_sort_on_ties(parsed, documents):
    """Equal scores keep the upstream order.

    Upstream rank is a real signal; dict or hash order is noise. A tiebreak by
    noise would also make two runs of the eval disagree, which would quietly
    widen every confidence interval.
    """
    scorer = ScriptedScorer({text: 5.0 for text in documents.values()})
    candidates = [_candidate("b003", 1), _candidate("b001", 2), _candidate("b002", 3)]

    reranked = CrossEncoderReranker(scorer).rerank(parsed, candidates, documents, top_k=3)

    assert [c.business_id for c in reranked] == ["b003", "b001", "b002"]

    # And the reverse input order must come back reversed, which is what proves
    # the order came from the input rather than from an incidental sort key.
    reversed_back = CrossEncoderReranker(scorer).rerank(
        parsed, list(reversed(candidates)), documents, top_k=3
    )
    assert [c.business_id for c in reversed_back] == ["b002", "b001", "b003"]


def test_rerank_respects_top_k(parsed, documents):
    scorer = ScriptedScorer(
        {documents["b001"]: 3.0, documents["b002"]: 2.0, documents["b003"]: 1.0}
    )
    candidates = [_candidate("b001", 1), _candidate("b002", 2), _candidate("b003", 3)]
    reranker = CrossEncoderReranker(scorer)

    assert len(reranker.rerank(parsed, candidates, documents, top_k=2)) == 2
    # Asking for more than exists is not an error; the cascade's `rerank` count
    # is a cap, and short candidate lists are normal for heavily filtered queries.
    assert len(reranker.rerank(parsed, candidates, documents, top_k=50)) == 3


def test_rerank_rejects_nonsense_top_k(parsed, documents):
    reranker = CrossEncoderReranker(ScriptedScorer({}))
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        reranker.rerank(parsed, [_candidate("b001", 1)], documents, top_k=0)


def test_rerank_of_nothing_is_nothing(parsed, documents):
    """An over-filtered query can legitimately retrieve zero candidates."""
    assert CrossEncoderReranker(ScriptedScorer({})).rerank(parsed, [], documents, top_k=10) == ()


def test_rerank_fails_loudly_when_a_document_is_missing(parsed, documents):
    """Index and document store are built together; drift between them is a build
    bug, and serving a silently shortened shortlist would hide it."""
    reranker = CrossEncoderReranker(ScriptedScorer({text: 1.0 for text in documents.values()}))
    with pytest.raises(KeyError, match="no document text"):
        reranker.rerank(parsed, [_candidate("b404", 1)], documents, top_k=10)


def test_rank_last_keeps_undocumented_candidates_without_inventing_a_score(parsed, documents):
    scorer = ScriptedScorer({documents["b001"]: 1.0})
    candidates = [_candidate("b404", 1), _candidate("b001", 2)]

    reranked = CrossEncoderReranker(scorer).rerank(
        parsed, candidates, documents, top_k=10, on_missing_document="rank_last"
    )

    assert [c.business_id for c in reranked] == ["b001", "b404"]
    # No cross-encoder score at all, rather than a 0.0 that Stage 4 would treat
    # as a measurement.
    assert SCORE_CROSS_ENCODER not in reranked[1].scores
    assert reranked[1].ranks[RANK_CROSS_ENCODER] == 2


def test_rerank_rejects_an_unknown_missing_document_policy(parsed, documents):
    reranker = CrossEncoderReranker(ScriptedScorer({}))
    with pytest.raises(ValueError, match="on_missing_document"):
        reranker.rerank(
            parsed, [_candidate("b404", 1)], documents, top_k=1, on_missing_document="shrug"
        )


@pytest.mark.parametrize("shape", ["mapping_of_str", "mapping_of_documents", "sequence"])
def test_rerank_accepts_every_document_shape_callers_have(parsed, documents, shape):
    """Retrieval hands around `BusinessDocument`s, the serving layer caches plain
    strings; both must work without a conversion dance at each call site."""
    scorer = ScriptedScorer({text: 1.0 for text in documents.values()})
    docs: object
    if shape == "mapping_of_str":
        docs = documents
    elif shape == "mapping_of_documents":
        docs = {bid: BusinessDocument(business_id=bid, text=t) for bid, t in documents.items()}
    else:
        docs = [BusinessDocument(business_id=bid, text=t) for bid, t in documents.items()]

    reranked = CrossEncoderReranker(scorer).rerank(parsed, [_candidate("b001", 1)], docs, top_k=1)
    assert reranked[0].scores[SCORE_CROSS_ENCODER] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Truncation diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def test_truncation_report_arithmetic():
    """max_length is shared by query + document + specials; the report must show
    how much of the document actually survived that split."""
    scorer = ScriptedScorer({}, max_length=10)  # budget = 10 - 3 specials - query
    reranker = CrossEncoderReranker(scorer)

    report = reranker.truncation_report(
        "quiet cafe",  # 2 tokens -> document budget 5
        ["one two three", "one two three four five six seven eight"],
    )

    assert report.document_token_budget == 5
    assert report.query_tokens == 2
    assert report.n_documents == 2
    assert report.n_truncated == 1
    assert report.fraction_truncated == 0.5
    assert report.stats[0].kept_fraction == 1.0
    assert report.stats[1].tokens_kept == 5
    assert report.stats[1].kept_fraction == pytest.approx(5 / 8)
    assert report.min_kept_fraction == pytest.approx(5 / 8)
    assert report.mean_kept_fraction == pytest.approx((1.0 + 5 / 8) / 2)
    # The summary is what ends up in a build log, so it must carry the numbers.
    assert "1/2 documents clipped" in report.summary()


def test_truncation_report_keyed_by_business_id(documents):
    scorer = ScriptedScorer({}, max_length=8)
    report = CrossEncoderReranker(scorer).truncation_report_for_documents("quiet cafe", documents)
    assert {s.business_id for s in report.stats} == set(documents)
    assert report.worst(1)[0].kept_fraction <= report.stats[0].kept_fraction


def test_empty_document_is_reported_as_whole_not_as_a_division_by_zero():
    stat = TruncationStat(business_id="b001", document_tokens=0, tokens_kept=0)
    assert stat.kept_fraction == 1.0
    assert not stat.truncated


def test_truncation_report_over_no_documents_is_vacuous():
    report = TruncationReport(max_length=512, query_tokens=3, document_token_budget=500, stats=())
    assert report.n_documents == 0
    assert report.fraction_truncated == 0.0
    assert report.mean_kept_fraction == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Weight resolution — which row of the ablation table is this?
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_weights_reports_the_pretrained_fallback_visibly(config, tmp_path, monkeypatch):
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path)
    name, source = resolve_weights(config)
    assert name == config.rerank.base_model
    # "cross-encoder (pretrained)" and "cross-encoder (fine-tuned)" are different
    # rows of the ablation table; a silent fallback would mislabel a result.
    assert "pretrained" in source


def test_resolve_weights_prefers_a_fine_tuned_checkpoint(config, tmp_path, monkeypatch):
    checkpoint = tmp_path / rerank_model.CHECKPOINT_DIRNAME
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path)

    name, source = resolve_weights(config)

    assert name == str(checkpoint)
    assert "fine-tuned" in source


def test_rerank_dir_is_keyed_by_config_fingerprint(config, tmp_path):
    directory = rerank_dir(config, root=tmp_path)
    assert directory == tmp_path / "rerank" / config.fingerprint_for(*RERANK_SECTIONS)
    # Changing a reranking hyperparameter must not reuse the old checkpoint.
    other = config.model_copy(update={"rerank": config.rerank.model_copy(update={"lr": 5e-5})})
    assert rerank_dir(other, root=tmp_path) != directory
    # And a knob the checkpoint cannot depend on must not orphan it: an hour of
    # Colab GPU is not something an HTTP port gets to invalidate (DECISIONS.md
    # D237).
    ported = config.model_copy(update={"serving": config.serving.model_copy(update={"port": 8001})})
    assert rerank_dir(ported, root=tmp_path) == directory


def test_load_run_metadata_absent_is_none(config, tmp_path, monkeypatch):
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path)
    assert rerank_model.load_run_metadata(config) is None
    (tmp_path / rerank_model.RUN_METADATA_FILENAME).write_text(json.dumps({"seed": 42}))
    assert rerank_model.load_run_metadata(config) == {"seed": 42}


# ─────────────────────────────────────────────────────────────────────────────
# train.py — synthetic data contract
# ─────────────────────────────────────────────────────────────────────────────


def _example_row(
    query: str,
    positive: str = "b001",
    negatives=("b002", "b003"),
    query_id: str | None = None,
) -> dict:
    """One synthetic-data row. ``query_id`` is omitted unless a test needs it.

    Omitted by default because the parser must keep accepting a row without one
    (``SynthExample.query_id`` is optional), and present in every test that
    goes through the split manifest — which is the only thing that can tie a
    row to a split, and is therefore required by anything that trains.
    """
    row = {
        "query": query,
        "positive_business_id": positive,
        "negative_business_ids": list(negatives),
    }
    if query_id is not None:
        row["query_id"] = query_id
    return row


def test_parse_example_reads_the_agreed_shape():
    example = rerank_train.parse_example(_example_row("quiet cafe"))
    assert example.query == "quiet cafe"
    assert example.positive_business_id == "b001"
    assert example.negative_business_ids == ("b002", "b003")
    assert example.query_id is None


def test_parse_example_names_what_was_missing():
    """A renamed key that defaulted to empty would train on positives with no
    negatives and still report a falling loss."""
    with pytest.raises(KeyError, match="positive_business_id"):
        rerank_train.parse_example({"query": "quiet cafe"})


def test_parse_example_rejects_an_empty_query():
    with pytest.raises(ValueError, match="empty query"):
        rerank_train.parse_example(_example_row("   "))


def test_parse_example_reads_the_file_the_miner_actually_writes(tmp_path):
    """Cross-module seam test: mine with the real writer, parse with this reader.

    `concierge.synth.negatives` stores each negative as an object carrying the
    retrieval rank and score that mined it, under the key ``negatives``. This
    parser used to read only ``negative_business_ids``, with a `.get(...) or ()`
    fallback, so the mined file parsed cleanly into *zero* negatives — and a
    cross-encoder fine-tuned on positives alone still reports a falling loss, so
    no later stage would have caught it.
    """
    from concierge.synth.negatives import HardNegative, TrainingExample, write_examples_jsonl
    from concierge.types import QueryType

    mined = TrainingExample(
        query_id="s001",
        query="quiet cafe with wifi",
        query_type=QueryType.VIBE,
        positive_business_id="b001",
        negatives=(
            HardNegative(
                business_id="b002", retrieval_rank=1, retrieval_score=0.9, source="hybrid"
            ),
            HardNegative(
                business_id="b003", retrieval_rank=4, retrieval_score=0.7, source="hybrid"
            ),
        ),
        positive_retrieval_rank=2,
    )
    path = write_examples_jsonl([mined], tmp_path / "crossencoder_train.jsonl")

    example = rerank_train.parse_example(rerank_train.read_jsonl(path)[0])

    assert example.query == "quiet cafe with wifi"
    assert example.positive_business_id == "b001"
    assert example.negative_business_ids == ("b002", "b003")
    assert example.query_id == "s001"


def test_parse_example_refuses_a_row_with_no_negatives_key():
    """The docstring's promise, now actually kept: a row that mentions no
    negatives at all is a key mismatch, not an example with zero negatives."""
    with pytest.raises(KeyError, match="no negatives"):
        rerank_train.parse_example({"query": "quiet cafe", "positive_business_id": "b001"})


def test_parse_example_allows_an_explicitly_empty_negatives_list():
    """Distinct from the case above: the miner records queries whose retrieval
    pool yielded nothing, and silently dropping them would hide weak recall."""
    example = rerank_train.parse_example(
        {"query": "quiet cafe", "positive_business_id": "b001", "negatives": []}
    )
    assert example.negative_business_ids == ()


def test_parse_example_rejects_a_negative_object_without_a_business_id():
    with pytest.raises(KeyError, match="business_id"):
        rerank_train.parse_example(
            {
                "query": "quiet cafe",
                "positive_business_id": "b001",
                "negatives": [{"retrieval_rank": 1}],
            }
        )


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_find_split_file_searches_both_accepted_layouts(config, tmp_path):
    _write_jsonl(tmp_path / "splits" / "crossencoder_train.jsonl", [_example_row("a")])
    found = rerank_train.find_split_file("crossencoder_train", config, roots=[tmp_path])
    assert found.name == "crossencoder_train.jsonl"


def test_missing_split_lists_every_path_it_searched(config, tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        rerank_train.find_split_file("crossencoder_train", config, roots=[tmp_path])
    message = str(excinfo.value)
    assert "crossencoder_train" in message
    assert str(tmp_path) in message
    # The trainer must never silently substitute another split for the one asked
    # for: that would break the out-of-fold guarantee Stage 4 depends on.
    assert "never re-splits" in message


def test_empty_split_is_an_error_not_an_empty_run(config, tmp_path):
    _write_jsonl(tmp_path / "crossencoder_train.jsonl", [])
    with pytest.raises(ValueError, match="is empty"):
        # manifest=None: this test is about the file's shape, and the keyword is
        # spelled out at the call site precisely so skipping the membership
        # check is never the quiet default.
        rerank_train.load_split("crossencoder_train", config, roots=[tmp_path], manifest=None)


def test_the_only_search_root_is_the_fingerprinted_one(config):
    """No fallback to an unfingerprinted or processed directory. Ever.

    The split manifest is keyed by the config fingerprint, so after any config
    change the fingerprinted directory is empty. A fallback chain then loads the
    PREVIOUS configuration's ``crossencoder_train.jsonl`` with no error: the
    cross-encoder fine-tunes on queries the current manifest assigns to
    LambdaMART, and every out-of-fold check still passes because each manifest
    is internally consistent on its own. Failing to find the file costs a
    minute; training on the wrong split is not detectable at all.
    """
    roots = rerank_train._search_roots(config)

    assert [r.name for r in roots] == [config.fingerprint_for(*synth_splits.SYNTH_SECTIONS)]
    # Named explicitly: the directories the old fallback chain searched.
    searched = {str(r) for r in roots}
    assert not any(s.endswith("/artifacts/synth") for s in searched)
    assert not any("processed" in s for s in searched)


# ─────────────────────────────────────────────────────────────────────────────
# train.py — the split manifest, which is what ties this model to a split
# ─────────────────────────────────────────────────────────────────────────────
#
# Stage 4's out-of-fold guarantee is a claim about which queries THIS model was
# fitted on. Before these tests the claim rested on a filename: a file called
# `crossencoder_train.jsonl` was assumed to hold the crossencoder_train split,
# the module never read the manifest at all, and a comment at the top of the
# file said splits were "read from disk here". Each test below plants one
# failure in an otherwise valid manifest and asserts the trainer refuses.


def _write_manifest(path, cfg, by_split, *, seed=None, config_fingerprint=None):
    """Write a manifest the real reader accepts, unless a keyword breaks it.

    Checksum, seed, fractions and fingerprint default to agreeing with ``cfg``
    because that is what ``concierge.synth.splits`` verifies. (The LambdaMART
    tests have their own copy of this helper: a shared fixture would couple the
    two suites, and each file is meant to be readable on its own.)
    """
    assignment = {qid: split for split, ids in by_split.items() for qid in ids}
    seed = cfg.seed if seed is None else seed
    fractions = synth_splits.fractions_from_config(cfg)
    group_by = "query"
    manifest = synth_splits.SplitManifest(
        seed=seed,
        group_by=group_by,
        fractions=fractions,
        assignment=assignment,
        checksum=synth_splits.compute_checksum(assignment, seed, fractions, group_by),
        n_groups=len(assignment),
        config_fingerprint=(
            # The synth key, matching what `synth.splits.assign_splits` records
            # and what both trainers now compare against (artifacts are keyed on
            # the sections they depend on, not the whole config — see
            # `Config.fingerprint_for` / DECISIONS.md D237).
            cfg.fingerprint_for(*synth_splits.SYNTH_SECTIONS)
            if config_fingerprint is None
            else config_fingerprint
        ),
    )
    return synth_splits.save_manifest(manifest, path)


def _verified(cfg, path):
    manifest, _ = rerank_train.load_verified_manifest(cfg, path)
    return manifest


def test_rows_the_manifest_assigns_to_this_split_load(config, tmp_path):
    manifest = _verified(
        config,
        _write_manifest(
            tmp_path / "splits.json",
            config,
            {rerank_train.TRAIN_SPLIT: ["s001"], rerank_train.VAL_SPLIT: ["s002"]},
        ),
    )
    _write_jsonl(
        tmp_path / "crossencoder_train.jsonl", [_example_row("quiet cafe", query_id="s001")]
    )

    examples = rerank_train.load_split(
        rerank_train.TRAIN_SPLIT, config, roots=[tmp_path], manifest=manifest
    )

    assert [e.query_id for e in examples] == ["s001"]


def test_a_row_the_manifest_assigns_elsewhere_is_refused(config, tmp_path):
    """The stale-file leak, in the shape it actually arrives in.

    A file named for the cross-encoder's split holding a query the manifest
    assigns to LambdaMART means the cross-encoder would be fine-tuned on
    queries LambdaMART treats as unseen — and LambdaMART's own check would
    still pass, because its manifest is internally consistent.
    """
    manifest = _verified(
        config,
        _write_manifest(
            tmp_path / "splits.json",
            config,
            {rerank_train.TRAIN_SPLIT: ["s001"], "ltr_train": ["s999"]},
        ),
    )
    _write_jsonl(
        tmp_path / "crossencoder_train.jsonl", [_example_row("quiet cafe", query_id="s999")]
    )

    with pytest.raises(synth_splits.SplitLeakageError, match="assigned to another split"):
        rerank_train.load_split(
            rerank_train.TRAIN_SPLIT, config, roots=[tmp_path], manifest=manifest
        )


def test_a_row_without_a_query_id_is_refused_not_skipped(config, tmp_path):
    """An id is the only thing checkable against the manifest.

    Skipping unidentified rows would mean the model's training set was partly
    unaccounted for, which is the state this check exists to end — so a file
    without ids is refused rather than partly verified.
    """
    manifest = _verified(
        config,
        _write_manifest(tmp_path / "splits.json", config, {rerank_train.TRAIN_SPLIT: ["s001"]}),
    )
    _write_jsonl(tmp_path / "crossencoder_train.jsonl", [_example_row("quiet cafe")])

    with pytest.raises(synth_splits.SplitLeakageError, match="no 'query_id'"):
        rerank_train.load_split(
            rerank_train.TRAIN_SPLIT, config, roots=[tmp_path], manifest=manifest
        )


def test_a_row_the_manifest_never_assigned_is_refused(config, tmp_path):
    manifest = _verified(
        config,
        _write_manifest(tmp_path / "splits.json", config, {rerank_train.TRAIN_SPLIT: ["s001"]}),
    )
    _write_jsonl(
        tmp_path / "crossencoder_train.jsonl", [_example_row("quiet cafe", query_id="s404")]
    )

    with pytest.raises(synth_splits.SplitLeakageError, match="not in the split manifest"):
        rerank_train.load_split(
            rerank_train.TRAIN_SPLIT, config, roots=[tmp_path], manifest=manifest
        )


def test_a_hand_edited_manifest_is_refused(config, tmp_path):
    """Only the checksum notices an edit, which is why the manifest is read
    through ``concierge.synth.splits`` rather than parsed here."""
    path = _write_manifest(
        tmp_path / "splits.json",
        config,
        {rerank_train.TRAIN_SPLIT: ["s001"], "ltr_train": ["s002"]},
    )
    edited = json.loads(path.read_text())
    edited["assignment"]["s002"] = rerank_train.TRAIN_SPLIT  # checksum left alone
    path.write_text(json.dumps(edited))

    with pytest.raises(synth_splits.SplitManifestCorruptError, match="checksum"):
        rerank_train.load_verified_manifest(config, path)


def test_a_missing_manifest_is_an_error_not_a_pass(config, tmp_path):
    with pytest.raises(synth_splits.SplitManifestCorruptError, match="no split manifest"):
        rerank_train.load_verified_manifest(config, tmp_path / "nope.json")


def test_a_manifest_from_another_seed_or_configuration_is_refused(config, tmp_path):
    seeded = _write_manifest(
        tmp_path / "seed.json",
        config,
        {rerank_train.TRAIN_SPLIT: ["s001"]},
        seed=config.seed + 1,
    )
    with pytest.raises(synth_splits.SplitLeakageError, match="seed"):
        rerank_train.load_verified_manifest(config, seeded)

    # Same ids, same seed, built under a different config: a different split.
    other_config = _write_manifest(
        tmp_path / "fingerprint.json",
        config,
        {rerank_train.TRAIN_SPLIT: ["s001"]},
        config_fingerprint="0" * 10,
    )
    with pytest.raises(synth_splits.SplitLeakageError, match="fingerprint"):
        rerank_train.load_verified_manifest(config, other_config)


def test_a_manifest_with_no_config_fingerprint_is_accepted_but_said_out_loud(
    config, tmp_path, capsys
):
    """An unchecked manifest must not look like a checked, clean one.

    A manifest predating the fingerprint field, or written by hand, cannot be
    tied to any configuration — only to its seed and fractions. It is still
    trainable, so this is a warning and not a refusal. But it was previously
    silent on this side while ``concierge.ltr.train`` warned about the same
    thing, and the consequence of the silence was not local: this fine-tune's
    metadata records a checksum, LambdaMART matches that checksum, and the two
    artifacts then agree with each other while neither is tied to a config.
    """
    untied = _write_manifest(
        tmp_path / "untied.json",
        config,
        {rerank_train.TRAIN_SPLIT: ["s001"]},
        config_fingerprint="",
    )

    manifest, resolved = rerank_train.load_verified_manifest(config, untied)

    assert resolved == untied
    assert manifest.config_fingerprint == ""
    assert "records no config fingerprint" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# Artifact layout — both trainers must resolve the paths the writer writes
# ─────────────────────────────────────────────────────────────────────────────
#
# `concierge.synth.splits` documents one directory for everything a trainer
# reads: `artifacts/synth/<config fingerprint>/`, holding the checksummed
# `splits.json` and one JSONL per split. That contract had been broken in three
# silent ways, which is why it is asserted here rather than trusted:
#
#   * the cross-encoder trainer spelled `artifacts/synth/<fingerprint>` out
#     itself instead of asking the writer, so relocating the synth artifacts
#     would leave the reader searching a directory nothing writes any more — a
#     FileNotFoundError on a clean tree, but a hit on the previous layout's
#     leftover file on a tree that has run before;
#   * the LambdaMART trainer defaulted to `data/processed/ltr_examples.jsonl`,
#     which is not keyed by the config fingerprint and which nothing in the
#     repository writes, so the only way to populate it was by hand — and a
#     hand-placed file is the one way to get rows into that trainer without
#     going through the manifest;
#   * nothing compared the two trainers' defaults, so "both models were trained
#     from one directory of artifacts" was an assumption rather than a check.
#
# The layout tests pass `root=tmp_path` instead of using the repo's own
# `artifacts/`, so they pin the *shape* of the layout and cannot pass or fail
# because of what happens to be on this machine's disk.


def test_the_documented_layout_is_what_every_path_function_returns(config, tmp_path):
    """One directory per configuration, and these are the names in it."""
    expected_dir = (
        tmp_path / "artifacts" / "synth" / config.fingerprint_for(*synth_splits.SYNTH_SECTIONS)
    )

    assert synth_splits.splits_path(config, root=tmp_path) == expected_dir / "splits.json"
    for split in synth_splits.SPLIT_NAMES:
        assert (
            synth_splits.split_example_path(split, config, root=tmp_path)
            == expected_dir / f"{split}.jsonl"
        )
    assert (
        ltr_train.default_examples_path(config, root=tmp_path)
        == expected_dir / ltr_train.EXAMPLES_FILENAME
    )


def test_both_trainers_resolve_into_the_one_fingerprinted_directory(config):
    """The split manifest, the cross-encoder's split files and LambdaMART's
    feature rows are artifacts of one configuration, so they must resolve into
    one directory keyed by that configuration's fingerprint.

    Asserted as a three-way equality rather than against a literal path: the
    point is that the three agree with each other and with the fingerprint, and
    a literal would have to be edited (and could be edited wrongly) if the
    layout ever moved.
    """
    manifest_dir = ltr_train.default_manifest_path(config).parent
    examples_dir = ltr_train.default_examples_path(config).parent
    (rerank_root,) = rerank_train._search_roots(config)

    assert manifest_dir == examples_dir == rerank_root
    assert manifest_dir.name == config.fingerprint_for(*synth_splits.SYNTH_SECTIONS)


def test_the_ltr_trainer_defaults_to_the_fingerprinted_examples_file(config, tmp_path, monkeypatch):
    """``--examples`` unset must mean the fingerprinted path, not a fixed one.

    Checked by recording the path the run asks for rather than by letting it
    read a file, so the assertion is about resolution and does not depend on
    whether this machine's `artifacts/` happens to be populated.
    """
    manifest_path = _write_manifest(
        tmp_path / "splits.json", config, {ltr_train.SPLIT_LTR_TRAIN: ["s001"]}
    )
    asked_for: list[Path] = []

    class _Stop(Exception):
        pass

    def _record(path):
        asked_for.append(Path(path))
        raise _Stop

    monkeypatch.setattr(ltr_train, "load_examples", _record)
    with pytest.raises(_Stop):
        ltr_train.run(config, None, manifest_path, allow_missing_gold=True)

    assert asked_for == [ltr_train.default_examples_path(config)]


def test_a_missing_examples_file_says_that_nothing_writes_it_yet(config, tmp_path):
    """The error must name the gap, not a stage that would fill it.

    This message used to say the rows were "produced by the synthetic-data
    stage over the ltr_train split". They are not: that stage writes
    query-level files and says so, and turning those into candidate-level
    feature rows needs the fine-tuned cross-encoder, so it can only happen
    after Stage 3. Sending the owner to `make synth` for a file `make synth`
    does not write costs an afternoon of looking for a bug that is not there.

    This test previously asserted the message said nothing writes the file,
    which was true until `concierge.ltr.build_examples` was added. It now
    asserts the message names that producer instead — the assertion moved with
    the behaviour rather than being deleted, so the message still cannot drift
    away from whatever actually builds the file.
    """
    missing = ltr_train.default_examples_path(config, root=tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        ltr_train.load_examples(missing)

    message = str(excinfo.value)
    assert str(missing) in message
    # Names the stage that builds it...
    assert "make ltr-examples" in message
    # ...and why that stage is not part of `make synth`, which is the mistake
    # the message exists to prevent.
    assert "make synth" in message


# ─────────────────────────────────────────────────────────────────────────────
# The cross-artifact seam: what the fine-tune recorded, read by LambdaMART
# ─────────────────────────────────────────────────────────────────────────────
#
# `concierge.ltr.train.crossencoder_run_check` reads this module's run metadata
# to decide whether the two trained models belong to the same experiment. The
# checksum comparison (tested in tests/test_ltr.py) covers the split. These
# tests cover the field added beside it: which config fingerprint the manifest
# carried when the fine-tune read it. The checksum cannot cover that —
# `compute_checksum` hashes the assignment, seed, fractions and grouping, not
# the fingerprint — so two manifests can be the same split and still have been
# written under different configurations.


def _rerank_metadata(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / rerank_model.RUN_METADATA_FILENAME
    path.write_text(json.dumps(payload))
    return path


def _ltr_manifest(config, tmp_path, **kwargs):
    """A manifest LambdaMART's loader accepts, with one ltr_train query."""
    return _write_manifest(
        tmp_path / "splits.json", config, {ltr_train.SPLIT_LTR_TRAIN: ["s001"]}, **kwargs
    )


def test_ltr_confirms_the_fine_tune_saw_this_configs_fingerprint(config, tmp_path, monkeypatch):
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path / "rerank")
    manifest, _ = ltr_train.load_verified_manifest(config, _ltr_manifest(config, tmp_path))
    _rerank_metadata(
        tmp_path / "rerank",
        {
            "split_manifest_checksum": manifest.checksum,
            "split_manifest_config_fingerprint": config.fingerprint_for(
                *synth_splits.SYNTH_SECTIONS
            ),
        },
    )

    result = ltr_train.crossencoder_run_check(config, manifest)

    assert result["checked"] is True
    assert result["config_fingerprint_tie"] == "matched"


def test_ltr_refuses_a_fine_tune_recorded_under_another_configuration(
    config, tmp_path, monkeypatch
):
    """Same split, other config: the out-of-fold property holds and the
    provenance does not.

    Reachable because the checksum is blind to the fingerprint, so a checkpoint
    copied in from another configuration — one with a different base model or
    ``max_length`` — passes every other check in that function. Its scores are
    then the feature column while this artifact records this config's
    hyperparameters.
    """
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path / "rerank")
    manifest, _ = ltr_train.load_verified_manifest(config, _ltr_manifest(config, tmp_path))
    _rerank_metadata(
        tmp_path / "rerank",
        {
            "split_manifest_checksum": manifest.checksum,
            "split_manifest_config_fingerprint": "0" * 10,
        },
    )

    with pytest.raises(ltr_train.LeakageError, match="fine-tuned"):
        ltr_train.crossencoder_run_check(config, manifest)


def test_ltr_reports_a_fine_tune_that_could_not_be_tied_to_a_config(
    config, tmp_path, monkeypatch, caplog
):
    """An untied fine-tune is reported, not refused — and it is distinguishable
    from an artifact that merely predates the field.

    Not refused because this run's own manifest check has already warned about
    the very same manifest: refusing twice for one cause is how an override
    flag gets added that then disables both. What must not happen is the
    LambdaMART artifact recording this as a clean check, so the two cases get
    two different values.
    """
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path / "rerank")
    manifest, _ = ltr_train.load_verified_manifest(
        config, _ltr_manifest(config, tmp_path, config_fingerprint="")
    )
    _rerank_metadata(
        tmp_path / "rerank",
        {
            "split_manifest_checksum": manifest.checksum,
            "split_manifest_config_fingerprint": "",
        },
    )

    with caplog.at_level("WARNING"):
        untied = ltr_train.crossencoder_run_check(config, manifest)

    assert untied["config_fingerprint_tie"] == "manifest_recorded_none"
    assert "no config fingerprint" in caplog.text

    # Same two models, but metadata written before the field existed: the key is
    # absent rather than empty, and that is a different problem (an old
    # artifact, not an untied one).
    _rerank_metadata(tmp_path / "rerank", {"split_manifest_checksum": manifest.checksum})

    older = ltr_train.crossencoder_run_check(config, manifest)

    assert older["config_fingerprint_tie"] == "not_recorded"


# ─────────────────────────────────────────────────────────────────────────────
# train.py — which query text is scored (train/serve skew)
# ─────────────────────────────────────────────────────────────────────────────


def test_training_query_text_is_the_string_serving_scores(parsed, documents):
    """THE train/serve skew test, asserted end to end through the reranker.

    Training used to build pairs from the raw synthetic query while serving
    scored the constraint-stripped ``semantic_text``. Nothing errored; the
    fine-tune was simply optimised on one input distribution and measured on
    another, which caps the fine-tuned-vs-pretrained delta that is one of this
    project's headline numbers.

    So the assertion is not "these two functions agree" but "the string the
    reranker hands the scorer at serving time is byte-identical to the string
    training put in the pair".
    """
    raw = "quiet cafe with wifi open past 10pm"
    scorer = ScriptedScorer({documents["b001"]: 1.0})
    reranker = CrossEncoderReranker(scorer)

    reranker.rerank(parse_rules(raw, today="Friday"), [_candidate("b001", 1)], documents, top_k=1)

    served_text = scorer.seen[0][0]
    assert served_text == rerank_train.training_query_text(raw)
    # And it really is the stripped text, not the raw query — otherwise the two
    # could agree by both being wrong.
    assert "10pm" not in served_text
    assert served_text != raw


def test_build_pairs_scores_the_stripped_query_on_both_labels(documents):
    """Every pair, positive and negative, carries the derived text."""
    raw = "quiet cafe with wifi open past 10pm"
    examples = [rerank_train.parse_example(_example_row(raw, query_id="s001"))]

    pairs, _ = rerank_train.build_pairs(examples, documents)

    expected = rerank_train.training_query_text(raw)
    assert {p.query for p in pairs} == {expected}
    assert expected != raw, "this query has constraints to strip, or the test proves nothing"


def test_training_text_does_not_depend_on_the_day(config):
    """The weekday handed to the rule parser must not reach the scored text.

    A day is needed because the parser leaves an hours phrase in the text when
    it cannot encode it, and an un-encoded phrase is an un-stripped phrase. The
    day itself lands only in ``ParsedQuery.open_at``, which the cross-encoder
    never sees — so this pins that invariance rather than assuming it, since
    the alternative would make the training text depend on when training ran.
    """
    raw = "quiet cafe with wifi open past 10pm on a weeknight"
    derived = {query_text_for(parse_rules(raw, today=day)) for day in DAYS}

    assert len(derived) == 1
    assert derived == {rerank_train.training_query_text(raw)}


def test_the_training_text_is_derived_from_the_public_training_parse():
    """``training_query_text`` must *be* ``query_text_for(training_parse(q))``.

    Both halves are public because both are needed: the reranker scores the text,
    and ``concierge.ltr.build_examples`` needs the whole parse the text came from
    — the same constraints and, crucially, the same weekday, so that the stage
    filters candidates for the day it scores text for. That stage used to
    re-derive the parse with its own copy of a private constant here
    (DECISIONS.md D228); this asserts the composition that replaced it, so a
    change to either function cannot leave the two with different parses.
    """
    for raw in ("quiet cafe with wifi open past 10pm", "somewhere to read", "cheap tacos"):
        assert training_query_text(raw) == query_text_for(training_parse(raw))

    constrained = training_parse("cheap tacos open past 10pm")
    assert constrained.open_at is not None, "the hours phrase must have been encoded"
    assert constrained.open_at.day in DAYS
    # Same query, same parse, every time — which is what lets the stage that
    # writes Stage 4's rows be byte-reproducible. `parse_latency_ms` is excluded
    # because it is a wall-clock measurement rather than part of the parse, and
    # it is for that same reason that `build_examples.record_for_candidate` does
    # not serialize it.
    again = training_parse("cheap tacos open past 10pm")
    assert replace(again, parse_latency_ms=0.0) == replace(constrained, parse_latency_ms=0.0)


def test_training_text_falls_back_to_the_raw_query_when_nothing_is_left(documents):
    """A query that is nothing but constraints must not become an empty string.

    ``query_text_for`` already falls back to the raw query for exactly this
    case; the point here is that training inherits the fallback instead of
    building a pair whose query side is "".
    """
    raw = "open past 10pm"
    assert rerank_train.training_query_text(raw).strip()


# ─────────────────────────────────────────────────────────────────────────────
# train.py — THE LEAKAGE GUARD
# ─────────────────────────────────────────────────────────────────────────────


def _write_gold(path: Path, queries: list[str]) -> Path:
    rows = ["query_id,query,query_type"]
    rows += [f"g{i:03d},{query},vibe" for i, query in enumerate(queries)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_leakage_check_passes_on_clean_training_data(tmp_path):
    gold = _write_gold(tmp_path / "gold_queries.csv", ["cozy spot to read"])
    examples = [rerank_train.parse_example(_example_row("quiet cafe with wifi"))]

    check = rerank_train.assert_no_gold_leakage(examples, gold)

    assert check.n_collisions == 0
    assert check.n_gold_queries == 1
    assert check.n_training_queries == 1
    # Recorded in the run metadata: a run with no evidence of this check should
    # not be trusted.
    assert check.gold_queries_path == str(gold)


def test_leakage_check_fires_on_a_planted_gold_query(tmp_path):
    """The most consequential test in this file.

    If a gold query trains the reranker, every NDCG@10 the project reports is
    inflated and nothing about the number looks wrong. A loud crash is infinitely
    better than a quietly inflated result, so this must fail, not warn.
    """
    gold = _write_gold(tmp_path / "gold_queries.csv", ["cozy spot to read", "good pho"])
    examples = [
        rerank_train.parse_example(_example_row("quiet cafe with wifi")),
        rerank_train.parse_example(_example_row("good pho")),  # planted
    ]

    with pytest.raises(rerank_train.GoldLeakageError, match="good pho"):
        rerank_train.assert_no_gold_leakage(examples, gold)


@pytest.mark.parametrize("variant", ["Good Pho", "  good   pho ", "GOOD PHO"])
def test_leakage_check_is_not_fooled_by_case_or_whitespace(tmp_path, variant):
    """Byte equality is a weaker guard than it looks: the same test query with
    different spacing would walk straight past it."""
    gold = _write_gold(tmp_path / "gold_queries.csv", ["good pho"])
    examples = [rerank_train.parse_example(_example_row(variant))]
    with pytest.raises(rerank_train.GoldLeakageError):
        rerank_train.assert_no_gold_leakage(examples, gold)


def test_leakage_check_refuses_to_run_without_the_gold_file(tmp_path):
    """ "We could not check" and "we checked and it was clean" must never look the
    same, so a missing gold file stops training instead of skipping the check."""
    examples = [rerank_train.parse_example(_example_row("quiet cafe"))]
    with pytest.raises(FileNotFoundError, match="Training is refused"):
        rerank_train.assert_no_gold_leakage(examples, tmp_path / "nope.csv")


def test_leakage_check_rejects_a_gold_file_without_a_query_column(tmp_path):
    path = tmp_path / "gold_queries.csv"
    path.write_text("query_id,text,query_type\ng001,good pho,specific\n")
    with pytest.raises(ValueError, match="no 'query' column"):
        rerank_train.assert_no_gold_leakage([], path)


def test_normalize_query_folds_case_whitespace_and_trailing_punctuation():
    assert rerank_train.normalize_query("  Quiet   Cafe ") == "quiet cafe"
    # "good pho?" and "good pho" are one query, not two.
    assert rerank_train.normalize_query("Good pho?") == "good pho"
    # It must NOT do fuzzy matching: near-duplicate removal is the synth stage's
    # job and uses embeddings, with a threshold in configs/base.yaml.
    assert rerank_train.normalize_query("quiet cafes") != rerank_train.normalize_query("quiet cafe")
    # Internal punctuation is meaning, not noise, and must survive.
    assert rerank_train.normalize_query("open 24/7, cheap") == "open 24/7, cheap"


def test_both_leakage_guards_normalize_identically():
    """Seam test: Stage 3 and Stage 4 each have a gold-leakage guard, and the
    weaker of the two is the one that decides whether a gold query reaches
    training. They drifted — Stage 4 stripped trailing punctuation and Stage 3
    did not — so "good pho?" was caught before LambdaMART training and waved
    through before an hour of cross-encoder fine-tuning. Pin them together.
    """
    from concierge.ltr.train import _normalize_query as ltr_normalize

    for text in (
        "Good pho?",
        "  Quiet   Cafe ",
        "open past 10pm.",
        "cheap ramen near downtown!",
        "WiFi + outdoor seating",
        "open 24/7, cheap",
    ):
        assert rerank_train.normalize_query(text) == ltr_normalize(text), text


# ─────────────────────────────────────────────────────────────────────────────
# train.py — pair building
# ─────────────────────────────────────────────────────────────────────────────


def test_build_pairs_labels_positives_and_negatives(documents):
    examples = [rerank_train.parse_example(_example_row("quiet cafe"))]
    pairs, stats = rerank_train.build_pairs(examples, documents)

    assert stats.n_positive_pairs == 1
    assert stats.n_negative_pairs == 2
    assert stats.n_pairs == 3
    assert pairs[0].label == 1.0
    assert pairs[0].document == documents["b001"]
    assert {p.label for p in pairs[1:]} == {0.0}


def test_build_pairs_counts_what_it_dropped(documents):
    """A shrinking dataset must be visible, not mysterious."""
    examples = [
        rerank_train.parse_example(_example_row("quiet cafe", negatives=("b002", "b999"))),
        rerank_train.parse_example(_example_row("no document", positive="b404")),
    ]
    pairs, stats = rerank_train.build_pairs(examples, documents)

    assert stats.n_examples == 2
    assert stats.n_examples_dropped_no_positive_doc == 1
    assert stats.n_negatives_dropped_no_doc == 1
    assert stats.n_pairs == len(pairs) == 2


def test_build_pairs_refuses_a_dataset_with_no_usable_text():
    examples = [rerank_train.parse_example(_example_row("quiet cafe"))]
    with pytest.raises(ValueError, match="no trainable pairs"):
        rerank_train.build_pairs(examples, {})


@pytest.mark.slow
def test_detect_device_honours_an_explicit_request():
    """``--device cpu`` is the documented escape hatch when an mps kernel
    misbehaves, so an explicit request must win over detection.

    Marked slow despite being instant because it is the only non-model test that
    imports torch, and on macOS a process that has imported torch segfaults
    inside LightGBM (two copies of libomp ship, one in faiss, one in torch — see
    ``build_index.duplicate_libomp_paths``). Keeping torch out of the default
    in-process run is what lets the Stage 3 and Stage 4 suites share one pytest
    invocation.
    """
    assert rerank_train.detect_device("cpu") == "cpu"
    assert rerank_train.detect_device(None) in {"cpu", "mps", "cuda"}


# ─────────────────────────────────────────────────────────────────────────────
# ONNX parity bookkeeping (no onnxruntime needed)
# ─────────────────────────────────────────────────────────────────────────────


def _parity(
    max_abs_diff: float,
    ordering_identical: bool = True,
    max_abs_diff_single_pair: float = 0.0,
) -> ParityResult:
    return ParityResult(
        n_pairs=12,
        tolerance=DEFAULT_TOLERANCE,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=max_abs_diff / 2,
        max_abs_diff_single_pair=max_abs_diff_single_pair,
        ordering_identical=ordering_identical,
        torch_ms_per_pair=10.0,
        onnx_ms_per_pair=4.0,
        speedup=2.5,
        max_length=512,
        weights_source="pretrained base_model",
        onnx_path="artifacts/rerank/x/model.onnx",
    )


def test_parity_tolerance_boundary_is_inclusive():
    assert _parity(DEFAULT_TOLERANCE).within_tolerance
    assert not _parity(DEFAULT_TOLERANCE * 10).within_tolerance


def test_parity_fails_when_only_the_unbatched_scores_disagree():
    """A graph that froze its batch or sequence axis can match perfectly inside the
    batch shape it was exported with and drift everywhere else."""
    result = _parity(0.0, max_abs_diff_single_pair=DEFAULT_TOLERANCE * 10)
    assert not result.within_tolerance
    assert "unbatched=1.000e-03" in result.summary()


def test_parity_summary_carries_the_measured_numbers():
    summary = _parity(1e-6).summary()
    assert "max|Δ|=1.000e-06" in summary
    assert "ordering identical" in summary
    assert "2.50x" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Slow tests: these really load a model
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_real_scorer_scores_and_measures_truncation(config):
    """Sanity-checks the real model end to end: the relevant document must
    outscore the irrelevant one, and the truncation diagnostic must report real
    tokenizer counts."""
    from concierge.rerank.model import TransformersCrossEncoderScorer

    scorer = TransformersCrossEncoderScorer(
        config.rerank.base_model, max_length=config.rerank.max_length, batch_size=4
    )
    reranker = CrossEncoderReranker(scorer, weights_source="pretrained (test)")

    relevant = "Corner Coffee. Coffee shop, cafe. free WiFi, quiet, outdoor seating."
    irrelevant = "City Auto Body. Car repair, towing, collision work."
    scores = reranker.score("quiet cafe with wifi", [relevant, irrelevant])
    assert scores[0] > scores[1]

    long_document = "Rich broth and springy noodles. " * 400
    report = reranker.truncation_report("quiet cafe with wifi", [relevant, long_document])
    assert report.document_token_budget < config.rerank.max_length
    assert report.n_truncated == 1
    assert report.stats[1].kept_fraction < 1.0


@pytest.mark.slow
def test_training_smoke_run_writes_a_checkpoint_and_metadata(config, tmp_path):
    """A real (tiny) fine-tune: two epochs over a handful of pairs.

    Not a quality test — the loss over six pairs means nothing. It pins the
    plumbing: splits are read from disk, the leakage check runs before training,
    and the checkpoint plus run metadata land where ``model.py`` looks for them.
    """
    gold = _write_gold(tmp_path / "gold_queries.csv", ["cozy spot to read"])
    _write_jsonl(
        tmp_path / "crossencoder_train.jsonl",
        [
            _example_row("quiet cafe with wifi", "b001", ("b002",), query_id="s001"),
            _example_row("loud sports bar", "b002", ("b001",), query_id="s002"),
        ],
    )
    # A distinct query: the manifest assigns each id to exactly one split, so a
    # validation row cannot reuse a training row's id — which is the property
    # that makes the split checkable in the first place.
    _write_jsonl(
        tmp_path / "crossencoder_val.jsonl",
        [_example_row("cafe with free wifi", "b001", ("b002",), query_id="s003")],
    )

    small = config.model_copy(
        update={
            "rerank": config.rerank.model_copy(
                update={"epochs": 1, "batch_size": 2, "max_length": 64}
            )
        }
    )
    manifest_path = _write_manifest(
        tmp_path / "splits.json",
        small,
        {rerank_train.TRAIN_SPLIT: ["s001", "s002"], rerank_train.VAL_SPLIT: ["s003"]},
    )
    meta = rerank_train.train(
        small,
        device="cpu",
        out_dir=tmp_path / "artifact",
        gold_path=gold,
        data_roots=[tmp_path],
        splits_manifest=manifest_path,
        documents={
            "b001": "Corner Coffee. quiet cafe with free wifi.",
            "b002": "Loud Sports Bar. loud bar with tvs.",
        },
    )

    assert meta.leakage_check["n_collisions"] == 0
    assert len(meta.loss_curve) == 1
    assert meta.wall_clock_seconds is not None and meta.wall_clock_seconds > 0
    checkpoint = tmp_path / "artifact" / rerank_model.CHECKPOINT_DIRNAME
    assert (checkpoint / "config.json").exists()
    saved = json.loads((tmp_path / "artifact" / rerank_model.RUN_METADATA_FILENAME).read_text())
    # The WHOLE config: this field is provenance for a reader of the artifact,
    # not the directory key. The scoped keys are asserted in
    # `test_rerank_dir_is_keyed_by_config_fingerprint`.
    assert saved["config_fingerprint"] == small.fingerprint()
    assert saved["train_split"] == rerank_train.TRAIN_SPLIT
    # The checksum is the field `concierge.ltr.train` compares against the
    # manifest it verifies, so an artifact without it cannot support the
    # out-of-fold claim. A path alone could not: the file at a path can be
    # regenerated.
    assert saved["split_manifest_checksum"] == synth_splits.load_manifest(manifest_path).checksum
    assert saved["split_manifest"] == str(manifest_path)
    assert saved["split_manifest_seed"] == small.seed
    # Recorded beside the checksum so the artifact itself says whether this
    # fine-tune was tied to a configuration: `concierge.ltr.train` compares it,
    # and the checksum cannot stand in for it (compute_checksum does not hash
    # the fingerprint, so one split can have been written under two configs).
    assert saved["split_manifest_config_fingerprint"] == small.fingerprint_for(
        *synth_splits.SYNTH_SECTIONS
    )
    assert saved["query_text_source"] == "rules_parse.semantic_text"


@pytest.mark.slow
def test_training_refuses_a_leaky_split_before_touching_weights(config, tmp_path):
    """Order of operations: the crash must come from the leakage check, not from
    an hour of wasted compute followed by a retracted number."""
    gold = _write_gold(tmp_path / "gold_queries.csv", ["good pho"])
    _write_jsonl(tmp_path / "crossencoder_train.jsonl", [_example_row("good pho", query_id="s001")])
    _write_jsonl(tmp_path / "crossencoder_val.jsonl", [_example_row("quiet cafe", query_id="s002")])
    # The manifest is clean, so the gold check is what fires. That ordering is
    # the point: both guards run before any weight is touched, and this test
    # would pass for the wrong reason if the manifest were the thing refusing.
    manifest_path = _write_manifest(
        tmp_path / "splits.json",
        config,
        {rerank_train.TRAIN_SPLIT: ["s001"], rerank_train.VAL_SPLIT: ["s002"]},
    )

    with pytest.raises(rerank_train.GoldLeakageError):
        rerank_train.train(
            config,
            device="cpu",
            save=False,
            gold_path=gold,
            data_roots=[tmp_path],
            splits_manifest=manifest_path,
            documents={"b001": "a", "b002": "b", "b003": "c"},
        )


@pytest.mark.slow
def test_onnx_export_matches_pytorch_scores(config, tmp_path):
    """The parity check this project's served results depend on."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from concierge.rerank import onnx_export

    small = config.model_copy(
        update={"rerank": config.rerank.model_copy(update={"max_length": 128})}
    )
    path, weights_source = onnx_export.export(small, out_path=tmp_path / "model.onnx")
    assert path.exists()
    assert "pretrained" in weights_source or "fine-tuned" in weights_source

    result = onnx_export.check_parity(
        small,
        path,
        pairs=[
            ("quiet cafe with wifi", "Corner Coffee. quiet cafe, free WiFi."),
            ("quiet cafe with wifi", "City Auto Body. car repair and towing."),
            ("cheap ramen", "Noodle House. Ramen. " + ("broth and noodles. " * 100)),
        ],
    )
    assert result.within_tolerance
    assert result.ordering_identical
    assert result.n_pairs == 3
