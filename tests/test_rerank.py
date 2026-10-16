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

Tests that genuinely load a model or export ONNX are marked ``slow``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

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
)
from concierge.rerank.onnx_export import DEFAULT_TOLERANCE, ParityResult
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
    assert directory == tmp_path / "rerank" / config.fingerprint()
    # Changing a reranking hyperparameter must not reuse the old checkpoint.
    other = config.model_copy(update={"rerank": config.rerank.model_copy(update={"lr": 5e-5})})
    assert rerank_dir(other, root=tmp_path) != directory


def test_load_run_metadata_absent_is_none(config, tmp_path, monkeypatch):
    monkeypatch.setattr(rerank_model, "rerank_dir", lambda cfg, root=None: tmp_path)
    assert rerank_model.load_run_metadata(config) is None
    (tmp_path / rerank_model.RUN_METADATA_FILENAME).write_text(json.dumps({"seed": 42}))
    assert rerank_model.load_run_metadata(config) == {"seed": 42}


# ─────────────────────────────────────────────────────────────────────────────
# train.py — synthetic data contract
# ─────────────────────────────────────────────────────────────────────────────


def _example_row(query: str, positive: str = "b001", negatives=("b002", "b003")) -> dict:
    return {
        "query": query,
        "positive_business_id": positive,
        "negative_business_ids": list(negatives),
    }


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
        rerank_train.load_split("crossencoder_train", config, roots=[tmp_path])


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
    rows = [
        _example_row("quiet cafe with wifi", "b001", ("b002",)),
        _example_row("loud sports bar", "b002", ("b001",)),
    ]
    _write_jsonl(tmp_path / "crossencoder_train.jsonl", rows)
    _write_jsonl(tmp_path / "crossencoder_val.jsonl", rows[:1])

    small = config.model_copy(
        update={
            "rerank": config.rerank.model_copy(
                update={"epochs": 1, "batch_size": 2, "max_length": 64}
            )
        }
    )
    meta = rerank_train.train(
        small,
        device="cpu",
        out_dir=tmp_path / "artifact",
        gold_path=gold,
        data_roots=[tmp_path],
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
    assert saved["config_fingerprint"] == small.fingerprint()
    assert saved["train_split"] == rerank_train.TRAIN_SPLIT


@pytest.mark.slow
def test_training_refuses_a_leaky_split_before_touching_weights(config, tmp_path):
    """Order of operations: the crash must come from the leakage check, not from
    an hour of wasted compute followed by a retracted number."""
    gold = _write_gold(tmp_path / "gold_queries.csv", ["good pho"])
    _write_jsonl(tmp_path / "crossencoder_train.jsonl", [_example_row("good pho")])
    _write_jsonl(tmp_path / "crossencoder_val.jsonl", [_example_row("quiet cafe")])

    with pytest.raises(rerank_train.GoldLeakageError):
        rerank_train.train(
            config,
            device="cpu",
            save=False,
            gold_path=gold,
            data_roots=[tmp_path],
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
