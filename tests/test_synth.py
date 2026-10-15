"""Tests for the synthetic training data pipeline.

No API key, no model download, no Yelp data. Every network and embedding
dependency is injected, which is the point of the protocols in
``concierge.synth`` — a leakage guard that could only be tested with a live
sentence-transformer would never actually be tested.

The five tests that exist because the project would be broken without them:

* ``test_gold_dedup_removes_planted_near_duplicate`` — the leakage guard. A
  paraphrase of a gold query is planted in the synthetic set and must be removed.
* ``test_splits_are_disjoint_*`` — the out-of-fold guarantee LambdaMART depends on.
* ``test_split_assignment_is_deterministic_*`` — a split that moved between runs
  would leak differently every time and no test could catch it afterwards.
* ``test_judge_never_writes_qrels`` (three of them) — the judge must be
  structurally incapable of writing gold labels.
* ``test_generation_resumes_after_rate_limit`` — a 429 halfway through must not
  throw away calls that were already paid for.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from concierge.synth import generate, judge, negatives, splits
from concierge.types import BusinessDocument, GoldQuery, QueryType

#: The ``needs_api`` marker is registered in pyproject.toml but nothing in
#: tests/conftest.py turns it into a skip, so the skip condition lives here. A
#: marker that does not actually skip would fail the suite on every machine
#: without a key, which is every machine in CI.
requires_api_key = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is not set; live-API test skipped",
)

# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────


class StubClient:
    """An ``LlmJsonClient`` that replays canned payloads and counts calls.

    ``fail_after`` makes the n-th call raise a rate limit, which is how the
    resumability test simulates a free-tier 429 without waiting for one.
    """

    name = "stub"
    model = "stub-model"

    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        fail_after: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payloads = payloads
        self.fail_after = fail_after
        self.error = error or generate.SynthRateLimitError("stub 429")
        self.calls: list[str] = []

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            self.calls.append(prompt)
            raise self.error
        self.calls.append(prompt)
        index = len(self.calls) - 1
        return self.payloads[index] if index < len(self.payloads) else {"businesses": []}


FAKE_EMBED_DIM = 512


def fake_embed(texts):
    """A hashed bag-of-words embedding: deterministic, no model, no download.

    Each word is hashed into one of ``FAKE_EMBED_DIM`` slots, so two texts sharing
    most of their words get a high cosine and two unrelated texts get a low one.
    That is the only property the near-duplicate guard relies on, so the guard can
    be tested without asserting anything about bge's actual geometry.

    The dimension is **fixed**, not derived from the input, because the real
    encoder's is: the guard embeds the gold set and each chunk of candidates in
    separate calls and multiplies the results, so a vocabulary-sized vector would
    not even be a valid stand-in.
    """
    import numpy as np

    matrix = np.zeros((len(texts), FAKE_EMBED_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in generate.normalize_query_text(text).split():
            slot = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % FAKE_EMBED_DIM
            matrix[row, slot] += 1.0
    return matrix


def doc(business_id: str, text: str = "a place") -> BusinessDocument:
    return BusinessDocument(business_id=business_id, text=text)


def synth_query(
    business_id: str,
    text: str,
    qtype: QueryType = QueryType.VIBE,
) -> generate.SyntheticQuery:
    return generate.SyntheticQuery(
        query_id=generate.synthetic_query_id(business_id, text),
        query=text,
        query_type=qtype,
        business_id=business_id,
    )


def payload_for(business_id: str, queries: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "businesses": [
            {
                "business_id": business_id,
                "queries": [{"query": q, "type": t} for q, t in queries],
            }
        ]
    }


@pytest.fixture
def store(tmp_path: Path) -> generate.GenerationStore:
    return generate.GenerationStore(
        responses=generate.JsonlCache(tmp_path / "responses.jsonl", "batch_key"),
        queries=generate.JsonlCache(tmp_path / "queries.jsonl", "query_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt and normalization
# ─────────────────────────────────────────────────────────────────────────────


def test_generation_prompt_renders_every_placeholder():
    prompt = generate.render_generation_prompt([doc("b001", "Lantern Coffee, quiet")], 4)
    assert "{{" not in prompt
    assert "b001" in prompt and "Lantern Coffee, quiet" in prompt
    assert "4" in prompt
    for qtype in QueryType:
        assert qtype.value in prompt


def test_generation_prompt_strips_the_repo_facing_header():
    """The leakage rule is documentation for whoever edits the file, not an
    instruction for the model: telling the model about the test set is the thing
    the rule forbids."""
    prompt = generate.render_generation_prompt([doc("b001")], 2)
    assert "prompt_version" not in prompt
    assert "LEAKAGE RULE" not in prompt


def test_prompt_contains_no_gold_queries(tmp_path: Path):
    """The first line of defence: no gold query may appear as a prompt example.

    The gold file does not exist yet (the owner writes it), so this test asserts
    the invariant against whatever is on disk and skips cleanly when it is absent
    rather than silently passing forever.
    """
    gold_path = Path(generate.REPO_ROOT) / "eval" / "gold_queries.csv"
    if not gold_path.exists():
        pytest.skip("eval/gold_queries.csv does not exist yet (owner writes it)")
    gold = generate.load_gold_queries(gold_path)
    body = generate.prompt_text().casefold()
    for query in gold:
        assert generate.normalize_query_text(query.query) not in body, (
            f"gold query {query.query_id} appears in generate_queries.txt"
        )


def test_judge_prompt_renders_rubric_and_pairs():
    j = judge.LlmJudge(client=None, prompt_version="test")
    pair = judge.JudgePair("q1", "quiet cafe", "b001", "Lantern Coffee")
    prompt = j.render_prompt([pair])
    assert "{{" not in prompt
    assert pair.pair_id in prompt
    assert "quiet cafe" in prompt and "Lantern Coffee" in prompt
    # The rubric must be the injected one, so judge and owner grade by one wording.
    assert judge.RUBRIC[2] in prompt


def test_normalize_query_text_folds_case_and_edges_only():
    assert generate.normalize_query_text("  Quiet Cafe?  ") == "quiet cafe"
    # Internal punctuation is meaning, not noise, and the messy query type depends on it.
    assert generate.normalize_query_text("cheap ramen, open late") == "cheap ramen, open late"


def test_query_ids_are_content_addressed():
    """Stable ids are what keep the split manifest stable as data is added."""
    a = generate.synthetic_query_id("b001", "quiet cafe")
    assert a == generate.synthetic_query_id("b001", "  Quiet Cafe  ")
    assert a != generate.synthetic_query_id("b002", "quiet cafe")


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_generation_response_accepts_valid_queries():
    batch = [doc("b001")]
    parsed = generate.parse_generation_response(
        payload_for("b001", [("quiet spot to read", "vibe"), ("open after 10pm", "constraint")]),
        batch,
    )
    assert [q.query for q in parsed.queries] == ["quiet spot to read", "open after 10pm"]
    assert parsed.queries[1].query_type is QueryType.CONSTRAINT
    assert parsed.n_rejected == 0


def test_parse_generation_response_rejects_unknown_business_id():
    """A hallucinated id must never be remapped by position: attaching a query to
    the wrong positive is a corrupt label, which is worse than a missing one."""
    parsed = generate.parse_generation_response(
        payload_for("b999", [("quiet spot", "vibe")]), [doc("b001")]
    )
    assert parsed.queries == ()
    assert parsed.n_rejected == 1
    assert "b999" in parsed.rejection_reasons[0]


def test_parse_generation_response_rejects_unknown_query_type():
    parsed = generate.parse_generation_response(
        payload_for("b001", [("quiet spot", "atmospheric")]), [doc("b001")]
    )
    assert parsed.queries == ()
    assert parsed.n_rejected == 1


def test_parse_generation_response_drops_within_reply_duplicates():
    parsed = generate.parse_generation_response(
        payload_for("b001", [("quiet spot", "vibe"), ("Quiet Spot", "vibe")]), [doc("b001")]
    )
    assert len(parsed.queries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Quota confirmation (CLAUDE.md §10)
# ─────────────────────────────────────────────────────────────────────────────


def test_generation_refuses_without_confirmation(config, store):
    client = StubClient([payload_for("b001", [("quiet spot", "vibe")])])
    with pytest.raises(generate.QuotaNotConfirmedError):
        generate.generate_queries(
            [doc("b001")], config, client=client, store=store, confirm=lambda plan: False
        )
    assert client.calls == [], "no API call may be made before confirmation"


def test_confirmation_states_the_exact_call_count(config, store):
    seen: list[generate.QuotaPlan] = []

    def confirm(plan: generate.QuotaPlan) -> bool:
        seen.append(plan)
        return True

    documents = [doc(f"b{i:03d}") for i in range(20)]
    generate.generate_queries(
        documents,
        config,
        client=StubClient([]),
        store=store,
        confirm=confirm,
    )
    expected_batches = math.ceil(20 / config.synth.batch_size)
    assert seen[0].n_api_calls == expected_batches
    assert str(expected_batches) in seen[0].describe()


def test_fully_cached_run_asks_nothing(config, store):
    """A resumed run that spends no quota must not prompt — otherwise the owner
    learns to click through the prompt that exists to protect them."""
    documents = [doc("b001")]
    generate.generate_queries(
        documents,
        config,
        client=StubClient([payload_for("b001", [("quiet spot", "vibe")])]),
        store=store,
        confirm=lambda plan: True,
    )

    def refuse(plan: generate.QuotaPlan) -> bool:  # pragma: no cover - must not run
        raise AssertionError("confirmation was requested for a fully cached run")

    run = generate.generate_queries(
        documents, config, client=StubClient([]), store=store, confirm=refuse
    )
    assert run.n_api_calls == 0
    assert len(run.queries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Resumability
# ─────────────────────────────────────────────────────────────────────────────


def test_generation_resumes_after_rate_limit(config, store):
    """A 429 partway through keeps every completed batch, and a rerun pays only
    for what is missing."""
    documents = [doc(f"b{i:03d}") for i in range(config.synth.batch_size * 3)]
    payloads = [
        payload_for(f"b{i:03d}", [(f"query for b{i:03d}", "vibe")])
        for i in range(0, len(documents), config.synth.batch_size)
    ]

    first_client = StubClient(payloads, fail_after=2)
    first = generate.generate_queries(
        documents, config, client=first_client, store=store, confirm=lambda p: True
    )
    assert first.stopped_early and "rate limited" in first.stop_reason
    assert first.n_api_calls == 2
    assert len(first.queries) == 2, "work completed before the 429 is kept"
    # Already on disk, not held in memory until the end of the run.
    assert len(store.load_queries()) == 2

    second_client = StubClient(payloads[2:])
    second = generate.generate_queries(
        documents, config, client=second_client, store=store, confirm=lambda p: True
    )
    assert second.n_api_calls == 1, "the two cached batches must not be re-requested"
    assert second.n_batches_cached == 2
    assert len(second.queries) == 3
    assert len(store.load_queries()) == 3


def test_one_failed_batch_does_not_end_the_run(config, store):
    """A malformed reply for one batch must not discard the calls already paid for."""

    class FlakyClient(StubClient):
        def complete_json(self, prompt: str) -> dict[str, Any]:
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise generate.SynthBadResponseError("not JSON")
            return payload_for("b002", [("second batch query", "vibe")])

    documents = [doc("b001"), doc("b002")]
    run = generate.generate_queries(
        documents,
        config.model_copy(update={"synth": config.synth.model_copy(update={"batch_size": 1})}),
        client=FlakyClient([]),
        store=store,
        confirm=lambda p: True,
    )
    assert run.n_batches_failed == 1
    assert not run.stopped_early
    assert len(run.queries) == 1


def test_jsonl_cache_skips_a_truncated_final_line(tmp_path: Path):
    """What a kill -9 mid-write actually leaves behind."""
    path = tmp_path / "c.jsonl"
    cache = generate.JsonlCache(path, "id")
    cache.append([{"id": "a", "v": 1}])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "b", "v":')  # truncated
    assert set(cache.load()) == {"a"}


# ─────────────────────────────────────────────────────────────────────────────
# THE LEAKAGE GUARD
# ─────────────────────────────────────────────────────────────────────────────


GOLD = (
    GoldQuery("g1", "quiet cafe with wifi open late", QueryType.CONSTRAINT),
    GoldQuery("g2", "cheap ramen near downtown", QueryType.SPECIFIC),
)


def test_gold_dedup_removes_exact_duplicate():
    queries = [
        synth_query("b001", "Quiet cafe with wifi open late!"),
        synth_query("b002", "dog park"),
    ]
    kept, report = generate.dedupe_against_gold(queries, GOLD, fake_embed, threshold=0.92)
    assert [q.business_id for q in kept] == ["b002"]
    assert report.n_removed_exact_gold == 1
    assert report.removed[0].kind == "exact_gold"


def test_gold_dedup_removes_planted_near_duplicate():
    """The single most important test in this file.

    A paraphrase shares no normalized form with its gold query, so exact matching
    waves it through. If the embedding half of the guard regresses, the
    cross-encoder gets fine-tuned on a test item and every NDCG in the README
    becomes an overstatement that nothing downstream can reveal.
    """
    planted = synth_query("b001", "cafe with wifi quiet open late")  # same words, reordered
    queries = [planted, synth_query("b002", "indoor rock climbing gym")]
    kept, report = generate.dedupe_against_gold(queries, GOLD, fake_embed, threshold=0.92)

    assert planted.query_id not in {q.query_id for q in kept}
    assert report.n_removed_near_gold == 1
    near = next(r for r in report.removed if r.kind == "near_gold")
    assert near.matched_id == "g1"
    assert near.cosine is not None and near.cosine >= 0.92
    # The count is a required README number, so it has to be in the report text.
    assert "1" in report.summary()


def test_gold_dedup_keeps_a_merely_similar_query():
    """The guard must not be a category filter: other cafés are legitimate
    training data, or hard negatives would have nothing to contrast with."""
    keeper = synth_query("b001", "somewhere calm to work through the afternoon")
    kept, report = generate.dedupe_against_gold([keeper], GOLD, fake_embed, threshold=0.92)
    assert [q.query_id for q in kept] == [keeper.query_id]
    assert report.n_removed_gold == 0


def test_gold_dedup_removes_internal_duplicates():
    """Duplicate training queries reweight the loss, and the same text landing in
    two different splits means the model is tested on its own training item."""
    a = synth_query("b001", "open late with outdoor seating")
    b = synth_query("b002", "with outdoor seating open late")  # same words, reordered
    kept, report = generate.dedupe_against_gold([a, b], GOLD, fake_embed, threshold=0.92)
    assert len(kept) == 1
    assert report.n_removed_internal == 1


def test_gold_dedup_is_order_independent():
    a = synth_query("b001", "open late with outdoor seating")
    b = synth_query("b002", "with outdoor seating open late")
    kept_one, _ = generate.dedupe_against_gold([a, b], GOLD, fake_embed, threshold=0.92)
    kept_two, _ = generate.dedupe_against_gold([b, a], GOLD, fake_embed, threshold=0.92)
    assert [q.query_id for q in kept_one] == [q.query_id for q in kept_two]


def test_gold_dedup_refuses_an_empty_gold_set():
    """ "No gold file, so nothing to remove" must be impossible to report."""
    with pytest.raises(generate.GoldSetMissingError):
        generate.dedupe_against_gold([synth_query("b001", "x")], (), fake_embed)


def test_load_gold_queries_fails_loudly_when_absent(tmp_path: Path):
    with pytest.raises(generate.GoldSetMissingError):
        generate.load_gold_queries(tmp_path / "nope.csv")


def test_load_gold_queries_reads_the_owner_format(tmp_path: Path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "query_id,query,query_type\ng1,quiet cafe,vibe\ng2,open after 10pm,constraint\n",
        encoding="utf-8",
    )
    gold = generate.load_gold_queries(path)
    assert [g.query_id for g in gold] == ["g1", "g2"]
    assert gold[1].query_type is QueryType.CONSTRAINT


def test_load_gold_queries_rejects_an_unknown_query_type(tmp_path: Path):
    path = tmp_path / "gold.csv"
    path.write_text("query_id,query,query_type\ng1,quiet cafe,atmosphere\n", encoding="utf-8")
    with pytest.raises(generate.SynthError):
        generate.load_gold_queries(path)


def test_dedup_rejects_a_misshaped_embedder():
    def bad_embed(texts):
        import numpy as np

        return np.zeros((len(texts) + 1, 4), dtype="float32")

    with pytest.raises(generate.SynthError):
        generate.dedupe_against_gold([synth_query("b001", "x")], GOLD, bad_embed)


# ─────────────────────────────────────────────────────────────────────────────
# Hard negatives
# ─────────────────────────────────────────────────────────────────────────────


def fake_retriever(ranked: list[str]):
    """A retriever returning a fixed ranking with descending scores."""

    def retrieve(query: str, top_k: int):
        return [(bid, 1.0 - i * 0.01) for i, bid in enumerate(ranked[:top_k])]

    return retrieve


def test_mining_excludes_the_positive_and_records_its_rank(config):
    query = synth_query("b002", "late night ramen")
    examples, report = negatives.mine_hard_negatives(
        [query], config, fake_retriever(["b001", "b002", "b003", "b004"])
    )
    ids = [n.business_id for n in examples[0].negatives]
    assert "b002" not in ids, "the source business must never be its own negative"
    assert ids == ["b001", "b003", "b004"]
    assert examples[0].positive_retrieval_rank == 2
    assert report.n_positive_missed == 0


def test_mining_records_a_missed_positive(config):
    """Stage 2 failing to retrieve a query's own source business is a real signal
    about retrieval recall, so it is reported rather than dropped."""
    query = synth_query("b009", "late night ramen")
    examples, report = negatives.mine_hard_negatives(
        [query], config, fake_retriever(["b001", "b002"])
    )
    assert examples[0].positive_retrieval_rank is None
    assert report.n_positive_missed == 1


def test_mining_respects_the_configured_count(config):
    query = synth_query("b000", "coffee")
    pool = [f"b{i:03d}" for i in range(1, 50)]
    examples, report = negatives.mine_hard_negatives([query], config, fake_retriever(pool))
    assert examples[0].n_negatives == config.synth.hard_negatives_per_query == 8
    # The pool is retrieved deeper than the ask, so the judge can remove some.
    assert report.pool_depth > config.synth.hard_negatives_per_query


def test_mining_skips_duplicate_ids(config):
    query = synth_query("b000", "coffee")
    examples, report = negatives.mine_hard_negatives(
        [query], config, fake_retriever(["b001", "b001", "b002"])
    )
    assert [n.business_id for n in examples[0].negatives] == ["b001", "b002"]
    assert report.n_duplicate_business_ids_skipped == 1


def test_negatives_keep_their_retrieval_rank(config):
    """The rank is what makes a negative "hard" and is how the judge chooses which
    ones are worth a grading call."""
    query = synth_query("b000", "coffee")
    examples, _ = negatives.mine_hard_negatives(
        [query], config, fake_retriever(["b001", "b002", "b003"])
    )
    assert [n.retrieval_rank for n in examples[0].negatives] == [1, 2, 3]


def test_random_negatives_are_the_measurable_control(config):
    """Random negatives exist so "hard beats random" is a measurement, not a claim."""
    query = synth_query("b001", "coffee")
    pool = [f"b{i:03d}" for i in range(20)]
    one = negatives.sample_random_negatives([query], pool, config)
    two = negatives.sample_random_negatives([query], list(reversed(pool)), config)
    assert [n.business_id for n in one[0].negatives] == [n.business_id for n in two[0].negatives]
    assert "b001" not in {n.business_id for n in one[0].negatives}
    assert {n.source for n in one[0].negatives} == {"random"}


def test_pairs_for_training_puts_the_positive_first(config):
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    rows = negatives.pairs_for_training(
        examples[0], {"b001": "doc one", "b002": "doc two", "b003": "doc three"}
    )
    assert rows[0] == ("coffee", "doc one", 1)
    assert [label for _, _, label in rows] == [1, 0, 0]


def test_pairs_for_training_skips_an_example_with_no_positive_document(config):
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002"]))
    assert negatives.pairs_for_training(examples[0], {"b002": "doc two"}) == []


def test_examples_round_trip_through_jsonl(config, tmp_path: Path):
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    path = negatives.write_examples_jsonl(examples, tmp_path / "ex.jsonl")
    assert negatives.read_examples_jsonl(path) == examples


# ─────────────────────────────────────────────────────────────────────────────
# The judge: false-negative filtering
# ─────────────────────────────────────────────────────────────────────────────


class StubJudgeClient:
    """Grades pairs from a ``{business_id: grade}`` table."""

    name = "stub"
    model = "stub-model"

    def __init__(self, grades_by_business: dict[str, int]) -> None:
        self.grades = grades_by_business
        self.calls = 0

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.calls += 1
        out = []
        for block in prompt.split("--- PAIR ")[1:]:
            pid = block.split(" ---", 1)[0].strip()
            # The stub reads the document text, which carries the business id.
            business = next((b for b in self.grades if b in block), None)
            out.append({"pair_id": pid, "grade": self.grades.get(business, 0), "reason": "stub"})
        return {"grades": out}


def _example(query_id: str, negative_ids: list[str]) -> negatives.TrainingExample:
    return negatives.TrainingExample(
        query_id=query_id,
        query="quiet cafe with wifi",
        query_type=QueryType.CONSTRAINT,
        positive_business_id="b001",
        negatives=tuple(
            negatives.HardNegative(
                business_id=b, retrieval_rank=i + 1, retrieval_score=1.0 - i / 10
            )
            for i, b in enumerate(negative_ids)
        ),
        positive_retrieval_rank=1,
    )


def test_false_negative_filter_removes_relevant_negatives():
    """Other quiet cafes genuinely exist; training the model to call them
    irrelevant teaches it something false."""
    examples = [_example("q1", ["b002", "b003"])]
    documents = {"b002": "doc for b002", "b003": "doc for b003"}
    stub = StubJudgeClient({"b002": 2, "b003": 0})
    filtered, report = judge.filter_false_negatives(
        examples,
        documents,
        judge.LlmJudge(stub, prompt_version="test"),
        top_n=None,
        assume_yes=True,
    )
    assert [n.business_id for n in filtered[0].negatives] == ["b003"]
    assert report.n_removed == 1
    assert report.n_graded == 2
    assert report.removed[0].business_id == "b002"
    assert "removed 1" in report.summary()


def test_false_negative_filter_keeps_partially_relevant_negatives():
    """Grade 1 is the near-boundary contrast the reranker has to learn; removing it
    would leave a training set closer to random negatives."""
    examples = [_example("q1", ["b002"])]
    stub = StubJudgeClient({"b002": 1})
    filtered, report = judge.filter_false_negatives(
        examples,
        {"b002": "doc for b002"},
        judge.LlmJudge(stub, prompt_version="test"),
        top_n=None,
        assume_yes=True,
    )
    assert filtered[0].n_negatives == 1
    assert report.n_removed == 0


def test_false_negative_filter_only_grades_the_hardest_negatives():
    """The grading budget goes to the top of the retrieval list, where false
    negatives concentrate; the rest are kept and the report says so."""
    examples = [_example("q1", ["b002", "b003", "b004", "b005"])]
    documents = {f"b{i:03d}": f"doc for b{i:03d}" for i in range(2, 6)}
    stub = StubJudgeClient({})
    _, report = judge.filter_false_negatives(
        examples,
        documents,
        judge.LlmJudge(stub, prompt_version="test"),
        top_n=2,
        assume_yes=True,
    )
    assert report.n_graded == 2
    assert report.n_negatives_in == 4


def test_judge_caches_verdicts_across_runs(tmp_path: Path):
    examples = [_example("q1", ["b002"])]
    documents = {"b002": "doc for b002"}
    cache = generate.JsonlCache(tmp_path / "judge.jsonl", "pair_id")
    stub = StubJudgeClient({"b002": 2})

    judge.filter_false_negatives(
        examples,
        documents,
        judge.LlmJudge(stub, cache=cache, prompt_version="test"),
        top_n=None,
        assume_yes=True,
    )
    assert stub.calls == 1

    second = StubJudgeClient({"b002": 2})
    _, report = judge.filter_false_negatives(
        examples,
        documents,
        judge.LlmJudge(second, cache=cache, prompt_version="test"),
        top_n=None,
        assume_yes=True,
    )
    assert second.calls == 0, "a cached verdict must not be re-purchased"
    assert report.n_removed == 1


def test_judge_rejects_a_verdict_for_a_pair_it_did_not_ask_about():
    pair = judge.JudgePair("q1", "quiet cafe", "b002", "doc")
    verdicts, unparsed = judge.parse_judge_response(
        {"grades": [{"pair_id": "p-deadbeef", "grade": 2}]}, [pair]
    )
    assert verdicts == {}
    assert unparsed == 1


def test_judge_rejects_an_out_of_scale_grade():
    pair = judge.JudgePair("q1", "quiet cafe", "b002", "doc")
    verdicts, unparsed = judge.parse_judge_response(
        {"grades": [{"pair_id": pair.pair_id, "grade": 7}]}, [pair]
    )
    assert verdicts == {} and unparsed == 1


def test_judge_grading_requires_confirmation():
    examples = [_example("q1", ["b002"])]
    stub = StubJudgeClient({"b002": 2})
    with pytest.raises(generate.QuotaNotConfirmedError):
        judge.filter_false_negatives(
            examples,
            {"b002": "doc"},
            judge.LlmJudge(stub, prompt_version="test"),
            top_n=None,
            confirm=lambda plan: False,
        )
    assert stub.calls == 0


# ─────────────────────────────────────────────────────────────────────────────
# The judge: label quality, and the read-only rule
# ─────────────────────────────────────────────────────────────────────────────


# One business per labeled pair, so a stub judge keyed by business id can
# reproduce the owner's grades exactly. (In real qrels the same business is
# labeled under several queries with different grades — which is precisely why
# the judge is keyed by (query, business) pair id and not by business.)
QRELS_CSV = """query_id,business_id,grade
g1,b001,2
g1,b002,1
g1,b003,0
g2,b004,2
g2,b005,0
"""


@pytest.fixture
def qrels_file(tmp_path: Path) -> Path:
    path = tmp_path / "qrels.csv"
    path.write_text(QRELS_CSV, encoding="utf-8")
    return path


GOLD_FOR_LABELS = (
    GoldQuery("g1", "quiet cafe with wifi open late", QueryType.CONSTRAINT),
    GoldQuery("g2", "cheap ramen near downtown", QueryType.SPECIFIC),
)
DOCUMENTS = {f"b{i:03d}": f"doc for b{i:03d}" for i in range(1, 6)}


def test_label_quality_reports_kappa_and_a_confusion_matrix(config, qrels_file: Path):
    labels = judge.GoldLabels.load(qrels_file)
    # A judge that agrees exactly: kappa must be 1.0 on a sample using >1 category.
    agreeing = StubJudgeClient({row.business_id: row.grade for row in labels.rows})
    report = judge.validate_label_quality(
        labels,
        GOLD_FOR_LABELS,
        DOCUMENTS,
        judge.LlmJudge(agreeing, prompt_version="test"),
        config,
        assume_yes=True,
    )
    assert report.n_sampled == 5
    assert report.kappa_quadratic.kappa == pytest.approx(1.0)
    assert report.kappa_unweighted.kappa == pytest.approx(1.0)
    # 3x3 whatever grades were used, so matrices are comparable between runs.
    assert len(report.matrix) == 3 and all(len(row) == 3 for row in report.matrix)
    assert report.qrels_sha256 == labels.sha256


def test_label_quality_reports_both_weightings(config, qrels_file: Path):
    """Quadratic kappa forgives 1-vs-2 near misses and is the flattering number;
    reporting only it would overstate agreement."""
    labels = judge.GoldLabels.load(qrels_file)
    # Judge shifts every grade one step down, so quadratic > unweighted.
    shifted = StubJudgeClient({row.business_id: max(0, row.grade - 1) for row in labels.rows})
    report = judge.validate_label_quality(
        labels,
        GOLD_FOR_LABELS,
        DOCUMENTS,
        judge.LlmJudge(shifted, prompt_version="test"),
        config,
        assume_yes=True,
    )
    assert report.kappa_quadratic.weights == "quadratic"
    assert report.kappa_unweighted.weights == "none"
    assert "quadratic" in report.summary()


def test_label_quality_sample_is_deterministic(config, qrels_file: Path):
    labels = judge.GoldLabels.load(qrels_file)
    first, grades_one, _, _ = judge.sample_labeled_pairs(
        labels, GOLD_FOR_LABELS, DOCUMENTS, n=3, seed=config.seed
    )
    second, grades_two, _, _ = judge.sample_labeled_pairs(
        labels, GOLD_FOR_LABELS, DOCUMENTS, n=3, seed=config.seed
    )
    assert [p.pair_id for p in first] == [p.pair_id for p in second]
    assert grades_one == grades_two


def test_label_quality_excludes_ungraded_pairs_rather_than_defaulting_them(
    config, qrels_file: Path
):
    """An API failure must not be recorded as agreement or disagreement."""

    class PartialClient(StubJudgeClient):
        def complete_json(self, prompt: str) -> dict[str, Any]:
            payload = super().complete_json(prompt)
            payload["grades"] = payload["grades"][:1]  # grade only the first pair
            return payload

    labels = judge.GoldLabels.load(qrels_file)
    report = judge.validate_label_quality(
        labels,
        GOLD_FOR_LABELS,
        DOCUMENTS,
        judge.LlmJudge(
            PartialClient({row.business_id: row.grade for row in labels.rows}),
            batch_size=5,
            prompt_version="test",
        ),
        config,
        assume_yes=True,
    )
    assert report.n_sampled == 1
    assert report.n_ungraded == 4
    assert "did not grade" in report.summary()


def test_gold_labels_reject_an_out_of_scale_grade(tmp_path: Path):
    path = tmp_path / "qrels.csv"
    path.write_text("query_id,business_id,grade\ng1,b001,5\n", encoding="utf-8")
    with pytest.raises(generate.SynthError):
        judge.GoldLabels.load(path)


def test_gold_labels_as_qrels_is_not_mutable(qrels_file: Path):
    """Structural read-only check 2: the labels arrive as an immutable value."""
    qrels = judge.GoldLabels.load(qrels_file).as_qrels()
    with pytest.raises(TypeError):
        qrels["g1"]["b001"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        qrels["g3"] = {}  # type: ignore[index]


def test_judge_never_writes_qrels(config, qrels_file: Path):
    """Structural read-only check 3: the file is byte-identical after a full run."""
    before = qrels_file.read_bytes()
    labels = judge.GoldLabels.load(qrels_file)
    stub = StubJudgeClient({row.business_id: row.grade for row in labels.rows})
    judge.validate_label_quality(
        labels,
        GOLD_FOR_LABELS,
        DOCUMENTS,
        judge.LlmJudge(stub, prompt_version="test"),
        config,
        assume_yes=True,
    )
    assert qrels_file.read_bytes() == before


def test_qrels_tripwire_fires_if_the_file_changes(config, qrels_file: Path):
    """If anything does write labels mid-run, the result is an exception rather
    than a quietly invalid kappa."""
    labels = judge.GoldLabels.load(qrels_file)
    qrels_file.write_text(QRELS_CSV + "g2,b005,2\n", encoding="utf-8")
    with pytest.raises(judge.QrelsMutatedError):
        labels.verify_unchanged()


def test_judge_module_contains_no_file_writer():
    """Structural read-only check 1: no writer exists in this module at all.

    Checked on the parsed syntax tree rather than by grepping text, so prose in a
    docstring cannot trip it and a real call cannot hide behind formatting. Every
    write in the package goes through ``generate.JsonlCache``, which cannot be
    pointed at the qrels schema — this test is the mechanical version of a promise
    nobody will remember in six months.
    """
    import ast

    tree = ast.parse(Path(judge.__file__).read_text(encoding="utf-8"))
    forbidden_names = {"open"}
    forbidden_attrs = {
        "write",
        "writelines",
        "writerow",
        "writerows",
        "write_text",
        "write_bytes",
        "to_csv",
        "touch",
        "mkdir",
        "unlink",
    }
    # Note: `replace` is deliberately NOT in this set — `str.replace` is used for
    # prompt rendering, so banning the name would ban legitimate string work and
    # the test would be deleted rather than fixed. `open` is the gate that matters:
    # nothing can be written without it.
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in forbidden_names:
            offenders.append(f"{func.id}() at line {node.lineno}")
        if isinstance(func, ast.Attribute) and func.attr in forbidden_attrs:
            offenders.append(f".{func.attr}() at line {node.lineno}")
    assert not offenders, f"concierge/synth/judge.py must not write anything: {offenders}"


def test_judge_exposes_no_way_to_write_labels():
    """``GoldLabels`` offers reads only — no save, no put, no setter."""
    api = {name for name in dir(judge.GoldLabels) if not name.startswith("_")}
    assert api == {"load", "as_qrels", "verify_unchanged", "path", "sha256", "rows"}


# ─────────────────────────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────────────────────────


def _queries(n: int) -> list[generate.SyntheticQuery]:
    return [synth_query(f"b{i:03d}", f"query number {i}") for i in range(n)]


def test_splits_cover_every_query_exactly_once(config):
    queries = _queries(200)
    manifest = splits.assign_splits_for_queries(queries, config)
    assert manifest.n_queries == 200
    assert sum(manifest.counts.values()) == 200
    assert set(manifest.assignment.values()) <= set(splits.SPLIT_NAMES)


def test_splits_are_disjoint(config):
    manifest = splits.assign_splits_for_queries(_queries(200), config)
    seen: set[str] = set()
    for name in splits.SPLIT_NAMES:
        ids = set(manifest.ids(name))
        assert not (ids & seen), f"{name} overlaps an earlier split"
        seen |= ids
    splits.verify_splits(manifest, cfg=config)


def test_splits_are_out_of_fold_for_the_crossencoder_feature(config):
    """This is what makes LambdaMART's cross-encoder feature honest."""
    manifest = splits.assign_splits_for_queries(_queries(300), config)
    ce = set(manifest.ids(*splits.CROSSENCODER_SPLITS))
    ltr = set(manifest.ids(*splits.LTR_SPLITS))
    assert ce and ltr
    assert not (ce & ltr)
    splits.assert_out_of_fold(manifest)


def test_out_of_fold_check_fails_when_it_should(config):
    """A guard that cannot fail proves nothing, so break it deliberately.

    The realistic way this leak arrives is a stale manifest: the cross-encoder was
    trained from one split and LambdaMART from a regenerated one, so a query ends
    up in both training sets even though neither manifest is self-contradictory.
    """
    manifest = splits.assign_splits_for_queries(_queries(50), config)
    ce_ids = manifest.ids(*splits.CROSSENCODER_SPLITS)
    ltr_ids = manifest.ids(*splits.LTR_SPLITS)

    splits.assert_no_overlap(ce_ids, ltr_ids)  # the honest case passes
    with pytest.raises(splits.SplitLeakageError) as excinfo:
        splits.assert_no_overlap(ce_ids, [*ltr_ids, ce_ids[0]])
    assert ce_ids[0] in str(excinfo.value)


def test_trained_ids_are_checked_against_the_manifest(config):
    """The stronger check: what the trainers actually used, not just the plan."""
    manifest = splits.assign_splits_for_queries(_queries(50), config)
    ce_ids = manifest.ids(*splits.CROSSENCODER_SPLITS)
    ltr_ids = manifest.ids(*splits.LTR_SPLITS)

    splits.assert_models_trained_on_disjoint_queries(ce_ids, ltr_ids, manifest)
    with pytest.raises(splits.SplitLeakageError):
        # A query the manifest assigns to the cross-encoder, used for LambdaMART.
        splits.assert_models_trained_on_disjoint_queries([], [ce_ids[0]], manifest)


def test_verify_splits_rejects_an_unknown_split_name(config):
    manifest = splits.assign_splits_for_queries(_queries(20), config)
    assignment = {**manifest.assignment, "sq-stray": "not_a_split"}
    broken = splits.SplitManifest(
        seed=manifest.seed,
        group_by=manifest.group_by,
        fractions=manifest.fractions,
        assignment=assignment,
        checksum=splits.compute_checksum(
            assignment, manifest.seed, manifest.fractions, manifest.group_by
        ),
        n_groups=manifest.n_groups,
    )
    with pytest.raises(splits.SplitLeakageError):
        splits.verify_splits(broken, cfg=config)


def test_split_assignment_is_deterministic_under_a_seed(config):
    queries = _queries(120)
    one = splits.assign_splits_for_queries(queries, config)
    two = splits.assign_splits_for_queries(list(reversed(queries)), config)
    assert one.assignment == two.assignment
    assert one.checksum == two.checksum


def test_split_assignment_is_stable_as_queries_are_added(config):
    """Hash-based assignment, not a reshuffle: generating more data later must not
    move an existing query from the cross-encoder's split into LambdaMART's."""
    first = splits.assign_splits_for_queries(_queries(50), config)
    second = splits.assign_splits_for_queries(_queries(200), config)
    for query_id, split in first.assignment.items():
        assert second.assignment[query_id] == split


def test_split_changes_with_the_seed(config):
    queries = _queries(100)
    one = splits.assign_splits_for_queries(queries, config)
    other = splits.assign_splits_for_queries(
        queries, config.model_copy(update={"seed": config.seed + 1})
    )
    assert one.assignment != other.assignment


def test_split_fractions_are_approximately_the_configured_ones(config):
    """Hash bucketing gives binomial scatter, not exact slices. The manifest
    reports realised counts for exactly this reason."""
    manifest = splits.assign_splits_for_queries(_queries(2000), config)
    realised = manifest.realised_fractions()
    for name, target in splits.fractions_from_config(config).items():
        assert abs(realised[name] - target) < 0.05, f"{name}: {realised[name]} vs {target}"


def test_grouping_by_business_keeps_a_businesss_queries_together(config):
    queries = [
        synth_query("b001", "query one"),
        synth_query("b001", "query two"),
        synth_query("b002", "query three"),
    ]
    manifest = splits.assign_splits_for_queries(queries, config, group_by="business")
    assert manifest.n_groups == 2
    assigned = {q.query_id: manifest.assignment[q.query_id] for q in queries}
    assert assigned[queries[0].query_id] == assigned[queries[1].query_id]


def test_manifest_round_trips_through_disk(config, tmp_path: Path):
    manifest = splits.assign_splits_for_queries(_queries(30), config)
    path = splits.save_manifest(manifest, tmp_path / "splits.json")
    loaded = splits.load_manifest(path)
    assert loaded.assignment == manifest.assignment
    assert loaded.checksum == manifest.checksum
    assert loaded.seed == config.seed


def test_manifest_checksum_catches_a_hand_edit(config, tmp_path: Path):
    manifest = splits.assign_splits_for_queries(_queries(30), config)
    path = splits.save_manifest(manifest, tmp_path / "splits.json")
    data = json.loads(path.read_text())
    victim = sorted(data["assignment"])[0]
    data["assignment"][victim] = (
        "ltr_train" if data["assignment"][victim] != "ltr_train" else "ltr_val"
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(splits.SplitManifestCorruptError):
        splits.load_manifest(path)


def test_verify_rejects_a_manifest_from_another_seed(config, tmp_path: Path):
    """A config change must not be answerable with a stale split."""
    other = config.model_copy(update={"seed": config.seed + 7})
    manifest = splits.assign_splits_for_queries(_queries(30), other)
    with pytest.raises(splits.SplitLeakageError):
        splits.verify_splits(manifest, cfg=config)


def test_require_disjoint_splits_is_what_the_trainers_call(config, tmp_path: Path):
    manifest, path = splits.build_and_save(_queries(40), config, root=tmp_path)
    verified = splits.require_disjoint_splits(config, root=tmp_path)
    assert verified.checksum == manifest.checksum
    assert path.exists()


def test_require_disjoint_splits_fails_with_no_manifest(config, tmp_path: Path):
    with pytest.raises(splits.SplitManifestCorruptError):
        splits.require_disjoint_splits(config, root=tmp_path)


def test_split_of_refuses_an_unassigned_query(config):
    manifest = splits.assign_splits_for_queries(_queries(10), config)
    with pytest.raises(splits.SplitLeakageError):
        splits.split_of(manifest, "sq-doesnotexist")


def test_partition_groups_training_examples_by_split(config):
    queries = _queries(40)
    manifest = splits.assign_splits_for_queries(queries, config)
    examples = [_example(q.query_id, ["b999"]) for q in queries]
    grouped = splits.partition(manifest, examples)
    assert sum(len(v) for v in grouped.values()) == len(examples)
    for name, items in grouped.items():
        for item in items:
            assert manifest.assignment[item.query_id] == name


def test_assign_splits_rejects_a_query_with_two_group_keys(config):
    with pytest.raises(splits.SplitError):
        splits.assign_splits([("q1", "b001"), ("q1", "b002")], config)


# ─────────────────────────────────────────────────────────────────────────────
# Live-API tests (skipped without a key; never run in CI)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.needs_api
@requires_api_key
def test_live_generation_returns_usable_queries(config):
    """One real batch, to prove the prompt and the schema work against the API.

    Marked needs_api so the default suite never spends quota. Asserts shape only:
    asserting anything about the *content* of a model's reply would be a flaky
    test and would tell us nothing we could report.
    """
    client = generate.GeminiJsonClient.from_config(
        config, response_schema=generate.generation_response_schema()
    )
    payload = client.complete_json(
        generate.render_generation_prompt(
            [doc("b001", "Lantern Coffee. Coffee & Tea, Cafes. free wifi, quiet, open to 11pm.")],
            2,
        )
    )
    parsed = generate.parse_generation_response(payload, [doc("b001")])
    assert parsed.queries, "the live model returned no usable queries"
