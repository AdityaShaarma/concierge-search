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

from concierge.rerank import train as rerank_train
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


def recording_retriever(ranked: list[str]) -> tuple[negatives.RetrieveFn, list[str]]:
    """``fake_retriever``, plus the list of query strings it was actually handed.

    The query text is the whole subject of the two tests below, and it is
    invisible in the mined output (the record keeps the raw query by design), so
    the only place to observe it is at the retriever boundary.
    """
    seen: list[str] = []

    def retrieve(query: str, top_k: int):
        seen.append(query)
        return [(bid, 1.0 - i * 0.01) for i, bid in enumerate(ranked[:top_k])]

    return retrieve, seen


#: A query the rule parser really does strip, so the assertions below can fail.
#: "quiet" (NoiseLevel), "with wifi" (WiFi) and "open past 10pm" (hours) all
#: become hard filters and leave the text. If this string were "coffee" the
#: stripped and raw forms would be identical and every test using it would pass
#: just as happily against the bug it exists to catch.
CONSTRAINED_QUERY = "quiet cafe with wifi open past 10pm"


def test_mining_retrieves_on_the_text_the_served_system_retrieves_on(config):
    """The negatives must be drawn from the candidate pool serving produces.

    Serving's Stage 2 retrieves on ``query_text_for(parsed)`` — the request with
    the hard constraints stripped out, because the filters enforce those
    arithmetically. Mining on the raw synthetic query instead draws negatives
    from a pool the served system never produces: the extra constraint words are
    a free relevance signal, so the top of that list is cleaner than the
    shortlist the reranker is really handed and the mined negatives come out
    systematically easier than the real ones. The fine-tune then optimises an
    input distribution the eval never measures, and the measured
    fine-tuned-vs-pretrained gain does not transfer.

    Asserted against ``rerank_train.training_query_text``, which is the single
    definition of that string: ``tests/test_rerank.py`` pins it to what the
    serving reranker scores, so agreeing with it here chains mining all the way
    to the served text rather than re-deriving it and hoping.
    """
    query = synth_query("b001", CONSTRAINED_QUERY)
    retrieve, seen = recording_retriever(["b002", "b003"])
    negatives.mine_hard_negatives([query], config, retrieve)

    stripped = rerank_train.training_query_text(CONSTRAINED_QUERY)
    assert stripped != CONSTRAINED_QUERY, (
        "CONSTRAINED_QUERY must be one the parser actually strips, or this test "
        "cannot distinguish the fix from the bug"
    )
    assert seen == [stripped]


def test_mined_examples_keep_the_raw_query(config):
    """Retrieval uses the stripped text; the *record* keeps the raw query.

    Deliberate, and the reason the test above has to watch the retriever. Every
    downstream consumer re-derives what it needs: the reranker trainer derives
    the scored text from this field, and the LLM judge wants the full request,
    constraints included, when it decides whether a negative is secretly
    relevant. Storing the stripped text would double-apply the derivation for
    the first consumer and hide the constraints from the second.
    """
    query = synth_query("b001", CONSTRAINED_QUERY)
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    assert examples[0].query == CONSTRAINED_QUERY


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
    """Row order and labels only. "coffee" has nothing for the parser to strip, so
    the query side is unchanged here on purpose — it is pinned separately, against
    a query that *is* stripped, in
    ``test_pairs_for_training_agrees_with_the_reranker_trainers_pair_builder``."""
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    rows = negatives.pairs_for_training(
        examples[0], {"b001": "doc one", "b002": "doc two", "b003": "doc three"}
    )
    assert rows[0] == ("coffee", "doc one", 1)
    assert [label for _, _, label in rows] == [1, 0, 0]


def test_pairs_for_training_skips_an_example_with_no_positive_document(config):
    """A per-example skip, not the batch-level error ``build_pairs`` raises.

    ``build_pairs`` treats a whole training file with nothing trainable in it as
    a configuration error, which is right for a file and wrong for one example —
    so ``pairs_for_training`` keeps its own ``[]`` contract for this one case.
    """
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002"]))
    assert negatives.pairs_for_training(examples[0], {"b002": "doc two"}) == []


def test_pairs_for_training_agrees_with_the_reranker_trainers_pair_builder(config, tmp_path: Path):
    """One pair builder, proved rather than intended.

    ``pairs_for_training`` was a second implementation that emitted the raw
    synthetic query long after ``rerank.train.build_pairs`` had switched to the
    constraint-stripped text serving scores. It was dead outside this file, which
    is exactly why nothing noticed. It now delegates; this test is what keeps the
    delegation honest, because a reimplementation would pass every other test in
    this section.

    The comparison goes through the real path on purpose — mined examples written
    to JSONL, read back by the *trainer's* reader (``parse_example``), flattened
    by the trainer's builder — so it also pins the on-disk contract between the
    two modules, which is where the earlier silent-empty-negatives bug lived.
    """
    query = synth_query("b001", CONSTRAINED_QUERY)
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    documents = {"b001": "doc one", "b002": "doc two", "b003": "doc three"}

    rows = negatives.pairs_for_training(examples[0], documents)

    path = negatives.write_examples_jsonl(examples, tmp_path / "ex.jsonl")
    on_disk = [
        rerank_train.parse_example(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs, _ = rerank_train.build_pairs(on_disk, documents)

    assert rows == [(p.query, p.document, int(p.label)) for p in pairs]

    # And the shared query side is the stripped text, not the raw query. Without
    # this the test above would also pass on two identical *wrong* builders.
    stripped = rerank_train.training_query_text(CONSTRAINED_QUERY)
    assert stripped != CONSTRAINED_QUERY
    assert {q for q, _, _ in rows} == {stripped}


def test_examples_round_trip_through_jsonl(config, tmp_path: Path):
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    path = negatives.write_examples_jsonl(examples, tmp_path / "ex.jsonl")
    assert negatives.read_examples_jsonl(path) == examples


def test_a_judged_verdict_survives_the_round_trip(config, tmp_path: Path):
    """The verdict is label provenance, so it has to be in the file.

    Stage 4 reads the filtered file and reports how many of its label-0 rows rest
    on an LLM verdict; a field that lived only in memory would make that count
    zero on every real run.
    """
    query = synth_query("b001", "coffee")
    examples, _ = negatives.mine_hard_negatives([query], config, fake_retriever(["b002", "b003"]))
    judged = [examples[0].with_verdicts({"b002": 1})]

    path = negatives.write_examples_jsonl(judged, tmp_path / "ex.jsonl")
    back = negatives.read_examples_jsonl(path)

    assert back == judged
    by_id = {n.business_id: n for n in back[0].negatives}
    assert by_id["b002"].verdict == 1 and by_id["b002"].judged
    assert by_id["b003"].verdict is None and not by_id["b003"].judged


def test_a_negative_from_a_file_without_the_verdict_field_reads_as_unjudged(tmp_path: Path):
    """An ``examples_filtered.jsonl`` written before the field must still load.

    And it must not pass its negatives off as graded: such a file records no
    grading provenance at all, so "never judged" is the only honest reading. The
    row below is the exact shape the old writer produced.
    """
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "quiet cafe with wifi",
                "query_type": QueryType.CONSTRAINT.value,
                "positive_business_id": "b001",
                "negatives": [
                    {
                        "business_id": "b002",
                        "retrieval_rank": 1,
                        "retrieval_score": 0.9,
                        "source": "hybrid",
                    }
                ],
                "positive_retrieval_rank": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [example] = negatives.read_examples_jsonl(path)

    assert example.negatives[0].verdict is None
    assert not example.negatives[0].judged


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


def test_a_surviving_negative_records_the_grade_it_survived_on():
    """A kept negative's label has to be traceable to an actual grading call.

    The judge grades only the hardest few negatives of each query, so the
    filtered file is a mixture of "graded and not called relevant" and "never
    looked at" — and both become label-0 rows in Stage 4's training file
    (DECISIONS.md D227). Recording the verdict is what lets a reader tell them
    apart afterwards, instead of trusting that the judge's ``top_n`` was deep
    enough (D228).
    """
    examples = [_example("q1", ["b002", "b003", "b004"])]
    documents = {f"b{i:03d}": f"doc for b{i:03d}" for i in (2, 3, 4)}
    # b002 is relevant and goes; b003 is partially relevant and stays, graded;
    # b004 is past `top_n` and stays ungraded.
    stub = StubJudgeClient({"b002": 2, "b003": 1})
    filtered, report = judge.filter_false_negatives(
        examples,
        documents,
        judge.LlmJudge(stub, prompt_version="test"),
        top_n=2,
        assume_yes=True,
    )

    kept = {n.business_id: n for n in filtered[0].negatives}
    assert set(kept) == {"b003", "b004"}
    assert kept["b003"].judged and kept["b003"].verdict == 1
    assert not kept["b004"].judged, "an ungraded negative must not claim a verdict"
    assert kept["b004"].verdict is None
    # The grade counts are unchanged: this records evidence, it does not regrade.
    assert report.n_graded == 2
    assert report.n_removed == 1


def test_an_ungraded_example_passes_through_the_filter_untouched():
    """No verdicts, no copies: the filter must not rewrite what it did not grade."""
    examples = [_example("q1", ["b002"])]
    filtered, report = judge.filter_false_negatives(
        examples,
        {"b002": "doc for b002"},
        # No document is withheld; `top_n=0` is what "nothing was graded" looks
        # like, and the example must come back as the same object.
        judge.LlmJudge(StubJudgeClient({}), prompt_version="test"),
        top_n=0,
        assume_yes=True,
    )

    assert filtered[0] is examples[0]
    assert report.n_graded == 0
    assert all(n.verdict is None for n in filtered[0].negatives)


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


def test_gold_labels_read_the_header_the_labeling_cli_actually_writes(tmp_path: Path):
    """Cross-module seam test: write with the real writer, read with the judge.

    Every other test here hand-writes a ``grade`` header, so all of them passed
    while the judge could not read the only qrels file that will ever exist —
    `eval.pooling.QrelsWriter` heads that column "relevance". The label-quality
    kappa is a headline deliverable, and it failed on the owner's own labels.
    """
    from eval.pooling import QRELS_COLUMNS, QrelsWriter

    path = tmp_path / "qrels.csv"
    with QrelsWriter(path, labeler="owner") as writer:
        writer.append("g1", "b001", 2)
        writer.append("g1", "b002", 0)

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(QRELS_COLUMNS), "writer header changed; this test pins the seam"

    labels = judge.GoldLabels.load(path)
    assert {(r.query_id, r.business_id, r.grade) for r in labels.rows} == {
        ("g1", "b001", 2),
        ("g1", "b002", 0),
    }


def test_gold_labels_name_a_missing_grade_column(tmp_path: Path):
    """A file with no grade-ish column at all must say so, rather than blaming
    row 2 for an empty value."""
    path = tmp_path / "qrels.csv"
    path.write_text("query_id,business_id,notes\ng1,b001,looks fine\n", encoding="utf-8")
    with pytest.raises(generate.SynthError, match="no grade column"):
        judge.GoldLabels.load(path)


def test_every_qrels_reader_agrees_on_the_grade_column() -> None:
    """One list, not three. The judge and the eval harness must resolve the
    grade column identically, and the writer's column must be in the list."""
    from eval.metrics import QRELS_GRADE_COLUMNS
    from eval.pooling import QRELS_COLUMNS
    from eval.run_eval import QRELS_GRADE_ALIASES

    assert QRELS_GRADE_ALIASES is QRELS_GRADE_COLUMNS
    written = set(QRELS_COLUMNS) & set(QRELS_GRADE_COLUMNS)
    assert len(written) == 1, f"writer's grade column is ambiguous or absent: {written}"


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
# Near-duplicate clustering: what may not straddle the split boundary
# ─────────────────────────────────────────────────────────────────────────────
#
# A query id is sha256(business_id, normalized text) and a split is a hash of
# that id, so two *paraphrases* of one query get unrelated ids and can be
# assigned to different splits: the cross-encoder fine-tuned on one, LambdaMART
# trained on the other, with every id-level out-of-fold check still passing
# because at the level of ids nothing is wrong.
#
# Two mechanisms close that, and the tests below are split along them because an
# interviewer will ask which one is doing the work:
#
#   1. the gold guard's third pass (``dedupe_within``) drops any query whose
#      cosine against a kept query is at or above the configured threshold, so an
#      above-threshold paraphrase never reaches the split stage at all — only one
#      of the pair survives, and the dedup report names it;
#   2. cluster grouping keeps whatever *does* reach the split stage together:
#      queries sharing a source business (which pass 3 does not touch — they are
#      different requests, not duplicates) and anything linked transitively.
#
# So pass 3 is the first line and the clustering is the second, and the invariant
# the pipeline actually guarantees is the one the end-to-end test asserts: no
# paraphrase of a training query lands on the other side of the boundary.

#: Plant one: one request, the same words in a different order, written from two
#: different businesses — what the generator produces when two similar places are
#: described in one batch. Nothing here is arbitrary: the two texts and the two
#: business ids were *searched for* (by enumerating word orders and bucketing the
#: resulting ids) so that id-level splitting puts them on opposite sides of the
#: cross-encoder/LambdaMART boundary. If the seed or the configured split
#: fractions ever change, ``test_paraphrase_pair_straddles_the_split_boundary_by_id``
#: is what tells you this pair no longer demonstrates the leak and a new one has
#: to be found.
PARAPHRASE_A = ("b001", "quiet cafe with free wifi open late")
PARAPHRASE_B = ("b002", "quiet cafe with free wifi late open")

#: Plant two: two queries written from ONE business document, which is what every
#: business in the run produces. They share no words, so neither exact matching
#: nor the embedding passes link them — the gold guard keeps both, and only the
#: clustering's same-business merge can hold them together. Searched for the same
#: way: by id these two also straddle the boundary.
SAME_BUSINESS_ID = "b003"
SAME_BUSINESS_QUERIES = ("birria tacos", "late night counter service")


def paraphrase_pair() -> tuple[generate.SyntheticQuery, generate.SyntheticQuery]:
    return synth_query(*PARAPHRASE_A), synth_query(*PARAPHRASE_B)


def same_business_pair() -> tuple[generate.SyntheticQuery, generate.SyntheticQuery]:
    first, second = SAME_BUSINESS_QUERIES
    return synth_query(SAME_BUSINESS_ID, first), synth_query(SAME_BUSINESS_ID, second)


def test_paraphrases_land_in_one_cluster(config):
    """Reordered words give a different id *and* a different normalized form, so
    only the embedding pass can link them — which is why the guard cannot be
    exact matching alone."""
    a, b = paraphrase_pair()
    assert a.query_id != b.query_id
    assert generate.normalize_query_text(a.query) != generate.normalize_query_text(b.query)
    report = generate.cluster_near_duplicates_from_config(
        [a, b], config, embed=fake_embed, merge_same_business=False
    )
    assert report.cluster_of[a.query_id] == report.cluster_of[b.query_id]
    assert report.n_clusters == 1
    assert report.n_multi_query_clusters == 1
    assert report.largest_cluster == 2
    assert report.threshold == config.synth.gold_dedup_cosine_threshold


def test_unrelated_queries_stay_in_their_own_clusters(config):
    """Over-merging is the safe direction but not a free one: it shrinks the usable
    splits, so the clustering has to leave unrelated queries alone."""
    a = synth_query("b001", "quiet cafe with free wifi open late")
    b = synth_query("b002", "birria tacos")
    report = generate.cluster_near_duplicates_from_config(
        [a, b], config, embed=fake_embed, merge_same_business=False
    )
    assert report.cluster_of[a.query_id] != report.cluster_of[b.query_id]
    assert report.n_clusters == 2
    assert report.n_multi_query_clusters == 0


def test_clustering_merges_one_businesss_queries_by_default(config):
    """Queries written from one business document carry the document-level leak as
    well as the paraphrase risk, so they move together unless asked otherwise."""
    a = synth_query("b001", "birria tacos")
    b = synth_query("b001", "late night counter service")
    merged = generate.cluster_near_duplicates_from_config([a, b], config, embed=fake_embed)
    assert merged.cluster_of[a.query_id] == merged.cluster_of[b.query_id]
    assert merged.merged_same_business is True
    apart = generate.cluster_near_duplicates_from_config(
        [a, b], config, embed=fake_embed, merge_same_business=False
    )
    assert apart.cluster_of[a.query_id] != apart.cluster_of[b.query_id]


def test_cluster_keys_do_not_depend_on_input_order(config):
    """The key is the cluster's lowest member id, so two runs over the same queries
    in different orders produce the same grouping — and therefore the same split."""
    a, b = paraphrase_pair()
    one = generate.cluster_near_duplicates_from_config([a, b], config, embed=fake_embed)
    two = generate.cluster_near_duplicates_from_config([b, a], config, embed=fake_embed)
    assert dict(one.cluster_of) == dict(two.cluster_of)


def test_clustering_is_transitive(config):
    """Union-find, so A~B and B~C puts all three together even when A and C fall
    below the threshold on their own.

    The threshold is passed directly here rather than taken from the config: with
    the hashed stand-in embedder, cosine between two equal-length word sets is the
    share of words they share, so a chain of overlaps is only constructible at a
    threshold the real pipeline would never use. The property under test is the
    transitive closure, not the value.
    """
    a = synth_query("b001", "alpha bravo charlie delta")
    b = synth_query("b002", "charlie delta echo foxtrot")
    c = synth_query("b003", "echo foxtrot golf hotel")
    report = generate.cluster_near_duplicates(
        [a, b, c], fake_embed, threshold=0.5, merge_same_business=False
    )
    assert report.n_clusters == 1, "A~B and B~C must put A and C in one cluster"
    assert report.largest_cluster == 3


def test_clustering_rejects_an_impossible_threshold():
    with pytest.raises(generate.SynthConfigError):
        generate.cluster_near_duplicates([synth_query("b001", "coffee")], fake_embed, threshold=0.0)


def test_paraphrase_pair_straddles_the_split_boundary_by_id(config):
    """The leak itself, demonstrated: split by query id and this paraphrase pair
    lands on opposite sides of the cross-encoder/LambdaMART boundary, while every
    out-of-fold assertion the trainers make still passes."""
    a, b = paraphrase_pair()
    manifest = splits.assign_splits_for_queries([a, b], config, group_by="query")
    assert manifest.assignment[a.query_id] in splits.CROSSENCODER_SPLITS
    assert manifest.assignment[b.query_id] in splits.LTR_SPLITS
    splits.assert_out_of_fold(manifest)  # passes, and that is exactly the problem


def test_cluster_grouping_keeps_the_paraphrase_pair_on_one_side(config):
    """The fix: group by near-duplicate cluster and the pair cannot straddle."""
    a, b = paraphrase_pair()
    report = generate.cluster_near_duplicates_from_config([a, b], config, embed=fake_embed)
    manifest = splits.assign_splits_for_queries(
        [a, b], config, group_by="cluster", cluster_of=report.cluster_of
    )
    assert manifest.assignment[a.query_id] == manifest.assignment[b.query_id]
    assert manifest.n_groups == 1
    assert manifest.group_by == "cluster"


def test_same_business_queries_straddle_the_split_boundary_by_id(config):
    """The second leak, demonstrated: two queries from one business document are
    not duplicates of each other, so no dedup pass removes either — and split by
    id they land on opposite sides of the boundary, which means the cross-encoder
    is fine-tuned on that business's document and then scores it for LambdaMART."""
    a, b = same_business_pair()
    manifest = splits.assign_splits_for_queries([a, b], config, group_by="query")
    assert manifest.assignment[a.query_id] in splits.CROSSENCODER_SPLITS
    assert manifest.assignment[b.query_id] in splits.LTR_SPLITS


def test_same_business_queries_share_a_split_under_cluster_grouping(config):
    a, b = same_business_pair()
    report = generate.cluster_near_duplicates_from_config([a, b], config, embed=fake_embed)
    manifest = splits.assign_splits_for_queries(
        [a, b], config, group_by="cluster", cluster_of=report.cluster_of
    )
    assert manifest.assignment[a.query_id] == manifest.assignment[b.query_id]
    assert manifest.n_groups == 1


def test_cluster_grouping_requires_a_cluster_map(config):
    with pytest.raises(splits.SplitError, match="cluster_of"):
        splits.assign_splits_for_queries(list(paraphrase_pair()), config, group_by="cluster")


def test_cluster_grouping_refuses_a_query_missing_from_the_map(config):
    """A query that quietly became its own group is precisely the paraphrase that
    then straddles the boundary, so a gap in the map is an error."""
    a, b = paraphrase_pair()
    with pytest.raises(splits.SplitError, match="cluster map"):
        splits.assign_splits_for_queries(
            [a, b], config, group_by="cluster", cluster_of={a.query_id: a.query_id}
        )


# ─────────────────────────────────────────────────────────────────────────────
# The four CLI entry points — `make synth`
# ─────────────────────────────────────────────────────────────────────────────
#
# Every guard above is worth exactly what the runnable pipeline exercises. These
# tests drive the four ``main`` functions the Makefile calls, end to end, with a
# stub model and the hashed embedder, and assert the three properties that make
# the pipeline trustworthy rather than merely present:
#
#   1. the gold leakage guard is on the training path (a stage reads only the
#      deduplicated query file, and the removal counts land in an artifact);
#   2. each stage refuses to run on a missing or hand-skipped predecessor;
#   3. a planted paraphrase pair ends up in one split, through the real chain.
#
# The quota rule (CLAUDE.md §10) is covered by the confirmation tests above; here
# the stages are run with --assume-yes, which is what states the call count
# without a terminal.


def cli_config(tmp_path: Path, **synth_overrides: Any) -> tuple[Path, Any]:
    """Write a config for the CLI tests: ``configs/base.yaml`` with its gold-label
    paths redirected into ``tmp_path`` and its batch sizes shrunk.

    Dumped from the real ``Config`` object rather than written as a YAML literal,
    so every value the test does not deliberately override — the cosine threshold
    and the split fractions above all — is the one the pipeline actually runs
    with, and cannot drift away from it.
    """
    import yaml

    from concierge.config import load_config

    cfg = load_config("configs/base.yaml")
    data = cfg.model_dump(mode="json")
    data["synth"].update(synth_overrides)
    data["evaluation"]["gold_queries"] = str(tmp_path / "gold_queries.csv")
    data["evaluation"]["qrels"] = str(tmp_path / "qrels.csv")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path, load_config(path)


def write_documents_file(directory: Path, business_ids: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "documents.jsonl"
    path.write_text(
        "".join(
            json.dumps({"business_id": b, "text": f"doc for {b}"}) + "\n" for b in business_ids
        ),
        encoding="utf-8",
    )
    return path


class PromptAwareStubClient:
    """A generation client that answers for whichever businesses the prompt names.

    ``StubClient`` replays payloads by call index, which couples a test to the
    order ``sample_documents`` happens to produce. The CLI tests care about what
    comes out of the pipeline, not about sampling order, so this stub reads the
    business ids out of the rendered prompt instead.
    """

    name = "stub"
    model = "stub-model"

    def __init__(self, queries_by_business: dict[str, list[tuple[str, str]]]) -> None:
        self.queries_by_business = queries_by_business
        self.calls: list[str] = []

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.calls.append(prompt)
        return {
            "businesses": [
                {
                    "business_id": business_id,
                    "queries": [{"query": q, "type": t} for q, t in rows],
                }
                for business_id, rows in self.queries_by_business.items()
                if business_id in prompt
            ]
        }


def write_gold_file(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "gold_queries.csv"
    path.write_text(
        "query_id,query,query_type\n"
        + "".join(f"{qid},{text},{qtype}\n" for qid, text, qtype in rows),
        encoding="utf-8",
    )
    return path


def test_generate_cli_runs_the_leakage_guard_and_reports_what_it_removed(tmp_path: Path):
    """``make synth-generate`` must write the guard's output, not the raw set.

    Two gold duplicates are planted: one that differs from the gold query only in
    case and punctuation (caught by exact matching) and one with an extra word
    (caught only by the embedding pass). Both have to be gone from the file the
    next stage reads, and the counts have to land in the report artifact — the
    removal count is a README figure, so it cannot be a log line.
    """
    gold_text = "cozy spot to read on a rainy afternoon"
    write_gold_file(tmp_path, [("g1", gold_text, "vibe")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=2, businesses_sampled=2, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001", "b002"])

    exact_dup = "Cozy Spot To Read On A Rainy Afternoon!"
    near_dup = f"{gold_text} downtown"
    keeper_a, keeper_b = PARAPHRASE_A[1], "birria tacos"
    client = PromptAwareStubClient(
        {
            "b001": [(exact_dup, "vibe"), (keeper_a, "constraint")],
            "b002": [(near_dup, "vibe"), (keeper_b, "specific")],
        }
    )

    code = generate.main(
        [
            "--config",
            str(cfg_path),
            "--documents",
            str(documents),
            "--assume-yes",
            "--artifacts-root",
            str(tmp_path),
        ],
        client=client,
        embed=fake_embed,
    )
    assert code == 0

    out_dir = generate.synth_dir(cfg, tmp_path)
    generated = {
        q.query for q in generate.GenerationStore.from_config(cfg, tmp_path).load_queries()
    }
    kept = {
        q.query for q in generate.read_queries_jsonl(out_dir / generate.QUERIES_DEDUPED_FILENAME)
    }
    assert {exact_dup, near_dup} <= generated, "the stub's queries should all be generated"
    assert kept == {keeper_a, keeper_b}, "both planted gold duplicates must be gone"

    report = json.loads((out_dir / generate.DEDUP_REPORT_FILENAME).read_text())
    assert report["n_removed_exact_gold"] == 1
    assert report["n_removed_near_gold"] == 1
    assert report["n_kept"] == len(kept)
    assert report["cosine_threshold"] == cfg.synth.gold_dedup_cosine_threshold
    assert {row["kind"] for row in report["removed"]} == {"exact_gold", "near_gold"}
    assert (out_dir / generate.QUERIES_DEDUPED_CSV_FILENAME).exists()


def test_generate_cli_with_skip_dedup_leaves_the_next_stage_nothing_to_read(tmp_path: Path):
    """--skip-dedup is a generation-only mode, not a way around the guard: it
    writes no training input, and the next stage says which target to rerun."""
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=1, businesses_sampled=1, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001"])
    client = StubClient([payload_for("b001", [("birria tacos", "specific")])])

    code = generate.main(
        [
            "--config",
            str(cfg_path),
            "--documents",
            str(documents),
            "--assume-yes",
            "--skip-dedup",
            "--artifacts-root",
            str(tmp_path),
        ],
        client=client,
        embed=fake_embed,
    )
    assert code == 0

    out_dir = generate.synth_dir(cfg, tmp_path)
    assert (out_dir / generate.QUERIES_FILENAME).exists()
    assert not (out_dir / generate.QUERIES_DEDUPED_FILENAME).exists()
    with pytest.raises(generate.SynthArtifactMissingError, match="synth-generate"):
        negatives.main(
            ["--config", str(cfg_path), "--artifacts-root", str(tmp_path)],
            retrieve=fake_retriever(["b002"]),
        )


def test_negatives_cli_mines_only_the_queries_the_guard_kept(tmp_path: Path):
    """The un-deduplicated store sits right beside the deduplicated file; mining
    must read the second one, which is what keeps the guard unavoidable."""
    cfg_path, cfg = cli_config(tmp_path, hard_negatives_per_query=2)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    kept = synth_query("b001", "quiet cafe with free wifi open late")
    removed = synth_query("b002", "cozy spot to read on a rainy afternoon")
    generate.write_queries_jsonl([kept], out_dir / generate.QUERIES_DEDUPED_FILENAME)
    generate.JsonlCache(out_dir / generate.QUERIES_FILENAME, "query_id").append(
        [kept.to_json(), removed.to_json()]
    )

    code = negatives.main(
        ["--config", str(cfg_path), "--artifacts-root", str(tmp_path)],
        retrieve=fake_retriever(["b003", "b004", "b005"]),
    )
    assert code == 0

    mined = negatives.read_examples_jsonl(out_dir / negatives.EXAMPLES_MINED_FILENAME)
    assert [e.query_id for e in mined] == [kept.query_id]
    assert [n.business_id for n in mined[0].negatives] == ["b003", "b004"]
    report = json.loads((out_dir / negatives.MINING_REPORT_FILENAME).read_text())
    assert report["n_queries"] == 1
    assert report["n_negatives_total"] == 2


def test_negatives_cli_resumes_without_re_retrieving(tmp_path: Path):
    """Mining spends no quota but it does run two index searches per query, so a
    second run must reuse what is on disk — and the report must still describe the
    whole file rather than only the queries this run mined."""
    cfg_path, cfg = cli_config(tmp_path, hard_negatives_per_query=1)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    first = synth_query("b001", "quiet cafe with free wifi open late")
    second = synth_query("b002", "birria tacos")
    generate.write_queries_jsonl([first], out_dir / generate.QUERIES_DEDUPED_FILENAME)
    argv = ["--config", str(cfg_path), "--artifacts-root", str(tmp_path)]
    assert negatives.main(argv, retrieve=fake_retriever(["b003", "b004"])) == 0

    # Both queries now, with a retriever that fails if it is asked about the first.
    generate.write_queries_jsonl([first, second], out_dir / generate.QUERIES_DEDUPED_FILENAME)

    def retrieve_only_the_new_one(query: str, top_k: int):
        assert query == second.query, f"re-retrieved a query already mined: {query!r}"
        return fake_retriever(["b003", "b004"])(query, top_k)

    assert negatives.main(argv, retrieve=retrieve_only_the_new_one) == 0
    mined = negatives.read_examples_jsonl(out_dir / negatives.EXAMPLES_MINED_FILENAME)
    assert {e.query_id for e in mined} == {first.query_id, second.query_id}
    report = json.loads((out_dir / negatives.MINING_REPORT_FILENAME).read_text())
    assert report["n_queries"] == len(mined), "the report must describe the whole file"
    assert report["n_negatives_total"] == sum(e.n_negatives for e in mined)


def test_judge_cli_writes_the_filtered_file_the_split_stage_reads(tmp_path: Path):
    cfg_path, cfg = cli_config(tmp_path)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    negatives.write_examples_jsonl(
        [_example("q1", ["b002", "b003"])], out_dir / negatives.EXAMPLES_MINED_FILENAME
    )
    documents = write_documents_file(tmp_path, ["b001", "b002", "b003"])
    stub = StubJudgeClient({"b002": 2})

    code = judge.main(
        [
            "--config",
            str(cfg_path),
            "--documents",
            str(documents),
            "--assume-yes",
            "--artifacts-root",
            str(tmp_path),
        ],
        client=stub,
    )
    assert code == 0

    filtered = negatives.read_examples_jsonl(out_dir / judge.EXAMPLES_FILTERED_FILENAME)
    assert [n.business_id for n in filtered[0].negatives] == ["b003"]
    report = json.loads((out_dir / judge.FALSE_NEGATIVE_REPORT_FILENAME).read_text())
    assert report["n_removed"] == 1
    assert report["n_graded"] == 2
    assert report["removed"][0]["business_id"] == "b002"


def test_judge_cli_honours_the_disabled_filter_and_still_writes_one_input(tmp_path: Path):
    """With the filter off the stage must still produce ``examples_filtered.jsonl``
    — the next stage has one input path either way — and the report must say the
    negatives were never graded rather than implying a clean sweep."""
    cfg_path, cfg = cli_config(tmp_path, false_negative_filter=False)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    mined = [_example("q1", ["b002", "b003"])]
    negatives.write_examples_jsonl(mined, out_dir / negatives.EXAMPLES_MINED_FILENAME)
    documents = write_documents_file(tmp_path, ["b001", "b002", "b003"])

    class NeverCalled:
        name = model = "never-called"

        def complete_json(self, prompt: str) -> dict[str, Any]:
            raise AssertionError("the judge must not spend a call when the filter is off")

    code = judge.main(
        [
            "--config",
            str(cfg_path),
            "--documents",
            str(documents),
            "--assume-yes",
            "--artifacts-root",
            str(tmp_path),
        ],
        client=NeverCalled(),
    )
    assert code == 0

    filtered = negatives.read_examples_jsonl(out_dir / judge.EXAMPLES_FILTERED_FILENAME)
    assert [n.business_id for n in filtered[0].negatives] == ["b002", "b003"]
    report = json.loads((out_dir / judge.FALSE_NEGATIVE_REPORT_FILENAME).read_text())
    assert report["filter_enabled"] is False
    assert report["n_graded"] == 0
    assert report["n_removed"] == 0


def test_judge_cli_label_quality_reads_the_owners_labels_without_writing_them(tmp_path: Path):
    """``make synth-label-quality`` is the job that licenses using judge grades as
    training labels, so it has to be runnable — and it must leave qrels untouched."""
    cfg_path, cfg = cli_config(tmp_path)
    write_gold_file(tmp_path, [(g.query_id, g.query, g.query_type.value) for g in GOLD_FOR_LABELS])
    qrels = tmp_path / "qrels.csv"
    qrels.write_text(QRELS_CSV, encoding="utf-8")
    before = qrels.read_bytes()
    documents = write_documents_file(tmp_path, sorted(DOCUMENTS))
    stub = StubJudgeClient({"b001": 2, "b002": 1, "b003": 0, "b004": 2, "b005": 0})

    code = judge.main(
        [
            "--config",
            str(cfg_path),
            "--job",
            "label-quality",
            "--documents",
            str(documents),
            "--assume-yes",
            "--artifacts-root",
            str(tmp_path),
        ],
        client=stub,
    )
    assert code == 0
    assert qrels.read_bytes() == before, "the judge must never write the owner's labels"

    out_dir = generate.synth_dir(cfg, tmp_path)
    report = json.loads((out_dir / judge.LABEL_QUALITY_REPORT_FILENAME).read_text())
    # The stub was given the owner's own grades, so agreement is perfect and the
    # kappa is the one value a test may assert without inventing a measurement.
    assert report["n_sampled"] == len(QRELS_CSV.strip().splitlines()) - 1
    assert report["kappa_quadratic"]["n"] == report["n_sampled"]
    assert report["kappa_unweighted"]["kappa"] == pytest.approx(1.0)
    assert len(report["kappa_unweighted"]["matrix"]) == len(
        report["kappa_unweighted"]["categories"]
    )
    assert report["qrels_sha256"]


def test_splits_cli_refuses_before_the_judge_has_run(tmp_path: Path):
    cfg_path, cfg = cli_config(tmp_path)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    generate.write_queries_jsonl(
        list(paraphrase_pair()), out_dir / generate.QUERIES_DEDUPED_FILENAME
    )
    with pytest.raises(splits.SplitError, match="synth-judge"):
        splits.main(
            ["--config", str(cfg_path), "--artifacts-root", str(tmp_path)], embed=fake_embed
        )


def test_splits_cli_writes_a_verified_manifest_and_one_file_per_split(tmp_path: Path):
    cfg_path, cfg = cli_config(tmp_path)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    queries = _queries(60)
    generate.write_queries_jsonl(queries, out_dir / generate.QUERIES_DEDUPED_FILENAME)
    negatives.write_examples_jsonl(
        [_example(q.query_id, ["b900"]) for q in queries],
        out_dir / judge.EXAMPLES_FILTERED_FILENAME,
    )

    code = splits.main(
        ["--config", str(cfg_path), "--artifacts-root", str(tmp_path)], embed=fake_embed
    )
    assert code == 0

    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))
    splits.verify_splits(manifest, cfg=cfg)
    assert manifest.group_by == "cluster"
    assert manifest.n_queries == len(queries)
    total = 0
    for name in splits.SPLIT_NAMES:
        rows = negatives.read_examples_jsonl(splits.split_example_path(name, cfg, tmp_path))
        total += len(rows)
        for row in rows:
            assert manifest.assignment[row.query_id] == name
    assert total == len(queries)

    report = json.loads((out_dir / splits.SPLIT_REPORT_FILENAME).read_text())
    assert report["group_by"] == "cluster"
    assert report["clustering"]["cosine_threshold"] == cfg.synth.gold_dedup_cosine_threshold
    assert sum(report["example_counts"].values()) == len(queries)


def test_synth_pipeline_lets_no_paraphrase_straddle_the_split_boundary(tmp_path: Path):
    """The whole chain, as ``make synth`` runs it, with both planted pairs in it.

    Both pairs straddle the cross-encoder/LambdaMART boundary when split by id
    (pinned above). Run through the four real stages, each pair has to be closed
    by one of the two mechanisms, and the test asserts which one did it:

    * the reordered-words pair is an above-threshold near-duplicate, so the gold
      guard's third pass drops one of the two and the dedup report names it. Only
      one of them is a training query at all, which is the strongest possible
      version of "it cannot be on both sides".
    * the two queries from one business document are not duplicates, so both
      survive the guard — and the clustering's same-business merge is what keeps
      them in one split.

    This is the test that makes the out-of-fold guarantee a property of the
    pipeline rather than of functions nothing calls.
    """
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, cfg = cli_config(
        tmp_path,
        queries_per_business=2,
        businesses_sampled=3,
        batch_size=1,
        hard_negatives_per_query=2,
    )
    # Two document files on purpose: generation is given only the businesses the
    # plants are written from, while the judge needs a document for every mined
    # negative as well — which is how the real stages are fed too, since mining
    # returns businesses generation never sampled.
    sources = write_documents_file(tmp_path, [PARAPHRASE_A[0], PARAPHRASE_B[0], SAME_BUSINESS_ID])
    documents = write_documents_file(tmp_path / "all", ["b001", "b002", "b003", "b004", "b005"])
    client = PromptAwareStubClient(
        {
            PARAPHRASE_A[0]: [(PARAPHRASE_A[1], "constraint")],
            PARAPHRASE_B[0]: [(PARAPHRASE_B[1], "constraint")],
            SAME_BUSINESS_ID: [
                (SAME_BUSINESS_QUERIES[0], "specific"),
                (SAME_BUSINESS_QUERIES[1], "vibe"),
            ],
        }
    )
    config_arg = ["--config", str(cfg_path)]
    root = ["--artifacts-root", str(tmp_path)]

    assert (
        generate.main(
            [*config_arg, "--documents", str(sources), "--assume-yes", *root],
            client=client,
            embed=fake_embed,
        )
        == 0
    )
    assert negatives.main([*config_arg, *root], retrieve=fake_retriever(["b004", "b005"])) == 0
    assert (
        judge.main(
            [*config_arg, "--documents", str(documents), "--assume-yes", *root],
            client=StubJudgeClient({}),
        )
        == 0
    )
    assert splits.main([*config_arg, *root], embed=fake_embed) == 0

    out_dir = generate.synth_dir(cfg, tmp_path)
    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))
    splits.verify_splits(manifest, cfg=cfg)

    # Mechanism 1: one half of the reordered pair never became a training query.
    a, b = paraphrase_pair()
    reached = {q for q in (a.query_id, b.query_id) if q in manifest.assignment}
    assert len(reached) == 1, "the gold guard's internal pass must drop one of the two"
    dedup = json.loads((out_dir / generate.DEDUP_REPORT_FILENAME).read_text())
    assert dedup["n_removed_near_internal"] == 1
    dropped = [row for row in dedup["removed"] if row["kind"] == "near_internal"]
    assert {row["query_id"] for row in dropped} == {a.query_id, b.query_id} - reached

    # Mechanism 2: the two queries from one business both survived, together.
    first, second = same_business_pair()
    assert {first.query_id, second.query_id} <= set(manifest.assignment)
    assert manifest.assignment[first.query_id] == manifest.assignment[second.query_id]

    # The invariant, stated the way the trainers need it: neither plant has a
    # member in a cross-encoder split and another in a LambdaMART split.
    crossencoder = set(manifest.ids(*splits.CROSSENCODER_SPLITS))
    ltr = set(manifest.ids(*splits.LTR_SPLITS))
    for plant in ({a.query_id, b.query_id}, {first.query_id, second.query_id}):
        assert not (plant & crossencoder and plant & ltr)

    # And the per-split files follow the manifest, since that is what the trainers
    # actually open.
    for name in splits.SPLIT_NAMES:
        rows = negatives.read_examples_jsonl(splits.split_example_path(name, cfg, tmp_path))
        for row in rows:
            assert manifest.assignment[row.query_id] == name
    assert manifest.n_queries == len(reached) + 2


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
