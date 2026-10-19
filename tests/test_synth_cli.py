"""The synthetic-data CLI entry points: ``make synth-generate`` and ``make synth-splits``.

``tests/test_synth.py`` tests the functions. This file tests the two *programs*
the Makefile runs, because that is where the properties the project's honesty
rests on either hold or do not:

1. **The quota gate** (CLAUDE.md §10). A stage that spends API calls states the
   count and refuses without confirmation; a stage that spends nothing still
   prints its plan.
2. **Resumability.** A rerun must cost nothing it has already paid for, and —
   the sharper half — must never rewrite the training input *smaller* than the
   resumable store, which is how a rate-limited run could quietly reduce the
   dataset three stages before anybody trains on it.
3. **The gold leakage guard's receipt travels with the data.** The guard runs in
   stage 1; the trainers open stage 4's artifacts. Unless the split manifest
   records what the guard did — threshold, counts, which gold file — "these rows
   were deduplicated against the gold set" is a step somebody remembers running
   rather than a property anything can check. These tests pin that the receipt is
   recorded, that it is inside the manifest's checksum, that it is held against
   the running config, and that a manifest carrying no receipt is refused for a
   reported training run.
4. **The split grouping is enforced by the config, not by an argparse default.**
   A single flag used to be able to rebuild the manifest in the mode that lets
   two paraphrases of one query straddle the cross-encoder/LambdaMART boundary.

No API key, no model download, no Yelp data: the LLM client and the embedder are
injected, exactly as the two ``main`` functions' signatures allow.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from concierge.config import REPO_ROOT, Config, load_config
from concierge.synth import generate, judge, negatives, splits
from concierge.types import QueryType

# ─────────────────────────────────────────────────────────────────────────────
# Stubs and fixtures
# ─────────────────────────────────────────────────────────────────────────────

FAKE_EMBED_DIM = 512


def fake_embed(texts: list[str]) -> Any:
    """A hashed bag-of-words embedding: deterministic, no model, no download.

    Two texts sharing most of their words get a high cosine and two unrelated
    texts a low one, which is the only property the guard relies on. The
    dimension is fixed rather than vocabulary-sized because the real encoder's
    is: the guard embeds the gold set and each chunk of candidates separately and
    multiplies the results, so the two matrices have to agree on a width.
    """
    import numpy as np

    matrix = np.zeros((len(texts), FAKE_EMBED_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in generate.normalize_query_text(text).split():
            slot = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % FAKE_EMBED_DIM
            matrix[row, slot] += 1.0
    return matrix


class StubGenerator:
    """A generation client that answers for whichever businesses the prompt names.

    Keyed on the prompt rather than on call order, so these tests do not depend
    on the order ``sample_documents`` happens to produce. ``raise_for`` makes the
    batch containing one business fail with a given error, which is how a
    free-tier 429 or a malformed reply is simulated without waiting for one.
    """

    name = "stub"
    model = "stub-model"

    def __init__(
        self,
        queries_by_business: dict[str, list[tuple[str, str]]],
        *,
        raise_for: dict[str, Exception] | None = None,
    ) -> None:
        self.queries_by_business = queries_by_business
        self.raise_for = raise_for or {}
        self.calls: list[str] = []

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.calls.append(prompt)
        for business_id, error in self.raise_for.items():
            if business_id in prompt:
                raise error
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


def cli_config(tmp_path: Path, **synth_overrides: Any) -> tuple[Path, Config]:
    """Write a config for the CLI tests: the real one, redirected into ``tmp_path``.

    Dumped from the real ``Config`` object rather than written as a YAML literal,
    so every value a test does not deliberately override — the cosine threshold,
    the split fractions and ``min_split_grouping`` above all — is the one the
    pipeline actually runs with and cannot drift away from it.
    """
    import yaml

    cfg = load_config("configs/base.yaml")
    data = cfg.model_dump(mode="json")
    data["synth"].update(synth_overrides)
    data["evaluation"]["gold_queries"] = str(tmp_path / "gold_queries.csv")
    data["evaluation"]["qrels"] = str(tmp_path / "qrels.csv")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path, load_config(path)


def write_gold_file(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "gold_queries.csv"
    path.write_text(
        "query_id,query,query_type\n"
        + "".join(f"{qid},{text},{qtype}\n" for qid, text, qtype in rows),
        encoding="utf-8",
    )
    return path


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


def write_filtered_examples(queries: list[generate.SyntheticQuery], out_dir: Path) -> Path:
    """Stand in for ``make synth-judge``: one example per query, one negative each.

    The split stage only needs each example's ``query_id`` (it partitions by the
    manifest), so mining and judging are not re-tested here — ``test_synth.py``
    covers those CLIs.
    """
    return negatives.write_examples_jsonl(
        [
            negatives.TrainingExample(
                query_id=q.query_id,
                query=q.query,
                query_type=q.query_type,
                positive_business_id=q.business_id,
                negatives=(
                    negatives.HardNegative(
                        business_id="b900", retrieval_rank=1, retrieval_score=0.5
                    ),
                ),
            )
            for q in queries
        ],
        out_dir / judge.EXAMPLES_FILTERED_FILENAME,
    )


GOLD_QUERY = "cozy spot to read on a rainy afternoon"
GOLD_DUPLICATE = f"{GOLD_QUERY} downtown"  # near-duplicate: caught by the embedding pass


def run_generate(
    tmp_path: Path,
    cfg_path: Path,
    documents: Path,
    client: StubGenerator,
    *extra: str,
) -> int:
    return generate.main(
        [
            "--config",
            str(cfg_path),
            "--documents",
            str(documents),
            "--artifacts-root",
            str(tmp_path),
            *extra,
        ],
        client=client,
        embed=fake_embed,
    )


@pytest.fixture
def split_inputs(tmp_path: Path):
    """A real stage-1 run, plus the judge's output, ready for ``make synth-splits``.

    Returns ``(cfg_path, cfg, out_dir)``. One planted gold near-duplicate is
    generated and removed by the guard, so the dedup report under test has a
    non-zero removal count and a removed row the split stage can cross-check.
    """
    write_gold_file(tmp_path, [("g1", GOLD_QUERY, "vibe")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=2, businesses_sampled=3, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001", "b002", "b003"])
    client = StubGenerator(
        {
            "b001": [("quiet cafe with free wifi open late", "constraint")],
            "b002": [(GOLD_DUPLICATE, "vibe"), ("birria tacos al pastor", "specific")],
            "b003": [("somewhere loud for a big group dinner", "vibe")],
        }
    )
    assert run_generate(tmp_path, cfg_path, documents, client, "--assume-yes") == 0

    out_dir = generate.synth_dir(cfg, tmp_path)
    kept = generate.read_queries_jsonl(out_dir / generate.QUERIES_DEDUPED_FILENAME)
    write_filtered_examples(kept, out_dir)
    return cfg_path, cfg, out_dir


def run_splits(cfg_path: Path, tmp_path: Path, *extra: str) -> int:
    return splits.main(
        ["--config", str(cfg_path), "--artifacts-root", str(tmp_path), *extra],
        embed=fake_embed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument handling
# ─────────────────────────────────────────────────────────────────────────────


def test_limit_overrides_the_configured_sample_size(tmp_path: Path):
    """``--limit`` is how a smoke run stays cheap, so it has to reach the sampler
    rather than only the log line: one document means one batch means one call."""
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=1, businesses_sampled=3, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001", "b002", "b003"])
    client = StubGenerator({b: [(f"a query for {b}", "vibe")] for b in ("b001", "b002", "b003")})

    assert run_generate(tmp_path, cfg_path, documents, client, "--assume-yes", "--limit", "1") == 0

    assert len(client.calls) == 1
    kept = generate.read_queries_jsonl(
        generate.synth_dir(cfg, tmp_path) / generate.QUERIES_DEDUPED_FILENAME
    )
    assert len({q.business_id for q in kept}) == 1


def test_an_unsupported_documents_format_is_refused_by_name(tmp_path: Path):
    """A wrong ``--documents`` extension must name the formats, not fail on a
    parse deep inside polars."""
    cfg_path, _ = cli_config(tmp_path)
    bad = tmp_path / "documents.csv"
    bad.write_text("business_id,text\nb001,doc\n", encoding="utf-8")
    with pytest.raises(generate.SynthConfigError, match="parquet"):
        run_generate(tmp_path, cfg_path, bad, StubGenerator({}), "--assume-yes")


def test_artifacts_root_keeps_every_write_out_of_the_repo(tmp_path: Path):
    """``--artifacts-root`` has to redirect the *paid* cache too.

    The response cache deliberately lives outside ``artifacts/`` (``make clean``
    must not be able to delete API calls already paid for), which makes it the
    one path a half-wired root flag would leave pointing at the repo — and a test
    run would then write into the owner's real cache.
    """
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=1, businesses_sampled=1, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001"])
    client = StubGenerator({"b001": [("birria tacos", "specific")]})

    assert run_generate(tmp_path, cfg_path, documents, client, "--assume-yes") == 0

    assert (tmp_path / ".llm_cache" / "synth_generate.jsonl").exists()
    out_dir = generate.synth_dir(cfg, tmp_path)
    assert out_dir.is_relative_to(tmp_path)
    for name in (
        generate.QUERIES_FILENAME,
        generate.QUERIES_DEDUPED_FILENAME,
        generate.QUERIES_DEDUPED_CSV_FILENAME,
        generate.DEDUP_REPORT_FILENAME,
    ):
        assert (out_dir / name).exists()
    assert not (REPO_ROOT / "artifacts" / "synth" / cfg.fingerprint()).exists()


# ─────────────────────────────────────────────────────────────────────────────
# The quota gate (CLAUDE.md §10)
# ─────────────────────────────────────────────────────────────────────────────


def test_generation_refuses_to_spend_quota_without_confirmation(tmp_path: Path):
    """Without ``--assume-yes`` and without a terminal, the run must refuse — and
    it must refuse *before* the call, not after it."""
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=1, businesses_sampled=1, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001"])
    client = StubGenerator({"b001": [("birria tacos", "specific")]})

    with pytest.raises(generate.QuotaNotConfirmedError, match="1 API call"):
        run_generate(tmp_path, cfg_path, documents, client)

    assert client.calls == []
    assert not (generate.synth_dir(cfg, tmp_path) / generate.QUERIES_FILENAME).exists()


def test_assume_yes_still_states_the_call_count(tmp_path: Path, capsys):
    """``--assume-yes`` skips the question, not the disclosure: CLAUDE.md §10 asks
    for the number to be stated, and the help text promises it regardless."""
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, _ = cli_config(tmp_path, queries_per_business=1, businesses_sampled=2, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001", "b002"])
    client = StubGenerator({b: [(f"a query for {b}", "vibe")] for b in ("b001", "b002")})

    assert run_generate(tmp_path, cfg_path, documents, client, "--assume-yes") == 0

    err = capsys.readouterr().err
    assert "synthetic query generation" in err
    assert "2 API call(s) will be made" in err


def test_a_fully_cached_rerun_asks_nothing_and_spends_nothing(tmp_path: Path, capsys):
    """The resumability promise, from the CLI's side: the second run replays the
    stored responses, makes no call, and is not asked for a confirmation it would
    refuse (there is no terminal here, so being asked would raise)."""
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=1, businesses_sampled=1, batch_size=1)
    documents = write_documents_file(tmp_path, ["b001"])
    client = StubGenerator({"b001": [("birria tacos", "specific")]})

    assert run_generate(tmp_path, cfg_path, documents, client, "--assume-yes") == 0
    out_dir = generate.synth_dir(cfg, tmp_path)
    first = (out_dir / generate.QUERIES_DEDUPED_FILENAME).read_bytes()
    store_lines = (out_dir / generate.QUERIES_FILENAME).read_text().strip().splitlines()

    capsys.readouterr()
    # No --assume-yes on the second run: a cached run must not even reach the
    # confirmation, which would refuse for want of a terminal.
    assert run_generate(tmp_path, cfg_path, documents, client) == 0

    assert len(client.calls) == 1, "the second run must not call the API again"
    assert (out_dir / generate.QUERIES_DEDUPED_FILENAME).read_bytes() == first
    # The store is append-only, so a replayed batch must not re-append its rows:
    # otherwise the file grows on every rerun and its line count means nothing.
    assert (out_dir / generate.QUERIES_FILENAME).read_text().strip().splitlines() == store_lines
    assert "0 API call(s) will be made" in capsys.readouterr().err


def test_a_stage_that_spends_nothing_still_prints_its_plan(split_inputs, tmp_path: Path, capsys):
    """The split stage makes no API call and its docstring says the plan is printed
    anyway. It was not: the early return skipped ``describe()``, so the item counts
    the docstring promises never reached the operator."""
    cfg_path, _cfg, _ = split_inputs
    capsys.readouterr()

    assert run_splits(cfg_path, tmp_path) == 0

    err = capsys.readouterr().err
    assert "split assignment" in err
    assert "queries to assign" in err
    assert "0 API call(s) will be made" in err


# ─────────────────────────────────────────────────────────────────────────────
# Resumability
# ─────────────────────────────────────────────────────────────────────────────


def test_a_resumed_run_cannot_shrink_the_training_input(tmp_path: Path, capsys):
    """The training input is a function of the *store*, not of this process's luck.

    Set up: a first run generates for b002 only, so its batch is cached and its
    query is in the store. A second run is asked for b001 and b002; the b001 batch
    (sorted first) is rate-limited, which stops the loop before the cached b002
    batch is replayed. ``run.queries`` is therefore empty.

    Deduplicating over ``run.queries`` would rewrite ``queries_deduped.jsonl`` —
    which is rewritten, not appended — with *nothing*, silently throwing away a
    paid-for query three stages before anything trains on it. Deduplicating over
    the store keeps it, and the artifact records that the run was partial.
    """
    write_gold_file(tmp_path, [("g1", GOLD_QUERY, "vibe")])
    cfg_path, cfg = cli_config(tmp_path, queries_per_business=1, businesses_sampled=2, batch_size=1)
    only_second = write_documents_file(tmp_path / "one", ["b002"])
    both = write_documents_file(tmp_path / "two", ["b001", "b002"])
    queries = {
        "b001": [("ramen open past midnight", "constraint")],
        "b002": [("good pho", "specific")],
    }

    assert (
        run_generate(tmp_path, cfg_path, only_second, StubGenerator(queries), "--assume-yes") == 0
    )
    out_dir = generate.synth_dir(cfg, tmp_path)
    assert {
        q.query for q in generate.read_queries_jsonl(out_dir / generate.QUERIES_DEDUPED_FILENAME)
    } == {"good pho"}

    rate_limited = StubGenerator(
        queries, raise_for={"b001": generate.SynthRateLimitError("stub 429")}
    )
    assert run_generate(tmp_path, cfg_path, both, rate_limited, "--assume-yes") == 0

    kept = generate.read_queries_jsonl(out_dir / generate.QUERIES_DEDUPED_FILENAME)
    assert {q.query for q in kept} == {"good pho"}, "the paid-for query must survive the resume"

    report = json.loads((out_dir / generate.DEDUP_REPORT_FILENAME).read_text())
    assert report["generation"]["complete"] is False
    assert report["generation"]["stopped_early"] is True
    assert report["generation"]["n_queries_in_store"] == 1
    assert report["n_input"] == 1, "the guard ran over the store, not over this run"
    assert "PARTIAL" in capsys.readouterr().out


def test_a_complete_run_says_so_in_the_artifact(split_inputs, tmp_path: Path):
    """The flip side: "partial" is only a useful field if "complete" is recorded
    too, otherwise a reader cannot tell a complete run from an old artifact."""
    _, _cfg, out_dir = split_inputs
    report = json.loads((out_dir / generate.DEDUP_REPORT_FILENAME).read_text())
    assert report["generation"]["complete"] is True
    assert report["generation"]["stopped_early"] is False
    assert report["generation"]["n_batches_failed"] == 0
    assert report["generation"]["prompt_version"]


# ─────────────────────────────────────────────────────────────────────────────
# The gold leakage guard's receipt, recorded in the split manifest
# ─────────────────────────────────────────────────────────────────────────────


def test_the_split_manifest_records_the_gold_dedup_receipt(split_inputs, tmp_path: Path):
    """The point of the whole exercise: a trainer opening ``splits.json`` can see
    that these rows were deduplicated, at what threshold, against which gold file.

    Before this, ``dedup_report.json`` was written and read by nothing — the
    guarantee was a property of the library (a step somebody ran) rather than of
    the artifact a trainer consumes.
    """
    cfg_path, cfg, out_dir = split_inputs

    assert run_splits(cfg_path, tmp_path) == 0

    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))
    record = manifest.gold_dedup
    assert record is not None
    report = json.loads((out_dir / generate.DEDUP_REPORT_FILENAME).read_text())
    assert record.cosine_threshold == cfg.synth.gold_dedup_cosine_threshold
    assert record.n_removed_gold == report["n_removed_gold"] == 1
    assert record.n_kept == report["n_kept"] == manifest.n_queries
    assert record.n_gold == 1
    assert record.gold_sha256 == generate.file_sha256(tmp_path / "gold_queries.csv")
    assert record.report_sha256 == generate.file_sha256(out_dir / generate.DEDUP_REPORT_FILENAME)
    assert record.generation_complete is True

    # And the manifest is fit to train a reported number: the receipt is present,
    # ties to these rows, and the grouping was enforced.
    splits.verify_splits_for_training(manifest, cfg)
    report_json = json.loads((out_dir / splits.SPLIT_REPORT_FILENAME).read_text())
    assert report_json["fit_for_reported_training"] == {"ok": True, "reason": ""}
    assert report_json["gold_dedup"]["gold_sha256"] == record.gold_sha256
    assert report_json["min_grouping"] == "cluster"


def test_the_receipt_is_covered_by_the_manifest_checksum(split_inputs, tmp_path: Path):
    """A receipt that can be edited or deleted in place is not evidence.

    Both directions are checked, because they fail differently: editing a field
    changes the hashed payload, and *removing* the whole record would also have
    to, or "delete the inconvenient paragraph" would be a clean edit.
    """
    cfg_path, cfg, _ = split_inputs
    assert run_splits(cfg_path, tmp_path) == 0
    path = splits.splits_path(cfg, tmp_path)
    original = json.loads(path.read_text())

    edited = json.loads(json.dumps(original))
    edited["gold_dedup"]["n_removed_gold"] = 999
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(splits.SplitManifestCorruptError, match="checksum"):
        splits.load_manifest(path)

    removed = json.loads(json.dumps(original))
    removed["gold_dedup"] = None
    path.write_text(json.dumps(removed), encoding="utf-8")
    with pytest.raises(splits.SplitManifestCorruptError, match="checksum"):
        splits.load_manifest(path)


def test_a_receipt_from_another_threshold_is_refused(split_inputs, tmp_path: Path):
    """ "The dedup ran at cosine x" only means something against the x this config
    names. A manifest whose receipt was produced under a different threshold
    describes a different experiment, so the trainers' own verification refuses it
    rather than reporting clean."""
    cfg_path, cfg, _ = split_inputs
    assert run_splits(cfg_path, tmp_path) == 0
    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))

    moved = cfg.model_copy(
        update={"synth": cfg.synth.model_copy(update={"gold_dedup_cosine_threshold": 0.80})}
    )
    with pytest.raises(splits.SplitLeakageError, match="not the guard this configuration"):
        splits.verify_splits(manifest, cfg=moved)


def test_a_receipt_that_describes_other_rows_is_refused(split_inputs, tmp_path: Path):
    """The tie between the receipt and the data: a kept-count that does not match
    the number of queries being split means the report belongs to some other run,
    and a receipt for other rows proves nothing about these."""
    cfg_path, _, out_dir = split_inputs
    report_path = out_dir / generate.DEDUP_REPORT_FILENAME
    data = json.loads(report_path.read_text())
    data["n_kept"] = data["n_kept"] + 7
    report_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(splits.SplitLeakageError, match="different set of rows"):
        run_splits(cfg_path, tmp_path)


def test_a_query_the_guard_removed_cannot_come_back(split_inputs, tmp_path: Path):
    """The failure the counts alone cannot see.

    A hand-edited (or half-regenerated) ``queries_deduped.jsonl`` that puts a
    removed gold duplicate back would be split and trained on while the receipt
    beside it still describes a clean dedup. The split stage cross-checks the
    report's removed ids against the file it is about to split, so the two cannot
    disagree in silence.
    """
    cfg_path, _cfg, out_dir = split_inputs
    report = json.loads((out_dir / generate.DEDUP_REPORT_FILENAME).read_text())
    removed = [row for row in report["removed"] if row["kind"] == "near_gold"]
    assert removed, "the fixture plants a near-duplicate of the gold query"

    queries_path = out_dir / generate.QUERIES_DEDUPED_FILENAME
    resurrected = generate.SyntheticQuery(
        query_id=removed[0]["query_id"],
        query=removed[0]["query"],
        query_type=QueryType.VIBE,
        business_id="b002",
    )
    with queries_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(resurrected.to_json()) + "\n")

    with pytest.raises(splits.SplitLeakageError, match="are back in"):
        run_splits(cfg_path, tmp_path)


def test_a_report_that_cannot_name_its_gold_file_is_refused(split_inputs, tmp_path: Path):
    """A receipt without ``gold_sha256`` cannot answer "deduplicated against
    *which* gold set", so it is refused rather than recorded as reassurance.
    Regenerating costs no API calls, which is why failing closed here is cheap."""
    cfg_path, _, out_dir = split_inputs
    report_path = out_dir / generate.DEDUP_REPORT_FILENAME
    data = json.loads(report_path.read_text())
    data["gold_sha256"] = ""
    report_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(splits.SplitError, match="synth-generate"):
        run_splits(cfg_path, tmp_path)


def test_a_manifest_with_no_receipt_is_not_fit_for_a_reported_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Hand-built inputs are a legitimate testing path, so the split stage still
    runs — but the absence of evidence travels with the artifact instead of being
    forgotten: the manifest records no receipt, every read warns, the stage's own
    report says the split is unfit, and the strict gate refuses it."""
    cfg_path, cfg = cli_config(tmp_path)
    out_dir = generate.synth_dir(cfg, tmp_path)
    out_dir.mkdir(parents=True)
    queries = [
        generate.SyntheticQuery(
            query_id=generate.synthetic_query_id(f"b{i:03d}", f"query number {i}"),
            query=f"query number {i}",
            query_type=QueryType.VIBE,
            business_id=f"b{i:03d}",
        )
        for i in range(12)
    ]
    generate.write_queries_jsonl(queries, out_dir / generate.QUERIES_DEDUPED_FILENAME)
    write_filtered_examples(queries, out_dir)

    with caplog.at_level(logging.ERROR, logger="concierge.synth.splits"):
        assert run_splits(cfg_path, tmp_path) == 0
    assert "no gold-dedup report" in caplog.text

    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))
    assert manifest.gold_dedup is None
    # Still internally consistent — verify_splits checks claims, and there is no
    # claim here to contradict — but it warns, every time it is read.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="concierge.synth.splits"):
        splits.verify_splits(manifest, cfg=cfg)
    assert "no gold-dedup receipt" in caplog.text

    with pytest.raises(splits.SplitLeakageError, match="cannot support a reported number"):
        splits.verify_splits_for_training(manifest, cfg)
    report = json.loads((out_dir / splits.SPLIT_REPORT_FILENAME).read_text())
    assert report["fit_for_reported_training"]["ok"] is False
    assert "gold_dedup" in report["fit_for_reported_training"]["reason"]


def test_the_dedup_report_can_be_pointed_at_explicitly(split_inputs, tmp_path: Path):
    """``--dedup-report`` exists so a dataset assembled outside the standard layout
    can still carry its receipt, rather than being split without one."""
    cfg_path, cfg, out_dir = split_inputs
    moved = tmp_path / "elsewhere" / "dedup_report.json"
    moved.parent.mkdir(parents=True)
    moved.write_bytes((out_dir / generate.DEDUP_REPORT_FILENAME).read_bytes())
    (out_dir / generate.DEDUP_REPORT_FILENAME).unlink()

    assert run_splits(cfg_path, tmp_path, "--dedup-report", str(moved)) == 0

    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))
    assert manifest.gold_dedup is not None
    assert manifest.gold_dedup.report_sha256 == generate.file_sha256(moved)


# ─────────────────────────────────────────────────────────────────────────────
# The grouping is a config decision, not an argparse default
# ─────────────────────────────────────────────────────────────────────────────


def test_grouping_strength_orders_the_three_modes():
    """The ordering the enforcement is defined over. ``cluster`` implies
    ``business`` only because the same-business merge is on by default, which is
    why dropping that merge is itself treated as a weakening."""
    assert splits.grouping_is_at_least("cluster", "business")
    assert splits.grouping_is_at_least("business", "query")
    assert splits.grouping_is_at_least("query", "query")
    assert not splits.grouping_is_at_least("query", "cluster")
    assert not splits.grouping_is_at_least("business", "cluster")
    with pytest.raises(splits.SplitError, match="unknown grouping"):
        splits.grouping_is_at_least("cluster", "paragraph")


def test_a_weaker_grouping_flag_is_refused(split_inputs, tmp_path: Path):
    """The leak this closes is one flag wide.

    ``--group-by query`` rebuilt the manifest in exactly the mode where two
    paraphrases of one query can sit on opposite sides of the
    cross-encoder/LambdaMART boundary — and every downstream check still reported
    clean, because at the level of ids nothing is wrong. The grouping now has to
    clear ``synth.min_split_grouping``, and no manifest is written when it does
    not.
    """
    cfg_path, cfg, _ = split_inputs
    for weaker in ("query", "business"):
        with pytest.raises(splits.SplitError, match="min_split_grouping"):
            run_splits(cfg_path, tmp_path, "--group-by", weaker)
    assert not splits.splits_path(cfg, tmp_path).exists()

    with pytest.raises(splits.SplitError, match="no-merge-business"):
        run_splits(cfg_path, tmp_path, "--no-merge-business")


def test_lowering_the_config_is_how_a_weaker_grouping_is_chosen(tmp_path: Path):
    """A comparison run is still possible — deliberately, in the config.

    That is strictly stronger than a flag: the setting is part of the config
    fingerprint, so the weak run writes into its own artifact directory instead of
    overwriting the real manifest, and the manifest records which minimum applied.
    The last assertion is the consumer side: the same manifest is refused the
    moment it is verified against a config that requires clustering.
    """
    write_gold_file(tmp_path, [("g1", "good pho", "specific")])
    cfg_path, weak_cfg = cli_config(
        tmp_path,
        queries_per_business=1,
        businesses_sampled=4,
        batch_size=1,
        min_split_grouping="query",
    )
    documents = write_documents_file(tmp_path, [f"b{i:03d}" for i in range(4)])
    client = StubGenerator(
        {f"b{i:03d}": [(f"a distinct query about place {i}", "vibe")] for i in range(4)}
    )
    assert run_generate(tmp_path, cfg_path, documents, client, "--assume-yes") == 0
    out_dir = generate.synth_dir(weak_cfg, tmp_path)
    write_filtered_examples(
        generate.read_queries_jsonl(out_dir / generate.QUERIES_DEDUPED_FILENAME), out_dir
    )

    assert run_splits(cfg_path, tmp_path, "--group-by", "query") == 0

    manifest = splits.load_manifest(splits.splits_path(weak_cfg, tmp_path))
    assert manifest.group_by == "query"
    assert manifest.min_grouping == "query"
    splits.verify_splits(manifest, cfg=weak_cfg)

    strict = weak_cfg.model_copy(
        update={"synth": weak_cfg.synth.model_copy(update={"min_split_grouping": "cluster"})}
    )
    with pytest.raises(splits.SplitLeakageError, match="weaker than the configured minimum"):
        splits.verify_splits(manifest, cfg=strict)


def test_a_manifest_that_contradicts_its_own_minimum_is_refused(split_inputs, tmp_path: Path):
    """Self-consistency, checked separately from the config comparison: a manifest
    claiming a minimum its own grouping does not meet is incoherent, so neither
    of its two claims can be trusted."""
    cfg_path, cfg, _ = split_inputs
    assert run_splits(cfg_path, tmp_path) == 0
    manifest = splits.load_manifest(splits.splits_path(cfg, tmp_path))
    broken = splits.SplitManifest(
        seed=manifest.seed,
        group_by="query",
        fractions=manifest.fractions,
        assignment=manifest.assignment,
        checksum=splits.compute_checksum(
            manifest.assignment,
            manifest.seed,
            manifest.fractions,
            "query",
            min_grouping="cluster",
            gold_dedup=manifest.gold_dedup,
        ),
        n_groups=manifest.n_groups,
        config_fingerprint=manifest.config_fingerprint,
        min_grouping="cluster",
        gold_dedup=manifest.gold_dedup,
    )
    with pytest.raises(splits.SplitLeakageError, match="contradicts itself"):
        splits.verify_splits(broken, cfg=cfg)
