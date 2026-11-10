"""Write ``artifacts/eval/runs.json``: the pool the labeling tool grades from.

**The gap this closes.** Pooled relevance judging needs, for each gold query,
the top results of *every system being compared* (CLAUDE.md §6). The labeling
CLI reads that from a runs file. Nothing in the repository wrote one. `make
eval` cannot: it refuses to report before labels exist, and labels cannot exist
before a pool, which is a cycle with the owner's 25-hour labeling task on the
far side of it. The documented workaround was to run a system over the gold
queries by hand and dump JSON in the right shape — a bespoke script standing
between the owner and the single most expensive step in the project.

So this module runs the systems and writes the file. It is deliberately thin:
every piece of the work already existed in `eval.run_eval` and is imported, not
reimplemented. If the pool were built by different code than the ablation
measures, the two could disagree about what a system returns, and the labels
would describe results no reported system ever produced.

**What it refuses to do.** A variant whose artifacts are missing is *skipped and
named*, never degraded. `SearchPipeline` will happily fall back when a model is
absent — that is right for serving, where a slightly worse answer beats an
error, and wrong here. A degraded run recorded under the name `full` would
attribute results to a system that did not produce them, and because pooling
tracks which system surfaced each candidate, the error would be baked into the
labels and survive every later correction. Writing fewer systems is honest;
writing the wrong label for one is not.

Incomplete pools are *expected*, not a failure mode: in Phase 1 only `bm25`
exists. Incremental pooling is the design (CLAUDE.md §6) — `--merge` keeps
earlier systems' results so adding a system adds only its newly surfaced
candidates to the grading queue.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from concierge.cli import PrerequisiteError, run_cli
from concierge.config import Config, load_config
from concierge.pipeline import SearchPipeline, Stage
from concierge.query.parser import count_cached_parses
from concierge.synth.generate import QuotaPlan, require_confirmation
from concierge.types import GoldQuery
from eval.run_eval import (
    DEFAULT_EVAL_NOW,
    CandidateDepths,
    PipelineFactory,
    SearcherFactory,
    SystemSpec,
    ablation_specs,
    execute_system,
    parse_now,
    read_gold_file,
    resolve_config_path,
)

#: How deep to pool per system.
#:
#: CLAUDE.md §6 says top 20, and that is a judging-cost decision rather than a
#: retrieval one: pooling depth multiplied by the number of systems and 150
#: queries is the size of the owner's labeling queue. Deeper pools find more
#: relevant documents (better Recall@100) at a directly proportional cost in
#: hours of their time.
DEFAULT_POOL_DEPTH = 20


def _emit(message: str) -> None:
    print(message, file=sys.stderr)


def plan_quota(specs: Sequence[SystemSpec], gold: Sequence[GoldQuery], cfg: Config) -> QuotaPlan:
    """What running these systems will cost in LLM parse calls.

    Only variants with `Stage.QUERY_LLM` call the API, and the parse cache is
    keyed on the query — not on the system — so two LLM variants over the same
    gold set cost one set of calls, not two. Counting per variant would roughly
    double the stated cost and train the owner to wave the confirmation through,
    which is the failure this gate exists to prevent.
    """
    llm_specs = [s for s in specs if s.variant.enabled(Stage.QUERY_LLM)]
    if not llm_specs:
        return QuotaPlan(
            job="build pool",
            n_items=0,
            items_label="queries needing an LLM parse",
            n_batches=0,
            n_cached_batches=0,
            n_api_calls=0,
            batch_size=1,
            note=(
                "no selected system enables Stage 1's LLM parser (rule-based parsing costs nothing)"
            ),
        )
    queries = [gq.query for gq in gold]
    cached = count_cached_parses(queries, cfg)
    return QuotaPlan(
        job="build pool",
        n_items=len(queries),
        items_label="queries needing an LLM parse",
        n_batches=len(queries),
        n_cached_batches=cached,
        n_api_calls=len(queries) - cached,
        batch_size=1,
        note=(
            f"{len(llm_specs)} selected system(s) parse with the LLM "
            f"({', '.join(s.key for s in llm_specs)}); they share one parse cache, "
            "so the cost is per query, not per system"
        ),
    )


def select_specs(keys: Sequence[str] | None) -> tuple[SystemSpec, ...]:
    """The systems to pool from, defaulting to every ablation row.

    Unknown keys are an error naming what is available, rather than being
    ignored: a typo that silently pooled from five systems instead of six would
    leave a system's results unjudged and show up only as an inflated
    "unjudged in top 10" count much later.
    """
    specs = ablation_specs(include_supplementary=True)
    if not keys:
        return specs
    by_key = {s.key: s for s in specs}
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        raise SystemExit(
            f"unknown system(s) {unknown}. Available: {sorted(by_key)}",
        )
    return tuple(by_key[k] for k in keys)


def partition_available(
    specs: Sequence[SystemSpec],
    loaded: SearchPipeline,
) -> tuple[list[SystemSpec], list[tuple[str, str]]]:
    """Split the systems into those the loaded pipeline can run and those it cannot.

    Asked of `PipelineFactory.missing_artifacts` — the same check the ablation
    uses — rather than re-deriving which stage needs which file. If the pool and
    the ablation disagreed about whether a system is available, the pool could
    contain a system `make eval` will not measure, or omit one it will.

    Returns `(runnable, [(key, what_is_missing)])`.
    """
    runnable: list[SystemSpec] = []
    unavailable: list[tuple[str, str]] = []
    for spec in specs:
        missing = PipelineFactory.missing_artifacts(loaded, spec.variant)
        if missing:
            unavailable.append((spec.key, "; ".join(missing)))
        else:
            runnable.append(spec)
    return runnable, unavailable


def read_existing_pool(path: Path) -> dict[str, dict[str, list[str]]]:
    """Read a runs file to merge onto, validated by the reader that will use it.

    Validated by the labeling CLI's own `load_runs` rather than by a check
    written here: that is what reads this file next, it names the offending key,
    and a second opinion about the format is exactly how two sides of a contract
    drift apart. A bare `isinstance(..., dict)` would wave `{"bm25": "oops"}`
    through — it *is* a JSON object — and `--merge` would then copy a malformed
    run forward and rewrite it as though it were sound.
    """
    from eval.label_cli import load_runs

    try:
        return load_runs(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"--merge given but the existing {path} is not a valid runs file: {exc}"
        ) from exc


def merge_runs(
    existing: dict[str, dict[str, list[str]]],
    fresh: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    """Fresh results win per system; systems absent from `fresh` are kept.

    Merged at the system level, not the query level. A system that ran is
    replaced wholesale, because a half-updated run — some queries from the
    current code, some from a checkout two weeks ago — is a system that never
    existed, and every number computed from it would describe nothing.
    """
    merged = {system: dict(run) for system, run in existing.items()}
    merged.update({system: dict(run) for system, run in fresh.items()})
    return merged


def build(
    cfg: Config,
    *,
    gold: Sequence[GoldQuery],
    specs: Sequence[SystemSpec],
    depth: int,
    factory: SearcherFactory | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Run each system over the gold queries. Returns `(runs, skip_notes)`.

    `depth` sets the final cut only. Retrieve and rerank widths come from the
    config, because the pool must be built at the depths the ablation will be
    measured at — pooling at other widths would grade a different system's
    results than the one that gets reported.

    `factory` is injectable for the same reason `run_eval` makes it injectable:
    the pooling logic here — depth, which systems are skipped, how a merge
    behaves — is worth testing without a dataset, an index or a trained model.
    """
    depths = CandidateDepths(
        retrieve=cfg.candidates.retrieve,
        rerank=cfg.candidates.rerank,
        final=depth,
    )
    if factory is None:
        factory = PipelineFactory(now=now) if now is not None else PipelineFactory()
    runs: dict[str, dict[str, list[str]]] = {}
    notes: list[str] = []

    for spec in specs:
        run = execute_system(spec, gold, factory, cfg, depths)
        if not run.ran:
            notes.append(f"{spec.key}: skipped — {run.skip_reason or 'did not run'}")
            _emit(f"  skip {spec.key}: {run.skip_reason or 'did not run'}")
            continue
        ranked = {qid: ids for qid, ids in run.ranked.items() if ids}
        if not ranked:
            notes.append(f"{spec.key}: ran but returned nothing for every query")
            _emit(f"  skip {spec.key}: returned no results for any query")
            continue
        runs[spec.key] = ranked
        n_results = sum(len(v) for v in ranked.values())
        failed = f", {len(run.errors)} query error(s)" if run.errors else ""
        _emit(f"  ok   {spec.key}: {len(ranked)} queries, {n_results} results{failed}")
    return runs, notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.build_runs",
        description=(
            "Run the ablation's systems over the gold queries and write the runs file "
            "the labeling CLI pools from."
        ),
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the runs file. Defaults to <evaluation.out_dir>/runs.json.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_POOL_DEPTH,
        help=f"Results per system per query (default {DEFAULT_POOL_DEPTH}; CLAUDE.md §6).",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="Comma-separated system keys. Default: every ablation row and diagnostic.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Keep systems already in the runs file instead of replacing it. "
            "This is what makes incremental pooling work: only the new system's "
            "newly surfaced results reach the grading queue."
        ),
    )
    parser.add_argument(
        "--now",
        default=None,
        metavar="ISO",
        help=(
            "the instant hours constraints are resolved against. Defaults to the "
            f"same pinned instant `make eval` uses ({DEFAULT_EVAL_NOW.isoformat(timespec='minutes')}, "
            "a Monday midday), NOT the wall clock. Pass the same value to both or "
            "neither."
        ),
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Do not prompt before spending LLM parse quota (the cost is still printed).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.depth < 1:
        raise SystemExit("--depth must be at least 1")

    cfg = load_config(args.config)
    gold, n_examples = read_gold_file(cfg.evaluation.gold_queries)
    if not gold:
        # The EXAMPLE rows are filtered by the reader, so an all-example file
        # arrives here as an empty list. Pooling it would write `{}` and let the
        # labeling tool report a clean, empty, meaningless session.
        raise PrerequisiteError(
            f"{resolve_config_path(cfg.evaluation.gold_queries)} contains no queries of the "
            f"owner's own ({n_examples} EXAMPLE row(s) skipped). The gold queries are "
            "written by hand — see eval/README.md and CLAUDE.md §6."
        )
    if n_examples:
        _emit(f"note: skipped {n_examples} EXAMPLE row(s) in the gold file")

    specs = select_specs(
        [k.strip() for k in args.systems.split(",") if k.strip()] if args.systems else None
    )
    out_path = resolve_config_path(
        Path(args.out) if args.out else cfg.evaluation.out_dir / "runs.json"
    )

    _emit(f"pooling top {args.depth} from {len(specs)} system(s) over {len(gold)} gold query(ies)")

    # Availability is settled *before* the quota prompt, for two reasons. The
    # prompt has to quote the cost of the work that will actually happen — a
    # figure that counts LLM systems which cannot be built overstates it, and a
    # confirmation that is routinely wrong is one the owner learns to wave
    # through. And on a fresh checkout every system is unavailable for the same
    # reason, so asking first means asking to spend quota on nothing.
    # Pinned to the same instant `make eval` scores at, not to the wall clock.
    # "open after 10pm" has a different answer on a Saturday than on a Tuesday,
    # so a pool built tonight and scored next week surfaces businesses the pool
    # never covered — which the metrics then score as 0 for every system.
    # Nothing warns about it: the report would simply be wrong and look fine.
    resolved_now = parse_now(args.now)
    _emit(
        f"hours resolved against {resolved_now.isoformat(timespec='minutes')} (pass the same --now to `make eval`)"
    )

    factory = PipelineFactory(now=resolved_now)
    try:
        loaded = factory.base(cfg)
    except Exception as exc:
        # One line, not one per system: nine copies of the same
        # "run `make data`" buries the single thing the owner has to do.
        raise PrerequisiteError(f"the search pipeline could not be loaded.\n  {exc}") from exc

    runnable, unavailable = partition_available(specs, loaded)
    for key, reason in unavailable:
        _emit(f"  skip {key}: needs {reason}")
    if not runnable:
        raise PrerequisiteError(
            "no selected system can run yet, so no runs file was written. "
            "Pooling needs at least one.\n  "
            + "\n  ".join(f"{key}: needs {reason}" for key, reason in unavailable)
        )

    require_confirmation(plan_quota(runnable, gold, cfg), assume_yes=args.assume_yes)

    runs, notes = build(
        cfg, gold=gold, specs=runnable, depth=args.depth, factory=factory, now=resolved_now
    )

    if not runs:
        # Not an empty file: that would look like a completed pool of nothing.
        raise PrerequisiteError(
            "every system that could run returned no results, so no runs file "
            "was written.\n  " + "\n  ".join(notes)
        )

    if args.merge and out_path.exists():
        existing = read_existing_pool(out_path)
        kept = sorted(set(existing) - set(runs))
        runs = merge_runs(existing, runs)
        if kept:
            _emit(f"merged: kept {len(kept)} earlier system(s) ({', '.join(kept)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(f"wrote {out_path}: {len(runs)} system(s)")
    _emit("next: `make label` to grade the pool.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run_cli(main))
