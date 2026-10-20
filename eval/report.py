"""Rendering for ``make eval``: markdown tables, a latency plot, and the raw JSON.

This module only formats what ``eval/run_eval.py`` measured. It computes
nothing, so a formatting change can never move a number — and nothing here can
invent one, because every value it prints comes out of a
:class:`~eval.run_eval.MetricCell` that is either a real measurement or
``None``.

Three formatting rules, each of them an honesty rule in disguise:

1. **A missing measurement renders as an em dash plus a reason, never as 0.**
   An unrun ablation row is a stated gap in the table.
2. **Every table states its own *n*.** The counts are printed above each table
   rather than once at the top of the document, because a table pasted into the
   README on its own must still carry its denominators.
3. **A hedged comparison keeps its words.** The sentence saying an interval
   crosses zero is built in ``run_eval`` and reproduced verbatim here, so the
   caveat cannot be lost in rendering.

The JSON output exists so no number is ever retyped by hand into the README:
every figure in the markdown is also a machine-readable field with full
precision.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from eval.run_eval import (
    AblationRow,
    EvalResults,
    LatencySummary,
    MetricCell,
)

#: Printed wherever a measurement does not exist. One symbol, used everywhere,
#: so a blank cell is never mistaken for a zero.
MISSING = "—"

#: Marks the one sweep row whose depths are the shipped operating point. The
#: served system's NDCG lives there and not in the ablation table (see
#: ``render_header``), so it has to be findable without matching integers by eye.
OPERATING_POINT_MARK = "◀"

#: Displayed precision. The JSON keeps full precision; three decimals is as far
#: as a metric with a confidence interval tens of thousandths wide can honestly
#: be read.
DECIMALS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Small formatters
# ─────────────────────────────────────────────────────────────────────────────


def fmt_float(value: float | None, decimals: int = DECIMALS) -> str:
    return MISSING if value is None else f"{value:.{decimals}f}"


def fmt_cell(cell: MetricCell | None) -> str:
    """A metric as ``mean [low, high]``, or a dash with its reason."""
    if cell is None:
        return MISSING
    if cell.mean is None:
        return f"{MISSING} ({cell.note or 'undefined'})"
    return f"{fmt_float(cell.mean)} [{fmt_float(cell.low)}, {fmt_float(cell.high)}]"


def fmt_n(cell: MetricCell | None) -> str:
    """The denominator behind a cell: how many queries it is defined on."""
    if cell is None:
        return MISSING
    if cell.n_undefined:
        return f"{cell.n_defined} of {cell.n_total}"
    return str(cell.n_defined)


def fmt_latency(latency: LatencySummary | None) -> str:
    if latency is None or latency.p95_ms is None:
        return MISSING
    return f"{latency.p95_ms:.1f}"


def fmt_fraction(value: float | None) -> str:
    return MISSING if value is None else f"{value * 100:.1f}%"


def fmt_count(n: int, singular: str, plural: str | None = None) -> str:
    """``1 gold query`` / ``5 gold queries``. The report is pasted into a README
    a hiring manager reads, so the grammar is part of the deliverable."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def denominator_note(results: EvalResults) -> str:
    """The *n* line that travels under a table.

    Repeated under each table rather than stated once at the top of the report,
    because these tables are meant to be pasted into a README individually and
    a metric separated from its denominators is not a result. It is also where
    the partial-run stamp reappears, so a smoke-test table cannot be lifted out
    of the document and read as a measurement.
    """
    cov = results.coverage
    note = (
        f"*n = {fmt_count(cov.n_queries_with_relevant, 'gold query', 'gold queries')} with "
        f"at least one relevant label (of {cov.n_gold_queries} in scope), "
        f"{fmt_count(cov.n_labeled_pairs, 'labeled pair')} "
        f"({cov.n_positive_pairs} graded relevant). Queries with no relevant label have "
        "no defined NDCG and are excluded from every mean, never scored 0.*"
    )
    if results.partial_run_note:
        note += f"\n\n*{results.partial_run_note}*"
    return note


def type_denominator_note(results: EvalResults, qtype: str, n_queries: int) -> str:
    """The *n* line for one query-type table."""
    cov = results.coverage
    pairs = cov.labeled_pairs_by_type.get(qtype)
    positives = cov.positive_pairs_by_type.get(qtype)
    counted = (
        f"{fmt_count(pairs, 'labeled pair')} ({positives} graded relevant)"
        if pairs is not None and positives is not None
        else "labeled pairs not counted for this type"
    )
    note = (
        f"*n = {fmt_count(n_queries, qtype + ' query', qtype + ' queries')} in scope, "
        f"{counted}. The `n defined` column is how many of them have a defined "
        "metric and so entered the mean.*"
    )
    if results.partial_run_note:
        note += f"\n\n*{results.partial_run_note}*"
    return note


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A pipe table. Written by hand rather than with a dependency because the
    output has to be pasteable into a README unchanged."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────


def render_header(results: EvalResults) -> str:
    cov = results.coverage
    lines = [
        "# Concierge — evaluation report",
        "",
        f"Generated {results.generated_at} · config `{results.config_path}` "
        f"(fingerprint `{results.config_fingerprint}`)",
        "",
    ]
    if results.partial_run_note:
        lines += [f"> **{results.partial_run_note}**", ""]
    lines += [
        "## What every number below is computed over",
        "",
        f"- **{cov.n_gold_queries}** gold {'query' if cov.n_gold_queries == 1 else 'queries'} in scope for this run.",
        f"- **{cov.n_queries_with_labels}** of them have any relevance label; "
        f"**{cov.n_queries_with_relevant}** have at least one *relevant* label and so "
        "have a defined NDCG.",
        f"- **{cov.n_labeled_pairs}** labeled (query, business) pairs, of which "
        f"**{cov.n_positive_pairs}** are graded relevant "
        f"(grade >= {results.relevance_threshold}).",
        f"- Mean labeled pairs per judged query: {fmt_float(cov.mean_pairs_per_labeled_query, 1)}.",
        f"- Headline metric: **NDCG@{results.ndcg_k}**; recall is measured at "
        f"**{results.recall_k}**.",
        "- Query types in scope: "
        + ", ".join(f"{name} = {n}" for name, n in sorted(results.query_type_counts.items()))
        + ".",
        "",
        "### How to read a cell",
        "",
        "`mean [low, high]` is the metric and its bootstrap confidence interval over "
        "**queries** (queries are the unit of resampling because they are the "
        "independent observations). A metric is **undefined** for a query with no "
        "relevant label — such queries are excluded from the mean, never scored 0, and "
        "the `n` column says how many survived.",
        "",
        f"Ablation rows return **{results.eval_depth}** results per query, because "
        f"Recall@{results.recall_k} cannot be computed from a shorter list. That "
        "widens the cascade: every ablation row runs at "
        f"retrieve={max(results.operating_depths.retrieve, results.eval_depth)} / "
        f"rerank={max(results.operating_depths.rerank, results.eval_depth)} / "
        f"final={results.eval_depth}, while the production operating point is "
        f"retrieve={results.operating_depths.retrieve} / "
        f"rerank={results.operating_depths.rerank} / "
        f"final={results.operating_depths.final}.",
        "",
        "**That difference is not only a latency difference, and this note used to "
        "say it was.** For the rows without Stage 4, a deeper cascade leaves the "
        f"top-{results.ndcg_k} unchanged — it is the same ranking cut deeper — so "
        "their NDCG and MRR describe the served system. For the row *with* Stage 4 it "
        "does not: LambdaMART scores and re-sorts every candidate the reranker kept, "
        f"so at the eval depth it re-sorts "
        f"{max(results.operating_depths.rerank, results.eval_depth)} candidates where "
        f"production hands it {results.operating_depths.rerank}, and the top-"
        f"{results.ndcg_k} differs whenever it promotes something from cross-encoder "
        f"rank {results.operating_depths.rerank + 1} or deeper. So the full row's "
        "NDCG and MRR in table 1 are **measured at the eval depth, not at the "
        "operating point**.",
        "",
        "The served system's own NDCG is the row marked "
        f"**{OPERATING_POINT_MARK} operating point** in section 3, which runs at "
        f"exactly {results.operating_depths.retrieve} / "
        f"{results.operating_depths.rerank} / {results.operating_depths.final} over "
        "the same queries and the same labels. Quote that one for the served system, "
        "and section 3's latency — not the ablation table's — as the serving cost.",
        "",
        "Latency is end-to-end wall clock around the search call, p95 by nearest rank "
        "(an observed value, not an interpolation). The first query of each system is "
        "excluded and reported separately as a cold start: it pays for model loading "
        "and a cold page cache, which is not steady-state serving.",
        "",
        "Hours constraints were resolved against a **pinned** instant, "
        f"`{results.resolved_now}` — not the wall clock, so two runs of this report "
        "describe the same night and are comparable. Change it with `--now`.",
        (
            "Each measurement had its own LLM parse cache (`--cold-parse-cache`), so "
            "Stage 1's latency below is a real call, not a cache read."
            if results.cold_parse_cache
            else "The LLM parse cache was **shared** across measurements (the default, "
            "because isolating it costs one API call per query per row). Each row and "
            "sweep point states where its Stage 1 answers came from; a row served from "
            "cache does not include Stage 1's network round trip in its latency."
        ),
    ]
    return "\n".join(lines)


def render_ablation(results: EvalResults) -> str:
    ladder = [r for r in results.rows if not r.supplementary]
    extra = [r for r in results.rows if r.supplementary]
    lines = [
        f"## 1. Ablation — what each stage is worth (NDCG@{results.ndcg_k})",
        "",
        "Each row adds exactly one stage to the row above it, and the comparison "
        "column is a **paired** bootstrap against that row — paired because both "
        "systems answered the same queries, so query difficulty cancels instead of "
        "swamping the difference.",
        "",
        _ablation_table(results, ladder),
        "",
        denominator_note(results),
        "",
        f"`unjudged@{results.ndcg_k}` is the share of each system's top-"
        f"{results.ndcg_k} that nobody has labeled. Those results are scored 0, so a "
        "row with a high unjudged share is being penalised for surfacing places the "
        "pool never covered — the comparison is not yet fair at that point, and the "
        "fix is more labeling, not a different metric.",
        "",
        "### Exactly what each row ran",
        "",
        "The variant names are `concierge.pipeline`'s own, so a row can be "
        "reproduced with `pipeline.search(query, variant=VARIANTS[name])`. A row "
        "whose model was missing is refused rather than quietly run with a "
        "shorter cascade, which is why a gap here is a gap and not a weak number.",
        "",
        _table(
            ("System", "variant", "stages enabled", "cross-encoder"),
            [
                [
                    row.label,
                    f"`{row.variant_name}`",
                    ", ".join(row.stages) or MISSING,
                    row.reranker or "off",
                ]
                for row in results.rows
            ],
        ),
        "",
        "### What the comparisons actually support",
        "",
    ]
    for row in ladder:
        if not row.ran:
            lines.append(f"- **{row.label}** — not run: {row.skip_reason}")
        elif row.vs_previous is None:
            lines.append(f"- **{row.label}** — baseline row; nothing to compare against.")
        else:
            lines.append(f"- **{row.label}** — {row.vs_previous.sentence}")
    if extra:
        lines += [
            "",
            "### Supplementary comparison (not a rung of the ladder)",
            "",
            _ablation_table(results, extra),
            "",
            denominator_note(results),
            "",
        ]
        for row in extra:
            if not row.ran:
                lines.append(f"- **{row.label}** — not run: {row.skip_reason}")
            else:
                lines.append(f"- **{row.label}** — {row.rationale}")
                if row.vs_previous is not None:
                    lines.append(f"  - {row.vs_previous.sentence}")
    errors = [(r.label, r.n_query_errors) for r in results.rows if r.ran and r.n_query_errors]
    if errors:
        lines += [
            "",
            "**Queries that raised an exception** (scored as if the system returned "
            "nothing, which is what happened): "
            + ", ".join(f"{label}: {n}" for label, n in errors)
            + ".",
        ]
    return "\n".join(lines)


def _ablation_table(results: EvalResults, rows: Sequence[AblationRow]) -> str:
    header = (
        "System",
        f"NDCG@{results.ndcg_k}",
        f"Recall@{results.recall_k}",
        "MRR",
        "p95 ms",
        "p50 ms",
        f"unjudged@{results.ndcg_k}",
        "n queries",
        "n latency samples",
        "Stage 1",
    )
    body: list[list[str]] = []
    for row in rows:
        if not row.ran:
            body.append(
                [row.label, f"{MISSING} (not run: {row.skip_reason})"]
                + [MISSING] * (len(header) - 2)
            )
            continue
        body.append(
            [
                row.label,
                fmt_cell(row.ndcg),
                fmt_cell(row.recall),
                fmt_cell(row.mrr),
                fmt_latency(row.latency),
                fmt_float(row.latency.p50_ms if row.latency else None, 1),
                f"{fmt_fraction(row.unjudged_fraction)} ({row.n_unjudged}/{row.n_top_k_results})",
                fmt_n(row.ndcg),
                str(row.latency.n_samples if row.latency else 0),
                # Where Stage 1's answers came from. In the table rather than a
                # footnote because it is what makes the two latency columns
                # beside it readable: a row whose parses were served from the
                # persistent cache has no network round trip in its p95, and
                # without this column a reader comparing it to a cold row would
                # attribute the gap to the stage being ablated.
                row.parse.sentence if row.parse else MISSING,
            ]
        )
    out = _table(header, body)
    caveats = [
        f"{r.label}: {r.latency.caveat}" for r in rows if r.ran and r.latency and r.latency.caveat
    ]
    cold = [
        f"{r.label}: {r.latency.cold_start_ms:.0f} ms"
        for r in rows
        if r.ran and r.latency and r.latency.cold_start_ms is not None
    ]
    if cold:
        out += "\n\nCold start (first query, excluded from the percentiles above): " + ", ".join(
            cold
        )
    if caveats:
        out += "\n\nLatency caveats: " + "; ".join(caveats)
    survival = [
        f"{r.label}: {r.filter_survival.sentence}"
        for r in rows
        if r.ran and r.filter_survival is not None and r.filter_survival.n_queries
    ]
    if survival:
        out += (
            "\n\n**After the hard filters** (the count the constraint backfill's "
            "default trigger does *not* look at — it counts candidates before "
            "filtering, so a query that retrieved a full list and kept five never "
            "fires it): " + "; ".join(survival)
        )
    return out


def render_by_type(results: EvalResults) -> str:
    """Per-query-type tables, one table per type.

    Grouped by type rather than by system so the question the breakdown exists
    to answer — *which* stage helps *which* kind of query — is one column
    comparison instead of a hunt across six tables.
    """
    lines = [
        "## 2. Per query type",
        "",
        "The stages do not help every kind of request equally, and this is a headline "
        "result rather than a diagnostic. Note the sample sizes: each type holds a "
        "fraction of the gold set, so these intervals are much wider than the overall "
        "ones and small gaps here are usually noise.",
        "",
    ]
    ran = [r for r in results.rows if r.ran]
    if not ran:
        return "\n".join([*lines, "No system ran, so there is no breakdown to show."])

    types = [bt.query_type for bt in ran[0].by_type]
    for qtype in types:
        n_queries = next((bt.n_queries for bt in ran[0].by_type if bt.query_type == qtype), 0)
        lines += [
            f"### {qtype} ({fmt_count(n_queries, 'gold query', 'gold queries')})",
            "",
        ]
        header = (
            "System",
            f"NDCG@{results.ndcg_k}",
            f"Recall@{results.recall_k}",
            "MRR",
            f"unjudged@{results.ndcg_k}",
            "n defined",
        )
        body: list[list[str]] = []
        sentences: list[str] = []
        for row in results.rows:
            if not row.ran:
                body.append([row.label] + [MISSING] * (len(header) - 1))
                continue
            bt = next((b for b in row.by_type if b.query_type == qtype), None)
            if bt is None:
                body.append([row.label] + [MISSING] * (len(header) - 1))
                continue
            body.append(
                [
                    row.label,
                    fmt_cell(bt.ndcg),
                    fmt_cell(bt.recall),
                    fmt_cell(bt.mrr),
                    fmt_fraction(bt.unjudged_fraction),
                    fmt_n(bt.ndcg),
                ]
            )
            if bt.vs_previous is not None:
                sentences.append(f"- **{row.label}** — {bt.vs_previous.sentence}")
        lines += [
            _table(header, body),
            "",
            type_denominator_note(results, qtype, n_queries),
            "",
        ]
        if sentences:
            lines += [*sentences, ""]
    return "\n".join(lines)


def render_sweep(results: EvalResults, plot_note: str) -> str:
    lines = [
        f"## 3. Quality vs latency (NDCG@{results.ndcg_k} against p95)",
        "",
        results.sweep_note,
        "",
    ]
    if not results.sweep:
        return "\n".join([*lines, "No sweep was run."])
    header = (
        "",
        "retrieve_k",
        "rerank_k",
        "final_k",
        f"NDCG@{results.ndcg_k}",
        "p95 ms",
        "p50 ms",
        "n defined",
        "Stage 1",
    )
    body: list[list[str]] = []
    for point in results.sweep:
        mark = f"{OPERATING_POINT_MARK} operating point" if point.is_operating_point else ""
        if not point.ran:
            body.append(
                [
                    mark,
                    str(point.retrieve_k),
                    str(point.rerank_k),
                    str(point.final_k),
                    f"{MISSING} (not run: {point.skip_reason})",
                    MISSING,
                    MISSING,
                    MISSING,
                    MISSING,
                ]
            )
            continue
        body.append(
            [
                mark,
                str(point.retrieve_k),
                str(point.rerank_k),
                str(point.final_k),
                fmt_cell(point.ndcg),
                fmt_latency(point.latency),
                fmt_float(point.latency.p50_ms if point.latency else None, 1),
                fmt_n(point.ndcg),
                point.parse.sentence if point.parse else MISSING,
            ]
        )
    lines += [
        _table(header, body),
        "",
        f"The **{OPERATING_POINT_MARK} operating point** row is the served system: "
        f"retrieve={results.operating_depths.retrieve} / "
        f"rerank={results.operating_depths.rerank} / "
        f"final={results.operating_depths.final}, the depths every document in this "
        "project quotes. Its NDCG is the one to cite for the system as served; the "
        "ablation table's full row is measured at the eval depth (see "
        '"How to read a cell").'
        + (
            ""
            if any(p.is_operating_point for p in results.sweep)
            else " **No grid point matched the operating point in this run**, so the "
            "served system's own NDCG was not measured here — add "
            f"{results.operating_depths.retrieve} to `candidates.sweep_retrieve` and "
            f"{results.operating_depths.rerank} to `candidates.sweep_rerank`."
        ),
        "",
        denominator_note(results),
        "",
        plot_note,
    ]
    return "\n".join(lines)


def render_failures(results: EvalResults) -> str:
    lines = [
        "## 4. Failure analysis",
        "",
        "The worst queries per type under "
        f"`{results.failure_reference_system or 'no system'}`, with what the system "
        "actually returned, what Stage 1 parsed, and where the good result was lost. "
        "The question each case answers is the one that decides what to fix: was the "
        "relevant place **never retrieved** (a Stage 2 or filter problem, which no "
        "reranker can repair) or **retrieved and then mis-ranked** (a Stage 3/4 "
        "problem)?",
        "",
        "These are the lowest-scoring queries of each type, which is not the same "
        "as bad ones: where a type is answered well, its worst query says so and "
        "no fault is invented for it.",
        "",
        denominator_note(results),
        "",
    ]
    if not results.failures:
        return "\n".join(
            [*lines, "No failure cases: no system ran, or no query has a defined NDCG."]
        )
    for case in results.failures:
        lines += [
            f"### [{case.query_type}] {case.query_id} — NDCG@{results.ndcg_k} = "
            f"{fmt_float(case.ndcg)}",
            "",
            f"> {case.query}",
            "",
            f"- **Lost at:** `{case.lost_stage}`",
            f"- **Diagnosis:** {case.diagnosis}",
            f"- **Parsed:** {case.parsed_summary}",
        ]
        if case.parse_latency_ms is not None:
            lines.append(f"- **Parse latency:** {case.parse_latency_ms:.1f} ms")
        lines += ["", f"Top {results.ndcg_k} returned:", ""]
        lines.append(
            _table(
                ("rank", "business", "business_id", "grade"),
                [
                    [
                        str(rank),
                        name,
                        bid,
                        str(grade) if judged else f"unjudged (scored {grade})",
                    ]
                    for rank, bid, name, grade, judged in case.top_results
                ]
                or [["—", "nothing returned", MISSING, MISSING]],
            )
        )
        lines += ["", "Where each known-relevant place ranked, by system:", ""]
        systems = list(case.relevant_traces[0].rank_by_system) if case.relevant_traces else []
        lines.append(
            _table(
                ("business", "grade", *systems),
                [
                    [
                        f"{trace.name} (`{trace.business_id}`)",
                        str(trace.grade),
                        *[
                            MISSING
                            if trace.rank_by_system[s] is None
                            else str(trace.rank_by_system[s])
                            for s in systems
                        ],
                    ]
                    for trace in case.relevant_traces
                ]
                or [["no relevant labels", MISSING]],
            )
        )
        lines.append("")
    lines += [
        f"`{MISSING}` in the per-system table means the place was not in that system's "
        "returned list at all.",
    ]
    return "\n".join(lines)


def render_warnings(results: EvalResults) -> str:
    if not results.warnings:
        return ""
    return "\n".join(["## Warnings from this run", ""] + [f"- {w}" for w in results.warnings])


def render_markdown(results: EvalResults, plot_note: str = "") -> str:
    """The whole report, ready to paste into the README."""
    sections = [
        render_header(results),
        render_ablation(results),
        render_by_type(results),
        render_sweep(results, plot_note),
        render_failures(results),
        render_warnings(results),
    ]
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# JSON
# ─────────────────────────────────────────────────────────────────────────────


def to_jsonable(obj: Any) -> Any:
    """Convert the result objects to plain JSON types.

    Hand-written rather than ``dataclasses.asdict`` alone so enums, paths and
    tuple-keyed oddities cannot slip through as ``repr`` strings — a JSON file
    that silently stores ``"QueryType.VIBE"`` is a file the README generator
    then has to parse by hand.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        # NaN/inf are not valid JSON and would be emitted as bare tokens that
        # most parsers reject. They should never occur; if one does, it must be
        # visible rather than written out as a number-shaped token.
        return None
    return obj


def results_to_json(results: EvalResults) -> str:
    return json.dumps(to_jsonable(results), indent=2, sort_keys=False) + "\n"


def write_sweep_csv(results: EvalResults, path: Path) -> Path:
    """The curve's data as CSV, so the figure can be redrawn (or drawn at all,
    when matplotlib is not installed) without rerunning the sweep."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "retrieve_k",
                "rerank_k",
                "final_k",
                "ran",
                "ndcg_mean",
                "ndcg_ci_low",
                "ndcg_ci_high",
                "n_queries_defined",
                "p50_ms",
                "p95_ms",
                "n_latency_samples",
                # Carried into the CSV as well as the table: a redrawn figure
                # that silently mixes a cold Stage 1 with fourteen warm ones is
                # the same defect one layer down.
                "is_operating_point",
                "parse_sources",
                "parse_mean_ms",
                "parse_p95_ms",
                "skip_reason",
            ]
        )
        for p in results.sweep:
            writer.writerow(
                [
                    p.retrieve_k,
                    p.rerank_k,
                    p.final_k,
                    int(p.ran),
                    "" if p.ndcg is None else p.ndcg.mean,
                    "" if p.ndcg is None else p.ndcg.low,
                    "" if p.ndcg is None else p.ndcg.high,
                    "" if p.ndcg is None else p.ndcg.n_defined,
                    "" if p.latency is None else p.latency.p50_ms,
                    "" if p.latency is None else p.latency.p95_ms,
                    0 if p.latency is None else p.latency.n_samples,
                    int(p.is_operating_point),
                    ""
                    if p.parse is None
                    else ";".join(f"{k}={v}" for k, v in sorted(p.parse.counts.items())),
                    "" if p.parse is None else p.parse.mean_ms,
                    "" if p.parse is None else p.parse.p95_ms,
                    p.skip_reason,
                ]
            )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PlotPoint:
    """One drawable point: both coordinates present, no optionals left."""

    retrieve_k: int
    rerank_k: int
    p95_ms: float
    mean: float
    low: float
    high: float
    n_defined: int
    confidence: float


def _plottable(results: EvalResults) -> list[PlotPoint]:
    """Sweep points that have both coordinates.

    Flattened into a small plain record rather than passed around as
    ``SweepPoint`` so the "this value exists" check happens once, here, instead
    of being re-asserted at every use in the drawing code below.
    """
    return [
        PlotPoint(
            retrieve_k=p.retrieve_k,
            rerank_k=p.rerank_k,
            p95_ms=p.latency.p95_ms,
            mean=p.ndcg.mean,
            low=p.ndcg.low if p.ndcg.low is not None else p.ndcg.mean,
            high=p.ndcg.high if p.ndcg.high is not None else p.ndcg.mean,
            n_defined=p.ndcg.n_defined,
            confidence=p.ndcg.confidence,
        )
        for p in results.sweep
        if p.ran
        and p.ndcg is not None
        and p.ndcg.mean is not None
        and p.latency is not None
        and p.latency.p95_ms is not None
    ]


def plot_latency_curve(results: EvalResults, path: Path) -> tuple[Path | None, str]:
    """Save the quality-vs-latency figure; return the path and a note for the report.

    matplotlib is imported here and nowhere else, and a missing install is a
    *note*, not a failure: the numbers are the deliverable, the figure is a
    convenience, and losing a whole eval run to a plotting dependency would be
    absurd. The CSV next to it always holds the same data.
    """
    points = _plottable(results)
    if not points:
        return None, ("No figure: no sweep point produced both an NDCG and a latency sample.")
    try:
        import matplotlib  # optional dependency, imported only to draw

        matplotlib.use("Agg")  # headless: this runs from a Makefile, not a notebook
        import matplotlib.pyplot as plt
    except ImportError:
        return None, (
            "No figure: matplotlib is not installed. The curve's data is in "
            f"`{path.with_suffix('.csv').name}` and the table above; install "
            "matplotlib and rerun `make eval` to render it."
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    by_retrieve: dict[int, list[PlotPoint]] = {}
    for p in points:
        by_retrieve.setdefault(p.retrieve_k, []).append(p)
    for retrieve_k in sorted(by_retrieve):
        series = sorted(by_retrieve[retrieve_k], key=lambda p: p.p95_ms)
        xs = [p.p95_ms for p in series]
        ys = [p.mean for p in series]
        # Asymmetric error bars: a percentile bootstrap interval is not
        # required to be symmetric about the mean, so it must not be drawn as
        # +/- one number.
        lows = [p.mean - p.low for p in series]
        highs = [p.high - p.mean for p in series]
        ax.errorbar(
            xs,
            ys,
            yerr=[lows, highs],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=f"retrieve_k = {retrieve_k}",
        )
        for p in series:
            ax.annotate(
                f"rerank {p.rerank_k}",
                (p.p95_ms, p.mean),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=7,
            )
    ax.set_xlabel("p95 end-to-end latency (ms)")
    ax.set_ylabel(f"NDCG@{results.ndcg_k}")
    ax.set_title(f"Quality vs latency · {results.coverage.n_queries_with_relevant} judged queries")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, (
        f"![Quality vs latency]({path.name}) — error bars are the "
        f"{points[0].confidence:.0%} bootstrap interval on NDCG@{results.ndcg_k} "
        f"over {points[0].n_defined} queries with a defined score."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Writing everything out
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReportPaths:
    directory: Path
    markdown: Path
    json: Path
    sweep_csv: Path
    plot: Path | None

    def all_paths(self) -> tuple[Path, ...]:
        paths = (self.markdown, self.json, self.sweep_csv)
        return paths + ((self.plot,) if self.plot else ())


def write_report(results: EvalResults, out_dir: Path) -> ReportPaths:
    """Write report.md, results.json, the sweep CSV and (if possible) the figure.

    Everything lands in ``<out_dir>/<config fingerprint>/`` so a figure in the
    README can always be traced back to the exact configuration that produced
    it (DECISIONS.md D008).
    """
    directory = out_dir / results.config_fingerprint
    directory.mkdir(parents=True, exist_ok=True)

    sweep_csv = write_sweep_csv(results, directory / "latency_curve.csv")
    plot_path, plot_note = plot_latency_curve(results, directory / "latency_curve.png")

    markdown = directory / "report.md"
    markdown.write_text(render_markdown(results, plot_note), encoding="utf-8")
    json_path = directory / "results.json"
    json_path.write_text(results_to_json(results), encoding="utf-8")

    return ReportPaths(
        directory=directory,
        markdown=markdown,
        json=json_path,
        sweep_csv=sweep_csv,
        plot=plot_path,
    )
