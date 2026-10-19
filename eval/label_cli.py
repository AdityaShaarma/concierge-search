"""The relevance labeling tool: `make label`.

The owner will spend many hours inside this program, and the labels it collects
are the ground truth under every number in the project. Two consequences shaped
every decision here:

**Speed is a quality feature, not a nicety.** A tool that needs two keystrokes
and a page of scrolling per judgement produces rushed labels in hour three, and
rushed labels do not announce themselves — they quietly flatten every ablation
gap and make an honest comparison impossible. So: one keystroke per judgement,
no Enter, no mouse, a fixed-position screen that does not reflow, and everything
needed to decide on one screen.

**You cannot judge what you cannot see.** Relevance here is not vibes. "Open
past 10pm with outdoor seating" is a factual question about hours and
attributes, and the labeler can only answer it if the hours and attributes are
in front of them. The display *is* the product of this file.

Three smaller decisions worth defending:

* **Grades are written to disk after every keystroke** (flushed and fsynced).
  An hour of labeling is irreplaceable human work; losing it to a crash is
  unacceptable in a way that losing a minute of compute never is.
* **The previous grade is never shown in ``--retest`` mode.** The whole point of
  re-labeling a subset weeks later is to measure self-agreement. Showing the old
  grade would measure the labeler's willingness to agree with a number on a
  screen instead.
* **Candidates are served in a system-independent shuffled order** (see
  ``pooling.build_pool``). Judging best-first would let the system under test
  set the labeler's expectations, which is a subtle way of marking your own
  homework.

What this tool needs to run, and where each piece comes from:

* ``eval/gold_queries.csv`` — written by the owner (never by a model).
* a **runs file**: ``{"system": {"query_id": ["business_id", ...]}}`` as JSON,
  produced by the retrieval baselines. Pooling needs at least one system.
* **business records**: newline-delimited JSON in ``business.json``'s own shape,
  so this runs directly against the raw download with no build step.
* optionally **reviews/tips** in their raw shape, for the evidence snippet.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Final, NoReturn

import typer
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from concierge.config import REPO_ROOT, Config, load_config
from concierge.data.attributes import parse_attributes
from concierge.data.documents import attribute_phrases
from concierge.data.hours import DAYS, MINUTES_PER_DAY, parse_hours
from concierge.types import Business, GoldQuery, HoursInterval
from eval import pooling
from eval.metrics import (
    ORDINAL_GRADES,
    BusinessId,
    Grade,
    QueryId,
    cohens_kappa,
    confusion_matrix,
)
from eval.pooling import PoolPair, QrelsWriter

# Raw-mode single-keystroke input is POSIX-only. On a platform without termios
# we fall back to line input (type the key, press Enter) rather than refusing to
# run — slower, but the tool still works.
try:  # pragma: no cover - platform dependent
    import termios
    import tty

    _RAW_INPUT_AVAILABLE = True
except ImportError:  # pragma: no cover - Windows
    _RAW_INPUT_AVAILABLE = False

app = typer.Typer(
    add_completion=False,
    help="Grade pooled search results for the gold queries (owner-only task).",
)

console = Console(highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# The grading scale — kept on screen at all times
# ─────────────────────────────────────────────────────────────────────────────

# The exact wording is duplicated in eval/README.md on purpose: the definitions
# must be identical in the tool and in the document the owner re-reads weeks
# later, because label consistency across sessions is the thing the test-retest
# check measures. If you edit one, edit the other.
GRADE_SCALE: Final[tuple[tuple[int, str, str], ...]] = (
    (
        2,
        "highly relevant",
        "I would send a friend here for this exact request",
    ),
    (
        1,
        "partially relevant",
        "right kind of place, but part of the request is unmet or unknown",
    ),
    (
        0,
        "irrelevant",
        "wrong kind of place, or it breaks a stated hard constraint",
    ),
)

GRADE_COLOR: Final[dict[int, str]] = {2: "green", 1: "yellow", 0: "red"}

ACTION_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("s", "skip (undecided — comes back next session)"),
    ("u", "undo the last judgement"),
    ("q", "save and quit"),
)

# How much review text to show. Long enough to carry evidence ("open until 2am
# which saved me after a late shift"), short enough that the screen never
# scrolls — a scrolling screen is what makes a labeling tool slow.
SNIPPET_CHARS: Final = 260
N_SNIPPETS: Final = 2

DEFAULT_RETEST_SIZE: Final = 30
DEFAULT_RETEST_FILENAME: Final = "qrels_retest.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Reading the inputs
# ─────────────────────────────────────────────────────────────────────────────


def _iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    """Stream newline-delimited JSON, skipping blank lines.

    Streaming rather than ``json.load`` because the same code path has to work
    on ``review.json``, which is about 5 GB. Nothing is kept except the records
    that match a pooled business id.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def business_from_raw(record: Mapping[str, Any]) -> Business:
    """Build a :class:`Business` from one raw ``business.json`` record.

    Uses the project's own parsers (``parse_attributes``, ``parse_hours``) so the
    labeler sees attributes and hours interpreted *exactly* as the search system
    interprets them. If the attribute parser is wrong about ``"u'free'"``, the
    labeling screen is wrong in the same way and the bug is visible here rather
    than hiding behind a metric.

    Categories are split inline (one comma-separated string, as the dataset
    stores them) to keep this tool's startup free of the DuckDB/Arrow import
    that ``concierge.data.load`` pulls in; the loader owns bulk reads, this owns
    one record at a time.
    """
    raw_categories = record.get("categories") or ""
    categories = tuple(part.strip() for part in raw_categories.split(",") if part.strip())
    return Business(
        business_id=str(record["business_id"]),
        name=str(record.get("name") or ""),
        city=str(record.get("city") or ""),
        state=str(record.get("state") or ""),
        latitude=_as_float(record.get("latitude")),
        longitude=_as_float(record.get("longitude")),
        stars=float(record.get("stars") or 0.0),
        review_count=int(record.get("review_count") or 0),
        categories=categories,
        attributes=parse_attributes(record.get("attributes")),
        hours=parse_hours(record.get("hours")),
        is_open=bool(record.get("is_open")),
    )


def _as_float(value: Any) -> float | None:
    """None, not 0.0, for a missing coordinate: (0, 0) is a real place."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_businesses(path: Path, keep: set[BusinessId]) -> dict[BusinessId, Business]:
    """Read only the pooled businesses out of a business NDJSON file."""
    out: dict[BusinessId, Business] = {}
    for record in _iter_ndjson(path):
        business_id = record.get("business_id")
        if business_id in keep and business_id not in out:
            out[str(business_id)] = business_from_raw(record)
            if len(out) == len(keep):
                break  # everything we need; no reason to read the rest
    return out


def truncate_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Cut to ``limit`` characters at a word boundary, with an ellipsis.

    Mid-word truncation reads as a rendering bug and costs the labeler a beat to
    parse; at hundreds of judgements those beats are the difference between one
    session and two.
    """
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def load_snippets(
    path: Path,
    keep: set[BusinessId],
    *,
    per_business: int = N_SNIPPETS,
    rank_key: str = "useful",
) -> dict[BusinessId, list[str]]:
    """Collect a few short text snippets per pooled business.

    Ranked by the dataset's own ``useful`` vote count (``compliment_count`` for
    tips) rather than by recency or at random: the most-upvoted review is the
    one other humans found most informative, which is exactly the property a
    labeler needs from a two-line summary. This mirrors the document builder's
    ``useful_then_recent`` selection rule, so the evidence on screen resembles
    the evidence the models read.
    """
    best: dict[BusinessId, list[tuple[float, str]]] = {}
    for record in _iter_ndjson(path):
        business_id = record.get("business_id")
        if business_id not in keep:
            continue
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        score = float(record.get(rank_key) or 0.0)
        bucket = best.setdefault(str(business_id), [])
        bucket.append((score, text))
        # Keep only the best few per business so a 5 GB file does not become a
        # 5 GB dict.
        if len(bucket) > per_business * 4:
            bucket.sort(key=lambda pair: -pair[0])
            del bucket[per_business:]
    out: dict[BusinessId, list[str]] = {}
    for business_id, bucket in best.items():
        bucket.sort(key=lambda pair: -pair[0])
        out[business_id] = [truncate_snippet(text) for _, text in bucket[:per_business]]
    return out


def load_runs(path: Path) -> dict[str, dict[QueryId, list[BusinessId]]]:
    """Read the runs JSON: ``{system: {query_id: [business_id, ...]}}``.

    Validated structurally with a message naming the offending key, because the
    file is produced by a different part of the pipeline and a silent
    misinterpretation here would mean pooling the wrong thing — the kind of bug
    that only shows up as a strange metric weeks later.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"{path} must be a non-empty JSON object "
            f'{{"system_name": {{"query_id": ["business_id", ...]}}}}'
        )
    runs: dict[str, dict[QueryId, list[BusinessId]]] = {}
    for system, run in data.items():
        if not isinstance(run, dict):
            raise ValueError(f"{path}: run for system {system!r} is not an object")
        parsed: dict[QueryId, list[BusinessId]] = {}
        for query_id, ranked in run.items():
            if not isinstance(ranked, list) or any(not isinstance(b, str) for b in ranked):
                raise ValueError(
                    f"{path}: {system!r}/{query_id!r} must be a list of business id strings"
                )
            parsed[str(query_id)] = list(ranked)
        runs[str(system)] = parsed
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Formatting the evidence
# ─────────────────────────────────────────────────────────────────────────────


def format_clock(minute_of_day: int) -> str:
    """Minutes from midnight as ``HH:MM``; 1440 prints as ``midnight``."""
    if minute_of_day >= MINUTES_PER_DAY:
        return "midnight"
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def format_interval(interval: HoursInterval) -> str:
    """One opening interval in words.

    Overnight intervals are labeled "(next day)" explicitly. A bare
    "17:00-02:00" invites the reader to see a five-hour gap in the middle of the
    afternoon, and "open past 10pm" queries are a large share of the gold set —
    this is the single most misreadable field on the screen.
    """
    if interval.start_min == 0 and interval.end_min >= MINUTES_PER_DAY:
        return "open 24 hours"
    end = format_clock(interval.end_min)
    if interval.crosses_midnight:
        return f"{format_clock(interval.start_min)}-{end} (next day)"
    return f"{format_clock(interval.start_min)}-{end}"


def format_hours(business: Business) -> list[tuple[str, str]]:
    """``(day, text)`` rows for the weekly hours table.

    Distinguishes the two kinds of "not open", which the ranking code also keeps
    apart (``ConstraintStatus.UNKNOWN`` vs ``VIOLATED``): a business with hours
    data but no Monday entry is **closed Monday**, while a business with no hours
    field at all tells us **nothing**. A labeler who cannot see that difference
    will grade the second group as failures and bake a wrong penalty into the
    ground truth.
    """
    if not business.hours:
        return [("—", "no hours data in the dataset")]
    rows: list[tuple[str, str]] = []
    for day in DAYS:
        intervals = business.hours.get(day, ())
        text = ", ".join(format_interval(i) for i in intervals) if intervals else "closed"
        rows.append((day[:3], text))
    return rows


def grade_legend() -> Panel:
    """The scale, always on screen, always in the same place.

    Permanently visible rather than behind a help key: drift in what "1" means
    between session one and session four is invisible in the data and fatal to
    it. Re-reading the definition has to cost nothing.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column(style="bold")
    table.add_column()
    for grade, name, gloss in GRADE_SCALE:
        table.add_row(str(grade), Text(name, style=GRADE_COLOR[grade]), gloss)
    for key, gloss in ACTION_KEYS:
        table.add_row(key, "", gloss)
    return Panel(table, title="press one key", title_align="left", border_style="dim")


def query_panel(query: GoldQuery, position: int, total_for_query: int) -> Panel:
    """The request being judged. Shown above every candidate, never scrolled off."""
    header = Text()
    header.append(query.query, style="bold cyan")
    header.append(f"\n{query.query_id} · type: {query.query_type.value}", style="dim")
    header.append(f" · candidate {position} of {total_for_query} for this query", style="dim")
    return Panel(header, title="the request", title_align="left", border_style="cyan")


def candidate_panel(
    business: Business,
    snippets: Sequence[str],
) -> Panel:
    """Everything needed to judge one business, on one screen.

    Field order follows how a human actually decides: what kind of place is it
    (name, categories), then the hard facts a query can contradict (attributes,
    hours), then the soft evidence (reviews). Rating and review count come last
    of the facts deliberately — they are quality signals, not relevance
    signals, and a 4.9-star place that is loud is still wrong for "somewhere
    quiet".
    """
    body = Table.grid(padding=(0, 2))
    body.add_column(justify="right", style="dim", width=12)
    body.add_column(overflow="fold")

    title = Text(business.name or "(no name)", style="bold")
    location = ", ".join(part for part in (business.city, business.state) if part)
    if location:
        title.append(f"  {location}", style="dim")

    body.add_row("categories", ", ".join(business.categories) or "—")

    phrases = attribute_phrases(business.attributes)
    body.add_row(
        "attributes",
        ", ".join(phrases) if phrases else Text("no attribute data", style="dim italic"),
    )

    hours_table = Table.grid(padding=(0, 1))
    hours_table.add_column(style="dim")
    hours_table.add_column()
    for day, text in format_hours(business):
        hours_table.add_row(day, text)
    body.add_row("hours", hours_table)

    body.add_row("rating", f"{business.stars:g} stars · {business.review_count} reviews")

    if snippets:
        quotes = Text()
        for i, snippet in enumerate(snippets):
            if i:
                quotes.append("\n")
            quotes.append(f'"{snippet}"', style="italic")
        body.add_row("reviews", quotes)
    else:
        body.add_row("reviews", Text("no review or tip text on file", style="dim italic"))

    return Panel(
        Group(title, "", body),
        title="candidate",
        title_align="left",
        border_style="white",
    )


def format_duration(seconds: float | None) -> str:
    """``12m 30s``, or ``—`` when unmeasured. Never a guessed number."""
    if seconds is None:
        return "—"
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LabelSession:
    """Cursor, counters, and undo history for one sitting.

    Separated from both the rendering and the file writing so the control flow
    (what does 'u' do after a skip?) is unit-testable without a terminal. The
    interesting invariant is that *undo* must reverse exactly the last action,
    whether that action wrote a row or not.
    """

    pairs: tuple[PoolPair, ...]
    writer: QrelsWriter
    historical_pace_s: float | None = None
    index: int = 0
    n_graded: int = 0
    n_skipped: int = 0
    # "grade" or "skip", most recent last. Only this session's actions.
    _history: list[str] = field(default_factory=list)
    # Seconds spent on each completed judgement, for this session's own ETA.
    _durations: list[float] = field(default_factory=list)
    _started_at: float = field(default_factory=time.monotonic)
    _item_started_at: float = field(default_factory=time.monotonic)

    # ── position ─────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.pairs)

    @property
    def done(self) -> bool:
        return self.index >= self.total

    def current(self) -> PoolPair | None:
        return None if self.done else self.pairs[self.index]

    def position_in_query(self) -> tuple[int, int]:
        """``(position, total)`` of the current pair within its query.

        Shown so the labeler knows whether they are two candidates into a query
        or nineteen — it paces the work and flags a query whose pool is
        unexpectedly huge.
        """
        pair = self.current()
        if pair is None:
            return (0, 0)
        same = [p for p in self.pairs if p.query_id == pair.query_id]
        return (same.index(pair) + 1, len(same))

    # ── actions ──────────────────────────────────────────────────────────────

    def grade(self, relevance: Grade) -> None:
        """Record a grade: written to disk before this returns."""
        pair = self.current()
        if pair is None:
            raise RuntimeError("nothing left to grade")
        self.writer.append(pair.query_id, pair.business_id, relevance)
        self._durations.append(time.monotonic() - self._item_started_at)
        self.n_graded += 1
        self._history.append("grade")
        self._advance()

    def skip(self) -> None:
        """Leave a pair unjudged.

        Nothing is written, so the pair reappears in the next session's pool —
        skipping is "I cannot decide yet", not "irrelevant". Grading a genuinely
        undecidable pair 0 to make it go away would be the worst possible
        outcome, so skipping has to be the cheaper option.
        """
        if self.current() is None:
            raise RuntimeError("nothing left to skip")
        self.n_skipped += 1
        self._history.append("skip")
        self._advance()

    def undo(self) -> bool:
        """Reverse the last action of this session. False if there is none."""
        if not self._history:
            return False
        action = self._history.pop()
        if action == "grade":
            if self.writer.undo() is None:  # pragma: no cover - kept in lockstep
                self._history.append(action)
                return False
            self.n_graded -= 1
            if self._durations:
                self._durations.pop()
        else:
            self.n_skipped -= 1
        self.index -= 1
        self._item_started_at = time.monotonic()
        return True

    def _advance(self) -> None:
        self.index += 1
        self._item_started_at = time.monotonic()

    # ── progress ─────────────────────────────────────────────────────────────

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started_at

    @property
    def pace_s(self) -> float | None:
        """Seconds per judgement: this session's own median, else history.

        This session first, because pace changes with fatigue and with how
        familiar the queries are, and the live number is the one that predicts
        the next hour. Falls back to the pace measured from past sessions'
        timestamps, and to None — displayed as "—" — when neither exists.
        Never a constant: an invented pace produces an invented ETA.
        """
        if len(self._durations) >= pooling.MIN_INTERVALS_FOR_PACE:
            ordered = sorted(self._durations)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        return self.historical_pace_s

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.index)

    @property
    def eta_s(self) -> float | None:
        pace = self.pace_s
        return None if pace is None else self.remaining * pace

    def progress_line(self) -> Text:
        line = Text()
        line.append(f"{self.index}/{self.total} served", style="bold")
        line.append(f"  ·  {self.n_graded} graded")
        if self.n_skipped:
            line.append(f"  ·  {self.n_skipped} skipped", style="yellow")
        line.append(f"  ·  elapsed {format_duration(self.elapsed_s)}")
        line.append(f"  ·  ETA {format_duration(self.eta_s)}")
        if self.pace_s is None:
            line.append("  (pace unmeasured)", style="dim")
        return line


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────────────

KeyReader = Callable[[], str]


def read_key() -> str:
    """Read one keypress, lowercased, without waiting for Enter.

    Single-keystroke input is deliberate: requiring Enter doubles the keystrokes
    per judgement and, more importantly, moves attention away from the candidate
    once per pair. How many judgements a session actually is depends on
    inter-system overlap in the pool, which is not derivable here —
    ``eval.pooling.build_pool`` returns it as ``PoolStats`` once a real pool
    exists. (Named precisely because the earlier version of this docstring
    quoted a session size nobody had measured.)

    Falls back to line input when stdin is not a terminal (a pipe, a test, CI)
    or when the platform has no ``termios``.
    """
    if not _RAW_INPUT_AVAILABLE or not sys.stdin.isatty():  # pragma: no cover - tty dependent
        line = sys.stdin.readline()
        return line.strip()[:1].lower() if line.strip() else "q"
    fd = sys.stdin.fileno()  # pragma: no cover - requires a real terminal
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    if char in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D: raw mode swallows the signal
        raise KeyboardInterrupt
    return char.lower()


# ─────────────────────────────────────────────────────────────────────────────
# The loop
# ─────────────────────────────────────────────────────────────────────────────


def render(
    session: LabelSession,
    query: GoldQuery,
    business: Business,
    snippets: Sequence[str],
    *,
    message: str = "",
) -> RenderableType:
    """Assemble the whole screen. Pure, so it can be snapshot-tested."""
    position, total_for_query = session.position_in_query()
    parts: list[RenderableType] = [
        session.progress_line(),
        query_panel(query, position, total_for_query),
        candidate_panel(business, snippets),
        grade_legend(),
    ]
    if message:
        parts.append(Text(message, style="magenta"))
    return Group(*parts)


def run_session(
    session: LabelSession,
    queries: Mapping[QueryId, GoldQuery],
    businesses: Mapping[BusinessId, Business],
    snippets: Mapping[BusinessId, Sequence[str]],
    *,
    key_reader: KeyReader = read_key,
    out: Console | None = None,
) -> LabelSession:
    """Serve candidates until the pool is exhausted or the owner quits.

    ``key_reader`` is injected so the loop can be driven by a scripted list of
    keys in a test; the production reader is :func:`read_key`.
    """
    screen = out or console
    message = ""
    while not session.done:
        pair = session.current()
        assert pair is not None  # loop condition guarantees it
        business = businesses.get(pair.business_id)
        if business is None:
            # A pooled id with no business record means the runs file and the
            # business file disagree. Skipping silently would hide a real data
            # bug, so it is counted as a skip and reported at the end.
            session.skip()
            message = f"no business record for {pair.business_id} — skipped (data mismatch)"
            continue
        screen.clear()
        screen.print(
            render(
                session,
                queries[pair.query_id],
                business,
                snippets.get(pair.business_id, ()),
                message=message,
            )
        )
        message = ""
        try:
            key = key_reader()
        except KeyboardInterrupt:
            screen.print("\ninterrupted — everything graded so far is saved.")
            return session
        if key in {"0", "1", "2"}:
            session.grade(int(key))
        elif key == "s":
            session.skip()
        elif key == "u":
            message = "undone" if session.undo() else "nothing from this session to undo"
        elif key == "q":
            return session
        else:
            message = f"unknown key {key!r} — use 0/1/2, s, u, q"
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────


def report_agreement(original_path: Path, retest_path: Path, out: Console | None = None) -> int:
    """Print test-retest self-agreement between two label files.

    Returns the number of pairs compared (0 when there is nothing to compare),
    so a caller can tell "they agreed perfectly" from "there was no overlap".
    """
    screen = out or console
    first = pooling.read_judgements(original_path)
    second = pooling.read_judgements(retest_path)
    grades_a, grades_b, keys = pooling.agreement_pairs(first, second)
    if not keys:
        screen.print(
            f"No pairs graded in both {original_path} and {retest_path}. "
            f"Run `--retest` first, ideally at least two weeks after the original labels."
        )
        return 0

    exact = sum(1 for a, b in zip(grades_a, grades_b, strict=True) if a == b)
    screen.print(f"Pairs graded twice: {len(keys)}")
    screen.print(f"Exact agreement:    {exact}/{len(keys)} ({exact / len(keys):.1%})")
    for weights in ("quadratic", "none"):
        # Both are printed because quadratic weighting is the more flattering
        # number on a 3-point ordinal scale and must never be quoted alone.
        # ORDINAL_GRADES rather than a re-spelled (0, 1, 2): the weighting is
        # defined over the project's grading scheme, so widening that scheme
        # must widen this call too, not leave it silently behind.
        result = cohens_kappa(grades_a, grades_b, weights=weights, categories=ORDINAL_GRADES)
        label = "weighted (quadratic)" if weights == "quadratic" else "unweighted"
        if result.kappa is None:
            screen.print(f"Cohen's kappa, {label}: undefined — {result.note}")
        else:
            screen.print(f"Cohen's kappa, {label}: {result.kappa:.3f}")

    table = Table(title="confusion matrix (rows = first pass, columns = retest)")
    table.add_column("")
    for grade in ORDINAL_GRADES:
        table.add_column(str(grade), justify="right")
    # Declared, not derived: a retest sample where neither pass used grade 1
    # must still print a 3x3 matrix with an empty middle row and column,
    # because "nobody said 1" is the finding.
    matrix = confusion_matrix(grades_a, grades_b, categories=ORDINAL_GRADES)
    for grade, row in zip(ORDINAL_GRADES, matrix, strict=True):
        table.add_row(str(grade), *[str(v) for v in row])
    screen.print(table)

    gaps = _relabel_gaps_days(first, second, keys)
    if gaps:
        screen.print(
            f"Days between first and second label: median {median(gaps):.0f} "
            f"(min {min(gaps):.0f}, max {max(gaps):.0f})"
        )
    else:
        screen.print("Days between labels: unmeasured (timestamps missing or unparseable)")
    return len(keys)


def _relabel_gaps_days(
    first: Sequence[pooling.Judgement],
    second: Sequence[pooling.Judgement],
    keys: Sequence[tuple[QueryId, BusinessId]],
) -> list[float]:
    """Days between the two labels for each re-graded pair.

    Reported because self-agreement measured an hour later is mostly memory;
    measured weeks later it is consistency. The gap is the number that tells a
    reader which one they are looking at, so it has to be measured, not claimed.
    """
    a = {j.key: j.labeled_at_dt for j in first}
    b = {j.key: j.labeled_at_dt for j in second}
    gaps: list[float] = []
    for key in keys:
        start, end = a.get(key), b.get(key)
        if start is not None and end is not None:
            gaps.append((end - start).total_seconds() / 86400.0)
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _resolve(path: Path) -> Path:
    """Resolve a project-relative path against the repo root, like the config does."""
    return path if path.is_absolute() else REPO_ROOT / path


def _fail(message: str) -> NoReturn:
    """Exit with an actionable message rather than a traceback.

    Every failure here is a missing input the owner can fix in one step, so the
    message names the file and the fix. A stack trace would say the same thing
    in a form that invites debugging instead of downloading a file.
    """
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=2)


@app.command()
def label(
    # Typer's Annotated form rather than defaults-as-calls: the default value
    # stays a plain value, which keeps the function callable from Python (and
    # from a test) exactly as written.
    config: Annotated[Path, typer.Option("--config", help="Config YAML.")] = Path(
        "configs/base.yaml"
    ),
    runs: Annotated[
        Path | None,
        typer.Option(
            "--runs",
            help='Runs JSON: {"system": {"query_id": ["business_id", ...]}}. '
            "Defaults to <eval.out_dir>/runs.json.",
        ),
    ] = None,
    businesses: Annotated[
        Path | None,
        typer.Option(
            "--businesses",
            help="business.json (NDJSON). Defaults to <raw_dir>/business.json.",
        ),
    ] = None,
    reviews: Annotated[
        Path | None,
        typer.Option("--reviews", help="review.json (NDJSON) for evidence snippets. Optional."),
    ] = None,
    tips: Annotated[
        Path | None, typer.Option("--tips", help="tip.json (NDJSON). Optional.")
    ] = None,
    labeler: Annotated[
        str, typer.Option("--labeler", help="Recorded in every qrels row.")
    ] = "owner",
    out: Annotated[Path | None, typer.Option("--out", help="qrels CSV to append to.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Stop after N candidates.")] = None,
    all_queries: Annotated[
        bool,
        typer.Option(
            "--all-queries", help="Pool every gold query instead of the random label subset."
        ),
    ] = False,
    stats: Annotated[bool, typer.Option("--stats", help="Print pool statistics and exit.")] = False,
    retest: Annotated[
        bool,
        typer.Option("--retest", help="Re-serve already-labeled pairs, previous grades hidden."),
    ] = False,
    retest_size: Annotated[
        int, typer.Option("--retest-size", help="Queries to re-label in --retest mode.")
    ] = DEFAULT_RETEST_SIZE,
    agreement: Annotated[
        bool, typer.Option("--agreement", help="Report test-retest self-agreement and exit.")
    ] = False,
) -> None:
    """Grade pooled results, one keystroke per judgement.

    Every judgement is appended to the qrels CSV immediately, so quitting (or
    crashing) never loses work and the next run resumes where this one stopped.
    """
    cfg: Config = load_config(config)
    seed = cfg.seed
    qrels_path = _resolve(out or cfg.evaluation.qrels)
    retest_path = qrels_path.with_name(DEFAULT_RETEST_FILENAME)

    if agreement:
        report_agreement(qrels_path, retest_path)
        return

    # ── the owner's queries ──────────────────────────────────────────────────
    gold_path = _resolve(cfg.evaluation.gold_queries)
    try:
        gold = pooling.load_gold_queries(gold_path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    if not gold.queries:
        _fail(
            f"{gold_path} contains no real queries"
            + (
                f" ({gold.n_examples_skipped} example rows skipped)"
                if gold.n_examples_skipped
                else ""
            )
            + ". The gold queries are written by you, by hand — see eval/README.md."
        )
    if gold.n_examples_skipped:
        console.print(
            f"[yellow]note:[/yellow] skipped {gold.n_examples_skipped} example row(s) in "
            f"{gold_path.name}; delete them once you no longer need the format."
        )

    qrels = pooling.load_qrels(qrels_path)
    history = pooling.read_judgements(qrels_path)
    pace = pooling.measure_pace(history)

    # ── what to serve ────────────────────────────────────────────────────────
    if retest:
        pool = pooling.build_retest_pool(
            qrels,
            pooling.load_qrels(retest_path),
            size=retest_size,
            seed=seed,
        )
        writer_path = retest_path
        console.print(
            Panel(
                "[bold]retest mode[/bold] — these pairs were graded before and the previous "
                "grade is deliberately not shown. Judge each one fresh; disagreeing with "
                "yourself is information, not a mistake.",
                border_style="magenta",
            )
        )
    else:
        runs_path = _resolve(runs or cfg.evaluation.out_dir / "runs.json")
        if not runs_path.exists():
            _fail(
                f"no runs file at {runs_path}. Pooling needs at least one system's results: "
                f'write {{"bm25": {{"q001": ["business_id", ...]}}}} there, or pass --runs.'
            )
        try:
            loaded_runs = load_runs(runs_path)
        except (ValueError, json.JSONDecodeError) as exc:
            _fail(str(exc))
        chosen = (
            gold.queries
            if all_queries
            else pooling.select_label_subset(gold.queries, cfg.evaluation.label_subset_size, seed)
        )
        pool = pooling.build_pool(
            loaded_runs,
            cfg.evaluation.pool_depth,
            qrels=qrels,
            query_ids=[q.query_id for q in chosen],
            shuffle_seed=seed,
        )
        scope = "all gold queries" if all_queries else f"random subset of {len(chosen)}"
        console.print(f"Labeling scope: {scope} of {len(gold.queries)} gold queries (seed {seed})")
        writer_path = qrels_path

    for line in pooling.format_pool_report(pool, seconds_per_pair=pace):
        console.print(line)
    if stats:
        return
    if not pool.pairs:
        console.print(
            "[green]Nothing new to judge — every pooled pair already has a grade.[/green]"
        )
        return

    # ── the evidence ─────────────────────────────────────────────────────────
    pairs = pool.pairs[:limit] if limit else pool.pairs
    needed = {p.business_id for p in pairs}
    businesses_path = _resolve(businesses or cfg.data.raw_dir / "business.json")
    if not businesses_path.exists():
        _fail(
            f"no business records at {businesses_path}. Download the Yelp Open Dataset into "
            f"{cfg.data.raw_dir} (see README), or pass --businesses."
        )
    console.print(f"Reading {len(needed)} business records from {businesses_path.name}…")
    records = load_businesses(businesses_path, needed)
    if missing := needed - records.keys():
        console.print(
            f"[yellow]warning:[/yellow] {len(missing)} pooled business id(s) are not in "
            f"{businesses_path.name}; they will be skipped. This means the runs file and the "
            f"business file were built from different data."
        )

    snippets: dict[BusinessId, list[str]] = {}
    for path_opt, default_name, rank_key in (
        (reviews, "review.json", "useful"),
        (tips, "tip.json", "compliment_count"),
    ):
        path = _resolve(path_opt or cfg.data.raw_dir / default_name)
        if not path.exists():
            continue
        console.print(f"Reading evidence snippets from {path.name} (this streams the file)…")
        for business_id, texts in load_snippets(path, needed, rank_key=rank_key).items():
            snippets.setdefault(business_id, []).extend(texts)
    if not snippets:
        console.print(
            "[yellow]note:[/yellow] no review or tip text available, so candidates will be "
            "judged on name, categories, attributes and hours alone."
        )

    # ── grade ────────────────────────────────────────────────────────────────
    by_id = gold.by_id
    with QrelsWriter(writer_path, labeler) as writer:
        session = LabelSession(pairs=pairs, writer=writer, historical_pace_s=pace)
        run_session(session, by_id, records, snippets)

    console.print()
    console.print(
        f"[green]Saved {session.n_graded} judgement(s)[/green] to {writer_path} "
        f"in {format_duration(session.elapsed_s)}."
    )
    if session.n_skipped:
        console.print(f"{session.n_skipped} skipped — they stay in the pool for next time.")
    if session.remaining:
        console.print(f"{session.remaining} pair(s) left. Run the same command again to resume.")


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
