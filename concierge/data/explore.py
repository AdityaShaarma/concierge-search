"""Phase 1 step 2: the dataset report the owner needs in order to pick a metro.

    python -m concierge.data.explore --config configs/base.yaml

**What question this answers.** Scope is a real tradeoff, and it is not simply
"pick the biggest city". A bigger metro means more businesses, which means a
harder retrieval problem and a more impressive headline number. But a search
engine that answers *"quiet cafe with wifi, open past 10pm"* can only answer it
for businesses that actually carry `NoiseLevel`, `WiFi` and `hours`. Attribute
coverage in this dataset is sparse and uneven between cities, so a large metro
with thin coverage can leave fewer *answerable* queries than a smaller one with
rich coverage. Nothing in the raw file makes that visible, so this report makes
it visible: counts *and* coverage, side by side, per metro.

**What it refuses to do.** Pick the metro. That is the owner's call at the
Phase 1 checkpoint (D009), made on these numbers.

Everything printed here is counted from the file in `data.raw_dir`. Where a
quantity is not measured — review row counts without `--count-reviews`, for
instance — the report says so rather than substituting an estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.table import Table

from concierge.config import REPO_ROOT, Config, load_config
from concierge.data.load import (
    ALL_METROS,
    BUSINESS_FILE,
    MissingDatasetError,
    assign_metro,
    count_reviews_by_business,
    load_businesses,
    median_or_none,
    raw_path,
)

# `concierge.data.attributes` and `concierge.data.hours` own every rule about
# how this dataset spells its values, and this report must not reimplement
# them: two definitions of "has a NoiseLevel" would eventually disagree, and the
# one in the report is the one the owner would be reading.
#
# Both imports are guarded because those modules are written in parallel with
# this one. A guarded import means the Phase 1 checkpoint is not blocked on
# build order: the report still runs, with a cruder structural fallback, and it
# always prints *which* path produced its numbers — so no figure here has
# ambiguous provenance.
try:  # pragma: no cover - which branch runs depends on build order
    from concierge.data import attributes as _attributes
except ImportError:  # pragma: no cover
    _attributes = None  # type: ignore[assignment]

try:  # pragma: no cover
    from concierge.data import hours as _hours
except ImportError:  # pragma: no cover
    _hours = None  # type: ignore[assignment]


#: The constraints the gold set leans on hardest, and the columns of the
#: per-metro coverage table. All four are flat keys in the raw data, so the
#: shared parser and the structural fallback agree about them.
HEADLINE_ATTRIBUTES: Final[tuple[str, ...]] = (
    "NoiseLevel",
    "WiFi",
    "OutdoorSeating",
    "RestaurantsPriceRange2",
)

#: Only used when the attributes module is unavailable. Deliberately limited to
#: flat keys: Yelp nests `Ambience` and `BusinessParking` as stringified dicts,
#: and `concierge.data.attributes` flattens those to dotted leaves
#: ("Ambience.casual"). Listing a container key here would report 0% for an
#: attribute that is in fact populated — a plausible-looking wrong number, which
#: is worse than an admitted gap. With the real module in place the overall
#: table is driven by whatever keys it finds, nested leaves included.
FALLBACK_ATTRIBUTES: Final[tuple[str, ...]] = (
    *HEADLINE_ATTRIBUTES,
    "GoodForKids",
    "RestaurantsReservations",
    "RestaurantsTakeOut",
    "Alcohol",
    "BikeParking",
    "RestaurantsGoodForGroups",
)

#: Spellings that mean "no value" in the raw data. `"None"` appears as a
#: *string* (fixture b008), which a plain `key in attributes` check would count
#: as data and quietly inflate every coverage number here.
_MISSING_SPELLINGS: Final[frozenset[str]] = frozenset(
    {"", "none", "null", "u'none'", "'none'", "nan"}
)

UNASSIGNED: Final = "(unassigned)"

#: Rows in the printed overall-coverage table. The JSON artifact keeps them all;
#: a 40-row table in a terminal buries the four that matter.
MAX_ATTRIBUTE_ROWS: Final = 25

#: A business is "answerable" when a time constraint *and* at least one
#: amenity/vibe constraint can be evaluated against it at all. This is a
#: definition chosen for this report, not a dataset property, so it is printed
#: next to the number it produces.
ANSWERABLE_AMENITIES: Final[tuple[str, ...]] = ("NoiseLevel", "WiFi", "OutdoorSeating")


# ─────────────────────────────────────────────────────────────────────────────
# Hours: delegate to concierge.data.hours when it is importable and sane
# ─────────────────────────────────────────────────────────────────────────────


def _structural_hours_field(raw: Any) -> bool:
    return isinstance(raw, Mapping) and len(raw) > 0


def _structural_usable_hours(raw: Any) -> bool:
    """Crude stand-in for a real parse: any day whose value looks like a range."""
    if not isinstance(raw, Mapping):
        return False
    return any(isinstance(v, str) and "-" in v for v in raw.values())


_HoursPredicate = Callable[[Any], bool]


def _resolve_hours_backend() -> tuple[_HoursPredicate, _HoursPredicate, str]:
    """Pick the hours predicates, preferring the shared module.

    The shared module is probed with known-answer cases before it is trusted.
    That is not suspicion of a colleague's code: this module can be imported
    while that one is half-written, and a coverage table that is quietly wrong
    is the one failure mode this project cannot absorb. A failed probe
    downgrades to the structural check *and says so* in the report.
    """
    if _hours is not None:
        has_data = getattr(_hours, "has_hours_data", None)
        parse = getattr(_hours, "parse_hours", None)
        if callable(has_data) and callable(parse):
            try:
                ok = (
                    bool(has_data({"Monday": "7:0-20:0"}))
                    and not has_data(None)
                    and len(parse({"Monday": "7:0-20:0"})) == 1
                    and len(parse(None)) == 0
                )
            except Exception:  # pragma: no cover - a broken dependency
                ok = False
            if ok:
                return (
                    lambda raw: bool(has_data(raw)),
                    lambda raw: len(parse(raw)) > 0,
                    "concierge.data.hours",
                )
    return (
        _structural_hours_field,
        _structural_usable_hours,
        "structural fallback in explore.py (hours module unavailable)",
    )


_HAS_HOURS_FIELD, _HAS_USABLE_HOURS, HOURS_SOURCE = _resolve_hours_backend()


def has_hours_field(raw: Mapping[str, Any] | None) -> bool:
    """Whether the record carries any hours data at all, parseable or not."""
    return _HAS_HOURS_FIELD(raw)


def has_usable_hours(raw: Mapping[str, Any] | None) -> bool:
    """Whether the hours field parses into at least one real interval.

    Reported separately from `has_hours_field` because the gap between the two
    is a finding, not noise: those are businesses that *look* like they have
    hours and still cannot answer "open past 10pm".
    """
    return _HAS_USABLE_HOURS(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Attribute coverage: delegate to concierge.data.attributes when possible
# ─────────────────────────────────────────────────────────────────────────────


def raw_attribute_present(attributes: Mapping[str, Any] | None, key: str) -> bool:
    """Fallback presence test against a *raw* attribute dict.

    Structural only: it does not interpret `"u'free'"` vs `"'quiet'"` (the
    attributes module's job) and it cannot see nested leaves. Used when the
    shared module is unavailable, and kept public because the test suite pins
    the `"None"`-as-a-string quirk against it.
    """
    if not attributes:
        return False
    value = attributes.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _MISSING_SPELLINGS
    return True


def _resolve_parser() -> Callable[[Any], Mapping[str, Any]] | None:
    """The shared per-record attribute parser, probed before it is trusted."""
    parse = getattr(_attributes, "parse_attributes", None) if _attributes else None
    if not callable(parse):
        return None
    try:
        # Known answers: a stringified unicode literal normalizes to a value,
        # and a null attributes field yields no keys at all (types.py: an
        # absent key means "no data").
        if parse({"WiFi": "u'free'"}).get("WiFi") is None or parse(None):
            return None
    except Exception:  # pragma: no cover - a broken dependency
        return None
    # Bound to a typed name because `getattr` hands back `Any`, and an
    # unchecked `Any` crossing a module boundary is how type errors hide.
    parser: Callable[[Any], Mapping[str, Any]] = parse
    return parser


_PARSE_ATTRIBUTES: Final = _resolve_parser()


def attribute_present(record: Mapping[str, Any], key: str) -> bool:
    """Per-record attribute presence, via the shared parser when available."""
    raw = record.get("attributes")
    if _PARSE_ATTRIBUTES is not None:
        return _PARSE_ATTRIBUTES(raw).get(key) is not None
    return raw_attribute_present(raw, key)


def is_constraint_answerable(record: Mapping[str, Any]) -> bool:
    """Usable hours plus at least one amenity — see `ANSWERABLE_AMENITIES`."""
    return has_usable_hours(record.get("hours")) and any(
        attribute_present(record, key) for key in ANSWERABLE_AMENITIES
    )


@dataclass(frozen=True, slots=True)
class Coverage:
    """Per-attribute fill rates plus the provenance of those numbers.

    `complete` separates "ranked over every attribute key present in the data"
    (what the shared module reports) from "only the keys this file knows to ask
    about" (the fallback). The report prints that difference rather than
    presenting a partial list as the whole truth.
    """

    rates: dict[str, float]
    complete: bool
    source: str

    def headline(self, keys: Sequence[str] = HEADLINE_ATTRIBUTES) -> dict[str, float]:
        """Fill rates for the per-metro table columns.

        A key absent from `rates` is a genuine zero: `attributes.coverage`
        counts the keys it saw, so a key it never saw is present on no business.
        """
        return {key: self.rates.get(key, 0.0) for key in keys}


def _as_fill_rates(result: Any) -> dict[str, float] | None:
    """Accept a coverage result only if it is unambiguously a fill-rate map.

    Every value must be a fraction in [0, 1]. A mapping of raw *counts* is
    rejected rather than divided by a guessed denominator: silently
    reinterpreting another module's units is exactly how a wrong number ends up
    looking right.
    """
    if not isinstance(result, Mapping):
        return None
    out: dict[str, float] = {}
    for key, value in result.items():
        if not isinstance(key, str):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not 0.0 <= float(value) <= 1.0:
            return None
        out[key] = float(value)
    return out


def measure_coverage(records: Sequence[Mapping[str, Any]]) -> Coverage:
    """Fill rate per attribute key over `records`, with its provenance.

    Prefers `attributes.coverage(records)`, which reports *every* key it finds
    (nested leaves included, e.g. "Ambience.casual"). That is strictly more
    informative than any hardcoded list here, and it is the same function the
    failure analysis will quote later — one definition, one number.
    """
    if not records:
        # Zero over zero. Labelled, so it can never be read as a measured 0%.
        return Coverage({}, complete=False, source="no records")

    fn = getattr(_attributes, "coverage", None) if _attributes else None
    if callable(fn):
        # Two call shapes: the current interface takes just the records, but a
        # revision taking an explicit key list should also work.
        for call in (lambda: fn(records), lambda: fn(records, FALLBACK_ATTRIBUTES)):
            try:
                validated = _as_fill_rates(call())
            except Exception:  # a half-written dependency must not break the report
                continue
            if validated is not None:
                return Coverage(
                    validated, complete=True, source="concierge.data.attributes.coverage"
                )

    n = len(records)
    rates = {
        key: sum(1 for r in records if raw_attribute_present(r.get("attributes"), key)) / n
        for key in FALLBACK_ATTRIBUTES
    }
    return Coverage(
        rates,
        complete=False,
        source="structural fallback in explore.py (attributes module unavailable)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MetroStats:
    """One row of the metro table.

    Every count is over *open* businesses except `n_closed`, because closed
    businesses never enter the searchable index and would only dilute the
    coverage figures the scoping decision rests on.
    """

    metro: str
    n_businesses: int
    n_closed: int
    n_cities: int
    top_cities: tuple[tuple[str, int], ...]
    # Yelp's own review totals, from the `review_count` field.
    yelp_review_count_total: int
    yelp_review_count_median: float | None
    # Rows actually present in review.json. None means "not measured this run".
    review_rows_total: int | None
    review_rows_median: float | None
    n_with_hours_field: int
    n_with_usable_hours: int
    n_constraint_answerable: int
    attribute_fill: dict[str, float] = field(default_factory=dict)

    @property
    def hours_field_share(self) -> float:
        return self.n_with_hours_field / self.n_businesses if self.n_businesses else 0.0

    @property
    def hours_share(self) -> float:
        return self.n_with_usable_hours / self.n_businesses if self.n_businesses else 0.0

    @property
    def answerable_share(self) -> float:
        return self.n_constraint_answerable / self.n_businesses if self.n_businesses else 0.0


@dataclass(frozen=True, slots=True)
class ExploreReport:
    generated_at: str
    config_fingerprint: str
    source_file: str
    attribute_source: str
    attribute_fill_is_complete: bool
    hours_source: str
    n_businesses_raw: int
    n_open: int
    n_closed: int
    drop_closed: bool
    review_rows_measured: bool
    overall_attribute_fill: dict[str, float]
    n_with_hours_field: int
    n_with_usable_hours: int
    n_constraint_answerable: int
    metros: tuple[MetroStats, ...]
    unassigned_top_cities: tuple[tuple[str, int], ...]
    notes: tuple[str, ...]

    @property
    def unassigned(self) -> MetroStats | None:
        return next((m for m in self.metros if m.metro == UNASSIGNED), None)


def _metro_key(record: Mapping[str, Any]) -> str:
    return assign_metro(record.get("city"), record.get("state")) or UNASSIGNED


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    config_fingerprint: str,
    source_file: str,
    drop_closed: bool,
    review_rows: Mapping[str, int] | None = None,
    top_cities: int = 8,
) -> ExploreReport:
    """Build the report from raw business records, closed ones included.

    Pure: takes records, returns a value, touches no files. That is what makes
    it testable against `tests/fixtures/` without the 5 GB download.

    `records` must be the *unfiltered* load (`include_closed=True`), because
    "how much of this metro is permanently closed" is one of the things the
    owner is deciding on. The searchable statistics are computed from the open
    subset inside this function.
    """
    open_records = [r for r in records if r.get("is_open") == 1]
    closed_by_metro: Counter[str] = Counter(_metro_key(r) for r in records if r.get("is_open") != 1)

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in open_records:
        grouped[_metro_key(record)].append(record)

    overall = measure_coverage(open_records)

    metros: list[MetroStats] = []
    for metro, group in grouped.items():
        cities = Counter(
            f"{(r.get('city') or '?').strip()}, {(r.get('state') or '?').strip()}" for r in group
        )
        yelp_counts = [int(r.get("review_count") or 0) for r in group]
        rows = (
            [int(review_rows.get(str(r.get("business_id")), 0)) for r in group]
            if review_rows is not None
            else None
        )
        metros.append(
            MetroStats(
                metro=metro,
                n_businesses=len(group),
                n_closed=closed_by_metro.get(metro, 0),
                n_cities=len(cities),
                top_cities=tuple(cities.most_common(top_cities)),
                yelp_review_count_total=sum(yelp_counts),
                yelp_review_count_median=median_or_none(yelp_counts),
                review_rows_total=sum(rows) if rows is not None else None,
                review_rows_median=median_or_none(rows) if rows is not None else None,
                n_with_hours_field=sum(1 for r in group if has_hours_field(r.get("hours"))),
                n_with_usable_hours=sum(1 for r in group if has_usable_hours(r.get("hours"))),
                n_constraint_answerable=sum(1 for r in group if is_constraint_answerable(r)),
                # Coverage is measured *within* the metro: that it varies by
                # city is the whole reason this report exists.
                attribute_fill=measure_coverage(group).headline(),
            )
        )

    # Largest first: the scoping decision starts from size, then checks coverage.
    metros.sort(key=lambda m: (-m.n_businesses, m.metro))
    unassigned = next((m for m in metros if m.metro == UNASSIGNED), None)

    return ExploreReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config_fingerprint=config_fingerprint,
        source_file=source_file,
        attribute_source=overall.source,
        attribute_fill_is_complete=overall.complete,
        hours_source=HOURS_SOURCE,
        n_businesses_raw=len(records),
        n_open=len(open_records),
        n_closed=len(records) - len(open_records),
        drop_closed=drop_closed,
        review_rows_measured=review_rows is not None,
        # Highest fill rate first, ties alphabetical, so the artifact is
        # byte-stable across runs and diffable between releases.
        overall_attribute_fill=dict(sorted(overall.rates.items(), key=lambda kv: (-kv[1], kv[0]))),
        n_with_hours_field=sum(1 for r in open_records if has_hours_field(r.get("hours"))),
        n_with_usable_hours=sum(1 for r in open_records if has_usable_hours(r.get("hours"))),
        n_constraint_answerable=sum(1 for r in open_records if is_constraint_answerable(r)),
        metros=tuple(metros),
        unassigned_top_cities=unassigned.top_cities if unassigned else (),
        notes=_notes(review_rows is not None, overall, unassigned, len(open_records)),
    )


def _notes(
    review_rows_measured: bool,
    coverage: Coverage,
    unassigned: MetroStats | None,
    n_open: int,
) -> tuple[str, ...]:
    """The caveats that must travel with the numbers rather than be remembered."""
    notes = [
        "Metro membership is OUR definition, not a dataset column: an explicit city "
        "list plus allowed states per metro (see concierge/data/load.py). The "
        f"'{UNASSIGNED}' row is the honesty check — those businesses are in cities no "
        "definition claims. If that row is large, extend the city lists using the "
        "per-city table below and rerun.",
        "'Yelp reviews' columns come from the review_count field in business.json, "
        "which is Yelp's own total. review.json contains only a subset of those "
        "reviews, so the two differ and are reported separately.",
        f"Attribute fill rates were computed by: {coverage.source}.",
        f"Hours figures were computed by: {HOURS_SOURCE}.",
        "'hours field' counts businesses with any hours data; 'usable hours' counts "
        "those whose hours parse into at least one interval. The gap between them is "
        "data we cannot use.",
        "'Answerable' = usable hours AND at least one of "
        f"{', '.join(ANSWERABLE_AMENITIES)}. A definition chosen for this report, not a "
        "dataset property: it approximates 'a constraint query could be evaluated "
        "against this business at all'.",
    ]
    if not coverage.complete:
        notes.append(
            "The attribute table lists only the keys this report asks about, not every "
            "key in the data, and it cannot see nested attributes such as "
            "Ambience.casual. Rerun once concierge/data/attributes.py is importable to "
            "get the complete ranking."
        )
    if not review_rows_measured:
        notes.append(
            "Review row counts from review.json were NOT measured in this run "
            "(pass --count-reviews; it streams the ~5GB file)."
        )
    if unassigned is not None and n_open:
        share = unassigned.n_businesses / n_open
        notes.append(
            f"{unassigned.n_businesses} open businesses ({share:.1%}) matched no metro definition."
        )
    return tuple(notes)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | None) -> str:
    """Never print a stand-in zero for something that was not measured."""
    if value is None:
        return "not measured"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _attribute_label(key: str) -> str:
    return "price" if key == "RestaurantsPriceRange2" else key


def _fill_scope_label(report: ExploreReport) -> str:
    return (
        "every attribute key found in the data"
        if report.attribute_fill_is_complete
        else "only the keys this report asks about — nested keys invisible"
    )


def render(report: ExploreReport, console: Console) -> None:
    """Print the report: three tables, then the tradeoff, then the caveats."""
    console.print()
    console.rule("[bold]Dataset exploration — Phase 1 step 2")
    closed_share = (
        f" ({report.n_closed / report.n_businesses_raw:.1%} excluded from the index)"
        if report.n_businesses_raw
        else ""
    )
    console.print(
        f"source: {report.source_file}\n"
        f"config fingerprint: {report.config_fingerprint}    generated: {report.generated_at}\n"
        f"businesses in file: {report.n_businesses_raw:,}   open: {report.n_open:,}   "
        f"permanently closed: {report.n_closed:,}{closed_share}\n"
        f"attribute fill via: {report.attribute_source}\n"
        f"hours via: {report.hours_source}"
    )

    scope = Table(title="Per-metro scope (open businesses only)", title_justify="left")
    scope.add_column("metro")
    scope.add_column("open", justify="right")
    scope.add_column("closed", justify="right")
    scope.add_column("cities", justify="right")
    scope.add_column("Yelp reviews\n(median/biz)", justify="right")
    scope.add_column("review.json rows\n(median/biz)", justify="right")
    for metro in report.metros:
        scope.add_row(
            metro.metro,
            f"{metro.n_businesses:,}",
            f"{metro.n_closed:,}",
            f"{metro.n_cities:,}",
            _num(metro.yelp_review_count_median),
            _num(metro.review_rows_median),
        )
    console.print(scope)

    coverage = Table(
        title="Constraint coverage — what makes a constraint query answerable",
        title_justify="left",
    )
    coverage.add_column("metro")
    coverage.add_column("open", justify="right")
    coverage.add_column("hours\nfield", justify="right")
    coverage.add_column("usable\nhours", justify="right")
    for key in HEADLINE_ATTRIBUTES:
        coverage.add_column(_attribute_label(key), justify="right")
    coverage.add_column("answerable", justify="right")
    for metro in report.metros:
        coverage.add_row(
            metro.metro,
            f"{metro.n_businesses:,}",
            _pct(metro.hours_field_share),
            _pct(metro.hours_share),
            *(_pct(metro.attribute_fill.get(key)) for key in HEADLINE_ATTRIBUTES),
            f"{metro.n_constraint_answerable:,} ({_pct(metro.answerable_share)})",
        )
    console.print(coverage)

    shown = list(report.overall_attribute_fill.items())[:MAX_ATTRIBUTE_ROWS]
    overall = Table(
        title=f"Attribute fill rate, all open businesses ({_fill_scope_label(report)})",
        title_justify="left",
    )
    overall.add_column("attribute")
    overall.add_column("fill rate", justify="right")
    for key, value in shown:
        overall.add_row(key, _pct(value))
    console.print(overall)
    hidden = len(report.overall_attribute_fill) - len(shown)
    if hidden > 0:
        console.print(f"[dim]({hidden} further attributes in the JSON artifact)[/dim]")

    if report.unassigned_top_cities:
        cities = Table(
            title=f"Top cities in '{UNASSIGNED}' — extend the metro definitions from here",
            title_justify="left",
        )
        cities.add_column("city, state")
        cities.add_column("open businesses", justify="right")
        for name, count in report.unassigned_top_cities:
            cities.add_row(name, f"{count:,}")
        console.print(cities)

    console.print(_tradeoff_text(report))
    console.print("\n[bold]Caveats that travel with these numbers[/bold]")
    for note in report.notes:
        console.print(f"  • {note}")
    console.print(
        "\n[bold]Next step (owner's call):[/bold] set data.metro in configs/base.yaml. "
        "Bigger is not automatically better — a metro only helps the project if its "
        "businesses carry the attributes the queries ask about."
    )


def _tradeoff_text(report: ExploreReport) -> str:
    """State the size-vs-coverage tradeoff using only computed values."""
    real = [m for m in report.metros if m.metro != UNASSIGNED and m.n_businesses > 0]
    if not real:
        return (
            "\n[bold]The tradeoff[/bold]\n  No metro definition matched any business — "
            "every record is unassigned, so the city lists in load.py do not fit this "
            "file. Fix those before choosing a scope."
        )
    biggest = max(real, key=lambda m: m.n_businesses)
    richest = max(real, key=lambda m: m.answerable_share)
    most_answerable = max(real, key=lambda m: m.n_constraint_answerable)
    lines = [
        "\n[bold]The tradeoff[/bold]",
        f"  Largest metro: {biggest.metro} — {biggest.n_businesses:,} open businesses, "
        f"{biggest.n_constraint_answerable:,} of them answerable "
        f"({biggest.answerable_share:.1%}).",
        f"  Best coverage: {richest.metro} — {richest.answerable_share:.1%} answerable "
        f"of {richest.n_businesses:,} open businesses.",
        f"  Most answerable businesses in absolute terms: {most_answerable.metro} "
        f"({most_answerable.n_constraint_answerable:,}).",
    ]
    if biggest.metro != most_answerable.metro:
        lines.append(
            "  Note: the largest metro is NOT the one with the most answerable "
            "businesses. Retrieval difficulty scales with the first number, but the "
            "gold set's constraint queries can only be satisfied from the second."
        )
    else:
        lines.append(
            "  Here the largest metro is also the one with the most answerable "
            "businesses, so size and coverage do not conflict in this file."
        )
    return "\n".join(lines)


def to_markdown(report: ExploreReport) -> str:
    """Markdown twin of the printed report, for pasting into notes or the README."""
    out: list[str] = [
        "# Dataset exploration — Phase 1 step 2",
        "",
        f"- source: `{report.source_file}`",
        f"- generated: {report.generated_at}",
        f"- config fingerprint: `{report.config_fingerprint}`",
        f"- businesses in file: {report.n_businesses_raw:,} "
        f"(open {report.n_open:,}, permanently closed {report.n_closed:,})",
        f"- `data.drop_closed` = {report.drop_closed}",
        f"- attribute fill computed by: {report.attribute_source}",
        f"- hours computed by: {report.hours_source}",
        "",
        "## Per-metro scope (open businesses)",
        "",
        "| metro | open | closed | cities | Yelp reviews median/biz | review.json rows median/biz |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in report.metros:
        out.append(
            f"| {m.metro} | {m.n_businesses:,} | {m.n_closed:,} | {m.n_cities:,} | "
            f"{_num(m.yelp_review_count_median)} | {_num(m.review_rows_median)} |"
        )

    headers = " | ".join(_attribute_label(k) for k in HEADLINE_ATTRIBUTES)
    out += [
        "",
        "## Constraint coverage",
        "",
        f"| metro | open | hours field | usable hours | {headers} | answerable | answerable % |",
        "|---|---:|---:|---:|" + "---:|" * len(HEADLINE_ATTRIBUTES) + "---:|---:|",
    ]
    for m in report.metros:
        cells = " | ".join(_pct(m.attribute_fill.get(k)) for k in HEADLINE_ATTRIBUTES)
        out.append(
            f"| {m.metro} | {m.n_businesses:,} | {_pct(m.hours_field_share)} | "
            f"{_pct(m.hours_share)} | {cells} | {m.n_constraint_answerable:,} | "
            f"{_pct(m.answerable_share)} |"
        )

    out += [
        "",
        f"## Attribute fill rate, all open businesses ({_fill_scope_label(report)})",
        "",
        "| attribute | fill rate |",
        "|---|---:|",
    ]
    for key, value in report.overall_attribute_fill.items():
        out.append(f"| {key} | {_pct(value)} |")

    if report.unassigned_top_cities:
        out += [
            "",
            f"## Top cities in `{UNASSIGNED}`",
            "",
            "| city, state | open businesses |",
            "|---|---:|",
        ]
        out += [f"| {name} | {count:,} |" for name, count in report.unassigned_top_cities]

    out += ["", "## Caveats", ""]
    out += [f"- {note}" for note in report.notes]
    out += [
        "",
        "## Next step",
        "",
        "Set `data.metro` in `configs/base.yaml`. This is the owner's decision "
        "(DECISIONS.md D009): a bigger metro is a harder, more impressive retrieval "
        "problem, but constraint queries are only answerable where the attribute and "
        "hours data exist.",
        "",
    ]
    return "\n".join(out)


def write_artifacts(report: ExploreReport, out_dir: Path) -> tuple[Path, Path]:
    """Write the JSON and markdown twins.

    Namespaced by config fingerprint (D008) so a report can always be traced
    back to the settings that produced it.
    """
    target = out_dir / report.config_fingerprint
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "dataset_report.json"
    md_path = target / "dataset_report.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=False) + "\n")
    md_path.write_text(to_markdown(report))
    return json_path, md_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m concierge.data.explore",
        description="Per-metro counts and attribute coverage, to choose data.metro.",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--out-dir",
        default="artifacts/explore",
        help="Artifact root; the report lands in <out-dir>/<config fingerprint>/.",
    )
    parser.add_argument("--top-cities", type=int, default=12)
    parser.add_argument(
        "--count-reviews",
        action="store_true",
        help="Also count rows per business in review.json. Streams the ~5GB file: "
        "slow, but the only way to know how much review text actually exists.",
    )
    parser.add_argument("--no-artifacts", action="store_true", help="Print only.")
    return parser


def run(cfg: Config, args: argparse.Namespace, console: Console) -> int:
    """The CLI body, separate from argument parsing so tests can drive it."""
    try:
        # include_closed=True: the closed share is part of the decision, and
        # this is the only full pass over business.json we make.
        records = load_businesses(cfg, ALL_METROS, include_closed=True)
        review_rows = None
        if args.count_reviews:
            console.print("[dim]streaming review.json to count rows per business…[/dim]")
            review_rows = count_reviews_by_business(cfg)
    except MissingDatasetError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    if not records:
        console.print(
            "[bold red]business.json parsed but contained no records.[/bold red] "
            "Check that the file is the newline-delimited JSON from the Yelp download."
        )
        return 2

    report = build_report(
        records,
        config_fingerprint=cfg.fingerprint(),
        source_file=str(raw_path(cfg, BUSINESS_FILE)),
        drop_closed=cfg.data.drop_closed,
        review_rows=review_rows,
        top_cities=args.top_cities,
    )
    render(report, console)

    if not args.no_artifacts:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        json_path, md_path = write_artifacts(report, out_dir)
        console.print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    return run(load_config(args.config), args, Console())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
