"""Streaming loaders for the raw Yelp Open Dataset files.

Three files matter here: ``business.json`` (~150K lines), ``tip.json``, and
``review.json`` — the last of which is roughly 5 GB of newline-delimited JSON.

**Why streaming is not optional.** A naive ``json.loads`` per line, or
``pandas.read_json``, builds a Python object for *every field of every review*:
text, user id, votes, timestamps, for ~7M rows. That is tens of gigabytes of
Python objects for a pipeline that only ever needs ``business_id``, ``text``,
``useful`` and ``date``, and only for the businesses of the one metro in scope
(how many that is comes out of ``make explore``; it is not asserted here).
So every read here goes through DuckDB with:

  * **explicit column projection** — ``read_json(..., columns={...})`` extracts
    only the listed keys and skips the rest of each line, so the unprojected
    fields never become Python objects at all. How much that saves is the byte
    share of the fields we skip, and ``text`` — the field most likely to
    dominate a review record — is one of the columns we *do* project, so the
    saving is not the kind of thing to guess at from the field count. Nothing
    here measures it; the mechanism is the reason for the call, not a number;
  * **a semi-join against the in-scope business ids** — the filter runs inside
    DuckDB while it streams the file, so out-of-scope rows are dropped before
    they cross into Python;
  * **Arrow record batches** — results arrive in chunks, so peak memory is one
    batch, not one result set.

The practical effect is that the only thing proportional to the 5 GB file is
*time*, never memory. (Those costs are not measured yet; this module makes the
measurement possible, it does not claim a number.)

``business.json`` is small enough to materialize as a list of dicts, and the
rest of the pipeline genuinely wants all of it at once (documents, indexes), so
``load_businesses`` returns a list. The asymmetry is deliberate, not an
oversight.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final, Literal

import duckdb
import pyarrow as pa

from concierge.config import REPO_ROOT, Config

# ─────────────────────────────────────────────────────────────────────────────
# File layout
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_FILE: Final = "business.json"
REVIEW_FILE: Final = "review.json"
TIP_FILE: Final = "tip.json"

#: Passed to ``load_businesses(metro=...)`` to mean "every metro, ignore the
#: config". Needed because ``metro=None`` already means "defer to the config",
#: and exploration has to scan the whole dataset *before* a metro is chosen.
ALL_METROS: Final = "*"

# Column projections. Declared explicitly rather than letting DuckDB sniff the
# schema, for three reasons: sniffing a 5 GB file costs a scan; an inferred
# schema can drift between dataset releases and silently change types; and only
# an explicit list actually skips the unwanted keys.
#
# ``attributes`` and ``hours`` are read as JSON (i.e. as text) rather than as
# DuckDB STRUCTs on purpose: their keys vary per business, so any struct schema
# would be a lossy guess. They are handed to Python as raw dicts and parsed by
# ``concierge.data.attributes`` / ``concierge.data.hours``, which own that mess.
BUSINESS_COLUMNS: Final[dict[str, str]] = {
    "business_id": "VARCHAR",
    "name": "VARCHAR",
    "address": "VARCHAR",
    "city": "VARCHAR",
    "state": "VARCHAR",
    "postal_code": "VARCHAR",
    # Carried from the very first build so geo never forces a reindex (D003).
    "latitude": "DOUBLE",
    "longitude": "DOUBLE",
    "stars": "DOUBLE",
    "review_count": "BIGINT",
    "is_open": "BIGINT",
    "attributes": "JSON",
    "categories": "VARCHAR",
    "hours": "JSON",
}

REVIEW_COLUMNS: Final[dict[str, str]] = {
    "review_id": "VARCHAR",
    "business_id": "VARCHAR",
    "stars": "DOUBLE",
    "useful": "BIGINT",
    "text": "VARCHAR",
    "date": "VARCHAR",
}

TIP_COLUMNS: Final[dict[str, str]] = {
    "business_id": "VARCHAR",
    "text": "VARCHAR",
    "date": "VARCHAR",
    "compliment_count": "BIGINT",
}

# Columns whose values arrive as JSON text and must be decoded in Python.
_JSON_COLUMNS: Final = frozenset({"attributes", "hours"})

#: How many rows per Arrow batch. Large enough that per-batch overhead is
#: negligible, small enough that one batch of review text is a few MB.
DEFAULT_BATCH_ROWS: Final = 50_000

ReviewOrder = Literal["useful_then_recent", "recent", "random"]


class MissingDatasetError(FileNotFoundError):
    """Raised when a raw Yelp file is absent.

    Its own exception type so callers (and the explore CLI) can catch exactly
    this and print the download instructions instead of a stack trace.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Metro scoping
#
# This is the judgement call in this module, so it is spelled out rather than
# buried. Yelp's "11 metropolitan areas" is a sentence in the dataset
# documentation, not a column: every record carries only `city` and `state`.
# A metro therefore has to be *defined* by us, and the definition changes the
# number of searchable businesses, so it must be visible and testable.
#
# Options considered:
#
#   (a) state only — wrong in both directions. One state can hold two of the
#       dataset's metros, and real metros cross state lines (the Philadelphia
#       area spills into NJ and DE, Saint Louis into IL).
#   (b) exact match on the core city string — defensible and simple, but drops
#       the suburbs that make up much of a metro (a Phoenix-area search that
#       excludes Tempe and Scottsdale is not a Phoenix-area search), and it is
#       brittle against the dirty city strings in the data.
#   (c) a radius around a city centroid using the lat/lon we already carry —
#       immune to city-string noise, but needs a centroid and a radius per
#       metro, and the radius is an unprincipled knob that silently includes
#       whatever else is nearby.
#   (d) chosen: an explicit, named list of city names plus the set of states
#       the metro may appear in, matched on a *normalized* city string.
#
# (d) wins because it is legible and checkable: the definition is data you can
# read, `explore` prints the per-city counts the lists were built from, and it
# also prints how many businesses no definition claims. That "unassigned" count
# is the honesty check — if it is large, these lists are wrong, and the report
# says so instead of hiding it.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MetroDefinition:
    """A metro area, defined as (normalized city names) x (allowed states).

    The state set is not redundant with the city list: city names repeat across
    states (Nashville exists in several), and requiring both makes a definition
    safe to extend with a short suburb name.
    """

    name: str
    states: frozenset[str]
    cities: frozenset[str]
    note: str = ""

    def matches(self, city: str | None, state: str | None) -> bool:
        # An empty state set means "any state" — that is the uncurated
        # single-city fallback from `resolve_metro`, which has no business
        # guessing which state the owner meant.
        if self.states and normalize_state(state) not in self.states:
            return False
        return normalize_city(city) in self.cities


_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[.,;]+$")
# "St. Louis" / "St Louis" / "Saint Louis" are the same city typed three ways,
# and all three spellings occur in the wild. Collapse them before matching.
_SAINT_PREFIX = re.compile(r"^(st|ste)\.?\s+")


def normalize_city(city: str | None) -> str:
    """Canonical form of a city string, for matching only (never for display).

    City strings in this dataset carry case noise, stray punctuation, accents,
    and the Saint/St. ambiguity. Matching on the raw string would quietly drop
    businesses, which looks like missing data rather than a lookup bug.
    """
    if not city:
        return ""
    # NFKD + dropping combining marks folds "Montréal" onto "montreal".
    folded = unicodedata.normalize("NFKD", city)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    out = _WHITESPACE.sub(" ", folded.strip().lower())
    out = _TRAILING_PUNCT.sub("", out)
    return _SAINT_PREFIX.sub(lambda m: "saint " if m.group(1) == "st" else "sainte ", out)


def normalize_state(state: str | None) -> str:
    return (state or "").strip().upper()


def _metro(
    name: str, states: Iterable[str], cities: Iterable[str], note: str = ""
) -> MetroDefinition:
    return MetroDefinition(
        name=name,
        states=frozenset(normalize_state(s) for s in states),
        cities=frozenset(normalize_city(c) for c in cities),
        note=note,
    )


# Seed definitions, NOT a claim about what is in the download.
#
# These cover the metros named in recent releases of the dataset, each as a core
# city plus nearby municipalities. The city lists are ordinary geography; which
# of them actually appear, and with how many businesses, is answered by
# `python -m concierge.data.explore`, whose per-city table exists precisely so
# the owner can correct these lists against the real file. Release-to-release
# the covered metros have changed, so treat an empty metro as "this release does
# not have it", not as a bug.
METRO_DEFINITIONS: Final[dict[str, MetroDefinition]] = {
    normalize_city(d.name): d
    for d in (
        _metro(
            "Philadelphia",
            ["PA", "NJ", "DE", "MD"],
            [
                "Philadelphia",
                "Cherry Hill",
                "Camden",
                "Bensalem",
                "King of Prussia",
                "Norristown",
                "Upper Darby",
                "Media",
                "Conshohocken",
                "Ardmore",
                "Wilmington",
                "Collingswood",
                "Haddonfield",
                "Voorhees",
                "Marlton",
            ],
            note="Genuinely multi-state; the state set is wide on purpose.",
        ),
        _metro(
            "Tucson",
            ["AZ"],
            ["Tucson", "Oro Valley", "Marana", "Vail", "Sahuarita", "Green Valley"],
        ),
        _metro(
            "Tampa",
            ["FL"],
            [
                "Tampa",
                "Saint Petersburg",
                "Clearwater",
                "Brandon",
                "Largo",
                "Riverview",
                "Pinellas Park",
                "Palm Harbor",
                "Dunedin",
                "Temple Terrace",
                "Lutz",
            ],
        ),
        _metro(
            "Indianapolis",
            ["IN"],
            [
                "Indianapolis",
                "Carmel",
                "Fishers",
                "Greenwood",
                "Noblesville",
                "Zionsville",
                "Avon",
                "Brownsburg",
                "Plainfield",
                "Westfield",
                "Beech Grove",
            ],
        ),
        _metro(
            "Nashville",
            ["TN"],
            [
                "Nashville",
                "Franklin",
                "Brentwood",
                "Hendersonville",
                "Murfreesboro",
                "Smyrna",
                "Mount Juliet",
                "Antioch",
                "Madison",
                "Goodlettsville",
            ],
        ),
        _metro(
            "New Orleans",
            ["LA"],
            [
                "New Orleans",
                "Metairie",
                "Kenner",
                "Gretna",
                "Harahan",
                "Marrero",
                "Chalmette",
                "Westwego",
                "River Ridge",
            ],
        ),
        _metro(
            "Reno",
            ["NV"],
            ["Reno", "Sparks"],
            note="Carson City is deliberately excluded: it is a separate metro.",
        ),
        _metro(
            "Edmonton",
            ["AB"],
            [
                "Edmonton",
                "Saint Albert",
                "Sherwood Park",
                "Spruce Grove",
                "Leduc",
                "Fort Saskatchewan",
                "Stony Plain",
                "Beaumont",
                "Nisku",
            ],
        ),
        _metro(
            "Saint Louis",
            ["MO", "IL"],
            [
                "Saint Louis",
                "Clayton",
                "Kirkwood",
                "Chesterfield",
                "Maplewood",
                "Webster Groves",
                "Saint Charles",
                "Ballwin",
                "Brentwood",
                "University City",
                "Belleville",
                "Edwardsville",
                "O'Fallon",
                "Florissant",
            ],
            note="Crosses into Illinois; both states are allowed.",
        ),
        _metro(
            "Boise",
            ["ID"],
            ["Boise", "Meridian", "Nampa", "Eagle", "Garden City", "Caldwell", "Kuna"],
        ),
        _metro(
            "Santa Barbara",
            ["CA"],
            ["Santa Barbara", "Goleta", "Carpinteria", "Montecito", "Isla Vista", "Summerland"],
        ),
    )
}


def resolve_metro(metro: str) -> MetroDefinition:
    """Look up a metro definition, falling back to "treat it as one city".

    The fallback is what makes the module usable before the metro question is
    settled, and what lets the tests scope to a synthetic city. It is explicit
    rather than an error because a single-city scope is a perfectly coherent
    (if narrower) thing to ask for — but it accepts *any* state, so it cannot
    silently masquerade as one of the curated definitions above.
    """
    key = normalize_city(metro)
    if key in METRO_DEFINITIONS:
        return METRO_DEFINITIONS[key]
    return MetroDefinition(
        name=metro,
        states=frozenset(),  # empty == "any state"; see _sql_metro_filter
        cities=frozenset({key}),
        note="Not a curated metro definition: matched as a single city name.",
    )


def assign_metro(city: str | None, state: str | None) -> str | None:
    """Which curated metro a business belongs to, or None if no definition claims it."""
    for definition in METRO_DEFINITIONS.values():
        if definition.matches(city, state):
            return definition.name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Paths and connection
# ─────────────────────────────────────────────────────────────────────────────


def raw_path(cfg: Config, filename: str) -> Path:
    """Absolute path to a raw file.

    Relative config paths resolve against the repo root (the same rule
    ``load_config`` uses), so ``make explore`` behaves identically whatever
    directory it is invoked from.
    """
    base = cfg.data.raw_dir
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / filename


_DOWNLOAD_HINT = (
    "Download the Yelp Open Dataset from https://www.yelp.com/dataset "
    "(choose the JSON download), then unzip it so the files land directly in "
    "{raw_dir}/ — expected: business.json, review.json, tip.json. "
    "The data is licensed for personal/educational use and is never committed to git."
)


def require_raw_file(cfg: Config, filename: str) -> Path:
    """Return the path to a raw file, or raise an actionable error."""
    path = raw_path(cfg, filename)
    if path.is_file():
        return path
    raw_dir = path.parent
    if not raw_dir.is_dir():
        detail = f"the directory {raw_dir} does not exist"
    elif not any(raw_dir.iterdir()):
        detail = f"{raw_dir} is empty"
    else:
        present = sorted(p.name for p in raw_dir.iterdir() if p.is_file())[:10]
        detail = (
            f"{raw_dir} exists but has no {filename} (found: {', '.join(present) or 'nothing'})"
        )
    raise MissingDatasetError(
        f"Raw dataset file not found: {path}\n  {detail}\n  "
        + _DOWNLOAD_HINT.format(raw_dir=raw_dir)
    )


def connect() -> duckdb.DuckDBPyConnection:
    """A fresh in-memory DuckDB connection.

    In-memory is right here: nothing is persisted, DuckDB spills to disk by
    itself if an aggregate outgrows RAM, and a fresh connection per call keeps
    the temporary scope table from leaking between loads.
    """
    return duckdb.connect()


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────


def _read_json_sql(alias: str, columns: dict[str, str]) -> str:
    """The FROM-clause fragment that streams one file with a fixed projection.

    The path and the column map are passed as query parameters ($1/$2) so no
    filename or identifier is ever string-formatted into SQL.
    """
    select = ", ".join(f"{alias}.{name}" for name in columns)
    return f"SELECT {select} FROM read_json($1, format='newline_delimited', columns=$2) AS {alias}"


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    """Turn JSON-text columns back into Python dicts, leaving NULL as None.

    A NULL ``attributes`` (b005 in the fixtures) must stay ``None``: "no
    attribute data at all" is a distinct state from "an empty attribute dict",
    and the three-valued constraint logic downstream depends on the difference.
    """
    for key in _JSON_COLUMNS:
        if key in row and isinstance(row[key], str):
            row[key] = json.loads(row[key])
    return row


def _stream_rows(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any],
    batch_rows: int,
) -> Iterator[dict[str, Any]]:
    """Execute a query and yield dicts one batch at a time.

    Never calls ``fetchall()``: the whole point of this module is that a result
    set the size of review.json is never resident at once.
    """
    result = con.execute(sql, params)
    # to_arrow_reader is the current API; fetch_record_batch is its pre-1.3 name.
    reader = getattr(result, "to_arrow_reader", None) or result.fetch_record_batch
    for batch in reader(batch_rows):
        for row in batch.to_pylist():
            yield _decode(row)


def _sql_metro_filter(definition: MetroDefinition) -> tuple[str, list[Any]]:
    """WHERE fragment implementing a metro definition inside DuckDB.

    Normalization is reimplemented in SQL (lower/trim/strip-punctuation, and
    the Saint/St. fold) rather than filtering in Python, so that out-of-scope
    rows are discarded during the scan. The Python and SQL versions must agree;
    ``test_load.py`` checks them against each other on the fixtures.
    """
    city_expr = (
        "regexp_replace("
        "  regexp_replace(lower(trim({col})), '[.,;]+$', ''),"
        "  '^(st|ste)\\.?\\s+', 'saint ')"
    ).format(col="b.city")
    clauses = [f"{city_expr} IN (SELECT UNNEST($3::VARCHAR[]))"]
    params: list[Any] = [sorted(definition.cities)]
    if definition.states:
        clauses.append("upper(trim(b.state)) IN (SELECT UNNEST($4::VARCHAR[]))")
        params.append(sorted(definition.states))
    return " AND ".join(clauses), params


def _resolve_scope(cfg: Config, metro: str | None) -> MetroDefinition | None:
    """Decide the geographic scope for a load.

    ``metro=None`` defers to ``cfg.data.metro`` (which is null in the committed
    config until the owner decides — D009), and a null config metro means "no
    filter". ``metro=ALL_METROS`` forces no filter regardless of the config,
    which is what exploration needs.
    """
    if metro == ALL_METROS:
        return None
    chosen = metro if metro is not None else cfg.data.metro
    return resolve_metro(chosen) if chosen else None


def load_businesses(
    cfg: Config,
    metro: str | None = None,
    *,
    include_closed: bool | None = None,
) -> list[dict[str, Any]]:
    """Raw business records, metro-scoped and (by default) open-only.

    Returns a list because ``business.json`` is ~150K rows and every consumer
    (document builder, BM25 index, FAISS index) needs all of them at once.

    ``include_closed`` defaults to the inverse of ``cfg.data.drop_closed``:
    permanently closed businesses (``is_open == 0``) must not reach the
    searchable index, because a perfect answer to "quiet café with wifi" that
    shut down two years ago is a wrong answer. Exploration passes
    ``include_closed=True`` on purpose — how much of the raw file is closed is
    itself a number the owner wants.
    """
    path = require_raw_file(cfg, BUSINESS_FILE)
    keep_closed = (not cfg.data.drop_closed) if include_closed is None else include_closed
    definition = _resolve_scope(cfg, metro)

    sql = _read_json_sql("b", BUSINESS_COLUMNS)
    params: list[Any] = [str(path), BUSINESS_COLUMNS]
    clauses: list[str] = []
    if not keep_closed:
        # NULL is_open would be malformed input; treat only an explicit 1 as open.
        clauses.append("b.is_open = 1")
    if definition is not None:
        fragment, extra = _sql_metro_filter(definition)
        clauses.append(fragment)
        params.extend(extra)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    con = connect()
    try:
        return list(_stream_rows(con, sql, params, DEFAULT_BATCH_ROWS))
    except duckdb.Error as exc:  # pragma: no cover - depends on a malformed download
        raise RuntimeError(
            f"Failed to read {path}. Expected newline-delimited JSON with the keys "
            f"{sorted(BUSINESS_COLUMNS)}. DuckDB said: {exc}"
        ) from exc
    finally:
        con.close()


def load_business_ids(cfg: Config, metro: str | None = None, **kwargs: Any) -> list[str]:
    """Just the ids in scope — the input to the review/tip loaders."""
    return [b["business_id"] for b in load_businesses(cfg, metro, **kwargs)]


def _scope_table(ids: Iterable[str]) -> pa.Table:
    """The id set, as an Arrow table DuckDB can semi-join against.

    Arrow rather than an ``IN (...)`` literal list: the literal form grows the
    query string linearly in the number of in-scope ids and filters row by row,
    while a semi-join against a registered table is a hash join whose cost does
    not depend on how the predicate is spelled. We do not yet know how many ids
    a metro produces (``make explore`` is what tells us), so the form that
    degrades gracefully is the one to write.
    """
    unique = sorted(set(ids))
    return pa.table({"business_id": pa.array(unique, type=pa.string())})


# Column names here are deliberately unqualified: the ORDER BY is used inside a
# window function over a subquery, where the table alias no longer exists.
_ORDER_SQL: Final[dict[str, str]] = {
    # Matches cfg.documents.review_selection == "useful_then_recent".
    "useful_then_recent": "useful DESC, date DESC",
    "recent": "date DESC",
    # Deterministic pseudo-random: hashing the id gives the same sample on every
    # run, which a plain random() would not. Reproducibility is a project rule.
    "random": "hash(review_id)",
}


def _iter_rows_for(
    cfg: Config,
    filename: str,
    columns: dict[str, str],
    business_ids: Iterable[str],
    *,
    alias: str,
    max_per_business: int | None,
    order_sql: str | None,
    batch_rows: int,
) -> Iterator[dict[str, Any]]:
    """Shared body of the review and tip loaders."""
    scope = _scope_table(business_ids)
    if scope.num_rows == 0:
        # No businesses in scope means no rows; do not touch the 5 GB file.
        return
    path = require_raw_file(cfg, filename)

    inner = _read_json_sql(alias, columns)
    sql = f"{inner} SEMI JOIN _scope s ON s.business_id = {alias}.business_id"
    if max_per_business is not None:
        if order_sql is None:
            raise ValueError("max_per_business requires an order")
        projected = ", ".join(columns)
        # The per-business cap is applied inside DuckDB so the rows that lose
        # the cut never become Python objects. DuckDB spills the window
        # operator to disk if it has to, which plain Python sorting cannot.
        sql = (
            f"SELECT {projected} FROM ("
            f"  SELECT {projected}, row_number() OVER ("
            f"    PARTITION BY business_id ORDER BY {order_sql}"
            f"  ) AS _rn FROM ({sql})"
            f") WHERE _rn <= {int(max_per_business)}"
        )

    con = connect()
    try:
        con.register("_scope", scope)
        yield from _stream_rows(con, sql, [str(path), columns], batch_rows)
    finally:
        con.close()


def iter_reviews_for(
    cfg: Config,
    business_ids: Iterable[str],
    *,
    max_per_business: int | None = None,
    order: ReviewOrder = "useful_then_recent",
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> Iterator[dict[str, Any]]:
    """Stream reviews belonging to ``business_ids``, newest/most-useful first.

    This is the function that must never materialize review.json. Prefer it
    over ``load_reviews_for`` whenever the caller can consume rows as they
    arrive, and pass ``max_per_business`` when only a sample is needed (the
    document builder caps at ``cfg.documents.max_reviews``) — the cap then runs
    inside DuckDB instead of after the fact.
    """
    if order not in _ORDER_SQL:
        raise ValueError(f"unknown review order {order!r}; expected one of {sorted(_ORDER_SQL)}")
    yield from _iter_rows_for(
        cfg,
        REVIEW_FILE,
        REVIEW_COLUMNS,
        business_ids,
        alias="r",
        max_per_business=max_per_business,
        order_sql=_ORDER_SQL[order],
        batch_rows=batch_rows,
    )


def load_reviews_for(
    cfg: Config,
    business_ids: Iterable[str],
    *,
    max_per_business: int | None = None,
    order: ReviewOrder = "useful_then_recent",
) -> list[dict[str, Any]]:
    """Materializing wrapper around :func:`iter_reviews_for`.

    Safe only because the caller has already narrowed the scope to one metro
    *and* usually capped reviews per business. With neither narrowing, this is
    the 5 GB file in RAM — hence the explicit warning rather than a convenient
    default.
    """
    return list(iter_reviews_for(cfg, business_ids, max_per_business=max_per_business, order=order))


def iter_tips_for(
    cfg: Config,
    business_ids: Iterable[str],
    *,
    max_per_business: int | None = None,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> Iterator[dict[str, Any]]:
    """Stream tips for the given businesses, most-complimented first.

    Tips are short, so they are ordered by ``compliment_count`` — the only
    quality signal a tip carries.
    """
    yield from _iter_rows_for(
        cfg,
        TIP_FILE,
        TIP_COLUMNS,
        business_ids,
        alias="t",
        max_per_business=max_per_business,
        order_sql="compliment_count DESC, date DESC",
        batch_rows=batch_rows,
    )


def load_tips_for(
    cfg: Config,
    business_ids: Iterable[str],
    *,
    max_per_business: int | None = None,
) -> list[dict[str, Any]]:
    """Materializing wrapper around :func:`iter_tips_for`. tip.json is small."""
    return list(iter_tips_for(cfg, business_ids, max_per_business=max_per_business))


def count_reviews_by_business(
    cfg: Config, business_ids: Iterable[str] | None = None
) -> dict[str, int]:
    """Count review rows per business by streaming review.json.

    Separate from the ``review_count`` field in business.json, which is Yelp's
    own total and is *larger* than what the download contains — review.json is
    a subset. Exploration reports both and labels which is which; conflating
    them would overstate how much review text the documents can draw on.

    Only ``business_id`` is projected, so this never reads review text.
    """
    path = require_raw_file(cfg, REVIEW_FILE)
    columns = {"business_id": "VARCHAR"}
    sql = (
        "SELECT r.business_id, count(*) AS n "
        "FROM read_json($1, format='newline_delimited', columns=$2) AS r"
    )
    con = connect()
    try:
        if business_ids is not None:
            scope = _scope_table(business_ids)
            if scope.num_rows == 0:
                return {}
            con.register("_scope", scope)
            sql += " SEMI JOIN _scope s ON s.business_id = r.business_id"
        sql += " GROUP BY r.business_id"
        rows = con.execute(sql, [str(path), columns]).fetchall()
    finally:
        con.close()
    return {bid: int(n) for bid, n in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Small shared helpers used by the exploration report
# ─────────────────────────────────────────────────────────────────────────────


def parse_categories(raw: str | None) -> tuple[str, ...]:
    """Yelp stores categories as one comma-separated string, or null."""
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def median_or_none(values: Sequence[float]) -> float | None:
    """Median, or None for an empty input — never a stand-in zero.

    A zero here would read as a measured value in the report. "No businesses in
    this metro" must look different from "a median of zero".
    """
    return float(median(values)) if values else None
