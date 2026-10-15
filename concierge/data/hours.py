"""Parse Yelp opening hours and answer "is this place open at this time?".

Yelp stores hours as one string per day, ``{"Monday": "7:0-20:0"}``, with
minutes that are **not** zero-padded. Three encodings in that format are easy to
get wrong, and all three appear in the gold set's "open late" queries:

* ``"17:0-2:0"`` — closes at 2am the *next* day.
* ``"17:0-0:0"`` — closes at midnight, i.e. end of *this* day. Treating the ``0``
  as "cross midnight" would claim the place is open at 1am.
* ``"0:0-0:0"`` — Yelp's encoding for open 24 hours.

Two kinds of "not open" must stay distinguishable, which is why
:func:`is_open_at` returns a three-valued :class:`ConstraintStatus`:

* a business with hours data where the requested day is absent is **closed**
  that day (``VIOLATED``);
* a business with ``hours == null`` tells us **nothing** (``UNKNOWN``).

Ranking the second group as if they were closed would delete a large slice of
the corpus from every time-constrained query — hours coverage in this dataset is
far from complete — and ranking them as if they were open would fill the results
with places that are shut. Both are wrong in different directions, so the
decision is pushed up to the retrieval policy instead of being baked in here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

from concierge.types import ConstraintStatus, DayOfWeek, HoursInterval

__all__ = [
    "DAYS",
    "MINUTES_PER_DAY",
    "OPEN_24H",
    "canonical_day",
    "has_hours_data",
    "is_open_at",
    "parse_hours",
    "parse_interval",
    "parse_user_time",
]

MINUTES_PER_DAY: Final = 24 * 60

# As spelled in the dataset, Monday first. `DayOfWeek` is just `str`, so this
# tuple is the only authority on valid day names.
DAYS: Final[tuple[DayOfWeek, ...]] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# Open all day. `end_min == MINUTES_PER_DAY` rather than 0 so that
# `HoursInterval.crosses_midnight` is False: a 24-hour day must not also spill
# over into the following morning (that would double-count the next day, which
# may well be closed).
OPEN_24H: Final = HoursInterval(0, MINUTES_PER_DAY)

# Accepts "Monday", "monday", "Mon", "mon". The LLM query parser emits
# abbreviations ("Fri") while the dataset uses full names, so normalizing in one
# place keeps that mismatch from becoming a silent zero-result bug.
_DAY_LOOKUP: Final[dict[str, DayOfWeek]] = {
    **{day.lower(): day for day in DAYS},
    **{day[:3].lower(): day for day in DAYS},
}

_PREVIOUS_DAY: Final[dict[DayOfWeek, DayOfWeek]] = {
    day: DAYS[(index - 1) % len(DAYS)] for index, day in enumerate(DAYS)
}

# "7:0-20:0". Hours and minutes are both unpadded in the dataset.
_INTERVAL_RE: Final = re.compile(r"^(\d{1,2}):(\d{1,2})-(\d{1,2}):(\d{1,2})$")

# "10pm", "22:00", "22", "10:30pm". Dots and spaces are stripped before matching
# so "10 p.m." arrives here as "10pm".
_USER_TIME_RE: Final = re.compile(r"^(\d{1,2})(?::(\d{1,2}))?(am|pm)?$")

_NAMED_TIMES: Final[dict[str, int]] = {
    "midnight": 0,
    "noon": 12 * 60,
    "midday": 12 * 60,
}


def canonical_day(day: str) -> DayOfWeek | None:
    """Map a day name or three-letter abbreviation to the dataset's spelling."""
    return _DAY_LOOKUP.get(day.strip().lower())


def has_hours_data(raw: Mapping[str, Any] | None) -> bool:
    """True if the *raw* record carries any hours information at all.

    Separate from :func:`parse_hours` because the parsed form cannot express the
    difference between "no hours field" and "hours field we could not parse" —
    both come back empty. Callers doing data-quality reporting need the raw
    answer; callers doing search do not, since :func:`is_open_at` already treats
    an empty parse as ``UNKNOWN``.
    """
    return isinstance(raw, Mapping) and len(raw) > 0


def parse_interval(text: str) -> HoursInterval | None:
    """Parse one ``"H:M-H:M"`` range, or ``None`` if it is malformed.

    Returns ``None`` rather than raising: one unparseable day in a 150K-row file
    must not abort the data build. The cost is that the day then reads as closed,
    which is a slightly wrong claim — but the alternative, inventing hours for
    it, is a worse one, and :func:`parse_hours` reports nothing it did not parse.
    """
    match = _INTERVAL_RE.match(text.strip())
    if match is None:
        return None

    start_hour, start_minute, end_hour, end_minute = (int(g) for g in match.groups())
    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
        return None
    if not (0 <= start_minute <= 59 and 0 <= end_minute <= 59):
        return None

    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute

    # Open 24 hours, handled as its own case on purpose. The generic rule below
    # (end == 0 means "midnight at the end of this day") happens to produce the
    # same interval, but relying on that coincidence would leave the single most
    # surprising encoding in the dataset undocumented and untested.
    if start == 0 and end == 0:
        return OPEN_24H

    if end == 0:
        # "17:0-0:0" closes at the end of this day, it does not wrap to 1am.
        return HoursInterval(start, MINUTES_PER_DAY)

    if start == end:
        # Rare, and genuinely ambiguous: "12:0-12:0" could mean a full 24 hours
        # starting at noon or a zero-length (closed) day. We read it as 24 hours,
        # consistent with "0:0-0:0" being Yelp's all-day encoding. Zero-length
        # would be unmatchable by any query, so the cost of being wrong here is
        # a few false positives rather than silently hiding businesses.
        return OPEN_24H

    return HoursInterval(start, end)


def parse_hours(
    raw: Mapping[str, Any] | None,
) -> dict[DayOfWeek, tuple[HoursInterval, ...]]:
    """Parse a raw Yelp hours blob into intervals per day.

    Args:
        raw: the ``hours`` field of a ``business.json`` record, or ``None``.

    Returns:
        ``{day: (interval, ...)}`` containing only days that parsed. Days absent
        from the input are absent from the output and mean closed; an empty dict
        means we have no hours data (see :func:`is_open_at`, which maps that to
        ``UNKNOWN``).

    The value is a tuple because :class:`Business` is frozen and because the
    shared contract allows for split hours (lunch and dinner service). This
    dataset supplies at most one range per day, so the tuples are all length one
    — the type leaves room without pretending we have data we do not.
    """
    if not isinstance(raw, Mapping):
        return {}

    parsed: dict[DayOfWeek, tuple[HoursInterval, ...]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        day = canonical_day(key)
        if day is None:
            continue
        interval = parse_interval(value)
        if interval is None:
            continue
        # Later duplicates of the same day accumulate rather than overwrite.
        parsed[day] = (*parsed.get(day, ()), interval)
    return parsed


def is_open_at(
    hours: Mapping[DayOfWeek, tuple[HoursInterval, ...]],
    day: str,
    minute_of_day: int,
) -> ConstraintStatus:
    """Three-valued "is this business open on ``day`` at ``minute_of_day``?".

    Args:
        hours: output of :func:`parse_hours`.
        day: full name or three-letter abbreviation, any case.
        minute_of_day: 0-1439.

    Returns:
        ``UNKNOWN`` when ``hours`` is empty (no data), otherwise ``SATISFIED`` or
        ``VIOLATED``.

    Raises:
        ValueError: for an unrecognized day or an out-of-range minute. Both are
            caller bugs rather than bad data, and swallowing them as ``UNKNOWN``
            would hide a broken query parser behind plausible-looking results.

    **Overnight spillover is handled.** A Friday interval of 17:00-02:00 means the
    business IS open at Saturday 01:00, so this function also inspects the
    previous day. Without that, "ramen at 1am on Saturday" would miss exactly the
    businesses it is looking for — the overnight places are the only ones open.
    """
    if not hours:
        return ConstraintStatus.UNKNOWN

    canonical = canonical_day(day)
    if canonical is None:
        raise ValueError(f"unrecognized day: {day!r}")
    if not 0 <= minute_of_day < MINUTES_PER_DAY:
        raise ValueError(f"minute_of_day out of range: {minute_of_day!r}")

    for interval in hours.get(canonical, ()):
        if interval.crosses_midnight:
            # Only the evening half belongs to this calendar day. `covers()`
            # answers "is this minute inside the interval on the interval's own
            # clock", which merges the two calendar days the interval touches —
            # correct for the interval, wrong for a day-aware question. Using it
            # here would report "open" at Saturday 01:00 purely because Saturday
            # also happens to have a 17:00-03:00 shift.
            if minute_of_day >= interval.start_min:
                return ConstraintStatus.SATISFIED
        elif interval.covers(minute_of_day):
            return ConstraintStatus.SATISFIED

    # The morning half of the previous day's overnight shift.
    for interval in hours.get(_PREVIOUS_DAY[canonical], ()):
        if interval.crosses_midnight and minute_of_day < interval.end_min:
            return ConstraintStatus.SATISFIED

    return ConstraintStatus.VIOLATED


def parse_user_time(text: str) -> int | None:
    """Parse a time the way a *person* writes it into a minute-of-day.

    Accepts ``"10pm"``, ``"10:30 p.m."``, ``"22:00"``, ``"22"``, ``"noon"``,
    ``"midnight"``.

    Returns:
        Minutes from midnight (0-1439), or ``None`` if the text is not a time we
        recognize. ``None`` rather than a default: a query parser that cannot
        read the time must drop the ``open_at`` constraint, not guess at one and
        then filter the corpus by a time the user never asked for.
    """
    cleaned = text.strip().lower().replace(".", "").replace(" ", "")
    if cleaned in _NAMED_TIMES:
        return _NAMED_TIMES[cleaned]

    match = _USER_TIME_RE.match(cleaned)
    if match is None:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if minute > 59:
        return None

    if meridiem is None:
        # Bare number: read as a 24-hour clock. "22" is 10pm; "9" is 9am, which
        # is the ambiguity we accept — an LLM parser asked for a 24-hour time
        # should emit one, and a human writing "9" without am/pm is ambiguous to
        # a human too.
        if hour > 23:
            return None
    else:
        if not 1 <= hour <= 12:
            return None
        # 12am is hour 0 and 12pm is hour 12; every other pm hour shifts by 12.
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0

    return hour * 60 + minute
