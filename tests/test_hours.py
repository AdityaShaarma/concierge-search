"""Tests for hours parsing and the three-valued open/closed check.

Overnight hours are the subtlest correctness issue in the data layer, so they
get boundary tests on both sides of every edge: exactly at open, one minute
before, exactly at close, one minute before close.
"""

from __future__ import annotations

import pytest

from concierge.data.hours import (
    DAYS,
    MINUTES_PER_DAY,
    OPEN_24H,
    canonical_day,
    has_hours_data,
    is_open_at,
    parse_hours,
    parse_interval,
    parse_user_time,
)
from concierge.types import ConstraintStatus, HoursInterval

SATISFIED = ConstraintStatus.SATISFIED
VIOLATED = ConstraintStatus.VIOLATED
UNKNOWN = ConstraintStatus.UNKNOWN


def at(hour: int, minute: int = 0) -> int:
    return hour * 60 + minute


# ─────────────────────────────────────────────────────────────────────────────
# parse_interval
# ─────────────────────────────────────────────────────────────────────────────


def test_simple_interval():
    assert parse_interval("7:0-20:0") == HoursInterval(at(7), at(20))


def test_minutes_are_not_zero_padded_in_the_dataset():
    assert parse_interval("7:5-20:5") == HoursInterval(at(7, 5), at(20, 5))
    assert parse_interval("7:05-20:05") == HoursInterval(at(7, 5), at(20, 5))


def test_overnight_interval_crosses_midnight():
    interval = parse_interval("17:0-2:0")
    assert interval == HoursInterval(at(17), at(2))
    assert interval.crosses_midnight


def test_interval_ending_exactly_at_midnight_does_not_cross():
    """``"17:0-0:0"`` closes at the end of this day.

    Reading the ``0`` as "cross midnight" would claim the place is open at 1am,
    which is wrong by a whole night.
    """
    interval = parse_interval("17:0-0:0")
    assert interval == HoursInterval(at(17), MINUTES_PER_DAY)
    assert not interval.crosses_midnight


def test_zero_to_zero_is_open_24h():
    assert parse_interval("0:0-0:0") == OPEN_24H
    assert not OPEN_24H.crosses_midnight
    assert OPEN_24H.covers(0)
    assert OPEN_24H.covers(MINUTES_PER_DAY - 1)


def test_equal_nonzero_endpoints_are_read_as_24h():
    """Documented assumption, not a dataset fact: "12:0-12:0" is ambiguous and
    we read it as all day, consistent with "0:0-0:0"."""
    assert parse_interval("12:0-12:0") == OPEN_24H


@pytest.mark.parametrize(
    "text",
    ["", "   ", "abc", "7:0", "7:0-", "-20:0", "25:0-20:0", "7:60-20:0", "7:0-20:0-21:0"],
)
def test_malformed_intervals_return_none(text):
    assert parse_interval(text) is None


# ─────────────────────────────────────────────────────────────────────────────
# parse_hours
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_hours_of_a_normal_business(raw_by_id):
    hours = parse_hours(raw_by_id["b001"]["hours"])
    assert set(hours) == set(DAYS)
    assert hours["Monday"] == (HoursInterval(at(7), at(20)),)
    assert hours["Thursday"] == (HoursInterval(at(7), at(23)),)


def test_null_hours_parse_to_empty(raw_by_id):
    assert raw_by_id["b005"]["hours"] is None
    assert parse_hours(raw_by_id["b005"]["hours"]) == {}
    assert parse_hours(None) == {}


def test_missing_days_are_simply_absent(raw_by_id):
    """b003 lists Monday, Saturday, Sunday only."""
    hours = parse_hours(raw_by_id["b003"]["hours"])
    assert set(hours) == {"Monday", "Saturday", "Sunday"}


def test_unparseable_days_are_dropped_not_invented():
    hours = parse_hours({"Monday": "7:0-20:0", "Tuesday": "garbage"})
    assert set(hours) == {"Monday"}


def test_unknown_day_keys_are_ignored():
    assert parse_hours({"Funday": "7:0-20:0"}) == {}


def test_abbreviated_day_keys_are_normalized():
    assert parse_hours({"Fri": "7:0-20:0"}) == {"Friday": (HoursInterval(at(7), at(20)),)}


def test_has_hours_data_separates_no_field_from_unparseable():
    """The parsed form cannot express this difference — both come back empty —
    so data-quality reporting needs the raw predicate."""
    assert has_hours_data(None) is False
    assert has_hours_data({}) is False
    assert has_hours_data({"Monday": "garbage"}) is True
    assert parse_hours({"Monday": "garbage"}) == {}


# ─────────────────────────────────────────────────────────────────────────────
# is_open_at — the three values
# ─────────────────────────────────────────────────────────────────────────────


def test_no_hours_data_is_unknown_never_closed(raw_by_id):
    """The single most important case in this module.

    b005 has ``hours == null``. Reporting it closed would delete every
    hours-less business from every time-constrained query on the strength of no
    evidence at all.
    """
    hours = parse_hours(raw_by_id["b005"]["hours"])
    for day in DAYS:
        assert is_open_at(hours, day, at(13)) is UNKNOWN


def test_day_absent_from_a_populated_schedule_is_closed(raw_by_id):
    """b003 has no Tuesday. That is a fact about Tuesday, not missing data —
    contrast with the UNKNOWN case above."""
    hours = parse_hours(raw_by_id["b003"]["hours"])
    assert is_open_at(hours, "Tuesday", at(13)) is VIOLATED
    assert is_open_at(hours, "Monday", at(13)) is SATISFIED


# ── boundaries on a plain daytime interval (b001 Monday 7:00-20:00) ──────────


@pytest.mark.parametrize(
    ("minute", "expected"),
    [
        (at(6, 59), VIOLATED),  # one minute before open
        (at(7, 0), SATISFIED),  # exactly at open: inclusive
        (at(7, 1), SATISFIED),
        (at(19, 59), SATISFIED),  # one minute before close
        (at(20, 0), VIOLATED),  # exactly at close: exclusive
        (at(20, 1), VIOLATED),
        (0, VIOLATED),
        (MINUTES_PER_DAY - 1, VIOLATED),
    ],
)
def test_daytime_interval_boundaries(raw_by_id, minute, expected):
    hours = parse_hours(raw_by_id["b001"]["hours"])
    assert is_open_at(hours, "Monday", minute) is expected


def test_open_past_10pm_query_separates_the_fixtures(raw_by_id):
    """The query shape that motivates this whole module."""
    cafe = parse_hours(raw_by_id["b001"]["hours"])  # Thu/Fri until 23:00
    assert is_open_at(cafe, "Friday", at(22)) is SATISFIED
    assert is_open_at(cafe, "Monday", at(22)) is VIOLATED  # Monday closes at 20:00


# ── overnight spillover (b002: Mon-Thu 17:00-02:00, Fri/Sat 17:00-03:00) ─────


def test_overnight_evening_half_is_open(raw_by_id):
    hours = parse_hours(raw_by_id["b002"]["hours"])
    assert is_open_at(hours, "Friday", at(17)) is SATISFIED
    assert is_open_at(hours, "Friday", at(23, 59)) is SATISFIED
    assert is_open_at(hours, "Friday", at(16, 59)) is VIOLATED


def test_overnight_spills_into_the_following_day(raw_by_id):
    """Friday 17:00-03:00 means the business IS open at Saturday 01:00.

    is_open_at handles this spillover by inspecting the previous day. Without
    it, "ramen at 1am on Saturday" would miss exactly the businesses it wants —
    the overnight places are the only ones open at that hour.
    """
    hours = parse_hours(raw_by_id["b002"]["hours"])
    assert is_open_at(hours, "Saturday", at(1)) is SATISFIED


@pytest.mark.parametrize(
    ("minute", "expected"),
    [
        (at(0, 0), SATISFIED),  # midnight, start of the spilled-over morning
        (at(2, 59), SATISFIED),  # one minute before the 03:00 close
        (at(3, 0), VIOLATED),  # exactly at close: exclusive
        (at(3, 1), VIOLATED),
        (at(10, 0), VIOLATED),  # mid-morning: nothing is open
    ],
)
def test_spillover_boundaries(raw_by_id, minute, expected):
    """Saturday morning is governed entirely by Friday's 17:00-03:00 shift."""
    hours = parse_hours(raw_by_id["b002"]["hours"])
    assert is_open_at(hours, "Saturday", minute) is expected


def test_a_days_own_overnight_shift_does_not_open_that_days_morning(raw_by_id):
    """The bug that `HoursInterval.covers()` alone would introduce.

    b002's Monday shift is 17:00-02:00. ``covers(60)`` is True for that interval,
    because it answers "is 01:00 inside the interval" on the interval's own
    clock — but that 01:00 belongs to Tuesday. Monday 01:00 is governed by
    *Sunday*, which closes at midnight, so Monday 01:00 must be VIOLATED.
    """
    hours = parse_hours(raw_by_id["b002"]["hours"])
    monday = hours["Monday"][0]
    assert monday.covers(at(1))  # the trap
    assert is_open_at(hours, "Monday", at(1)) is VIOLATED


def test_interval_ending_at_midnight_does_not_spill(raw_by_id):
    """b002's Sunday is 17:00-00:00, so Monday 00:00 is closed."""
    hours = parse_hours(raw_by_id["b002"]["hours"])
    assert is_open_at(hours, "Sunday", at(23, 59)) is SATISFIED
    assert is_open_at(hours, "Monday", at(0)) is VIOLATED


def test_spillover_across_the_week_boundary(raw_by_id):
    """Sunday → Monday wraps the day list; b006 is open Sat 11:00-01:00."""
    hours = parse_hours(raw_by_id["b006"]["hours"])
    assert is_open_at(hours, "Sunday", at(0, 30)) is SATISFIED
    assert is_open_at(hours, "Sunday", at(12)) is VIOLATED  # no Sunday hours


# ── open 24 hours (b007) ────────────────────────────────────────────────────


@pytest.mark.parametrize("minute", [0, at(3), at(12), MINUTES_PER_DAY - 1])
def test_open_24h_is_always_open(raw_by_id, minute):
    hours = parse_hours(raw_by_id["b007"]["hours"])
    for day in DAYS:
        assert is_open_at(hours, day, minute) is SATISFIED


def test_open_24h_does_not_spill_into_a_closed_day():
    """A 24-hour Monday says nothing about Tuesday.

    This is why ``OPEN_24H`` uses ``end_min = 1440`` instead of 0: an interval
    with ``end_min == 0`` would report as crossing midnight and leak into
    Tuesday morning.
    """
    hours = parse_hours({"Monday": "0:0-0:0"})
    assert is_open_at(hours, "Monday", at(1)) is SATISFIED
    assert is_open_at(hours, "Tuesday", at(1)) is VIOLATED


# ── split shifts (not in this dataset, but the contract allows them) ────────


def test_multiple_intervals_per_day_are_all_checked():
    hours = {"Monday": (HoursInterval(at(11), at(14)), HoursInterval(at(17), at(22)))}
    assert is_open_at(hours, "Monday", at(12)) is SATISFIED
    assert is_open_at(hours, "Monday", at(15)) is VIOLATED
    assert is_open_at(hours, "Monday", at(18)) is SATISFIED


# ── caller errors ───────────────────────────────────────────────────────────


def test_day_argument_accepts_abbreviations_and_any_case(raw_by_id):
    """The LLM parser emits "Fri"; the dataset says "Friday"."""
    hours = parse_hours(raw_by_id["b001"]["hours"])
    assert is_open_at(hours, "Fri", at(22)) is SATISFIED
    assert is_open_at(hours, "friday", at(22)) is SATISFIED
    assert is_open_at(hours, " FRIDAY ", at(22)) is SATISFIED


def test_bad_day_raises_rather_than_reporting_unknown(raw_by_id):
    """A caller bug must not hide behind a plausible-looking UNKNOWN."""
    hours = parse_hours(raw_by_id["b001"]["hours"])
    with pytest.raises(ValueError, match="unrecognized day"):
        is_open_at(hours, "Funday", at(12))


@pytest.mark.parametrize("minute", [-1, MINUTES_PER_DAY, 10_000])
def test_out_of_range_minute_raises(raw_by_id, minute):
    hours = parse_hours(raw_by_id["b001"]["hours"])
    with pytest.raises(ValueError, match="minute_of_day"):
        is_open_at(hours, "Monday", minute)


def test_canonical_day():
    assert canonical_day("Mon") == "Monday"
    assert canonical_day("SUNDAY") == "Sunday"
    assert canonical_day("Smarch") is None


# ─────────────────────────────────────────────────────────────────────────────
# parse_user_time
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10pm", at(22)),
        ("10 pm", at(22)),
        ("10 p.m.", at(22)),
        ("10PM", at(22)),
        ("22:00", at(22)),
        ("22", at(22)),
        ("10:30pm", at(22, 30)),
        ("9am", at(9)),
        ("12am", 0),
        ("12pm", at(12)),
        ("12:01am", at(0, 1)),
        ("midnight", 0),
        ("noon", at(12)),
        ("0:00", 0),
        ("23:59", at(23, 59)),
        ("7:5", at(7, 5)),
    ],
)
def test_parse_user_time(text, expected):
    assert parse_user_time(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "late", "24:00", "25", "13pm", "0pm", "10:60", "tenpm", "half past nine"],
)
def test_unparseable_user_time_returns_none(text):
    """``None`` rather than a default: a parser that cannot read the time must
    drop the constraint, not filter the corpus by a time nobody asked for."""
    assert parse_user_time(text) is None


def test_user_time_round_trips_into_is_open_at(raw_by_id):
    """The actual call path for "open past 10pm on a Friday"."""
    minute = parse_user_time("10pm")
    assert minute is not None
    assert is_open_at(parse_hours(raw_by_id["b001"]["hours"]), "Fri", minute) is SATISFIED
    assert is_open_at(parse_hours(raw_by_id["b003"]["hours"]), "Fri", minute) is VIOLATED
