"""Shared test fixtures.

`tests/fixtures/*.json` are newline-delimited JSON in exactly the shape the Yelp
Open Dataset uses, including its awkward parts, so the parsing code is exercised
against the real quirks without needing the 5 GB download.

Every quirk in the fixture corpus is there on purpose:

  b001  stringified unicode literals (``"u'free'"``) and nested dict strings
        (``"{'casual': True}"``)
  b002  overnight hours (``"17:0-2:0"``) plus an interval ending exactly at
        midnight (``"17:0-0:0"``)
  b003  plain quoted literals (``"'quiet'"``) and several weekdays missing
        entirely from the hours dict
  b004  ``is_open == 0`` — must never appear in a searchable index
  b005  ``attributes`` and ``hours`` both null — every constraint is UNKNOWN
  b006  a different metro, so metro scoping can be tested
  b007  ``"0:0-0:0"``, Yelp's encoding for open 24 hours
  b008  ``"None"`` as an attribute *string*, which must normalize to missing

If you add a quirk to the fixtures, document it here and say which test covers it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _read_ndjson(name: str) -> list[dict]:
    path = FIXTURE_DIR / name
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="session")
def raw_businesses() -> list[dict]:
    """Business records exactly as they appear in `business.json`."""
    return _read_ndjson("businesses.json")


@pytest.fixture(scope="session")
def raw_reviews() -> list[dict]:
    return _read_ndjson("reviews.json")


@pytest.fixture(scope="session")
def raw_tips() -> list[dict]:
    return _read_ndjson("tips.json")


@pytest.fixture(scope="session")
def raw_by_id(raw_businesses: list[dict]) -> dict[str, dict]:
    return {b["business_id"]: b for b in raw_businesses}


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def config():
    """A config pointing at the fixture corpus, with a metro set.

    Metro is set here (unlike the committed YAML, which deliberately has none)
    because tests must not depend on an owner decision that has not been made.
    """
    from concierge.config import load_config

    cfg = load_config("configs/base.yaml")
    return cfg.model_copy(update={"data": cfg.data.model_copy(update={"metro": "Metroville"})})
