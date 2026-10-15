"""Tests for the streaming loaders and the exploration report.

The real dataset is a ~5 GB download that is not in the repo, so these tests
run against `tests/fixtures/`, copied into a temp directory under the *real*
filenames (`business.json`, not `businesses.json`). That matters: it exercises
the actual path resolution and the actual "where is my data" error, instead of
a test-only shortcut around them.

Tests that genuinely need the download are marked `needs_data` and skip.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
from rich.console import Console

from concierge.config import Config, load_config
from concierge.data import explore
from concierge.data import load as L

# ─────────────────────────────────────────────────────────────────────────────
# Fixture plumbing
# ─────────────────────────────────────────────────────────────────────────────

#: fixture filename -> the name the Yelp download uses.
_FIXTURE_TO_RAW = {
    "businesses.json": L.BUSINESS_FILE,
    "reviews.json": L.REVIEW_FILE,
    "tips.json": L.TIP_FILE,
}


@pytest.fixture
def raw_dir(tmp_path: Path, fixture_dir: Path) -> Path:
    """A `data/raw`-shaped directory holding the fixture corpus."""
    target = tmp_path / "raw"
    target.mkdir()
    for src, dst in _FIXTURE_TO_RAW.items():
        shutil.copyfile(fixture_dir / src, target / dst)
    return target


def _with_raw(cfg: Config, raw_dir: Path, **data_overrides: object) -> Config:
    """Point a config at a raw directory. Config is frozen, so copy."""
    return cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"raw_dir": raw_dir, **data_overrides})}
    )


@pytest.fixture
def cfg(config: Config, raw_dir: Path) -> Config:
    """The shared `config` fixture (metro = "Metroville") over the fixture files."""
    return _with_raw(config, raw_dir)


def _ids(records: list[dict]) -> list[str]:
    return sorted(r["business_id"] for r in records)


# ─────────────────────────────────────────────────────────────────────────────
# Missing-data failure mode
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingDataset:
    """The first thing a reader of this repo will hit is an empty data/raw. The
    message has to tell them what to do, not print a DuckDB parse error."""

    def test_missing_directory_names_the_download(self, config: Config, tmp_path: Path):
        cfg = _with_raw(config, tmp_path / "does-not-exist")
        with pytest.raises(L.MissingDatasetError) as excinfo:
            L.load_businesses(cfg)
        message = str(excinfo.value)
        assert "yelp.com/dataset" in message
        assert L.BUSINESS_FILE in message
        assert "does not exist" in message

    def test_empty_directory_says_empty(self, config: Config, tmp_path: Path):
        empty = tmp_path / "raw"
        empty.mkdir()
        with pytest.raises(L.MissingDatasetError, match="is empty"):
            L.load_businesses(_with_raw(config, empty))

    def test_wrong_filenames_list_what_was_found(self, config: Config, tmp_path: Path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "businesses.json").write_text("{}\n")  # plural: the wrong name
        with pytest.raises(L.MissingDatasetError) as excinfo:
            L.load_businesses(_with_raw(config, raw))
        assert "businesses.json" in str(excinfo.value)

    def test_missing_dataset_is_a_file_not_found_error(self):
        # Callers that only catch FileNotFoundError must still work.
        assert issubclass(L.MissingDatasetError, FileNotFoundError)


# ─────────────────────────────────────────────────────────────────────────────
# City / metro resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestCityNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Metroville", "metroville"),
            ("  metroville  ", "metroville"),
            ("METROVILLE", "metroville"),
            ("Tucson.", "tucson"),
            ("New  Orleans", "new orleans"),
            # All three spellings of the same city must collapse, or a metro
            # definition silently drops a third of its businesses.
            ("St. Louis", "saint louis"),
            ("St Louis", "saint louis"),
            ("Saint Louis", "saint louis"),
            ("Ste. Anne", "sainte anne"),
            # Accent folding, for the Canadian metros.
            ("Montréal", "montreal"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_normalization(self, raw, expected):
        assert L.normalize_city(raw) == expected

    def test_saint_fold_does_not_eat_real_words(self):
        # "Stockton" starts with "st" but is not "Saint ockton".
        assert L.normalize_city("Stockton") == "stockton"


class TestMetroResolution:
    def test_curated_metro_is_found_case_insensitively(self):
        definition = L.resolve_metro("saint louis")
        assert definition.name == "Saint Louis"
        # Saint Louis genuinely spans two states; that is the point of the state set.
        assert {"MO", "IL"} <= definition.states

    def test_curated_metro_includes_suburbs(self):
        indy = L.resolve_metro("Indianapolis")
        assert indy.matches("Carmel", "IN")
        assert indy.matches("Indianapolis", "IN")

    def test_state_set_blocks_a_same_named_city_elsewhere(self):
        indy = L.resolve_metro("Indianapolis")
        # Carmel, California is not in the Indianapolis metro.
        assert not indy.matches("Carmel", "CA")

    def test_unknown_name_falls_back_to_a_single_city(self):
        definition = L.resolve_metro("Metroville")
        assert definition.matches("Metroville", "AZ")
        assert not definition.matches("Othertown", "NV")
        # An uncurated fallback must not pretend to be a curated definition.
        assert definition.states == frozenset()
        assert "single city" in definition.note

    def test_assign_metro_returns_none_when_no_definition_claims_the_city(self):
        assert L.assign_metro("Metroville", "AZ") is None
        assert L.assign_metro("Carmel", "IN") == "Indianapolis"


# ─────────────────────────────────────────────────────────────────────────────
# load_businesses
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadBusinesses:
    def test_drop_closed_removes_b004(self, cfg: Config):
        # b004 is the is_open == 0 fixture. cfg.data.drop_closed is true, so a
        # permanently closed business must never reach a searchable index.
        ids = _ids(L.load_businesses(cfg, L.ALL_METROS))
        assert "b004" not in ids
        assert ids == ["b001", "b002", "b003", "b005", "b006", "b007", "b008"]

    def test_drop_closed_false_keeps_b004(self, config: Config, raw_dir: Path):
        cfg = _with_raw(config, raw_dir, drop_closed=False)
        assert "b004" in _ids(L.load_businesses(cfg, L.ALL_METROS))

    def test_include_closed_overrides_the_config(self, cfg: Config):
        assert "b004" in _ids(L.load_businesses(cfg, L.ALL_METROS, include_closed=True))
        assert "b004" not in _ids(L.load_businesses(cfg, L.ALL_METROS, include_closed=False))

    def test_metro_argument_scopes_to_one_city(self, cfg: Config):
        metroville = _ids(L.load_businesses(cfg, "Metroville"))
        assert "b006" not in metroville  # b006 is the Othertown fixture
        assert metroville == ["b001", "b002", "b003", "b005", "b007", "b008"]
        assert _ids(L.load_businesses(cfg, "Othertown")) == ["b006"]

    def test_metro_scoping_is_case_insensitive_in_sql_too(self, cfg: Config):
        # The metro filter runs inside DuckDB, so SQL normalization has to agree
        # with normalize_city(). Same scope from three spellings or they do not.
        expected = _ids(L.load_businesses(cfg, "Metroville"))
        assert _ids(L.load_businesses(cfg, "metroville")) == expected
        assert _ids(L.load_businesses(cfg, "  METROVILLE. ")) == expected

    def test_metro_defaults_to_the_config(self, cfg: Config):
        # The shared config fixture sets metro = "Metroville".
        assert _ids(L.load_businesses(cfg)) == _ids(L.load_businesses(cfg, "Metroville"))

    def test_all_metros_ignores_the_config_metro(self, cfg: Config):
        # Exploration must see every metro even when a metro is configured.
        assert "b006" in _ids(L.load_businesses(cfg, L.ALL_METROS))

    def test_null_config_metro_means_no_filter(self, config: Config, raw_dir: Path):
        cfg = _with_raw(config, raw_dir, metro=None)
        assert "b006" in _ids(L.load_businesses(cfg))

    def test_unmatched_metro_returns_empty_not_an_error(self, cfg: Config):
        # Silence here is correct: a metro that this release does not contain is
        # a fact about the data, not a crash. The explore report surfaces it.
        assert L.load_businesses(cfg, "Atlantis") == []

    def test_sql_filter_agrees_with_python_matching(self, cfg: Config):
        """The metro predicate exists twice — in SQL (for pushdown) and in
        Python (`MetroDefinition.matches`, used by the report). Divergence
        would make counts disagree with the index, so pin them together."""
        everything = L.load_businesses(cfg, L.ALL_METROS)
        for metro in ("Metroville", "Othertown"):
            definition = L.resolve_metro(metro)
            in_python = sorted(
                r["business_id"] for r in everything if definition.matches(r["city"], r["state"])
            )
            assert _ids(L.load_businesses(cfg, metro)) == in_python


class TestBusinessFieldsSurvive:
    @pytest.fixture
    def by_id(self, cfg: Config) -> dict[str, dict]:
        return {
            r["business_id"]: r for r in L.load_businesses(cfg, L.ALL_METROS, include_closed=True)
        }

    def test_geo_is_carried_through_as_floats(self, by_id):
        # D003: coordinates come along from the first build so geo never needs
        # a reindex. A string here would break distance maths silently.
        b001 = by_id["b001"]
        assert isinstance(b001["latitude"], float)
        assert isinstance(b001["longitude"], float)
        assert b001["latitude"] == pytest.approx(33.4484)
        assert b001["longitude"] == pytest.approx(-112.0740)

    def test_attributes_and_hours_arrive_as_dicts(self, by_id):
        b001 = by_id["b001"]
        assert isinstance(b001["attributes"], dict)
        assert isinstance(b001["hours"], dict)
        # Values stay raw: parsing "u'free'" belongs to the attributes module.
        assert b001["attributes"]["WiFi"] == "u'free'"
        assert b001["hours"]["Monday"] == "7:0-20:0"

    def test_arbitrary_attribute_keys_are_preserved(self, by_id):
        # Read as JSON rather than a fixed STRUCT precisely so unusual keys
        # survive. A struct schema inferred from a sample would drop these.
        assert "Ambience" in by_id["b001"]["attributes"]
        assert "BusinessParking" in by_id["b001"]["attributes"]

    def test_null_attributes_stay_none_not_empty_dict(self, by_id):
        # b005 has attributes: null and hours: null. "No data at all" must stay
        # distinguishable from "an empty dict" — the whole three-valued
        # constraint design (D006) rests on that difference.
        b005 = by_id["b005"]
        assert b005["attributes"] is None
        assert b005["hours"] is None

    def test_overnight_hours_string_is_not_mangled(self, by_id):
        assert by_id["b002"]["hours"]["Monday"] == "17:0-2:0"
        assert by_id["b007"]["hours"]["Monday"] == "0:0-0:0"

    def test_scalar_columns_have_the_expected_types(self, by_id):
        b002 = by_id["b002"]
        assert isinstance(b002["stars"], float)
        assert isinstance(b002["review_count"], int)
        assert b002["name"] == "Midnight Ramen"
        assert b002["state"] == "AZ"

    def test_load_business_ids_matches_load_businesses(self, cfg: Config):
        assert sorted(L.load_business_ids(cfg, L.ALL_METROS)) == _ids(
            L.load_businesses(cfg, L.ALL_METROS)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reviews and tips
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadReviews:
    def test_only_rows_for_the_requested_businesses(self, cfg: Config):
        got = L.load_reviews_for(cfg, ["b001"])
        assert sorted(r["review_id"] for r in got) == ["r001", "r002"]
        assert {r["business_id"] for r in got} == {"b001"}

    def test_multiple_businesses(self, cfg: Config):
        got = L.load_reviews_for(cfg, ["b001", "b003"])
        assert sorted(r["review_id"] for r in got) == ["r001", "r002", "r004"]

    def test_duplicate_ids_do_not_duplicate_rows(self, cfg: Config):
        # The scope table is deduplicated; a semi-join would otherwise be fine,
        # but an inner join would fan out. Pin the behaviour.
        assert len(L.load_reviews_for(cfg, ["b001", "b001", "b001"])) == 2

    def test_empty_id_set_returns_nothing(self, cfg: Config):
        assert L.load_reviews_for(cfg, []) == []
        assert L.load_tips_for(cfg, []) == []

    def test_empty_id_set_never_opens_the_file(self, config: Config, tmp_path: Path):
        # No businesses in scope must short-circuit *before* touching the 5 GB
        # file: with no raw data at all this still returns empty, not an error.
        cfg = _with_raw(config, tmp_path / "nothing-here")
        assert L.load_reviews_for(cfg, []) == []

    def test_unknown_business_id_returns_nothing(self, cfg: Config):
        assert L.load_reviews_for(cfg, ["nope"]) == []

    def test_only_the_projected_columns_come_back(self, cfg: Config):
        # Projection is the thing that keeps review.json out of memory, so
        # assert it rather than trusting it. user_id/funny/cool are in the
        # fixture file and must not be read.
        row = L.load_reviews_for(cfg, ["b001"])[0]
        assert set(row) == set(L.REVIEW_COLUMNS)
        assert "user_id" not in row

    def test_max_per_business_keeps_the_most_useful(self, cfg: Config):
        # b001 has r001 (useful=12) and r002 (useful=2). The cap runs inside
        # DuckDB, so the losing row never becomes a Python object.
        got = L.load_reviews_for(cfg, ["b001"], max_per_business=1)
        assert [r["review_id"] for r in got] == ["r001"]

    def test_max_per_business_applies_per_business(self, cfg: Config):
        got = L.load_reviews_for(cfg, ["b001", "b003"], max_per_business=1)
        assert sorted(r["review_id"] for r in got) == ["r001", "r004"]

    def test_recent_order_picks_the_newest(self, cfg: Config):
        got = L.load_reviews_for(cfg, ["b001"], max_per_business=1, order="recent")
        assert [r["review_id"] for r in got] == ["r002"]  # 2022 beats 2021

    def test_random_order_is_deterministic(self, cfg: Config):
        # Hash-based, not random(): the same sample on every run, or no result
        # that depends on review sampling is reproducible.
        first = L.load_reviews_for(cfg, ["b001"], max_per_business=1, order="random")
        second = L.load_reviews_for(cfg, ["b001"], max_per_business=1, order="random")
        assert [r["review_id"] for r in first] == [r["review_id"] for r in second]

    def test_unknown_order_is_rejected(self, cfg: Config):
        with pytest.raises(ValueError, match="unknown review order"):
            list(L.iter_reviews_for(cfg, ["b001"], order="by_vibes"))  # type: ignore[arg-type]

    def test_iter_streams_rather_than_materializing(self, cfg: Config):
        # The streaming API is the contract; the list version is a convenience
        # wrapper. They must agree.
        streamed = [r["review_id"] for r in L.iter_reviews_for(cfg, ["b001", "b003"])]
        assert sorted(streamed) == ["r001", "r002", "r004"]

    def test_small_batch_size_does_not_change_the_result(self, cfg: Config):
        # Batch size is a memory knob, never a correctness knob.
        one_at_a_time = [
            r["review_id"] for r in L.iter_reviews_for(cfg, ["b001", "b003"], batch_rows=1)
        ]
        assert sorted(one_at_a_time) == ["r001", "r002", "r004"]


class TestLoadTips:
    def test_only_rows_for_the_requested_businesses(self, cfg: Config):
        got = L.load_tips_for(cfg, ["b003"])
        assert len(got) == 1
        assert got[0]["text"] == "Second floor is the quietest."

    def test_projection(self, cfg: Config):
        assert set(L.load_tips_for(cfg, ["b003"])[0]) == set(L.TIP_COLUMNS)

    def test_max_per_business(self, cfg: Config):
        got = L.load_tips_for(cfg, ["b001", "b002", "b003"], max_per_business=1)
        assert len(got) == 3


class TestCountReviews:
    def test_counts_rows_not_the_yelp_review_count_field(self, cfg: Config):
        # b001 has review_count=312 in business.json but two rows in
        # review.json. Conflating the two would overstate how much review text
        # the business documents can draw on.
        counts = L.count_reviews_by_business(cfg, ["b001", "b002"])
        assert counts == {"b001": 2, "b002": 1}

    def test_counts_every_business_when_no_scope_given(self, cfg: Config):
        counts = L.count_reviews_by_business(cfg)
        assert counts["b001"] == 2
        assert set(counts) == {"b001", "b002", "b003", "b005", "b006"}

    def test_empty_scope_short_circuits(self, cfg: Config):
        assert L.count_reviews_by_business(cfg, []) == {}


class TestSmallHelpers:
    def test_parse_categories(self):
        assert L.parse_categories("Coffee & Tea, Cafes") == ("Coffee & Tea", "Cafes")
        assert L.parse_categories(None) == ()
        assert L.parse_categories("") == ()
        assert L.parse_categories(" Ramen ,, Japanese ") == ("Ramen", "Japanese")

    def test_median_or_none_never_fakes_a_zero(self):
        # A zero here would read as a measured median in the report.
        assert L.median_or_none([]) is None
        assert L.median_or_none([1, 2, 3]) == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Exploration report
# ─────────────────────────────────────────────────────────────────────────────


class TestPredicates:
    """These delegate to `concierge.data.attributes` / `concierge.data.hours`
    when those are importable, and to a structural fallback otherwise. The
    assertions below hold either way — if a shared module ever disagrees with
    one of them, that is a contract problem worth failing on."""

    def test_usable_hours(self, raw_by_id):
        assert explore.has_usable_hours(raw_by_id["b001"]["hours"])
        assert explore.has_usable_hours(raw_by_id["b003"]["hours"])  # days missing, still usable
        assert explore.has_usable_hours(raw_by_id["b007"]["hours"])  # "0:0-0:0" == open 24h
        assert explore.has_usable_hours(raw_by_id["b002"]["hours"])  # overnight "17:0-2:0"
        assert not explore.has_usable_hours(raw_by_id["b005"]["hours"])  # null
        assert not explore.has_usable_hours({})
        assert not explore.has_usable_hours(None)

    def test_hours_field_is_weaker_than_usable_hours(self):
        # The gap between the two is a reportable finding: data that looks
        # present and cannot answer "open past 10pm".
        unparseable = {"Monday": "whenever we feel like it"}
        assert explore.has_hours_field(unparseable)
        assert not explore.has_usable_hours(unparseable)
        assert not explore.has_hours_field(None)
        assert not explore.has_hours_field({})

    def test_none_as_a_string_counts_as_missing(self, raw_by_id):
        # b008 has OutdoorSeating: "None" — a *string*. Counting that as data
        # would inflate every coverage number in the report.
        attributes = raw_by_id["b008"]["attributes"]
        assert "OutdoorSeating" in attributes
        assert not explore.raw_attribute_present(attributes, "OutdoorSeating")
        assert not explore.attribute_present(raw_by_id["b008"], "OutdoorSeating")
        assert explore.attribute_present(raw_by_id["b008"], "NoiseLevel")

    def test_false_is_data(self, raw_by_id):
        # "has outdoor seating: no" is an answer. Only absence is missing.
        assert explore.raw_attribute_present({"OutdoorSeating": "False"}, "OutdoorSeating")
        assert explore.attribute_present(raw_by_id["b002"], "OutdoorSeating")  # "False"

    def test_null_attributes_are_all_missing(self, raw_by_id):
        assert not explore.raw_attribute_present(raw_by_id["b005"]["attributes"], "WiFi")
        assert not explore.attribute_present(raw_by_id["b005"], "WiFi")

    def test_stringified_literals_are_data(self, raw_by_id):
        # "u'free'" and "'quiet'" are both real values, differently mangled.
        assert explore.attribute_present(raw_by_id["b001"], "WiFi")
        assert explore.attribute_present(raw_by_id["b003"], "NoiseLevel")

    def test_constraint_answerable_needs_hours_and_an_amenity(self, raw_by_id):
        assert explore.is_constraint_answerable(raw_by_id["b001"])  # hours + wifi + noise
        assert not explore.is_constraint_answerable(raw_by_id["b005"])  # no hours, no attrs
        # b007 has hours and NoiseLevel, so a constraint query can be evaluated.
        assert explore.is_constraint_answerable(raw_by_id["b007"])
        # b004 has hours but only a price attribute: no amenity to match on.
        assert not explore.is_constraint_answerable(raw_by_id["b004"])


class TestMeasureCoverage:
    def test_fractions_over_the_records_given(self, raw_businesses):
        coverage = explore.measure_coverage(raw_businesses)
        # NoiseLevel is present on b001, b002, b003, b006, b007, b008 = 6 of 8.
        assert coverage.headline(["NoiseLevel"])["NoiseLevel"] == pytest.approx(6 / 8)
        assert coverage.source

    def test_rates_are_fractions_not_counts(self, raw_businesses):
        coverage = explore.measure_coverage(raw_businesses)
        assert all(0.0 <= v <= 1.0 for v in coverage.rates.values())

    def test_a_key_no_business_has_is_a_real_zero(self, raw_businesses):
        coverage = explore.measure_coverage(raw_businesses)
        assert coverage.headline(["NotAnAttribute"])["NotAnAttribute"] == 0.0

    def test_empty_input_is_labelled_not_zero_percent(self):
        coverage = explore.measure_coverage([])
        assert coverage.source == "no records"
        assert coverage.complete is False
        assert coverage.rates == {}

    def test_coverage_result_in_counts_is_rejected(self, monkeypatch, raw_businesses):
        """A shared `coverage()` returning counts instead of fractions must be
        refused, not divided by a guessed denominator — reinterpreting another
        module's units is how a wrong number ends up looking right."""

        class FakeAttributes:
            @staticmethod
            def coverage(records):
                return {"NoiseLevel": 999}

        monkeypatch.setattr(explore, "_attributes", FakeAttributes)
        coverage = explore.measure_coverage(raw_businesses)
        assert "fallback" in coverage.source
        assert coverage.complete is False
        assert coverage.rates["NoiseLevel"] == pytest.approx(6 / 8)

    def test_valid_coverage_result_is_used_and_attributed(self, monkeypatch, raw_businesses):
        class FakeAttributes:
            @staticmethod
            def coverage(records):
                return {"NoiseLevel": 0.5}

        monkeypatch.setattr(explore, "_attributes", FakeAttributes)
        coverage = explore.measure_coverage(raw_businesses)
        assert coverage.rates == {"NoiseLevel": 0.5}
        assert coverage.complete is True
        assert coverage.source == "concierge.data.attributes.coverage"

    def test_a_keyed_coverage_signature_also_works(self, monkeypatch, raw_businesses):
        # Tolerated so a later revision of the shared module that takes an
        # explicit key list does not silently downgrade this report.
        class FakeAttributes:
            @staticmethod
            def coverage(records, keys):
                return {key: 0.25 for key in keys}

        monkeypatch.setattr(explore, "_attributes", FakeAttributes)
        coverage = explore.measure_coverage(raw_businesses)
        assert coverage.complete is True
        assert coverage.headline()["WiFi"] == 0.25

    def test_a_broken_shared_module_does_not_break_the_report(self, monkeypatch, raw_businesses):
        class FakeAttributes:
            @staticmethod
            def coverage(records):
                raise RuntimeError("still being written")

        monkeypatch.setattr(explore, "_attributes", FakeAttributes)
        coverage = explore.measure_coverage(raw_businesses)
        assert "fallback" in coverage.source
        assert coverage.rates["NoiseLevel"] == pytest.approx(6 / 8)

    def test_fallback_list_contains_no_nested_container_keys(self):
        # Yelp nests Ambience/BusinessParking as stringified dicts, which the
        # attributes module flattens to dotted leaves. Asking for the container
        # key would print a confident 0% for populated data.
        assert "Ambience" not in explore.FALLBACK_ATTRIBUTES
        assert "BusinessParking" not in explore.FALLBACK_ATTRIBUTES
        assert set(explore.HEADLINE_ATTRIBUTES) <= set(explore.FALLBACK_ATTRIBUTES)


class TestBuildReport:
    @pytest.fixture
    def report(self, raw_businesses) -> explore.ExploreReport:
        return explore.build_report(
            raw_businesses,
            config_fingerprint="testfp",
            source_file="tests/fixtures/businesses.json",
            drop_closed=True,
        )

    def test_open_and_closed_are_counted_separately(self, report):
        assert report.n_businesses_raw == 8
        assert report.n_open == 7  # b004 is closed
        assert report.n_closed == 1

    def test_closed_businesses_are_excluded_from_coverage_stats(self, report):
        # b004 carries only a price attribute; including it would drag every
        # other fill rate down for a business that can never be returned.
        # Six of the seven open fixtures have hours; only b005 has hours: null.
        assert report.n_with_usable_hours == 6
        assert report.n_with_hours_field == 6
        assert report.overall_attribute_fill["NoiseLevel"] == pytest.approx(6 / 7)

    def test_fixture_cities_are_unassigned(self, report):
        # Metroville and Othertown are synthetic, so no curated definition
        # claims them. That is the point of the "(unassigned)" row: it is how a
        # wrong set of metro definitions announces itself instead of hiding.
        assert [m.metro for m in report.metros] == [explore.UNASSIGNED]
        assert report.unassigned is not None
        assert report.unassigned.n_businesses == 7
        assert report.unassigned.n_closed == 1

    def test_unassigned_share_is_reported_in_the_notes(self, report):
        assert any("matched no metro definition" in note for note in report.notes)

    def test_top_cities_let_the_owner_fix_the_definitions(self, report):
        cities = dict(report.unassigned_top_cities)
        assert cities["Metroville, AZ"] == 6
        assert cities["Othertown, NV"] == 1

    def test_review_rows_are_absent_unless_measured(self, report):
        assert report.review_rows_measured is False
        assert all(m.review_rows_total is None for m in report.metros)
        assert any("NOT measured" in note for note in report.notes)
        # And they must not be silently rendered as zero.
        assert explore._num(None) == "not measured"

    def test_review_rows_when_measured(self, raw_businesses):
        report = explore.build_report(
            raw_businesses,
            config_fingerprint="testfp",
            source_file="x",
            drop_closed=True,
            review_rows={"b001": 2, "b002": 1, "b003": 1, "b005": 1, "b006": 1},
        )
        assert report.review_rows_measured is True
        metro = report.metros[0]
        assert metro.review_rows_total == 6
        # b007 and b008 have no reviews, so the median over 7 businesses is 1.
        assert metro.review_rows_median == 1.0

    def test_yelp_review_count_is_kept_distinct_from_rows(self, report):
        metro = report.metros[0]
        # Sum of review_count over the 7 open fixtures.
        assert metro.yelp_review_count_total == 312 + 880 + 45 + 150 + 540 + 77 + 210
        assert metro.yelp_review_count_median == 210.0

    def test_answerable_count(self, report):
        metro = report.metros[0]
        # b001,b002,b003,b006,b007,b008 have hours + an amenity; b005 has neither.
        assert metro.n_constraint_answerable == 6
        assert metro.answerable_share == pytest.approx(6 / 7)

    def test_shares_do_not_divide_by_zero(self):
        empty = explore.MetroStats(
            metro="x",
            n_businesses=0,
            n_closed=0,
            n_cities=0,
            top_cities=(),
            yelp_review_count_total=0,
            yelp_review_count_median=None,
            review_rows_total=None,
            review_rows_median=None,
            n_with_hours_field=0,
            n_with_usable_hours=0,
            n_constraint_answerable=0,
        )
        assert empty.hours_share == 0.0
        assert empty.hours_field_share == 0.0
        assert empty.answerable_share == 0.0

    def test_metros_are_sorted_largest_first(self, raw_businesses):
        # Relabel the fixtures into two real metros so the ordering is visible.
        relabelled = [dict(r) for r in raw_businesses]
        for record in relabelled:
            record["city"], record["state"] = (
                ("Boise", "ID") if record["business_id"] == "b006" else ("Tucson", "AZ")
            )
        report = explore.build_report(
            relabelled, config_fingerprint="fp", source_file="x", drop_closed=True
        )
        assert [m.metro for m in report.metros] == ["Tucson", "Boise"]
        assert report.unassigned is None
        assert report.unassigned_top_cities == ()

    def test_provenance_of_every_number_is_recorded(self, report):
        # A report whose numbers cannot be traced to the code that produced
        # them is not usable at a checkpoint.
        assert report.attribute_source
        assert report.hours_source
        assert any(report.attribute_source in note for note in report.notes)
        assert any(report.hours_source in note for note in report.notes)

    def test_incomplete_coverage_is_flagged_with_a_caveat(self, monkeypatch, raw_businesses):
        # Without the attributes module the table cannot see nested leaves, and
        # the report has to say so instead of looking complete.
        monkeypatch.setattr(explore, "_attributes", None)
        monkeypatch.setattr(explore, "_PARSE_ATTRIBUTES", None)
        report = explore.build_report(
            raw_businesses, config_fingerprint="fp", source_file="x", drop_closed=True
        )
        assert report.attribute_fill_is_complete is False
        assert any("Ambience.casual" in note for note in report.notes)
        # The headline four are flat keys, so they are still correct.
        assert report.overall_attribute_fill["NoiseLevel"] == pytest.approx(6 / 7)

    def test_report_is_pure_and_json_serializable(self, report):
        # The JSON artifact is the machine-readable twin of the printed table;
        # if it cannot serialize, the artifact silently stops being written.
        from dataclasses import asdict

        payload = json.dumps(asdict(report))
        assert "(unassigned)" in payload


class TestRendering:
    @pytest.fixture
    def report(self, raw_businesses) -> explore.ExploreReport:
        relabelled = [dict(r) for r in raw_businesses]
        for record in relabelled:
            record["city"], record["state"] = (
                ("Boise", "ID") if record["business_id"] == "b006" else ("Tucson", "AZ")
            )
        return explore.build_report(
            relabelled, config_fingerprint="fp", source_file="x", drop_closed=True
        )

    def test_rich_output_names_the_metros_and_the_tradeoff(self, report):
        console = Console(record=True, width=200)
        explore.render(report, console)
        text = console.export_text()
        assert "Tucson" in text
        assert "Boise" in text
        assert "The tradeoff" in text
        # The owner decision must be stated, not implied.
        assert "data.metro" in text

    def test_markdown_has_both_tables_and_the_caveats(self, report):
        md = explore.to_markdown(report)
        assert "# Dataset exploration" in md
        assert "## Per-metro scope" in md
        assert "## Constraint coverage" in md
        assert "## Caveats" in md
        assert "Tucson" in md

    def test_markdown_states_where_the_numbers_came_from(self, report):
        md = explore.to_markdown(report)
        assert "attribute fill computed by:" in md
        assert "hours computed by:" in md

    def test_tradeoff_text_handles_an_all_unassigned_file(self, raw_businesses):
        report = explore.build_report(
            raw_businesses, config_fingerprint="fp", source_file="x", drop_closed=True
        )
        text = explore._tradeoff_text(report)
        assert "do not fit this file" in text

    def test_artifacts_are_written_under_the_config_fingerprint(self, report, tmp_path: Path):
        json_path, md_path = explore.write_artifacts(report, tmp_path)
        assert json_path.parent.name == "fp"  # D008: traceable to the config
        payload = json.loads(json_path.read_text())
        assert payload["config_fingerprint"] == "fp"
        assert payload["metros"][0]["metro"] == "Tucson"
        assert md_path.read_text().startswith("# Dataset exploration")


class TestExploreCli:
    def _args(self, **overrides) -> argparse.Namespace:
        base = {
            "config": "configs/base.yaml",
            "out_dir": "artifacts/explore",
            "top_cities": 12,
            "count_reviews": False,
            "no_artifacts": True,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_runs_against_the_fixture_corpus(self, cfg: Config):
        console = Console(record=True, width=200)
        assert explore.run(cfg, self._args(), console) == 0
        assert "Dataset exploration" in console.export_text()

    def test_reports_every_metro_even_though_config_sets_one(self, cfg: Config):
        # Exploration happens *before* the metro is chosen, so a configured
        # metro must not narrow the report.
        console = Console(record=True, width=200)
        explore.run(cfg, self._args(), console)
        assert "Othertown" in console.export_text()

    def test_count_reviews_streams_the_review_file(self, cfg: Config):
        console = Console(record=True, width=200)
        assert explore.run(cfg, self._args(count_reviews=True), console) == 0
        assert "review.json" in console.export_text()

    def test_missing_data_exits_nonzero_with_instructions(self, config: Config, tmp_path: Path):
        console = Console(record=True, width=200)
        cfg = _with_raw(config, tmp_path / "absent")
        assert explore.run(cfg, self._args(), console) == 2
        assert "yelp.com/dataset" in console.export_text()

    def test_writes_artifacts_when_asked(self, cfg: Config, tmp_path: Path):
        out = tmp_path / "artifacts"
        console = Console(record=True, width=200)
        code = explore.run(cfg, self._args(no_artifacts=False, out_dir=str(out)), console)
        assert code == 0
        written = sorted(p.name for p in out.rglob("*") if p.is_file())
        assert written == ["dataset_report.json", "dataset_report.md"]

    def test_parser_defaults(self):
        args = explore._build_parser().parse_args([])
        assert args.config == "configs/base.yaml"
        assert args.count_reviews is False  # the 5 GB scan is opt-in


# ─────────────────────────────────────────────────────────────────────────────
# Tests that need the real download
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.needs_data
class TestAgainstRealDataset:
    """Skipped unless the Yelp download is in `data.raw_dir`.

    These check the claims that only the real file can check: that the curated
    metro definitions actually match it, and that the streaming read holds up
    at 5 GB.
    """

    @pytest.fixture
    def real_cfg(self) -> Config:
        cfg = load_config("configs/base.yaml")
        if not L.raw_path(cfg, L.BUSINESS_FILE).is_file():
            pytest.skip("Yelp Open Dataset not present in data/raw")
        return cfg

    def test_metro_definitions_claim_most_of_the_file(self, real_cfg: Config):
        records = L.load_businesses(real_cfg, L.ALL_METROS, include_closed=True)
        report = explore.build_report(
            records,
            config_fingerprint=real_cfg.fingerprint(),
            source_file=str(L.raw_path(real_cfg, L.BUSINESS_FILE)),
            drop_closed=real_cfg.data.drop_closed,
        )
        unassigned = report.unassigned
        share = unassigned.n_businesses / report.n_open if unassigned else 0.0
        # Not a tuned threshold: a definition set that leaves most of the
        # dataset unclaimed is simply wrong, and this is the assertion that
        # says so. The real share is reported by `make explore`, not asserted.
        assert share < 0.5, (
            f"{share:.1%} of open businesses matched no metro definition — "
            "extend the city lists in concierge/data/load.py using the "
            "per-city table from `make explore`."
        )

    def test_review_streaming_stays_bounded(self, real_cfg: Config):
        ids = L.load_business_ids(real_cfg, L.ALL_METROS)[:50]
        seen = 0
        for _ in L.iter_reviews_for(real_cfg, ids, batch_rows=1000):
            seen += 1
        assert seen >= 0  # the assertion is that this completes without OOM
