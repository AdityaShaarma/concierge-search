"""Tests for the business document builder.

The document is the only thing BM25, the embedder, and the cross-encoder ever
see, so these tests guard the properties the rest of the pipeline assumes:

  * byte-identical output for identical input (index artifacts are keyed by the
    config hash, so a document that changes between runs makes a "reproducible"
    artifact a fiction);
  * attributes read as English, and negative attributes never appear — BM25 has
    no notion of negation, so "no outdoor seating" would match a request *for*
    outdoor seating;
  * the configured caps are actually enforced, not approximately;
  * truncation never cuts a word in half;
  * the structured signal (name, categories, attribute phrases) precedes the
    review prose, because the reranker reads the head and drops the tail;
  * a business with no attributes and no hours (fixture b005) still produces a
    usable document rather than an empty string.

Businesses are built from the raw fixtures through
`concierge.data.attributes.parse_attributes`, the same path the real data build
takes, so these tests also cover the seam between the two modules rather than a
convenient idealization of it.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from concierge.data.attributes import PRICE_ATTRIBUTE, parse_attributes
from concierge.data.documents import (
    attribute_phrases,
    build_document,
    build_documents,
)
from concierge.types import Business

# ─────────────────────────────────────────────────────────────────────────────
# Test-local scaffolding
# ─────────────────────────────────────────────────────────────────────────────


def business_from_fixture(raw: dict) -> Business:
    """Build a `Business` from a raw fixture record.

    `hours` is left empty on purpose: hours are parsed by another module and,
    by design, never reach the document (see `test_hours_are_not_in_the_document`).
    """
    categories = tuple(c.strip() for c in (raw.get("categories") or "").split(",") if c.strip())
    return Business(
        business_id=raw["business_id"],
        name=raw["name"],
        city=raw["city"],
        state=raw["state"],
        latitude=raw.get("latitude"),
        longitude=raw.get("longitude"),
        stars=raw["stars"],
        review_count=raw["review_count"],
        categories=categories,
        attributes=parse_attributes(raw.get("attributes")),
        hours={},
        is_open=bool(raw["is_open"]),
    )


def _reviews_for(raw_reviews: list[dict], business_id: str) -> list[dict]:
    return [r for r in raw_reviews if r["business_id"] == business_id]


def _tips_for(raw_tips: list[dict], business_id: str) -> list[dict]:
    return [t for t in raw_tips if t["business_id"] == business_id]


def _with_documents(config, **overrides):
    """A copy of the config with `documents` fields overridden."""
    return config.model_copy(update={"documents": config.documents.model_copy(update=overrides)})


def _long_review(review_id: str, word: str, n_words: int, useful: int, date: str) -> dict:
    """A review made of one repeated distinctive word, for truncation tests."""
    return {
        "review_id": review_id,
        "business_id": "b001",
        "useful": useful,
        "date": date,
        "text": " ".join([word] * n_words) + ".",
    }


def phrases_of(doc_text: str) -> str:
    """The attribute-phrase line of a document (line 2), or ""."""
    lines = doc_text.splitlines()
    return lines[1] if len(lines) > 1 and not lines[1].startswith(("Tips:", "Reviews:")) else ""


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_gives_byte_identical_text(self, raw_by_id, raw_reviews, raw_tips, config):
        """Index artifacts are namespaced by the config hash. If the document
        text for a fixed business and config can differ between runs, that
        namespacing promises reproducibility it cannot deliver."""
        business = business_from_fixture(raw_by_id["b001"])
        reviews = _reviews_for(raw_reviews, "b001")
        tips = _tips_for(raw_tips, "b001")

        first = build_document(business, reviews, tips, config)
        second = build_document(business, reviews, tips, config)

        assert first.text == second.text
        assert (first.n_reviews_used, first.n_tips_used, first.truncated) == (
            second.n_reviews_used,
            second.n_tips_used,
            second.truncated,
        )

    def test_input_order_does_not_change_the_document(
        self, raw_by_id, raw_reviews, raw_tips, config
    ):
        """Reviews arrive from a group-by whose order is not guaranteed, so the
        selection rule must impose a total order of its own."""
        business = business_from_fixture(raw_by_id["b001"])
        reviews = _reviews_for(raw_reviews, "b001")
        tips = _tips_for(raw_tips, "b001")

        forward = build_document(business, reviews, tips, config)
        reversed_ = build_document(
            business,
            list(reversed(reviews)),
            list(reversed(tips)),
            config,
        )
        assert forward.text == reversed_.text

    def test_random_selection_is_seed_stable(self, raw_by_id, config):
        """`random` exists only as an ablation baseline for the selection rule,
        so it still has to rebuild identically."""
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [
            _long_review(f"r{i}", f"word{i}", 6, useful=1, date=f"2021-01-0{i} 10:00:00")
            for i in range(1, 6)
        ]
        cfg = _with_documents(config, review_selection="random", max_reviews=2)

        first = build_document(business, reviews, (), cfg)
        second = build_document(business, reviews, (), cfg)
        assert first.text == second.text


# ─────────────────────────────────────────────────────────────────────────────
# Caps
# ─────────────────────────────────────────────────────────────────────────────


class TestCaps:
    def test_max_chars_is_never_exceeded(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [
            _long_review(f"r{i}", "alpha", 200, useful=10 - i, date=f"2021-01-0{i} 10:00:00")
            for i in range(1, 5)
        ]
        for max_chars in (40, 120, 300, 1800):
            cfg = _with_documents(config, max_chars=max_chars)
            doc = build_document(business, reviews, (), cfg)
            assert len(doc.text) <= max_chars, f"overran the cap at max_chars={max_chars}"

    def test_max_reviews_and_max_tips_are_respected(self, raw_by_id, raw_reviews, config):
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [
            _long_review(f"r{i}", f"word{i}", 4, useful=i, date=f"2021-01-0{i} 10:00:00")
            for i in range(1, 7)
        ]
        tips = [
            {
                "business_id": "b001",
                "text": f"Tip number {i}.",
                "compliment_count": i,
                "date": f"2021-02-0{i} 10:00:00",
            }
            for i in range(1, 6)
        ]
        cfg = _with_documents(config, max_reviews=2, max_tips=1, max_chars=1800)
        doc = build_document(business, reviews, tips, cfg)

        assert doc.n_reviews_used == 2
        assert doc.n_tips_used == 1
        # The cap is selection, not truncation: nothing was shortened to fit.
        assert doc.truncated is False

    def test_counts_reflect_what_actually_survived_the_cap(self, raw_by_id, config):
        """n_reviews_used must count snippets in the text, not snippets chosen —
        it feeds the "why this result" UI and the failure analysis."""
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [
            _long_review(f"r{i}", "beta", 100, useful=10 - i, date=f"2021-01-0{i} 10:00:00")
            for i in range(1, 5)
        ]
        # Enough room for the head and attribute phrases, not for four reviews.
        cfg = _with_documents(config, max_chars=140, max_reviews=4)
        doc = build_document(business, reviews, (), cfg)

        assert len(doc.text) <= 140
        assert doc.n_reviews_used <= 4
        assert doc.truncated is True
        if doc.n_reviews_used == 0:
            assert "Reviews:" not in doc.text
        else:
            # Every counted snippet is really in the text.
            assert doc.text.count("beta") >= doc.n_reviews_used

    def test_zero_caps_leave_only_structured_signal(self, raw_by_id, raw_reviews, raw_tips, config):
        """max_reviews=0 is the ablation that asks "do reviews help at all?" —
        it must produce a document, not a crash."""
        business = business_from_fixture(raw_by_id["b001"])
        cfg = _with_documents(config, max_reviews=0, max_tips=0)
        doc = build_document(
            business,
            _reviews_for(raw_reviews, "b001"),
            _tips_for(raw_tips, "b001"),
            cfg,
        )
        assert doc.n_reviews_used == 0
        assert doc.n_tips_used == 0
        assert "Reviews:" not in doc.text and "Tips:" not in doc.text
        assert "Lantern Coffee" in doc.text


# ─────────────────────────────────────────────────────────────────────────────
# Truncation behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestTruncation:
    def test_never_truncates_mid_word(self, raw_by_id, config):
        """Half a word is noise: it matches no BM25 term and tokenizes into
        subword junk for the embedder."""
        business = business_from_fixture(raw_by_id["b001"])
        # Distinctive, long, equal-length words so a mid-word cut is obvious.
        reviews = [
            _long_review("r1", "sarsaparilla", 60, useful=5, date="2021-01-01 10:00:00"),
            _long_review("r2", "pumpernickel", 60, useful=4, date="2021-01-02 10:00:00"),
        ]
        tips = [
            {
                "business_id": "b001",
                "text": "Mezzanine seating upstairs.",
                "compliment_count": 2,
                "date": "2021-03-01 10:00:00",
            }
        ]

        allowed = set()
        for source in [
            business.name,
            " ".join(business.categories),
            " ".join(attribute_phrases(business.attributes)),
            "$$ Tips: Reviews:",
            *[r["text"] for r in reviews],
            *[t["text"] for t in tips],
        ]:
            allowed.update(_words(source))

        for max_chars in range(30, 400, 7):
            cfg = _with_documents(config, max_chars=max_chars)
            doc = build_document(business, reviews, tips, cfg)
            unknown = _words(doc.text) - allowed
            assert not unknown, f"mid-word cut at max_chars={max_chars}: {sorted(unknown)}"

    def test_prefers_a_sentence_boundary(self, raw_by_id, config):
        """A snippet that ends at a sentence boundary reads like language, which
        is what the cross-encoder was pretrained on."""
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [
            {
                "review_id": "r1",
                "business_id": "b001",
                "useful": 9,
                "date": "2021-01-01 10:00:00",
                "text": "Great espresso. Terrible parking. The patio is sunny all afternoon.",
            }
        ]
        # Room for the first two sentences of the review but not the third.
        cfg = _with_documents(config, max_chars=110, max_reviews=1, max_tips=0)
        doc = build_document(business, reviews, (), cfg)

        if "Reviews:" in doc.text:
            tail = doc.text.split("Reviews: ", 1)[1]
            assert tail.endswith((".", "!", "?")), f"snippet ended mid-sentence: {tail!r}"
            assert "patio" not in tail  # the third sentence did not fit

    def test_truncated_flag_is_false_when_everything_fits(
        self, raw_by_id, raw_reviews, raw_tips, config
    ):
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(
            business,
            _reviews_for(raw_reviews, "b001"),
            _tips_for(raw_tips, "b001"),
            config,
        )
        assert doc.truncated is False
        assert doc.n_reviews_used == 2
        assert doc.n_tips_used == 1

    def test_truncated_flag_is_set_when_content_is_cut(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [_long_review("r1", "cardamom", 120, useful=3, date="2021-01-01 10:00:00")]
        cfg = _with_documents(config, max_chars=150)
        doc = build_document(business, reviews, (), cfg)
        assert doc.truncated is True

    def test_one_long_review_does_not_crowd_out_the_others(self, raw_by_id, config):
        """Reviews disagree with each other usefully, so breadth beats depth:
        a single enormous review must not swallow the whole window."""
        business = business_from_fixture(raw_by_id["b001"])
        reviews = [
            _long_review("r1", "tamarind", 400, useful=9, date="2021-01-01 10:00:00"),
            _long_review("r2", "saffron", 10, useful=8, date="2021-01-02 10:00:00"),
            _long_review("r3", "nutmeg", 10, useful=7, date="2021-01-03 10:00:00"),
        ]
        cfg = _with_documents(config, max_chars=600, max_reviews=3, max_tips=0)
        doc = build_document(business, reviews, (), cfg)

        assert doc.n_reviews_used == 3
        assert "saffron" in doc.text and "nutmeg" in doc.text


# ─────────────────────────────────────────────────────────────────────────────
# Content and ordering
# ─────────────────────────────────────────────────────────────────────────────


class TestContentAndOrdering:
    def test_attribute_phrases_come_before_review_text(
        self, raw_by_id, raw_reviews, raw_tips, config
    ):
        """The ordering claim the whole design rests on: a 512-token
        cross-encoder truncates the tail, so structured signal goes first."""
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(
            business,
            _reviews_for(raw_reviews, "b001"),
            _tips_for(raw_tips, "b001"),
            config,
        )
        text = doc.text
        assert text.index("Lantern Coffee") < text.index("free WiFi")
        assert text.index("free WiFi") < text.index("Reviews:")
        assert text.index("Coffee & Tea") < text.index("Reviews:")
        assert text.index("Tips:") < text.index("Reviews:")

    def test_attributes_read_as_english_not_key_values(self, raw_by_id, config):
        """The pretraining-distribution argument, asserted: no "Key=value" and
        no raw dataset key names in the text."""
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(business, (), (), config)
        assert "=" not in doc.text
        # Dataset key names must not survive into the text. ("WiFi" is absent
        # from this list on purpose: "free WiFi" is how a person writes it.)
        for key in ("NoiseLevel", "OutdoorSeating", "Ambience", PRICE_ATTRIBUTE):
            assert key not in doc.text
        assert "free WiFi" in doc.text  # the phrase, not the key, is present
        assert "quiet" in doc.text
        assert "True" not in doc.text  # a boolean flag says nothing about a venue

    @pytest.mark.parametrize(
        ("business_id", "expected"),
        [("b002", "$"), ("b001", "$$"), ("b008", "$$$$")],
    )
    def test_price_renders_as_dollar_signs(self, raw_by_id, config, business_id, expected):
        """A user types "cheap" or "$$", never "RestaurantsPriceRange2: 2", so
        the price needs lexical surface in the document."""
        business = business_from_fixture(raw_by_id[business_id])
        doc = build_document(business, (), (), config)
        # The price phrase is a standalone token at the end of the phrase line.
        assert re.search(rf"(^|[\s,]){re.escape(expected)}($|[\s,.])", doc.text), doc.text

    def test_price_is_not_duplicated_when_the_phraser_already_rendered_it(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(business, (), (), config, phrase_fn=lambda attrs: ["quiet", "$$"])
        assert doc.text.count("$$") == 1

    def test_hours_are_not_in_the_document(self, raw_by_id, raw_reviews, config):
        """ "Open past 10pm" is arithmetic over HoursInterval. A review that says
        "open late" must not be able to fake a satisfied hard constraint."""
        business = business_from_fixture(raw_by_id["b002"])
        doc = build_document(
            business,
            _reviews_for(raw_reviews, "b002"),
            (),
            config,
        )
        assert "17:0" not in doc.text
        assert "Monday" not in doc.text

    def test_business_with_no_attributes_or_hours_is_still_usable(
        self, raw_by_id, raw_reviews, config
    ):
        """Fixture b005: attributes and hours both null. Sparse attribute
        coverage is the norm in this dataset, so the no-data path is a main
        path, not an edge case."""
        business = business_from_fixture(raw_by_id["b005"])
        doc = build_document(
            business,
            _reviews_for(raw_reviews, "b005"),
            (),
            config,
        )
        assert doc.business_id == "b005"
        assert "No Data Bistro" in doc.text
        assert "Bistros" in doc.text and "French" in doc.text
        assert doc.n_reviews_used == 1
        assert "French bistro" in doc.text
        # No attribute line at all, rather than an empty or placeholder one.
        assert "$" not in doc.text
        assert len(doc.text.splitlines()) == 2

    def test_review_text_from_another_business_is_never_used(self, raw_by_id, raw_reviews, config):
        """Guard against an upstream grouping bug, which would otherwise show up
        as plausible-looking but wrong results."""
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(business, raw_reviews, (), config)
        assert "tonkotsu" not in doc.text  # b002's review
        assert "al pastor" not in doc.text  # b006's review
        assert doc.n_reviews_used == 2


# ─────────────────────────────────────────────────────────────────────────────
# Selection modes
# ─────────────────────────────────────────────────────────────────────────────


class TestSelectionModes:
    def _reviews(self) -> list[dict]:
        return [
            {
                "review_id": "r_old_useful",
                "business_id": "b001",
                "useful": 50,
                "date": "2018-01-01 10:00:00",
                "text": "Durian pancakes are the move.",
            },
            {
                "review_id": "r_new_useless",
                "business_id": "b001",
                "useful": 0,
                "date": "2024-01-01 10:00:00",
                "text": "Lychee soda is new here.",
            },
        ]

    def test_useful_then_recent_prefers_the_upvoted_review(self, raw_by_id, config):
        """A highly-upvoted review explains *why* people go there; the newest
        one is often a one-line complaint about parking."""
        business = business_from_fixture(raw_by_id["b001"])
        cfg = _with_documents(config, review_selection="useful_then_recent", max_reviews=1)
        doc = build_document(business, self._reviews(), (), cfg)
        assert "Durian" in doc.text and "Lychee" not in doc.text

    def test_recent_prefers_the_newest_review(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        cfg = _with_documents(config, review_selection="recent", max_reviews=1)
        doc = build_document(business, self._reviews(), (), cfg)
        assert "Lychee" in doc.text and "Durian" not in doc.text

    def test_all_three_modes_are_implemented(self, raw_by_id, config):
        """All three exist so the selection rule can be ablated rather than
        asserted. None of them may raise."""
        business = business_from_fixture(raw_by_id["b001"])
        for mode in ("useful_then_recent", "recent", "random"):
            cfg = _with_documents(config, review_selection=mode, max_reviews=1)
            doc = build_document(business, self._reviews(), (), cfg)
            assert doc.n_reviews_used == 1, mode

    def test_ties_are_broken_deterministically(self, raw_by_id, config):
        """Equal useful counts and equal timestamps must still order the same
        way on every run, or the corpus is not reproducible."""
        business = business_from_fixture(raw_by_id["b001"])
        tied = [
            {
                "review_id": "ra",
                "business_id": "b001",
                "useful": 3,
                "date": "2022-01-01 10:00:00",
                "text": "Alpha tie.",
            },
            {
                "review_id": "rb",
                "business_id": "b001",
                "useful": 3,
                "date": "2022-01-01 10:00:00",
                "text": "Bravo tie.",
            },
        ]
        cfg = _with_documents(config, max_reviews=1)
        first = build_document(business, tied, (), cfg)
        second = build_document(business, list(reversed(tied)), (), cfg)
        assert first.text == second.text


# ─────────────────────────────────────────────────────────────────────────────
# Batch builder
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildDocuments:
    def test_preserves_input_order(self, raw_businesses, raw_reviews, raw_tips, config):
        """BM25 and FAISS address documents by row number, so document i must
        be business i. A misaligned id map returns nonsense, not an error."""
        businesses = [business_from_fixture(b) for b in raw_businesses]
        reviews_by_business: dict[str, list[dict]] = {}
        for review in raw_reviews:
            reviews_by_business.setdefault(review["business_id"], []).append(review)
        tips_by_business: dict[str, list[dict]] = {}
        for tip in raw_tips:
            tips_by_business.setdefault(tip["business_id"], []).append(tip)

        docs = build_documents(
            businesses,
            reviews_by_business,
            tips_by_business,
            config,
        )
        assert [d.business_id for d in docs] == [b.business_id for b in businesses]
        assert all(d.text for d in docs), "every business must get a non-empty document"

    def test_matches_build_document_one_at_a_time(self, raw_businesses, config):
        """The batch path is a convenience, not a second implementation."""
        businesses = [business_from_fixture(b) for b in raw_businesses]
        batch = build_documents(businesses, {}, {}, config)
        singles = [build_document(b, (), (), config) for b in businesses]
        assert [d.text for d in batch] == [d.text for d in singles]

    def test_missing_reviews_and_tips_are_fine(self, raw_businesses, config):
        businesses = [business_from_fixture(b) for b in raw_businesses]
        docs = build_documents(businesses, None, None, config)
        assert all(d.n_reviews_used == 0 and d.n_tips_used == 0 for d in docs)


# ─────────────────────────────────────────────────────────────────────────────
# Attribute phrasing
# ─────────────────────────────────────────────────────────────────────────────


class TestAttributePhrases:
    """`attribute_phrases` is the half of the document that answers constraint
    queries, so it gets tested directly and not only through a document."""

    def test_phrases_the_fixture_business_as_english(self, raw_by_id):
        attributes = parse_attributes(raw_by_id["b001"]["attributes"])
        assert attribute_phrases(attributes) == [
            "free WiFi",
            "quiet",
            "outdoor seating",
            "casual",
            "street parking",
            "$$",
        ]

    def test_negative_attributes_are_omitted(self, raw_by_id):
        """b002 has OutdoorSeating=False and WiFi="no". Writing "no outdoor
        seating" into the text would make this ramen shop a strong BM25 match
        for "outdoor seating" — BM25 scores terms and cannot see the "no"."""
        attributes = parse_attributes(raw_by_id["b002"]["attributes"])
        phrases = attribute_phrases(attributes)
        assert phrases == ["loud", "$"]
        joined = " ".join(phrases)
        assert "outdoor" not in joined and "WiFi" not in joined

    def test_unknown_attributes_still_read_as_english(self):
        """Yelp has ~40 attributes and adds more; an unmapped one must degrade
        to readable words, not vanish and not appear as a raw key."""
        phrases = attribute_phrases({"DriveThru": True, "AgesAllowed": "21plus"})
        assert "drive through" in phrases
        assert any("21plus" in p and "ages allowed" in p for p in phrases), phrases

    def test_nested_attributes_use_their_parent_for_sense(self):
        phrases = attribute_phrases(
            {
                "Ambience.romantic": True,
                "BusinessParking.valet": True,
                "GoodForMeal.brunch": True,
                "Ambience.touristy": False,
            }
        )
        assert "romantic" in phrases
        assert "valet parking" in phrases  # "valet" alone is not an amenity
        assert "good for brunch" in phrases
        assert "touristy" not in phrases  # False leaf, negation rule

    def test_weekday_attributes_are_skipped(self):
        """A stray "friday" in the text is a false match for "open friday
        night", which is an hours constraint the filter answers exactly."""
        assert attribute_phrases({"BestNights.friday": True}) == []

    def test_phrase_order_is_independent_of_key_order(self):
        """Attribute order comes from the raw record, so phrase order must not:
        a document that reorders between rebuilds breaks the config-hash
        reproducibility guarantee."""
        forward = {"WiFi": "free", "NoiseLevel": "quiet", "OutdoorSeating": True}
        backward = dict(reversed(list(forward.items())))
        assert attribute_phrases(forward) == attribute_phrases(backward)

    def test_price_is_last(self):
        phrases = attribute_phrases({PRICE_ATTRIBUTE: 3, "WiFi": "free"})
        assert phrases[-1] == "$$$"

    @pytest.mark.parametrize(("level", "expected"), [(1, "$"), (2, "$$"), (3, "$$$"), (4, "$$$$")])
    def test_every_price_level_renders(self, level: int, expected: str):
        assert attribute_phrases({PRICE_ATTRIBUTE: level}) == [expected]

    @pytest.mark.parametrize("bad", [0, 5, "", "free", None, True])
    def test_unusable_price_renders_nothing(self, bad: Any):
        """An out-of-range or non-numeric price is no data, and no data must
        never become a confident "$"."""
        assert attribute_phrases({PRICE_ATTRIBUTE: bad}) == []

    def test_no_attributes_gives_no_phrases(self, raw_by_id):
        assert attribute_phrases(parse_attributes(raw_by_id["b005"]["attributes"])) == []


class TestPhraseFnInjection:
    """`phrase_fn` is the seam that lets document phrasing be ablated."""

    def test_an_injected_phraser_replaces_the_default(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(business, (), (), config, phrase_fn=lambda attrs: ["key=value style"])
        assert "key=value style" in doc.text
        assert "free WiFi" not in doc.text

    def test_price_is_not_duplicated_when_the_phraser_renders_it(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(business, (), (), config, phrase_fn=lambda attrs: ["quiet", "$$"])
        assert doc.text.count("$$") == 1

    def test_price_survives_a_phraser_that_only_renders_price(self, config):
        """The price must not be lost just because it arrived from the phraser
        rather than from the attributes dict."""
        business = Business(
            business_id="x1",
            name="Hand Built",
            city="Metroville",
            state="AZ",
            latitude=None,
            longitude=None,
            stars=4.0,
            review_count=1,
            categories=("Cafes",),
            attributes={"WiFi": "free"},
            hours={},
            is_open=True,
        )
        doc = build_document(business, (), (), config, phrase_fn=lambda attrs: ["$$$"])
        assert "$$$" in doc.text

    def test_duplicate_phrases_are_said_once(self, raw_by_id, config):
        business = business_from_fixture(raw_by_id["b001"])
        doc = build_document(
            business, (), (), config, phrase_fn=lambda attrs: ["quiet", "quiet", "cozy"]
        )
        assert doc.text.count("quiet") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9&$']+")


def _words(text: str) -> set[str]:
    """Words in `text`, punctuation stripped, for the mid-word-cut check."""
    return set(_WORD_RE.findall(text))
