"""Stage 2, third path: candidate generation from the hard constraints alone.

Why this module exists
----------------------
Stage 1 splits a query into *soft text* and *hard constraints*, and Stage 2's
two retrievers only ever see the soft text. For some queries there is almost no
soft text left. Measured on the fixture corpus with the rule parser and the
lexical retriever alone:

    "somewhere with a patio"  -> semantic_text "somewhere"  -> nothing retrieved
    "cheap place"             -> semantic_text "place"      -> nothing retrieved
    "free wifi"               -> semantic_text "free wifi"  -> four retrieved

In the first two cases the constraints were parsed *correctly*
(``OutdoorSeating=True``; ``price_max``). They simply never got applied, because
`concierge.retrieval.filters` filters a candidate list and there was no
candidate list to filter. The query returns an empty page.

That matters more than the raw count suggests. ``constraint`` is one of the four
gold query types, and CLAUDE.md's own example of one — "open after 10pm with
outdoor seating" — is almost pure constraint. So the failure targets a whole row
of the per-query-type breakdown, and it fails by returning *nothing* rather than
something wrong, which is the shape of failure least likely to be noticed in a
demo.

This module supplies the missing candidates: given a `ParsedQuery`, it returns
the businesses that the constraint checks actually admit, ordered by a quality
prior. It is the "constraint-only retriever" of DECISIONS.md D224.

Which of D224's three candidate fixes this is, and why
------------------------------------------------------
D224 listed three:

(a) *Fall back to constraint-driven candidate generation when retrieval returns
    too few.*
(b) *Stop stripping constraint phrases out of ``semantic_text``* and let fusion
    absorb the redundancy.
(c) *Keep a constraint-only retrieval path fused by RRF* alongside BM25 and
    dense.

This module is **(a), built with (c)'s interface.** The reasoning, because it is
the part worth defending:

*Against (b).* Stripping exists so that a constraint is not also scored as
lexical evidence — a café whose *reviews* say "quiet" would otherwise be
rewarded for a property the filters already check arithmetically, while a café
with ``NoiseLevel=quiet`` and no reviewer who mentions it would be rewarded for
nothing. Un-stripping trades a measured, contained defect for a diffuse
double-counting problem, and it would not even fix the worst case here: the
documents render the attribute as "outdoor seating", so leaving the word "patio"
in the query buys no lexical match at all.

*Against (c) as the wiring.* RRF's premise is that each input list is an
independent *estimate of relevance to this query*. A constraint list is not
that. It is the filters' own admissible set, ordered by a query-independent
prior. Fusing it in unconditionally would:

  1. let a high-prior business that merely satisfies a constraint outscore a
     business that one retriever is genuinely confident about — the dilution the
     task of this module explicitly has to avoid;
  2. count the constraint three times — once here, once in
     `filters.apply_filters`, which already drops violators, and once in the
     Stage 4 ``n_constraints_satisfied`` feature — while also smuggling a
     popularity prior into Stage 2, where `concierge.ltr.features` already
     carries rating and review count as their own features;
  3. change the fused score of *every* constrained query, so the
     ``bm25 -> hybrid_rrf`` ablation step would no longer isolate "+dense".

*For (a).* The defect is an empty candidate list. Repairing exactly that — and
nothing else — is the change whose effect can be attributed. So `backfill`
appends constraint-only candidates *below* everything retrieval produced,
scored so that they cannot displace a retrieved candidate (see
`_backfill_scores`), and only when retrieval under-delivered.

The interface is still (c)'s: `ConstraintIndex.as_retriever` satisfies the same
``search(query_text, top_k) -> [(business_id, score)]`` protocol as
`concierge.retrieval.bm25.BM25Index`, so if measurement later favours full
fusion, `fusion.fuse` can take this list as a third system with no change here.
It is an adapter rather than the primary method because the honest signature
takes a `ParsedQuery`: there is no query *text* in a constraint.

Agreement with the filters is by construction, not by convention
----------------------------------------------------------------
A backfilled candidate that `filters.apply_filters` then drops would leave the
page empty again — the bug, reintroduced one stage later. So this module does
**not** reimplement any check. It calls `filters.evaluate_constraints`, the same
function `apply_filters` calls, and admits a business under the same two rules:

* any VIOLATED match -> never admitted (`apply_filters` drops it unconditionally);
* any UNKNOWN match -> admitted only when
  ``cfg.retrieval.missing_attribute_policy`` is not ``exclude``, because under
  ``exclude`` `apply_filters` would drop it.

The inverted index below therefore holds no comparison logic at all. It holds
only *has this business any data for this field*, which is the one thing Yelp's
attribute sparsity makes selective, and every hit is verified by
`evaluate_constraints` before it is returned. A posting list that is too
generous costs a few wasted checks; it cannot produce a wrong answer.

What is deliberately **not** done here, and what to measure next
----------------------------------------------------------------
The default trigger fires on *how few candidates retrieval returned*
(``TRIGGER_FEW_CANDIDATES``). Under the default ``penalize`` policy that is a
conservative trigger: `apply_filters` keeps a candidate whose attribute data is
merely missing, so a query like "somewhere with a patio" against a *dense* index
returns a full page of candidates, none of which is known to have outdoor
seating, and the count-based trigger stays quiet. Detecting that case needs a
satisfaction-aware trigger, which is implemented as ``TRIGGER_FEW_SATISFIED``
and is **not** the default: it changes the candidate set of every constrained
query, and whether that helps or hurts is a labelled measurement this project
has not run. Both triggers are parameters of one function so the comparison is
an ablation rather than a rewrite — the same way
``cfg.retrieval.missing_attribute_policy`` keeps all three missing-data policies
runnable (DECISIONS.md D007).
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from concierge.config import Config
from concierge.retrieval.filters import (
    CONSTRAINT_CATEGORY,
    CONSTRAINT_GEO,
    CONSTRAINT_OPEN_AT,
    CONSTRAINT_PRICE_MAX,
    PRICE_ATTRIBUTE,
    evaluate_constraints,
)
from concierge.types import (
    RANK_RRF,
    SCORE_RRF,
    Business,
    Candidate,
    ConstraintMatch,
    ConstraintStatus,
    ParsedQuery,
)

log = logging.getLogger(__name__)

#: Score key for this path's own number, kept out of `SCORE_RRF` so a reader of
#: a candidate can always tell a backfilled candidate from a retrieved one. It
#: is a module constant rather than an entry in `concierge.types` because
#: nothing downstream consumes it yet: `concierge.ltr.features` reads only the
#: keys it names, and an unknown key is ignored. Promoting it to a Stage 4
#: feature would be a feature-set change, which needs a measurement.
SCORE_CONSTRAINT: Final = "constraint"

#: Which condition makes `backfill` fire. See the module docstring for why the
#: conservative one is the default.
TRIGGER_FEW_CANDIDATES: Final = "few_candidates"
TRIGGER_FEW_SATISFIED: Final = "few_satisfied"
Trigger = Literal["few_candidates", "few_satisfied"]

#: The top of the Yelp rating scale. A domain fact (ratings are 1.0-5.0 in
#: half-steps), not a measurement — it is used only to keep the quality prior
#: strictly inside one unit so that it can serve as a tie-break below an integer
#: count without ever overturning it (see `_composite_score`).
_STARS_CEILING: Final = 5.0

# Field keys for the inverted index. The four named constraints borrow the
# filters' own constant names so a posting and a `ConstraintMatch` are talking
# about the same thing; attribute keys are namespaced because an attribute could
# in principle be called "geo".
_ATTRIBUTE_PREFIX: Final = "attr:"


def _attribute_field(key: str) -> str:
    return f"{_ATTRIBUTE_PREFIX}{key}"


# ─────────────────────────────────────────────────────────────────────────────
# The quality prior
# ─────────────────────────────────────────────────────────────────────────────


def shrunk_rating(stars: float, review_count: int, *, corpus_mean: float, weight: float) -> float:
    """A star rating pulled towards the corpus mean in proportion to its evidence.

    ``(v/(v+m)) * R + (m/(v+m)) * C`` — the standard weighted-rating/empirical-
    Bayes shrinkage, where ``R`` is this business's rating, ``v`` its review
    count, ``C`` the corpus mean rating and ``m`` the shrinkage weight.

    Why any shrinkage at all: the constraint path has no relevance signal to
    rank with, so the ordering *is* the prior, and the obvious priors are both
    wrong on their own. Raw stars puts a 5.0 with three reviews above a 4.5 with
    two thousand; raw review count ranks by fame and ignores quality. Shrinkage
    is the one-line fix and it is the ordering every "best of" list in the
    industry uses.

    Why ``m`` is the corpus *median* review count (`ConstraintIndex.build`):
    ``m`` is by definition the review count at which a business's own rating and
    the corpus mean carry equal weight. Setting it to the median says that half
    the corpus is pulled more than halfway towards the mean — which is the honest
    statement about how little a sparse rating tells you. It is read off the
    corpus rather than tuned, so it costs no label budget and no hyperparameter
    search.
    """
    if weight <= 0.0:
        # A weight of zero collapses the formula back to the raw rating and
        # divides by zero for a business with no reviews at all.
        raise ValueError(f"shrinkage weight must be positive, got {weight!r}")
    denominator = review_count + weight
    return (review_count / denominator) * stars + (weight / denominator) * corpus_mean


def _composite_score(n_established: int, prior: float) -> float:
    """One number whose ordering *is* the selection rule.

    `select` returns ``(business_id, score)`` pairs promised to be best-first, so
    the score has to be monotone with the order — a caller that min-max scales
    them (`fusion.weighted_sum_scores` does) would otherwise reorder the list.
    Two criteria have to fit in one float: how many of the query's constraints
    the data actually *established*, and the quality prior.

    Dividing the prior by ``_STARS_CEILING + 1`` keeps it strictly inside
    ``[0, 1)``, so the integer count always dominates and the prior only ever
    breaks ties within a count. That is lexicographic order expressed as
    arithmetic, which is what keeps it monotone.
    """
    return n_established + prior / (_STARS_CEILING + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Which fields a parsed query asks about
# ─────────────────────────────────────────────────────────────────────────────


def _requested_fields(parsed: ParsedQuery) -> tuple[str, ...]:
    """The index fields this query constrains, mirroring `evaluate_constraints`.

    Used for two things only: deciding whether there is anything to select on at
    all, and picking posting lists to intersect. It is never used to *count*
    constraints — `select` takes that from the `ConstraintMatch` tuple
    `evaluate_constraints` actually returned, so a drift between this function
    and that one can cost a posting-list narrowing but cannot change a verdict.
    """
    fields: list[str] = []
    if parsed.open_at is not None:
        fields.append(CONSTRAINT_OPEN_AT)
    if parsed.price_max is not None:
        fields.append(CONSTRAINT_PRICE_MAX)
    for key in parsed.attributes:
        # Price arrives as `price_max` and is checked as an inequality;
        # `evaluate_constraints` skips it here for the same reason.
        if key == PRICE_ATTRIBUTE:
            continue
        fields.append(_attribute_field(key))
    if parsed.categories:
        fields.append(CONSTRAINT_CATEGORY)
    if parsed.geo is not None:
        fields.append(CONSTRAINT_GEO)
    return tuple(fields)


# ─────────────────────────────────────────────────────────────────────────────
# The index
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ConstraintIndex:
    """Businesses in quality-prior order, plus "has data for this field" postings.

    Built in one pass over records the pipeline already holds in memory, and
    deliberately **not** persisted to disk. The BM25 and FAISS indexes are saved
    because rebuilding them means re-tokenizing or re-embedding every document;
    this one is a sort and a few set builds, so a saved copy would buy nothing
    and add a staleness failure mode (an index keyed by a config fingerprint
    that no longer matches the business file).
    """

    #: Businesses, best prior first. Position in this tuple is the posting id.
    ordered: tuple[Business, ...]
    #: ``field -> positions``, every position whose business has *some* data for
    #: that field. A superset of the positions that could be SATISFIED; see the
    #: module docstring on why a generous posting list is safe.
    postings: Mapping[str, frozenset[int]]
    #: The prior per position, parallel to `ordered`.
    priors: tuple[float, ...]
    #: ``business_id -> Business``, for the satisfaction-aware trigger. Built
    #: once here rather than per query: rebuilding it inside `backfill` would be
    #: a corpus-sized allocation on a per-request path.
    by_id: Mapping[str, Business]
    #: What the prior was built from, so a number traced back to this ordering
    #: can be explained without re-deriving it.
    meta: Mapping[str, float | int]

    @property
    def n_indexed(self) -> int:
        return len(self.ordered)

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, businesses: Iterable[Business]) -> ConstraintIndex:
        """Index every *open* business.

        ``is_open == 0`` businesses are excluded here for the same reason
        `apply_filters` drops them and the document build never indexes them:
        showing a permanently closed business is the most embarrassing wrong
        answer available, and this path has no other gate in front of it.
        """
        open_businesses = [b for b in businesses if b.is_open]
        if not open_businesses:
            # An empty index is a legitimate state (an over-narrow metro scope),
            # and `select` on it returns nothing rather than raising: the
            # constraint path is a fallback and must never be the thing that
            # breaks a search.
            log.warning("constraint index built over zero open businesses")
            return cls(ordered=(), postings={}, priors=(), by_id={}, meta={"n_indexed": 0})

        corpus_mean = statistics.fmean(b.stars for b in open_businesses)
        median_reviews = statistics.median(b.review_count for b in open_businesses)
        # A median of zero (a corpus of unreviewed businesses) would make the
        # shrinkage weight illegal; one review is the smallest weight that keeps
        # the formula defined, and at that weight the prior is essentially the
        # raw rating — which is the right reading when nothing has any reviews.
        weight = max(float(median_reviews), 1.0)

        scored = [
            (
                shrunk_rating(b.stars, b.review_count, corpus_mean=corpus_mean, weight=weight),
                b,
            )
            for b in open_businesses
        ]
        # Descending prior, then business_id. The id tie-break is not cosmetic:
        # ties are common (two businesses with the same stars and review count)
        # and an unreproducible order would make an A/B difference in the
        # ablation un-attributable, exactly as in `fusion.fuse`.
        scored.sort(key=lambda pair: (-pair[0], pair[1].business_id))

        ordered = tuple(business for _prior, business in scored)
        priors = tuple(prior for prior, _business in scored)
        postings = _build_postings(ordered)

        return cls(
            ordered=ordered,
            postings=postings,
            priors=priors,
            by_id={b.business_id: b for b in ordered},
            meta={
                "n_indexed": len(ordered),
                "corpus_mean_stars": corpus_mean,
                "shrinkage_weight_reviews": weight,
            },
        )

    # ── selection ────────────────────────────────────────────────────────────

    def select(self, parsed: ParsedQuery, top_k: int, cfg: Config) -> list[tuple[str, float]]:
        """Up to ``top_k`` ``(business_id, score)`` pairs the constraints admit.

        Returns ``[]`` when the query constrains nothing: this path has no
        relevance signal of its own, so with no constraint to select on it would
        be answering a different question ("which businesses are good?") than
        the one asked.

        The ordering rule, stated once:

        1. **Established first.** Businesses whose every constraint came back
           SATISFIED, best prior first. "This place definitely has a patio"
           outranks "this place might have a patio", which is the same judgement
           ``missing_attribute_policy=penalize`` makes in the fused score.
        2. **Then the merely-not-refuted**, fewest UNKNOWN matches first, then by
           prior. Skipped entirely under ``missing_attribute_policy=exclude``,
           because `apply_filters` would drop every one of them and the page
           would be empty again.

        Both groups stop as soon as ``top_k`` is reached, which is what keeps
        this cheap on a large corpus: group 1 iterates only the intersection of
        the relevant posting lists, and group 2 iterates the prior order and
        stops early, so neither scans the corpus in the common case. Group 2's
        order is therefore exact among the businesses it examined — the
        highest-prior non-refuted ones — rather than globally optimal in
        UNKNOWN count. That is a deliberate latency choice: a constraint we have
        no data for is not a reason to promote a low-prior business over a
        high-prior one.

        The worst case, stated rather than left to be discovered: when almost
        nothing satisfies the constraints, both groups run to exhaustion and
        this costs one pass over the index, calling `evaluate_constraints` per
        business. It is reachable only on a query where retrieval already
        under-delivered *and* the answer is genuinely near-empty — a query like
        "open at 3am" against a corpus that mostly closes at midnight — and
        there is no cheaper exact answer, because establishing that a business
        is closed requires checking it. No cap is imposed here: an arbitrary one
        would silently truncate recall, and this path is on the measured latency
        surface (`concierge.pipeline` times Stage 2 on every search), so the cost
        shows up as a number rather than as a guess. If that number turns out to
        need a bound, the bound belongs here, on the group-2 scan, and it must be
        reported alongside the recall it costs.
        """
        if top_k <= 0 or not self.ordered:
            return []
        fields = _requested_fields(parsed)
        if not fields:
            return []

        # Drift guard, checked once rather than per candidate. `_requested_fields`
        # mirrors `evaluate_constraints` by hand; if it ever says "there is a
        # constraint" where that function finds none, every business would come
        # back unrefuted and this would quietly become a popularity ranking for
        # a query that constrained nothing. The check count does not depend on
        # the business, so one probe settles it.
        if not evaluate_constraints(self.ordered[0], parsed):
            log.error("constraint fields %r produced no checks; selecting nothing", fields)
            return []

        policy = cfg.retrieval.missing_attribute_policy
        # (position, number of constraints checked). Every match in this group
        # is SATISFIED, so the count *is* the number established.
        established: list[tuple[int, int]] = []
        claimed: set[int] = set()

        for position in self._narrowed(fields):
            matches = evaluate_constraints(self.ordered[position], parsed)
            if _refuted(matches) or _n_unknown(matches):
                continue
            established.append((position, len(matches)))
            claimed.add(position)
            if len(established) >= top_k:
                break

        unresolved: list[tuple[int, int, int]] = []  # (n_unknown, position, n_matches)
        if len(established) < top_k and policy != "exclude":
            for position in range(len(self.ordered)):
                if position in claimed:
                    continue
                matches = evaluate_constraints(self.ordered[position], parsed)
                if _refuted(matches):
                    continue
                unresolved.append((_n_unknown(matches), position, len(matches)))
                if len(established) + len(unresolved) >= top_k:
                    break
            # Fewest guesses first, then prior order (position *is* prior order).
            unresolved.sort(key=lambda row: (row[0], row[1]))

        out: list[tuple[str, float]] = [
            (
                self.ordered[position].business_id,
                _composite_score(n_matches, self.priors[position]),
            )
            for position, n_matches in established
        ]
        out.extend(
            (
                self.ordered[position].business_id,
                _composite_score(n_matches - n_unknown, self.priors[position]),
            )
            for n_unknown, position, n_matches in unresolved
        )
        return out[:top_k]

    def as_retriever(self, parsed: ParsedQuery, cfg: Config) -> _ConstraintRetriever:
        """This index bound to one parsed query, as a plain `Retriever`.

        The returned object's ``search(query_text, top_k)`` matches
        `concierge.pipeline.Retriever` exactly, so `fusion.fuse` could take it as
        a third system if D224's option (c) is ever measured and preferred. The
        text argument is ignored, which is the whole point: a constraint has no
        query text, and pretending otherwise is what got this stage into
        trouble.
        """
        return _ConstraintRetriever(index=self, parsed=parsed, cfg=cfg)

    def _narrowed(self, fields: Sequence[str]) -> Iterator[int]:
        """Positions, in prior order, that have data for *every* requested field.

        The intersection starts from the smallest posting list, because that
        bounds the work by the most selective constraint — which, with Yelp's
        attribute sparsity, is usually a very small set. A field with no posting
        list at all (nothing in the corpus has data for it) makes the
        intersection empty, which is correct: nothing can establish it.
        """
        lists = [self.postings.get(field, frozenset()) for field in fields]
        if not lists:
            return iter(())
        lists.sort(key=len)
        narrowed = set(lists[0])
        for other in lists[1:]:
            narrowed &= other
            if not narrowed:
                break
        return iter(sorted(narrowed))


@dataclass(frozen=True, slots=True)
class _ConstraintRetriever:
    """`ConstraintIndex` + a parsed query, behind the `Retriever` protocol."""

    index: ConstraintIndex
    parsed: ParsedQuery
    cfg: Config

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Up to ``top_k`` ``(business_id, score)`` pairs, best first.

        ``query_text`` is accepted and ignored — the constraint domain does not
        depend on it. Taking the argument anyway is what makes this object
        substitutable for `BM25Index` and `DenseIndex` in `fusion.fuse`.
        """
        del query_text
        return self.index.select(self.parsed, top_k, self.cfg)


def _build_postings(ordered: Sequence[Business]) -> dict[str, frozenset[int]]:
    """``field -> positions with some data for that field``.

    Each posting list is a *superset* of the positions that could satisfy a
    constraint on that field, never an exact match set — the exact answer comes
    from `evaluate_constraints`. Two of them are worth explaining:

    * **Hours** are indexed as "has any hours data at all" rather than per
      weekday. `concierge.data.hours.is_open_at` resolves a late-night minute
      against the *previous* day's interval ("17:0-2:0"), so a business with no
      Monday hours can still be open on Monday at 00:30. A per-day posting would
      have to replicate that spillover rule to stay a superset, which is exactly
      the duplication this module refuses; and since Yelp hours are typically
      present or absent wholesale, the coarse list is nearly as selective.
    * **Categories** are indexed as "has any category", because
      `filters.check_categories` matches a requested name against a business
      category's *words* ("coffee" matches "Coffee & Tea"). A posting keyed by
      category name would miss those, i.e. it would not be a superset.
    """
    postings: dict[str, set[int]] = {}

    def add(field: str, position: int) -> None:
        postings.setdefault(field, set()).add(position)

    for position, business in enumerate(ordered):
        # `Business.hours` is the *parsed* form, where a day with no usable
        # interval is simply absent, so a non-empty mapping is exactly "has some
        # hours". `concierge.data.hours.has_hours_data` answers the same
        # question about the *raw* record and is documented for data-quality
        # reporting, so it is not reused here.
        if business.hours:
            add(CONSTRAINT_OPEN_AT, position)
        if business.has_attribute_data(PRICE_ATTRIBUTE):
            add(CONSTRAINT_PRICE_MAX, position)
        for key in business.attributes:
            if business.has_attribute_data(key):
                add(_attribute_field(key), position)
        if business.categories:
            add(CONSTRAINT_CATEGORY, position)
        if business.latitude is not None and business.longitude is not None:
            add(CONSTRAINT_GEO, position)

    return {field: frozenset(positions) for field, positions in postings.items()}


def _refuted(matches: Sequence[ConstraintMatch]) -> bool:
    return any(m.status is ConstraintStatus.VIOLATED for m in matches)


def _n_unknown(matches: Sequence[ConstraintMatch]) -> int:
    return sum(1 for m in matches if m.status is ConstraintStatus.UNKNOWN)


def _all_satisfied(matches: Sequence[ConstraintMatch]) -> bool:
    return bool(matches) and all(m.status is ConstraintStatus.SATISFIED for m in matches)


# ─────────────────────────────────────────────────────────────────────────────
# The backfill
# ─────────────────────────────────────────────────────────────────────────────


def _backfill_scores(count: int, *, ceiling: float, k: int, budget: int) -> list[float]:
    """``count`` strictly decreasing positive scores, all strictly below ``ceiling``.

    This is the whole non-displacement guarantee, so it is worth being precise
    about. `filters.apply_filters` re-sorts its survivors by ``SCORE_RRF``, so
    "appended at the end" is not enough — the ordering has to live in the score
    itself.

    The shape is RRF's own decay, continued past the retrieval budget: a
    backfilled candidate is scored as though some system had returned it at rank
    ``budget + 1 + i``, and the whole block is then scaled to sit under the worst
    retrieved candidate. ``(k + budget) / (k + budget + 1 + i)`` is in ``(0, 1)``
    and strictly decreasing in ``i``, so every backfilled score is strictly
    positive, strictly below ``ceiling``, and in the order `select` chose.

    Strictly *positive* is load-bearing and not just tidiness: under
    ``missing_attribute_policy=penalize`` `apply_filters` multiplies
    ``SCORE_RRF`` by a factor below one, which moves a negative score *up*. A
    negative encoding of "below everything" would invert the penalty.
    """
    base = max(k + budget, 1)
    return [ceiling * base / (base + 1 + i) for i in range(count)]


def _ceiling(fused: Sequence[Candidate], *, k: int, budget: int) -> float:
    """The score the backfill block must stay strictly below.

    The smallest *positive* fused score, or a single RRF quantum when there is
    no retrieved candidate to stay below.

    Why positive-only: under ``fusion.method = "weighted_sum"`` each system's
    scores are min-max scaled, so the worst document a system returned scores
    exactly 0.0 — `fusion.py` says in its own docstring that this method cannot
    distinguish "returned last" from "absent". Where fusion itself cannot tell
    them apart, this function does not pretend to either, and such a candidate
    may tie with the backfill block. Under the default RRF every fused score is
    strictly positive, so the guarantee is unconditional there.
    """
    positives = [
        score
        for score in (candidate.scores.get(SCORE_RRF, 0.0) for candidate in fused)
        if score > 0.0
    ]
    if positives:
        return min(positives)
    return 1.0 / max(k + budget, 1)


def _n_satisfying(
    fused: Sequence[Candidate],
    index: ConstraintIndex,
    parsed: ParsedQuery,
) -> int:
    """How many retrieved candidates establish *every* constraint in the query."""
    total = 0
    for candidate in fused:
        business = index.by_id.get(candidate.business_id)
        if business is None:
            continue
        if _all_satisfied(evaluate_constraints(business, parsed)):
            total += 1
    return total


def backfill(
    fused: Sequence[Candidate],
    *,
    index: ConstraintIndex,
    parsed: ParsedQuery,
    cfg: Config,
    min_candidates: int | None = None,
    trigger: Trigger = TRIGGER_FEW_CANDIDATES,
) -> list[Candidate]:
    """Extend an under-delivering Stage 2 candidate list with constraint-only hits.

    Returns ``fused`` unchanged — same objects, same order — whenever the
    trigger does not fire, which is the normal case. When it does fire, the
    extra candidates are appended with ``SCORE_RRF`` values strictly below every
    retrieved candidate's (`_backfill_scores`), so a retrieved candidate can
    never be displaced by one. That is the guarantee this function exists to
    make: a query with real lexical or semantic evidence must not have it
    diluted.

    The exact scope of that guarantee, because overstating it would be worse
    than not making it: it holds on the fused Stage-2 score, which is where this
    function writes. `filters.apply_filters` runs *after* this and, under
    ``missing_attribute_policy=penalize``, discounts a retrieved candidate whose
    constraint data is merely missing — which can leave it below a backfilled
    candidate that *establishes* the constraint. That reordering is the policy
    doing its job (D007), not the backfill displacing anything, and it is the
    same discount the retrieved candidate would have taken with no backfill at
    all.

    Args:
        fused: `fusion.fuse`'s output, best first.
        index: the constraint index over the same corpus.
        parsed: Stage 1's output. With no hard constraints this is a no-op.
        cfg: supplies the missing-data policy, the fusion ``k`` and the Stage 2
            candidate budget.
        min_candidates: the floor the trigger compares against. Defaults to
            ``cfg.candidates.final`` — the number of results the user is
            actually promised. That is the threshold tied to the user-visible
            symptom rather than a tuned constant: if retrieval already produced
            a full page, there is no empty-page failure to repair, and if it did
            not, the page is short however it is ranked. There is no config key
            for it because `Config` is a frozen ``extra="forbid"`` contract owned
            elsewhere; an experiment that wants to sweep it passes it in.
        trigger: ``TRIGGER_FEW_CANDIDATES`` (default) fires on how many
            candidates retrieval returned. ``TRIGGER_FEW_SATISFIED`` fires on
            how many of them establish every constraint, which catches the case
            a dense retriever masks by returning a full page of candidates that
            merely fail to contradict the request. See the module docstring for
            why the aggressive one is not the default.

    The caller is responsible for not invoking this when hard filtering is
    disabled: a ``no_filters`` ablation row that received constraint-driven
    candidates would be measuring constraint handling under another name.
    """
    candidates = list(fused)
    if not _requested_fields(parsed):
        return candidates

    floor = cfg.candidates.final if min_candidates is None else min_candidates
    if trigger == TRIGGER_FEW_SATISFIED:
        have = _n_satisfying(candidates, index, parsed)
    else:
        have = len(candidates)
    if have >= floor:
        return candidates

    budget = cfg.candidates.retrieve
    room = budget - len(candidates)
    if room <= 0:
        return candidates

    present = {candidate.business_id for candidate in candidates}
    # Ask for `room` plus what we already hold, because `select` knows nothing
    # about the retrieved list and its best hits are likely to be in it.
    selected = [
        (business_id, score)
        for business_id, score in index.select(parsed, room + len(present), cfg)
        if business_id not in present
    ][:room]
    if not selected:
        return candidates

    k = cfg.retrieval.fusion.k
    scores = _backfill_scores(
        len(selected), ceiling=_ceiling(candidates, k=k, budget=budget), k=k, budget=budget
    )
    first_rank = len(candidates) + 1
    candidates.extend(
        Candidate(
            business_id=business_id,
            # SCORE_RRF carries the ordering (and is what `apply_filters` sorts
            # on); SCORE_CONSTRAINT keeps this path's own number, which is on a
            # different scale and must not be confused with a fused score.
            scores={SCORE_RRF: rrf_score, SCORE_CONSTRAINT: own_score},
            # A rank below every fused rank. `apply_filters` recomputes
            # RANK_RRF over its survivors, so this matters only to a caller that
            # skips filtering — but a candidate with no rank at all would reach
            # Stage 4's `rrf_rank` feature as a missing value.
            ranks={RANK_RRF: first_rank + offset},
        )
        for offset, ((business_id, own_score), rrf_score) in enumerate(
            zip(selected, scores, strict=True)
        )
    )
    log.debug(
        "constraint backfill added %d candidates to %d retrieved (trigger=%s, floor=%d)",
        len(selected),
        len(fused),
        trigger,
        floor,
    )
    return candidates
