"""Stage 2, fusion: combining the lexical and dense result lists.

Why Reciprocal Rank Fusion is the default
-----------------------------------------
The two retrievers return numbers that are not on the same scale and cannot be
put on the same scale without inventing a hyperparameter:

* A BM25 score is unbounded above and depends on the corpus — it is a sum over
  query terms of an IDF weight computed from *this* collection's term
  statistics. A score of 8.0 means nothing on its own; it is not comparable
  across queries, let alone across systems.
* A cosine similarity between normalized embeddings sits in [-1, 1], and in
  practice bunches into a narrow band near the top (0.7-0.9 for a small bge
  model).

So ``bm25_score + cosine`` is dominated by BM25 for long queries and by nothing
in particular for short ones. Fixing that requires normalizing each system's
scores and weighting them — and both the normalization scheme (min-max over
what population? z-score with what mean?) and the weights are themselves
untuned hyperparameters. Tuning them needs labels, and labels are the scarcest
thing in this project.

Reciprocal Rank Fusion sidesteps the problem: it throws the scores away and
uses only *ranks*, which every ranked list produces on the same scale by
construction. It is therefore scale-invariant and parameter-light (one constant,
``k``), which is exactly what you want from a component you cannot afford to
tune. The cost is real and worth stating: discarding scores discards the
*margins*, so a document that BM25 is wildly confident about looks the same as
the one just below it.

``weighted_sum`` is implemented anyway, behind ``cfg.retrieval.fusion.method``,
so that claim can be measured rather than asserted — the ablation can run both.

The role of k
-------------
``score = Σ 1/(k + rank)``. ``k`` damps the top of each list: with ``k = 60``
the gap between rank 1 and rank 2 is 1/61 - 1/62 ≈ 0.00026, while between rank
1 and rank 100 it is 1/61 - 1/160 ≈ 0.010. A small ``k`` makes rank 1 nearly
unbeatable, so a single system can dominate; a large ``k`` flattens everything
towards "appeared in many lists". 60 is the value from the original Cormack et
al. (2009) paper and the common default; it is a starting point to sweep, not a
result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from concierge.config import Config
from concierge.types import RANK_RRF, SCORE_RRF, Candidate

# A system's output: ``(business_id, score)`` pairs, best first. Keys of the
# `systems` mapping must be the canonical names from `concierge.types`
# (SCORE_BM25 / SCORE_DENSE, whose string values are shared with RANK_BM25 /
# RANK_DENSE), because that is what the Stage 4 feature builder looks for.
SystemResults = Sequence[tuple[str, float]]


def rank_map(results: SystemResults) -> dict[str, int]:
    """``business_id -> 1-based rank`` for one system's result list.

    Ranks are 1-based because RRF's ``1/(k + rank)`` assumes it: a 0-based rank
    would give the top document ``1/k``, silently shifting every weight by one
    position and making the implementation disagree with the formula in the
    paper (and with every other implementation it would be compared against).

    A business repeated in one list keeps its *best* rank. That should not
    happen with a correct retriever, but if it does, the alternative (last
    occurrence wins) would quietly demote a top hit.
    """
    ranks: dict[str, int] = {}
    for position, (business_id, _score) in enumerate(results, start=1):
        ranks.setdefault(business_id, position)
    return ranks


def score_map(results: SystemResults) -> dict[str, float]:
    """``business_id -> score`` for one system's result list.

    The companion of `rank_map`, and it exists so the two cannot disagree. A
    business repeated in one list keeps the score of its *best*-ranked
    occurrence, matching `rank_map`'s "best rank wins".

    `dict(results)` would be shorter and wrong: it keeps the *last* occurrence,
    so a duplicate would leave the candidate carrying rank 1 together with the
    score from rank 180. Those two go into `Candidate.scores` and
    `Candidate.ranks` and are both read as Stage 4 features (``bm25_score`` next
    to ``rrf_rank``), so the model would be trained on a row describing a
    retrieval result that never happened. Duplicates should not occur with a
    correct retriever; this makes "should not" cost nothing if it does.
    """
    scores: dict[str, float] = {}
    for business_id, score in results:
        scores.setdefault(business_id, score)
    return scores


def rrf_scores(rankings: Mapping[str, Mapping[str, int]], k: int) -> dict[str, float]:
    """``business_id -> Σ_systems 1/(k + rank)``.

    The one rule that matters: a system that did **not** return a document
    contributes *nothing* to that document's score — no term at all.

    This is the classic RRF bug. It is tempting to treat "missing" as rank 0
    (giving a ``1/k`` bonus, the largest possible contribution — a document is
    rewarded for being absent) or as "rank = list length + 1" (a smaller but
    still unearned bonus that also makes the score depend on how long each
    list happens to be). Both make a document found by one system beat a
    document found by one system plus a near-miss in the other. Omitting the
    term is the only version that means "no evidence".
    """
    scores: dict[str, float] = {}
    for ranks in rankings.values():
        for business_id, rank in ranks.items():
            scores[business_id] = scores.get(business_id, 0.0) + 1.0 / (k + rank)
    return scores


def _minmax(values: Mapping[str, float]) -> dict[str, float]:
    """Scale scores into [0, 1] per system.

    Only used by `weighted_sum`. Min-max is the crudest possible choice and is
    sensitive to a single outlier at the top of the list; it is here to make the
    ablation runnable, not because it is good. A degenerate range (every score
    equal, or a single result) maps to 1.0 — anything else would turn "one
    result" into "a result worth zero".
    """
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo <= 0.0:
        return dict.fromkeys(values, 1.0)
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def weighted_sum_scores(
    systems: Mapping[str, SystemResults],
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Min-max normalize each system, then take a weighted sum.

    Weights default to 1.0 per system. They are a function argument rather than
    a config key because `FusionConfig` carries only ``method`` and ``k``; an
    ablation that wants to sweep them passes them in explicitly.

    A document missing from a system contributes 0 from that system. Note that
    this is *not* the same kind of choice as in RRF: after min-max scaling, 0 is
    the value of the worst returned document, so "absent" and "returned last"
    are treated alike. That conflation is one more reason this method needs
    tuning to be competitive, and one more reason RRF is the default.
    """
    out: dict[str, float] = {}
    for name, results in systems.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for business_id, score in _minmax(score_map(results)).items():
            out[business_id] = out.get(business_id, 0.0) + weight * score
    return out


def fuse(
    systems: Mapping[str, SystemResults],
    cfg: Config,
    top_k: int | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """Fuse per-system result lists into ranked `Candidate` objects.

    Every upstream score and rank is written through to the candidate, not just
    the fused value: Stage 4 consumes them as features, and the ablation has to
    be able to attribute a win to a specific stage.

    Ties in the fused score are broken by ``business_id``. An arbitrary but
    *deterministic* tie-break matters more than it looks: ties are common with
    RRF (two documents found at the same ranks score identically), and
    dict-iteration order would otherwise make a run unreproducible and an
    A/B difference un-attributable.

    ``top_k`` defaults to ``cfg.candidates.retrieve`` — the Stage 2 budget.
    """
    rankings = {name: rank_map(results) for name, results in systems.items()}
    # Same first-occurrence rule as `rank_map`, so a candidate's score and its
    # rank always describe the same position in the same list.
    raw_scores = {name: score_map(results) for name, results in systems.items()}

    method = cfg.retrieval.fusion.method
    if method == "rrf":
        fused = rrf_scores(rankings, cfg.retrieval.fusion.k)
    elif method == "weighted_sum":
        fused = weighted_sum_scores(systems, weights)
    else:  # pragma: no cover — pydantic's Literal already rejects anything else.
        raise ValueError(f"unknown fusion method: {method!r}")

    order = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    limit = cfg.candidates.retrieve if top_k is None else top_k
    order = order[:limit]

    candidates: list[Candidate] = []
    for position, (business_id, fused_score) in enumerate(order, start=1):
        scores = {
            # The fused score always lands in SCORE_RRF, whichever method
            # produced it: it is the "Stage 2 fused score" slot that Stage 4
            # reads by name. Which method produced a given number is not
            # ambiguous — it is in the config, and artifacts are keyed by the
            # config fingerprint.
            SCORE_RRF: fused_score,
        }
        ranks = {RANK_RRF: position}
        for name in systems:
            if business_id in raw_scores[name]:
                scores[name] = raw_scores[name][business_id]
                ranks[name] = rankings[name][business_id]
        candidates.append(Candidate(business_id=business_id, scores=scores, ranks=ranks))
    return candidates
