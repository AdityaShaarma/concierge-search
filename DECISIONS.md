# Decisions

Every meaningful choice, the alternatives considered, why this one, and how it could be
tested. Append — never rewrite history. This file is interview preparation: if a decision
here cannot be explained in plain English, it does not belong in the project.

Status legend: **Settled** (measured or structural) · **Provisional** (chosen as a
default, to be decided by measurement) · **Deferred** (deliberately not decided yet).

---

## D001 — `uv` as the package manager · Settled
*2026-10-15*

**Decision.** `uv` for dependency resolution, virtualenv management, and locking.

**Alternatives.** `pip` + `venv` (already installed, nothing to learn); Poetry (mature,
widely known); conda (best binary story for FAISS historically).

**Why.** The project's core promise is "every number is regenerable with one command."
That needs a real lockfile, so an evaluation run six weeks from now resolves the exact
same dependency versions. `pip` + `requirements.txt` pins direct dependencies but not
the transitive graph. `uv` also installs its own Python, so the environment does not
depend on whatever the laptop's system Python happens to be.

**How it could be tested.** Delete `.venv`, run `make setup`, run `make eval`, confirm
identical metrics. That is the actual reproducibility claim, and it is worth running once
before the project is called done.

---

## D002 — Python pinned to `>=3.11,<3.13` · Settled
*2026-10-15*

**Decision.** Pin below 3.13 even though the laptop has 3.13.3 installed; `uv` fetches
3.12.

**Alternatives.** Use 3.13 and deal with wheel gaps as they appear.

**Why.** `faiss-cpu` and some `torch` builds lag a few months behind each new Python
minor release. Debugging a missing wheel is pure lost time that teaches nothing about
search. 3.12 has universal wheel coverage for every dependency here.

**Revisit when.** Phase 5, if anything else forces a Python bump.

---

## D003 — Location/distance is in scope, minimally · Settled
*2026-10-15*

**Decision.** Carry `latitude`/`longitude` from the first data build. A geo filter and a
distance feature land in Phase 2. Until they exist, `near X` queries stay out of the gold
set.

**Alternatives.** (a) Drop geo entirely and reword the project pitch. (b) Full
neighborhood resolution with a gazetteer and per-city centroids.

**Why.** The project's own pitch includes *"cheap ramen near downtown"*, so a system that
cannot reason about location fails its own example. Coordinates are already in
`business.json`, so carrying them costs nothing now and avoids a reindex later. The
gazetteer version is scope the project does not need: a city-centroid plus radius handles
"downtown" and "near X" well enough to be honest about.

**How it could be tested.** Hold a geo-constrained slice of the gold set and compare
NDCG@10 with the distance feature on and off. If it does not move, say so and keep the
simpler system.

**Consequence to watch.** If gold queries containing `near X` get written and labeled
before the filter exists, they will look like model failures when they are really missing
functionality. Noted in the gold-query instructions.

---

## D004 — Gold set: write ~150, fully label a random 50 first · Settled
*2026-10-15*

**Decision.** The owner writes all ~150 queries up front (a few hours), then labels a
**random, fixed-seed** 50-query subset at full pool depth 20, expanding toward 150 as
time allows. All reported metrics state how many queries they are computed over.

**Alternatives.** (a) Label all 150 before reporting anything (~25–30 h of labeling
before a single number exists). (b) Cut pool depth from 20 to 10 across all 150 queries.
(c) Label whichever 50 queries are quickest to judge.

**Why.** (a) delays all feedback until the end, which is how label quality degrades — the
last queries get rushed. (c) is disqualifying: a non-random subset biases every number,
and an interviewer will ask how the subset was chosen. (b) looked attractive but weakens
Recall@100, and recall is the measurement that justifies the retrieval stage existing at
all. A random subset of a larger set is a standard, defensible sampling design; the cost
is wider confidence intervals, which get reported honestly.

**How it could be tested.** When labeling expands from 50 to 100 queries, check whether
the ablation's conclusions change. If the ranking of systems is stable, the subset was
large enough — and that stability is itself a result worth reporting.

---

## D005 — Frozen dataclasses with per-stage score dictionaries · Settled
*2026-10-15*

**Decision.** `Candidate` carries `scores: dict[str, float]` and `ranks: dict[str, int]`
keyed by stage, and is immutable; stages return copies.

**Alternatives.** A single mutable `score` field each stage overwrites; a pandas DataFrame
threaded through the pipeline.

**Why.** Stage 4 consumes *every* upstream score as a feature, so nothing may be discarded
between stages. Immutability means a bug in the reranker cannot silently corrupt a
retrieval score, which would make the ablation untraceable — and the ablation is the
project's main artifact. A DataFrame would be faster in bulk but makes a single query's
path through the cascade much harder to inspect when debugging a specific failure.

---

## D006 — Constraint satisfaction is three-valued, not boolean · Settled
*2026-10-15*

**Decision.** `ConstraintStatus ∈ {SATISFIED, VIOLATED, UNKNOWN}`. A missing attribute is
`UNKNOWN` and never silently a pass or a fail, in ranking or in the UI.

**Alternatives.** Boolean matching, treating missing data as a failure (or as a pass).

**Why.** Yelp attribute coverage is sparse: many businesses have no `NoiseLevel`, no
`WiFi`, sometimes no hours at all. Treating missing as "fails the filter" discards large
numbers of genuinely good results; treating it as "passes" fills results with places that
may well violate the request. Neither is defensible, and the distinction is also an
honesty requirement for the "why this result" explanation — "no data" is the truth.

**How it could be tested.** `retrieval.missing_attribute_policy` is a config knob
(`exclude` / `penalize` / `neutral`). Run the ablation under each and report the effect.
Default is `penalize`, chosen as a provisional middle ground — see D007.

---

## D007 — Missing-attribute policy defaults to `penalize` · Provisional
*2026-10-15*

**Decision.** Keep businesses with missing data for a requested constraint, but multiply
their fused score by `missing_attribute_penalty` (0.5).

**Why provisional.** This is a guess, not a result. It is wired as a config option
precisely so it can be measured rather than asserted. The number 0.5 has no
justification yet beyond "between keeping and dropping."

**How it could be tested.** Three eval runs, one per policy, plus a small sweep over the
penalty. Report the chosen value with the measurement that chose it. **Until that
measurement exists, no claim about this setting goes in the README.**

---

## D008 — Strict config validation, artifacts keyed by config hash · Settled
*2026-10-15*

**Decision.** Config is a pydantic model with `extra="forbid"`; `Config.fingerprint()`
hashes the whole config and namespaces every index, model, and eval report.

**Alternatives.** Plain YAML dict; argparse flags.

**Why.** A silently-ignored config typo means an experiment that did not test what it
claimed to test — the worst possible failure for this project, because it produces a
number that looks fine and is wrong. Hashing the config into artifact paths means any
figure can be traced back to the settings that produced it, which is what makes "rerun
and get the same answer" an actual guarantee rather than a hope.

---

## D009 — `data.metro` ships as `null` and fails loudly · Settled
*2026-10-15*

**Decision.** The committed config has no metro. `Config.require_metro()` raises with an
instruction to run `make explore` first.

**Why.** Picking the metro is the owner's call at the Phase 1 checkpoint, made after
seeing business counts and attribute coverage per city. A default here would quietly
decide a scoping question that should be decided on evidence. Failing loudly is better
than indexing the wrong city and discovering it after labeling has started.

---

## D010 — RRF over weighted score fusion · Settled (restates the brief's reasoning)
*2026-10-15*

**Decision.** Fuse BM25 and dense rankings with Reciprocal Rank Fusion,
`score = Σ 1/(k + rank)`, `k = 60`.

**Alternatives.** Weighted sum of normalized scores; learned fusion.

**Why.** BM25 scores are unbounded and corpus-dependent; cosine similarities sit in
[-1, 1]. Adding them requires a normalization choice that is itself a hyperparameter, and
min-max normalizing per query makes scores depend on which documents happened to be
retrieved. RRF uses only *ranks*, so it is invariant to both scales. `k = 60` is the value
from the original RRF paper (Cormack et al., 2009) and is a deliberate starting point, not
a tuned result. Learned fusion is what Stage 4 does anyway, with more features.

**How it could be tested.** Compare RRF against weighted-sum fusion on the gold set, and
sweep `k`. Worth doing because "why RRF?" is a near-certain interview question, and
"because I measured the alternative" is a much better answer than citing the paper.

---

## D011 — Exact FAISS index first (`IndexFlatIP`) · Settled
*2026-10-15*

**Decision.** Start with exact inner-product search over L2-normalized vectors. Move to
HNSW only if latency demands it, and measure the recall loss when doing so.

**Why.** One metro is on the order of tens of thousands of businesses — small enough that
exact search is fast. Starting approximate would mean every later quality number carries
an unmeasured approximation error baked in, and it would be impossible to tell whether a
disappointing result came from the ranking or from the index. Exact first gives a
correctness baseline that the approximate version can then be measured against.

---

## D012 — Parallel module development against frozen contracts · Settled
*2026-10-15*

**Decision.** `concierge/types.py` and `concierge/config.py` were written first and are
treated as fixed interfaces; the stage modules were then developed concurrently against
them.

**Why.** The four stages are genuinely independent given a shared vocabulary for
businesses, parsed queries, candidates, and results. Without that vocabulary fixed first,
each module invents its own and the integration seams become the buggiest part of the
system.

**Consequence to watch.** Any module that wants to change a shared type must change it in
one place, for everyone. If a contract turns out to be wrong, fix the contract — do not
let a stage work around it locally.
