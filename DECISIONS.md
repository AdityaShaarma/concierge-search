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

**Decision.** The owner writes all ~150 queries up front, then labels a
**random, fixed-seed** 50-query subset at full pool depth 20, expanding toward 150 as
time allows. All reported metrics state how many queries they are computed over.

**Alternatives.** (a) Label all 150 before reporting anything (many hours of labeling
before a single number exists; `concierge-label --stats` reports the measured pace once
there is history). (b) Cut pool depth from 20 to 10 across all 150 queries.
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
penalty — but only on the **retrieval-only** ablation rows, and that caveat is part of
the experiment rather than a footnote to it. `penalize` multiplies the fused Stage-2
score (`SCORE_RRF`) and nothing else, and `SCORE_RRF` is not a Stage 4 feature; so on a
reranked row `penalize` differs from `neutral` only through the `rrf_rank` feature and
through tie-breaks, and a small gap there says how little of the Stage-2 ordering
survives reranking, *not* that missing attribute data does not matter. The experiment to
run is therefore the retrieval-only triple, plus — separately — a Stage-4 run with and
without the `has_unknown_constraint_data` feature, which is where a *learned* discount
would live. See the "Where the `penalize` policy acts" section of
`concierge/retrieval/filters.py` for the full argument.

One confound has to be removed before any of those numbers mean what D007 says. A
lexical category miss is `UNKNOWN` by design (D-filters, `check_categories`), so wherever
it reaches the missing-data test it trips the penalty and sets
`has_unknown_constraint_data` alongside genuine attribute sparsity — and the policy is
then partly measuring gaps in Yelp's category taxonomy rather than sparse attributes.
`concierge/retrieval/filters.py` names the intended scope as `POLICY_EXEMPT_CONSTRAINTS`
and reports category UNKNOWNs separately in `FilterReport.unknown_by_constraint`; the
policy and the Stage-4 feature both have to honour it before `n_penalized` means
"attribute data is missing" and not "the category lexicon missed". Confirm they do
(`grep -n POLICY_EXEMPT_CONSTRAINTS concierge/retrieval/filters.py concierge/ltr/features.py`)
before running the three policies, rather than assuming it. Report the chosen value with
the measurement that chose it. **Until that measurement exists, no claim about this
setting goes in the README.**

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

**Why.** The index covers every business in the metro (`make explore` prints how many),
and exact search over one metro is expected to be fast enough that approximation buys
nothing — a claim the latency curve measures rather than assumes. Starting approximate
would mean every later quality number carries an unmeasured approximation error baked in,
and it would be impossible to tell whether a disappointing result came from the ranking or
from the index. Exact first gives a correctness baseline that the approximate version can
then be measured against.

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


---

# Module decisions
These were produced while building each module and are grouped by subsystem. They are the
detail behind the structural decisions D001-D012 above. Read the group for whatever you are
being asked about rather than front to back.
- [Data layer — attributes and hours](#data-layer--attributes-and-hours) — D013–D028
- [Data layer — loading and exploration](#data-layer--loading-and-exploration) — D029–D043
- [Data layer — business documents](#data-layer--business-documents) — D044–D051
- [Stage 1 — query schema and rule parser](#stage-1--query-schema-and-rule-parser) — D052–D074
- [Stage 1 — LLM client, cache and fallback](#stage-1--llm-client-cache-and-fallback) — D075–D085
- [Stage 2 — BM25, fusion and filters](#stage-2--bm25-fusion-and-filters) — D086–D099
- [Stage 2 — dense embeddings and FAISS](#stage-2--dense-embeddings-and-faiss) — D100–D111
- [Stage 3 — cross-encoder reranker](#stage-3--cross-encoder-reranker) — D112–D126
- [Stage 4 — LambdaMART and features](#stage-4--lambdamart-and-features) — D127–D136
- [Training data — synthetic queries, negatives, judge](#training-data--synthetic-queries-negatives-judge) — D137–D153
- [Evaluation — metrics and statistics](#evaluation--metrics-and-statistics) — D154–D164
- [Evaluation — pooling and labeling CLI](#evaluation--pooling-and-labeling-cli) — D165–D176
- [Evaluation — ablation runner and reporting](#evaluation--ablation-runner-and-reporting) — D177–D187
- [Pipeline and data build](#pipeline-and-data-build) — D188–D200
- [Serving — FastAPI](#serving--fastapi) — D201–D208
- [Frontend](#frontend) — D209–D217
- [Documentation](#documentation) — D218–D219

## Data layer — attributes and hours

### D013 — ast.literal_eval with a narrow failure-path fallback, never eval

**Decision.** Attribute values are decoded with ast.literal_eval inside try/except. eval is never used. A regex that strips legacy string prefixes runs ONLY on strings literal_eval has already rejected, not on every value.

**Alternatives.** (a) eval — rejected: these strings come from a third-party data file nobody here wrote, so eval is arbitrary code execution. (b) Blanket regex pre-processing of every value before parsing — rejected: a regex applied to all values also rewrites the interiors of legitimate strings; restricting it to the failure path bounds the blast radius to values that were already unparseable.

**Why.** literal_eval walks the parsed AST and accepts only literal nodes, so the worst case on hostile input is a ValueError rather than code execution. Keeping the repair regex on the failure path means the common case is handled by the standard library alone.

**How it could be tested.** test_literal_eval_does_not_execute_payloads feeds a payload that would write a file to disk under eval, and asserts the file does not exist and the value is kept as inert text.

### D014 — The u-prefix needs no stripping — the common claim about it is wrong

**Decision.** Do not pre-strip the u-prefix from "u'free'". PEP 414 restored the u'...' prefix to Python 3, so ast.literal_eval("u'free'") returns 'free' unaided. The regex fallback exists for the legacy combined prefix ur'...', which genuinely is a Python 3 SyntaxError.

**Alternatives.** (a) Regex-strip the u before every parse, as the dataset folklore (and this task's brief) assumes it is required — rejected after measuring: it is not required, and the regex adds a way to corrupt real strings for no benefit. (b) Strip with str.removeprefix('u') — rejected: it does not reach u-prefixes nested inside a stringified dict such as "{u'garage': False}".

**Why.** Verified empirically on the project interpreter (CPython 3.11): u'free', {u'a': u'b'} and u'can\'t' all parse directly; only ur'x' fails. Encoding a false premise into the parser would add untested dead code on the hot path.

**How it could be tested.** test_u_prefixed_literal_is_valid_python3 pins ast.literal_eval("u'free'") == 'free' so the fallback cannot silently become the hot path; test_legacy_ur_prefix_uses_the_regex_fallback asserts ur'free' raises SyntaxError bare and still normalizes through parse_attributes.

### D015 — Missing means the key is absent from the returned dict

**Decision.** parse_attributes never maps a key to None. The string "None", a JSON null, an empty string, and an unusable price all cause the key to be omitted. A sentinel class (_MISSING), not None, is used internally to carry that state.

**Alternatives.** (a) Keep the key with a None value — rejected: every consumer would then need `attrs.get(k) is not None` and the first one to write `attrs.get(k)` turns 'no data' into a falsy fail. (b) Use None as the internal 'missing' marker — rejected: None is itself one of the inputs being classified, so `value is None` would be ambiguous inside the normalizer.

**Why.** Membership (`k in attrs`) is the cheapest possible encoding of the three-valued contract in types.py: present-and-true, present-and-false, absent. ConstraintStatus.UNKNOWN downstream is derived from absence. Conflating missing with False is called out in types.py as the single most common source of wrong results in this dataset.

**How it could be tested.** test_none_string_becomes_missing_not_a_value (fixture b008), test_json_null_value_becomes_missing, test_empty_string_value_becomes_missing, and test_false_is_a_value_not_a_missing (fixture b002 OutdoorSeating=False stays present).

### D016 — Alcohol='none' is a value; only Python None is missing

**Decision.** The parsed *string* 'none' is kept as a value. Only Python None (which the literal string "None" decodes to) is treated as missing.

**Alternatives.** Treat the string 'none' as missing too, since it reads like absence — rejected: Alcohol='none' is a real Yelp value meaning 'serves no alcohol', and dropping it would make a dry restaurant indistinguishable from one with no alcohol data, breaking both filtering and the 'why this result' explanation.

**Why.** In the raw file "None" and "u'none'" look almost identical and mean opposite things. literal_eval already separates them (NoneType vs str), so the rule is 'trust the type, not the spelling'.

**How it could be tested.** test_alcohol_none_is_a_real_value asserts {'Alcohol': "u'none'"} -> {'Alcohol': 'none'} and {'Alcohol': 'None'} -> {}.

### D017 — Nested attribute dicts are flattened to dotted keys and the parent is dropped

**Decision.** "{'romantic': False, 'casual': True}" under Ambience becomes Ambience.romantic / Ambience.casual. The separator is the module constant ATTRIBUTE_PATH_SEP. Flattening recurses, and None leaves are dropped like any other missing value. The parent key is not retained.

**Alternatives.** (a) Keep the nested dict as-is — rejected: filters, the document renderer, the coverage report and the LambdaMART feature builder would each re-walk it and each invent its own leaf naming. (b) Keep both parent and flattened leaves — rejected: it makes coverage() double-count and leaves two sources of truth for one fact.

**Why.** A user constraint targets exactly one leaf ('casual ambience', 'street parking'), so the leaf is the right unit of addressing. One flat dict[str, Any] also matches Business.attributes in the shared contract without special cases.

**How it could be tested.** test_nested_dicts_flatten_to_dotted_keys (fixture b001), test_nested_none_leaves_are_dropped, test_nested_dict_of_only_none_leaves_yields_nothing, test_deeply_nested_dicts_flatten_recursively, and test_coverage_includes_flattened_keys.

### D018 — A canonical value vocabulary exported as a module constant, with out-of-vocabulary values kept

**Decision.** ATTRIBUTE_VOCABULARY maps each string attribute to its allowed spellings (WiFi free/paid/no, NoiseLevel quiet/average/loud/very_loud, Alcohol, RestaurantsAttire, Smoking, BYOBCorkage). Values are slugged (lowercase, non-alphanumerics to underscores) then alias-mapped. A value that still falls outside the vocabulary is KEPT, not dropped. canonical_value() is exported so the query parser normalizes LLM output through the identical code path.

**Alternatives.** (a) Drop out-of-vocabulary values — rejected: absence means UNKNOWN downstream, which is a stronger claim than the data supports; we know something, just not a spelling we recognize. (b) Let each module normalize its own spellings — rejected: a 'very loud' vs 'very_loud' mismatch produces zero results, raises nothing, and is nearly invisible in review.

**Why.** Constraint matching compares a parsed query value against a parsed business value. If the two sides can disagree about spelling, the filter silently empties the candidate set. One constant, one function, both sides.

**How it could be tested.** test_spelling_variants_canonicalize, test_out_of_vocabulary_value_is_kept_not_dropped, test_canonical_value_matches_parse_attributes, and test_fixture_values_all_land_inside_the_declared_vocabulary, which re-parses every fixture business and asserts each vocabulary-keyed value is in the declared tuple.

### D019 — Booleans on string-valued attributes are translated only where the reading is unambiguous

**Decision.** A bool where the vocabulary expects a string is left as a bool, except for WiFi, where True->free and False->no via an explicit _BOOL_ALIASES table.

**Alternatives.** (a) Guess for every attribute (NoiseLevel=True -> 'loud'?) — rejected as inventing data. (b) Guess for none, leaving WiFi=True as a bool — rejected: it would hide those businesses from every 'with wifi' query, and 'has WiFi' has exactly one reading.

**Why.** The project rule is never to invent data; the narrow exception is documented in one table with the reasoning inline, so an interviewer can see where the line was drawn and why.

**How it could be tested.** test_wifi_boolean_is_translated_but_noiselevel_boolean_is_not asserts WiFi True->'free', False->'no', and NoiseLevel True stays the bool True.

### D020 — Price is coerced to int; anything outside 1-4 reads as missing

**Decision.** RestaurantsPriceRange2 is coerced to an int and must land in 1..4, otherwise the key is omitted.

**Alternatives.** Keep the raw string — rejected: ParsedQuery.price_max is compared arithmetically, and '2' < 2 raises TypeError in Python 3, so the coercion has to happen somewhere; doing it once here beats doing it in every filter.

**Why.** An unusable price is not data. Letting '0' or '$$' through as a value would make a business look cheap rather than unknown, which is a wrong pass rather than an honest UNKNOWN.

**How it could be tested.** test_price_coerces_to_int over '1'-'4' and ints; test_unusable_price_is_missing over '0', '5', 'abc', "'$$'", 'None'.

### D021 — coverage() divides by every business seen, including those with a null attributes blob

**Decision.** Fill rate = (businesses where the key is present) / (all businesses passed in). Output is ordered by descending fill rate then alphabetically, and the helper accepts raw records or already-parsed Business objects.

**Alternatives.** Divide only by businesses that have at least one attribute — rejected: it flatters the dataset and misstates the number the failure analysis actually needs.

**Why.** The question the failure analysis asks is 'what fraction of the corpus can answer a NoiseLevel constraint at all', and b005-style records with attributes == null are precisely the ones that cannot. Stable ordering keeps the generated report byte-comparable between runs, which the reproducibility rule requires.

**How it could be tested.** test_coverage_counts_all_businesses_in_the_denominator checks hand-counted rates against the 8-business fixture corpus (price 7/8, NoiseLevel 6/8, WiFi 5/8, OutdoorSeating 3/8 — b008's "None" string excluded); plus tests for sort order, flattened keys, the empty input, and Business objects. No real-data coverage number is claimed anywhere.

### D022 — is_open_at DOES handle overnight spillover, by inspecting the previous day

**Decision.** At (day D, minute m) a business is open if (a) a non-crossing interval on D covers m, or (b) a crossing interval on D has start_min <= m (its evening half only), or (c) a crossing interval on day D-1 has m < end_min (its morning half). So a Friday 17:00-02:00 shift makes the business open at Saturday 01:00.

**Alternatives.** (a) Ignore spillover and treat each day's string in isolation — rejected: 'ramen at 1am on Saturday' would miss exactly the businesses it is looking for, since overnight places are the only ones open then. (b) Call HoursInterval.covers() on the requested day and stop there — rejected as actively wrong: covers() answers 'is this minute inside the interval on the interval's own clock', which merges the two calendar dates a crossing interval touches. It would report Monday 01:00 open merely because Monday also has a 17:00-02:00 shift, when Monday 01:00 is in fact governed by Sunday's hours.

**Why.** This is the one place where the shared contract's helper is not sufficient on its own: covers() is interval-relative, while the question asked is day-relative. Splitting a crossing interval into its evening half (today) and morning half (tomorrow) is the only formulation that gets both directions right.

**How it could be tested.** Fixture b002 drives it: test_overnight_spills_into_the_following_day (Saturday 01:00 open via Friday), test_spillover_boundaries (00:00 open, 02:59 open, 03:00 closed, 03:01 closed, 10:00 closed), test_a_days_own_overnight_shift_does_not_open_that_days_morning (asserts monday.covers(60) is True — the trap — while is_open_at returns VIOLATED), and test_spillover_across_the_week_boundary on b006 for the Sunday->Monday wrap.

### D023 — "0:0-0:0" is open 24h, represented as HoursInterval(0, 1440) and branched on explicitly

**Decision.** parse_interval has a dedicated branch for start == 0 and end == 0 returning OPEN_24H = HoursInterval(0, MINUTES_PER_DAY). Separately, any interval with end == 0 (e.g. '17:0-0:0') gets end_min = 1440, meaning 'midnight at the end of this day'.

**Alternatives.** (a) Let 0:0-0:0 fall out of the generic rule, which coincidentally produces the same interval — rejected: the most surprising encoding in the dataset would then be undocumented and untested. (b) Keep end_min = 0 for midnight closes — rejected, and this is a real bug it prevents: HoursInterval.crosses_midnight is `end_min <= start_min`, so end_min = 0 makes a 24-hour day and a midnight-closing day both report as crossing, which leaks them into the next morning. b002's Sunday 17:00-00:00 would wrongly show as open Monday 01:00, and a 24-hour Monday would wrongly show as open on a closed Tuesday.

**Why.** end_min = 1440 makes crosses_midnight False for both cases, so neither spills, while covers() still answers correctly within the day. One representation choice fixes two otherwise separate off-by-a-night bugs.

**How it could be tested.** test_zero_to_zero_is_open_24h, test_interval_ending_exactly_at_midnight_does_not_cross, test_open_24h_is_always_open across all seven days of b007, test_open_24h_does_not_spill_into_a_closed_day, and test_interval_ending_at_midnight_does_not_spill (b002 Sunday 23:59 open, Monday 00:00 closed).

### D024 — Empty parsed hours means UNKNOWN; a day absent from a populated schedule means closed

**Decision.** parse_hours returns {} for hours == null. is_open_at returns ConstraintStatus.UNKNOWN when the dict is empty, and VIOLATED when the dict is non-empty but the requested day (and the previous day's overnight) do not cover the minute. has_hours_data(raw) is exported as a separate predicate over the RAW blob.

**Alternatives.** (a) Return a sentinel object or Optional from parse_hours to carry 'no data' — rejected: Business.hours in the fixed contract is typed dict[DayOfWeek, tuple[HoursInterval, ...]], so a sentinel cannot be stored there. (b) Have a business with no hours read as closed — rejected: it deletes every hours-less business from every time-constrained query on no evidence. (c) Read it as open — rejected: it fills results with places that are shut.

**Why.** The distinction is expressible from the parsed dict alone — 'no hours at all' is an empty dict, 'closed Tuesdays' is a non-empty dict without Tuesday — so no contract change is needed. Whether UNKNOWN is excluded, penalized or kept neutral is left to retrieval.missing_attribute_policy in the config, which is a measured variable rather than something baked into the parser. has_hours_data exists because the parsed form cannot distinguish 'no hours field' from 'hours field we could not parse' (both empty), which data-quality reporting needs.

**How it could be tested.** test_no_hours_data_is_unknown_never_closed (b005, all seven days), test_day_absent_from_a_populated_schedule_is_closed (b003 has no Tuesday -> VIOLATED, not UNKNOWN), test_has_hours_data_separates_no_field_from_unparseable.

### D025 — A malformed day string is dropped rather than raising or guessed at

**Decision.** parse_interval returns None for anything that does not match ^H:M-H:M$ with in-range components, and parse_hours omits that day. Unknown day keys are ignored.

**Alternatives.** (a) Raise — rejected: one bad row anywhere in the raw file would abort the whole data build. (b) Substitute default hours — rejected as inventing data.

**Why.** The cost is that the day then reads as closed, which is a slightly wrong claim. The alternative is a worse claim, and parse_hours reports exactly what it parsed, so the loss is measurable rather than hidden.

**How it could be tested.** test_malformed_intervals_return_none over nine malformed inputs, test_unparseable_days_are_dropped_not_invented, test_unknown_day_keys_are_ignored.

### D026 — is_open_at raises ValueError on a bad day or out-of-range minute, rather than returning UNKNOWN

**Decision.** An unrecognized day name or a minute outside 0..1439 raises ValueError. Bad *data* is tolerated; bad *caller input* is not.

**Alternatives.** Return UNKNOWN for both — rejected: a broken query parser emitting 'Funday' or minute 1500 would then produce plausible-looking results forever and never surface in a test.

**Why.** These two inputs come from our own code (the Stage 1 parser), not from the dataset, so they are programmer errors. Failing loudly is how a cross-stage contract break gets noticed on the first run instead of in the failure analysis.

**How it could be tested.** test_bad_day_raises_rather_than_reporting_unknown, test_out_of_range_minute_raises parametrized over -1, 1440 and 10000.

### D027 — Day names are normalized centrally, accepting both 'Fri' and 'Friday' in any case

**Decision.** canonical_day() maps full names and three-letter abbreviations, case- and whitespace-insensitively, to the dataset's spelling. Both parse_hours keys and the is_open_at day argument go through it.

**Alternatives.** Require the dataset's exact spelling everywhere — rejected: CLAUDE.md's own Stage 1 JSON example emits {"day": "Fri"} while the dataset writes "Friday", so the mismatch is already designed into the pipeline and would surface as a silent zero-result bug.

**Why.** A mismatch between what the LLM emits and what the dataset stores is the kind of integration bug that returns an empty list and raises nothing. One lookup table removes the whole class.

**How it could be tested.** test_day_argument_accepts_abbreviations_and_any_case ('Fri', 'friday', ' FRIDAY '), test_abbreviated_day_keys_are_normalized, test_canonical_day.

### D028 — parse_user_time returns None for unrecognized input instead of a default

**Decision.** parse_user_time accepts '10pm', '10 p.m.', '22:00', '22', '10:30pm', 'noon', 'midnight' and returns a minute-of-day, or None. A bare number is read on a 24-hour clock.

**Alternatives.** (a) Default to a fixed time or raise — rejected: a query parser that cannot read the time must drop the open_at constraint entirely, not filter the whole corpus by a time the user never asked for (a hard filter on the wrong hour is far more damaging than no filter). (b) Guess am/pm for a bare number from context — rejected as unmeasurable guesswork; an LLM asked for a 24-hour time should emit one, and a human writing '9' is ambiguous to a human too.

**Why.** Making the failure explicit in the return type forces the caller to decide, and the decision it should make (drop the constraint) is written into the docstring.

**How it could be tested.** test_parse_user_time over 16 accepted forms including the 12am/12pm edge cases; test_unparseable_user_time_returns_none over '', 'late', '24:00', '25', '13pm', '0pm', '10:60', 'tenpm', 'half past nine'; test_user_time_round_trips_into_is_open_at exercises the real 'open past 10pm on a Friday' call path end to end.

## Data layer — loading and exploration

### D029 — A metro is an explicit city list plus allowed states, matched on a normalized city string

**Decision.** `MetroDefinition(name, states, cities)` in concierge/data/load.py, with 11 seed definitions keyed by normalized name. `resolve_metro(name)` returns a curated definition if one exists, otherwise falls back to treating the name as a single city (any state). The filter is pushed into DuckDB SQL; the same predicate also exists in Python (`MetroDefinition.matches`) for the report, and a test pins the two against each other on the fixtures.

**Alternatives.** (a) state only - wrong in both directions: one state can hold two of the dataset's metros, and real metros cross state lines (Philadelphia into NJ/DE, Saint Louis into IL). (b) exact match on the core city string - drops the suburbs that make up much of a metro and is brittle against dirty city strings. (c) radius around a city centroid using the lat/lon already carried - immune to city-string noise but needs a per-metro centroid and an unprincipled radius knob that silently sweeps in whatever is nearby.

**Why.** Yelp's "11 metro areas" is a sentence in the dataset documentation, not a column: records carry only city and state, so a metro has to be defined by us, and the definition changes how many businesses are searchable. An explicit city list is legible and checkable - the definition is data you can read, and explore prints the per-city counts the lists were built from. Crucially it also prints an "(unassigned)" row: businesses in cities no definition claims. That is the honesty check - if it is large, the lists are wrong and the report says so instead of hiding it. The city lists are seed geography, NOT a claim about what is in the download; which metros actually appear is answered by running explore.

**How it could be tested.** tests/test_load.py::TestMetroResolution and TestLoadBusinesses - suburbs included, same-named city in another state excluded by the state set, unknown name falls back to single-city, unmatched metro returns empty rather than raising, and test_sql_filter_agrees_with_python_matching pins the SQL and Python predicates together. A needs_data test asserts that under 50% of open businesses are unassigned on the real download, with a message pointing at the per-city table; the actual share is reported by `make explore`, never asserted as a number.

### D030 — City-string normalization folds case, accents, trailing punctuation and Saint/St.

**Decision.** `normalize_city` applies NFKD + combining-mark stripping, lowercase, whitespace collapse, trailing-punctuation strip, and a leading `st.`/`st `/`ste.` to `saint `/`sainte ` fold. The same normalization is reimplemented in SQL so the filter runs during the scan rather than in Python.

**Alternatives.** Match the raw string; or lowercase only.

**Why.** "St. Louis", "St Louis" and "Saint Louis" are one city typed three ways and all three occur in the wild; "Montreal" with and without the accent likewise. Matching raw strings would silently drop a slice of a metro, which looks like missing data rather than a lookup bug. Duplicating the rule in SQL is a real duplication cost, accepted because it is what makes the filter a pushdown rather than a post-filter - and a test pins the two implementations together so they cannot drift.

**How it could be tested.** tests/test_load.py::TestCityNormalization (parametrized, including that "Stockton" is not read as "Saint ockton") and test_metro_scoping_is_case_insensitive_in_sql_too.

### D031 — Every read goes through DuckDB with an explicit column projection and an Arrow-batch stream

**Decision.** `read_json(path, format='newline_delimited', columns={...})` with a hardcoded column map per file, results consumed via `to_arrow_reader(batch_rows)` (falling back to `fetch_record_batch` on older DuckDB). review.json/tip.json reads semi-join a registered Arrow table of in-scope business ids. `fetchall()` is never used on a review query.

**Alternatives.** `read_json_auto` with schema sniffing; Polars `scan_ndjson`; line-by-line `json.loads`; pandas.

**Why.** review.json is ~5GB / ~7M rows and the pipeline only ever needs business_id, text, useful and date, and only for the businesses in the one metro in scope (`make explore` prints how many that is). An explicit column list makes DuckDB skip the unwanted keys so they never become Python objects; the semi-join drops out-of-scope rows inside the scan; Arrow batches keep peak memory at one batch. Explicit columns are also preferred over read_json_auto because sniffing a 5GB file costs a scan and an inferred schema can drift between dataset releases and silently change types. Time stays proportional to the file; memory does not. Those costs are not measured yet - the module makes the measurement possible and claims no number.

**How it could be tested.** tests/test_load.py::TestLoadReviews - test_only_the_projected_columns_come_back asserts user_id/funny/cool never arrive; test_small_batch_size_does_not_change_the_result pins batch size as a memory knob, not a correctness knob; test_empty_id_set_never_opens_the_file checks the short-circuit. A needs_data test streams real reviews to assert it completes without OOM.

### D032 — `attributes` and `hours` are read as JSON text and decoded in Python, not as DuckDB STRUCTs

**Decision.** Both columns are declared `JSON` in the column map and json.loads-ed per row; a SQL NULL stays Python `None`.

**Alternatives.** Let DuckDB infer a STRUCT per column; flatten the attributes into fixed columns at read time.

**Why.** Attribute and hours keys vary per business, so any struct schema is a lossy guess - a schema inferred from a sample would drop unusual keys. Keeping `None` distinct from `{}` matters because D006's three-valued constraint logic rests on "no data at all" being different from "an empty dict", and b005 in the fixtures exists to pin exactly that.

**How it could be tested.** tests/test_load.py::TestBusinessFieldsSurvive - arbitrary keys (Ambience, BusinessParking) preserved, null attributes/hours stay None, overnight "17:0-2:0" and 24h "0:0-0:0" strings unmangled, lat/lon arrive as floats.

### D033 — load_businesses materializes a list; reviews and tips are generators with a materializing wrapper

**Decision.** `load_businesses` returns `list[dict]`. Reviews/tips expose `iter_reviews_for` / `iter_tips_for` as the primary API with `load_*_for` as thin `list()` wrappers whose docstrings state when they are safe.

**Alternatives.** Return iterators everywhere; return a DuckDB relation and let callers decide.

**Why.** business.json holds one row per business in the whole download (the dataset's own documentation says ~150K) and every downstream consumer (document builder, BM25 index, FAISS index) needs all of it at once, so an iterator would only force each caller to materialize it anyway. review.json is far larger - the same documentation puts it at ~5GB / ~7M rows - and is consumed per-business, so there the streaming form has to be the default and the convenience form has to carry a warning. Neither file has been counted here (`make explore` prints the real figures once the download exists); the asymmetry is what the design rests on, not the ratio.

**How it could be tested.** test_iter_streams_rather_than_materializing and test_load_business_ids_matches_load_businesses pin the two forms to the same results.

### D034 — Per-business review cap and ordering are applied inside DuckDB, with a hash-based deterministic "random"

**Decision.** `max_per_business` wraps the scan in `row_number() OVER (PARTITION BY business_id ORDER BY <order>)`. Orders: useful_then_recent (default, matches cfg.documents.review_selection), recent, and random implemented as `ORDER BY hash(review_id)`.

**Alternatives.** Load all reviews for the in-scope businesses and sort/slice in Python; use SQL `random()`.

**Why.** The rows that lose the cut never become Python objects, and DuckDB spills the window operator to disk if it has to - plain Python sorting cannot. `hash(review_id)` rather than `random()` because the project requires every result to be regenerable: a randomly-sampled review set would make document text, and therefore every downstream metric, unreproducible across runs.

**How it could be tested.** test_max_per_business_keeps_the_most_useful, test_max_per_business_applies_per_business, test_recent_order_picks_the_newest, test_random_order_is_deterministic, test_unknown_order_is_rejected.

### D035 — count_reviews_by_business is separate from the review_count field, and explore reports both

**Decision.** A streaming `GROUP BY business_id` over review.json projecting only business_id, exposed behind an opt-in `--count-reviews` flag. The report carries Yelp's `review_count` and the measured row counts in separate, separately-labelled columns; when the flag is off, the row columns print "not measured" and a caveat says so.

**Alternatives.** Report `review_count` alone and call it "reviews per business"; always scan review.json.

**Why.** `review_count` is Yelp's own total on their site, while review.json contains only a subset - so the two genuinely differ, and conflating them would overstate how much review text the business documents can actually draw on. Making the scan opt-in keeps `make explore` fast at the checkpoint while leaving the real number one flag away. A "not measured" string rather than a zero is the point: an unmeasured quantity must never render as a measured one.

**How it could be tested.** test_counts_rows_not_the_yelp_review_count_field (b001 has review_count=312 but 2 rows), test_review_rows_are_absent_unless_measured, test_yelp_review_count_is_kept_distinct_from_rows, and `median_or_none([]) is None` so an empty metro never shows a median of 0.

### D036 — explore delegates attribute and hours logic to the shared modules, probes them before trusting them, and always prints which path produced each number

**Decision.** concierge.data.attributes and concierge.data.hours are imported in guarded try/except blocks. Attribute coverage prefers `attributes.coverage(records)`; the per-record parser and the hours predicates are resolved once at import and only after known-answer probes pass. Every path sets a source label that appears in the report header, the caveats, and the JSON artifact (attribute_source, hours_source, attribute_fill_is_complete). A coverage result is accepted only if every value is a fraction in [0,1].

**Alternatives.** (a) Hard-import the shared modules and let a half-written one crash the checkpoint report. (b) Reimplement attribute and hours parsing locally. (c) Trust whatever coverage() returns.

**Why.** Those modules own every rule about how this dataset spells its values; two definitions of "has a NoiseLevel" would eventually disagree, and the one in the report is the one the owner reads at the checkpoint. But the modules are written in parallel with this one, so a hard import makes the report hostage to build order. The guarded import plus a labelled fallback gets both. Validation matters concretely: the landed coverage() returns fractions, and a revision returning counts is refused rather than divided by a guessed denominator - silently reinterpreting another module's units is exactly how a wrong number ends up looking right.

**How it could be tested.** tests/test_load.py::TestMeasureCoverage covers all four paths - valid fractions used and attributed, counts rejected into the fallback, a raising module tolerated, an alternative keyed signature accepted - plus TestBuildReport::test_provenance_of_every_number_is_recorded and test_incomplete_coverage_is_flagged_with_a_caveat.

### D037 — The overall attribute table is driven by whatever keys the shared coverage() finds, not a hardcoded list

**Decision.** Only the four headline keys (NoiseLevel, WiFi, OutdoorSeating, RestaurantsPriceRange2) are hardcoded, as the per-metro columns. The overall ranked table comes from coverage() output, sorted by fill rate then alphabetically. The fallback list is restricted to flat keys and deliberately excludes Ambience and BusinessParking.

**Alternatives.** Hardcode a list of ~10 interesting attribute keys for both tables.

**Why.** This was caught by running the report: the attributes module flattens nested attributes to dotted leaves ("Ambience.casual"), so a hardcoded Ambience row printed a confident 0.0% for data that is in fact populated - a plausible-looking wrong number, the worst failure mode for this project. Letting coverage() drive the table is both honest and strictly more informative (it surfaces every leaf), and the ranking by fill rate makes the artifact byte-stable and diffable between dataset releases. When the fallback is in use, the table title and a caveat both state that nested keys are invisible.

**How it could be tested.** test_fallback_list_contains_no_nested_container_keys, and test_incomplete_coverage_is_flagged_with_a_caveat asserts the caveat names Ambience.casual. Verified by eye against the fixtures: Ambience.casual/intimate/romantic and BusinessParking.* now appear with real rates.

### D038 — Hours are reported twice: "hours field" and "usable hours"

**Decision.** `has_hours_field` (any hours data, via hours.has_hours_data) and `has_usable_hours` (parses into at least one interval, via hours.parse_hours) are separate columns in both tables.

**Alternatives.** One "has hours" column, using either definition.

**Why.** The gap between the two is a finding, not noise: those are businesses that look like they have hours and still cannot answer "open past 10pm". Collapsing them would hide unusable data behind a reassuring percentage, and "open late" is a large share of the intended gold set.

**How it could be tested.** test_hours_field_is_weaker_than_usable_hours uses an unparseable hours value; test_closed_businesses_are_excluded_from_coverage_stats pins both counts on the fixtures.

### D039 — "Constraint-answerable" is a stated definition, printed next to its own number

**Decision.** A business is answerable if it has usable hours AND at least one of NoiseLevel / WiFi / OutdoorSeating. The definition is printed in the caveats block of every report and in the markdown artifact.

**Alternatives.** Report raw per-attribute fill rates only; require all three amenities; invent a weighted coverage score.

**Why.** The scoping decision needs one number that combines size and coverage, because the two can point at different metros - but any such number is a choice, not a measurement. Stating the definition next to the figure is what keeps it from being read as a dataset property. A weighted score would have been worse: it would look authoritative while hiding arbitrary weights.

**How it could be tested.** test_constraint_answerable_needs_hours_and_an_amenity (b004 has hours but only price, so not answerable) and test_answerable_count on the fixtures.

### D040 — explore loads with include_closed=True and computes searchable stats from the open subset internally

**Decision.** `load_businesses(cfg, ALL_METROS, include_closed=True)` once, then build_report filters to is_open == 1 for every coverage figure while reporting the closed count and share separately. build_report itself is pure - records in, value out, no file access.

**Alternatives.** Honor cfg.data.drop_closed in the load and never see closed businesses; make build_report do its own loading.

**Why.** "How much of this metro is permanently closed" is part of what the owner is deciding on, so the report needs the closed rows - but coverage percentages must be computed over businesses that can actually be returned, or a closed business with one price attribute drags down every rate for nothing. One pass over business.json serves both. Keeping build_report pure is what makes the whole report testable against tests/fixtures/ without the dataset download at all.

**How it could be tested.** test_open_and_closed_are_counted_separately, test_closed_businesses_are_excluded_from_coverage_stats, test_report_is_pure_and_json_serializable, and the CLI tests driving explore.run against the fixture corpus.

### D041 — ALL_METROS sentinel so exploration can ignore a configured metro

**Decision.** `metro=None` defers to cfg.data.metro (null in the committed config, D009, meaning no filter); `metro=ALL_METROS` ("*") forces no filter regardless of config.

**Alternatives.** A separate `all_metros: bool` flag; a private `_UNSET` sentinel object; require callers to pass cfg.model_copy(metro=None).

**Why.** Three states are genuinely needed - defer to config, scope to this metro, ignore the config - because exploration runs before the metro exists and must still see everything even after the owner sets one. `None` is already taken by "defer", and a sentinel object makes the type hint unreadable for a student who has to defend it.

**How it could be tested.** test_metro_defaults_to_the_config, test_all_metros_ignores_the_config_metro, test_null_config_metro_means_no_filter.

### D042 — MissingDatasetError subclasses FileNotFoundError and names the download, the directory state, and the expected filenames

**Decision.** `require_raw_file` distinguishes "directory does not exist" / "directory is empty" / "present but no business.json (found: ...)" and appends the yelp.com/dataset instructions plus the no-commit reminder. The explore CLI catches it and exits 2 with the message instead of a traceback.

**Alternatives.** Let DuckDB's parse error surface; a plain FileNotFoundError.

**Why.** An empty data/raw is the first thing any reader of this repo hits, including the owner on a new machine. A DuckDB IO error teaches them nothing. Subclassing FileNotFoundError means callers that only catch the builtin still work. Listing what was found catches the likely mistake of unzipping into a nested folder or keeping the plural fixture names.

**How it could be tested.** tests/test_load.py::TestMissingDataset (four cases) and TestExploreCli::test_missing_data_exits_nonzero_with_instructions; also run end to end - `python -m concierge.data.explore` against the real empty data/raw exits 2 with the instructions.

### D043 — Artifacts namespaced by config fingerprint; JSON and markdown twins

**Decision.** artifacts/explore/<cfg.fingerprint()>/dataset_report.json and dataset_report.md. The JSON keeps every attribute key; the terminal table caps at 25 rows with a line saying how many more are in the artifact.

**Alternatives.** A single latest.md; fingerprint in the filename rather than a directory.

**Why.** Follows D008 so any figure can be traced back to the settings that produced it. Two formats because they serve different readers: the markdown is for pasting into notes and the README, the JSON is machine-readable for later comparison between dataset releases. The display cap exists because a table with one row per attribute key found buries the four rows that matter.

**How it could be tested.** test_artifacts_are_written_under_the_config_fingerprint, test_writes_artifacts_when_asked, test_markdown_has_both_tables_and_the_caveats, test_markdown_states_where_the_numbers_came_from.

## Data layer — business documents

### D044 — Attributes render as English phrases, not key=value pairs

**Decision.** Normalized attributes become comma-joined phrases ("free WiFi, quiet, outdoor seating, casual, street parking, $$") via a public `attribute_phrases(attributes)` in concierge/data/documents.py, built on explicit phrase tables plus a camel-case fallback ("AcceptsInsurance" -> "accepts insurance").

**Alternatives.** Key-value dump ("WiFi=free NoiseLevel=quiet"); JSON blob in the text; omitting attributes from the document and relying only on hard filters.

**Why.** Both the bi-encoder and the cross-encoder were pretrained on natural language, so text closer to that distribution gets a better representation; "OutdoorSeating" also tokenizes into subword fragments and "True" carries no venue meaning. The lexical half matters just as much: BM25 matches terms, and users type "quiet cafe with wifi", never "NoiseLevel=quiet" — so one phrasing edit helps both retrieval arms. Omitting attributes entirely would leave vibe queries ("quiet spot with wifi") with nothing to match, since those are soft preferences the filters do not always get as hard constraints.

**How it could be tested.** Unit-tested for exact phrase output per fixture business (tests/test_documents.py::TestAttributePhrases) and asserted absent of "=", raw key names, and "True". The real test is an ablation: index the corpus with `attribute_phrases` vs. an injected key=value phraser (the `phrase_fn` seam exists for exactly this) and compare NDCG@10 and Recall@100. Unmeasured so far.

### D045 — Negative attributes are omitted from the document

**Decision.** A False boolean (OutdoorSeating=False) and a negating enum value (WiFi="no", Alcohol="none", Smoking="no") contribute no text at all. Only affirmative or descriptive values are phrased (quiet / loud / moderate noise are all descriptive, so all are kept).

**Alternatives.** Write the negation into the text ("no outdoor seating"); write it with a marker token; include everything and let the model sort it out.

**Why.** BM25 scores terms and has no notion of negation, so "no outdoor seating" is a strong lexical match for a query asking FOR outdoor seating — it would actively promote the wrong places. Dense bi-encoders handle negation poorly too. Nothing is lost: a hard "with outdoor seating" request is enforced arithmetically against Business.attributes by the Stage 2 filter, which is where a correct answer has to come from anyway.

**How it could be tested.** tests/test_documents.py::TestAttributePhrases::test_negative_attributes_are_omitted asserts b002 (OutdoorSeating=False, WiFi="no") renders only ["loud", "$"]. Measurable later by indexing both ways and checking how often a negative-attribute business appears in the top 10 for the corresponding positive constraint query.

### D046 — Document order: name, categories, attribute phrases, tips, review snippets

**Decision.** Fixed four-line layout, newline separated, with structured signal first and user prose last. Hours are not in the document at all.

**Alternatives.** Reviews first (they are the longest and most distinctive text); interleaving attributes with reviews; a single unstructured concatenation; including "Rated 4.5 stars by 312 reviewers" in the text.

**Why.** The cross-encoder's window is 512 tokens (rerank.max_length), so the tail of a long document is simply not read by the last stage that reads the document text at all — whatever has to survive must be at the head. Hours are excluded deliberately: "open past 10pm" is answered by comparing HoursInterval values, and a review that merely says "open late" must not be able to fake a satisfied hard constraint. Stars and review_count are also excluded: LambdaMART consumes them as numeric features, so duplicating them as text would double-count the signal in a stage less able to use it well.

**How it could be tested.** tests/test_documents.py::TestContentAndOrdering asserts phrase text precedes "Reviews:" and that no hours strings or weekday names appear. The ordering claim itself is ablatable: rebuild with reviews-first and compare NDCG@10 at the reranker stage. Unmeasured.

### D047 — Review selection: useful_then_recent by default, all three modes implemented

**Decision.** Two stable sorts (recency descending, then usefulness descending) give usefulness priority with recency as the tiebreak and review text as a final tiebreak so the order is total. `recent` and `random` are implemented too; `random` seeds from f"{cfg.seed}:{business_id}" over a canonically sorted pool so it is reproducible.

**Alternatives.** Random sampling (the common default); newest-first; highest-star reviews; longest reviews.

**Why.** A heavily-upvoted review is the one that explains why people go somewhere ("open until 2am which saved me after a late shift") — exactly the signal vibe and constraint queries need — whereas the newest review is often a one-line complaint. Random was rejected as a default but kept as an ablation baseline, and even it has to be reproducible because index artifacts are keyed by the config hash. Highest-star was rejected because it biases the document toward promotional prose and correlates with the rating feature LambdaMART already has.

**How it could be tested.** tests/test_documents.py::TestSelectionModes asserts each mode picks the expected review, and TestDeterminism asserts random is seed-stable and that reversing the input order never changes the document. The choice between modes is an ablation (documents.review_selection is a config key) and is unmeasured.

### D048 — Per-review share of the character budget, recomputed as snippets are added

**Decision.** After the head, attribute phrases, and tips are placed, the remaining budget is divided equally among the selected reviews, and the share is recomputed after each snippet so a short review hands its unused budget to the next one.

**Alternatives.** A separate max_chars_per_review config key; first-come-first-served until the budget runs out; a fixed character cap per review.

**Why.** Without a per-review share, a single 4000-character review swallows the whole window and crowds out the other three — and reviews disagree with each other usefully, so breadth beats depth here. The share is derived from existing config (max_chars, max_reviews) rather than being a new invented number, which keeps the config honest; if a fixed per-review cap ever looks better it should be added as a real config key and ablated, not hidden as a default here.

**How it could be tested.** tests/test_documents.py::TestTruncation::test_one_long_review_does_not_crowd_out_the_others builds one 400-word review plus two short ones and asserts all three appear. The alternative policies are unmeasured.

### D049 — Truncate at sentence, then clause, then word boundary; never mid-word

**Decision.** `_truncate` prefers the last sentence mark (. ! ?) in the window, then a clause mark (; , :), then the last space; a single word longer than the limit yields an empty string rather than a fragment. A mark only counts as a boundary when it ends the window or is followed by whitespace, so decimals and "p.m." are not mistaken for sentence ends. `BusinessDocument.truncated` is set whenever content was shortened or dropped — but NOT when max_reviews/max_tips merely selected fewer items, which is selection, not truncation.

**Alternatives.** Hard character cut; token-count-aware truncation using the reranker's tokenizer; no truncation (let each model truncate on its own).

**Why.** Half a word matches no BM25 term and tokenizes into subword junk for the embedder, so it is pure cost. Capping in characters rather than tokens keeps this module free of a tokenizer dependency (it is used by the data build, which should not need torch); the character cap is deliberately conservative relative to 512 tokens. Letting each model truncate independently would mean BM25, the embedder, and the cross-encoder scored different text, which would make the ablation uninterpretable.

**How it could be tested.** tests/test_documents.py::TestTruncation sweeps max_chars from 30 to 400 and asserts every word in the output is a word from the input (no mid-word cut), that the cap is never exceeded, and that the truncated flag is set only when content was actually cut. A token-vs-character comparison against the reranker's tokenizer is worth doing once the model is downloaded; unmeasured.

### D050 — Phrase order is sorted by (priority, key), not by raw attribute order

**Decision.** Phrases are emitted in a fixed priority order of attribute keys (WiFi, NoiseLevel, OutdoorSeating, Ambience, ... then everything else alphabetically), with price always last.

**Alternatives.** Keep the raw record's attribute order; alphabetical order throughout; no ordering guarantee.

**Why.** Two reasons. Reproducibility: attribute order in the dict comes from the raw JSON record, so sorting here means the document cannot change shape if the upstream key order ever changes — and index artifacts are namespaced by the config hash, which only means something if the inputs are stable. Quality: the earliest phrases are the ones certain to survive the reranker's window, so the attributes people actually search for should go first. Price goes last because a bare "$$" in the middle of a run of amenity words reads like a typo.

**How it could be tested.** tests/test_documents.py::TestAttributePhrases::test_phrase_order_is_independent_of_key_order reverses the dict and asserts identical output; test_price_is_last covers the price position.

### D051 — build_documents returns a list in input order

**Decision.** The batch builder returns list[BusinessDocument] aligned with the input businesses rather than a dict keyed by business_id.

**Alternatives.** Return a dict keyed by business_id; return a generator.

**Why.** BM25 and FAISS both address documents by row number, so the caller keeps one list of businesses and one list of documents and row i means the same business in the index, in the id map, and here. A dict would push that alignment onto the caller, and a misaligned id map is a bug that surfaces as plausible-looking nonsense results rather than as an error.

**How it could be tested.** tests/test_documents.py::TestBuildDocuments asserts ids line up with the input and that the batch path produces byte-identical text to calling build_document one at a time.

## Stage 1 — query schema and rule parser

### D052 — Constraint phrases are stripped from semantic_text; category words are not

**Decision.** After a rule encodes a constraint, its matched phrase is removed from `semantic_text` and the connector words left dangling go with it ("quiet café with wifi that's open past 10pm" -> "café"). Category words are the deliberate exception and are never stripped.

**Alternatives.** (a) Leave the query intact and embed it whole. (b) Strip everything a rule matched, categories included. (c) Replace stripped text with a canonical phrase such as "late night".

**Why.** Leaving "open past 10pm" in the embedded text does three distinct harms: it double-counts (the hours filter already enforces it exactly, so scoring it again as text similarity over-weights it against the rest of the request and the ablation can no longer attribute the effect to one stage); it rewards talking over being (a document containing a review that says "open late!" outranks one whose structured hours actually satisfy the request); and it dilutes BM25, spending term weight on "open", "past", "10pm" so the discriminative word ("café") gets a smaller share. Categories are excluded because the category filter is coarse (Yelp category lists are noisy and incomplete) and the word carries meaning the category cannot -- "ramen" is far more specific than "Japanese" -- and stripping it would leave "cheap ramen" with nothing to embed at all.

**How it could be tested.** tests assert the exact stripped text for all three CLAUDE.md example queries and for the multi-intent case; a separate test asserts "cheap ramen" keeps "ramen". The real measurement is an ablation row: rule-parsed retrieval with stripping vs. without, on the gold set.

### D053 — Validation is two layers: payload-level, then ParsedQuery-level

**Decision.** `validate_parse_payload(payload) -> dict` validates strictly and returns a canonical JSON-safe payload (all six keys, None where absent). `validate_parsed_query(..., raw_query, today, geo_resolver, known_categories)` layers on it and builds the frozen ParsedQuery.

**Alternatives.** One function returning ParsedQuery only (my first version); or letting each caller validate what it needs.

**Why.** The LLM client and the parse cache need a validated *payload* and have no raw query, no clock and no gazetteer; the pipeline needs a ParsedQuery and has all three. With one function, the cache would have had to store an object (breaking the SQLite round trip) or skip validation (so an unusable entry could poison a query forever). Splitting also keeps all strictness in one place, so the client, the cache and the pipeline cannot disagree about what a valid parse is. `validate_payload` is kept as an alias because the sibling module resolves the validator by name.

**How it could be tested.** tests assert the canonical payload survives a json round trip, that re-validating a canonical payload is idempotent, and that junk is rejected at the payload layer too. Verified end to end against the sibling LLM/parser modules: the full suite passes.

### D054 — The weekday is left unresolved in the payload and resolved at request time

**Decision.** `open_at.day` is optional and may be null in both the advertised schema and the payload. The day is supplied by the caller through `today=` (a parameter, never a clock read); `time` is required.

**Alternatives.** Require the model to always emit a day; or read `datetime.now()` inside the validator.

**Why.** "open past 10pm" names a time and no day, and the day it means is the day the question is asked. Resolving it before the cache would bake the first caller's weekday into every later cache hit -- a bug that surfaces only as wrong results the next morning. Taking it as a parameter also keeps the module clock-free and therefore testable. This reverses my initial design (both fields required) after reading the sibling module's reasoning, which was better.

**How it could be tested.** tests cover: today supplies a missing day; an explicit day wins over today; no day and no today declines the constraint rather than guessing. A cache-hit test in the sibling module pins the day being re-resolved per read.

### D055 — A hours constraint with no time refuses the whole parse

**Decision.** `open_at` present with a day but no `time` raises SchemaValidationError, so the caller falls back to the rule parser.

**Alternatives.** Drop just the hours field and keep the rest of the parse (my first version); or invent 00:00.

**Why.** Inventing a time filters on something nobody asked about. Dropping the field looked kinder but is worse in practice: an `open_at` block at all means the query almost certainly *has* an hours constraint, so silently dropping it gives confidently wrong results -- a place that shuts at 6pm ranked top for "open past 10pm". The fallback is the rule parser, which reads exactly that phrase correctly, so refusing yields a better parse than salvaging one does. It is also consistent with the module's own strictness rule, since the advertised schema marks `time` required.

**How it could be tested.** A test asserts the refusal; the sibling module's test asserts the resulting search is answered by rules rather than returning a parse with the constraint missing.

### D056 — Strict and total on attributes, lenient on categories

**Decision.** An unknown attribute key or an out-of-vocabulary value raises. An out-of-vocabulary *category* is dropped (with a debug log) when the caller supplies the index's real vocabulary.

**Alternatives.** Reject both; pass both through; coerce unknown values to the nearest known one.

**Why.** The two failure modes are not symmetric. A bad attribute value such as NoiseLevel="pretty quiet" matches no business at all, so it silently empties the result set and looks exactly like a ranking failure -- the hardest class of bug to find later, and one that would corrupt the ablation table without ever raising. A hallucinated category is advisory noise the retrieval text signal routes around. Attribute-key *casing* is the one thing forgiven: the concept is right, the lookup is otherwise exact, and there is nothing to guess.

**How it could be tested.** tests cover unknown key, bad enum value, non-boolean flag, out-of-range price, non-string category, and the forgiving cases (casing, stringified booleans, "full bar" -> "full_bar", digit-string price).

### D057 — One schema table, two provider dialects

**Decision.** `query_json_schema(dialect)` emits the OpenAPI 3.0 subset Gemini's `response_schema` takes (`nullable: true`) or OpenAI-style strict JSON Schema for Groq (`["object","null"]` type unions, `additionalProperties: false`, every property in `required`). Both are generated from `SUPPORTED_ATTRIBUTES`, and every call returns a deep copy.

**Alternatives.** Ship only the Gemini shape and let the fallback provider go unstructured; hand-maintain two schemas.

**Why.** The config names Gemini with Groq as fallback, and the two disagree about how optionality is spelled, so a single dialect silently breaks the provider that is only exercised when the primary is rate-limited -- the worst possible time to discover it. Generating both from one table is what stops the prompt advertising a value the validator rejects, which would be a 100% failure rate on that constraint that looks like a model-quality problem. Copies are returned because a client that mutates a shared dict breaks the *other* provider.

**How it could be tested.** tests assert each dialect's shape, that every supported attribute is advertised with its exact vocabulary, that `describe_attributes()` lists the same values, and that mutating one returned dict does not affect the next.

### D058 — Strict mode makes optional properties nullable

**Decision.** When the strict dialect forces every property into `required`, any property that was not genuinely required is made nullable -- type union plus `null` appended to its enum. `semantic_text` and `open_at.time` stay non-nullable.

**Alternatives.** Leave the types as they were (the bug I found while reviewing the generated output).

**Why.** Strict mode has no notion of an optional property, so without this the model is obliged to assert something about WiFi, dogs and happy hour on every single query -- which is precisely the hallucinated-constraint failure the validator exists to prevent. An enumerated field needs null in the enum as well as the type, or a strict decoder has no legal way to say "the request did not ask for this".

**How it could be tested.** A regression test asserts every attribute property accepts null (and that enums include null) while the genuinely required fields do not.

### D059 — "open now" is recognized and deliberately not resolved

**Decision.** The phrase is matched so no looser rule misreads it, but no constraint is produced and the phrase stays in `semantic_text`.

**Alternatives.** Read the clock inside the parser; or ignore the phrase entirely.

**Why.** "Now" needs a clock *and* a timezone. Reading them here would make the parser's output depend on when the tests run, and the only component that knows the user's timezone is the serving layer -- which can translate "now" into an explicit day and time before calling. Leaving the phrase in the text is the honest treatment of a constraint we cannot enforce: retrieval gets a soft shot at it instead of it vanishing.

**How it could be tested.** A test asserts no `open_at` and that "open now" survives in `semantic_text`. Determinism is pinned by a test that parses the same query twice and compares.

### D060 — Bare hours are promoted to pm only in the "open past N" family

**Decision.** In "open past/after/till/until N", a bare hour of 1-11 is read as pm and 12 as midnight. In "open at N", a bare 1-11 is refused outright; 13-23 are unambiguous everywhere.

**Alternatives.** Promote bare hours everywhere; refuse bare hours everywhere; default to am.

**Why.** The construction carries the information: "open past 11" is never 11am, so refusing it would lose a common and unambiguous request. "open at 9" genuinely could be either, and a coin flip there removes good results with no visible symptom -- the asymmetry is in the English, not in the code. Refusing leaves the phrase in the text, so the request is weakened rather than falsified.

**How it could be tested.** Parametrized tests over eleven hour phrasings; separate tests assert "open at 9" is refused and "open past 99" is ignored rather than clamped.

### D061 — Vague cheapness maps to price_max 2; an explicit "$" maps to 1

**Decision.** `VAGUE_CHEAP_PRICE_MAX = 2` for "cheap/affordable/budget/inexpensive"; `$`/`$$`/`$$$`/`$$$$` are taken at face value; the tightest ceiling wins when both appear.

**Alternatives.** Map "cheap" to tier 1 as well; treat it as a soft preference and not a hard filter at all.

**Why.** Yelp's own published meaning for the tiers puts tier 1 at roughly under $10 a head and tier 2 at roughly $11-30, so most of what people call cheap -- ramen, tacos, a casual breakfast -- sits in tier 2. Filtering vague cheapness at tier 1 would therefore throw away much of what the user would have accepted, to satisfy a word they used loosely (how much exactly is unmeasured: `make explore` reports what share of businesses carry a price at all, not how they split across the four tiers), while an explicit "$" is a precise request and deserves to be honoured precisely. The number 2 is a convention and is marked UNMEASURED in the code; it is a sweep, not a fact.

**How it could be tested.** tests pin both mappings, the tightest-ceiling rule, and that "$30" is not read as a tier. The honest test is a sweep of the ceiling against the gold set's cheap queries.

### D062 — "expensive" is recognized but cannot be encoded

**Decision.** Matched so no other rule misreads it, recorded internally as unencodable, left in `semantic_text`, and reported as a contract gap. No filter is produced.

**Alternatives.** Emit `price_max=4` (filters nothing); invent a `price_min` field locally; ignore the word.

**Why.** `ParsedQuery` has `price_max` and no `price_min`, so a price *floor* has nowhere to live. `price_max=4` would appear in the ablation as a working filter while doing nothing at all -- a fake measurement, which is the one thing this project cannot afford. Working around the contract locally is also what D012 forbids.

**How it could be tested.** A test asserts no `price_max` and that "expensive" stays in the embeddable text. Also asserts "not expensive" reads as cheap rather than as expensive (the negation rule claims the span first).

### D063 — Rules are ordered and claim spans; specific patterns run first

**Decision.** A single ordered rule list; the first rule to match a span owns it, and a rule that recognizes a phrase without encoding it still claims the span so no looser rule re-reads it. Stripping is separate from claiming.

**Alternatives.** Independent regexes applied in any order; one monolithic pattern.

**Why.** Without ordering plus span claiming, "somewhere that isn't loud" is matched by the bare "loud" pattern and the filter inverts the user's request -- the most damaging failure this parser could have. The same mechanism makes "wine bar" win over "bar" and keeps "not expensive" out of the "expensive" rule. Separating claiming from stripping is what lets a recognized-but-unencodable phrase stay in the text.

**How it could be tested.** tests cover negated-noise vs. plain "loud", "wine bar" vs. "bar", and "not expensive". A guard test asserts the parser only ever emits keys present in SUPPORTED_ATTRIBUTES.

### D064 — A parsed clock time must end on a boundary

**Decision.** The time pattern ends in `(?![\w:])`.

**Alternatives.** Leave the pattern open-ended (what I wrote first).

**Why.** This was a real bug found by a test, not a hypothetical: "open past 1Opm" (letter O for zero) matched the leading "1", was promoted to pm, and became a silent 13:00 filter on a query the user never made. A garbled time must fail to match so the phrase falls through to retrieval instead of inventing a constraint. Side effect: the dataset's own "22:0" style no longer matches in prose, which is correct -- people do not type it, and `parse_clock_time` still accepts it where the dataset's format actually appears.

**How it could be tested.** A test asserts "open past 1Opm" produces no constraint and keeps the text, while "opne past 10pm" (typo in the verb) still parses -- the bare-preposition rule carries no verb.

### D065 — An unresolvable geo label is folded back into semantic_text

**Decision.** The model is asked for a location *label* and never for coordinates. Without a resolver, or with an unknown label, the geo constraint is dropped and the label is appended to `semantic_text` if not already there.

**Alternatives.** Raise and fall back to rules (which also cannot do geo, so strictly worse); drop the label silently.

**Why.** The model has no reliable way to know where "downtown" is for the metro in scope, and a confidently wrong latitude is worse than no filter -- hence labels, with resolution left to the caller per D003. Dropping the words silently would make a geo query look like a ranking failure; putting them back at least lets BM25 match an address or city field. This is the mirror of the stripping rule and consistent with it: strip what the filter enforces, keep what it cannot.

**How it could be tested.** tests cover a resolver building the constraint, the default radius, an unresolvable label folding into the text, and no duplication when the label is already present.

### D066 — Flat attribute keys only; vibe stays in the embedding

**Decision.** The vocabulary is 17 flat Yelp attribute keys. The nested attributes (Ambience, GoodForMeal, BusinessParking, Music) are excluded.

**Alternatives.** Support dotted paths such as `Ambience.romantic`.

**Why.** `Business.attributes` is matched key for key, so a nested request would need a dotted path the shared type does not express, and every filter would need a translation table. "Somewhere good for a first date" therefore lands in `semantic_text`, which is the right home for a vibe anyway -- an embedding handles "romantic" far better than a sparse, mostly-missing boolean does. Keeping the list to attributes the gold set plausibly asks for also limits how many ways the model has to be wrong.

**How it could be tested.** The CLAUDE.md first-date example asserts only NoiseLevel is filtered and the vibe survives in the text. Expanding the vocabulary is measurable: add a key, re-run the ablation per query type.

### D067 — The LambdaMART trainer could not read the split manifest the synth stage writes

**Decision.** Taught `load_split_manifest` (concierge/ltr/train.py) the canonical `assignment` shape that `concierge.synth.splits.save_manifest` actually emits, checked before the two hand-written shapes, and added a `_MANIFEST_METADATA_KEYS` skip-list so the writer's bookkeeping keys (checksum, fractions, counts, version, group_by, n_groups, config_fingerprint) are not read as split names. Fixed the reader, not the writer.

**Alternatives.** (a) Change `synth.splits.to_json` to emit `{"splits": {...}}` — rejected: that file carries a checksum over its own contents and is the documented, human-readable artifact; the reader was the side with the wrong assumption. (b) Have ltr/train.py call `synth.splits.load_manifest` and drop its own reader — rejected as a larger refactor: ltr deliberately re-derives nothing and does not want the checksum re-verification responsibility.

**Why.** `uv run pytest` was fully green while the real integration was broken: every existing manifest test hand-built `{"seed": .., "splits": {..}}`, a shape nothing in the repo writes. Against the real file the reader raised `ValueError: split 'assignment' must map to a list of query ids`, so `make train-ltr` could only crash at startup — and, worse, the out-of-fold guarantee was never once exercised against a real artifact. A safety check that has never run on real input is reassurance it has not earned.

**How it could be tested.** New `test_reads_the_manifest_the_synth_stage_actually_writes` writes with the real `synth_splits.assign_splits` + `save_manifest`, reads with `load_split_manifest`, asserts all four split id sets match the writer exactly, and then runs `assert_out_of_fold` on it. The bare `{id: split}` form now also round-trips with a `seed` key present, which it previously did not (extended `test_manifest_accepts_id_to_split_form`).

### D068 — The cross-encoder trainer silently read zero hard negatives from the mined file

**Decision.** Rewrote `parse_example` in concierge/rerank/train.py to pull negative ids from either `negatives` (the list-of-objects form `synth.negatives.write_examples_jsonl` actually writes, each carrying retrieval_rank/score) or `negative_business_ids` (the flat form), via a new `_negative_ids` helper. A row carrying neither key now raises `KeyError`; an explicitly empty list is still allowed through.

**Alternatives.** (a) Change `synth.negatives.TrainingExample.to_json` to emit flat ids — rejected: the rank and score of each negative are the mining evidence, and the miner's report quality depends on keeping them. (b) Leave `.get(...) or ()` and document the required key — rejected: that is exactly the failure mode the function's own docstring promised to prevent.

**Why.** This was the most dangerous defect found. `write_examples_jsonl` stores `negatives`; the trainer read only `negative_business_ids` with a `.get(...) or ()` fallback, so the mined file parsed cleanly into **zero** negatives. A cross-encoder fine-tuned on positives alone still reports a falling loss, so nothing downstream would have flagged it — the owner would have shipped a reranker trained on no contrast and an ablation row attributing its (absent) gain to fine-tuning.

**How it could be tested.** New `test_parse_example_reads_the_file_the_miner_actually_writes` mines a `TrainingExample` with the real writer and asserts the parser recovers both negative ids and the query_id. Three more tests pin the strictness the docstring claims: no negatives key at all raises, an explicitly empty list is permitted, and a negative object without `business_id` raises.

### D069 — The LLM judge crashed on the qrels file the owner's labeling CLI produces

**Decision.** Added `QRELS_GRADE_COLUMNS = ("relevance", "grade", "label", "rel")` to eval/metrics.py as the single source of truth, made `judge.GoldLabels.load` resolve the grade column from the header once (naming the header when none matches), and re-pointed `eval.run_eval.QRELS_GRADE_ALIASES` at the same tuple instead of re-declaring it.

**Alternatives.** (a) Change `eval.pooling.QRELS_COLUMNS` to write `grade` — rejected: `relevance` is the clearer column name and pooling is the writer, so changing it would invalidate any labels already collected. (b) Add `"relevance"` to the judge's own local list — rejected: that is a third copy of one contract, and three copies are how this bug happened.

**Why.** `eval.pooling.QrelsWriter` heads the grade column `relevance`; `judge.GoldLabels.load` read `row.get("grade")` and so failed with `grade '' is not an integer` on line 2 of the owner's real labels. The label-quality kappa is a headline deliverable (CLAUDE.md §6-7) and it could not run at all. `eval.run_eval` had independently discovered the problem and kept its own alias list, which is why the drift went unnoticed.

**How it could be tested.** New `test_gold_labels_read_the_header_the_labeling_cli_actually_writes` writes real rows with `QrelsWriter` and reads them with the judge; `test_gold_labels_name_a_missing_grade_column` pins the new error; `test_every_qrels_reader_agrees_on_the_grade_column` asserts run_eval and metrics share one object and that the writer's header contains exactly one grade column. Also confirmed by hand that `eval.pooling.load_qrels` and `eval.run_eval.load_qrels` return identical grades for the same file.

### D070 — `make train-ltr`'s default manifest path pointed where nothing writes

**Decision.** Added `default_manifest_path(cfg)` to concierge/ltr/train.py: it returns `concierge.synth.splits.splits_path(cfg)` (artifacts/synth/<fingerprint>/splits.json), falls back to the legacy `data/processed/synth_splits.json` with a warning when such a file exists, and otherwise names the canonical path in the error. `--splits` now defaults to None so an explicit flag is distinguishable from the default.

**Alternatives.** Leave the constant and rely on the owner passing `--splits` — rejected: `make train-ltr` passes no flags, so the only path the Makefile can reach was one that never exists. The failure was loud, but it was a guaranteed stumble on the first real training run.

**Why.** The writer keys the manifest by config fingerprint on purpose (a config change must not silently reuse an old split), so the consumer cannot hard-code a path — it has to ask the writer. Keeping the legacy fallback means a hand-placed manifest still works, and the error message now names the path the generator will actually produce.

**How it could be tested.** `test_default_manifest_path_points_where_the_synth_stage_writes` asserts equality with `synth.splits.splits_path(config)`; `test_default_manifest_path_still_honours_a_legacy_file` monkeypatches the legacy constant at an existing file and asserts it wins.

### D071 — The two gold-leakage guards normalized query text differently

**Decision.** Made `concierge.rerank.train.normalize_query` strip trailing `.?!` as well as folding case and whitespace, matching `concierge.ltr.train._normalize_query` exactly, and added a test that pins the two functions to each other over six inputs.

**Alternatives.** (a) Extract one shared normalizer into a new module — rejected for now: it would add a file for three lines and a cross-stage import, and the test-level pin catches drift just as reliably. (b) Leave the difference and only report it — rejected: of two guards over one invariant, the weaker one decides, and this one guards a whole GPU fine-tuning run.

**Why.** Stage 4 caught a gold query differing only by a trailing '?'; Stage 3 waved it through. CLAUDE.md §6 treats gold leakage as never-violate, and leakage into cross-encoder training corrupts the headline NDCG@10 in the flattering direction, where nobody goes looking. Note this is a narrow hole in practice — the synth stage's embedding dedup (cosine >= 0.92) would likely catch the same query first — but the guards should not disagree.

**How it could be tested.** `test_both_leakage_guards_normalize_identically` compares both functions on six inputs. The renamed `test_normalize_query_folds_case_whitespace_and_trailing_punctuation` keeps the original no-fuzzy-matching assertion ('quiet cafes' != 'quiet cafe') and adds that internal punctuation ('open 24/7, cheap') survives.

### D072 — Stage 4's trained-artifact seam was only ever tested with a fake booster

**Decision.** Added `test_a_saved_ranker_loads_into_the_serving_adapter`: trains a real LambdaMART on the known-answer monotone dataset, saves it with `ltr.train.save_ranker`, loads it with `pipeline.LightGbmRanker.load`, and asserts both copies of the feature order agree and that the reloaded model still ranks the known problem correctly. No production code changed — the seam was sound.

**Alternatives.** Trust the existing `check_feature_order` unit test — rejected: it is called with hand-made name lists, so it never proved that a real LightGBM round trip preserves the order `feature_names.json` records.

**Why.** `save_ranker` writes the column order twice specifically so serving can assert it, and permuted columns produce plausible predictions with no error — the project's own comments call it the most dangerous silent bug available in Stage 4. A guarantee that exists to be checked should be checked against a real model file.

**How it could be tested.** The new test itself; it passes, confirming the seam was already correct.

### D073 — A test asserted `pytest.raises(Exception)` for config validation

**Decision.** Tightened `tests/test_contracts.py::test_unknown_key_is_rejected` to `pytest.raises(ValidationError, match="extra_forbidden")` and added a docstring explaining what the check buys.

**Alternatives.** Add `# noqa: B017` — rejected: the lint rule is right here.

**Why.** A bare `Exception` assertion also passes if `Config` fails to import at all, so it could go green for the wrong reason. This strengthens the test; it does not relax it.

**How it could be tested.** Confirmed the actual raise is `pydantic_core.ValidationError` with `type=extra_forbidden`, then re-ran the suite.

### D074 — Verified the eight required correctness properties independently, not via the existing tests

**Decision.** Wrote a throwaway probe (in the scratchpad, not the repo) that exercises the production code directly for all eight properties plus edge cases around them, every assertion passing, rather than reading the tests that claim to cover them.

**Alternatives.** Read the relevant tests and confirm they assert the right things — rejected as the weaker check: the same eight agents wrote the modules and their tests, and three of the four seam bugs found here were invisible precisely because the tests agreed with the code they were written beside.

**Why.** A green suite is evidence that code matches its tests, not that it is correct. The probe is independent: overnight '17:0-2:0' open at 01:00 and closed at 03:00 (checked at interval, day-aware and filter level, plus that '17:0-0:0' does not wrap into the next morning); '0:0-0:0' open at every minute and not flagged as crossing midnight; missing attribute/hours/price data yielding UNKNOWN under all three missing-data policies and never a silent pass; the bge prefix applied to queries only, including through batching and `DenseIndex.build`; RRF giving no term at all for a non-retrieving system (absent score < 1/k) and a doc found by both systems beating one found by one; the FAISS position->business_id mapping round-tripping through save/load with a canned encoder, a short id list rejected, and no -1 sentinel leaking; both out-of-fold checkers firing on planted overlaps (CE-train and CE-val, ltr_train and ltr_val) and the enforce=False hatch logging instead; gold leakage firing by text, by id, after normalization, and failing closed on a missing gold file.

**How it could be tested.** Probe run three times — before any changes, after the production fixes, and after the format pass — passing each time. It is a scratchpad file, deliberately not added to the repo, so that run is evidence from this session and not something the owner can re-run: the durable check is `uv run pytest`, and anything the probe caught that the suite does not assert is a gap worth closing there.

## Stage 1 — LLM client, cache and fallback

### D075 — Cache key = (provider, model, prompt_version, query), with prompt_version = declared version + hash of body AND injected vocabulary

**Decision.** The SQLite key is a SHA-256 of the JSON-encoded tuple (provider, model, prompt_version, normalized query). prompt_version() returns "v1+<sha256[:12]>" where the hash covers the prompt body plus the attribute vocabulary rendered by schema.describe_attributes(), and deliberately excludes the file's leading '#' comment block.

**Alternatives.** (a) Key on the query alone. (b) Key on a hand-maintained prompt version only. (c) Hash the whole prompt file including comments. (d) Hash only the file, ignoring the injected vocabulary.

**Why.** A cache entry is only valid for the exact parser that produced it, and the prompt is part of that parser. (a) and (b) both allow the silent failure this project cannot tolerate: editing a prompt and forgetting to bump a version leaves no trace except metrics that describe a mixture of two parsers inside one experiment. The content hash means the key changes whether or not a human remembers. (d) was the near-miss: the attribute list is injected from schema.describe_attributes() rather than written in the file, so adding an attribute would change the prompt the model actually sees while leaving the version untouched — the same corruption arriving through the other module. (c) would throw away a warm cache every time a repo-facing comment was edited, which punishes documenting.

**How it could be tested.** tests/test_query_llm.py::test_prompt_version_tracks_the_body_and_the_vocabulary asserts a comment edit does not change the version while a body edit, a vocabulary change and a declared bump each do; ::test_prompt_version_change_forces_a_new_api_call shows a changed version misses the cache and re-calls the API with both entries coexisting.

### D076 — Retry only timeouts, 429 and 5xx; never a 400, and never a malformed response

**Decision.** A typed exception hierarchy (LlmTimeoutError, LlmTransportError, LlmHttpError with status_code, LlmBadResponseError, LlmConfigError) drives a one-line retry predicate. tenacity is used imperatively (Retrying(...)) with stop_after_attempt(max_retries + 1) and wait_exponential_jitter. JSON decoding happens outside the retry block.

**Alternatives.** (a) Retry everything a fixed number of times. (b) A tenacity decorator with constants. (c) Retry malformed JSON too, hoping for a better sample.

**Why.** Only timeouts, connection errors, 429s and 5xx are transient. A 400 means our own request is malformed: retrying it burns free-tier quota to earn the same rejection once per attempt, with the backoff waits on top. Decoding sits outside the retry loop because at temperature 0 a second identical request returns the same bytes, so (c) spends quota to fail again when the rule parser could already have answered. A decorator (b) cannot read cfg.max_retries, which is runtime data. Jitter is included because the eval harness parses in a loop and a fixed backoff would line every retry up at the same instant and re-trigger the limit.

**How it could be tested.** ::test_http_400_is_not_retried (exactly one call), ::test_http_429_is_retried_then_succeeds (exactly three with max_retries=2), ::test_http_5xx_is_retried, ::test_retry_count_follows_config (max_retries=0 means one attempt), ::test_malformed_json_falls_back_to_rules (one call, no retry), ::test_timeout_falls_back_to_rules.

### D077 — Stage 1 never raises: every failure degrades to rules, and ParseSource records which path answered

**Decision.** parser.parse() catches every exception from the cache, the API and the validator. With fallback_to_rules it calls rules.parse_rules(query, today=...); without it, it returns a pass-through ParsedQuery (raw query as semantic_text, zero constraints). llm.parse_with_llm() by contrast raises on every failure.

**Alternatives.** (a) Let API errors propagate to the API layer. (b) Have llm.py fall back internally. (c) Return None on failure and make callers handle it.

**Why.** A search that 500s because Gemini rate-limited is a worse product than a search answered with rule parsing. But the fallback decision must live in exactly one place: if llm.py fell back internally (b), nothing could tell an LLM parse from a rule parse, and the ablation's LLM-parsing row would be measuring an unknown mixture. So the lower layer raises, the composer decides, and ParseSource is set on every path — including CACHE, whose latency is a local SQLite read rather than a network round trip - a difference the ablation's per-stage timings report rather than asserting a ratio here.

**How it could be tested.** Eight degradation tests assert a ParsedQuery comes back and ParseSource is correct for: timeout, connection error, 400, exhausted 429, malformed JSON, schema rejection, safety-blocked reply, missing API key. ::test_rule_parser_failure_degrades_to_passthrough monkeypatches parse_rules to raise and still gets a usable result.

### D078 — provider: "rules" and the empty query short-circuit before any network or cache access

**Decision.** parse() returns the rule parse immediately when cfg.query_parsing.provider == "rules" or the query is blank, touching neither the API nor the cache.

**Alternatives.** Let the normal path run and have build_providers() refuse, relying on the fallback to produce the same answer.

**Why.** The rules-only setting is how the ablation measures the rule parser on its own. Any API call on that path would contaminate both its quality number (via a cached LLM parse) and its latency number. The alternative reaches the same answer but pays a cache lookup and an exception to get there, so the latency row would overstate the rule parser's cost.

**How it could be tested.** ::test_provider_rules_skips_the_api_entirely and ::test_empty_query_skips_the_api use a transport whose handler raises AssertionError if it is ever called, and additionally assert the cache stays empty.

### D079 — Gemini primary with a Groq fallback, switched only on "provider unavailable" errors

**Decision.** build_providers() returns the configured provider first and the other one second when its key is present. parse_with_llm() moves to the fallback only when the error is retryable (timeout, transport, 429, 5xx) and only after the primary's retry budget is exhausted; a 400 or a malformed reply raises straight through to the rule fallback.

**Alternatives.** (a) One provider, config only. (b) Fall back on any error. (c) Race both providers and take the first answer.

**Why.** On a free tier the realistic failure is a 429, and a demo that dies when the daily Gemini quota runs out is worse than one that finishes on Groq after the extra round trip (how long that costs is one of the per-stage latencies the eval harness reports). (b) is wrong because a malformed request fails the same way everywhere, so trying the second provider only adds latency before the rule fallback. (c) doubles quota consumption on every query to save latency on a rare path.

**How it could be tested.** ::test_groq_answers_when_gemini_is_rate_limited asserts the exact host sequence (three Gemini attempts, then Groq) and an LLM-sourced result; ::test_400_does_not_trigger_the_fallback_provider asserts only Gemini was contacted.

### D080 — Cache the canonical payload keyed on the configured provider, recording who actually answered

**Decision.** The cache stores schema.validate_parse_payload()'s canonical JSON payload, written only after validation has succeeded. The key carries the configured provider and model; an answered_by column records "<provider>:<model>" of whoever replied.

**Alternatives.** (a) Store the rendered ParsedQuery. (b) Store the raw model reply. (c) Key on the provider that actually answered. (d) Write to the cache before validating.

**Why.** (a) breaks the JSON round trip and freezes our payload-to-ParsedQuery mapping, so a later mapping bug could only be fixed by re-spending quota. (c) would mean a Groq reply fetched under a Gemini config could never be reused by that config, defeating the fallback's purpose — so the key describes what was requested and the column records what happened, which also keeps the provenance auditable. (d) would let an out-of-vocabulary reply poison a query permanently: every later call would read it, fail validation, and silently fall back.

**How it could be tested.** ::test_fallback_answer_is_cached_under_the_configured_provider reads the entry back under the Gemini key and asserts answered_by == "groq:<default model>"; ::test_only_successful_parses_are_cached asserts an invalid-attribute reply leaves the cache empty; ::test_cached_entry_rejected_by_the_validator_is_dropped_and_refetched asserts a now-invalid entry is deleted and re-fetched.

### D081 — open_at.day stays null in the stored payload and is resolved from the clock on every read

**Decision.** The prompt asks for "day": null whenever the request names no specific weekday ("open past 10pm", "open late", "tonight", "open now"). parse() derives today from its injected `now` parameter and passes it to schema.validate_parsed_query(today=...) on every call, cache hits included. Queries containing an explicitly day-shifted word (tomorrow, yesterday) bypass the cache entirely.

**Alternatives.** (a) Let the model write today's weekday into open_at.day. (b) Read datetime.now() inside the parser with no injection point. (c) Drop the hours constraint when no day is named. (d) Include the date in the cache key.

**Why.** "Open past 10pm" is one of the most common constraint shapes in the gold set, so (c) throws away the headline constraint type. (a) looks fine and is the trap: the weekday would be baked into the cached payload, and from the next midnight the filter would silently check the wrong day's hours — wrong results with no visible symptom. Resolving per read costs nothing and is always correct. (d) would fragment the cache daily for every query, not just the day-sensitive ones. "tomorrow" is the one case the clock cannot re-resolve, because the model already folded it into a concrete weekday, so those queries skip the cache instead of being silently a day stale. `now` is a parameter so the suite does not depend on the day it runs.

**How it could be tested.** ::test_a_null_day_is_resolved_from_the_clock, ::test_cache_hit_resolves_the_day_for_the_day_it_is_read (one API call; Thursday on write, Saturday on a later read), ::test_a_weekday_the_model_names_is_respected_not_overwritten, ::test_queries_about_another_day_bypass_the_cache (two API calls, empty cache).

### D082 — Constrained decoding with the schema module's own JSON schema, per provider dialect

**Decision.** Gemini requests carry generationConfig.response_schema = schema.query_json_schema("openapi") with temperature 0 and response_mime_type application/json. Groq requests carry response_format json_schema built from schema.query_json_schema("jsonschema"); on a 400 from that strict form only, the request is re-sent once in json_object mode.

**Alternatives.** (a) Ask for JSON in the prompt only and validate client-side. (b) Send the strict schema to Groq with no downgrade. (c) Always use Groq's looser json_object mode.

**Why.** Constrained decoding makes an out-of-vocabulary attribute value impossible rather than merely unlikely, and the same table generates the schema, the prompt's vocabulary and the validator, so none of the three can drift. The downgrade exists because strict json_schema support varies by Groq model: with (b) a model that rejects it would 400, and the one provider that exists to survive a 429 would be dead. The downgrade is not a retry of the same request — it is one different, weaker request, attempted once, with client-side validation still catching anything it lets through.

**How it could be tested.** ::test_request_carries_the_schema_and_deterministic_decoding compares the sent body against schema.query_json_schema("openapi"); ::test_groq_response_shape_is_parsed asserts json_schema mode; ::test_groq_downgrades_once_when_strict_schema_is_rejected asserts exactly two calls with response_format types [json_schema, json_object] — one downgrade, never a loop.

### D083 — Leakage-safe few-shot examples: off-domain categories, with a mechanical test

**Decision.** The prompt's four examples use hardware stores, laundromats, barbers and pharmacies — categories that cannot appear in a restaurant/cafe/bar gold set. The file's header states the rule and why, and a test asserts no gold query string appears in the prompt file.

**Alternatives.** (a) In-domain cafe/restaurant examples. (b) No examples at all. (c) Examples drawn from a held-out slice of the owner's gold queries.

**Why.** (c) is disqualifying: gold queries are test-only, and tuning the prompt on them would make every metric downstream of parsing a measurement of the prompt's memory of the test set — undetectable afterwards from the metrics alone. (a) risks accidental overlap once the owner writes ~150 queries, since cafe phrasings are exactly what they will write. The cost of off-domain examples is that they teach the output format rather than in-domain phrasing; that cost is unmeasured and was accepted because the output format is domain-independent while the leakage risk is not recoverable.

**How it could be tested.** ::test_prompt_contains_no_gold_query reads eval/gold_queries.csv when it exists and asserts no query of 8+ characters appears in the prompt file (currently skips: the gold file is owner-owned and not yet written). Exact-substring matching catches copy-paste but not paraphrase, which is why the off-domain-category rule is written into the file header for humans.

### D084 — API keys: environment only, sent as a header, never logged, never in a test

**Decision.** Keys are read via python-dotenv with override=False and sent as x-goog-api-key / Authorization headers. No code path logs, stores or returns a key; failure messages name the variable ("GEMINI_API_KEY is not set"), never the value. Tests force an _ENV_LOADED flag so dotenv never reads the owner's .env, and use an obvious dummy string.

**Alternatives.** (a) Gemini's documented ?key= query parameter. (b) Load dotenv at import time. (c) Let tests read the real .env and skip when no key is present.

**Why.** (a) puts the secret in a URL, and URLs reach logs, traces and HTTP error bodies — including the error strings this module builds. (c) would make the suite spend quota and pass or fail according to a rate limit rather than the code, and would leave a real key one logging change away from the console. override=False is what lets a test's environment variable beat the file, which is the mechanism that makes hermeticity possible at all.

**How it could be tested.** ::test_api_key_is_sent_as_a_header_never_in_the_url asserts the dummy key is absent from the request URL and present in the header; the autouse hermetic_env fixture pins _ENV_LOADED, and the whole file passes with no network and no key. A grep of every log call in llm.py and parser.py confirms none takes a key value.

### D085 — One SQLite connection per cache path, lock-guarded and WAL-mode

**Decision.** ParseCache opens one connection with check_same_thread=False and guards every statement with a threading.Lock; WAL is enabled for file-backed databases. parser._cache_for() is lru_cached so a process reuses one instance per path. A row whose payload will not decode is deleted and reported as a miss.

**Alternatives.** (a) A connection per call. (b) A connection per thread. (c) A JSON file on disk. (d) Raise on a corrupt row.

**Why.** Parse latency is a reported number, so opening SQLite per query (a) adds measurable cost for nothing; (b) leaks handles in a long-lived FastAPI process. (c) has no atomic write, and the service is multi-threaded while `make eval` may parse in parallel. The lock — not the flag — is what makes cross-thread sharing safe, which is noted at the line so a future edit cannot quietly remove it. (d) would turn one interrupted write into a permanently failing query; costing a single API call instead is the cheaper failure.

**How it could be tested.** ::test_cache_roundtrip_and_miss, ::test_cache_persists_across_instances (reopen the file and read the entry back), ::test_cache_treats_a_corrupt_payload_as_a_miss (corrupt a row directly, assert a miss and that the row is gone).

## Stage 2 — BM25, fusion and filters

### D086 — BM25 variant: Lucene (bm25s default), fixed in code not config

**Decision.** Use method="lucene" for the BM25 scoring variant, recorded in index metadata. k1/b come from cfg.retrieval.bm25.

**Alternatives.** Robertson/Sparck-Jones (textbook) IDF; ATIRE; BM25L; BM25+.

**Why.** The Lucene IDF is clamped to stay positive. The textbook IDF goes negative for a term occurring in more than half the corpus, which lets a very common term actively push a document down. Business documents here are highly repetitive ("Restaurants", "Food", "wifi" everywhere), so that case would be hit constantly. It also makes the baseline comparable to what a reader would get from Elasticsearch, which ships the same variant.

**How it could be tested.** Plumbing is tested (test_k1_and_b_come_from_config, test_b_zero_disables_length_normalization asserts b actually changes scores). The variant choice itself is unmeasured — comparing variants would need the gold set.

### D087 — Tokenization: regex + English stopwords + optional Snowball stemming, with a load-time mismatch guard

**Decision.** bm25s default token pattern, English stopword removal, Snowball stemming via PyStemmer when importable and none otherwise. The resolved stemmer name is written into index metadata and a mismatch between index and query-time environment raises rather than degrading silently.

**Alternatives.** No stemming at all (deterministic but worse recall); hard-require PyStemmer; make these config keys.

**Why.** Stemming is a real lever on lexical recall here - the one this project has actually caught failing, see D223: without it "cafe" and "Cafes", "bookstore" and "Bookstores", "noodle" and "noodles" are unrelated tokens, and short queries have no other terms to compensate. PyStemmer is a C extension and is not a declared dependency, so it cannot be assumed. The dangerous failure is a corpus stemmed one way and queries another — that loses recall quietly and produces plausible-looking but worse numbers, so it is made a hard error. Stopword removal is safe in this system specifically because negation and every other hard constraint is enforced arithmetically by filters.py, never by lexical overlap.

**How it could be tested.** test_tokenization_mismatch_is_refused_at_query_time; test_stemmer_name_is_recorded_in_metadata; test_stemming_matches_singular_query_to_plural_document (skipped when PyStemmer is absent, which it currently is). The recall cost of no stemming is unmeasured.

### D088 — BM25 drops zero-score documents from its result list

**Decision.** search() returns only documents with a strictly positive score.

**Alternatives.** Return the full top-k as bm25s does.

**Why.** A BM25 score of exactly 0 means no query term occurred in the document at all. bm25s still returns such rows to pad top-k. Passing them on would give them a fusion rank and therefore a real 1/(k+rank) contribution purely for existing, which is noise entering the fused ranking rather than evidence.

**How it could be tested.** test_zero_score_documents_are_dropped ("ramen" matches one fixture business; asking for 8 returns 1).

### D089 — BM25 index artifacts keyed by Config.fingerprint(), corrupt cache falls back to rebuild

**Decision.** artifacts/bm25/<fingerprint>/ holds the bm25s files plus business_ids.json (row -> id) and concierge_meta.json. load_or_build() rebuilds on any read failure.

**Alternatives.** A single shared index path; failing loudly on a corrupt index.

**Why.** Two configs must never read each other's index, or a measured number gets attributed to the wrong settings. The id list is saved next to the matrix because bm25s identifies documents by row number, and a separately-stored id list is how row-offset bugs happen. An index is derived data, so an unreadable cache should never be able to break a run.

**How it could be tested.** test_index_dir_is_keyed_by_config_fingerprint, test_save_and_load_round_trip, test_load_or_build_writes_then_reuses, test_corrupt_index_falls_back_to_rebuilding.

### D090 — RRF is the default fusion method; weighted_sum implemented for ablation only

**Decision.** score = sum over systems of 1/(k + rank), 1-based ranks, k from cfg.retrieval.fusion.k. weighted_sum (per-system min-max then weighted sum) is selectable via cfg.retrieval.fusion.method.

**Alternatives.** Weighted sum of raw scores; z-score or min-max normalization then sum; learned fusion weights.

**Why.** BM25 scores are unbounded and corpus-dependent (a sum of IDF weights computed from this collection), while cosine on normalized embeddings sits in [-1,1] and bunches near the top. Summing them requires a normalization scheme AND weights, and both are themselves untuned hyperparameters that would need labels to tune — labels are the scarcest resource in this project. RRF uses ranks only, so it is scale-invariant and has one constant. The cost is real and stated: discarding scores discards margins, so a document BM25 is wildly confident about looks like the one just below it.

**How it could be tested.** test_rank_map_ignores_scores (scale invariance); test_weighted_sum_score_depends_on_unrelated_results asserts the concrete mechanism — a document's weighted_sum score moves when a different document leaves the list, while its RRF score does not. Which method actually ranks better is unmeasured; both run through the same code path so the ablation can compare them.

### D091 — A document found by only one system contributes exactly one RRF term

**Decision.** A system that did not return a document contributes no term at all for it — not a 1/k "rank 0" bonus and not a "rank = len(list)+1" term.

**Alternatives.** Missing rank treated as 0; missing rank treated as list length + 1.

**Why.** This is the classic RRF implementation bug. Missing-as-rank-0 awards the largest possible contribution for being absent. Missing-as-last awards a smaller unearned bonus and additionally makes the score depend on how long each list happens to be. Either way a document found by one system beats a document found by one system plus a near-miss in the other, which inverts the point of fusing.

**How it could be tested.** test_single_system_document_gets_no_phantom_bonus asserts the score is exactly 1/63 and explicitly asserts it is neither 1/63+1/60 nor 1/63+1/63. Every fusion arithmetic test is written as the formula (1/61 + 1/62), not as a decimal.

### D092 — The fused score always lands in SCORE_RRF regardless of method; ties break on business_id

**Decision.** fuse() writes the fused value to Candidate.scores[SCORE_RRF] and the fused position to ranks[RANK_RRF], and also passes every per-system raw score and rank through under its own canonical key. Ties sort by business_id.

**Alternatives.** A separate score key per fusion method; leaving ties in dict order.

**Why.** SCORE_RRF is the "Stage 2 fused score" slot that the Stage 4 feature builder reads by name; adding a second key would make the feature name depend on a config value. Which method produced a number is not ambiguous — it is in the config, and artifacts are fingerprinted. Deterministic tie-breaking matters more than it looks: RRF produces exact ties routinely, and a dict-order-dependent result makes an A/B difference un-attributable.

**How it could be tested.** test_fuse_writes_canonical_score_and_rank_keys (including that an absent system writes no key at all, rather than a 0.0 that Stage 4 would learn from); test_fuse_tie_break_does_not_depend_on_input_order.

### D093 — Filters: VIOLATED always drops; UNKNOWN follows the configured policy

**Decision.** A VIOLATED hard constraint removes the business unconditionally. UNKNOWN is routed by cfg.retrieval.missing_attribute_policy: exclude drops, penalize multiplies SCORE_RRF by missing_attribute_penalty, neutral leaves the score untouched. All three implemented, none privileged in code.

**Alternatives.** Boolean constraint checking; one hardcoded missing-data behaviour.

**Why.** "No NoiseLevel data" and "loud" are different statements. Missing-as-pass returns loud bars for "quiet"; missing-as-fail shrinks the answer set to the biased subset of businesses whose owners filled in the attribute form, which on a sparse attribute can be a large share of the corpus - `make explore` prints the per-attribute fill rates, and until it has run the share is unknown. Keeping the two outcomes on different mechanisms is the whole point of the three-valued design. Which policy is best is D007, an open measurement — the 0.5 penalty has no justification yet beyond "between keeping and dropping", and the code says so rather than implying otherwise.

**How it could be tested.** test_violated_constraint_always_removes_the_business runs all three policies on one query; test_policy_exclude_drops_unknowns / test_policy_penalize_keeps_unknowns_with_a_multiplied_score / test_policy_neutral_leaves_scores_untouched run the same query and the same four fixtures (b001 SATISFIED, b002 VIOLATED, b005 UNKNOWN-no-attributes, b008 UNKNOWN-via-"None"); test_a_penalty_of_one_is_equivalent_to_neutral; test_penalty_can_reorder_results.

### D094 — Boolean recognition in attribute comparison is strict: only bools and "True"/"False"

**Decision.** _as_bool accepts Python bools and the strings "true"/"false" only. "yes"/"no"/"1"/"0" are treated as ordinary strings. A boolean on one side and a vocabulary word on the other yields UNKNOWN, not a guess.

**Alternatives.** A permissive truthy/falsy string set (the first version of this code did exactly that).

**Why.** Caught by fixture b002, whose WiFi is "no" — a real value in the WiFi vocabulary (free/paid/no), not a boolean. The permissive version coerced it, so asking for WiFi="free" against a stored "no" became a boolean-vs-string shape mismatch and returned UNKNOWN instead of VIOLATED, i.e. a provably-failing business was handed to the missing-data policy. Symmetric strictness is the only rule that keeps both "OutdoorSeating: True" and "WiFi: no" comparable.

**How it could be tested.** test_string_attribute_match_and_violation (this test is what found the bug), test_boolean_attribute, test_shape_mismatch_is_unknown_not_a_guess.

### D095 — Defensive handling of the literal string "None" in attributes

**Decision.** filters treat "", "none", "null", "nan", "na", "n/a" (case-insensitive) as missing, even though Business.attributes is contractually normalized by the data layer.

**Alternatives.** Trust the contract and compare whatever arrives.

**Why.** Defence in depth against the one variant of this quirk that fails silently: the string "None" is truthy in Python, so a leaked OutdoorSeating of "None" reads as "yes, outdoor seating" and produces a confidently wrong answer, which is worse than no answer. Fixture b008 carries exactly this value.

**How it could be tested.** test_b008_none_string_is_unknown_not_false asserts UNKNOWN, asserts actual is None, and asserts that requesting OutdoorSeating=False is also UNKNOWN (reporting False would be an equally invented fact).

### D096 — Category matching: normalized exact match OR token-subset, any-of across requested categories

**Decision.** Lowercase and strip non-alphanumerics (dropping the filler word "and"), then match if the requested category equals a business category or if its token set is a subset of a business category's tokens. Any one requested category matching is enough. A business with no categories at all is UNKNOWN.

**Alternatives.** Exact string match against Yelp's taxonomy; all-of semantics; fuzzy/embedding category matching.

**Why.** Case- and separator-insensitivity makes "Coffee & Tea", "coffee and tea", "COFFEE-TEA" equivalent. Token-subset matching exists because an LLM parser emits "coffee", not Yelp's verbatim "Coffee & Tea" — it does not know Yelp's category taxonomy verbatim. Any-of because a parser asked about ramen will emit several near-synonyms and all-of would reject the obvious answer. Stated limitation: this is lexical and will not match "noodles" to "Ramen" or "pub" to "Bars" — that gap is the dense retriever's job, not this filter's.

**How it could be tested.** test_category_match_is_separator_and_case_insensitive, test_category_match_accepts_a_token_subset, test_categories_are_any_of_not_all_of, test_category_violation, test_category_unknown_when_the_business_has_none.

### D097 — Geo: haversine, and a business with no coordinates is UNKNOWN rather than dropped by the check

**Decision.** haversine_km against GeoConstraint.radius_km. Missing latitude/longitude returns UNKNOWN, never VIOLATED.

**Alternatives.** Equirectangular/flat-earth approximation; Vincenty/geodesic; treating missing coordinates as out of radius.

**Why.** Haversine is three lines and removes the whole "why is this wrong away from the equator" class of question; geodesic sub-metre accuracy is irrelevant next to the precision of a "near downtown" radius. Returning VIOLATED for missing coordinates would hide an unconditional drop inside a geo check, taking that decision out of the missing-data measurement where it belongs.

**How it could be tested.** test_haversine_one_degree_of_longitude_at_the_equator asserts the known 111.195 km; plus zero-distance and symmetry; test_geo_outside_radius_is_violated computes the fixture distance and tests a radius either side of it; test_geo_without_coordinates_is_unknown_never_violated.

### D098 — Filters own two non-constraint drops, and recompute RANK_RRF over survivors

**Decision.** apply_filters also drops a candidate with no Business record (DROP_UNKNOWN_BUSINESS) and any business with is_open == 0 (DROP_PERMANENTLY_CLOSED), under every policy. Survivors are re-sorted and RANK_RRF is renumbered 1..N. A FilterReport carries the per-business drop reasons.

**Alternatives.** Rely solely on the index build to exclude closed businesses; preserve the pre-filter RRF ranks; return only the surviving candidates.

**Why.** The index build already excludes is_open == 0, but showing a permanently closed business is the most embarrassing possible wrong answer and the check costs one boolean. Filtering is part of Stage 2, so its output ranks are the post-filter ranks — leaving gaps would hand Stage 4 a rank feature corresponding to no list that ever existed. The penalty is deliberately not stored as a separate score key: it is recoverable as a function of the attached ConstraintMatch statuses and the config, so nothing is lost. Drop reasons are not debug noise — "this query returned three results" needs an explanation in the failure analysis.

**How it could be tested.** test_b004_is_dropped_as_permanently_closed (under neutral, to show it is not a missing-data case), test_candidate_without_a_business_record_is_dropped, test_ranks_are_recomputed_contiguously, test_upstream_scores_and_ranks_are_preserved, test_drop_counts_summarize_the_reasons.

### D099 — Hours logic is fully delegated; a bad OpenAt is allowed to raise

**Decision.** check_open_at calls concierge.data.hours.is_open_at(business.hours, open_at.day, open_at.minute_of_day) and maps its ConstraintStatus straight onto a ConstraintMatch. The ValueError that function raises for an unrecognized day or out-of-range minute propagates.

**Alternatives.** Re-derive the three-valued answer locally from HoursInterval.covers(); catch the ValueError and return UNKNOWN.

**Why.** Two implementations of the same calendar arithmetic can disagree, and that is exactly how "open past 10pm" queries break — including the previous-day overnight spillover, which a locally-written version got wrong during development. Swallowing the ValueError would hide a broken Stage 1 parser behind plausible-looking results; the parser must validate its own output against its schema.

**How it could be tested.** test_b002_is_open_at_1am_via_its_overnight_interval (Saturday 01:00 via Friday's 17:0-3:0), test_b001_is_closed_at_1am (VIOLATED, data says no), test_b005_open_at_is_unknown_not_closed (hours null), test_b007_open_24_hours ("0:0-0:0"), test_b003_missing_weekday_is_closed_not_unknown.

## Stage 2 — dense embeddings and FAISS

### D100 — D-dense-1 — The encoder is injectable; the bge prefix rule lives one layer above it

**Decision.** Split the embedding path in two: a `TextEncoder` protocol (`dim`, `encode(texts) -> ndarray`) that knows nothing about queries vs. passages, and a `DenseEmbedder` that owns the query prefix, the batching, and the L2 normalization. `DenseEmbedder.from_config(cfg, encoder=None)` downloads the real model only when no encoder is passed in.

**Alternatives.** (a) One class that loads SentenceTransformer in __init__ and applies the prefix inline. (b) Keep one class and monkeypatch `SentenceTransformer` in tests. (c) Pass a `is_query: bool` flag down into the encoder.

**Why.** The prefix asymmetry cannot be tested at all if the only way to run the code is to download the real model — and it is the single highest-risk bug in the module, because getting it wrong raises no exception and produces plausible-looking results that are simply worse. With the rule above the encoder, a fake that records the strings it was handed is enough to assert it. (b) was rejected because monkeypatching an import is a test of the mock, not the contract; (c) was rejected because it pushes the convention into every encoder implementation instead of keeping it in one place.

**How it could be tested.** `test_query_prefix_applied_to_queries_only` asserts the literal text that reached the model: bare for passages, prefixed for queries. `test_prefix_is_not_applied_twice_to_queries` and `test_prefixed_query_and_bare_passage_differ_in_vector_space` cover the two adjacent mistakes. The real model is exercised by the opt-in slow test, which checks that 'somewhere quiet to work on my laptop' ranks the quiet café above the noisy ramen counter.

### D101 — D-dense-2 — L2-normalized vectors + IndexFlatIP, so inner product IS cosine similarity

**Decision.** Normalize every vector to unit length (`l2_normalize`) and use `faiss.IndexFlatIP`. Zero-norm rows are left at zero rather than divided.

**Alternatives.** (a) Unnormalized vectors with IndexFlatL2 (Euclidean). (b) Unnormalized with IndexFlatIP. (c) Let sentence-transformers normalize via `normalize_embeddings=True`.

**Why.** a·b = |a||b|cos(a,b), so for unit vectors the dot product *is* the cosine — an inner-product index becomes a cosine index with no cosine metric needed, and the scores land in [-1, 1] where they are interpretable and comparable across queries. (b) is the silent-bug version: long documents get larger magnitudes and outrank better matches. (c) was rejected deliberately — normalization is this module's invariant and we want it asserted by our own test rather than delegated to a library flag whose default could change. The zero-vector guard exists because dividing by a zero norm yields NaN, and one NaN row propagates through FAISS and ruins whole result lists; scoring 0.0 against every query is wrong for that one document but harmlessly so.

**How it could be tested.** `test_embeddings_are_unit_length`, `test_inner_product_of_identical_text_is_one` (identical text scores exactly 1.0), `test_search_returns_cosine_scores_in_range`, `test_normalize_disabled_leaves_magnitudes_alone` (the config knob is real, so it can be ablated), and `test_zero_vector_does_not_become_nan` / `test_l2_normalize_handles_mixed_rows`.

### D102 — D-dense-3 — Exact search by default; HNSW supported but its M/efSearch are untuned and unmeasured

**Decision.** `IndexFlatIP` is the default. `HNSWFlat` is built with `METRIC_INNER_PRODUCT`, M=32, efConstruction=200, efSearch=64 — all three hardcoded in `dense.py` with comments saying they are starting points, not tuned results.

**Alternatives.** HNSW from the start for speed; IVF-Flat; expose M/efSearch in configs/base.yaml.

**Why.** Exact brute force means every retrieval quality number this project reports contains zero approximation error, so a later HNSW recall loss can be attributed to HNSW rather than confounded with it. At one metro's business count an exhaustive scan is expected to be fast enough that approximation buys nothing yet - the quality-vs-latency curve is what turns that expectation into a number, and `make explore` is what says how many vectors the scan is over. The HNSW parameters are hardcoded because `configs/base.yaml` is a fixed contract that exposes only `index_type` — see blocked_on; I did not edit it.

**How it could be tested.** `test_default_index_type_is_exact` pins the default; `test_hnsw_index_builds_and_searches` checks HNSW works and explicitly does NOT claim it matches exact search; `test_unknown_index_type_is_rejected` covers the error path. The recall cost of HNSW versus exact is currently UNMEASURED and must be measured by the eval harness before HNSW is turned on.

### D103 — D-dense-4 — Index, id mapping, and provenance metadata are saved as one unit and validated on load

**Decision.** `save()` writes `index.faiss`, `ids.json` (position -> business_id, in document order), and `meta.json` (model, index_type, dim, n_documents, normalize, query_prefix, config_fingerprint). `load()` rejects a dim mismatch or a query_prefix mismatch. The constructor rejects any id list whose length differs from `index.ntotal`. `build()` rejects duplicate business_ids.

**Alternatives.** Store only the FAISS file and recover ids from the document order at load time; use FAISS's own `IndexIDMap`.

**Why.** FAISS returns integer row positions, never business ids. An off-by-one in that mapping corrupts every result while the system keeps looking healthy — each query still returns ten businesses, they are just the wrong ten, and no smoke test notices. Recovering ids from document order at load time would make correctness depend on a file elsewhere staying byte-identical. The prefix and dim checks catch the other invisible failure: querying a bge index with a differently-trained 384-dim encoder (many small models are 384-wide) returns confidently-ranked nonsense. Duplicate ids are rejected because one copy then becomes unreachable and the other can be double-counted in the eval pool.

**How it could be tested.** `test_save_load_round_trip_returns_identical_results` compares ids and scores across a trip through disk; `test_id_mapping_preserves_document_order`; `test_id_mapping_length_mismatch_is_rejected`; `test_load_rejects_an_encoder_of_the_wrong_width`; `test_load_rejects_a_different_query_prefix`; `test_build_rejects_duplicate_business_ids`; `test_saved_metadata_records_what_built_the_index`.

### D104 — D-dense-5 — Dense artifacts moved to artifacts/dense/<fingerprint> to match the BM25 half

**Decision.** `dense_index_dir(cfg)` returns `artifacts/dense/<fingerprint>`. I originally wrote `artifacts/<fingerprint>/dense` and changed it after `bm25.py` landed with `artifacts/bm25/<fingerprint>`.

**Alternatives.** Keep `artifacts/<fingerprint>/<half>` (groups one experiment's artifacts together) and ask the BM25 owner to move.

**Why.** A consistent tree matters more than which of the two shapes is better, and the BM25 module was already written. `artifacts/<half>/<fingerprint>` also makes 'delete every dense index' a single glob. Only my own file's path changed, so no other agent's code was touched.

**How it could be tested.** `test_dense_and_bm25_artifact_layouts_match` asserts both halves resolve to `<root>/<half>/<fingerprint>` for the same config; `test_artifact_dir_changes_with_the_config` asserts a config change (exact -> HNSW) changes the directory, so a stale index can never be silently reused.

### D105 — D-dense-6 — Batching is done by DenseEmbedder, and progress reports only measured quantities

**Decision.** `DenseEmbedder` chunks texts at `cfg.retrieval.dense.batch_size` and calls `encoder.encode` per chunk; the real encoder is told to encode its whole chunk at once. Progress is a single `\r` line on stderr showing `done/total` and elapsed seconds, nothing else.

**Alternatives.** Let sentence-transformers do the batching with `show_progress_bar=True`; use rich.progress; print a rate or an ETA.

**Why.** Chunking above the encoder makes batching behaviour identical for the real model and the test fake, so a batching bug cannot hide behind 'the fake does it differently'. stderr keeps `make index > log.txt` readable and lets the test capture it. No rate or ETA is printed because a rate extrapolated from a half-finished loop is not a measurement, and this project does not print numbers it has not measured.

**How it could be tested.** `test_encoding_respects_configured_batch_size` asserts the fake saw chunk sizes [3, 3, 2] for 8 texts at batch_size 3; `test_empty_input_returns_correctly_shaped_array` covers the (0, dim) case; `test_progress_goes_to_stderr` asserts stdout stays empty (stdout carries the build report).

### D106 — D-build-7 — One command builds both halves from one in-memory document list, and refuses to disagree

**Decision.** `build_index.py` loads the documents once, builds BM25 and FAISS in the same process from that same list, then raises if the two halves report different document counts. It calls `bm25.build_index` + `save` rather than `bm25.load_or_build`.

**Alternatives.** Separate `make index-bm25` / `make index-dense` targets; use `load_or_build` so repeat runs are fast.

**Why.** If the two indexes were ever built from different document sets, Reciprocal Rank Fusion would still happily fuse their rankings and the system would be subtly wrong in a way no test on either half could catch — the count check turns that into a loud failure. `load_or_build` was rejected because `make index` means 'build the index now': silently reusing a cache would make the reported build time meaningless and could leave a stale index in place after a change to the document builder. `--only bm25|dense` exists for iterating on one half.

**How it could be tested.** `test_build_index_builds_and_saves_the_bm25_half` runs the BM25 half end to end through the real module and reloads the result. End-to-end smoke test with 4 synthetic documents and the real bge-small model: both halves built, 4 docs each, reloaded and queried in a second process — 'somewhere quiet to work with wifi' returned The Quiet Shelf then Lantern Coffee, 'noodles late at night' returned Midnight Ramen, 'cheap sandwich any hour' returned Always Open Deli. That is a correctness smoke test on 4 documents, NOT a quality measurement.

### D107 — D-build-8 — require_metro() is called before anything else, including the preflight

**Decision.** `main()` calls `cfg.require_metro()` immediately after loading the config, before locating documents or warning about anything.

**Alternatives.** Validate the metro after loading documents, so the owner sees all problems at once.

**Why.** Indexing the wrong scope is not merely wasted compute: every relevance label the owner collects against that index has to be thrown away, and labeling is hand work that cannot be regenerated, unlike every other artifact in the project. Failing before a single file is opened is the cheapest possible failure.

**How it could be tested.** `test_build_index_refuses_to_run_without_a_metro` calls `main(['--config', 'configs/base.yaml'])` against the committed config (metro is null there by design) and asserts it raises before touching the filesystem.

### D108 — D-build-9 — Build report prints only measured values; index size is read from disk

**Decision.** The report prints document count, wall-clock seconds, and `directory_size_bytes()` summed from the filesystem, in binary units labeled KiB/MiB.

**Alternatives.** Estimate index size from n_documents x dim x 4 bytes; report embeddings/second.

**Why.** The computed estimate would be wrong (it ignores the HNSW graph, FAISS headers, and the id/meta JSON) and would be a fabricated number in a project whose entire value is that its numbers are real. Index size is a genuine serving constraint — it bounds what fits in a free-tier container — so it is worth measuring properly. Binary units are labeled so the printed figure matches `du -h`.

**How it could be tested.** `test_directory_size_is_measured_not_estimated` (1000-byte file in a subdirectory + 24-byte file = exactly 1024) and `test_format_bytes_uses_binary_units`.

### D109 — D-test-10 — The real-model test is gated twice: the `slow` marker AND an env var

**Decision.** `test_real_bge_small_end_to_end` carries `@pytest.mark.slow` and `@pytest.mark.skipif(not os.environ.get('CONCIERGE_SLOW_TESTS'))`. Run it with `CONCIERGE_SLOW_TESTS=1 pytest -m slow`.

**Alternatives.** Marker only; delete the test; `-m 'not slow'` in the Makefile.

**Why.** The marker alone does not skip anything — `addopts` in pyproject.toml is `-q --strict-markers`, with no `-m 'not slow'`, so a plain `pytest` RUNS every slow test (I hit this: the first run reached for the network and then killed the interpreter). A plain `pytest` must never need the network, and pyproject.toml is not my file to change, so the guard went in my own file where it is unambiguous. The test is kept because the fakes verify the wiring and only the real model can confirm the prefix convention produces a sensible ranking.

**How it could be tested.** `pytest tests/test_dense.py` passes with exactly one skip, and the skip reason names the env var. `CONCIERGE_SLOW_TESTS=1 pytest -k real_bge -s` passes against the real bge-small: dim 384, both ranking assertions hold, and the save/load round trip matches.

### D110 — D-env-11 — faiss-cpu and torch ship separate OpenMP runtimes; this segfaults at realistic scale (measured)

**Decision.** Diagnosed it, did NOT apply the fix (dependencies are the orchestrator's), and added a `warn_on_duplicate_libomp()` preflight to `build_index.py` that names both copies and prints the one-line fix before a long build starts. The venv was left byte-identical to how I found it (libomp sha256 verified before and after).

**Alternatives.** (a) Set KMP_DUPLICATE_LIB_OK=TRUE. (b) Symlink faiss's bundled libomp at torch's copy. (c) Embed in a subprocess and build the index in another, keeping torch and faiss in separate processes. (d) Drop faiss for hnswlib or a numpy brute-force scan.

**Why.** Measured on this machine, not assumed. Both wheels bundle their own LLVM libomp (different builds: sha256 bb055b57... vs 6256bee0...). Once both runtimes initialize, the process dies — `OMP: Error #15` + `Abort trap: 6` whichever order they are imported. Scale matters: 4 and 500 vectors survive (FAISS does not enter a parallel region), 20,000 vectors segfault with exit code 139 and NO error message at all, which is worse than the abort. How a Yelp metro compares to that 20,000-vector threshold is not measured here (`make explore` prints the business count once the metro is picked), so the condition is stated as one: if the metro index exceeds the 20,000 vectors that segfaulted on this machine, `make eval` and `make serve` both die silently once the index is real. (a) is disqualified on evidence: with KMP_DUPLICATE_LIB_OK=TRUE a trivial workload deadlocked for over 10 minutes instead of aborting, and the OpenMP warning itself says it may silently produce incorrect results — unacceptable in a project whose value is correct measurements. (b) was verified to work at n=20000 (exit 0) and is the cheapest real fix. (c) fixes indexing but not serving, unless the ONNX Runtime export already planned for Phase 5 removes torch from the serving process entirely — worth noting as an argument for doing that export. (d) is the fallback if (b) proves fragile; an exact numpy matmul has no OpenMP runtime to conflict with at all, and whether it is fast enough over this metro's vectors is unmeasured - `make explore` gives the vector count and the latency curve would settle it.

**How it could be tested.** Reproduced three ways and bisected by scale: `scale.py N` loads the real bge model, warms the torch thread pool, then builds and searches a FAISS IndexFlatIP of N vectors. N=4 and N=500 return normally; N=20000 exits 139 (SIGSEGV) with empty stdout. Re-running N=20000 with faiss's libomp symlinked to torch's returns normally (exit 0); the symlink was then removed and the original file restored (hash re-verified). The preflight's own contract is tested by `test_duplicate_libomp_check_returns_paths_or_nothing`, which asserts only the shape of the result — whether this machine has the collision is an environment fact, not something a test should assert.

### D111 — D-env-12 — sentence-transformers 6.x renamed the embedding-dimension getter

**Decision.** `SentenceTransformerEncoder` tries `get_embedding_dimension()` and falls back to `get_sentence_embedding_dimension()`.

**Alternatives.** Call the new name only; call the old name and live with the FutureWarning; pin sentence-transformers.

**Why.** The installed version (6.1.0) emits a FutureWarning for the old name, so the old name will eventually break; the new name does not exist on 3.x, which is what pyproject.toml's floor allows. The lockfile will outlive this particular minor release, and a renamed method here means the index cannot be built at all. Two lines buys version-boundary safety.

**How it could be tested.** Observed the FutureWarning on 6.1.0, then confirmed the slow test passes with no warning after the change. Both branches are not separately unit-tested — only the installed version's branch is exercised.

## Stage 3 — cross-encoder reranker

### D112 — Plain transformers for the training loop, not sentence-transformers CrossEncoder

**Decision.** concierge/rerank/train.py implements an explicit PyTorch loop over AutoModelForSequenceClassification: tokenize, forward, loss, backward, clip, optimizer.step, scheduler.step, validate.

**Alternatives.** sentence_transformers.CrossEncoder.fit() (three lines); HuggingFace Trainer.

**Why.** The owner must defend every line in an interview, and fit()/Trainer hide the loss, the batching, the scheduler and the evaluator behind version-dependent defaults. The explicit loop is longer and fully visible. It also removes a train/serve seam: model.py serves with the same model class, so a checkpoint is loaded at serving time by the code path that saved it.

**How it could be tested.** tests/test_rerank.py::test_training_smoke_run_writes_a_checkpoint_and_metadata (slow) runs a real one-epoch fine-tune on injected documents and asserts the checkpoint plus run metadata land where model.py looks for them.

### D113 — Pointwise binary cross-entropy objective

**Decision.** Positive business label 1.0, each mined hard negative 0.0, BCE-with-logits on the single-logit head (CrossEntropy if a 2-class head is ever used; any other head count is refused).

**Alternatives.** Listwise/group softmax (MultipleNegativesRanking, LambdaRank-style) which optimizes ordering directly, i.e. what NDCG measures.

**Why.** BCE continues the objective cross-encoder/ms-marco-MiniLM-L-6-v2 was pretrained with, rather than bolting a new objective onto pretrained weights, and it is easy to reason about. The listwise alternative is likely stronger but needs each query's whole group intact in one batch; it is named in the module docstring and is a contained change in _loss_for_batch, to be judged by the eval harness rather than by assertion.

**How it could be tested.** Not a quality claim, so not measured here: the smoke run only pins that the loop runs and the head reduction matches the label shape. The pretrained-vs-fine-tuned comparison belongs to eval/run_eval.py.

### D114 — Gold-leakage guard is a hard crash with no override flag

**Decision.** train() loads eval/gold_queries.csv and raises GoldLeakageError if any training OR validation query matches a gold query after casefold + whitespace collapse. A missing gold file raises FileNotFoundError and refuses to train. The check runs before any weights are loaded, and its result (gold count, training count, 0 collisions) is written into the run metadata.

**Alternatives.** Warn and continue; a --allow-leakage escape hatch; byte-exact comparison; checking only the train split; skipping the check when the gold file is absent.

**Why.** A leaked gold query inflates every reported NDCG@10 and nothing about the number looks wrong. Every escape hatch here is a future retracted result. Byte equality is weaker than it looks (differing case/spacing walks past it), so normalized exact match is used — still exact, strictly stronger; fuzzy near-duplicate removal stays with the synth stage and its cosine threshold. The val split is checked too because it tunes decisions. 'We could not check' must never look like 'we checked and it was clean', hence the refusal on a missing file.

**How it could be tested.** tests/test_rerank.py: fires on a planted gold query; fires on 'Good Pho'/' good pho '/'GOOD PHO' variants; passes on clean data and records the check; refuses without the gold file; rejects a gold CSV with no query column; and (slow) test_training_refuses_a_leaky_split_before_touching_weights proves the crash precedes training.

### D115 — Splits are read from disk and never re-derived

**Decision.** load_split() reads crossencoder_train.jsonl / crossencoder_val.jsonl, searching artifacts/synth/<fingerprint>/, artifacts/synth/, data/processed/synth/ and data/processed/ (each also with a splits/ subdirectory). A missing file lists every path searched and states that the trainer never re-splits.

**Alternatives.** Re-splitting from a single pairs.jsonl using cfg.synth.splits and cfg.seed; silently falling back to any available JSONL.

**Why.** Stage 4's out-of-fold guarantee (the cross-encoder feature must come from queries the cross-encoder never saw) only holds if both trainers read the same saved split. Re-splitting with the same seed would look safe and break the moment either side's data changed. Guessing which file is the training split could silently train on the LambdaMART split.

**How it could be tested.** tests/test_rerank.py: finds a file in the splits/ layout; a missing split lists the searched paths and the 'never re-splits' message; an empty split raises instead of running an empty training job.

### D116 — Truncation is exposed as a measurable diagnostic, not a hope

**Decision.** PairScorer carries count_tokens() and document_token_budget() alongside score_pairs(); CrossEncoderReranker.truncation_report() / truncation_report_for_documents() return per-document tokens, tokens kept, kept fraction, count and fraction truncated, and worst offenders, plus a number-only summary() line.

**Alternatives.** Trusting the tokenizer's silent truncation; a boolean 'was anything truncated' flag; computing it inside rerank() on every query.

**Why.** max_length (512 tokens) covers query + document; documents.max_chars is 1800 characters, and whether that fits is a tokenizer fact, not something to guess. Silent truncation caps achievable quality with no error. Kept as a separate explicit call so serving never tokenizes twice unasked, and the three-method protocol lets a test fake answer with whitespace tokens so the arithmetic is tested without a model.

**How it could be tested.** tests/test_rerank.py: exact arithmetic over a 10-token budget (budget 5, one of two documents clipped, kept fraction 5/8, mean/min, summary text); business-id-keyed variant; empty document reports 1.0 rather than dividing by zero; vacuous report over no documents; slow test asserts the real tokenizer reports a real budget and one clipped document.

### D117 — rerank() adds scores, never overwrites, and ties keep upstream order

**Decision.** rerank() returns new Candidates carrying every upstream score/rank plus SCORE_CROSS_ENCODER and a 1-based RANK_CROSS_ENCODER; ordering is a stable sort on descending cross-encoder score; top_k clamps without erroring; top_k < 1 raises.

**Alternatives.** Collapsing to one score; tiebreaking by business_id or by dict order; erroring when top_k exceeds the candidate count.

**Why.** Stage 4 consumes BM25/dense/RRF as features, so losing them would make the ablation untraceable. Upstream (RRF) rank is a real signal, whereas hash/dict order is noise that would make two eval runs disagree and quietly widen every confidence interval. Short candidate lists are normal for heavily filtered queries, so a cap is a cap.

**How it could be tested.** tests/test_rerank.py: upstream scores/ranks preserved and inputs unmutated; reordering actually overturns retrieval order; tie test asserts input order is kept AND that reversed input comes back reversed (which is what proves the order came from the input); top_k truncation and over-asking; top_k=0 raises.

### D118 — A candidate with no document fails loudly by default

**Decision.** on_missing_document='error' (default) raises KeyError naming examples; 'rank_last' keeps such candidates at the end of the list with a cross-encoder rank but NO cross-encoder score.

**Alternatives.** Silently dropping them; scoring them 0.0 or -inf.

**Why.** The retrieval index and the document store are built from the same file, so a candidate without a document means they have drifted — a build bug that a silently shortened shortlist would hide. When the lenient mode is used, a missing score is honest while a 0.0 would be a fabricated measurement that Stage 4 would learn from.

**How it could be tested.** tests/test_rerank.py: default raises; rank_last orders the undocumented candidate last and asserts SCORE_CROSS_ENCODER is absent from it; an unknown policy string raises.

### D119 — The cross-encoder scores semantic_text, not the raw query

**Decision.** query_text_for(parsed) returns ParsedQuery.semantic_text, falling back to raw only when semantic_text is blank.

**Alternatives.** Always the raw user string; concatenating raw + rendered constraints.

**Why.** Hard constraints (open past 10pm, price) are enforced arithmetically by the Stage 2 filters. Leaving them in the text asks the model to judge something it cannot verify from prose and double-counts constraints already applied. The fallback exists because scoring every candidate against an empty string would make the stage a no-op that still costs full latency. It is one named function so the eval harness can ablate the choice.

**How it could be tested.** tests/test_rerank.py: prefers semantic_text, falls back to raw when blank, and rerank() is asserted to hand the semantic text to the scorer.

### D120 — ONNX export uses the legacy TorchScript exporter (dynamo=False)

**Decision.** torch.onnx.export(..., dynamic_axes=..., opset_version=17, dynamo=False) when the parameter exists, with a comment stating what to change when the legacy exporter is removed.

**Alternatives.** The dynamo (torch.export) exporter, which is the default from torch 2.9; optimum's exporter.

**Why.** Measured on this project's env (torch 2.14): the dynamo path additionally requires onnxscript, which is not in pyproject's onnx extra; it writes weights to a sidecar model.onnx.data file (59 KB graph + 91 MB data) so the served artifact becomes two files that must travel together, where the legacy exporter writes one self-contained file; and it raised the opset to 18 and converted back down, i.e. rewrote the graph we asked for. The legacy path is deprecated, so the switch instructions and the fact that the parity check is the test for the new graph are written into the comment.

**How it could be tested.** Ran the real export and parity check with onnx/onnxruntime installed into the session scratchpad only (never into the project venv): export succeeded as a single 91 MB file, and tests/test_rerank.py::test_onnx_export_matches_pytorch_scores passed (it is skipped unless onnx is importable).

### D121 — Parity check also re-scores every pair unbatched, and asserts ordering

**Decision.** check_parity compares PyTorch vs ONNX batched (max/mean abs diff) AND re-scores the same pairs one at a time; within_tolerance requires both to be within the stated 1e-4, and a changed ranking raises even when the differences are within tolerance.

**Alternatives.** A single batched max-abs-diff comparison; relying on tolerance alone and inferring ordering.

**Why.** Inside one batch every pair is padded to the same length, so a batched comparison cannot tell whether the graph is genuinely shape-dynamic — and the TorchScript tracer warned during export that it was baking a shape-dependent branch into a constant. Batch 1 changes both the batch size and the sequence length, so a frozen axis or padding dependence shows up. Ordering is what the user sees, so it is asserted separately.

**How it could be tested.** Measured on this machine with a pretrained checkpoint at max_length 256 over 12 probe pairs: max|Δ| 4.768e-06 batched, 6.676e-06 unbatched, ordering identical — inside the 1e-4 tolerance. Unit test test_parity_fails_when_only_the_unbatched_scores_disagree pins that the unbatched number alone can fail the check.

### D122 — No assumption that ONNX is faster; the CLI says when it is not

**Decision.** The export script times both backends and prints measured ms/pair and a speedup; when the measured speedup is below 1.0 it prints that ONNX was measured slower and that serving.use_onnx should stay false.

**Alternatives.** Documenting an expected speedup; flipping serving.use_onnx on after a successful export.

**Why.** In the verification run above, ONNX Runtime measured SLOWER than PyTorch on this machine (torch 8.89 ms/pair vs onnx 21.01 ms/pair, 0.42x, pretrained weights, probe documents, default thread settings) — plausibly because torch uses fused SDPA attention that the exported graph does not. Serving a slower backend for no reason is a regression, and a project whose value is its measurements must not ship an assumed win. Phase 5 should re-measure on the serving host with real documents before enabling ONNX.

**How it could be tested.** The numbers above are from running export + check_parity directly; they are machine- and input-specific and are not a project result. Nothing in the code hardcodes them.

### D123 — Expected runtime is extrapolated from measured steps, never guessed

**Decision.** After the first 20 real training steps, train.py records measured seconds/step and prints an extrapolated total, labelled as an extrapolation, and stores both in the run metadata.

**Alternatives.** A hardcoded 'about X minutes on CPU' note; timing a throwaway warmup; printing nothing.

**Why.** The owner needs to know within a minute whether to move to Colab, and rule 4 forbids invented numbers. Extrapolating from real steps costs no extra compute and is honest about being an extrapolation.

**How it could be tested.** Exercised by the slow smoke run (which has fewer than 20 steps, so the field correctly stays null rather than reporting a made-up estimate); the wall-clock and loss curve fields are asserted in that test.

### D124 — Smoke-run checkpoints are quarantined

**Decision.** --limit-examples writes to artifacts/rerank/<fingerprint>/smoke/ and sets is_smoke_run in the metadata; resolve_weights only looks at <artifact dir>/model.

**Alternatives.** Writing to the normal artifact directory and relying on a printed warning.

**Why.** resolve_weights treats any checkpoint it finds as 'fine-tuned', which is a distinct ablation row. A checkpoint trained on fifty pairs landing in the normal path would silently become the served model and produce a fine-tuned-looking result from a run that learned nothing.

**How it could be tested.** Not covered by an automated test (would require a second slow training run); the path logic is a two-line branch and the slow test pins the explicit out_dir case.

### D125 — Weight provenance is a returned value, not a log line

**Decision.** resolve_weights returns (name_or_path, source_label) and CrossEncoderReranker.weights_source carries it; the label distinguishes an explicit path, a fine-tuned checkpoint, and 'pretrained base_model (no fine-tuned checkpoint found)'.

**Alternatives.** Silently falling back to the base model; raising when no checkpoint exists.

**Why.** The ablation table has separate pretrained and fine-tuned cross-encoder rows. A silent fallback would mislabel a headline result — the worst kind of reporting error, one that looks like a result. Falling back is still the right runtime behaviour (the pipeline works and it is a legitimate row), so it is made visible instead of fatal.

**How it could be tested.** tests/test_rerank.py: fallback label contains 'pretrained'; a planted checkpoint directory yields the 'fine-tuned' label and its path; rerank_dir is keyed by fingerprint and changes when cfg.rerank.lr changes.

### D126 — Device detection cuda > mps > cpu, overridable

**Decision.** detect_device() honours an explicit request first, else cuda, else mps, else cpu; inference defaults to CPU, which is the required serving path.

**Alternatives.** CPU only; always mps on Apple Silicon.

**Why.** mps is generally faster than CPU for a model this small - not measured here, and the export run in D122 is a reminder that assumed speedups do not always survive measurement - but some ops still misbehave depending on the torch build, and the fix must be --device cpu rather than a code change. Inference stays CPU-first because the laptop serves on CPU; the device argument exists for offline batch feature generation.

**How it could be tested.** tests/test_rerank.py::test_detect_device_honours_an_explicit_request (marked slow because it is the only non-model test that imports torch — see blocked_on).

## Stage 4 — LambdaMART and features

### D127 — D0xx — Stage 4 features are one pure function of (ParsedQuery, Candidate, Business)

**Decision.** `concierge.ltr.features.build_feature_row(parsed, candidate, business) -> dict[str, float]` is the only place features are produced. Training materializes the contract dataclasses from its JSONL records and calls it; serving calls the same function on live objects. No precomputed offline feature table exists.

**Alternatives.** (a) A precomputed feature table written by the data stage and read by the trainer, with the serving path computing features live. (b) A feature class holding config/indexes, so features could depend on tunables.

**Why.** Training/serving feature skew is the classic silent production-ML bug: the offline job encodes a missing score as 0, the online path as NaN (or vice versa), offline metrics look fine, served quality is worse, and nothing errors because the two code paths are never compared. With one pure function there is no second implementation to drift from — a feature can be wrong, but it is wrong identically in both places, which is findable. Purity also makes every feature trivially unit-testable with hand-built objects and no fixtures.

**How it could be tested.** tests/test_ltr.py::test_example_from_json_materializes_the_contract_types builds features from a raw training record and checks the values; the end-to-end run test trains through the same path. The remaining skew risk is that no test yet compares a served row against a trained row — add that once serving exists (pipeline passes a Candidate through both).

### D128 — D0xx — Missing feature values are NaN, never a sentinel like 0 or -1

**Decision.** Absent BM25/dense/cross-encoder scores, absent RRF rank, an undefined constraint fraction (0/0), an unchecked or unknown category match, and distance with no geo constraint or no coordinates are all NaN. Real measured zeros stay 0.0.

**Alternatives.** (a) 0.0 for missing scores. (b) -1 as an explicit out-of-range sentinel. (c) A paired is_missing indicator column per feature.

**Why.** LightGBM handles NaN natively — it learns per split which side missing values go to and ignores them when choosing the threshold. A sentinel is a real ordered value, and a tree only asks `feature <= t`: 0.0 for a missing dense score asserts the candidate is weakly similar (a claim the data never made — the dense retriever simply never returned it), and -1 for a missing rating invents an order between 'bad' and 'unknown'. Worse, a split at `stars <= 0.5` fuses missing with 0-star forever; the information is destroyed before the model sees it. Indicator columns (c) would work but double the feature count and make the importance plot unreadable, which is the plot's whole purpose here.

**How it could be tested.** test_absent_stage_scores_are_nan_not_zero, test_present_zero_score_stays_zero (the mirror image — a measured 0 must not become NaN), test_unconstrained_query_has_nan_fraction_but_defined_count, test_distance_feature_matches_haversine_and_is_nan_without_geo.

### D129 — D0xx — Constraint features are three, not one, and are read from the candidate rather than recomputed

**Decision.** Three features: n_constraints_satisfied (count), frac_constraints_satisfied (count / checked, NaN when nothing was checked), has_unknown_constraint_data (indicator). All read `Candidate.constraints` as decided by `concierge.retrieval.filters`; nothing is re-derived. Geo and category are both excluded from the counts, each because it already has a feature of its own (`distance_km`, `category_match`). A query that requested categories but whose candidate carries no category check is the one place something *is* re-derived: `_category_match` calls the same `filters.check_categories` the filter stage calls, because refusing instead made every ablation query containing a category word score zero - a row that measured a crash while reading as "the filters are worth a lot".

**Alternatives.** (a) One boolean 'all constraints satisfied'. (b) Recompute constraint outcomes inside the feature builder. (c) Treat the undefined fraction as 0.0.

**Why.** The count and the fraction say different things and neither is redundant: 2-of-2 and 2-of-6 are very different answers, and a threshold on a raw count means different things across queries of different constraint counts. The UNKNOWN indicator is what lets the model learn, from labels, how much to discount missing data — the discount D007 currently hard-codes as a 0.5 penalty. Re-deriving outcomes (b) would create a second authority that can disagree with the filter stage, so the ranker's features would describe a different system than the one that produced the results and the 'why this result' explanation. 0.0 for the undefined fraction (c) would claim an unconstrained query satisfied none of its request.

**Consequence to watch — the feature's value set changed.** `filters.check_categories` emits only SATISFIED or UNKNOWN for a category (a lexical miss is "we cannot tell", not "wrong kind of place"), so `category_match` is now two-valued, {1.0, NaN}, and never 0.0; the VIOLATED branch in `_category_match` is unreachable safety net. Any feature file written while a miss was VIOLATED carries 0.0s and a different `has_unknown_constraint_data` distribution than the live path now produces, which is train/serve skew arriving through a data file rather than a second code path. Such a file must be regenerated before the next LambdaMART run, and the durable fix is to fingerprint the constraint semantics into the examples file and refuse a mismatch on load, the way the split manifest already is.

**How it could be tested.** test_constraint_counts_three_valued, test_all_constraints_satisfied_gives_fraction_one, test_unconstrained_query_has_nan_fraction_but_defined_count, test_geo_constraint_excluded_from_counts, plus the tests pinning the recomputed category check. Whether the three features beat one boolean is unmeasured — answerable from the importance table once real data exists.

### D130 — D0xx — Out-of-fold enforcement checks both cross-encoder splits and both LambdaMART splits

**Decision.** `assert_out_of_fold` raises `LeakageError` if any query in ltr_train OR ltr_val appears in crossencoder_train OR crossencoder_val. It reads the manifest written by the synth stage and never writes or re-splits. `cfg.ltr.enforce_oof_crossencoder_feature: false` downgrades it to a logged ERROR, and the model metadata records which mode produced it.

**Alternatives.** (a) The brief's minimum: ltr_train vs crossencoder_train only. (b) Check scores statistically (e.g. flag a suspiciously high mean cross-encoder score) instead of checking ids. (c) Let the trainer re-split when it detects overlap.

**Why.** crossencoder_val never fitted the model, but early stopping selected a checkpoint with it, so scores there are optimistic too — same bias, smaller magnitude; strict is the right default for a check whose failure mode is a silently better-looking number. ltr_val is included because it chooses the boosting iteration, so contamination there biases the model that actually ships. (b) cannot distinguish inflation from a genuinely good reranker. (c) is disqualifying: a trainer that can re-split its own data can re-split its way out of a leakage failure, so the split must stay owned by the fixed-seed code that generated the queries.

**How it could be tested.** test_planted_overlap_with_crossencoder_train_hard_fails plants 'q2' in both splits and asserts the refusal (and that the message explains the consequence, not just the fact); test_planted_overlap_with_crossencoder_val_also_fails; test_overlap_in_ltr_val_also_fails; test_out_of_fold_check_can_be_disabled_but_screams; test_run_refuses_a_leaky_manifest_before_training asserts no artifact is written for a leaky run.

### D131 — D0xx — Both leakage checks fail closed when their input is missing

**Decision.** A missing split manifest is a FileNotFoundError, and a missing eval/gold_queries.csv is a LeakageError unless `--allow-missing-gold` is passed; the override is logged and recorded in metadata as `gold_leakage_check_ran: false`.

**Alternatives.** Skip the check with a warning when the file is absent (the usual convenience default).

**Why.** A safety check that passes because it had nothing to compare against is worse than no check at all: it prints reassurance it has not earned, and six weeks later nobody remembers the file was missing. Failing closed forces the override to be a visible, deliberate act that ends up in the artifact's metadata, so a reader of the metadata can tell a verified model from an unverified one.

**How it could be tested.** test_missing_manifest_is_an_error_not_a_pass, test_missing_gold_file_fails_closed, test_metadata_records_a_skipped_gold_check_rather_than_implying_it_passed.

### D132 — D0xx — group_sizes refuses non-contiguous query ids instead of guessing

**Decision.** `group_sizes(query_ids)` run-length-encodes the ids and raises ValueError if any id reappears in a later block, or if the input is empty; `sort_by_query` is the stable, first-appearance-ordered way to make rows contiguous. `RankingData.__post_init__` re-checks that groups sum to the row count.

**Alternatives.** Silently sort inside group_sizes; or accept interleaved ids and emit one group per run.

**Why.** LightGBM's `group` is a list of consecutive run lengths and it never sees query ids, so ['a','b','a'] has no valid group array. Guessing [1,1,1] trains on pairs formed across different queries — a meaningless objective that still converges and still reports a plausible NDCG. That is the textbook silently-wrong LambdaMART bug, so the only honest response is to refuse. Sorting silently would hide a caller bug that likely indicates the upstream data is shuffled.

**How it could be tested.** test_group_sizes_hand_built_three_query_example (3/1/2 → [3,1,2], sum == rows), test_group_sizes_refuses_interleaved_rows, test_group_sizes_refuses_empty_input, test_sort_by_query_makes_blocks_contiguous_in_first_seen_order, test_ranking_data_rejects_inconsistent_pieces.

### D133 — D0xx — The evaluated training set must be named "training" so early stopping ignores it

**Decision.** In `train_lambdamart` the training Dataset is passed in valid_sets under the name "training" (not "train"), with the validation set named "val".

**Alternatives.** Leave the training set out of valid_sets entirely; or name it anything and trust early stopping.

**Why.** LightGBM's early-stopping callback skips the training data by *name*, comparing the eval entry's dataset name against `Booster._train_data_name`, whose default is exactly "training". Name it "train" and the callback treats it as a validation set: train NDCG rises monotonically, so at the final round the callback reports that entry's best iteration — the last one. Early stopping then silently has no effect and the saved model is the overfitted final model. Keeping the set in valid_sets (correctly named) is worth it because the train/val gap is recorded for diagnosing overfit.

**How it could be tested.** test_recovers_monotone_label asserts 1 <= best_iteration <= n_estimators and that a real validation NDCG@10 was recorded; test_ablated_model_cannot_see_the_non_text_features asserts val_ndcg is None when there is no validation split. A direct test that best_iteration < n_estimators on a deliberately overfitting problem is not written yet.

### D134 — D0xx — Determinism is pinned in LightGBM params, and asserted byte-for-byte

**Decision.** `lgb_params` sets seed, bagging_seed, feature_fraction_seed and data_random_seed all to cfg.seed, plus deterministic=True, force_row_wise=True, num_threads=1.

**Alternatives.** Set only `seed` and rely on LightGBM deriving the rest; leave threading at the default.

**Why.** The project promises every number is regenerable with one command. Without these, LightGBM may pick a different histogram construction path depending on thread count and data layout, and float summation order changes with thread count, so the model drifts run to run even with a fixed seed. Deriving the sub-seeds from `seed` is version-dependent behaviour, so they are pinned explicitly. The cost is speed at a scale where speed does not matter.

**How it could be tested.** test_training_is_deterministic_under_a_fixed_seed asserts byte-identical predictions (assert_array_equal, not approx) and identical best_iteration and val NDCG across two runs; test_lgb_params_pin_every_seed pins the params themselves.

### D135 — D0xx — Non-text ablation drops columns rather than masking them, into a separate artifact directory

**Decision.** `--ablate-non-text` trains on TEXT_FEATURE_NAMES only (bm25, dense, rrf_rank, cross_encoder); `FeatureMatrix.select` physically drops the other columns. Artifacts go to `artifacts/ltr/<config fingerprint>-textonly`, the full model to `artifacts/ltr/<config fingerprint>`.

**Alternatives.** NaN-fill or zero-fill the ablated columns; or write both runs to the same directory.

**Why.** A dropped column is provably unusable by the model; an all-NaN column leaves the feature in the model's schema and makes the ablated model's importance table misleading. Separate directories exist because two runs overwriting each other would make the ablation table unreproducible in the most confusing possible way (the numbers would look plausible and belong to different models). Keyed by config fingerprint per D008.

**How it could be tested.** test_ablated_model_cannot_see_the_non_text_features (booster.feature_name() == text features; stars absent from importances), test_select_drops_columns_rather_than_masking_them, test_run_with_ablation_writes_a_separate_artifact, test_train_val_feature_mismatch_is_caught. The actual ablation *result* is unmeasured — it needs real training data.

### D136 — D0xx — The training loop is verified on a synthetic problem with a known answer

**Decision.** `monotone_dataset` builds a ranking problem where the graded label is a monotone function of `stars` alone and every other feature is noise drawn independently of the label; the test asserts the model ranks each query's labels perfectly and that `stars` tops the gain-importance table.

**Alternatives.** Mock LightGBM and assert it was called with the right arguments; or only test on real data once it exists.

**Why.** This verifies the plumbing — group array, label alignment, column/name alignment — rather than LightGBM itself, and it does so with no data download and no API key. Mocks would pass even if the group array were wrong. The assertion is on label order and not on stars order on purpose: the label buckets ties (stars 4 and 5 are both label 2), so the data contains no evidence for an order inside a bucket, and a test demanding one would reward overfitting. Importance is reported as 'gain' rather than 'split' because split count rewards merely high-cardinality features.

**How it could be tested.** test_recovers_monotone_label (perfect within-query label ordering, plus every high-star candidate outranking every low-star one, plus stars as top gain feature). Tiny-data LightGBM minimums (min_data_in_leaf etc.) are passed only via the test-only `extra_params` argument, so production runs use the config's values untouched.

## Training data — synthetic queries, negatives, judge

### D137 — Leakage guard fails loudly instead of passing silently

**Decision.** load_gold_queries raises GoldSetMissingError when eval/gold_queries.csv is absent, and dedupe_against_gold raises if the gold list is empty. There is no "no gold set, nothing to remove" path.

**Alternatives.** Warn and return every query untouched (the usual default); skip dedup when gold is missing.

**Why.** The gold file does not exist yet (the owner writes it), so the common case TODAY is a missing file. A warning would let a training set be built with the guard effectively disabled while the logs read '0 removed', which is indistinguishable from a clean run. A missing test set must stop the pipeline, not degrade it.

**How it could be tested.** test_gold_dedup_refuses_an_empty_gold_set, test_load_gold_queries_fails_loudly_when_absent.

### D138 — Dedup is exact + embedding cosine, both, in one auditable function

**Decision.** dedupe_against_gold runs four passes in order: exact-vs-gold, cosine-vs-gold (>= cfg.synth.gold_dedup_cosine_threshold = 0.92), exact-internal, cosine-internal. It returns a DedupReport carrying every removal with its matched gold query and cosine.

**Alternatives.** Exact match only (cheap, nearly useless against paraphrases); embedding only (loses the free, auditable half); dedup inside generate_queries (couples the quota-spending step to a model download).

**Why.** Exact matching cannot catch 'quiet cafe with wifi open late' vs 'a quiet coffee shop with wifi thats open late', which is the same test item for every purpose that matters. Keeping dedup separate from generation means the expensive half can be resumed without re-embedding, and the removal count (a required README number) comes from one function with one report.

**How it could be tested.** test_gold_dedup_removes_planted_near_duplicate (a planted paraphrase of a gold query must be removed), plus exact/internal/order-independence/merely-similar tests. Threshold behaviour is tested with an injected hashed bag-of-words embedder, so no model download.

### D139 — Embedding function and LLM client are injected protocols

**Decision.** generate.LlmJsonClient (prompt -> JSON) and generate.EmbedFn (texts -> vectors) are protocols; the real Gemini client and the bge embedder are defaults built lazily from config.

**Alternatives.** Call concierge.query.llm.parse_with_llm; import DenseEmbedder at module top.

**Why.** A leakage guard that can only be tested against a live sentence-transformer and a live API would never actually be tested. Lazy imports also keep torch out of the test suite's import path. concierge.query.llm's Provider implementations hard-code the query-parsing response schema in the request body, so they cannot be reused for generation (see blocked_on).

**How it could be tested.** Every test in tests/test_synth.py runs with stubs; only one needs_api test touches the network and it is skipped without GEMINI_API_KEY.

### D140 — Quota confirmation is enforced in the library, not the CLI

**Decision.** require_confirmation(QuotaPlan, confirm, assume_yes) gates generation and judging. The plan states the API-call count excluding cached batches; a plan with 0 calls never prompts; confirm_on_stdin refuses when there is no terminal.

**Alternatives.** Put the prompt in the CLI only; always prompt; a quiet --yes default.

**Why.** CLAUDE.md section 10 requires stating the call count and asking before spending quota. In the library it cannot be bypassed by a notebook or a cron job. Excluding cached batches matters behaviourally: quoting the total would overstate a resumed run's cost and train the owner to click through the prompt that exists to protect them.

**How it could be tested.** test_generation_refuses_without_confirmation (asserts zero client calls), test_confirmation_states_the_exact_call_count, test_fully_cached_run_asks_nothing, test_judge_grading_requires_confirmation.

### D141 — Resumability via append-only JSONL, and a 429 stops the run rather than backing off

**Decision.** JsonlCache appends + flushes + fsyncs per batch. Raw responses live in .llm_cache/ (spent money, never touched by make clean); derived queries live in artifacts/<fingerprint>/ (rebuildable with no API call). SynthRateLimitError breaks the loop and is reported in GenerationRun.stopped_early/stop_reason; other per-batch errors are counted and the run continues.

**Alternatives.** Collect in memory and write once at the end; exponential backoff on 429; abort the whole run on any batch error.

**Why.** On a free tier a 429 usually means the daily quota is gone, so sleeping burns wall-clock without changing the answer. Append-per-batch makes 'the batch came back' and 'the batch is on disk' the same event, so a 429 one batch later cannot discard a call already paid for. A truncated final line (kill -9 mid-write) is skipped on load rather than poisoning the file.

**How it could be tested.** test_generation_resumes_after_rate_limit (fails on call 3, keeps 2 batches, rerun makes exactly 1 more call), test_one_failed_batch_does_not_end_the_run, test_jsonl_cache_skips_a_truncated_final_line.

### D142 — Generation temperature 0.7, judge temperature 0.0

**Decision.** Different temperatures for the two jobs, as module constants with the reasoning inline.

**Alternatives.** 0.0 everywhere for reproducibility; 0.7 everywhere for variety.

**Why.** At temperature 0 the model returns its single most likely query per type, so across a whole corpus of cafes the training set collapses to a handful of generic phrasings and the cross-encoder learns a template. Reproducibility is instead guaranteed by storing every raw response: a rerun replays the exact bytes the dataset was built from. Grading is the opposite case - it is a measurement, so it must be the modal label and must not move between runs.

**How it could be tested.** Not measured. The claim that higher temperature yields more diverse training data is untested here; the two constants and their reasoning are explicit so the alternative can be run and compared.

### D143 — Hard-negative pool is oversampled 3x and capped at cfg.candidates.retrieve

**Decision.** mine_hard_negatives retrieves hard_negatives_per_query * 3 (default 24), capped at the Stage 2 budget, and takes the top 8 after excluding the positive and duplicate ids.

**Alternatives.** Retrieve exactly 8; retrieve very deep (1000) for a bigger pool.

**Why.** Oversampling is a requirement, not a margin: the judge removes some negatives as false, so mining exactly 8 would leave the most confidently-retrieved queries (whose top results are most plausible and most likely judged relevant) with the fewest negatives - exactly backwards. The cap exists because the served reranker only ever sees the top cfg.candidates.retrieve, so mining deeper would train it on documents it will never be shown.

**How it could be tested.** test_mining_respects_the_configured_count (asserts pool_depth > per-query count), test_mining_excludes_the_positive_and_records_its_rank, test_mining_skips_duplicate_ids.

### D144 — A missed positive is recorded, not dropped

**Decision.** TrainingExample.positive_retrieval_rank is None when the retriever never surfaced the source business for its own query; MiningReport.n_positive_missed counts those. The example is still trainable.

**Alternatives.** Drop such examples; treat the absence as a zero rank.

**Why.** It is a direct (if synthetic) read on Stage 2 recall and an early warning that the documents or the retriever are weak - the sort of finding the failure analysis needs. The positive label does not depend on retrieval finding it, so dropping the example would discard a valid training row and hide the signal.

**How it could be tested.** test_mining_records_a_missed_positive.

### D145 — Random negatives are shipped as a measurable control

**Decision.** sample_random_negatives produces the same TrainingExample shape with source='random', rank 0 and score 0.0, seeded per query id.

**Alternatives.** Only implement hard negatives and assert in prose that they are better.

**Why.** 'Hard negatives beat random ones' is the interview question this project exists to answer with a number. Training one cross-encoder on each and comparing on the gold set turns the claim into a measurement. Per-query seeding means a query's negatives do not change when the query list is filtered or reordered.

**How it could be tested.** test_random_negatives_are_the_measurable_control (determinism under reordering, positive excluded, source tagged). The hard-vs-random NDCG difference is UNMEASURED - it needs the reranker trainer and the gold labels.

### D146 — False-negative filter removes only judge grade 2 by default, and only the top 3 hardest negatives are graded

**Decision.** DEFAULT_FALSE_NEGATIVE_GRADE = RELEVANCE_HIGH (2); DEFAULT_JUDGE_TOP_N = 3; ungraded negatives are kept. Both are module constants, exposed as parameters.

**Alternatives.** Remove grade >= 1 as well; grade all 8 negatives per query; drop every ungraded negative.

**Why.** A grade-1 negative ('right kind of place, misses part of the request') is the near-boundary contrast a reranker has to learn and is still genuinely worse than a source business satisfying the whole request - removing those would strip out most of what makes a hard negative useful. Only a grade-2 negative is an outright label error. Grading the top 3 is where a finite free-tier budget buys the most, because false negatives concentrate at the top of the retrieval list; keeping the ungraded ones is the conservative direction (some label noise beats discarding most of the training signal on no evidence).

**How it could be tested.** test_false_negative_filter_removes_relevant_negatives, test_false_negative_filter_keeps_partially_relevant_negatives, test_false_negative_filter_only_grades_the_hardest_negatives (report carries n_graded vs n_negatives_in so the limitation is visible, not silent).

### D147 — The judge is structurally incapable of writing gold labels

**Decision.** Three independent mechanisms: (1) concierge/synth/judge.py contains no file-write call at all - every write in the package goes through generate.JsonlCache; (2) labels arrive as the frozen GoldLabels value whose as_qrels() returns MappingProxyType, with no save/put and no path parameter; (3) GoldLabels records the file's SHA-256 at load and validate_label_quality re-hashes before returning, raising QrelsMutatedError on any change.

**Alternatives.** A docstring rule; a read-only file permission; passing a path and trusting the code.

**Why.** A judge that could write labels makes the evaluation circular - the system graded against grades it produced. A docstring will not survive six months of edits, so the rule has to be a property of the code that a test can break on purpose.

**How it could be tested.** test_judge_module_contains_no_file_writer (AST walk for open()/write*/to_csv calls - checked on the syntax tree, not by grepping text, so docstring prose cannot trip it), test_judge_exposes_no_way_to_write_labels, test_gold_labels_as_qrels_is_not_mutable, test_judge_never_writes_qrels (file bytes identical after a full validation run), test_qrels_tripwire_fires_if_the_file_changes.

### D148 — Label quality reports both quadratic-weighted and unweighted kappa, and excludes ungraded pairs

**Decision.** validate_label_quality returns both KappaResults plus the 3x3 confusion matrix (categories pinned to 0/1/2), the per-rater grade counts, and n_ungraded. Pairs the judge failed to grade are excluded from kappa, never defaulted to a grade.

**Alternatives.** Report quadratic kappa alone (the usual choice for ordinal labels); default ungraded pairs to 0; let the matrix shape follow the grades used.

**Why.** Quadratic weighting is the flattering number on a 3-point scale (it forgives 1-vs-2 near misses), so quoting it alone without saying it is weighted would overstate agreement. Defaulting an API failure to a grade would invent agreement or disagreement out of a network error. Pinning the categories keeps the matrix comparable between runs instead of silently changing dimensions.

**How it could be tested.** test_label_quality_reports_kappa_and_a_confusion_matrix (exact agreement gives kappa 1.0, matrix is 3x3), test_label_quality_reports_both_weightings, test_label_quality_excludes_ungraded_pairs_rather_than_defaulting_them, test_label_quality_sample_is_deterministic. The actual kappa on the owner's labels is UNMEASURED - it needs eval/qrels.csv.

### D149 — Splits assigned by hashing (seed, group_key), not by a seeded shuffle

**Decision.** bucket(group_key, seed) = sha256 -> [0,1), bucketed into the four fraction bands in fixed SPLIT_NAMES order. A query's split depends only on its id and the seed.

**Alternatives.** random.Random(seed).shuffle then slice by fraction (exact split sizes).

**Why.** With a shuffle, generating more synthetic data later reshuffles everything, so an existing query can move from the cross-encoder's training split into LambdaMART's - silent leakage introduced by a bookkeeping choice, with nothing downstream able to notice. Hash assignment is stable as the dataset grows *for a fixed group key* - which is the whole guarantee when the key is the query id, and less than that under `--group-by cluster`, where the key is a property of the near-duplicate cluster and can therefore move when new queries arrive (see D151). sha256 rather than hash() because hash() on a string is salted per process, which would make splits differ between runs. The cost: split sizes are only approximately the configured fractions, so the manifest reports realised counts rather than nominal ones.

**How it could be tested.** test_split_assignment_is_deterministic_under_a_seed, test_split_assignment_is_stable_as_queries_are_added, test_split_changes_with_the_seed, test_split_fractions_are_approximately_the_configured_ones (within 5 points at n=2000).

### D150 — Split manifest is checksummed, config-fingerprinted, and verified by the function the trainers call

**Decision.** SplitManifest carries a sha256 over (assignment, seed, fractions, group_by). load_manifest refuses a mismatch; verify_splits checks disjointness, coverage, out-of-fold, checksum, and (given a config) that the seed and fractions still match; require_disjoint_splits(cfg) is the single entry point for concierge.rerank.train and concierge.ltr.train. assert_no_overlap takes two id collections so the failure path is testable, and assert_models_trained_on_disjoint_queries checks the ids a trainer actually used against the manifest.

**Alternatives.** Trust the manifest file; assert disjointness inside each trainer; verify the manifest only.

**Why.** cfg.ltr.enforce_oof_crossencoder_feature is a promise; this is the check that makes it real. A single manifest cannot represent one query in two splits (the assignment is a dict), so the realistic leak is a stale manifest - cross-encoder trained from one split, LambdaMART from a regenerated one. assert_models_trained_on_disjoint_queries is the check that catches that, and taking id collections is the only shape a test can break deliberately.

**How it could be tested.** test_splits_are_disjoint, test_splits_are_out_of_fold_for_the_crossencoder_feature, test_out_of_fold_check_fails_when_it_should, test_trained_ids_are_checked_against_the_manifest, test_manifest_checksum_catches_a_hand_edit, test_verify_rejects_a_manifest_from_another_seed, test_require_disjoint_splits_is_what_the_trainers_call.

### D151 — Grouping unit is a parameter; 'query' is the default and its residual risk is stated

**Decision.** GroupBy = 'query' (the project's original specification), 'business' (all of one document's queries move together) or 'cluster' (all members of a near-duplicate cluster move together). `make synth-splits` runs the CLI, whose `--group-by` defaults to `cluster`.

**Alternatives.** Hard-code query-level grouping; hard-code business-level grouping.

**Why.** Query-level grouping leaves a smaller, document-level leak: the cross-encoder fine-tuned on a query written from business X has seen X's document, so its score for X during LambdaMART training is mildly optimistic. That is real, unmeasured, and one argument away from being eliminated - so the option exists and the risk is documented rather than hidden. `cluster` closes the paraphrase version of the same hole (two near-identical queries landing either side of the split), which is why it is what actually runs.

**The trade `cluster` makes, stated because an interviewer will ask.** A cluster's key is a property of the cluster, not of one query, so adding synthetic data can change it: a new member with a lower id moves the key, and linking two clusters merges them. Queries can therefore migrate between folds when the dataset grows - the stability D149 buys by hashing is a guarantee about a *fixed* query set, not about growth. That is paraphrase safety bought at the price of split stability. It is not silent: `concierge.ltr.train` compares the cross-encoder's recorded split-manifest checksum against the current manifest and raises, so regenerating synthetic data after fine-tuning costs a forced cross-encoder retrain rather than a quiet leak.

**How it could be tested.** test_grouping_by_business_keeps_a_businesss_queries_together. The size of the document-level leak is UNMEASURED, as is how often real regeneration moves a query between folds - countable by diffing two manifests built before and after adding queries.

### D152 — Hallucinated business ids and unknown pair ids are dropped, never remapped by position

**Decision.** parse_generation_response rejects any business_id not in the batch; parse_judge_response ignores any pair_id it did not ask about. Both count rejections rather than guessing.

**Alternatives.** Match replies to requests by array position when the id does not match.

**Why.** A query attached to the wrong positive trains the cross-encoder to score an unrelated business as the right answer - a corrupt label is strictly worse than a missing one, and it is invisible afterwards. 'It was probably the second one' is a guess, and a guess here corrupts labels.

**How it could be tested.** test_parse_generation_response_rejects_unknown_business_id, test_parse_generation_response_rejects_unknown_query_type, test_judge_rejects_a_verdict_for_a_pair_it_did_not_ask_about, test_judge_rejects_an_out_of_scale_grade.

### D153 — Prompt examples are off-domain, versioned by content hash, and the header is stripped before sending

**Decision.** Both prompts follow the house convention from concierge/query/prompts/parse_query.txt: a '#' header with the leakage rule (stripped at render time), few-shot examples only from categories the gold set cannot cover (hardware store, laundromat, dog groomer), the query-type vocabulary and the judge rubric injected from code, and prompt_version_for hashing body + injected vocabulary.

**Alternatives.** In-domain few-shot examples (better phrasing); rubric written in the prompt file.

**Why.** A gold query or paraphrase used as an example would be copied into the training set, and no metric could reveal it afterwards. The judge rubric is injected because it must be the same wording the owner labels by - otherwise a low kappa measures prompt drift rather than judge quality. Hashing the injected vocabulary into the version means editing the enum descriptions invalidates the response cache, which is the same silent-corruption guard the version exists for. The cost of off-domain examples (possibly blander generated queries) is UNMEASURED and accepted.

**How it could be tested.** test_generation_prompt_renders_every_placeholder, test_generation_prompt_strips_the_repo_facing_header, test_judge_prompt_renders_rubric_and_pairs, test_prompt_contains_no_gold_queries (asserts no gold query text appears in the prompt file; skips cleanly while eval/gold_queries.csv does not exist).

## Evaluation — metrics and statistics

### D154 — Exponential gain (2**rel - 1) rather than linear gain

**Decision.** DCG uses gain(g) = 2**g - 1, so grades 0/1/2 are worth 0/1/3. Discount is log2(rank + 1) with 1-based ranks.

**Alternatives.** Linear gain (gain = g), which makes a highly relevant result worth exactly twice a partially relevant one instead of three times.

**Why.** Exponential gain is the formulation in the LambdaMART line of work and the default in most IR toolkits, so our NDCG is comparable to published numbers. It also matches the task: a place satisfying all of 'quiet, wifi, open past 10pm' is worth far more than two places satisfying one each. The cost is that NDCG becomes more sensitive to a single mislabeled 2, which is part of why label agreement is measured.

**How it could be tested.** test_gain_is_exponential_not_linear asserts gain(2)/gain(1) == 3.0 (would be 2.0 under linear gain, so a silent switch fails); test_discount_is_log2_of_rank_plus_one pins discount(1)=1, discount(3)=2, discount(7)=3; test_dcg_at_k_hand_computed pins DCG([2,0,1])@3 = 3/1 + 0/log2(3) + 1/2 = 3.5.

### D155 — NDCG returns None (not 0.0) for a query with no relevant documents, and the mean excludes those queries

**Decision.** ndcg_at_k, recall_at_k and reciprocal_rank return None when the qrels row contains no document at/above the relevance threshold. aggregate() averages only defined scores and carries n_defined / n_undefined in the Aggregate result so a mean can never be quoted without its denominator.

**Alternatives.** Return 0.0 for such queries (common in quick eval scripts); or drop them silently without reporting the count.

**Why.** Ideal DCG is 0 there, so NDCG is literally 0/0 — undefined, not zero. Scoring 0.0 would mean a system that correctly found nothing good is scored identically to one that returned garbage, and would drag every reported mean down by a constant that says nothing about ranking quality. Dropping them silently is dishonest in the other direction: the reader needs to know the mean is over the queries that had a relevant document rather than over the whole gold set, and how many that was.

**How it could be tested.** test_ndcg_is_none_when_no_document_is_relevant and test_ndcg_is_none_for_an_entirely_unjudged_query; the contrast case test_ndcg_is_zero_not_none_when_the_run_finds_nothing_relevant (relevant docs exist but none retrieved -> a real 0.0); test_aggregate_excludes_undefined_queries_from_the_mean asserts mean 0.75 for [0.5, None, 1.0] and explicitly asserts it is NOT the naive 0.5.

### D156 — Ideal DCG is built from the full qrels row, never from what the system retrieved

**Decision.** ideal_dcg_at_k sorts all grades in the query's qrels row descending and takes the top k.

**Alternatives.** Normalize by the best ordering of only the documents the system returned.

**Why.** Per-system denominators would let a system that missed every good result still score 1.0 by ordering its own three mediocre hits perfectly. One shared ceiling per query is what makes the ablation rows comparable.

**How it could be tested.** test_ideal_dcg_is_independent_of_what_the_system_retrieved; test_ndcg_penalizes_a_short_run_that_misses_a_relevant_doc (run = [a] with grade 2, qrels also hold b at grade 1, so NDCG@10 = 3 / (3 + 1/log2(3)) = 0.82624, not 1.0).

### D157 — MRR counts grade >= 1 as relevant by default, and the threshold is a parameter

**Decision.** reciprocal_rank takes relevance_threshold, defaulting to 1 (partially relevant counts). 0.0 means 'we looked and found nothing relevant'; None means 'this query has no relevant documents at all'. An optional k caps the search depth. The threshold is stored in RunReport so a table cannot mislabel it.

**Alternatives.** Hard-code rel >= 1; or hard-code rel == 2 ('how fast do we surface a great result').

**Why.** For a concierge query a partially relevant place is genuinely useful to the user, so >= 1 is the better default. But the strict variant answers a different and also interesting question, and neither is 'the' MRR — so the choice must be visible in the output rather than buried in the code.

**How it could be tested.** test_reciprocal_rank_at_threshold_one (first relevant is b at rank 2 -> 0.5) and test_reciprocal_rank_at_threshold_two (same run, b no longer counts, a at rank 3 -> 1/3); plus the 0.0-vs-None pair of tests and test_reciprocal_rank_depth_cap.

### D158 — Unjudged results score 0 but are counted and reported as a first-class number

**Decision.** report_unjudged(run, qrels, k) returns the overall fraction of top-k results with no qrel entry, the raw counts, and a per-query breakdown. evaluate_run computes it at the NDCG cutoff and puts it in RunReport next to the metrics. A graded 0 counts as judged, not unjudged.

**Alternatives.** Score unjudged as 0 silently (the usual TREC assumption, unreported); or exclude unjudged documents from the ranking before scoring (condensed-list evaluation).

**Why.** Pooling labels only what the compared systems returned, so a newly added system is the only one whose freshly surfaced documents are unlabeled — those score 0 and the new system looks worse than it is. The fraction is the size of that bias; if it is high, the comparison is not yet fair and more labeling is needed, a conclusion that is invisible unless the number is printed. Condensed lists were rejected because they change the denominator per system and are harder to defend.

**How it could be tested.** test_report_unjudged_hand_computed ((2 + 2)/(3 + 2) = 0.8 across two queries); test_a_zero_graded_label_is_judged_not_unjudged; test_report_unjudged_handles_an_empty_result_list (0/0 reported as 0.0, no crash); test_report_unjudged_deduplicates_before_counting.

### D159 — Duplicate business ids in a run are deduplicated, keeping the best rank, before the k cutoff

**Decision.** dedupe_preserving_order runs inside every metric, before truncation to k.

**Alternatives.** Score duplicates as-is; or deduplicate after truncating to k; or raise on a duplicate.

**Why.** A buggy fusion step can emit the same business twice; scoring it twice can push NDCG above 1.0, which is nonsense. Deduplicating mirrors what the user sees (one card per place). Doing it before truncation matters: deduping after the cutoff would leave a short list and understate recall. Raising was rejected because the metric code should not be the thing that fails when an upstream stage misbehaves — the invariant NDCG <= 1 is asserted instead.

**How it could be tested.** test_duplicate_ids_are_not_scored_twice (the comment works out that without dedupe the run [a,a,c] scores 5.39/3.63 = 1.485, and asserts both NDCG == 1.0 and NDCG <= 1.0); test_dedupe_runs_before_the_k_cutoff; test_recall_counts_a_duplicate_hit_once.

### D160 — Standard-library-only implementation: no numpy in the metrics module

**Decision.** Bootstrap loops, percentile interpolation and the kappa weight matrix are written in plain Python with random.Random and math.

**Alternatives.** numpy vectorized resampling and np.percentile (numpy is already a project dependency).

**Why.** 2000 resamples (evaluation.bootstrap_samples) over ~150 queries is a few hundred thousand operations, and the suite that exercises them runs in a fraction of a second — so there is no speed argument, and the owner has to read and defend the bootstrap loop line by line in an interview. The percentile interpolation rule is written out explicitly (the same linear-interpolation convention numpy uses) so it is visible rather than delegated. It also keeps the module importable with no scientific stack at all.

**How it could be tested.** The whole file runs with no third-party imports beyond pytest; test_bootstrap_ci_on_two_extreme_scores_spans_the_whole_range pins the percentile behaviour exactly (scores {0.0, 1.0} -> interval [0.0, 1.0], mean 0.5).

### D161 — Bootstrap resamples queries, with the seed taken from config

**Decision.** bootstrap_ci resamples the list of per-query scores with replacement n times and reads the middle `confidence` of the resulting means (percentile bootstrap). Queries are the resampling unit. Seeded from cfg.seed via a thin bootstrap_ci_from_config wrapper; the pure function keeps literal defaults mirroring configs/base.yaml.

**Alternatives.** Resample query-document pairs; a normal-approximation CI from the standard error; BCa bootstrap; passing the Config object into every metric function.

**Why.** Queries are the independent observations — documents within a query are not independent of each other, and resampling them would give an interval far too narrow. The percentile bootstrap makes no normality assumption, which matters because per-query NDCG is bounded in [0,1] and often piles up at the ends. BCa adds bias correction the owner would struggle to defend for a marginal gain. Config is kept out of the metric functions so they can be unit-tested with literals; the wrapper exists so run_eval cannot accidentally report two intervals computed with different resample counts.

**How it could be tested.** test_bootstrap_ci_of_identical_scores_is_a_point (all scores 0.6 -> zero-width interval, proving no noise is injected); test_bootstrap_ci_on_two_extreme_scores_spans_the_whole_range (hand-derived from the three possible resample means 0/0.5/1 at probabilities 0.25/0.5/0.25); test_bootstrap_ci_is_reproducible_for_a_fixed_seed; test_bootstrap_ci_ignores_undefined_queries; test_bootstrap_ci_from_config_uses_config_values asserts the wrapper matches a direct call with the same settings.

### D162 — Paired bootstrap for system comparisons, reported as b - a with a CI and a non-zero-floored p-value

**Decision.** paired_bootstrap draws one index per resample row and uses it for BOTH systems, so each resample compares the same queries. It returns mean_a, mean_b, mean_diff (b - a), the CI of the difference, and a two-sided bootstrap p-value computed as min(1, 2 * (n_wrong_side + 1) / (n + 1)). Query ids are sorted before resampling. Mismatched query sets raise; pairs where either side is None are dropped and counted in n_dropped.

**Alternatives.** Two independent CIs on the two means; a paired t-test; an approximate randomization (permutation) test; silently intersecting the two query sets.

**Why.** The queries are identical across systems, so each query's difficulty affects both equally; resampling pairs lets that shared difficulty cancel, removing query difficulty as a confound. Independent CIs would carry query-difficulty variance in both, be much wider, and overlap even when every single query improved. The t-test assumes normal differences, which bounded per-query NDCG differences are not. The (count+1)/(n+1) convention exists because n resamples cannot distinguish 'very unlikely' from 'impossible' — a literal p = 0 in a README is a claim the method cannot support, so it bottoms out at 2/(n+1) instead. Sorting ids makes the result independent of dict insertion order. Intersecting mismatched sets silently would report a paired test over a query set neither system was evaluated on.

**How it could be tested.** test_paired_bootstrap_with_a_constant_improvement (B beats A by exactly 0.1 everywhere -> every resample difference is exactly 0.1, zero-width CI, and p = 2*(0+1)/(999+1) = 0.002 asserted exactly); test_paired_bootstrap_of_a_system_against_itself_finds_nothing (diff 0, CI [0,0], p capped at 1.0, significant False); test_paired_bootstrap_sign_convention_is_b_minus_a; test_paired_bootstrap_drops_pairs_where_either_side_is_undefined; test_paired_bootstrap_is_insensitive_to_dict_ordering; test_paired_bootstrap_requires_the_same_query_set.

### D163 — Quadratically weighted Cohen's kappa by default for the 0/1/2 label agreement, with the degenerate case returning None

**Decision.** cohens_kappa(labels_a, labels_b, weights='quadratic'|'linear'|'none') computes kappa = 1 - (weighted observed disagreement / weighted expected disagreement) with w_ij = ((i-j)/(C-1))**2, returns the confusion matrix and the weighting scheme alongside the number, and returns kappa=None with an explanatory note when expected disagreement is zero.

**Alternatives.** Unweighted Cohen's kappa; linear weights; Krippendorff's alpha; returning 1.0 or 0.0 in the single-category case; dividing by zero.

**Why.** Grades 0/1/2 are ordinal, not categorical: calling a 2 a 1 is a near miss, calling it a 0 is a real error. Unweighted kappa treats both as equally wrong and understates agreement on exactly this kind of data. Quadratic weights penalize the two-step disagreement four times as much as the one-step one, matching the intuition that 0-vs-2 is the mistake that matters. All three schemes are available because quadratic weighting on a 3-point scale is the more flattering number and must not be quoted without saying it is weighted — the result object carries the scheme so it cannot be. For the degenerate single-category case, chance already explains all the agreement, so there is no evidence of skill: None with a note is the honest answer, and reporting 1.0 would be the most flattering possible lie about label quality. The confusion matrix is returned because kappa hides the shape of the disagreement, and the shape is the useful part (an LLM judge confusing 1 and 2 is usable for training data; one calling 0s highly relevant is not).

**How it could be tested.** test_unweighted_kappa_textbook_2x2 uses the classic matrix [[20,5],[10,15]]: Po = 35/50 = 0.70, Pe = 0.5*0.6 + 0.5*0.4 = 0.50, kappa = 0.20/0.50 = 0.40. test_quadratic_weights_equal_unweighted_on_a_two_category_scale proves the weighted code reduces to the classic formula when C = 2. test_quadratic_kappa_on_ordinal_grades_hand_computed works the full 3x3 weight matrix out in the comment (observed weighted disagreement 0.10, expected 0.30, kappa = 2/3) with the unweighted (4/9) and linear (6/11) variants on the same data asserted in the two tests after it, showing the ordering unweighted < linear < quadratic. Also test_perfect_agreement_is_kappa_one, test_systematic_opposite_labels_give_kappa_minus_one (-1.0), and two degenerate-case tests.

### D164 — Metrics module is pure: no file I/O, no config file reads, no dataset dependency

**Decision.** eval/metrics.py contains only pure functions and frozen result dataclasses. Loading gold_queries.csv and qrels.csv, and all reporting/formatting, belong to run_eval. evaluate_run / subset are provided so the per-query-type breakdown and the bootstrap consume one set of per-query scores rather than recomputing them.

**Alternatives.** A metrics module that loads qrels itself and takes file paths.

**Why.** Purity is what makes every number in this file verifiable against a hand-computed example with no dataset on disk — which is the whole reason the test file can exist before any Yelp data is downloaded. Returning per-query scores (not just means) from evaluate_run also prevents the classic bug where the ablation table and the CI are computed from separately recomputed scores that quietly disagree.

**How it could be tested.** The entire metrics suite runs with no data files and no network; test_evaluate_run_end_to_end_hand_computed checks the three headline metrics, the unjudged fraction and the recorded cutoffs together from one call.

## Evaluation — pooling and labeling CLI

### D165 — Label subset selected by hash ordering, not random.sample

**Decision.** select_label_subset orders queries by sha256(seed|"subset"|query_id) and takes the first `size`. Random, reproducible, and nested: the 50-query subset is a prefix of the 100-query subset.

**Alternatives.** random.Random(seed).sample(...) over the gold list; stratifying by query_type; taking the first 50 rows.

**Why.** Random is non-negotiable because every convenience-based subset biases the headline metric in a specific direction (first-50 = the easy prototypical queries written first; quick-to-judge = excludes the borderline cases where systems actually differ; interesting-looking = inflates whichever stage fixes what the owner suspects). Among random schemes, hash ordering adds one property random.sample lacks: when D004's plan to expand 50 -> 100 happens, every label already collected stays inside the new subset instead of some falling outside a fresh draw. It is also independent of CSV row order, so editing the file cannot reshuffle who is in the sample.

**How it could be tested.** tests/test_pooling.py::TestSubsetSelection (stability, nestedness, order-independence, clamping, duplicate-id rejection) and ::TestSubsetIsUnbiased, which averages 200 fixed seeds and asserts the mean normalized write-order position of the sample is 0.5 +/- 0.03 and each query_type's share is 0.25 +/- 0.03 — a first-50 selector gives 0.08 and fails loudly.

### D166 — Pooled candidates are served shuffled within a query, never best-first

**Decision.** build_pool groups pairs by query (all of one query's candidates together) but orders them within the query by sha256(seed|"pool"|query_id|business_id) — deterministic and unrelated to any system's rank. best_rank and the contributing systems are kept on the pair for statistics but never displayed.

**Alternatives.** Serve in best-rank order (fastest to judge); serve interleaved across queries; shuffle with an unseeded RNG.

**Why.** Grouping by query is a speed and consistency win: judging "cozy spot to read" means holding a standard in your head, and swapping queries every pair means rebuilding it every pair. But ordering *within* a query by rank would let the system under test set the labeler's expectations — a grader shown the best results first anchors high and drifts down, so the label would partly encode the ranking it is supposed to evaluate. A fixed seed rather than an unseeded shuffle because a resumed session must continue along the same path and a pool has to be reproducible months later.

**How it could be tested.** tests/test_pooling.py::TestServingOrder — order is identical across calls with the same seed, differs with a different seed, and is neither the rank order nor its reverse for a 20-candidate run. The CLI loop tests derive expected ids from the pool rather than assuming b001 comes first.

### D167 — Incremental pooling excludes judged pairs; the qrels reader is last-write-wins

**Decision.** build_pool takes the loaded qrels and drops any pair that already has a grade (membership test, so a grade of 0 counts as judged). load_qrels resolves a repeated pair to its last row.

**Alternatives.** Re-serve everything and deduplicate later; keep a separate "done" ledger; make a repeated pair an error.

**Why.** The owner must never be asked to re-grade a pair. That is partly about hours, but mostly about correctness: a second look invites a different answer, and then one pair has two labels and every metric depends on which row won. Membership rather than truthiness avoids the classic falsy-value bug where grade 0 reads as unjudged and every irrelevant pair is re-served forever. Last-write-wins is the read rule for hand-edited files and for a crash between an undo and the next grade — the writer's undo keeps the normal file free of duplicates entirely.

**How it could be tested.** tests/test_pooling.py::TestIncrementalPooling — judged pairs excluded, a 0 counts as judged, judgements do not leak across query ids, and the real scenario (label the BM25 pool, add dense, pool again) asks only about what dense newly surfaced. Verified end to end: a second --stats run on the smoke corpus reported 4 already judged / 1 new.

### D168 — qrels is append-only CSV, fsynced after every single judgement

**Decision.** QrelsWriter appends one row per keystroke and calls flush() + os.fsync() before the next candidate is drawn. Columns: query_id, business_id, relevance, labeled_at (ISO-8601 UTC), labeler.

**Alternatives.** Buffer in memory and write on exit; sqlite; JSON lines; flush without fsync.

**Why.** A labeling session is hours of irreplaceable human work — a crash, a Ctrl-C, or a closed lid must cost at most the pair on screen, and nobody re-labels a session's worth of pairs with the same care. flush() alone hands bytes to the OS; only fsync gets them to disk. The synchronous write is invisible next to the seconds a human spends reading a candidate. CSV so the owner can open and grep the file and see nothing was lost; the timestamp is what makes the ETA and the test-retest gap measurable rather than claimed, and `labeler` leaves room for a second annotator without silently merging the two.

**How it could be tested.** tests/test_pooling.py::TestQrelsWriter — a reader opening the file mid-session sees every grade already entered; header written once across sessions; invalid grades refused at the keystroke and on read; parent directories created.

### D169 — Undo truncates the session's own last row by byte offset

**Decision.** The writer records the file offset before each append; undo() truncates back to it and returns the removed judgement. It can only undo rows this process wrote — on a fresh session undo() returns None.

**Alternatives.** Append a superseding row and let last-write-wins resolve it; rewrite the whole file; no undo at all.

**Why.** A mis-keyed grade is the one error the owner will want to fix instantly, so undo has to exist. A superseding row would work but leaves two rows for one pair, which makes "how many pairs have I judged?" a question you have to think about. Truncation keeps the file clean. Scoping it to the current session is deliberate: an undo that could reach into last week's labels is a footgun, and hand-editing the CSV is always available for a real correction.

**How it could be tested.** tests/test_pooling.py::TestQrelsWriter (undo removes only the last row, undo-then-regrade leaves one clean row, undo cannot reach a previous session) and ::TestLabelSession/::TestRunSession, which check that 'u' reverses a grade in the file and also reverses a skip.

### D170 — Labeling pace and ETA are measured, or reported as unmeasured

**Decision.** measure_pace takes the median gap between consecutive qrels timestamps, dropping gaps over 180s as breaks, and returns None below 3 usable intervals. estimate_minutes returns None when the pace is None, and the pool report prints "unmeasured" with an explanation. The live ETA prefers this session's own median once it has 3 judgements, then falls back to history, then to "—".

**Alternatives.** A hard-coded seconds-per-pair default; the mean instead of the median; no ETA.

**Why.** CLAUDE.md forbids invented numbers, and an ETA is exactly the kind of small number that gets believed: the owner would plan an evening around a pace nobody measured. Median with a break cutoff because the interval distribution has a long right tail of interruptions and one lunch break would double a mean. This session's pace first because pace changes with fatigue and familiarity, and the live number is the one that predicts the next hour.

**How it could be tested.** tests/test_pooling.py::TestEffortEstimates — None below the interval threshold, median arithmetic, a 4000-second break ignored, unparseable timestamps skipped rather than fatal; ::TestPoolStats checks the report says "unmeasured" without a pace and quotes the measured pace with one; ::TestLabelSession checks the progress line shows "ETA —" before a pace exists.

### D171 — --retest writes to a separate file and never receives the previous grades

**Decision.** pooling.build_retest_pool re-serves already-judged pairs for a hash-selected random 30 of the labeled queries, excluding pairs already in the retest file, and writes to eval/qrels_retest.csv. The previous grade is not a parameter of the function and PoolPair has no grade field. --agreement then prints exact agreement, Cohen's kappa both quadratic-weighted and unweighted, the confusion matrix, and the measured median day gap.

**Alternatives.** Re-label in place in qrels.csv; show the old grade for comparison; pick the retest subset by hand.

**Why.** The gold set has one annotator, so the only available second rater is the same person later. Writing into a separate file is essential: re-labeling in place would destroy the baseline the agreement is measured against. Hiding the old grade structurally (rather than by UI discipline) is what makes the number mean consistency instead of willingness to agree with a number on screen. Both kappa variants are printed because quadratic weighting is the more flattering figure on a 3-point ordinal scale and must never be quoted alone. The day gap is reported because self-agreement an hour later measures memory, not consistency.

**How it could be tested.** tests/test_pooling.py::TestRetest (only labeled queries eligible, stable and clamped selection, re-serves exactly the judged pairs, resumes past what was already retested, a pair carries no grade attribute, retest order differs from the labeling order) and ::TestAgreementReport. Verified end to end against a scratch corpus: 4 pairs compared, both kappas and the matrix printed.

### D172 — The labeling screen is built from the parsed Business, not from the BusinessDocument

**Decision.** business_from_raw builds a Business with the project's own parse_attributes / parse_hours, and the screen renders attributes via concierge.data.documents.attribute_phrases plus a full weekly hours table, rating, review count, categories, and up to two review/tip snippets.

**Alternatives.** Show the BusinessDocument text the models actually read; show the raw JSON record.

**Why.** Showing the model's document would hide document bugs: an attribute missing from the document would also be missing from the screen, so the labeler would grade "no data" and the ground truth would absorb the bug. Building from the parsed Business means the labeler sees the dataset's truth as the search system interprets it — if the attribute parser mishandles "u'free'", it is visible here rather than hidden behind a metric. Raw JSON would be unreadable at labeling speed. Field order follows how a human decides: what kind of place, then the facts a query can contradict, then the soft evidence; rating comes late because it is a quality signal, not a relevance signal.

**How it could be tested.** tests/test_pooling.py::TestBusinessDisplay and ::TestHoursDisplay against the fixture quirks (stringified unicode literals, nested dict strings, null attributes/hours), plus ::TestRunSession::test_attributes_and_hours_are_on_the_first_screen, which asserts the rendered screen contains the attributes, the weekly hours, the rating and the snippet. Confirmed visually in a real run of `python -m eval.label_cli` over the fixtures.

### D173 — Hours display separates "closed" from "no hours data", and labels overnight ranges

**Decision.** format_hours prints one row per weekday ("closed" for a day absent from a business that has hours) and a single "— no hours data in the dataset" row when the business has none. format_interval renders "17:00-02:00 (next day)", "17:00-midnight", and "open 24 hours" as distinct cases.

**Alternatives.** Print the raw Yelp strings; omit days with no entry; render both missing cases identically.

**Why.** This mirrors the three-valued ConstraintStatus the ranking code uses. A labeler who cannot tell "shut on Monday" from "the dataset says nothing" will grade the second group as failures and bake a wrong penalty into the ground truth — and hours coverage in this dataset is far from complete. A bare "17:00-02:00" invites the reader to see a five-hour afternoon gap, and the gold set is meant to be full of "open past 10pm" queries (`constraint` is one of its four types), so this is the most misreadable field on the screen.

**How it could be tested.** tests/test_pooling.py::TestHoursDisplay covers the overnight fixture (b002 Monday 17:0-2:0), the midnight-ending interval (b002 Sunday 17:0-0:0), Yelp's 24-hour encoding (b007 0:0-0:0), a missing weekday reading as closed (b008), and the no-hours business (b005).

### D174 — Inputs are raw NDJSON streams plus a runs JSON, so the tool needs no build step

**Decision.** The CLI reads business/review/tip records as newline-delimited JSON in the dataset's own shape (streamed, keeping only pooled ids) and systems' results from {"system": {"query_id": ["business_id", ...]}}. Snippets are ranked by the dataset's own `useful` / `compliment_count` votes.

**Alternatives.** Depend on processed parquet from `make data`; use concierge.data.load (DuckDB) for bulk reads; require a prebuilt pool file.

**Why.** Labeling is Phase 1 step 4-5 work, before a processed corpus necessarily exists, so the tool runs straight off the raw download — and off tests/fixtures, which are byte-identical in shape, which is what makes the whole thing unit-testable with no data. Streaming keeps review.json (~5 GB) workable while retaining only the few hundred pooled businesses. Snippets ranked by upvotes rather than recency or at random because the most-upvoted review is the one other humans found most informative, mirroring the document builder's useful_then_recent rule. The runs file is validated structurally with a message naming the offending key: it is produced elsewhere in the pipeline, and a silent misread would pool the wrong thing and only surface as a strange metric weeks later.

**How it could be tested.** tests/test_pooling.py::TestRunsFile (valid file plus five malformed shapes rejected), ::TestSnippets (most-useful review chosen, word-boundary truncation), ::TestBusinessDisplay. End-to-end run against tests/fixtures/{businesses,reviews,tips}.json produced the expected qrels rows.

### D175 — gold_queries.csv ships 5 EXAMPLE rows that every loader skips by id prefix

**Decision.** The file has a header plus five rows with ids EXAMPLE001-EXAMPLE005 covering all four query_types. load_gold_queries drops any id starting with "EXAMPLE" and reports how many it dropped; duplicate ids, empty queries and unknown query_types are hard errors naming the line.

**Alternatives.** Ship a header only; mark examples with a comment line or an extra column; write the ~150 queries.

**Why.** The ~150 real queries are explicitly owner-owned work (CLAUDE.md section 6) — queries written by a model would make the test set self-graded and destroy the project's credibility, so none were written. Examples still have to exist because the owner needs the format and the type definitions in front of them, and the id-prefix rule makes them structurally incapable of reaching a metric. A comment line would break CSV reading; the prefix is visible, greppable and testable. Duplicate ids are fatal rather than skipped because labels are keyed by id: two queries sharing a row of labels gives a score that belongs to neither.

**How it could be tested.** tests/test_pooling.py::TestGoldQueryFile, including a test over the shipped file itself asserting at most 5 rows, that every row is an EXAMPLE, that loading it yields zero real queries, and that all four query_types are demonstrated. The CLI refuses to start with "contains no real queries" and points at eval/README.md.

### D176 — One Typer command with mode flags; skipping writes nothing

**Decision.** label_cli exposes a single command (so the Makefile's `python -m eval.label_cli --config ...` keeps working with no subcommand), with --stats / --retest / --agreement / --limit / --all-queries as flags, and Typer's Annotated option form. 's' records no row at all.

**Alternatives.** Separate subcommands (label / stats / retest / agreement) with a callback for --config; a grade of 0 for "undecided"; an interactive menu.

**Why.** The Makefile target is owned by another agent and invokes the module with a bare --config; multiple Typer commands would make that call fail with "Missing command". The Annotated form keeps defaults as plain values, so the function stays callable from Python and from a test. Skipping writes nothing so the pair reappears next session: a guessed 0 silently becomes ground truth, while a skipped pair is counted and reported as unjudged — so skipping has to be the cheaper option for the owner.

**How it could be tested.** Verified `uv run python -m eval.label_cli --config <cfg> --stats`, a full piped-key labeling run, --retest, --agreement, and the `concierge-label --help` entry point. tests/test_pooling.py::TestLabelSession and ::TestRunSession cover skip/undo/quit/unknown-key behaviour and that a skip leaves the qrels file untouched.

## Evaluation — ablation runner and reporting

### D177 — The ablation table is not defined in eval/ — it is read from concierge.pipeline.ABLATION

**Decision.** eval/run_eval.py iterates concierge.pipeline.ABLATION and attaches only presentation (row label, rationale) plus each row's comparison baseline (the row above it). ablation_specs() derives the ladder order and the compare_to chain from that tuple; it never restates which rows exist. Supplementary variants (no_filters, dense_only) are read from VARIANTS, flagged supplementary=True, and rendered in a separate table so they can never be read as further rungs.

**Alternatives.** Listing the six systems in run_eval.py with their own stage toggles (simplest), or reimplementing each row's cascade locally so eval controls exactly what runs.

**Why.** The table and the code that produced it must not be able to drift. If eval/ owned its own list, adding a stage to the pipeline would silently leave the README table describing an older architecture, and a locally reimplemented cascade would be measuring run_eval.py rather than the system. A stage toggle through the real pipeline is the only configuration that can honestly be labelled '+cross-encoder (fine-tuned)'.

**How it could be tested.** tests/test_run_eval.py::test_the_ladder_is_the_pipelines_own_ablation_in_its_own_order (keys and order equal ABLATION), ::test_each_rung_changes_exactly_one_thing_from_the_one_above (consecutive rows differ by exactly one Stage flag or by the checkpoint, never both), ::test_every_rung_compares_against_the_row_directly_above_it, ::test_supplementary_rows_are_marked_and_kept_off_the_ladder.

### D178 — An ablation row whose model is missing is refused, not degraded

**Decision.** PipelineFactory checks the loaded pipeline for the artifacts a variant needs (BM25 index, FAISS index, the specific cross-encoder checkpoint, the LambdaMART booster) and raises StageUnavailable rather than letting SearchPipeline's own fallback chain (fine-tuned -> pretrained -> none) run. execute_system records the row as 'not run' with the reason, the rest of the ablation continues, and report.py renders that row as an em dash plus the reason with a `make train-*` hint.

**Alternatives.** Let the pipeline degrade and report whatever it produced; or crash the whole eval run when any stage is missing.

**Why.** Graceful degradation is right for serving and fatal for an ablation: a '+cross-encoder (fine-tuned)' row produced without the fine-tuned checkpoint is a fabricated measurement with a real-looking number. Crashing is also wrong, because `make eval` has to be useful in Phase 2 when three of the six rows do not exist yet — a half table with stated gaps is honest, a traceback is merely unhelpful.

**How it could be tested.** tests/test_run_eval.py::test_a_variant_whose_model_is_missing_is_refused_rather_than_degraded (missing_artifacts against a stand-in loaded pipeline), ::test_a_stage_that_cannot_run_is_a_stated_gap_not_a_zero, ::test_every_system_failing_stops_the_run (NoSystemRan), ::test_a_system_that_raises_on_every_query_is_not_run_rather_than_scored_zero.

### D179 — The hedging sentence is data, built next to the arithmetic, not a rendering choice

**Decision.** compare_systems() returns a Comparison whose `sentence` field already contains the words. When the paired-bootstrap CI includes zero the sentence contains the phrase NOT_DISTINGUISHABLE_PHRASE ('not distinguishable from') plus '...at this sample size: the 95% interval includes zero, so the observed difference is consistent with noise and must not be reported as an improvement.' report.py reproduces that string verbatim in markdown, and it is serialized into results.json.

**Alternatives.** Return only (mean_diff, low, high, p) and let the renderer decide the wording; or expose a boolean `significant` and let the README author phrase it.

**Why.** The requirement is that a CI crossing zero says so in words wherever the number appears. If the wording lived in the renderer, the JSON and any future consumer (the README generator, the API, a notebook) would carry a bare positive difference with no caveat, which is exactly how an inflated claim reaches a resume bullet.

**How it could be tested.** tests/test_run_eval.py::test_a_ci_crossing_zero_is_hedged_in_words, ::test_an_interval_spanning_zero_over_enough_queries_says_it_may_be_noise, ::test_a_uniform_improvement_over_enough_queries_is_called_distinguishable, ::test_the_report_states_n_on_the_tables_and_keeps_the_hedge (the phrase survives rendering).

### D180 — A floor of 6 paired queries before any difference may be called distinguishable

**Decision.** MIN_PAIRS_FOR_A_CLAIM = 6. Below that, Comparison.distinguishable is False and the sentence hedges even when the bootstrap interval excludes zero, naming the sample size as the reason.

**Alternatives.** Trust the percentile bootstrap alone; or pick a round number like 10 or 30 by convention.

**Why.** Arithmetic, not taste. With the most favourable evidence possible — system B better on every one of n paired queries — a two-sided sign test gives 2*(1/2)^n = 1/2^(n-1): n=5 is 0.0625, still above 0.05, and only n=6 reaches 0.031. A percentile bootstrap over five uniformly positive differences returns an interval excluding zero (every resample is positive) because it has no way to express 'there were barely any observations'. The floor matters most in the per-query-type breakdown, where one type can hold a handful of queries and a zero-width interval would otherwise read as certainty.

**How it could be tested.** tests/test_run_eval.py::test_too_few_paired_queries_can_never_be_called_distinguishable, ::test_a_single_query_interval_is_labelled_degenerate_not_narrow.

### D181 — Missing or nearly-empty labels stop the run before any search executes

**Decision.** run_eval() loads the gold set and qrels and calls require_enough_labels() first, before building any pipeline. A missing qrels.csv, a missing gold_queries.csv, or fewer than MIN_LABELED_QUERIES (10) gold queries carrying at least one *relevant* label raises LabelsNotReady; main() prints it and exits 2 (distinct from 1 for an unexpected failure) and writes no files at all. The threshold counts queries with a relevant label, not labeled rows, because a query whose whole pool was graded 0 has an undefined NDCG and contributes nothing to any mean.

**Alternatives.** Run the searches and report whatever is scoreable; emit a clearly-marked example table so the report's shape is visible before labeling; or exit 1 like any other error.

**Why.** An example table is the single worst outcome available to this project — a fabricated number that reaches the README destroys the only thing it is selling — so no code path produces one. Checking before searching matters practically too: discovering after twenty minutes of CPU reranking that nothing could be scored is waste, and 'label more queries' is the actionable message. Exit code 2 is separate because 'not labeled yet' is a normal state of this project, not a crash.

**How it could be tested.** tests/test_run_eval.py::test_missing_qrels_refuses_to_report, ::test_missing_gold_queries_refuses_and_does_not_write_them_for_the_owner, ::test_nearly_empty_labels_refuse_to_report, ::test_a_pool_graded_entirely_zero_does_not_count_as_a_labeled_query, ::test_main_exits_with_a_distinct_code_and_writes_nothing_when_labels_are_missing, ::test_nothing_in_the_report_is_a_placeholder_number; plus a live run of `python -m eval.run_eval --config configs/base.yaml` with no qrels.csv on disk, which printed the owner-action message and exited 2.

### D182 — Ablation rows are measured at a depth of 100; the production operating point is measured in the sweep

**Decision.** eval_depth_for(cfg) returns max(recall_cutoff, pool_depth, ndcg_k, candidates.final) = 100 with the committed config, and the ablation runs every row at that depth. The quality-vs-latency sweep runs at the production final depth (candidates.final = 10) across sweep_retrieve x sweep_rerank, skipping any combination that would widen the cascade. The report states in words that the sweep's latency, not the ablation table's, is the serving cost.

**Alternatives.** Run everything at top-10 and drop Recall@100; or run the sweep at depth 100 too, for one consistent number.

**Why.** Recall@100 cannot be computed from a ten-item list, and asking the production configuration for a hundred results changes the very latency being measured. Separating them keeps both numbers honest: the ablation answers 'what does each stage contribute?', the sweep answers 'what should we actually serve?'.

**How it could be tested.** tests/test_run_eval.py::test_sweep_runs_the_valid_grid_at_the_production_final_depth, ::test_depths_are_applied_as_a_config_copy_not_a_mutation (the caller's frozen config is never mutated, so a result's recorded fingerprint stays truthful), ::test_the_headline_cutoff_comes_from_the_config_not_from_this_code.

### D183 — Latency p95 is a nearest-rank percentile over the caller's own wall clock, with the first query excluded

**Decision.** percentile_nearest_rank() returns the smallest observation with at least q of the samples at or below it — an observed value, not an interpolation (deliberately different from metrics._percentile, which interpolates for bootstrap intervals). Timing is perf_counter around searcher.search(), not SearchResponse.total_ms. The first successful query of each system is recorded separately as cold_start_ms, and fewer than 20 samples attaches a caveat string saying p95 is really the top observation.

**Alternatives.** Interpolated percentiles for consistency with the bootstrap code; reading the pipeline's own reported total_ms; including the first query in the distribution.

**Why.** A latency percentile is an SLO statement — '95% of requests finished within this many ms' — so it must be a value that actually happened. Measuring from outside the pipeline means the number does not depend on the pipeline's own bookkeeping being correct, which is what a caller experiences. The first query pays for lazy index mmap and a cold page cache; folding that multi-second cost into a steady-state distribution would make every row's p95 meaningless.

**How it could be tested.** tests/test_run_eval.py::test_percentile_nearest_rank_is_an_observed_value, ::test_percentile_rejects_a_quantile_outside_the_unit_interval, ::test_the_cold_start_is_separated_from_the_steady_state_samples.

### D184 — The failure analysis separates 'never retrieved' from 'retrieved then mis-ranked', and refuses to invent a fault

**Decision.** diagnose_query() traces every labeled-relevant business through every system's ranking and returns one of: never retrieved (absent from all candidate lists -> Stage 2 or a hard filter), retrieved but never ranked up (in a pool, never in anyone's top k), mis-ranked by a named stage (in the top k under an earlier row, out of it under a later one — the later row's key is returned as lost_stage), or mis-ordered inside the top k. When the worst query of a type has NDCG@10 == 1.0, it returns lost_stage='none' and says the type has no failure to show in the labeled set.

**Alternatives.** List the worst queries with their results and leave the diagnosis to the reader; or always name a culprit stage so every case reads as a finding.

**Why.** That distinction is the entire diagnostic value: a relevant place absent everywhere calls for better retrieval or a looser filter, while one sitting at rank 40 calls for reranker work — opposite fixes. And inventing a fault for a perfectly-ranked query would be exactly as dishonest as inventing a number; 'the worst query of this type' is a statement about the type, not evidence of a defect.

**How it could be tested.** tests/test_run_eval.py::test_a_relevant_result_absent_everywhere_is_diagnosed_as_never_retrieved, ::test_a_result_pushed_out_of_the_top_k_names_the_stage_that_lost_it, ::test_a_result_in_the_pool_but_never_surfaced_is_a_ranking_failure, ::test_a_relevant_result_inside_the_top_k_is_diagnosed_as_mis_ordering, ::test_a_perfectly_ranked_worst_query_is_not_given_an_invented_fault, ::test_failure_cases_show_the_actual_results_with_their_grades, ::test_failure_analysis_skips_queries_with_no_defined_ndcg.

### D185 — Every table carries its own denominators, including per-type labeled-pair counts (added this pass)

**Decision.** LabelCoverage gained labeled_pairs_by_type and positive_pairs_by_type; report.py gained denominator_note() and type_denominator_note(), rendered under the ablation table, the supplementary table, each per-query-type table, the sweep table and the failure section. A partial run's PARTIAL RUN stamp is repeated in every one of those notes.

**Alternatives.** State the counts once in the report header (what the file did before this pass).

**Why.** These tables exist to be pasted into the README individually, and a metric separated from its denominators is not a result. A per-type table especially needed its own counts: a 'vibe NDCG@10 = x' computed over n queries and m labeled pairs is a different claim from the same x computed over a much larger n and m. Repeating the partial-run stamp means a smoke-test table cannot be lifted out of the document and read as a measurement.

**How it could be tested.** tests/test_run_eval.py::test_each_query_type_carries_its_own_label_counts (per-type counts hand-counted from the synthetic qrels, and asserted to sum to the overall counts so the breakdown cannot drift from the headline), ::test_every_table_carries_its_denominators_on_its_own, ::test_a_partial_run_stamp_rides_along_with_every_table.

### D186 — A missing matplotlib costs the figure and nothing else

**Decision.** report.py imports matplotlib only inside plot_latency_curve(), with Agg set before pyplot, and on ImportError returns a note saying matplotlib is not installed and pointing at latency_curve.csv. The sweep's data is always written as CSV, and the error bars are drawn asymmetric (mean-low, high-mean) because a percentile bootstrap interval is not required to be symmetric.

**Alternatives.** Import matplotlib at module scope and let `make eval` fail without it; or draw symmetric +/- error bars.

**Why.** The numbers are the deliverable and the figure is a convenience — losing a whole eval run to a plotting dependency would be absurd, and matplotlib is currently not in pyproject (see blocked_on). Symmetric bars would misreport the interval.

**How it could be tested.** tests/test_run_eval.py::test_a_missing_matplotlib_costs_the_figure_and_nothing_else (monkeypatched import failure: plot is None, the markdown says so, the CSV still holds the numbers). Also confirmed live: matplotlib is genuinely absent from .venv, and a full six-row smoke render printed the stated note instead of a figure.

### D187 — qrels.csv is parsed by the labeling CLI's own module when the file carries its header

**Decision.** load_qrels() reads the header first. If it contains the labeling CLI's full column set (query_id, business_id, relevance, labeled_at, labeler) it delegates to eval.pooling.load_qrels; otherwise it reads a minimal query_id,business_id,grade file itself (accepting grade/relevance/label/rel as the grade column). Last row wins for a repeated pair, matching the CLI's append-only file. Grades outside {0,1,2} and non-integer grades are rejected with the file and line number.

**Alternatives.** One parser in run_eval for both shapes; or require the CLI format exclusively.

**Why.** Two independent interpretations of one file format is the seam where two disagreeing numbers come from, so the writer's own module owns its format. A hand-assembled minimal file is still valid input (tests, a quick correction) and should not require the CLI's bookkeeping columns. A stray grade of 3 would change every DCG gain with no visible error, hence rejection rather than clamping.

**How it could be tested.** tests/test_run_eval.py::test_a_file_written_by_the_labeling_cli_is_read_by_its_own_module, ::test_a_repeated_pair_takes_the_last_row_as_the_current_judgement, ::test_a_grade_outside_the_labeling_scheme_is_rejected, ::test_a_non_integer_grade_is_rejected_with_its_line_number, ::test_a_relevance_column_alias_is_accepted, ::test_an_unknown_query_type_is_rejected_with_its_line_number, ::test_duplicate_query_ids_are_rejected, ::test_labels_for_queries_outside_the_gold_set_are_ignored_but_reported.

## Pipeline and data build

### D188 — Stage toggles are a Flag enum inside a frozen Variant, and the ablation is data

**Decision.** `Stage` is an `enum.Flag` (QUERY_LLM / BM25 / DENSE / FILTERS / RERANK / LTR). A frozen `Variant(name, stages, reranker)` is one system configuration, and `ABLATION` is a 6-tuple of Variants defined once in concierge/pipeline.py; `VARIANTS` is a name->Variant registry that also holds non-table diagnostics (`no_filters`, `dense_only`). `eval/run_eval.py` should iterate `ABLATION` and never construct its own row list.

**Alternatives.** Six boolean keyword arguments on search() (the obvious version); a config section per ablation row; a subclass per variant.

**Why.** Booleans cannot be stored, compared, printed or composed, so each result row would have to re-describe its own configuration and would eventually disagree with the code that produced it. A Flag makes "full system minus hard filters" an expression (`_FULL & ~Stage.FILTERS`) rather than a new code path, and a frozen Variant is hashable and printable so a result can carry the exact configuration that produced it.

**How it could be tested.** test_ablation_rows_add_exactly_one_thing_at_a_time asserts each row is the previous row plus exactly one change (one new stage, or the same stages with a different checkpoint) and that no row removes a stage; test_ablation_covers_the_six_configurations_the_readme_reports pins the row names and the full stage set.

### D189 — Which cross-encoder checkpoint is an enum field, not a flag

**Decision.** `Variant.reranker: RerankCheckpoint` (PRETRAINED | FINETUNED), read only when `Stage.RERANK` is enabled.

**Alternatives.** Two separate flags (RERANK_PRETRAINED, RERANK_FINETUNED).

**Why.** Pretrained and fine-tuned are two values of one choice: exactly one set of weights runs. As two flags, "both on" would be a meaningless state the type system actively invited, and an ablation row could silently be in it.

**How it could be tested.** test_missing_finetuned_checkpoint_falls_back_to_pretrained; the cumulative-ablation test treats a checkpoint change as a change only once RERANK is on.

### D190 — Hard filters stay ON in every ablation row

**Decision.** All six rows enable `Stage.FILTERS`. Row 5 ("+LLM query parsing") therefore measures LLM parsing against the rule parser, not against having no constraint handling. Filters-vs-no-filters is `VARIANTS["no_filters"]`, a separate experiment.

**Alternatives.** Turn filters on only in the query-parsing row, so that row absorbs both the parser and the constraint checking.

**Why.** If the LLM row added two things at once (a parser and constraint enforcement), the gap between row 4 and row 5 could not be attributed to either. Rows 1-4 already apply whatever constraints the rule parser finds, which is also the honest baseline: the rule parser is a real system, not a null.

**How it could be tested.** test_ablation_covers_the_six_configurations_the_readme_reports; test_violated_constraints_remove_candidates shows constraints are enforced in the pre-LLM rows.

### D191 — A disabled stage is a pass-through that neither reorders nor truncates

**Decision.** With RERANK or LTR off, candidates flow through unchanged, and `n_candidates` records the carried-forward count. Only the final assembly truncates (to `cfg.candidates.final`). Timings are still recorded for the skipped stage, so every search reports the same stage list.

**Alternatives.** Truncate at every stage width regardless of whether the stage ran.

**Why.** The top-10 would be identical either way, but the candidate counts would be fiction, and those counts are part of the latency analysis ("what did Stage 3 actually score?"). Non-increasing counts still hold, so the cascade invariant is testable.

**How it could be tested.** test_candidate_counts_narrow_down_the_cascade; test_bm25_only_variant_calls_neither_dense_nor_the_reranker; test_timings_are_recorded_for_every_stage.

### D192 — Stage 3 scores everything Stage 2 retrieved, then keeps cfg.candidates.rerank

**Decision.** The whole post-filter candidate list is handed to the reranker with `top_k=cfg.candidates.rerank` (200 scored -> 50 kept), rather than scoring only the first 50.

**Alternatives.** Shortlist to 50 first and rerank within it (cheaper per query).

**Why.** Scoring only the head makes the reranker unable to rescue a candidate that retrieval ranked 180th, which is the main thing a reranker is for. It is also what CLAUDE.md's architecture says, and the per-stage latency it costs is measured rather than avoided.

**How it could be tested.** test_reranker_runs_once_on_the_shortlist_and_reorders_it asserts the number of scored pairs equals the post-filter candidate count and that the output is capped at the rerank width.

### D193 — Stage 3's ordering is delegated to concierge.rerank.model.CrossEncoderReranker, not reimplemented

**Decision.** The pipeline declares a `Reranker` protocol with `rerank(parsed, candidates, documents, top_k)` and calls the real class. Scoring, the stable sort (ties keep RRF order), `SCORE_CROSS_ENCODER`/`RANK_CROSS_ENCODER` bookkeeping and top_k all live in the rerank module. The pipeline's own first draft had a sort with a `business_id` tie-break; it was deleted. `query_text_for` is likewise imported rather than restated, so retrieval and reranking score the same string.

**Alternatives.** A narrow `score(query, documents)` protocol with the sorting done in the pipeline (simpler to fake in tests).

**Why.** Two implementations of the same stage ordering would drift (they already disagreed on the tie-break), and the eval would be measuring whichever one it happened to call — exactly the failure DECISIONS D012 warns about.

**How it could be tested.** Tests inject a fake `PairScorer` into the real `CrossEncoderReranker`, so the production ordering contract runs while the spy still proves per call whether a forward pass happened.

### D194 — The LambdaMART serving adapter lives in pipeline.py and asserts the saved feature order

**Decision.** `LightGbmRanker` loads `model.txt` + `feature_names.json` from `concierge.ltr.train.default_out_dir(cfg, False)` and scores with `concierge.ltr.features.feature_matrix` in the model's own column order. `check_feature_order` is a module-level function that fails if the booster's names and the JSON disagree. There is no inference entry point in `concierge/ltr` yet, so this is wiring, not modelling: every feature value still comes from the ltr module.

**Alternatives.** Wait for `concierge.ltr` to expose a loader and keep Stage 4 permanently degraded; or build feature rows in the pipeline.

**Why.** Building feature rows here would create train/serve skew, which is invisible in the output — it just ranks worse. Loading a file and checking a column order is not a modelling decision, and permuted columns produce plausible predictions with no error at all, which makes the assertion worth more than the code it costs.

**How it could be tested.** test_ltr_adapter_builds_features_in_the_saved_column_order checks the matrix handed to the booster (the star column really is the star rating); test_permuted_feature_order_is_refused; test_untrained_lambdamart_says_which_command_to_run.

### D195 — Degradation ladder, warned exactly once per process

**Decision.** Missing fine-tuned checkpoint -> pretrained (warning says results are NOT the fine-tuned system); no cross-encoder at all -> skip Stage 3; no LambdaMART -> keep Stage 3 order; no dense index -> lexical-only (`from_config` rebuilds a missing BM25 index, but never silently re-embeds a missing FAISS index). Warnings are deduplicated by key on the instance.

**Alternatives.** Raise on a missing artifact; or warn per request.

**Why.** The pipeline must be runnable at every point in the project's own construction, and a load test would drown in per-request warnings while a silent fallback would make a degraded ablation row look like a real measurement.

**How it could be tested.** test_missing_finetuned_checkpoint_falls_back_to_pretrained, test_missing_reranker_entirely_keeps_retrieval_order, test_missing_ltr_model_keeps_cross_encoder_order, test_from_config_without_a_dense_index_degrades_to_lexical (all assert the warning text via caplog).

### D196 — Latency and filter drop reasons ride on the existing SearchResponse contract

**Decision.** A `StageTiming` per stage plus `total_ms` on every search, never behind a flag. `n_candidates` carries the cascade counts plus `final`, and also the filter stage's drop reasons as `dropped_<reason>` keys and `penalized_missing_data`.

**Alternatives.** A debug flag for timings; adding fields to `SearchResponse` (a frozen shared contract) for the filter report.

**Why.** p50/p95 per stage is a reported result, so a measurement that only happens when someone asks for it will be missing from the README. The drop reasons answer "why did this query return three results?" for the failure analysis, and `FilterReport` already counts them — surfacing them through the existing dict avoided editing a contract another agent owns.

**How it could be tested.** test_timings_are_recorded_for_every_stage (stage list, non-negative, total >= sum); test_permanently_closed_business_never_reaches_the_results asserts `dropped_permanently_closed == 1`; test_zero_surviving_candidates_returns_an_empty_response asserts `dropped_unknown_business > 0`.

### D197 — Unanswerable queries return an empty response, never an exception

**Decision.** A blank query returns after the parse stage with zeroed cascade counts and no retrieval call. A query whose candidates are all dropped returns `results=()` with the counts that explain why. Stage 4 skips (rather than invents a score for) a candidate with no business record.

**Alternatives.** Raise on an empty query; let a missing business record raise a KeyError.

**Why.** A search engine that 500s on a narrow query is worse than one that says nothing matched, and the eval harness runs ~150 queries unattended. A zero score for an unscorable candidate would be an invented measurement that Stage 4's features would then learn from.

**How it could be tested.** test_blank_query_returns_an_empty_response_without_retrieving; test_zero_surviving_candidates_returns_an_empty_response (pipeline built with an empty business map).

### D198 — businesses.parquet stores attributes and hours as JSON strings, written atomically under an explicit schema

**Decision.** `businesses.parquet` has a fixed 12-column schema with `attributes_json` and `hours_json` as UTF-8 JSON (`{day: [[start_min, end_min], ...]}`), `categories` as a list column, and explicit polars dtypes. Writes go to `<file>.tmp` then `replace`. `documents.parquet` matches exactly the columns `concierge.retrieval.build_index.load_documents` requires. `read_businesses`/`read_documents` convert back into the shared dataclasses so a renamed column fails at load, once.

**Alternatives.** Nested parquet structs/maps; one column per attribute; pickling the Business objects; inferring the schema.

**Why.** Attributes are an open-ended heterogeneous mapping whose key set differs per business (a struct would need a column per key) and hours are variable-length per day. Neither column is ever filtered on in SQL — both are read straight back into Python. Inferred schemas change when a column happens to be all-null in one scope and not in the next; an interrupted write leaving a half-written index input fails much later and much less obviously than a stale one.

**How it could be tested.** test_business_parquet_round_trip_is_exact (list equality over all 8 fixtures); test_round_trip_preserves_the_awkward_fixtures (overnight interval still crosses midnight, null attributes/hours stay empty, open-24h still covers 03:00); test_documents_parquet_matches_what_the_index_builder_reads; plus a manual CLI smoke run that produced byte-identical parquet on a second run.

### D199 — The build report contains only counted and measured quantities

**Decision.** `build_report.json` + the terminal tables carry: businesses in scope, documents written, how many raw records had any hours data, how many have any attributes, review/tip rows attached, documents truncated, mean document length, per-attribute coverage, and wall-clock per phase. `--limit` is recorded in the report and printed as a warning so a sampled build can never be mistaken for a full one. No totals are reported for data this build never reads (e.g. the whole raw file), and `_pct` prints "n/a" rather than 0% when there is nothing to divide by.

**Alternatives.** Also report throughput or projected index size; estimate out-of-scope counts.

**Why.** Project rule: never a placeholder number. `make explore` owns the whole-file numbers; this command owns what it actually built.

**How it could be tested.** Manual run over the fixture corpus (6 businesses in scope: 8 minus one closed and one out-of-metro, 5 with hours data); test_build_cli_without_a_metro_exits_with_an_actionable_error and test_build_cli_without_raw_data_prints_the_download_instructions cover the two refusal paths (exit code 2, no traceback).

### D200 — Tests use a crc32 hashing encoder and the real BM25/FAISS indexes

**Decision.** Only the models are faked: a deterministic hashed-bag-of-words `TextEncoder` (crc32, not `hash()`, which Python salts per process) and a fake `PairScorer`. The real `bm25s` index, the real FAISS `IndexFlatIP`, the real fusion, the real filters and the real reranker all run.

**Alternatives.** Fake the retrievers too (faster, simpler).

**Why.** A fake retriever would let the fusion, filter and rank-bookkeeping plumbing stay unverified while the tests passed. Salted hashing would make rankings differ between runs, which would make a flaky test look like a real behaviour change.

**How it could be tested.** The whole file passes with one skip, in a couple of seconds, with no model download and no Yelp data.

## Serving — FastAPI

### D201 — Pipeline injected through a create_app() factory

**Decision.** serve/app.py exposes create_app(cfg=, variant=, pipeline_factory=, cors_origins=) and a module-level `app = create_app()` for uvicorn. The pipeline is built inside an asynccontextmanager lifespan, never at import time.

**Alternatives.** (a) module-level pipeline built at import, (b) FastAPI dependency_overrides in tests, (c) a global settable singleton. (a) makes the module unimportable without Yelp data and indexes, so nothing about the API could be tested; (c) leaks state between the several apps the test file builds in one process.

**Why.** The API's own behaviour (status codes, response shape, degradation reporting) has to be testable separately from the models, or it is only tested by hand. The factory is also what lets a load failure be *recorded* instead of killing the process, which is what keeps /health able to explain itself.

**How it could be tested.** tests/test_serve.py runs the real app via TestClient against a FakePipeline for every case; test_health_degraded_returns_200_but_unavailable_returns_503 covers the recorded-failure path. Also smoke-tested with the real default factory in this repo (no processed data): /health -> 503 unavailable, /search -> 503 whose detail names `make data`.

### D202 — Endpoints are sync `def`, not `async def`

**Decision.** Both handlers are plain `def`, so Starlette runs them in its worker threadpool.

**Alternatives.** `async def` handlers (and optionally wrapping the search in run_in_threadpool or an explicit executor by hand).

**Why.** A search is CPU-bound and never awaits: a cross-encoder forward pass over ~200 documents would block the event loop and every other in-flight request, so the load test's concurrency levels would measure a queue rather than the service. The honest framing is that this service is thread-pool-parallel, not async-parallel, because nothing in the cascade is I/O-bound except the optional Stage 1 LLM call.

**How it could be tested.** Not unit-tested: it is a concurrency property, not a value. The
measurement that would settle it is `serve/load_test.py` against a live uvicorn, checking
whether throughput rises with concurrency instead of staying flat — flat throughput is the
signature of a blocked event loop. **That measurement has not been made on a representative
workload**, and no claim about it belongs here until it has.

What HAS been run is the load test against the fixture corpus
(`.audit/verification-run.md`): 360 requests over concurrency 1/4/16, zero errors, with
server-reported latency flat at p50 0.39-0.40 ms while client-observed p95 rose to 84 ms.
That run says nothing about this decision in either direction — the fixture corpus is eight
businesses and the per-request work is microseconds, so there is no CPU-bound stage to block
anything with. It is evidence about the single-process asyncio *client*, which the load test's
own note predicts. Run it against a real index and a loaded cross-encoder to get evidence
about the server.

### D203 — /health reports per-artifact load status, with `required` judged against the served variant

**Decision.** A free function inspect_artifacts(pipeline, variant) reads the loaded pipeline's attributes and returns one ArtifactStatus per artifact (businesses, documents, bm25_index, dense_index, cross_encoder_pretrained, cross_encoder_finetuned, ltr_model), each with loaded, required and an actionable detail naming the make target. status is ok / degraded / unavailable; warnings are derived from required-and-missing artifacts and are repeated on every /search response. I added a `required: bool` field to ArtifactStatus in serve/schemas.py (my own file) for this.

**Alternatives.** (a) a boolean {"status":"ok"}; (b) checking the filesystem for artifact paths; (c) treating every missing artifact as degraded regardless of variant.

**Why.** The pipeline degrades rather than crashes, which is correct for development and dangerous for a demo: a deployment serving retrieval order while the README claims a fine-tuned reranker looks exactly like success. (b) is untrustworthy — a file that failed to deserialize passes a path check and still serves nobody, so the only real answer is whether the object is in memory. (c) would cry wolf on every ablation-row deployment, where Stage 4 is off by design.

**How it could be tested.** tests/test_serve.py: degraded case asserts the three missing artifacts and that the warnings name `make index`, `make train-reranker`, `make train-ltr` (and that there are exactly three); test_missing_artifact_is_not_required_when_the_variant_skips_that_stage; test_health_distinguishes_the_two_cross_encoder_checkpoints (pretrained-fallback must not read as "reranker loaded"); test_inspect_artifacts_is_total_for_an_unknown_object.

### D204 — Status codes: degraded is 200, unavailable is 503, empty results are 200, LLM parse failure is not an error

**Decision.** /health returns 503 while the pipeline is unloaded and 200 otherwise (including degraded). /search returns 503 with a detail naming `make data`/`make index`/`GET /health` when unloaded, 422 for a malformed body (pydantic, extra="forbid", limit bounded by MAX_LIMIT=50), 200 with zero results for an unanswerable query, and 500 with a short message (traceback to the log only) for a genuine bug. A rules fallback after an LLM failure is a 200 with parsed.source == "rules".

**Alternatives.** A 404 for an empty result set; a 5xx on parse failure; a single 200 /health with the nuance only in the body; echoing the exception text on 500.

**Why.** A container probe keys on the status code and must not restart a service happily serving a BM25-only configuration, while a human needs the nuance in the body. "Nothing is open at 3am with a patio" is a correct answer to a well-formed question, and candidate_counts (with the filter stage's dropped_* reasons) explains where the candidates went — a 404 would send the failure analysis to the logs. Turning the parser's contractual fallback into a 5xx would make the service less reliable than the pipeline it wraps.

**How it could be tested.** Parametrized 422 cases (missing query, empty query, misspelled field, limit 0, limit > MAX_LIMIT, non-integer limit, non-JSON body); test_empty_result_set_is_a_200_not_a_404; test_llm_parse_failure_is_not_an_http_error; test_unexpected_pipeline_failure_is_a_500_without_a_traceback; test_search_before_startup_is_503_not_an_attribute_error.

### D205 — Load test: asyncio + httpx, nearest-rank percentiles, both client and server latency, health snapshot embedded

**Decision.** serve/load_test.py is stdlib + httpx (already a project dependency). Per level: untimed warmup, then a deterministic cycle through the query file dispatched by N worker tasks pulling from a queue. It reports client-observed p50/p95/p99 plus throughput counting only successes, the server's own timings.total_ms, per-stage p50/p95 lifted from each response, status counts, error examples, and the /health snapshot of the system under test. Percentiles are nearest-rank. Output is printed and written to artifacts/load_test/<timestamp>.json plus latest.json. --queries is required, there is no default workload, and a run in which nothing succeeded sets measurement_valid=false and exits 1.

**Alternatives.** locust or k6 (a dependency and a config language to explain for one table); asyncio.gather over all requests (measures the accept backlog, not a fixed concurrency); interpolated percentiles (numpy); a hardcoded default query file.

**Why.** A latency number is only interpretable with its workload and with what was actually loaded — a p95 measured without the fine-tuned reranker belongs to a different system, so the health snapshot is part of the record. Nearest-rank means a reported p95 is a time some request actually took. Counting only successful requests stops a service that 500s instantly from looking fast. With fewer than 100 timed requests the nearest-rank p99 *is* the maximum, so the record says so instead of dressing it up. No latency figure appears anywhere in the file — not in help text, not in a docstring.

**How it could be tested.** Unit tests for percentile (including refusing an empty sample or an out-of-range q), summarize, throughput-counts-successes-only with the small-n caveat, CSV/text query loading by column name, and write_record's timestamped + latest.json outputs. End-to-end: ran it against a live uvicorn serving the real app with a fake pipeline (scratchpad only) at concurrency 1/4/8 — table, per-stage breakdown, degraded-health banner and JSON all produced; then against a dead port, which produced a record with no latency numbers and exit code 1.

### D206 — Docker: multi-stage uv build, non-root, artifacts and data mounted rather than baked

**Decision.** Builder stage ghcr.io/astral-sh/uv:python3.12-bookworm-slim runs `uv sync --frozen --no-dev --no-install-project` into /app/.venv; runtime stage python:3.12-slim-bookworm copies that venv plus concierge/, serve/, configs/, runs as uid 10001, exposes 8000, HEALTHCHECKs /health with urllib, and CMDs uvicorn in exec form. data/processed and artifacts/ are mount points; the run command is documented in the file header.

**Alternatives.** A single-stage image; baking the processed parquet and model artifacts into layers; pip + requirements.txt; running as root.

**Why.** The processed files are derived from the Yelp Open Dataset, which is licensed for personal/educational use — baking them into a pushable image would be redistribution, and that reason alone settles it. They are also large and fully regenerable from make targets keyed by the config fingerprint. Mounting also makes an unmounted run fail *visibly*: the service still starts and /health names what is missing. --frozen means the served dependency set is the locked one; --no-dev keeps pytest/ruff/mypy out. OMP_NUM_THREADS is deliberately left unset because it materially changes CPU inference latency and so is a knob of the experiment, not a value to freeze in an image.

**How it could be tested.** NOT BUILT — the Docker daemon is not running on this machine, so the image build is unverified. No image size is quoted anywhere because none was measured.

### D207 — CORS allow-list rather than a wildcard

**Decision.** Default origins are http://localhost:3000 and http://127.0.0.1:3000, overridable via CONCIERGE_CORS_ORIGINS or the cors_origins argument; methods GET/POST, header Content-Type, allow_credentials=False.

**Alternatives.** allow_origins=["*"].

**Why.** A wildcard cannot be combined with credentials, hides who the client is, and this API has exactly one client. Both spellings are listed because a browser treats localhost and 127.0.0.1 as different origins, and which one the owner gets depends on what they typed in the address bar.

**How it could be tested.** Parametrized preflight test for both origins asserting access-control-allow-origin echoes the request origin; test_cors_rejects_an_unlisted_origin asserts the header is absent for an unlisted origin.

### D208 — Fake-vs-real drift guard in the tests

**Decision.** test_health_reads_attributes_the_real_pipeline_actually_has constructs a real SearchPipeline (cheap: empty businesses/documents, no models) and asserts both it and FakePipeline expose every attribute inspect_artifacts reads, then asserts the empty real pipeline reports its artifacts as not loaded.

**Alternatives.** Trusting the fake; inspecting SearchPipeline.__init__ bytecode names (what I wrote first, and discarded as clever rather than readable).

**Why.** inspect_artifacts reads the pipeline by attribute name. If SearchPipeline renamed `rerankers`, /health would report a perfectly healthy service with every model missing, and every other test in the file would still pass because they all use the fake. This is the cheapest test that fails on that rename.

**How it could be tested.** The test itself; it passes against the current SearchPipeline.

## Frontend

### D209 — Browser talks to Next route handlers, never to FastAPI directly

**Decision.** The page posts to this app's own `/api/search` and `/api/health` route handlers (`web/app/api/*/route.ts`), which forward to the Python service at `CONCIERGE_API_URL` (default `http://localhost:8000`). The backend address is a server-side env var with no `NEXT_PUBLIC_` prefix.

**Alternatives.** (a) `fetch('http://localhost:8000/search')` straight from the browser with a `NEXT_PUBLIC_` env var; (b) a Next `rewrites` proxy in next.config.ts.

**Why.** Two concrete wins. CORS: same-origin requests mean the FastAPI service needs no CORS configuration for the demo to work on a laptop with the API on :8000 and the UI on :3000. Failure shaping: a route handler can turn ECONNREFUSED into 'Cannot reach the search API at http://localhost:8000 — start it with `make serve`', which a `rewrites` proxy and a direct browser fetch cannot. The requirement was explicitly that the demo must not render a stack trace. The handler also maps ENOTFOUND, AbortError (timeout, 30s) and non-OK upstream statuses, and flattens pydantic 422 field errors into one sentence.

**How it could be tested.** Unverified against a live backend — no API, dataset or index exists on disk yet. The error paths are reachable by starting `npm run dev` with no API running (expect the amber 'not answering' banner on load and the red 'Search failed' banner with the `make serve` hint on submit).

### D210 — The three constraint states are distinguished by icon, word and style — colour is never the only cue

**Decision.** Each constraint row in `ConstraintList.tsx` carries an inline-SVG glyph plus a literal word: check + 'met', cross + 'not met', dashed-circle-with-dash + 'no data'. The no-data state uses a neutral slate palette with a dashed border; red is reserved strictly for a measured violation. A footnote appears whenever any no-data row is present, saying it is reported separately from a violation and not treated as one. The three colours are named `met`/`unmet`/`nodata` in the Tailwind theme rather than spelled as raw palette classes at each use site.

**Alternatives.** (a) Green/red/grey with colour alone; (b) green/red/amber, i.e. no-data as a weaker warning; (c) hiding no-data rows entirely to keep cards clean.

**Why.** The accessibility requirement rules out colour alone, and the project's own design decision rules out (b) and (c). Yelp's attribute coverage is sparse: rendering a missing NoiseLevel as any shade of failure claims a quiet cafe is loud, which is the exact bug the three-valued ConstraintStatus exists to prevent, and hiding unknowns would erase the sparseness finding that the failure analysis depends on. Naming the colours in the theme makes a stray `text-red-600` on a no-data badge a visible mistake rather than an invisible one. The explanation strings themselves come from the backend's `ConstraintView.explanation`, never rebuilt client-side, so the 'no data' wording cannot drift from the constraint logic.

**How it could be tested.** Not visually verified — no install, no build, no live response. Checkable on first run by searching a query with an attribute constraint and confirming greyscale screenshots still distinguish all three states.

### D211 — lib/types.ts is a hand-written mirror of serve/schemas.py, not OpenAPI codegen

**Decision.** The wire contract is transcribed into TypeScript by hand, field-for-field with snake_case preserved so the two files diff side by side, with a comment pointing at `serve/schemas.py` as the source to follow.

**Alternatives.** openapi-typescript or similar codegen from the running service's /openapi.json; or `any` everywhere.

**Why.** Codegen would mean the frontend cannot be built or typechecked unless the Python API is running — and this UI had to be written and reviewed with no backend, no dataset and no API key present. The contract is roughly eight types and changes rarely, so hand-sync costs less than that coupling. Worth revisiting if the API gains more endpoints.

**How it could be tested.** Typechecked clean. Drift between the two files is not currently caught by anything automated — that is a real gap if schemas.py changes.

### D212 — The loading skeleton contains no text and no sample data

**Decision.** `ResultsSkeleton` renders grey blocks only, `aria-hidden`, with the 'Searching…' state announced from a polite live region instead. No fixture data exists anywhere in committed source.

**Alternatives.** A skeleton populated with a labelled synthetic business (the task permitted this if clearly marked as a fixture).

**Why.** A skeleton holding plausible names, ratings or constraint rows puts numbers on screen that no measurement produced, and a screenshot taken a moment too early becomes a screenshot of fiction — in a repo whose rule is that every number is real. Grey blocks cannot lie, so the labelled-fixture escape hatch was not needed.

**How it could be tested.** Verified by inspection: no business names, ratings, scores or constraint values appear in any file under web/.

### D213 — Plain useState + fetch, with no data-fetching library and no client-side cache

**Decision.** `SearchExperience.tsx` is the only stateful component: one phase enum, one response object, one in-flight request guarded by a monotonic request id so a slow earlier response cannot overwrite a faster later one.

**Alternatives.** React Query / SWR, or `useTransition` with a server action.

**Why.** One endpoint, one in-flight request, nothing to invalidate. More importantly a client-side cache would actively damage the demo: the backend's parse cache and the per-stage latency figures are the things we want the viewer to observe, and serving a repeat query out of React state would hide a real cache hit behind a fake one. The request-id guard is not theoretical — clicking two example chips in quick succession is the obvious thing a demo viewer does.

**How it could be tested.** Typechecked only. The out-of-order case is checkable once the API runs by clicking two chips in quick succession and confirming the rendered results match the last query submitted.

### D214 — Client-side normalisation fills missing containers but invents no values

**Decision.** `lib/api.ts` coerces every response field defensively (missing lists become `[]`, missing dicts `{}`, unrecognised constraint statuses collapse to `unknown`, non-finite scores stay NaN and render as '–' or 'rating unavailable').

**Alternatives.** Trusting the payload shape and letting a malformed response throw; or a runtime validator such as zod.

**Why.** A live demo must not blank out because one field arrived in an unexpected shape, but the fix must not be to substitute a plausible number. Empty containers degrade a card to 'less shown'; a default score or rating would misreport the system. An unrecognised status collapsing to `unknown` rather than `satisfied` is the conservative direction — it under-claims rather than over-claims. zod was skipped to avoid a dependency for roughly eighty lines of narrowing.

**How it could be tested.** Typechecked; the narrowing functions are pure and are the obvious first unit-test targets if a frontend test harness is added.

### D215 — One light palette, no system-following dark mode

**Decision.** `color-scheme: light` and an explicit light palette; no `dark:` variants anywhere.

**Alternatives.** Tailwind `dark:` variants driven by prefers-color-scheme.

**Why.** The three constraint colours are the only colours in this app that carry meaning. A second theme doubles the contrast surface that has to be re-verified every time they change, for a UI that will be shown on one laptop and screen-recorded for the README. One known-contrast rendering is worth more here than two unverified ones.

**How it could be tested.** Not verified — contrast ratios were chosen by eye from the Tailwind palette, not measured.

### D216 — Deployment warnings and /health degradation are surfaced, not hidden

**Decision.** A `/health` check runs once on mount: an unreachable or unready backend shows an amber banner before the viewer types; a ready-but-degraded one lists its warnings. `SearchResponseModel.warnings` renders above the results on every search with the line 'Results are real, but not from the full four-stage system.'

**Alternatives.** Logging warnings to the console, or showing health only in a debug panel.

**Why.** CLAUDE.md names this the worst failure mode for the project: a deployment serving retrieval order while the README claims a fine-tuned reranker looks like success. If the demo is screenshotted or recorded while degraded, the banner has to be in the frame.

**How it could be tested.** Unverified — depends on what the (not yet written) /health endpoint reports. The synthesised 'unavailable' payload returned when nothing answers is labelled via its `error` field as the UI's own conclusion, not a server report.

### D217 — Typechecked by borrowing an existing local TypeScript install rather than running npm install

**Decision.** Per instructions no install or build was run. Instead `tsc --noEmit -p tsconfig.json` was run against the project's own tsconfig using the TypeScript 5.7.3 and React 19 type packages already present in a neighbouring project on this machine, via a temporary `web/node_modules` symlink that was removed in the same command (trap on EXIT). Zero errors over all 19 source files; the check was proven live by injecting a type error, seeing TS2322, and removing it.

**Alternatives.** Shipping unvalidated source; or running npm install against the instruction.

**Why.** Two real type errors were found and fixed this way that would otherwise have broken the first `npm run dev`: `React.ReactNode` used as a UMD global inside modules (fixed to `import type { ReactNode }`), and `Record<string, unknown>` assigned to the typed-optional `open_at`/`geo` views (fixed with explicit per-field narrowing). The borrowed types were next 16 rather than the pinned next 15.1.6, so next-version-specific API changes would not have been caught — but nothing in this code uses more than `NextResponse.json`, `NextConfig` and `Metadata`.

**How it could be tested.** The typecheck is the test. Not a substitute for `npm install && npm run build`, which remains unrun.

## Documentation

### D218 — Results tables ship with real structure and empty, reasoned cells

**Decision.** The RESULTS section prints the actual ablation table — six real row labels from `pipeline.ABLATION`, real column headers — with every cell as "— not yet measured", rather than omitting the tables until numbers exist or showing an example table with illustrative values.

**Alternatives.** (a) Omit the tables entirely and write a sentence saying metrics will appear later. (b) Show a filled example table labeled "illustrative". (c) Keep placeholder tokens like `X.XX` in the cells.

**Why.** (b) is disqualifying: an illustrative number is the single worst failure available to this project, and readers screenshot tables out of context. (a) hides the shape of the claim, which is itself most of the rigor — a reader can see from the empty table that six systems will be compared on four measures with CIs and paired tests. (c) reads as an unfinished draft and invites someone to fill it in by hand. The em-dash-plus-reason cell is also exactly what `eval/report.py` renders for a missing measurement, so the README is formatted the way the generated report will be, and no number in it will ever be retyped by hand.

**How it could be tested.** After `make eval` runs on real labels, the README tables should be replaceable verbatim by the generated `report.md` tables. If the structures do not line up, the README was promising a table the harness does not produce.

### D219 — Two documents with a hard split: README sells and justifies, architecture.md explains mechanism

**Decision.** The README carries the hook, the cascade diagram, one short section per stage, the quickstart, the empty results sections, the evaluation methodology, limitations, and the online-evaluation design. `docs/architecture.md` carries data flow, the artifact/config-hash scheme, model provenance, a long careful treatment of the out-of-fold cross-encoder feature, and a table of silent failure modes with their structural preventions.

**Alternatives.** One long README containing everything; or three documents (README + architecture + a separate evaluation document, duplicating eval/README.md).

**Why.** A recruiter or interviewer reads the first screen and skims the rest, so the README has to be self-contained and persuasive without becoming a manual. But the out-of-fold argument needs several hundred words to land, and compressing it into a README bullet would turn the project's subtlest correctness property into an unexplained assertion — the opposite of what it is for. Three documents would duplicate `eval/README.md`, which already owns the labeling workflow and is the owner's working document; the README links to it rather than restating it, so there is one place for each claim to be corrected.

**How it could be tested.** Read the README alone and ask whether the project is understandable and whether any claim in it is unsupported by a linked document. Read architecture.md alone and ask whether it repeats the README rather than deepening it.

---

# Environment and process decisions

## D220 — Consolidate OpenMP runtimes rather than suppressing the warning · Settled
*2026-10-15*

**Decision.** Four dependencies pull in an OpenMP runtime: scikit-learn, faiss-cpu and torch
each vendor a private `libomp.dylib`, and lightgbm links `@rpath/libomp.dylib` against an
rpath listing only Homebrew's prefix. All four are pointed at one runtime by
`scripts/fix_openmp.py`, run automatically by `make setup`.

**Alternatives.** (a) `KMP_DUPLICATE_LIB_OK=TRUE`, which is what essentially every search
result recommends. (b) Serialize the work into separate processes so the libraries never
co-load. (c) Drop faiss in favour of a numpy matmul, removing one runtime.

**Why.** The crash was real — the suite segfaulted in `__kmp_join_barrier` — but the crash is
not the dangerous part. LLVM's own warning says duplicate runtimes "can degrade performance or
cause incorrect results", and describes the environment-variable workaround as "unsafe,
unsupported, undocumented". A project whose entire value is trustworthy measurements cannot
rest on a flag whose documented failure mode is silently wrong numbers: there would be no
answer to "how do you know that NDCG is right?". Option (b) is disproportionate. Option (c) is
attractive on its merits and still open — at one metro's scale, exact search is a single
matrix multiply, and FAISS only earns its dependency cost once approximate search is needed —
but whether numpy is fast enough there is exactly the benchmark nobody has run, and it changes
the architecture `CLAUDE.md` specifies, so it belongs to the owner and should be decided on
that benchmark rather than on convenience.

**How it could be tested.** `make check-openmp` asserts that exactly one `libomp` image is
loaded, that FAISS returns the same neighbours as brute-force numpy, and that LightGBM
recovers a planted ranking signal. Absence of a crash is deliberately not accepted as
evidence, because silent wrongness was the actual risk.

**Mechanism note.** LightGBM needed different treatment from the other three. It has no
`@loader_path` rpath entry, so a symlink beside it would never be found; the dependency is
rewritten to an absolute path with `install_name_tool` and the image re-signed ad-hoc, which
Apple silicon requires after any Mach-O modification.

## D221 — `.gitignore` patterns for the dataset must be root-anchored · Settled
*2026-10-15*

**Decision.** The dataset is excluded with `/data/`, not `data/`.

**Why.** An unanchored `data/` matches a directory of that name at any depth, so it also
excluded `concierge/data/` — the entire data layer. The repository committed clean and would
have been unbuildable on clone, including on Colab. Worth recording because the failure was
invisible locally: everything worked, `git status` was empty, and only `git check-ignore`
revealed it.

**How it could be tested.** Clone the repo into a fresh directory and run `make setup && make
test`. That is the actual claim being made, and it is worth running once before the project is
called finished.

## D222 — Adversarial audit as a build step · Settled
*2026-10-15*

**Decision.** After the modules were written, three independent read-only auditors reviewed the
codebase along separate axes — fabricated numbers, train/test leakage, and correctness — with
no authority to edit. Their findings were then fixed by separate agents and re-audited.

**Alternatives.** Trust the passing test suite. Review inline while writing each module.

**Why.** The suite was green — every test passing — and the audit still found 33 real issues
(countable in `.audit/findings.md`), including a candidate-depth sweep that selected the
serving operating point on the gold test set, a cross-encoder trained on raw query text but
served constraint-stripped text, and several leakage guards that existed but were never
invoked on the real code path. None of these would fail a test, because each produces a
plausible number rather than an error. A separate reader with an adversarial brief finds a
different class of defect than the author of the code does — which is the whole argument for
code review, applied here to a codebase written quickly by many parallel agents.

**What the re-audits then found.** Two further read-only passes re-examined the same three
dimensions after the fixes landed, tracing live code paths rather than trusting that a guard
exists, and raised 32 more items (`.audit/outstanding.md`): 10 where the fix had been applied
to some of the sites a finding named and not the rest, and 22 new ones. The dominant pattern
was *relocation* — a docstring corrected while the copy of the same sentence in prose, or in
this file, survived — which is itself the argument for auditing by grepping the claim rather
than by visiting the line a previous report cited.

**How it could be tested.** The re-audit passes are exactly that test, and the only honest
exit condition is a pass that finds nothing in the live code paths rather than one that finds
nothing in the files it last edited.

## D223 — PyStemmer is a required dependency, not an optional one · Settled
*2026-10-17*

**Decision.** `PyStemmer` is declared in `pyproject.toml` rather than left as an optional
extra of `bm25s`.

**Why.** The BM25 module degraded gracefully without it — it warned, recorded `stemmer: none`
in the index metadata, and carried on. That is good engineering and it still produced a
silently worse system: a smoke run over the fixture corpus returned **zero results** for
"quiet cafe with wifi", because the document says `Cafes` and the query says `cafe`. With
stemming installed the same query returns both matching businesses. Graceful degradation is
the right behavior for a missing optional component, but lexical recall is not optional here —
a dependency whose absence costs recall silently should not be optional in the first place.

**How it could be tested.** The index metadata records the resolved stemmer name, so an
evaluation run can assert it is not `none`. Worth adding to `make eval` as a precondition
check: a result produced by an unstemmed index is not comparable to one that was.

**Found by.** Running the pipeline end to end rather than reading it. The unit tests all
passed with the stemmer absent, because each tests its own layer and none asked "does a
plausible user query actually return anything?".

## D224 — Open gap: constraint-only queries retrieve nothing · Settled (fix chosen: option (a))
*2026-10-17, resolved 2026-10-19*

> **Resolution.** Option (a) is implemented in `concierge/retrieval/constraints_index.py`,
> built behind option (c)'s `Retriever` interface so that fusing the list as a third system
> stays reachable if measurement later favours it. `concierge.pipeline._retrieve` calls
> `backfill` at the end of Stage 2, gated on `Stage.FILTERS`. The reasoning for (a) over (b)
> and (c) is in that module's docstring; the measured defect table below is reproduced as
> `D224_TABLE` in `tests/test_constraint_retrieval.py`, so the queries in it are now
> executed rather than quoted. The "stated known limitation" paragraph at the end of this
> entry is therefore superseded — see D225 for the one seam the fix had left open.
>
> **The secondary observation is resolved too** (*2026-10-19*). The strip rule now lives in
> exactly one place, `concierge.query.rules.strips_phrase`: a phrase the parser lifted leaves
> `semantic_text` **if and only if** it became a constraint the filters can *reject* a
> business for. Each rule declares one `Writes` field and the strip decision is derived from
> it, so no rule can hold a private policy; `TestTheStripRule` in
> `tests/test_query_rules.py` is parametrized over every kind the rule table declares, and
> `test_the_strip_rules_premise_holds_in_the_filters` pins the premise (a category check
> returns SATISFIED or UNKNOWN, never VIOLATED — D225) against the real filter functions.
> The behaviour change this required: an all-constraint query ("free wifi") now yields
> `semantic_text == ""` instead of the raw query it had just stripped. That raw-query
> fallback was the whole of the inconsistency this observation describes — it fired only
> when stripping emptied the text, which is why "patio" and "cheap" were removed while
> "free wifi" came back whole. Two things make the empty string safe, and both are
> executed rather than argued: an empty residue proves the query carries at least one hard
> constraint (`test_an_empty_residue_always_carries_a_hard_constraint`), which is exactly
> what the constraint-only retriever above generates candidates from; and no model is ever
> handed the field directly, because `concierge.rerank.model.query_text_for` is the single
> definition of a stage's query text and falls back to `raw`. Re-measured on the fixture
> corpus after the change, the `retrieved` column of the table below is unchanged
> (0 / 0 / 0 / 4 / 3) — what the retrievers are given did not change, only what the parse
> *reports* having kept, which the API response and the eval dumps now show honestly.
>
> **One correction to the argument, not the decision.** Option (b)'s entry below, and the
> docstrings in `concierge.retrieval.constraints_index` and `concierge.query.rules`, used to
> claim that leaving "patio" in the query buys no lexical match because documents render
> the attribute as "outdoor seating". Measured on `tests/fixtures/`, that is false: the two
> documents containing "patio" are b001 (in a review) and b006 (whose name is "Patio
> Taqueria"), which are exactly the two businesses with `OutdoorSeating=True`. Both
> docstrings now state the measurement. The choice is unaffected — (b) is rejected for the
> double-counting argument and, decisively, because the rules arm would then embed the full
> query while the LLM arm embeds stripped text, making the parser ablation row
> unattributable — but the fixture corpus is **not** evidence against (b), and nothing here
> is evidence for either policy against relevance labels.

**The defect.** When the rule parser strips constraint phrases out of the query, what remains
is sometimes a generic residue that matches no document. Retrieval then returns zero
candidates, and the hard constraints — which were parsed correctly — never get applied,
because there is nothing left to filter. Measured on the fixture corpus:

| query | semantic_text after stripping | constraints parsed | retrieved |
|---|---|---|---|
| `somewhere with a patio` | `somewhere` | 1 | 0 |
| `cheap place` | `place` | 1 | 0 |
| `a nice place to sit outside with wifi` | `a nice place to` | 2 | 0 |
| `free wifi` | `free wifi` | 1 | 4 |
| `open past 10pm` | `open past 10pm` | 1 | 3 |

**Why it matters more than it looks.** `constraint` is one of the four gold query types, and
CLAUDE.md's own example of it is "open after 10pm with outdoor seating" — a query whose
semantic content is almost entirely constraints. So this defect targets an entire row of the
per-query-type breakdown, and it fails by returning an empty list rather than a wrong one,
which is the shape of failure least likely to be noticed in a demo.

**Secondary observation.** The stripping is applied inconsistently: `patio` and `cheap` are
removed while `free wifi`, `quiet` and `outdoor seating` are left in. Whatever the fix, the
rule should be uniform and stated, because D-level reasoning about double-counting (see the
rule parser's decisions) assumes it is.

**Candidate fixes, none yet chosen.** (a) When the parsed query has hard constraints and
retrieval returns fewer than some threshold, fall back to constraint-driven candidate
generation over the attribute index. (b) Do not strip constraint phrases from `semantic_text`
at all, and let the fusion handle the redundancy. (c) Keep a constraint-only retrieval path as
a third retriever fused by RRF alongside BM25 and dense.

**How it would be tested.** A constraint-only query must return every business satisfying the
constraints, ordered sensibly, and the per-query-type breakdown for `constraint` queries is
the measurement that would confirm it. Until then this is a stated known limitation, not a
solved problem — and it must be in the README's limitations section.

---

## D225 — The constraint path reads the filters' policy exemption, it does not re-derive it · Settled
*2026-10-19*

**Context.** Two changes landed concurrently from different sides of one seam. On the filters
side, a lexical category miss became *exempt* from `missing_attribute_policy`
(`filters.POLICY_EXEMPT_CONSTRAINTS`): "none of this business's categories matched the words
the parser emitted" is a fact about our matcher and Yelp's taxonomy, not about the dataset's
coverage of the business, so the missing-data knob is the wrong instrument for it. On the
retrieval side, the new constraint-only path (D224) decided admissibility itself, with the
rule "any UNKNOWN match is admitted only when the policy is not `exclude`, because under
`exclude` the filters would drop it".

That second sentence stopped being true when the first change landed, and nothing failed.

**The consequence, measured on the fixture corpus.** For `ramen` under
`missing_attribute_policy=exclude`, `apply_filters` keeps seven of the eight open fixture
businesses; `ConstraintIndex.select` offered one. Six businesses the filters would have kept
were withheld by the stage whose job is to supply them. So the constraint path was *stricter*
than the filters it exists to feed, and it reinstated D224's empty-page defect for exactly the
queries most likely to need it — the ones naming a category the lexicon misses.

Why no test caught it: the agreement test asserted only that nothing offered gets dropped
("a backfilled candidate the filters then delete would leave the page empty again"). Being
over-strict passes that assertion. The direction that was never asserted is the one that
broke.

**Why this is worse than a recall bug.** `exclude` is one of the three arms of D007's
missing-data comparison. A shortfall in that arm reads as a property of the policy — "excluding
unknowns costs a lot of recall" — when it was really a disagreement between two modules. That
is a wrong conclusion drawn from a real number, which is the failure mode this project is
built to avoid.

**Decision.** `select` imports `POLICY_EXEMPT_CONSTRAINTS` from `filters` and admits a
business unless the policy is `exclude` *and* it has an UNKNOWN the policy governs. The
exempt set is imported, never restated, so the two modules cannot drift about it again. The
established-first ordering is unchanged: an exempt UNKNOWN still means "not established", so
such a business ranks below one whose data settles every check.

**Alternatives considered.** (a) Restate the exempt set in `constraints_index`. Rejected: that
is the drift that caused this, written down a second time. (b) Have `select` ignore the policy
entirely and let `apply_filters` be the only gate. Rejected: under `exclude` it would then
offer candidates certain to be deleted, spending the backfill budget on rows that cannot
appear — the mirror of this bug. (c) Drop the exemption from `filters` instead, restoring the
old agreement. Rejected on the merits argued in `filters.POLICY_EXEMPT_CONSTRAINTS`: it makes
the D007 knob partly a measurement of taxonomy gaps.

**A note on cost.** Under `exclude`, group 2 of `select` used to be skipped outright. It now
has to run so exempt-only unknowns can be admitted — except when the query constrains no
exempt field at all, in which case every UNKNOWN is governed, the group provably admits
nothing, and the scan is skipped exactly as before. So the old fast path is preserved
wherever it was valid rather than traded away.

**How it is tested.** `test_selection_admits_exactly_what_the_filters_keep_under_every_policy`
asserts set *equality* between what `select` offers and what `apply_filters` keeps over the
whole open corpus, parametrized over all three policies, on a query carrying a category
constraint. Set equality rather than one-way containment, because one-way containment is what
missed this. Verified to fail against the pre-fix logic with the six withheld ids named in the
message. `test_a_constraint_only_query_returns_results_under_every_policy` pins the
end-to-end property through the real pipeline in all three arms, which is the coverage gap
that let a policy-arm-specific bug through in the first place.

---

## D226 — `training_query_text` lives beside `query_text_for`, not in the trainer · Settled
*2026-10-19*

**Context.** "The text a stage scores" must have exactly one definition, or training, mining
and serving agree only by coincidence. The function that derives it from a raw query,
`training_query_text`, was written in `concierge/rerank/train.py` because that is where the
train/serve skew was first caught. Hard-negative mining then needed the same string — mining
on the raw query draws negatives from a candidate pool the served system never produces, so
they are systematically easier than the real ones — and `concierge/synth/negatives.py`
correctly imported the helper rather than re-deriving it.

**The problem that created.** `concierge.rerank.train` imports `concierge.synth.splits` at
module scope, and `concierge.synth.negatives` then imported `concierge.rerank.train` at module
scope, so the two packages referenced each other. It resolved only because
`concierge/synth/__init__.py` is empty and `splits` imports `negatives` *inside functions*
rather than at module scope. One future module-scope import on that last edge turns a working
tree into an ImportError, and the tempting local fix at that point is to re-derive the query
text in `negatives` — which is the original defect, restored.

**Decision.** Move `training_query_text` (and its `_PARSE_DAY_FOR_TRAINING_TEXT` constant) to
`concierge/rerank/model.py`, immediately below `query_text_for`. That module imports nothing
from `concierge.synth` or from the trainer, so the edge is removed rather than merely
survived. Two further reasons this is the right home and not just a safe one: the two
functions that must agree about query text now sit in one file, where a change to either is
visible to the other; and `model.py` is already imported by the serving reranker, the trainer,
the ONNX export and the mining stage, so every consumer reaches the same object.
`rerank/train.py` re-exports the name, because "the text this trainer trains on" is a fact
worth being able to read off the trainer, and because that is how the existing tests refer to
it.

**Alternatives considered.** (a) Leave it and rely on the import order holding. Rejected: the
property is invisible, and the failure arrives as an ImportError in a module nobody edited.
(b) Make `negatives`' import of the trainer function-scoped. Rejected: it hides the coupling
instead of removing it, and the stated preference in that file was for a cycle to fail loudly
at import time rather than mid-run inside a mining job — which is only a virtue if the cycle
exists at all. (c) Move the whole pair-building API out of the trainer too. Rejected as scope:
`negatives.pairs_for_training` delegating to `rerank.train.build_pairs` is a *genuine*
dependency — the trainer owns what a training pair is — and `synth -> rerank` is the intended
direction.

**How it is tested.** `TestPackageLayering` in `tests/test_contracts.py`: an AST check that
`concierge/rerank/model.py` imports neither `concierge.synth` nor the trainer at module scope
(verified to fail when such an import is added); an identity check that the trainer's and
`negatives`' names are the *same object* as `model`'s, so a second definition is a failure
rather than a coincidence; and a subprocess import of each of the four modules on the former
cycle, one per fresh interpreter — because importing them in-process proves nothing once the
suite has already populated `sys.modules`.

## D225 — The `no_filters` diagnostic measures something narrower · Settled by the owner
*2026-10-19*

**What changed.** Fixing the dead constraint features (D226) moved constraint evaluation to
the end of Stage 2, so it runs in *every* variant. `Stage.FILTERS` now gates only whether a
candidate is dropped or penalised, not whether its constraints are checked.

**The consequence for evaluation.** The `no_filters` diagnostic used to compare "the full
system" against "no constraint handling at all". It now compares against "nothing is dropped
and nothing is penalised", because Stage 4 sees the constraint features in both arms. That is
a different question, and a narrower one.

**Why it was done this way.** The alternative — leaving Stage 4 to score candidates whose
constraint features are absent — feeds the model feature values it never saw in training.
That makes the comparison uninterpretable in an unknown direction rather than merely narrower.
Between a narrower question answered correctly and a broader question answered meaninglessly,
the first is better.

**This is an evaluation-methodology change, which CLAUDE.md section 10 says to ask about
before making.** It was made by an agent that could not ask, so it was put to the owner.

**The owner's ruling: accept the narrower definition** (2026-10-19). `no_filters` now measures
what the drop/penalise half of the hard filters is worth, holding the constraint features
constant. The broader question moves to a Stage 4 feature ablation.

**A gap that ruling exposed, now closed.** The existing ablation
(`--ablate-non-text`) drops rating, popularity, constraint satisfaction and distance *as one
block*, so it answers "do the non-text signals matter?" and cannot answer "does constraint
arithmetic matter, as distinct from similarity *and* from quality signals?". Pointing at it as
the instrument for the broader question would have been wrong. A `CONSTRAINT_FEATURE_NAMES`
block and a `--ablate-constraints` training mode now exist, keeping rating, popularity and
distance while removing only the four constraint features. The two ablations are mutually
exclusive — requesting both raises, because the resulting feature set answers no stated
question — and each writes to its own artifact directory so the runs cannot overwrite each
other.

**How it could be tested.** Run `make train-ltr` three ways (full, `--ablate-non-text`,
`--ablate-constraints`) and compare NDCG@10 with paired bootstrap intervals. The constraint
block's contribution is the full-versus-`--ablate-constraints` difference. Until those runs
exist, no claim about what the constraint features are worth belongs in the README.

## D226 — Constraint checks run in every variant · Settled
*2026-10-19*

**Decision.** `pipeline._filter` always evaluates hard constraints and attaches them to each
candidate. `Stage.FILTERS` controls only dropping and penalising. Evaluation happens once, at
the end of Stage 2, and `_assemble` reads what the candidate carries rather than recomputing.

**Why.** Three Stage 4 features — `n_constraints_satisfied`, `frac_constraints_satisfied`,
`has_unknown_constraint_data` — are computed from `candidate.constraints`. Attaching
constraints during result assembly meant any variant that skipped the filter stage handed
LambdaMART constant values. Evaluating once also means the "why this result" explanation the
user reads is provably the same data Stage 4 scored, rather than a second, independently
recomputed answer that could drift from it.

**Scope correction, recorded because it was first reported wrongly.** This was initially
described as affecting every candidate on every query. It did not. Measured on the pre-fix
commit, the production `full` path and every ablation row already carried constraints
(`[3, 3]` per candidate) because `apply_filters` attached them; only the filters-off
diagnostic was affected (`[0, 0]`). The bug was real and worth fixing, but its blast radius
was one supplementary row, not the headline table.

**How it could be tested.** A regression test asserts the three features *vary* across
candidates when Stage 4 scores them — a test that merely checked the row was well-formed
would have passed throughout the bug. It was verified to go red against the old code.

## D227 — Grade only the labeled candidates in the LambdaMART feature rows · Settled
*2026-10-19*

**Context.** `concierge/ltr/build_examples.py` turns the query-level LambdaMART folds into
candidate-level rows. Per query it runs the real cascade and sees roughly
`candidates.rerank` survivors, of which only two kinds carry label evidence: the synthetic
query's source business (relevant by construction — the query was generated *from* its
document) and the hard negatives that survived the LLM judge's false-negative filter. The rest
have never been judged by anyone.

**Decision.** Label the source business 2 (`RELEVANCE_HIGH`) and each surviving hard negative 0
(`RELEVANCE_IRRELEVANT`), on the same 0/1/2 scale the gold set uses. **Emit no row at all for a
candidate that is neither**, and report how many were dropped
(`n_unlabeled_candidates_dropped`). Skip a query entirely — counted by id — when the positive
did not survive the cascade, or when no labeled negative did.

**Alternatives considered.**
1. *Label every other candidate 0 and emit the full shortlist* — the common LTR default.
   Rejected: those candidates are precisely the ones retrieval and the cross-encoder ranked
   highest, which is the population most likely to be genuinely relevant. That is the premise
   of the LLM judge, which spends quota removing exactly such false negatives from the mined
   set; labeling the unjudged remainder 0 reintroduces them at larger scale with no filter.
   Worse, under `lambdarank` each pair's gradient is scaled by how much swapping the pair would
   change NDCG, so mislabels near the top of the list are weighted most heavily of all — the
   noise lands on the rows that decide the model.
2. *Grade the unjudged remainder 1 ("partially relevant").* Rejected outright: nothing measured
   that, and an invented label is the one thing this project may not produce.
3. *Emit the positive anyway, with no stage scores, when it did not survive the cascade.*
   Rejected: at serving time a candidate with no scores never reaches Stage 4, so such a row
   teaches the model that missing scores mean relevance — a shortcut that shows up as excellent
   training NDCG and nowhere else.

**Why this one.** It is the only option in which every label traces to evidence that exists:
construction for the positive, an LLM verdict for the negative. Nothing is asserted that was
not checked.

**The cost, stated rather than glossed.** Each group becomes one positive against a handful of
judged-irrelevant negatives — a narrower and easier list than the shortlist the served system
ranks, because the plausible-but-unjudged middle is absent. LambdaMART therefore trains on a
biased sample of the serving distribution. A second, smaller residual: the judge grades only
the top-N hardest negatives per example and `HardNegative` records no verdict, so some label-0
rows are "mined but never judged" rather than "judged irrelevant", and nothing downstream can
distinguish them.

**How it could be tested.** Judge a random sample of the dropped candidates with the same LLM
judge and measure what fraction it grades relevant. A small fraction makes labeling them 0
defensible and the groups can widen toward the serving distribution; a large fraction confirms
the decision and is itself a finding for the failure analysis. Separately, train Stage 4 both
ways and compare NDCG@10 on the gold set with a paired bootstrap — that is the measurement that
settles it rather than this argument. The cheap preliminary is already reported per run:
`n_unlabeled_candidates_dropped` over `n_candidates_scored` bounds how much of the serving
distribution training never sees.

## D228 — Three contract gaps the examples producer had to work around · Settled (all three closed)
*2026-10-19*

> **Resolution.** All three interfaces named below now exist, and each workaround was deleted
> rather than kept alongside them.
>
> 1. **`SearchPipeline.candidates(parsed, *, variant, limit, record)`** runs Stages 2 and 3 and
>    returns `Candidate`s with every per-stage score *and* rank attached. `search()` calls it
>    and adds Stage 4 and assembly on top, so the cascade order is written once.
>    `concierge.ltr.build_examples.Cascade` is now a thin wrapper over it — a loader that
>    refuses missing artifacts plus the per-stage counts — and restates nothing.
>    `test_the_producer_and_search_agree_on_the_cascade` runs one pipeline both ways; the old
>    test compared two implementations, which could catch a drift but not prevent one.
>    Checked during integration by running the pre-refactor `search()` and the current one over
>    the same injected fixture parts for every ablation variant: 216 (variant, query, k)
>    responses, identical in results, per-stage scores, constraint matches, stage list and
>    candidate counts.
> 2. **`concierge.rerank.model.training_parse(query) -> ParsedQuery`** is public, and
>    `training_query_text = query_text_for(training_parse(q))`. `build_examples.parse_for_training`
>    calls it, so the weekday constant is no longer copied and the text scored by the
>    cross-encoder in the Stage 4 producer is the string the checkpoint was fine-tuned on by
>    construction. Deliberately uncached, unlike the text: `ParsedQuery.attributes` is a plain
>    dict, and a shared instance would let one caller's in-place mutation change another's
>    filters.
> 3. **`HardNegative` records the judge's verdict**, so `rows_for_query` takes the mined records
>    instead of bare ids and the run report splits label-`0` rows into
>    `n_negative_rows_with_judge_verdict` and `n_negative_rows_unjudged`. No label changed —
>    D227 settled the policy — but the evidence behind a `0` is now recoverable from the
>    artifact instead of requiring a reader to know the judge's `top_n`. A file written before
>    negatives carried verdicts reports every label-`0` row as unjudged, which is the honest
>    reading of a file with no grading provenance in it.

Recorded because each is a real seam, and because the workaround in each case is a test pinning
behaviour that a public interface should have guaranteed instead.

1. **`concierge/pipeline.py` has no public candidate-level entry point.** `search()` returns
   `SearchResult`s, which carry per-stage scores but have dropped per-stage *ranks*
   (`rrf_rank`, cross-encoder rank) — both Stage 4 features. So the offline row producer cannot
   reuse the pipeline and restates the order of `_retrieve`/`_filter`/`_rerank`. Every stage
   *decision* is still delegated, and a test pins the restatement against a real
   `SearchPipeline`, but a public `SearchPipeline.candidates(parsed, variant)` that `search()`
   itself called would mean the producer restates nothing. This is the highest-value of the
   three: a duplicated cascade order is exactly the kind of thing that drifts silently.
2. **`concierge/rerank/model.py` keeps the training-text weekday private.**
   `training_query_text` returns only the text, not the `ParsedQuery`, so any stage needing the
   full parse consistent with the checkpoint's training text must re-derive it with the same
   day. A public `training_parse(query) -> ParsedQuery`, with
   `training_query_text = query_text_for(training_parse(q))`, would remove the duplicated
   knowledge.
3. **`synth/negatives.HardNegative` does not record whether the judge graded it.** The judge
   grades only the top-N hardest negatives per example, so negatives surviving into the
   filtered file are a mix of "judged and kept as irrelevant" and "never judged", and nothing
   downstream can tell which. A `judged`/`verdict` field would make each negative's label
   provenance checkable — see D227, where this weakens the evidence behind some label-0 rows.

## D229 — Uniform stripping does not reach the string retrieval scores · Correction
*2026-10-19*

**The correction.** D224's remaining half was closed by stripping every lifted constraint
phrase from `semantic_text`. The commit that did it claimed this fixes the double-counting the
stripping argument rests on. **At the layer that matters, it does not**, and the claim was
checked only at the `semantic_text` field rather than at the string retrieval actually scores.

`query_text_for(parsed)` — not `semantic_text` — is what BM25 and the embedder receive, and it
falls back to the raw query when `semantic_text` is empty. Measured over the fixture queries:

| query | `semantic_text` | string actually scored |
|---|---|---|
| `free wifi` | `''` | `free wifi` |
| `outdoor seating` | `''` | `outdoor seating` |
| `open past 10pm` | `''` | `open past 10pm` |
| `quiet cafe with wifi` | `cafe` | `cafe` |
| `cheap ramen open past 1am` | `ramen` | `ramen` |

So for **mixed** queries stripping does what it claims. For **pure-constraint** queries it does
not: the text empties, the fallback returns the raw query, and the constraint phrase is scored
after all.

**Why the fallback is still right.** Scoring every candidate against `''` makes the stage a
no-op that costs full latency — worse than scoring slightly redundant text. The alternative,
skipping retrieval entirely when `semantic_text` empties, would hand those queries wholly to
the constraint retriever and give the fusion nothing to fuse.

**What is therefore true, stated precisely.** The rule is uniform at the field, and uniform at
the scoring layer too — "score `semantic_text`, or the raw query when nothing remains" is one
rule applied consistently. What is *not* true is that double-counting has been eliminated: it
is eliminated for mixed queries and unavoidable for pure-constraint ones, short of scoring an
empty string. That is a narrower claim than the commit made.

**How it could be tested.** Compare NDCG@10 on the `constraint` slice of the gold set under
three policies: strip-and-fall-back (current), strip-and-skip-retrieval, and no stripping. The
constraint retriever now backstops all three, so the comparison is finally meaningful — before
it existed, two of the three would have returned nothing.

**Found by** an adversarial auditor that compared `query_text_for` across the change rather
than `semantic_text`, explicitly because the field is what the commit message, DECISIONS and
README all described. Comparing the field would have confirmed the framing instead of testing
it. That is the generalisable lesson: verify at the layer the system consumes, not the layer
the change was written at.

## D230 — A StageRecorder measures one cascade run and refuses reuse · Settled
*2026-10-19*

**Decision.** `StageRecorder.measure()` raises if asked to measure the same stage twice.

**The bug it closes.** The recorder became public when `candidates()` did. Its two fields
disagreed about what reuse meant: `timings` is a list and appended, `counts` is a dict keyed by
stage and overwrote. Measured: two `candidates()` calls on one recorder produced six timings
and one query's counts. Nothing raised. Any per-stage p50 or p95 computed from that recorder
would have been wrong — and wrong in the direction of looking plausible, since the numbers are
real measurements of real work, just double-counted.

**Alternatives.** Document single-use and trust callers; make `counts` accumulate too for
consistency; reset on each `measure`. All leave a public API whose misuse is silent, and
latency is a reported result of this project.

**Why.** The cheapest honest contract is the one that cannot be used incorrectly.

**How it could be tested.** `test_a_stage_recorder_refuses_reuse_across_queries` asserts the
raise, that the refusal leaves no partial entry, and that a *different* stage on the same run
is still accepted — the normal case must keep working.
