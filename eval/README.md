# Evaluation: your gold set, and how to label it

This directory holds the ground truth for the whole project. Every number in the
README, every ablation row, and every resume bullet is computed from two files
you create by hand:

| file | who writes it | what it is |
| --- | --- | --- |
| `gold_queries.csv` | **you** | ~150 test queries |
| `qrels.csv` | **you**, via `make label` | graded relevance labels |
| `qrels_retest.csv` | **you**, via `--retest` | a second pass, for the consistency check |
| `pooling.py` | code | decides which pairs you are asked about |
| `label_cli.py` | code | the grading tool |
| `metrics.py` | code | NDCG@10, Recall@100, MRR, bootstrap CIs, kappa |
| `run_eval.py` | code | the ablation harness behind `make eval` |
| `report.py` | code | renders `report.md`, `results.json`, the latency curve |

Nothing in this project is allowed to write the queries or the labels for you.
A model-written test set graded by a model is a system marking its own homework,
and an interviewer will find that out in one question. The hours you spend here
are the reason the rest of the numbers mean anything.

---

## 1. Writing the gold queries

Open `gold_queries.csv` and replace the five `EXAMPLE*` rows with your own. The
format is three columns:

```csv
query_id,query,query_type
q001,cozy spot to read on a rainy afternoon,vibe
```

* `query_id` — `q001`, `q002`, … Unique, and **never reused or renumbered**:
  labels are keyed by it, so renumbering silently reassigns your grades to the
  wrong queries. The loader refuses duplicates.
* `query` — exactly what a person would type. Wrap it in double quotes if it
  contains a comma.
* `query_type` — one of `vibe`, `constraint`, `specific`, `messy` (below). A
  typo here is a hard error with the line number, not a silently created fifth
  type that would quietly shrink one of the four reported breakdowns.

**Delete the `EXAMPLE` rows before you start labeling.** The labeling pool skips
ids beginning `EXAMPLE`, so those rows can never be graded — but
`run_eval.py`'s own gold reader does not skip them yet, so an example row left
in the file is counted in the query-type totals and in the coverage denominator
while being permanently unlabelable. That makes "NDCG@10 over *n* of *m* gold
queries" overstate *m*, and it lands unevenly: two of the five shipped examples
are `constraint` queries. Having the two readers agree by construction is the
real fix and it is being wired up; deleting the rows is correct either way.

### The four query types

Every metric is reported per type, because the four stages help them by very
different amounts — and that breakdown is one of the project's headline results.
Aim for a roughly even spread; exact balance does not matter, but a type with
only a handful of queries will have a confidence interval too wide to say
anything about.

**`vibe`** — the request is about atmosphere and fit, with no checkable
constraint. The ranking has to come from text understanding.
> *cozy spot to read on a rainy afternoon*

**`constraint`** — the request contains at least one fact the dataset can verify:
hours, price, an attribute. A result that violates one is wrong no matter how
good it looks.
> *open after 10pm with outdoor seating*

**`specific`** — a concrete thing the user wants, usually a cuisine, dish, or
category. Closest to classic keyword search.
> *good pho*

**`messy`** — what people actually type: typos, vagueness, two intents at once,
or words the dataset never uses.
> *somewhere chill for a frist date not too expensive*

### What makes a good test query

1. **Write it as you would type it**, not as you would phrase it for a search
   engine. "quiet café with wifi that's open past 10pm" beats
   "cafe quiet wifi late".
2. **More than one place should be able to satisfy it.** If exactly one business
   can be correct, NDCG@10 over that query is nearly binary and tells you
   almost nothing about ranking quality.
3. **No business names.** "Lantern Coffee" tests string matching, not search.
4. **Write it before you look at any results.** A query written because you
   noticed a nice result is a query the system is guaranteed to do well on.
   This is the cheapest way to accidentally fake a good number.
5. **Make some of them hard.** A set of only easy queries compresses every
   ablation gap toward zero and hides exactly the differences you are trying to
   measure.
6. **Keep it in scope.** Queries have to be answerable from the metro you chose
   and from data the dataset actually carries (hours, price, noise, WiFi,
   outdoor seating, categories, review text).
7. **One request per query, except in `messy`**, where multi-intent is the point.

### Do not write "near X" queries yet

The geo filter and the distance feature both exist — `check_geo` is a haversine
test against the request's radius, and `distance_km` is a Stage 4 feature. What
is missing is the piece in between: turning a phrase like *downtown* or *near
the train station* into coordinates needs a gazetteer, and the repository has
none. `validate_parsed_query` accepts a `geo_resolver`, and nothing supplies
one, so `parsed.geo` is always `None` on the live path and `distance_km` is
`NaN` for every candidate.

The behaviour is graceful rather than broken — an unresolved location label is
folded back into `semantic_text`, so *cheap ramen near downtown* still retrieves
on the words — but it is retrieval on text, not a location constraint. If you
write and label such queries now, you will be grading missing functionality as
if it were a ranking failure, which then shows up as a permanent, unexplained
hole in the per-type breakdown (DECISIONS.md D003).

Write the rest of the set now, and add geo queries as a clearly separated batch
once a resolver is wired in. Then they can be labeled and reported on their own.

### Leakage — the one rule with no exceptions

Gold queries and their labels are **test-only**. They must never appear in
synthetic training data, in a prompt example, in hyperparameter tuning, or in
feature engineering. `make synth-generate` deduplicates synthetic queries
against this file — exactly, and by embedding similarity — and reports how many
were removed; it fails loudly if this file is missing rather than letting a
missing gold set be the thing that disables the guard. If you ever find yourself
tempted to tune something until a gold query looks better, that is the moment
the test set stops being a test set.

### The operating point is a tuned parameter too

The easiest way to tune on the test set without noticing is not to touch a model
at all — it is to read the quality-vs-latency curve on the gold set, pick the
candidate depths that look best, put them in `configs/base.yaml`, and then
report the headline NDCG@10 at those depths. Nothing in the code had to change;
the number is still inflated, because the depths were selected using the labels
they are being scored against. An interviewer who asks "how did you pick those
candidate depths?" is asking exactly this.

So the rule is split in two, and `eval/run_eval.py` makes you say which one you
are doing:

```bash
# CHOOSE the depths. Synthetic ltr_val split, no gold label involved.
# Needs `make synth` to have run (it writes the queries and the split manifest).
uv run python -m eval.run_eval --config configs/base.yaml --sweep-on validation

# MEASURE the chosen depths, for the report. Gold set. Prints a warning saying
# its numbers must not be used to pick depths.
make eval
```

`--sweep-on validation` runs the grid over the synthetic `ltr_val` split, whose
only positive per query is the business that query was generated from. That
makes the absolute level **not** comparable to a gold NDCG — it has one positive
per query, and other genuinely good places score as irrelevant — but the
ordering of operating points is what you need, and that ordering is not
contaminated. Read the curve, decide, write `candidates.retrieve` /
`candidates.rerank` / `candidates.final` into the config yourself, and record the
decision in `DECISIONS.md`. Nothing writes the config for you, on purpose: the
choice should be a deliberate act you can describe, not a side effect of running
an evaluation.

The gold curve stays in the report — it is a required deliverable (CLAUDE.md §6,
output 3) and it is the only honest answer to "what does a user actually pay for
that quality?". It is published as a *measurement of the configured operating
point*, not as the menu it was chosen from, and the warning that says so travels
in the result object so it cannot be lost in formatting.

---

## 2. Labeling

```bash
make label                      # resume (or start) a labeling session
uv run python -m eval.label_cli --config configs/base.yaml --stats   # pool size only
```

Two things have to exist first:

* **`data/raw/business.json`.** The tool shows you the real record for each
  candidate — name, categories, parsed attributes, the week's hours, rating, and
  review or tip snippets — so it reads the raw dataset, not the index. Override
  the paths with `--businesses` / `--reviews` / `--tips` if yours live elsewhere.
* **A runs file.** See [where the candidates come from](#where-the-candidates-come-from)
  below; this is the one genuinely manual step in the loop.

### What you see, and what you press

One query at a time, then its candidates one at a time, with the name,
categories, parsed attributes, the full week's hours, the rating, and a review
snippet or two. Those fields are what you judge against — if the screen says
"no hours data", the dataset genuinely has none for that business.

| key | meaning |
| --- | --- |
| `2` | **highly relevant** — I would send a friend here for this exact request |
| `1` | **partially relevant** — right kind of place, but part of the request is unmet or unknown |
| `0` | **irrelevant** — wrong kind of place, or it breaks a stated hard constraint |
| `s` | skip — undecided; the pair comes back next session |
| `u` | undo the last judgement |
| `q` | save and quit |

No Enter key. Every grade is written to `qrels.csv` and flushed to disk before
the next candidate appears, so a crash or a closed laptop costs you at most the
pair you were looking at. Quitting and rerunning resumes exactly where you
stopped — already-graded pairs are never served again.

### Grading the hard middle case

`2` and `0` are usually obvious. Almost all of your difficulty — and almost all
of your inconsistency between sessions — lives in `1`. Use these rules, and use
them the same way every time:

* **A violated hard constraint is a `0`, however good the place is.** "Open
  after 10pm" and the place closes at 9 — that is a `0`, even for a
  five-star café you love. The user asked a factual question and the answer is
  no.
* **Missing data caps a result at `1`.** The query asks for quiet, the business
  has no `NoiseLevel`: you cannot verify it, so it cannot be a `2`. It is not a
  `0` either — the dataset is silent, not negative. (This is the same
  three-valued distinction the ranking code makes, `satisfied` / `violated` /
  `unknown`.)
* **Right category, missing one requested feature → `1`.** A great ramen place
  with no outdoor seating, for "ramen with outdoor seating".
* **Right category, nothing else matching → `1`, not `2`.** For a `vibe` query,
  "a café" is not automatically a match for "cozy spot to read"; a sports bar
  café with a TV wall is a `0`.
* **Judge the request, not the quality of the business.** Star rating is a
  quality signal, not a relevance signal. The ranker is allowed to use ratings
  as a feature; your labels must not, or the two are measuring the same thing
  and the LambdaMART ablation becomes meaningless.
* **`s` rather than a guessed `0`.** If you cannot decide, skip it. A guessed
  grade is worse than a missing one: missing pairs are counted and reported as
  unjudged, guessed ones silently become ground truth.

Two habits that pay for themselves:

* Keep a note of any new rule you invent mid-session (e.g. "a bar counts as
  partially relevant for 'dinner'"), and re-read your notes when you start the
  next session. Consistency across sessions is a measurable property of your
  labels — see section 4.
* Label a whole query before taking a break. The grading standard lives in your
  head, and it is cheaper to keep it there than to rebuild it.

### Where the candidates come from

You are not shown every business in the searchable corpus — that is one metro
filtered to `is_open == 1`, and `make explore` prints how many that is. The
tool **pools**: it takes the top 20 results (`evaluation.pool_depth`) from every
system being compared, takes the union, drops anything already graded, and asks
you about the rest. Anything outside the pool counts as irrelevant when scoring,
so `metrics.py` reports the unjudged fraction next to every metric rather than
hiding that assumption.

Within a query, candidates are shuffled with a fixed seed rather than shown
best-first. That is deliberate: seeing the best results first would anchor your
standard high and let the system under test shape your labels.

The tool reads systems' results from a runs JSON (default
`<evaluation.out_dir>/runs.json`, i.e. `artifacts/eval/runs.json`; override with
`--runs`):

```json
{"bm25": {"q001": ["business_id_1", "business_id_2"]}}
```

**Nothing in the repository writes that file yet, and the bootstrap order is the
reason.** `make eval` refuses to run before labels exist, so the first pool
cannot come from an eval report — you have to run a system over your gold
queries and write its ranked ids into the runs file yourself. `label_cli` fails
with that exact instruction if the file is absent rather than pooling nothing. A
one-off script that imports `concierge.pipeline.SearchPipeline`, loops over
`gold_queries.csv` and dumps the shape above is enough; keeping it out of the
labeling tool is deliberate, because a labeling tool that could silently
regenerate its own pool would make "which systems was this pool built from?"
unanswerable after the fact.

When a new stage lands, run it, write its results into that file, and run
`make label` again: only the pairs nobody has graded yet are served
(**incremental pooling**). You will never be asked to re-grade a pair.

### You label a random subset first

Labeling all ~150 queries at depth 20 is many hours, so
`evaluation.label_subset_size` (50 in the shipped config) queries are served
first, chosen **at random** from a fixed seed — not the first 50 rows, not the
quick ones, not the interesting-looking ones. Every convenience-based choice
biases the headline metric in a specific direction (see `select_label_subset` in
`pooling.py` for the three ways it goes wrong). Raising the number later only
adds queries: the ones you have already labeled stay in the subset.

Every reported metric states how many queries it was computed over. "NDCG@10
= x over a random 50 of 150 gold queries" is an honest sentence. Dropping the
second half of it is not.

`--stats` prints how many judgements a session will be before you commit an
evening to it. It also prints an estimated time, measured from the timestamps of
your own past labeling — until there is enough history it says **unmeasured**
rather than guessing a pace.

Other switches, all optional:

| flag | what it does |
| --- | --- |
| `--stats` | print the pool statistics and exit, judging nothing |
| `--limit N` | stop after N candidates — useful for a short sitting |
| `--all-queries` | pool every gold query instead of the random subset |
| `--labeler NAME` | recorded in every row; defaults to `owner` |
| `--out PATH` | append to a different qrels file than `evaluation.qrels` |
| `--runs`, `--businesses`, `--reviews`, `--tips` | override the input paths |
| `--retest`, `--retest-size N`, `--agreement` | the consistency check, section 4 |

---

## 3. The files you produce

`qrels.csv` is append-only CSV, one row per judgement:

```csv
query_id,business_id,relevance,labeled_at,labeler
q001,abc123,2,2026-10-15T19:04:11+00:00,owner
```

Append-only so nothing can be lost by an overwrite, and plain CSV so you can
grep it. The timestamp is what measures your labeling pace and dates the
consistency check; `labeler` is there so a second annotator could be added later
without the two being merged. If the same pair ever appears twice, the last row
wins.

Only this CLI writes labels. The LLM judge (section 5) reads `qrels.csv` and
never writes it, because a judge that could write labels would make the
evaluation circular — and that rule is enforced at runtime rather than by
convention: the judge opens the file read-only, records its SHA-256, and
re-hashes it afterwards, so a file that changed underneath a judging run is an
error instead of a silently mixed result.

`make eval` reads both files and writes nothing back into this directory; its
output goes to `artifacts/eval/<config fingerprint>/`. It refuses to produce a
report until enough queries have at least one *relevant* label — a query whose
entire pool was graded `0` has an undefined NDCG and contributes nothing to any
mean, however many rows it added to the CSV. Two of its switches are worth
knowing while labeling is still in progress: `--systems` runs a subset of the
ablation, and `--max-queries N` smoke-tests the harness on the first N gold
queries while stamping `PARTIAL RUN` on every table it renders, so the output
cannot be mistaken for a result.

---

## 4. The consistency check (test-retest)

Your gold set has **one annotator**. An interviewer will ask how you know the
labels are any good, and "I was careful" is not an answer. With one annotator
there is no second rater to compare against, so compare against the only other
rater available: yourself, later.

**How to run it.**

1. Finish your first labeling pass.
2. **Wait at least two weeks.** This is the whole experiment. Re-labeling the
   next day measures your memory, not your standard.
3. `uv run python -m eval.label_cli --retest` — this re-serves pairs you already
   graded, for a random subset of your labeled queries (`--retest-size`, 30 by
   default), **without showing the previous grade**. Grade them fresh.
   Disagreeing with yourself is information, not a mistake, and nudging toward
   what you think you said last time destroys the measurement. Retest grades go
   to `qrels_retest.csv`, never to `qrels.csv`, so the second pass cannot
   overwrite the ground truth it is measuring.
4. `uv run python -m eval.label_cli --agreement` — prints the comparison.

**What to report**, all of it, in the README:

* how many pairs were graded twice, and the median gap in days;
* exact agreement (the plain percentage);
* Cohen's kappa, **both** quadratic-weighted and unweighted. Weighted kappa is
  the more flattering number on a 3-point ordinal scale, so it must never be
  quoted on its own;
* the confusion matrix, which shows the *shape* of your disagreement.
  Disagreement between `1` and `2` means your middle category is fuzzy;
  disagreement between `0` and `2` means something is wrong with the grading
  rules themselves and the affected queries deserve a second look.

Report the numbers you measure, and nothing else. There are commonly cited
interpretation bands for kappa in the statistics literature, but they are
conventions, not thresholds — do not translate your number into a word like
"substantial" and present that as a result. If the agreement is low, say so and
say what you changed: a low number you reported honestly is a far better
interview answer than a high number nobody can check.

---

## 5. Judge-vs-owner agreement

The same machinery answers a second, different question. An LLM judge grades a
sample of the pairs you labeled, and the kappa between the LLM and you is what
justifies — or fails to justify — using LLM labels inside the training data,
where the judge is what removes false negatives from the mined hard negatives.

```bash
make synth-label-quality        # needs eval/qrels.csv to exist
```

Read the result the same way as section 4: both kappas, the confusion matrix, no
adjectives. The two checks are complementary and neither substitutes for the
other — self-agreement bounds how noisy *your* labels are, and judge agreement
bounds how far an LLM's grades can stand in for them. If judge agreement comes
out low, the honest response is to drop the false-negative filtering step and
report that, not to relabel until the numbers agree.
