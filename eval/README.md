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
* `query_type` — one of `vibe`, `constraint`, `specific`, `messy` (below).

Rows whose id starts with `EXAMPLE` are skipped by every tool, so the shipped
examples can never reach a metric. Delete them once you no longer need the
format.

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

Location filtering lands in Phase 2 (see `DECISIONS.md` D003). Until then a
query like *cheap ramen near downtown* has no mechanism behind it, and when you
label it you will be grading missing functionality as if it were a ranking
failure — which then shows up as a permanent, unexplained hole in the
per-type breakdown.

Write the rest of the set now, and add geo queries as a clearly separated batch
after the geo filter exists. Then they can be labeled and reported on their own.

### Leakage — the one rule with no exceptions

Gold queries and their labels are **test-only**. They must never appear in
synthetic training data, in a prompt example, in hyperparameter tuning, or in
feature engineering. Phase 3 deduplicates synthetic queries against this file
(exactly, and by embedding similarity) and reports how many were removed. If you
ever find yourself tempted to tune something until a gold query looks better,
that is the moment the test set stops being a test set.

---

## 2. Labeling

```bash
make label                      # resume (or start) a labeling session
uv run python -m eval.label_cli --config configs/base.yaml --stats   # pool size only
```

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

You are not shown all 150K businesses. The tool **pools**: it takes the top 20
results (`evaluation.pool_depth`) from every system being compared, takes the
union, drops anything already graded, and asks you about the rest. Anything
outside the pool counts as irrelevant when scoring, so `metrics.py` reports the
unjudged fraction next to every metric rather than hiding that assumption.

Within a query, candidates are shuffled with a fixed seed rather than shown
best-first. That is deliberate: seeing the best results first would anchor your
standard high and let the system under test shape your labels.

The tool reads systems' results from a runs JSON (default
`artifacts/eval/runs.json`, override with `--runs`):

```json
{"bm25": {"q001": ["business_id_1", "business_id_2"]}}
```

When a new stage lands, run it, write its results into that file, and run
`make label` again: only the pairs nobody has graded yet are served
(**incremental pooling**). You will never be asked to re-grade a pair.

### You label a random 50 first

Labeling all ~150 queries at depth 20 is many hours, so
`evaluation.label_subset_size` (50) queries are served first, chosen **at
random** from a fixed seed — not the first 50 rows, not the quick ones, not the
interesting-looking ones. Every convenience-based choice biases the headline
metric in a specific direction (see `select_label_subset` in `pooling.py` for
the three ways it goes wrong). Raising the number later only adds queries: the
ones you have already labeled stay in the subset.

Every reported metric states how many queries it was computed over. "NDCG@10
= 0.61 over a random 50 of 150 gold queries" is an honest sentence. Dropping the
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
| `--runs`, `--businesses`, `--reviews`, `--tips` | override the input paths |
| `--retest`, `--agreement` | the consistency check, section 4 |

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
   graded, for a random 30 of your labeled queries, **without showing the
   previous grade**. Grade them fresh. Disagreeing with yourself is information,
   not a mistake, and nudging toward what you think you said last time destroys
   the measurement.
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

The same machinery is reused in Phase 3 for a different question: an LLM judge
grades a sample of the pairs you labeled, and the kappa between the LLM and you
is what justifies (or does not justify) using LLM labels in the training data.
