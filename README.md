# Concierge — natural-language place search

Concierge answers requests for a place the way you would ask a hotel concierge, not the way
you would phrase a keyword query. You type **"quiet café with wifi that's open past 10pm"**,
**"cheap ramen near downtown with outdoor seating"**, or **"somewhere good for a first date
that isn't loud"**, and it returns ten ranked places together with the part of the request
each one actually satisfies — and, just as importantly, the part it *cannot verify*, because
the underlying data is often silent. It is a four-stage retrieval-and-ranking cascade built
on the [Yelp Open Dataset](https://www.yelp.com/dataset): an LLM turns the request into
semantic text plus hard filters, hybrid lexical + dense retrieval produces a candidate set, a
fine-tuned cross-encoder reranks the shortlist, and a LambdaMART model makes the final
ordering by blending relevance with quality and constraint signals.

Two of the four stages are models trained here (the cross-encoder and the LambdaMART ranker).
The evaluation — a hand-labeled gold set with pooled graded judgements, bootstrap confidence
intervals, paired comparisons, and label-quality checks — is the part of this project I would
most want to be asked about.

> ### Status: the system is built; the numbers are not in yet
>
> **This README contains no results, and no estimated, simulated or illustrative stand-ins
> for results.** Every metric in it appears only after the gold set has been labeled by hand,
> and arrives by running `make eval`, which writes a markdown report, a machine-readable
> `results.json` and a latency figure. Sections below that will hold numbers are marked
> **PLACEHOLDER** and state the exact command that fills them. A number invented to make a
> table look finished would make every other number here worthless, so there are none.

---

## Architecture

```
                         user types a request
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ STAGE 1  Query understanding          LLM (Gemini Flash) → JSON     │
    │          "quiet café with wifi open past 10pm"                      │
    │            ├─ semantic_text: "quiet cafe with wifi"   (for models)  │
    │            └─ hard filters: NoiseLevel=quiet, WiFi=free,            │
    │                             open_at=(today, 22:00)     (arithmetic) │
    │          cache → LLM → rule fallback; never raises                  │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼        every business in the metro
    ┌─────────────────────────────────────────────────────────────────────┐
    │ STAGE 2  Hybrid retrieval                          → top ~200       │
    │          BM25 over business documents   (exact terms, rare words)   │
    │          + dense bi-encoder via FAISS   (meaning, paraphrase)       │
    │          fused by Reciprocal Rank Fusion, then hard filters         │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ STAGE 3  Cross-encoder reranker (fine-tuned here)  → top ~50        │
    │          reads (query, document) jointly in one forward pass;       │
    │          one transformer pass per candidate, so it can only run     │
    │          over a shortlist — never over the whole corpus             │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ STAGE 4  LambdaMART final ranker (trained here)    → top 10         │
    │          features: every upstream score and rank + stars +          │
    │          log(review_count) + constraints satisfied / violated /     │
    │          unknown + category match + distance                        │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            10 results + "why this result": satisfied / unmet / no data
```

**Why it is shaped like this.** Accuracy and cost move together, so the cascade spends its
compute where it buys the most ranking quality. The cheap methods run over everything: a BM25
query is a few sparse-column slices, and a dense query is one embedding plus an inner-product
scan over precomputed document vectors — both are affordable across the whole metro. The
cross-encoder is the opposite: it concatenates the query with a document and runs the
transformer over the pair, so nothing can be precomputed and the cost is one forward pass per
*(query, document)* pair. Scoring every business that way per query is out of the question;
scoring a 200-document shortlist is routine. Stage 4 is cheap again (a few hundred boosted
trees over ~50 feature rows), but it can only be cheap *because* it sits at the end, where it
has upstream scores to consume as features.

Each boundary trades recall for precision: Stage 2 maximizes the chance the right place is
*somewhere* in 200, and Stages 3 and 4 spend progressively more compute per candidate to get
it to rank 1. The counts 200 / 50 / 10 are config values (`candidates.retrieve`,
`candidates.rerank`, `candidates.final` in `configs/base.yaml`), not constants in the code,
and they are swept in the quality-vs-latency curve — the measurement that says whether
200 was the right place to draw the line.

---

## The four stages

### Stage 1 — Query understanding (`concierge/query/`)

Splits the request into **soft text** the models embed and score, and **hard filters** checked
arithmetically. The split is the whole point of the stage: an embedding will happily return a
café that closes at 6pm for "open past 10pm", because the prose about late-night studying
matches. Hours and price are facts, so they are checked as facts.

- **Three paths, in a fixed order: cache → LLM → rules.** The LLM is Gemini Flash (free
  tier), with Groq as a second provider, asked for strict JSON validated against a closed
  schema (`schema.py`): an unknown attribute key or an out-of-vocabulary value such as
  `NoiseLevel="pretty quiet"` is rejected rather than passed through, because a value that
  matches no business looks exactly like a ranking failure.
- **A parse failure never reaches the user.** A rate-limited API falls back to a regex parser
  (`rules.py`) that handles the genuinely regular constraints — hours, wifi, outdoor seating,
  noise, price, obvious category words. Which path answered is recorded on every
  `ParsedQuery` (`ParseSource.LLM / CACHE / RULES`), so the ablation can report LLM parsing
  as its own row and attribute both its quality *and* its latency honestly.
- **Key design decision:** the rule parser is not just a safety net, it is the baseline the
  LLM has to beat. "We used an LLM" is a claim; the gap between the `llm_parse` row and the
  rules row is a measurement.

### Stage 2 — Hybrid retrieval (`concierge/retrieval/`)

- **BM25** (`bm25s`) over one text document per business, and a **dense bi-encoder**
  (`BAAI/bge-small-en-v1.5`) behind a FAISS index. They fail in opposite directions: the
  embedding knows "ramen" and "noodle soup" are related but is weak on rare literal tokens
  ("birria", a neighbourhood name); BM25 is the reverse. Fusing them is cheaper than fixing
  either.
- **Reciprocal Rank Fusion**, `score = Σ 1/(k + rank)`, `k = 60`. BM25 scores are unbounded
  and corpus-dependent; cosine similarities sit in [-1, 1] and bunch near the top. Adding them
  requires a normalization choice that is itself an untuned hyperparameter, and min-max
  normalizing per query makes a document's score depend on which other documents happened to
  be retrieved. RRF uses only ranks, so it is invariant to both scales. `weighted_sum_scores`
  is implemented alongside it so the alternative can be measured rather than dismissed.
- **Exact FAISS first** (`IndexFlatIP` over L2-normalized vectors, so inner product *is*
  cosine). One metro is small enough that exact search is fast, and starting approximate would
  bake an unmeasured approximation error into every quality number downstream. HNSW is a later
  move, to be made only with the recall cost measured.
- **Hard constraints are three-valued, not boolean** (`filters.py`). `SATISFIED` /
  `VIOLATED` / `UNKNOWN`, because attribute coverage in this dataset is sparse: "this café has
  no `NoiseLevel` field" is a different statement from "this café is loud". Treating missing
  data as a failure throws away large numbers of good results; treating it as a pass fills the
  list with places that may well violate the request. Neither is defensible, so the policy is
  a config knob (`exclude` / `penalize` / `neutral`) whose effect gets measured, and `UNKNOWN`
  is surfaced in the UI as "no data" rather than quietly rounded either way.

### Stage 3 — Cross-encoder reranker, fine-tuned here (`concierge/rerank/`)

Initialized from `cross-encoder/ms-marco-MiniLM-L-6-v2` and fine-tuned on synthetic queries
(below) with pointwise binary cross-entropy — the same objective the checkpoint was pretrained
with on MS MARCO, so fine-tuning continues its objective instead of bolting a new head onto
pretrained weights.

- **Written as an explicit PyTorch loop** over `transformers`, not `CrossEncoder.fit()`.
  Forty more lines, and every one of them — labels, forward, loss, backward, clip, step,
  schedule, validate — is visible and defensible. It also removes a train/serve seam: training
  and serving use the same `AutoModelForSequenceClassification` class, so the checkpoint is
  loaded by the code path that saved it.
- **Key design decision:** the training signal is hard negatives mined from the *top of the
  hybrid retrieval list* for each synthetic query. A random business — a tyre shop for "quiet
  spot to read" — is separable on vocabulary alone, so the gradient is near zero and the step
  teaches nothing. At serving time the reranker only ever sees documents that already look
  right; that is the distribution it has to be trained on.
- The cost of that choice is admitted and then handled: some mined negatives are genuinely
  relevant (other quiet cafés with wifi do exist), so an LLM judge grades the hardest ones and
  likely false negatives are removed before training. How many were removed is a reported
  number, not a footnote.
- `model.py` also measures how much of each document actually fits inside `max_length` (512
  tokens). Document text past the window is simply not read by the cross-encoder — the last
  stage that reads document text at all — which puts a ceiling on quality *upstream of every
  model choice*.

### Stage 4 — LambdaMART final ranker, trained here (`concierge/ltr/`)

LightGBM with `objective="lambdarank"`, queries as groups. Stages 2 and 3 answer "does this
text match the request"; they say nothing about whether the place is any good, open, or
nearby. Stage 4 learns that blend from labels instead of a hand-written formula with hand-set
weights. Gradients are formed over pairs within a query and scaled by how much swapping the
pair would change NDCG, so the model spends capacity at the top of the list, where the metric
is sensitive.

- **Key design decision — the out-of-fold cross-encoder feature.** The cross-encoder's score
  should be one of the strongest features available — `make train-ltr` reports feature
  importance, which is where that stops being a hypothesis and becomes a measurement — and if
  LambdaMART trains on queries the cross-encoder was fine-tuned on, that feature is inflated
  exactly where the labels are. LambdaMART would learn "trust the cross-encoder completely",
  and on unseen queries that lesson is wrong — a
  model that looks excellent in training and mediocre on the gold set, with no single line of
  code to blame. Synthetic queries are therefore split by a hash of the query id into
  disjoint cross-encoder and LambdaMART folds, the manifest is checksummed to disk, and
  `assert_out_of_fold` / `verify_splits` make the trainers *fail* rather than warn
  (`ltr.enforce_oof_crossencoder_feature`). This is the subtlest correctness property in the
  system; it is explained in full in [docs/architecture.md](docs/architecture.md).
- **One feature function, used by both training and serving.** `build_feature_row` is a pure
  function of `(ParsedQuery, Candidate, Business)` and there is no second implementation, so
  training/serving feature skew — the failure where offline metrics stay fine while served
  quality quietly drops — has nowhere to live. Missing values are `NaN`, which LightGBM routes
  natively per split, rather than a sentinel `0` or `-1` that the trees would read as a real
  magnitude.
- Feature importance and an ablation that drops the non-text features are part of the
  deliverable, so "ratings and popularity matter" becomes a measurement instead of an
  assumption.

### Serving (`serve/`, `web/`) — current phase

The HTTP wire contract is defined in `serve/schemas.py`, deliberately separate from
`concierge/types.py`: the internal types are shaped for making the ablation traceable (frozen
dataclasses, a per-stage `scores` dict, three-valued constraint status), and a browser needs
none of that. The dependency runs one way — the API layer imports the pipeline, never the
reverse. Responses carry the parsed query, per-stage timings, and the three constraint buckets
(satisfied / unmet / no data) that drive "why this result". The FastAPI app, the load test, the
ONNX export's parity check, and the Next.js demo are the work in progress; `make serve` and
`make eval` are the ground truth for what currently runs end to end.

---

## Quickstart

### 1. Get the data yourself

**No dataset files are committed to this repository, and none ever will be.** The Yelp Open
Dataset is licensed for **educational and non-commercial use**, so you have to download it
yourself and accept its terms:

1. Go to **<https://www.yelp.com/dataset>** and download the dataset archive.
2. Extract `business.json`, `review.json`, `tip.json` into `data/raw/`.
3. `data/` is gitignored (`.gitignore` keeps only `data/.gitkeep`). Please keep it that way —
   and read the Yelp Dataset Terms of Use before publishing a demo that displays any of it.

`review.json` is several gigabytes of newline-delimited JSON. Every read in
`concierge/data/load.py` goes through DuckDB with explicit column projection and a semi-join
against the in-scope business ids, so the pipeline never materializes it.

### 2. Set up

```bash
make setup                 # uv sync --all-groups (fetches its own Python 3.12)
cp .env.example .env       # then paste a free Gemini key from aistudio.google.com/apikey
make test                  # unit tests; nothing here needs the dataset or an API key
```

Tests that need the real dataset are marked `needs_data` and tests that need a live API are
marked `needs_api`; both skip cleanly, so a fresh clone with no data and no keys still has a
green suite.

### 3. Build, train, evaluate

```bash
make explore               # dataset report: per-metro counts, attribute missingness
                           #   → then set data.metro in configs/base.yaml
make data                  # raw JSON → businesses.parquet + documents.parquet
make index                 # BM25 + FAISS indexes over the business documents
make train                 # train-reranker (cross-encoder) then train-ltr (LambdaMART)
make label                 # grade pooled results by hand — see eval/README.md
make eval                  # ablation table, per-type breakdown, latency curve, failures
make serve                 # FastAPI service on :8000
```

`make help` lists every target; `make lint`, `make fmt` and `make typecheck` are the quality
gates. `configs/base.yaml` holds every tunable, and `data.metro` deliberately ships as `null`:
`make data` refuses to run until you have looked at `make explore` and chosen a scope, so the
project can never silently index the wrong city.

**Order matters, and the tools say so.** Each command fails with an actionable message naming
what is missing rather than defaulting to something plausible. The one exception is the
pipeline itself, which degrades on purpose: with no fine-tuned checkpoint it uses the
pretrained cross-encoder, with no LambdaMART model it ranks by cross-encoder score, and it
*warns* about every degradation — because the pipeline has to be runnable at every point in
the project's own construction.

**One environment hazard worth knowing.** On macOS, the `faiss-cpu` and `torch` wheels each
bundle their own copy of LLVM's OpenMP runtime, and loading both into one process aborts the
interpreter at the C level with no Python traceback. `make index` detects the duplicate copies
and prints the diagnosis before starting a long build (see `duplicate_libomp_paths` in
`concierge/retrieval/build_index.py`). If something dies with a segfault or `OMP: Error #15`
instead of a Python error, check that first.

---

## Results

> ## PLACEHOLDER — no numbers exist yet
>
> Everything in this section is produced by **`make eval`**, which refuses to write a report
> until enough of the gold set is labeled (`eval/run_eval.py` raises `LabelsNotReady` below
> `MIN_LABELED_QUERIES`). Labeling is hand work and it is mine to do: `eval/gold_queries.csv`
> still contains only format examples, which every tool in the repo skips by `EXAMPLE` prefix
> so they can never reach a metric.
>
> Nothing here is estimated, simulated, or shown as an example value. The tables below are
> printed with their real structure and empty cells so the shape of the claim is visible
> before the claim is made.

Outputs land in `artifacts/eval/<config-fingerprint>/`: `report.md`, `results.json` (every
figure machine-readable at full precision, so no number is ever retyped by hand),
`latency_curve.csv` and `latency_curve.png`.

### Ablation ladder — `make eval`

One row per cascade configuration, defined once as data in `concierge/pipeline.py`
(`ABLATION`) so the table cannot drift from the code that produced it. Each metric arrives
with a bootstrap 95% CI, the number of queries it is defined on, and a paired-bootstrap
comparison against the row above it.

| # | System | NDCG@10 | Recall@100 | MRR | p95 latency |
|---|---|---|---|---|---|
| 1 | BM25 only | — not yet measured | — | — | — |
| 2 | + dense retrieval (RRF fusion) | — not yet measured | — | — | — |
| 3 | + cross-encoder (pretrained) | — not yet measured | — | — | — |
| 4 | + cross-encoder (fine-tuned here) | — not yet measured | — | — | — |
| 5 | + LLM query parsing | — not yet measured | — | — | — |
| 6 | + LambdaMART (full system) | — not yet measured | — | — | — |

Supplementary rows the same harness runs: the full system with hard filters switched off, and
dense-only retrieval. An em dash in any generated table means "not measured, for this stated
reason" — `eval/report.py` has no code path that renders a missing measurement as `0`.

### Per-query-type breakdown — `make eval`

The same metrics restricted to each of `vibe` / `constraint` / `specific` / `messy`. **Not yet
measured.** This breakdown is a headline result rather than a diagnostic: the stages should
help these four very differently — hard filters ought to earn their keep on `constraint`
queries and do nothing for `vibe`, and the fine-tuned cross-encoder is the stage expected to
matter most on `vibe` and `messy` — and a system that improves overall while getting *worse*
on one type is a finding the aggregate number hides.

### Quality vs. latency — `make eval`

NDCG@10 against p95 latency across `candidates.sweep_retrieve` × `candidates.sweep_rerank`
(the 200/50 defaults are one point on this surface). **Not yet measured.** Writes
`latency_curve.csv` and `latency_curve.png`; points with a missing coordinate are dropped
rather than drawn at zero.

### Failure analysis — `make eval`

For the worst queries per type, the harness separates the two failure modes that look
identical from the top of the result list: **the good result was never retrieved** (a Stage 2
recall failure, which no amount of reranking can fix) versus **it was retrieved and then
mis-ranked** (a Stage 3/4 failure). **Not yet measured.**

### Label quality — `eval/label_cli.py --agreement`, and the LLM judge

Three numbers that exist to show the labels can be trusted, none of them measured yet:

- **Test-retest self-agreement.** Exact agreement, Cohen's kappa both unweighted *and*
  quadratic-weighted (the weighted figure is the flattering one on a 3-point ordinal scale, so
  it is never quoted alone), and the confusion matrix, over pairs I graded twice at least two
  weeks apart.
- **LLM-judge vs. my labels.** The same kappa and matrix between the judge and me on a sample
  of pairs I labeled by hand. This is the entire justification for using LLM grades to filter
  training data; if it comes out low, the honest conclusion is to drop that filtering step,
  and that is what will be reported.
- **Leakage and filtering counts.** How many synthetic training queries were removed as
  duplicates of gold queries (exact and by embedding similarity), and how many mined hard
  negatives the judge removed as false negatives.

### Serving latency

p50/p95 per stage and end to end, under load, plus the PyTorch-vs-ONNX parity check and
backend timing from `python -m concierge.rerank.onnx_export`. **Not yet measured.**
`serving.use_onnx` stays `false` until a measurement on the serving host says ONNX is actually
faster there — it is not always, and the export script prints what it measured rather than
assuming.

---

## Evaluation methodology

This is the part of the project I would defend first, because it is what separates a measured
search system from a demo that looks good on the queries its author happened to try.

**Gold queries.** ~150 queries I write myself, before looking at any results, spread across
four types: `vibe` ("cozy spot to read on a rainy afternoon"), `constraint` ("open after 10pm
with outdoor seating"), `specific` ("good pho"), and `messy` — typos, vagueness, two intents
at once. Writing a query after noticing a nice result is the cheapest way to accidentally fake
a good number, so the order is fixed. No code in this repository writes or edits gold queries;
`eval/README.md` is the full brief, including what makes a query worth testing.

**Pooled graded labels.** 150 queries × a whole metro cannot be judged exhaustively, so the
harness pools: take the top 20 from every system being compared, union them, and grade only
that (TREC's method since 1992). Grades are 0 irrelevant / 1 partially relevant / 2 highly
relevant. Pooling is biased in a known direction — a new system that surfaces good documents
nobody pooled looks worse than it is — so `eval/metrics.py` reports the **unjudged fraction
next to every metric** instead of hiding the assumption. Incremental pooling means a new stage
only costs the newly surfaced pairs, and an already-graded pair is never served again (two
labels for one pair would make every metric depend on which row won).

**A random labeled subset, stated with every number.** Labeling 150 queries at pool depth 20
is many hours of hand work, so a **random, fixed-seed** 50-query subset is labeled first and expands
later. Random is the load-bearing word: the first 50, the short ones, or the
interesting-looking ones each bias the headline metric in a specific direction. Every reported
metric states how many queries it was computed over — *"NDCG@10 = x over a random 50 of 150
gold queries"* is an honest sentence and the clause after the comma is not optional.

**Metrics, with their uncertainty.** NDCG@10 primary, Recall@100 for the retrieval stages, MRR.
Every mean carries a bootstrap 95% CI, and consecutive systems are compared with a **paired**
bootstrap over per-query differences, which removes query difficulty from the comparison. With
~50–150 queries, small gaps are noise, and the harness says so in words: a comparison whose
interval crosses zero is reported as not distinguishable, and that sentence is carried verbatim
into the rendered report so it cannot be lost in formatting. A query with no relevant document
has *no* NDCG — not zero — so it is excluded from the mean and counted in the open.

**Self-agreement, because there is one annotator.** "I was careful" is not evidence. The only
second rater available is me, later: `label_cli.py --retest` re-serves pairs I already graded
(for a random subset of queries, without showing the old grade) after a two-week gap, and
`--agreement` prints exact agreement, both kappas, and the confusion matrix. The *shape* of the
disagreement is the interesting part — 1-vs-2 confusion means my middle category is fuzzy,
while 0-vs-2 confusion means the grading rules themselves are broken. Reported numbers are
reported as numbers; kappa interpretation bands are conventions, not thresholds, so no number
here gets translated into a flattering adjective.

**LLM-vs-owner agreement, as a licence to use LLM labels.** The same judge, prompt and rubric
that filter false negatives from the training data are run over a sample of pairs I graded by
hand, and the kappa between us is the justification for using LLM grades at all. The judge is
**read-only with respect to `eval/qrels.csv`** — only my labeling CLI writes labels, because a
judge that could write them would make the evaluation circular.

**Leakage rules, enforced in code and not by discipline.**

- Gold queries and labels are **test-only**: never in training data, prompt examples,
  hyperparameter tuning, or feature engineering. Tuning happens on a synthetic validation
  split.
- Synthetic training queries are deduplicated against the gold set **exactly and by embedding
  similarity** (cosine ≥ 0.92), the guard **fails loudly if the gold file is missing** rather
  than passing everything through, and the number removed is a required reported figure.
- Both trainers call `assert_no_gold_leakage` and refuse to run on a match, by query id or by
  normalized text.
- The cross-encoder and LambdaMART folds are disjoint by construction, checksummed, and
  verified at train time (see the out-of-fold discussion above).
- Splits are assigned by **hash of the query id**, not a seeded shuffle, so generating more
  synthetic data later cannot migrate an existing query from one model's fold into the
  other's — the leakage a reshuffle would introduce silently.

**Reproducibility.** Fixed seeds; every tunable in `configs/base.yaml`; and every index,
checkpoint, split manifest and eval report written under a directory keyed by a hash of the
whole config, so any figure can be traced back to the exact settings that produced it. Config
parsing forbids unknown keys, because a silently-ignored typo means an experiment that did not
test what it claimed to.

---

## Limitations

Stated plainly, because each of these bounds what the numbers will be able to claim:

- **One annotator — me.** Every relevance judgement in this project comes from one person, so
  "relevance" here means my judgement applied consistently, not a consensus. The test-retest
  kappa measures the consistency; it cannot measure whether my standard matches anyone else's.
  A second annotator on even a slice of the pool would be the single biggest upgrade available.
- **Synthetic training queries.** Both trained models learn from queries an LLM wrote from a
  business document, not from a query log — this is a cold-start search engine with no users.
  LLM-written queries are plausibly cleaner, better spelled, and more
  single-intent than real ones, which likely makes the `messy` query type the weakest part of
  the system and is a specific thing to look for in the per-type breakdown. The gold set is
  mine, so the *measurement* is not synthetic, but the training distribution is.
- **One metro area.** Scope is a single metro chosen on evidence from `make explore`. The
  pipeline is metro-agnostic (changing scope is a config change plus a rebuild), but nothing
  here demonstrates that quality transfers to a city with different category mix, different
  attribute coverage, or different review density.
- **No online evaluation.** There are no users, no clicks, and no A/B test. Every number is
  offline relevance against my labels, which cannot capture whether people would actually be
  satisfied by these results. The next section is how I would measure that, framed as design
  reasoning rather than results, because that is all it can be here.
- **Attribute sparsity in the source data.** Many businesses carry no `NoiseLevel`, no `WiFi`,
  and sometimes no hours at all. The system handles this honestly (three-valued constraints,
  "no data" in the UI, a measured missing-data policy) but honesty is not coverage: for a
  constraint the dataset is silent about, no ranker can verify the user's request. The
  `make explore` missingness report is the first thing to read before trusting any
  constraint-query result, and it bounds how many gold queries are answerable *in principle*.
- **A snapshot, not a live index.** The dataset is a static export; hours and attributes are
  as of its release, so "open past 10pm" is answered against the data's claim, not reality.
- **Known-unknowns deliberately left as config knobs.** The missing-attribute penalty (0.5),
  RRF's `k = 60`, the document length cap, and the 200/50/10 depths are starting points from
  the literature or from reasoning, not tuned results. `DECISIONS.md` marks each of them
  **Provisional** and names the experiment that would settle it. None of them is presented as
  a finding.

---

## How I would evaluate this online

The project has no users, so none of this is a result — it is the design I would propose if it
shipped, and it is here because "how would you measure this in production?" is a near-certain
interview question and offline NDCG is not a sufficient answer.

**Interleaving before A/B tests.** For ranking changes, I would start with team-draft
interleaving: for each query, build one list by alternately letting the two rankers draft their
next-best unused result, then attribute each click to the ranker that contributed that
document. Every user sees one blended list, so the comparison is *within* user and within
query, which removes almost all of the variance that makes ranking A/B tests need enormous
traffic. The interleaving literature's central claim is that it detects ranking differences
with far less traffic than an A/B test on a session metric — I have measured nothing of the
kind here, and I would want to verify the sensitivity on our own logs before relying on it,
but it is the reason to reach for interleaving first on a low-traffic product. Its limits are
real too: it measures *preference between two
rankers*, not the absolute quality of either, it cannot evaluate changes to the result card or
the number of results shown, and clicks on a blended list are not evidence about how a user
would behave on a page they would actually see — so I would use it to choose between rankers
and an A/B test to decide whether the change is worth shipping.

**Position bias is the main threat to any click-based conclusion.** Users click the top result
because it is on top, so raw click-through rate ranks the ranker's own past decisions rather
than relevance. Two complementary handles:

- *Randomization to estimate the bias.* Swap adjacent pairs, or shuffle within a small window,
  on a small traffic slice, and compare click rates for the same document at different
  positions. That gives a position propensity curve.
- *Inverse-propensity weighting to correct for it.* Weight each observed click by 1/propensity
  for its position, which makes the resulting estimator of relevance unbiased under the
  examination assumption (a document is clicked only if examined, and examination depends on
  position). The price is variance at low propensities, which means clipping the weights and
  being honest that clipping reintroduces some bias.

**Click models as noisy relevance signals, never as labels.** A cascade model (the user scans
top-down and stops at a click) or a dynamic Bayesian network (separating attractiveness —
was the snippet appealing — from satisfaction — did they come back) can turn click logs into
per-document relevance estimates good enough to train a ranker on. Three cautions I would hold
onto: these estimates are *downward biased for anything the current system ranks low*, because
documents that are never shown are never examined and therefore never clicked, so training on
them alone makes the system progressively more confident in its own earlier choices; a click
is attraction, not satisfaction, so a dwell-time or return-click threshold is a better target
than a raw click; and nothing stops a click model from encoding the biases of whoever wrote the
snippet. I would keep a small human-labeled set — the same kind of gold set as here — as the
calibration anchor that click-derived labels get checked against.

**Metrics I would commit to in advance.** A primary that reflects the product (successful
sessions: a click with sufficient dwell, or a tap through to directions or a phone number),
plus guardrails that catch the failure modes a ranking change causes: query reformulation rate,
abandonment, zero-result rate, and p95 latency — because this cascade can buy quality with
latency, and a quality win that costs 300ms may be a product loss. Rankings for constraint
queries would get a dedicated guardrail on "hard constraint violated in the top 3", since that
is the error users punish hardest and an aggregate relevance metric barely notices it.

**What would make me distrust a win.** A result that appears only in the aggregate and not in
any query segment; a win that coincides with a latency regression; a win driven by a handful
of head queries; and anything that moves before the fixed horizon of a sequentially-monitored
test, where repeated peeking inflates false positives unless the test is designed for it.

---

## Repository layout

```
concierge/
  types.py          shared contracts every stage speaks (frozen dataclasses)
  config.py         typed pydantic config; artifacts keyed by its hash
  data/             loading (DuckDB, streaming), attribute + hours parsing, documents
  query/            Stage 1: LLM client, JSON schema, SQLite parse cache, rule fallback
  retrieval/        Stage 2: BM25, dense + FAISS, RRF fusion, three-valued hard filters
  rerank/           Stage 3: training loop, inference, ONNX export + parity check
  ltr/              Stage 4: pure feature function, LambdaMART training
  synth/            synthetic queries, hard negatives, LLM judge, fold assignment
  pipeline.py       the four stages wired into one callable, with ablation variants
eval/               gold queries, labeling CLI, metrics + statistics, make eval harness
serve/              HTTP wire contract (FastAPI app in progress)
web/                Next.js demo frontend (in progress)
configs/base.yaml   every tunable
tests/              unit tests, including fixtures covering the dataset's quirks
```

Two companion documents:

- **[docs/architecture.md](docs/architecture.md)** — data flow end to end, the
  artifact/config-hash scheme, where each trained model comes from, and a careful treatment of
  the out-of-fold cross-encoder feature requirement.
- **[DECISIONS.md](DECISIONS.md)** — every meaningful choice with the alternatives considered,
  the reason, and how it could be tested. Decisions are marked **Settled**, **Provisional** (a
  default awaiting a measurement) or **Deferred**, so nothing chosen by reasoning alone can
  pass itself off as a result.
- **[eval/README.md](eval/README.md)** — how the gold set is written and labeled, including
  the grading rubric for the hard middle case.

## Data and licensing

Built on the **Yelp Open Dataset**, which is licensed for educational and non-commercial use
and must be obtained directly from <https://www.yelp.com/dataset>. No dataset file is
committed here, in any form, derived or otherwise. This project is not affiliated with,
endorsed by, or branded as Yelp; the dataset is an input, not the product. Anyone deploying a
public demo of this code should read the Yelp Dataset Terms of Use first to confirm what may
be displayed.
