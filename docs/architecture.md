# Architecture

The technical companion to the [README](../README.md). The README says what the system does
and how it is evaluated; this document says how the pieces fit, where every artifact comes
from, and — in the last and longest section — why the out-of-fold cross-encoder feature is the
subtlest correctness property in the whole project.

Nothing here contains a measurement. Where a number would belong, this document names the
command that produces it.

**Contents**

1. [Data flow, end to end](#1-data-flow-end-to-end)
2. [The shared contracts, and why they are frozen](#2-the-shared-contracts-and-why-they-are-frozen)
3. [A query's path through the cascade](#3-a-querys-path-through-the-cascade)
4. [Artifacts and the config-hash scheme](#4-artifacts-and-the-config-hash-scheme)
5. [Where each trained model comes from](#5-where-each-trained-model-comes-from)
6. [The out-of-fold cross-encoder feature](#6-the-out-of-fold-cross-encoder-feature)
7. [Failure modes this design is built to prevent](#7-failure-modes-this-design-is-built-to-prevent)

---

## 1. Data flow, end to end

Two pipelines meet at the business document: an **offline** one that turns raw dataset files
into indexes and trained models, and an **online** one that answers a query. They share the
document builder and the feature function, which is deliberate — those are the two places
where offline and online could drift apart and produce numbers that do not describe the served
system.

```
  data/raw/business.json ──┐
  data/raw/review.json  ──┼─► concierge.data.load  (DuckDB, streaming, projected columns)
  data/raw/tip.json     ──┘        │
                                   │  metro scope + is_open filter
                                   ▼
                        concierge.data.attributes   "u'quiet'" → {"NoiseLevel": "quiet"}
                        concierge.data.hours        "17:0-2:0" → interval crossing midnight
                                   │
                                   ▼
                        concierge.data.documents    one text document per business
                                   │
            ┌──────────────────────┴───────────────────────┐
            ▼                                              ▼
   data/processed/businesses.parquet              data/processed/documents.parquet
   (facts: the filters read this)                 (text: every model reads this)
            │                                              │
            │                     ┌────────────────────────┼────────────────────┐
            │                     ▼                        ▼                    ▼
            │            artifacts/bm25/<fp>      artifacts/dense/<fp>   concierge.synth
            │            (sparse index)           (FAISS + meta)         (LLM writes queries
            │                     │                        │              from documents)
            │                     │                        │                    │
            │                     │                        │          artifacts/synth/<fp>
            │                     │                        │          queries, negatives,
            │                     │                        │          splits.json
            │                     │                        │                    │
            │                     │                        │          ┌─────────┴────────┐
            │                     │                        │          ▼                  ▼
            │                     │                        │   artifacts/rerank/<fp>  artifacts/ltr/<fp>
            │                     │                        │   (fine-tuned CE)        (LambdaMART)
            └─────────────────────┴────────────────────────┴──────────┴──────────────────┘
                                             │
                                             ▼
                              concierge.pipeline.SearchPipeline
                                   (loads everything once)
                                             │
                        ┌────────────────────┴───────────────────┐
                        ▼                                        ▼
                eval.run_eval  (make eval)               serve/  →  web/
                artifacts/eval/<fp>/report.md             HTTP + demo UI
```

`<fp>` is the config fingerprint — see [section 4](#4-artifacts-and-the-config-hash-scheme).

**Why DuckDB rather than pandas or plain `json.loads`.** `review.json` is several gigabytes of
newline-delimited JSON — millions of rows — and the pipeline needs four fields of it
(`business_id`, `text`, `useful`, `date`) for the businesses in one metro. A naive
read builds a Python object for every field of every review — tens of gigabytes of objects to
reach the four fields the pipeline reads. How much that saves is a byte share nothing here has
measured (`make explore --count-reviews` reports the real row counts); the mechanism is the
reason for the call. Every read in `concierge/data/load.py` therefore pushes
the projection *and* the business-id semi-join down into DuckDB, and streams results in
batches, so peak memory is set by the batch size rather than by the file.

**Why the data build has two outputs rather than one.** `businesses.parquet` holds facts
(stars, review count, categories, normalized attributes, parsed hours, coordinates) and is what
the hard filters read arithmetically. `documents.parquet` holds the rendered text and is what
BM25, the bi-encoder and the cross-encoder read. They are separate because document rendering
is an *experimental variable* — which reviews to include, how attributes are phrased, where to
truncate — and a change to it must be able to invalidate the indexes without invalidating the
parsed facts.

**The three dataset quirks that are handled once, at the source.**

| Quirk | Where | Why it matters |
| --- | --- | --- |
| Attributes are stringified Python literals: `"u'free'"`, `"{'casual': True}"` | `data/attributes.py`, via `ast.literal_eval` (never `eval` — this is untrusted third-party text) | Every stage reads attributes; parsing them in more than one place guarantees two normalizations that disagree |
| Hours are unpadded strings, some crossing midnight: `"17:0-2:0"`, `"0:0-0:0"` = open 24h, `"17:0-0:0"` = closes at midnight | `data/hours.py`, `HoursInterval.covers()` | "Open late" is a large share of the gold set; treating the `0` in `"17:0-0:0"` as a midnight crossing would claim the place is open at 1am |
| Attribute coverage is sparse and uneven between cities | measured by `make explore`; handled by three-valued `ConstraintStatus` everywhere | A missing `NoiseLevel` is not a loud café. Collapsing the two is the most common way this dataset produces confidently wrong results |

---

## 2. The shared contracts, and why they are frozen

`concierge/types.py` defines the vocabulary every stage speaks: `Business`,
`BusinessDocument`, `ParsedQuery`, `Candidate`, `ConstraintMatch`, `SearchResult`,
`SearchResponse`. Three properties of that module do real work:

- **Frozen dataclasses.** Stages return copies rather than mutating in place
  (`Candidate.with_score`). A reranker that could quietly overwrite a retrieval score would
  make the ablation untraceable, and the ablation is the project's main artifact.
- **Scores and ranks are dictionaries keyed by stage**, not one collapsed number. Stage 4
  consumes *every* upstream score and rank as a feature, so nothing may be discarded between
  stages. `SCORE_BM25`, `SCORE_DENSE`, `SCORE_RRF`, `SCORE_CROSS_ENCODER`, `SCORE_LTR` are
  defined once as constants so the feature builder can rely on the names.
- **Constraint outcomes are three-valued.** `SATISFIED` / `VIOLATED` / `UNKNOWN` travels from
  the filter, through the Stage 4 features (which count each bucket separately), into the API
  response and the "why this result" UI, without ever being flattened to a boolean.

The HTTP layer deliberately does **not** reuse these types: `serve/schemas.py` defines its own
pydantic views, and the dependency runs one way only (the API imports the pipeline; nothing in
`concierge/` imports `serve/`). Internal types are shaped for traceable ablations; a wire
format is a public contract. Serializing the internals would mean a field rename inside the
pipeline is a breaking API change, and would couple the frontend to the reranker's score-key
constants.

---

## 3. A query's path through the cascade

`SearchPipeline.search(query, variant=..., k=...)` returns a `SearchResponse` carrying the
parsed query, the results, per-stage timings and per-stage candidate counts. Latency is
measured on **every** search, not behind a debug flag, because p50/p95 per stage is a reported
result and a measurement that only happens when someone remembers to ask for it is a
measurement that ends up missing.

1. **Parse** (`concierge.query.parser.parse`) — cache → LLM → rules, in that fixed order, and
   it never raises. The returned `ParsedQuery` records which path answered (`ParseSource`) and
   what the parse cost in milliseconds, so the ablation can attribute both quality and latency.
   `semantic_text` is the only string the text models see; `open_at`, `attributes`,
   `price_max`, `categories` and `geo` are checked arithmetically.
2. **Retrieve** — BM25 and the dense index each return `(business_id, score)` lists through
   the *same* `search(query_text, top_k)` shape, so `fusion.fuse` never needs to know which
   retriever it is talking to. RRF combines them by rank into `Candidate` objects carrying
   `SCORE_BM25`, `SCORE_DENSE`, `SCORE_RRF` and the matching ranks.
3. **Filter** (`retrieval.filters.apply_filters`) — each candidate's constraints are evaluated
   into `ConstraintMatch` tuples, and the configured missing-data policy
   (`exclude` / `penalize` / `neutral`) is applied. The `FilterReport` records *why* each
   dropped candidate was dropped, which is what makes a query that returns nothing diagnosable
   rather than mysterious.
4. **Rerank** (`rerank.model.CrossEncoderReranker`) — the shortlist is scored jointly,
   stably re-sorted, and truncated to `candidates.rerank`. The pipeline delegates the whole of
   Stage 3 to this object rather than reimplementing the sort, so there is only one definition
   of the stage's ordering.
5. **Final ranking** (`ltr` → `LightGbmRanker`) — one feature row per candidate from the
   single shared feature function, one booster prediction, truncate to `candidates.final`.
6. **Explain** — each `SearchResult` carries its constraint matches split into satisfied,
   violated and unknown, plus every stage score, for the UI and for debugging one query.

**The ablation is the same code path, parameterized.** `Stage` is a flag enum
(`QUERY_LLM | BM25 | DENSE | FILTERS | RERANK | LTR`) and a `Variant` is a named combination of
flags plus which cross-encoder checkpoint to use. The six ablation rows are *data*
(`pipeline.ABLATION`), so the published table cannot drift from the code that produced it, and
"the full system minus hard filters" is an expression (`VARIANT_FULL.stages & ~Stage.FILTERS`)
rather than a second code path that could diverge. `Variant.__post_init__` rejects a
configuration with no retriever at all, because such a row would produce a plausible-looking
set of zeros instead of an error.

**Missing models degrade, they do not crash.** The pipeline must be runnable at every point in
the project's own construction: before the dense index exists, before the cross-encoder is
fine-tuned, before LambdaMART is trained. So a missing artifact produces a warning naming the
artifact and a shorter cascade (fine-tuned → pretrained → no reranking; LambdaMART →
cross-encoder order). The warning matters as much as the fallback: a silent degradation would
surface only as a mysteriously weak ablation row.

---

## 4. Artifacts and the config-hash scheme

`configs/base.yaml` is loaded into a pydantic model with `extra="forbid"` and `frozen=True`.
Two consequences, both deliberate:

- **A typo is a startup error, not a shrug.** A silently-ignored config key means an experiment
  that did not test what it claimed to test — the worst possible failure for a project whose
  value is its measurements, because it produces a number that looks fine and is wrong.
- **`Config.fingerprint()` is the artifact namespace.** It is a SHA-256 over the whole config
  serialized with sorted keys, truncated to 10 hex characters, and every generated artifact
  lives under a directory keyed by it:

| Artifact | Path | Written by |
| --- | --- | --- |
| Processed businesses + documents | `data/processed/*.parquet`, `build_report.json` | `make data` |
| Exploration report | `artifacts/explore/` | `make explore` |
| BM25 index | `artifacts/bm25/<fp>/` | `make index` |
| Dense index + `DenseIndexMeta` | `artifacts/dense/<fp>/` | `make index` |
| Synthetic queries, raw LLM responses, fold manifest | `artifacts/synth/<fp>/` | `concierge.synth.generate` |
| Fine-tuned cross-encoder + `RunMetadata` | `artifacts/rerank/<fp>/` | `make train-reranker` |
| LambdaMART booster + `feature_names.json` | `artifacts/ltr/<fp>/`, `…-textonly/` for the ablation | `make train-ltr` |
| Eval report, `results.json`, latency CSV + figure | `artifacts/eval/<fp>/` | `make eval` |
| LLM parse cache | `.llm_cache/query_parse.sqlite` (not fingerprinted; see below) | Stage 1, at query time |

Why this is worth the small inconvenience: **any figure can be traced back to the exact
settings that produced it**, and changing the embedding model, the query prefix, the document
length cap or the FAISS index type cannot quietly reuse a stale index. That is the mechanism
behind the project's reproducibility claim — not a promise to be careful.

Three refinements the scheme needs in practice:

- **The fingerprint is coarse on purpose.** It hashes the *whole* config, so changing an
  unrelated knob gives every artifact a new path and forces rebuilds that are not strictly
  necessary. The alternative — a per-artifact hash over only its relevant subtree — is more
  efficient and much easier to get subtly wrong, and a stale-artifact bug is far more expensive
  than a rebuild. Coarse and obviously correct wins.
- **Hashes cannot catch everything, so artifacts also carry their own metadata.**
  `DenseIndexMeta` stores the model name, index type, dimension, document count, normalization
  flag, query prefix and fingerprint next to the FAISS file, and a mismatched load fails
  loudly. This matters because many small embedding models are 384-dimensional: an index built
  by a different model can load without error and return confident garbage.
- **The parse cache is keyed separately**, by `(provider, model, prompt_version, normalized
  query)`, where `prompt_version` is the declared version plus a hash of the prompt body and
  the attribute vocabulary. It is not under the config fingerprint because an unrelated config
  change must not throw away a cache whose answers are still valid — but *editing the prompt*
  must, and the prompt hash is what ensures it does. `make clean` removes `artifacts/` and
  `data/processed/`, and deliberately leaves `data/raw/` and the hand-made label files alone.

---

## 5. Where each trained model comes from

Two models are trained here. Both learn from **synthetic queries**, because this is a
cold-start search engine: there is no query log, so the problem is inverted — show a language
model one business document and ask what a person would type to find *that* place. The source
business is then the positive label for free, which is the only reason a few thousand labeled
training pairs are obtainable at zero cost.

```
documents.parquet
      │
      │  concierge.synth.generate — batched LLM calls, raw responses appended to
      │  disk as they arrive (a 429 mid-run must not cost the completed batches)
      ▼
synthetic queries  ──►  dedupe_against_gold  ──►  kept queries
      │                 exact + embedding cosine ≥ 0.92, against eval/gold_queries.csv
      │                 fails loudly if the gold file is missing
      ▼
concierge.synth.negatives — hard negatives = top hybrid-retrieval results for that
      │  query that are not the source business
      ▼
concierge.synth.judge — LLM judge grades the hardest negatives; likely false
      │  negatives (other genuinely relevant places) are removed before training
      ▼
concierge.synth.splits — hash-based fold assignment, checksummed to splits.json
      │
      ├── crossencoder_train / crossencoder_val  ──►  rerank.train   (Stage 3)
      └── ltr_train / ltr_val                    ──►  ltr.train      (Stage 4)
```

**Stage 3, the cross-encoder.** Initialized from `cross-encoder/ms-marco-MiniLM-L-6-v2` and
fine-tuned with pointwise binary cross-entropy over a single logit: the source business is
label 1, each surviving hard negative is label 0. That matches the head the checkpoint already
has — it was pretrained exactly this way on MS MARCO — so fine-tuning continues an objective
rather than bolting a new one onto pretrained weights. The loop is explicit `transformers`
code (labels, forward, loss, backward, clip, step, schedule, validate) rather than
`CrossEncoder.fit()`, both because every line has to be defensible and because training and
serving then use the same `AutoModelForSequenceClassification` class — no train/serve seam at
the checkpoint boundary. Every run writes a `RunMetadata` record: base model, config
fingerprint, seed, device, hyperparameters, split names, example counts, pair-build statistics,
the loss curve, and the result of the gold-leakage check. A run whose metadata lacks the
leakage check is a run whose numbers cannot be used, so the field is not optional. The metadata
also carries a standing note that loss values are training diagnostics only: retrieval quality
is measured by `eval/run_eval.py` and is never inferred from a loss curve.

**Stage 4, LambdaMART.** LightGBM with `objective="lambdarank"` and queries as groups. Features
come from `concierge/ltr/features.py` — one pure function of `(ParsedQuery, Candidate,
Business)`, called by both the training job and the serving path, because
**training/serving feature skew** is the classic way a ranking system goes quietly wrong:
offline the feature is computed one way, online another, nothing crashes, offline metrics look
fine, and served quality is worse than the numbers claim with no visible discrepancy. One
implementation means a wrong feature is wrong identically in both places — a bug you can find
rather than a gap you cannot see. Missing values are `NaN`, which LightGBM routes natively per
split, rather than `0` or `-1`, which the trees would read as real magnitudes.

`save_ranker` writes the feature order **twice** — inside the model file and in
`feature_names.json` — specifically so `pipeline.check_feature_order` can compare them at load
time. A permuted feature matrix produces plausible predictions and no error whatsoever, which
makes it the most dangerous silent bug available in Stage 4.

---

## 6. The out-of-fold cross-encoder feature

This is the subtlest correctness property in the system. It is worth reading slowly, because
the failure it prevents cannot be detected after the fact from any metric.

### The setup

The cross-encoder's score is expected to be one of the strongest features LambdaMART has — an
expectation, not a result: `make train-ltr` reports feature importance, and until it runs
nobody here knows the ordering. The expectation is at least reasonable, because that score is
the output of a transformer that read the query and the document together, while the other
features are a star rating, a review count, some constraint counts. If the cross-encoder's
score were a perfect relevance oracle, the optimal LambdaMART model would be "sort by the
cross-encoder score and ignore everything else".

Note that the leakage argument below does not actually need the feature to be *the* strongest
one. It needs only that LambdaMART is free to lean on it heavily, which the model decides for
itself from the data it is given. So the fix stands whatever `make train-ltr` ends up
reporting.

### The failure

Suppose LambdaMART is trained on the same synthetic queries the cross-encoder was fine-tuned
on. For those queries the cross-encoder has partially *memorized* the right answer: it has
seen this exact query paired with this exact positive document, with a label, and it was
optimized to score that pair high. So on the training rows:

- the cross-encoder feature is **inflated exactly where the labels are positive**;
- it looks almost perfectly predictive of the label — not because it generalizes, but because
  both the feature and the label derive from the same memorized example;
- LambdaMART, doing precisely its job, learns to put almost all its weight on that feature and
  to ignore stars, review count, and the constraint signals that were the entire reason for
  adding Stage 4;
- at serving time the query is new, the cross-encoder's score is merely *good* rather than
  memorized, and the learned "trust it completely" rule is wrong.

The result is a model that looks excellent during training and mediocre on the gold set, with
**no single line of code to blame**. Nothing raises. Training NDCG is high. The feature
importances look reasonable ("of course the cross-encoder matters most"). The only symptom is a
final stage that fails to add anything, which is indistinguishable from "Stage 4 was simply not
a good idea" — so the natural reaction is to delete the stage instead of fixing the leak.

This is the ranking-specific form of a general rule: **a feature produced by a model must be
computed on data that model did not train on.** It is the same reasoning behind out-of-fold
predictions in stacked ensembles, and the same reason a target-encoded categorical computed on
the full training set destroys a gradient-boosting model.

### The fix, and why it is code rather than discipline

The synthetic query set is partitioned into four folds — `crossencoder_train`,
`crossencoder_val`, `ltr_train`, `ltr_val` — and the cross-encoder and LambdaMART halves are
**disjoint by construction**. Every cross-encoder score that becomes a LambdaMART training
feature is therefore produced for a query the cross-encoder never saw, which is exactly the
condition the feature will be in at serving time.

`configs/base.yaml` states the requirement (`ltr.enforce_oof_crossencoder_feature: true`), and
three functions enforce it:

- **`splits.assert_out_of_fold(manifest)`** — no query id appears in both a cross-encoder split
  and a LambdaMART split. This proves the *plan* is sound.
- **`splits.assert_models_trained_on_disjoint_queries(ce_ids, ltr_ids, manifest)`** — checks the
  ids a trainer *actually used* against the manifest. This proves the *run followed the plan*,
  which is a different claim: a model trained before the manifest was regenerated is exactly
  how an out-of-fold promise quietly stops being true, and this is the check that catches it.
- **`splits.verify_splits(manifest, cfg=...)`** — the full invariant set in one place, called
  by the trainers, which **raise** rather than warn. A warning in a log is not a guarantee; a
  refusal to train is.

### Why folds are assigned by hash, not by a seeded shuffle

Each query's fold is `hash(seed, query_id)` bucketed by the configured fractions, so a query's
fold depends **only on its own id** — not on how many other queries exist, and not on their
order.

The alternative, `random.shuffle(queries)` with a fixed seed and then four slices, is wrong in
a way that is easy to miss: generating more synthetic data later reshuffles the whole list, and
a query that was in `crossencoder_train` for the trained checkpoint can land in `ltr_train` for
the next LambdaMART run. The cross-encoder checkpoint on disk has still seen it. The manifest
says it is a LambdaMART query. The out-of-fold guarantee is now false, nothing downstream
notices, and the system has silently returned to the failure mode above. Hash assignment makes
that impossible: adding data can only add new ids to folds, never move existing ones.

The cost is honest and small: fold sizes are only approximately the configured fractions
(binomial scatter around them), so the manifest records the **realised** counts and fractions
rather than restating the nominal ones. A manifest also carries a checksum over
`(version, seed, group_by, fractions, assignment)`, and a manifest that does not match its own
checksum is treated as corrupt — because a hand-edited split file would be an undetectable way
to produce an inflated result.

### The second leakage boundary: the gold set

Folds keep the two *trained models* independent of each other. A separate guard keeps both of
them independent of the **test set**:

- `synth.generate.dedupe_against_gold` removes synthetic queries that duplicate a gold query,
  exactly and by embedding cosine ≥ `synth.gold_dedup_cosine_threshold` (0.92), using the same
  embedding model Stage 2 retrieves with. It raises if `eval/gold_queries.csv` is absent rather
  than passing everything through — a missing gold file must not be the thing that silently
  disables the leakage guard.
- `rerank.train.assert_no_gold_leakage` and `ltr.train.assert_no_gold_leakage` independently
  refuse to train if any training query matches a gold query by id or by normalized text.
- The count removed by dedup is a **required reported figure**, not an implementation detail,
  because the number itself is the evidence that the guard ran.

Why the paranoia is proportionate: if a gold query leaks into training, every number in the
ablation is inflated and **no metric computed afterwards can reveal it**. The test set is the
only honest measurement in the project, and it is the one thing that cannot be repaired once
contaminated.

---

## 7. Failure modes this design is built to prevent

A summary of the structural choices, each stated as the bug it exists to make impossible. Every
one of these is a *silent* failure — none of them raises an exception on its own, and none is
visible in an aggregate metric.

| Failure | What it would look like | Structural prevention |
| --- | --- | --- |
| Leaked gold query | Every ablation number inflated; undetectable afterwards | Exact + embedding dedup that fails loudly; both trainers assert; removal count reported |
| In-fold cross-encoder feature | Stage 4 looks useless and gets deleted for the wrong reason | Hash-assigned disjoint folds, checksummed manifest, trainers raise |
| Train/serve feature skew | Offline metrics fine, served quality worse, no discrepancy visible | One pure feature function called by both paths |
| Permuted feature columns | Plausible predictions, no error | Feature order written twice and compared at load |
| Index/model mismatch | Confident garbage rankings, no error (384 dims is a common size) | `DenseIndexMeta` beside the index; loud failure on mismatch |
| Stale artifact reuse | A figure that does not correspond to the config that claims it | Every artifact path keyed by the config fingerprint |
| Silently ignored config key | An experiment that did not test what it claimed | `extra="forbid"` on every config model |
| Missing data read as a "no" | Good results discarded, or bad results presented as verified | Three-valued `ConstraintStatus` end to end, including in the UI |
| Hours crossing midnight | "Open late" queries wrong in both directions | `HoursInterval.covers()` as the only way to ask; unit-tested against each encoding |
| Mutated upstream score | The ablation becomes untraceable | Frozen dataclasses; stages return copies |
| An unmeasured number in a report | A fabricated result | `MetricCell` carries its own CI, *n* and exclusions; a missing measurement renders as an em dash with a reason, never `0` |
| Wrong scope indexed | Labeling effort spent on the wrong city | `data.metro` ships as `null`; `require_metro()` refuses to build |

The common thread: every one of these is prevented by the *shape* of the code — a type, a
required field, an assertion the trainer calls — rather than by remembering to be careful. That
is the point. A convention survives until the evening someone is tired; a frozen dataclass and
a trainer that refuses to run do not need anyone to remember anything.
