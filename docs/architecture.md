# Architecture

The technical companion to the [README](../README.md). The README says what the system does
and how it is evaluated; this document says how the pieces fit, where every artifact comes
from, and — in the longest section — why the out-of-fold cross-encoder feature is the subtlest
correctness property in the whole project.

Nothing here contains a measurement. Where a number would belong, this document names the
command that produces it. The only figures that appear are *settings* (values read out of
`configs/base.yaml`), and they are labelled as such.

**Contents**

1. [Data flow, end to end](#1-data-flow-end-to-end)
2. [The business document](#2-the-business-document)
3. [The shared contracts, and why they are frozen](#3-the-shared-contracts-and-why-they-are-frozen)
4. [A query's path through the cascade](#4-a-querys-path-through-the-cascade)
5. [Artifacts and the config-hash scheme](#5-artifacts-and-the-config-hash-scheme)
6. [Where each trained model comes from](#6-where-each-trained-model-comes-from)
7. [The out-of-fold cross-encoder feature](#7-the-out-of-fold-cross-encoder-feature)
8. [Serving](#8-serving)
9. [Failure modes this design is built to prevent](#9-failure-modes-this-design-is-built-to-prevent)

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

`<fp>` is the config fingerprint — see [section 5](#5-artifacts-and-the-config-hash-scheme).

Note the arrow that is *not* in the diagram: nothing flows from `eval/` back into
`concierge/`. The gold queries and their labels are read by the evaluation harness and by one
leakage guard that exists to *exclude* them, and by nothing else. Section 7 is about keeping
that arrow absent.

**Why DuckDB rather than pandas or plain `json.loads`.** `review.json` is the large file in the
dataset — newline-delimited JSON, one object per review — and the pipeline needs four of its
fields (`business_id`, `text`, `useful`, `date`) for the businesses in one metro. A naive read
constructs a Python object for every field of every review in the file, including the fields
nothing downstream reads, and holds the result in memory. What share of the bytes that wastes
is not something this repository has measured, and it is not stated here as if it had been;
`make explore --count-reviews` streams the file and reports the real per-business row counts.
The mechanism is the reason for the call, and it does not need a number: every read in
`concierge/data/load.py` pushes the column projection *and* the business-id semi-join down into
DuckDB and streams results in batches, so peak memory is set by the batch size rather than by
the file.

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
| Attributes are stringified Python literals: `"u'free'"`, `"{'casual': True}"` | `concierge/data/attributes.py`, via `ast.literal_eval` (never `eval` — this is untrusted third-party text) | Every stage reads attributes; parsing them in more than one place guarantees two normalizations that disagree |
| Hours are unpadded strings, some crossing midnight: `"17:0-2:0"`, `"0:0-0:0"` = open 24h, `"17:0-0:0"` = closes at midnight | parsed in `concierge/data/hours.py`; queried only through `HoursInterval.covers()` in `concierge/types.py` | "Open late" is a whole gold query type's worth of requests; treating the `0` in `"17:0-0:0"` as a midnight crossing would claim the place is open at 1am |
| Attribute coverage is sparse and uneven between cities | measured by `make explore`; handled by three-valued `ConstraintStatus` everywhere | A missing `NoiseLevel` is not a loud café. Collapsing the two produces results that are confidently wrong rather than visibly empty, and the sparser the attribute, the more of them there are |

`is_open_at` is three-valued for a reason that is easy to miss: a business *with* hours data
whose requested day is absent is closed that day (`VIOLATED`), while a business whose `hours`
field is null tells us nothing (`UNKNOWN`). Those two are the same shape in the raw JSON and
mean opposite things to a ranker.

---

## 2. The business document

BM25, the bi-encoder and the cross-encoder score a query against one text document per
business and against nothing else, so `concierge/data/documents.py` sets a ceiling on
achievable quality *upstream of every model choice*. A better reranker cannot recover
information that never entered its window. Five decisions, each argued at the place it
happens:

1. **Attributes become English phrases**, not key-value dumps: `free WiFi, quiet, outdoor
   seating, $$` rather than `WiFi=free NoiseLevel=quiet RestaurantsPriceRange2=2`. The query
   side is natural language, and both retrievers score text against text — the bi-encoder was
   trained on prose, and BM25 needs the query's own words to appear.
2. **Negative attributes are omitted entirely.** There is no phrase for "no outdoor seating",
   because BM25 cannot see a negation: the document would contain the token `outdoor` and match
   the query asking for it. Absence is the only honest encoding of a negative in a bag of
   words, and the arithmetic filter is where a negative is actually enforced.
3. **Structured signal first, user prose last.** Name, categories and attribute phrases go at
   the head; sampled tips and review snippets go at the tail. The cross-encoder's window is
   `rerank.max_length` tokens (a setting, 512 in the shipped config), and truncation takes the
   tail — so the ordering decides what gets cut. `BusinessDocument.truncated` records when
   content was shortened to fit, and the module measures how much of each document actually
   fits, which turns "is the cap hurting us?" into a question with an answer.
4. **Hours are deliberately absent from the text.** "Open past 10pm" is arithmetic, and a
   document that spelled out its hours in prose would let a near-miss place score well on the
   phrase while failing the actual check. Facts belong in the filter, not in the embedding.
5. **Every cap is config** (`documents.max_chars`, `max_reviews`, `max_tips`,
   `review_selection`), and the rendering is byte-stable for a fixed input. Both properties
   exist so document design can be ablated like any other hyperparameter and so an index keyed
   by `Config.fingerprint()` is genuinely reproducible.

Review selection is `useful_then_recent` rather than random: reviews are ranked by the
dataset's own `useful` vote count and then by date. The reason to state the rule rather than
just pick one is that it is a *selection* and not a truncation —
`BusinessDocument.truncated` is deliberately not set by reviews that `max_reviews` left out,
because those two situations call for different fixes.

The division of labour with `concierge/data/attributes.py` is worth noting: that module owns
attribute *parsing and normalization* (resolving `"u'free'"`, flattening `Ambience` into dotted
keys, dropping keys with no data); this one owns attribute *phrasing*, because how an attribute
reads in a document is a rendering choice to be measured, not a property of the data.
`build_document` takes an optional `phrase_fn` so an alternative phrasing can be swapped in and
compared without editing the module.

---

## 3. The shared contracts, and why they are frozen

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
  the filter, through the Stage 4 features, into the API response and the "why this result" UI,
  without ever being flattened to a boolean.

On that last point, the Stage 4 features summarise the three-valued statuses rather than
mirroring them one-to-one, and the asymmetry is deliberate rather than an oversight. There is a
satisfied **count**, a satisfied **fraction** (`NaN` for an unconstrained query, because 0/0 is
not zero), and a **`has_unknown_constraint_data` indicator** — but no violated count, because
by the time Stage 4 runs the filter stage has normally dropped the candidates that violate
something. The indicator is the interesting one: it is what lets the model learn from labels
how much to discount "we have no data" relative to "we checked and it passes", a discount that
Stage 2 currently makes with a hand-set penalty.

The HTTP layer deliberately does **not** reuse these types: `serve/schemas.py` defines its own
pydantic views, and the dependency runs one way only (the API imports the pipeline; nothing in
`concierge/` imports `serve/`). Internal types are shaped for traceable ablations; a wire
format is a public contract. Serializing the internals would mean a field rename inside the
pipeline is a breaking API change, and would couple the frontend to the reranker's score-key
constants.

---

## 4. A query's path through the cascade

`SearchPipeline.search(query, variant=..., k=...)` returns a `SearchResponse` carrying the
parsed query, the results, per-stage timings and per-stage candidate counts. Latency is
measured on **every** search, not behind a debug flag, because p50/p95 per stage is a reported
result and a measurement that only happens when someone remembers to ask for it is a
measurement that ends up missing.

1. **Parse** (`concierge.query.parser.parse`) — cache → LLM → rules, in that fixed order, and
   it never raises. The returned `ParsedQuery` records which path answered (`ParseSource.LLM /
   CACHE / RULES`) and what the parse cost in milliseconds, so the ablation can attribute both
   quality and latency. `semantic_text` — the request with every *rejectable* constraint
   phrase removed (`concierge.query.rules.strips_phrase`) — is the only string the text models
   see, via the one definition of it, `concierge.rerank.model.query_text_for`, which falls
   back to the raw query when stripping left nothing (an all-constraint request such as "free
   wifi"). `open_at`, `attributes`, `price_max`, `categories` and `geo` are checked
   arithmetically instead.
2. **Retrieve** — BM25 and the dense index each return `(business_id, score)` lists through
   the *same* `search(query_text, top_k)` shape, so `fusion.fuse` never needs to know which
   retriever it is talking to. RRF combines them by rank into `Candidate` objects carrying
   `SCORE_BM25`, `SCORE_DENSE`, `SCORE_RRF` and the matching ranks.
3. **Filter** (`retrieval.filters.apply_filters`) — each candidate's constraints are evaluated
   into `ConstraintMatch` tuples, and the configured missing-data policy
   (`retrieval.missing_attribute_policy`: `exclude` / `penalize` / `neutral`) is applied. The
   `FilterReport` records *why* each dropped candidate was dropped, which is what makes a query
   that returns nothing diagnosable rather than mysterious.
4. **Rerank** (`rerank.model.CrossEncoderReranker`) — the shortlist is scored jointly,
   stably re-sorted, and truncated to `candidates.rerank`. The pipeline delegates the whole of
   Stage 3 to this object rather than reimplementing the sort, so there is only one definition
   of the stage's ordering.
5. **Final ranking** (`ltr` → `LightGbmRanker`) — one feature row per candidate from the
   single shared feature function, one booster prediction, truncate to `candidates.final`.
6. **Explain** — each `SearchResult` carries its constraint matches split into satisfied,
   violated and unknown, plus every stage score, for the UI and for debugging one query.

**Why the sorts are stable.** Stages 2, 3 and 4 each re-sort the candidate list, and every sort
is stable, so a tie is broken by the upstream order rather than by whatever the sort
implementation happens to do. That is what makes an ablation difference attributable: if two
candidates score identically at Stage 3, the ordering they keep is the one Stage 2 gave them,
not an arbitrary one that could change between runs.

**The ablation is the same code path, parameterized.** `Stage` is a flag enum
(`QUERY_LLM | BM25 | DENSE | FILTERS | RERANK | LTR`) and a `Variant` is a named combination of
flags plus which cross-encoder checkpoint to use. The six ablation rows are *data*
(`pipeline.ABLATION`), so the published table cannot drift from the code that produced it, and
"the full system minus hard filters" is an expression (`VARIANT_FULL.stages & ~Stage.FILTERS`)
rather than a second code path that could diverge. `Variant.__post_init__` rejects a
configuration with no retriever at all, because such a row would produce a plausible-looking
set of zeros instead of an error.

One property of the ladder is easy to misread and worth stating twice: **`Stage.FILTERS` is on
in every ablation row.** The "+ LLM query parsing" row therefore measures the LLM parser
against the *rule* parser, not against having no constraint handling — rows below it still
apply whatever constraints the rule parser found. Filters-versus-no-filters is a separate
diagnostic variant (`VARIANTS["no_filters"]`), reported alongside the ladder rather than inside
it, because a row that changed two things at once could not attribute either.

**Missing models degrade, they do not crash.** The pipeline must be runnable at every point in
the project's own construction: before the dense index exists, before the cross-encoder is
fine-tuned, before LambdaMART is trained. So a missing artifact produces a warning naming the
artifact and a shorter cascade (fine-tuned → pretrained → no reranking; LambdaMART →
cross-encoder order). The warning matters as much as the fallback: a silent degradation would
surface only as a mysteriously weak ablation row.

**Where geo currently stops.** The parser, the filter and the feature are all in place —
`check_geo` is a haversine test against the request's radius, and `distance_km` is a Stage 4
feature — but turning a phrase like "downtown" into coordinates needs a gazetteer, and this
repository has none. `validate_parsed_query` takes an optional `geo_resolver`; nothing wires one
in, so `parsed.geo` is `None` on the live path, `check_geo` does not run, and `distance_km` is
`NaN` for every candidate. The degradation is deliberate rather than silent: an unresolved
location label is folded back into `semantic_text`, so "pharmacy near the train station" still
retrieves on the words instead of dropping the phrase. It is a stated limitation in the README
rather than a feature that half-works without saying so.

---

## 5. Artifacts and the config-hash scheme

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
| Exploration report | `artifacts/explore/<fp>/` | `make explore` |
| BM25 index | `artifacts/bm25/<fp>/` | `make index` |
| Dense index + `DenseIndexMeta` | `artifacts/dense/<fp>/` | `make index` |
| Synthetic queries, raw LLM responses, dedup report | `artifacts/synth/<fp>/` | `make synth-generate` |
| Mined hard negatives | `artifacts/synth/<fp>/` | `make synth-negatives` |
| Judge verdicts, filtered examples | `artifacts/synth/<fp>/` | `make synth-judge` |
| Fold manifest + one training file per split | `artifacts/synth/<fp>/splits.json`, `*_train.jsonl`, `*_val.jsonl` | `make synth-splits` |
| Fine-tuned cross-encoder + `RunMetadata` | `artifacts/rerank/<fp>/` | `make train-reranker` |
| LambdaMART booster + `feature_names.json` | `artifacts/ltr/<fp>/`, `…-textonly/` for the ablation | `make train-ltr` |
| Eval report, `results.json`, latency CSV + figure | `artifacts/eval/<fp>/` | `make eval` |
| Load-test record | `artifacts/load_test/` | `python -m serve.load_test` |
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
  by a different model can load without error and return confident garbage. The same idea
  appears again in `RunMetadata` for the cross-encoder and in the split manifest's checksum: a
  hash of the inputs tells you *which* config produced an artifact, while metadata inside the
  artifact tells you whether it is the artifact you think it is.
- **The parse cache is keyed separately**, by `(provider, model, prompt_version, normalized
  query)`, where `prompt_version` is the declared version plus a hash of the prompt body and
  the attribute vocabulary. It is not under the config fingerprint because an unrelated config
  change must not throw away a cache whose answers are still valid — but *editing the prompt*
  must, and the prompt hash is what ensures it does. `make clean` removes `artifacts/` and
  `data/processed/`, and deliberately leaves `data/raw/` and the hand-made label files alone.

---

## 6. Where each trained model comes from

Two models are trained here. Both learn from **synthetic queries**, because this is a
cold-start search engine: there is no query log, so the problem is inverted — show a language
model one business document and ask what a person would type to find *that* place. The source
business is then the positive label for free, which is the only reason a labeled training set
is obtainable at zero cost.

`make synth` runs the four stages below in order. Each reads only the previous stage's artifact
out of `artifacts/synth/<fp>/`, which is what keeps the gold-leakage guard on the training
path: the only query file the later stages read is the deduplicated one that `synth-generate`
writes. All four resume — generation replays cached responses, mining reuses examples already
on disk, the judge caches a verdict per pair, and splitting is deterministic — and the two that
spend API quota print the exact call count and ask first.

```
documents.parquet
      │
      │  make synth-generate — batched LLM calls, raw responses appended to
      │  disk as they arrive (a 429 mid-run must not cost the completed batches)
      ▼
synthetic queries  ──►  dedupe_against_gold  ──►  kept queries
      │                 exact + embedding cosine ≥ synth.gold_dedup_cosine_threshold,
      │                 against eval/gold_queries.csv; fails loudly if that file is missing
      ▼
make synth-negatives — hard negatives = top hybrid-retrieval results for that
      │  query that are not the source business (no API calls; needs `make index`)
      ▼
make synth-judge — LLM judge grades the hardest negatives; likely false
      │  negatives (other genuinely relevant places) are removed before training
      ▼
make synth-splits — near-duplicate clustering, then hash-based fold assignment,
      │  checksummed to splits.json
      │
      ├── crossencoder_train / crossencoder_val  ──►  make train-reranker  (Stage 3)
      └── ltr_train / ltr_val                    ──►  make train-ltr       (Stage 4)
```

The two arrows at the bottom are not symmetrical, and the difference is the reason section 7
exists. The cross-encoder trains on *(query, document)* text straight out of its two splits.
LambdaMART trains on *feature rows*, and one of those features is the fine-tuned
cross-encoder's score — so its input cannot be built until the reranker exists, and it has to
be built by running retrieval and the fine-tuned cross-encoder over each LambdaMART-fold
query's candidates. That is a separate pass over the data, between the two trainers, rather
than something `make synth-splits` can emit.

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

**Stage 4, LambdaMART.** LightGBM with `objective="lambdarank"` and queries as groups. The
objective is the point of using it: gradients are formed over pairs *within* a query and
weighted by how much swapping that pair would change NDCG, so the model spends its capacity at
the top of the list, where the metric is sensitive, instead of fitting the tail equally.
Features come from `concierge/ltr/features.py` — one pure function of `(ParsedQuery, Candidate,
Business)`, called by both the training job and the serving path, because
**training/serving feature skew** is the classic way a ranking system goes quietly wrong:
offline the feature is computed one way, online another, nothing crashes, offline metrics look
fine, and served quality is worse than the numbers claim with no visible discrepancy. One
implementation means a wrong feature is wrong identically in both places — a bug you can find
rather than a gap you cannot see. Missing values are `NaN`, which LightGBM routes natively per
split, rather than `0` or `-1`, which the trees would read as real magnitudes.

The feature set is `FEATURE_NAMES`, and it is split into two named groups so the "do ratings
and popularity actually matter?" question is an ablation rather than an opinion:

| group | features |
| --- | --- |
| `TEXT_FEATURE_NAMES` — what a pure similarity system already has | `bm25_score`, `dense_score`, `rrf_rank`, `cross_encoder_score` |
| `NON_TEXT_FEATURE_NAMES` — what Stage 4 adds | `stars`, `log1p_review_count`, `n_constraints_satisfied`, `frac_constraints_satisfied`, `has_unknown_constraint_data`, `category_match`, `distance_km` |

A second run with `--ablate-non-text` trains on the first group alone and writes to
`artifacts/ltr/<fp>-textonly/`, suffixed so the two runs cannot overwrite each other; the gap
between them is the measurement behind "do ratings and popularity matter?". Feature importance
comes out of the same trainer.

The stage that sits between these two trainers is `make ltr-examples`
(`concierge/ltr/build_examples.py`). The trainer's `--examples` input defaults to
`artifacts/synth/<fp>/ltr_examples.jsonl`, beside the manifest those rows must agree with, and
that producer now writes it: retrieval plus the fine-tuned cross-encoder over each
LambdaMART-fold query. It is deliberately not part of `make synth`, because every row needs a
cross-encoder score and so cannot exist until the reranker is fine-tuned. `make train-ltr`
still fails naming the missing path rather than training on something improvised. Stage 4 is
code and tests without a trained artifact. It fails closed, which is the right direction: a
hand-assembled examples file would be exactly the stale-feature-distribution hazard the shared
feature function exists to prevent.

`save_ranker` writes the feature order **twice** — inside the model file and in
`feature_names.json` — specifically so `pipeline.check_feature_order` can compare them at load
time. A permuted feature matrix produces plausible predictions and no error whatsoever, which
makes it the most dangerous silent bug available in Stage 4.

---

## 7. The out-of-fold cross-encoder feature

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

There is a fourth link that is easy to overlook: LambdaMART records the checksum of the split
manifest its cross-encoder was fine-tuned under, and compares it against the manifest in front
of it. Two artifacts that each pass their own checks can still disagree with each other, and
this is the check that notices.

### Why folds are assigned by hash, not by a seeded shuffle

Each group's fold is `hash(seed, group_key)` bucketed by the configured fractions, so a
query's fold depends **only on its group key** — not on how many other queries exist, and not
on their order.

The alternative, `random.shuffle(queries)` with a fixed seed and then four slices, is wrong in
a way that is easy to miss: generating more synthetic data later reshuffles the whole list, and
a query that was in `crossencoder_train` for the trained checkpoint can land in `ltr_train` for
the next LambdaMART run. The cross-encoder checkpoint on disk has still seen it. The manifest
says it is a LambdaMART query. The out-of-fold guarantee is now false, and nothing in the ids
looks wrong.

### What a "group" is, and the stability this does and does not buy

The unit being hashed is a **group**, not necessarily a single query, because query ids are not
the only thing that can straddle the fold boundary. `make synth-splits` defaults to
`--group-by cluster`:

| grouping | what may not span a fold | residual leak |
| --- | --- | --- |
| `query` | one query id | a paraphrase of a cross-encoder query can land in a LambdaMART fold; so can a *different* query written from the same business, whose document the cross-encoder has already read |
| `business` | all queries written from one business | paraphrases generated from different businesses can still straddle |
| `cluster` (default) | a near-duplicate cluster: identical text, embedding near-duplicates above `synth.gold_dedup_cosine_threshold`, and — by default — queries sharing a source business | paraphrases from different businesses that score *below* the clustering threshold; unmeasured, and accepted |

`configs/base.yaml` declares the floor (`synth.min_split_grouping: cluster`) so that the
requirement is a config fact rather than an argparse default. It is enforced in both
directions: `make synth-splits` refuses to *build* a manifest below the floor, and
`verify_splits` — which `require_disjoint_splits` runs on every trainer's path — refuses to
*read* one, so a manifest produced under a weaker setting cannot be trained on later. A manifest also carries a checksum
over `(version, seed, group_by, fractions, assignment)`, so a hand-edited split file is
detectable rather than an undetectable way to produce an inflated result.

Two costs, both real and both worth stating plainly rather than discovering in an interview:

- **Fold sizes are only approximately the configured fractions.** Groups are not single
  queries, so the realised fractions scatter further from the nominal ones. The manifest
  therefore records the **realised** counts and fractions rather than restating the configured
  ones.
- **Cluster membership is not monotonic under data growth.** A cluster's key is its lowest
  member id, so adding synthetic queries can admit a lower id into an existing cluster or link
  two previously separate clusters — and either changes the group key, which changes the
  bucket, which moves already-assigned queries between folds. So the stability property is
  "stable for a fixed query set", not "stable under growth": with `--group-by query` a query's
  fold really does depend on nothing but its own id, and the clustering trades some of that
  away to close the paraphrase leak. Growing the dataset means re-running the cross-encoder
  fine-tune. That is not silent — the manifest-checksum comparison above turns it into a
  refusal to train rather than a quietly broken guarantee — but it is a cost, and the reason to
  pay it is that an undetected paraphrase straddle inflates a reported number while a forced
  retrain only costs time.

### The second leakage boundary: the gold set

Folds keep the two *trained models* independent of each other. A separate guard keeps both of
them independent of the **test set**:

- `synth.generate.dedupe_against_gold` removes synthetic queries that duplicate a gold query,
  exactly and by embedding cosine ≥ `synth.gold_dedup_cosine_threshold`, using the same
  embedding model Stage 2 retrieves with. It raises if `eval/gold_queries.csv` is absent rather
  than passing everything through — a missing gold file must not be the thing that silently
  disables the leakage guard.
- `rerank.train.assert_no_gold_leakage` and `ltr.train.assert_no_gold_leakage` independently
  refuse to train if any training query matches a gold query by id or by normalized text.
- The count removed by dedup is a **required reported figure**, not an implementation detail,
  because the number itself is the evidence that the guard ran.

The threshold is worth being precise about, since "strict" points in two directions. It is a
*high* bar for calling two queries the same, so dedup removes only close matches and a
distant paraphrase of a gold query can survive into training. Nothing here has measured what
it catches on real embeddings, so that residual is an accepted, unquantified risk rather than a
solved problem.

Why the paranoia is proportionate: if a gold query leaks into training, every number in the
ablation is inflated and **no metric computed afterwards can reveal it**. The test set is the
only honest measurement in the project, and it is the one thing that cannot be repaired once
contaminated.

---

## 8. Serving

`serve/app.py` exposes a `create_app(...)` factory — `app = create_app()` at the bottom is just
the default instance `uvicorn serve.app:app` needs — so a test can construct a service around a
stub pipeline instead of loading real indexes, and a second variant can be served from the same
code. Models and indexes load once, in the lifespan handler rather than at import or per
request, because a cross-encoder loaded per request would make the latency table meaningless.

- **`GET /health` reports per-artifact load status, not just a boolean.** The status *code* is
  503 only when the service cannot answer searches at all; a service happily serving a
  BM25-only configuration returns 200, because a container probe keys on the code and must not
  restart a deliberately degraded deployment. The *body* carries the nuance a human needs:
  which artifacts loaded, which warnings fired, the config fingerprint, and the variant name.
  Whether an artifact counts as *required* is judged against the variant being served — a
  missing LambdaMART model is a degradation of the full system and irrelevant to the
  `hybrid_rrf` ablation row.
- **The degradation is visible in every response, not just at `/health`.** This is the one
  serving decision that is really an honesty decision: the pipeline falls back silently well
  (pretrained cross-encoder for fine-tuned, cross-encoder order for LambdaMART), and a
  screenshot of the degraded system looks exactly like a screenshot of the full one. So every
  `/search` response repeats the load warnings. A demo cannot accidentally claim four stages
  while running two.
- **`POST /search` returns 200 with zero results** when nothing matches. "Nothing is open at
  3am with outdoor seating" is a correct answer, not a 404, and the response still carries the
  parsed query and the per-stage candidate counts so the empty list is explainable.
- **The handlers are synchronous `def`.** The cascade is CPU-bound: a cross-encoder forward pass
  over the shortlist holds the GIL and never awaits. Declared `async def`, one such call would
  block the event loop and every other in-flight request with it, and the load test's
  concurrency levels would measure a queue rather than the service. Declared `def`, Starlette
  runs the handler in its worker threadpool. The honest framing is that this service is
  thread-pool-parallel, not async-parallel, because nothing in the cascade is I/O-bound except
  the optional Stage 1 LLM call.
- **`serve/load_test.py`** drives the API with asyncio and `httpx` at several concurrency
  levels and records client-observed *and* server-reported latency: what a user waits (HTTP,
  queueing, the threadpool, serialization) versus what the cascade costs (the per-stage table).
  A single end-to-end number cannot tell you which one to go fix. Percentiles are nearest-rank
  rather than interpolated, so "95% of requests finished within X" names a time some request
  actually took, and warmup requests are excluded because first-request costs are one-time. It
  measures and never asserts a target; the known limitation — at high concurrency the client
  shares a GIL with its own measurement loop — is recorded in the output rather than argued
  away.
- **`concierge/rerank/onnx_export.py`** exports the fine-tuned cross-encoder and then does two
  things in order: checks that ONNX scores match PyTorch within a tolerance, and *measures*
  ms/pair on both backends. `serving.use_onnx` stays `false` until a measurement on the serving
  host says ONNX is faster there, because it is not always, and the script prints what it
  observed rather than assuming a win.
- **`serve/Dockerfile` mounts everything the cascade loads rather than baking it in.** The
  processed parquet files are derived from the Yelp Open Dataset, so an image containing them
  could not be shared; separately, it keeps the image independent of which fingerprint's
  artifacts you happen to be serving.
- **`web/` is a Next.js App Router page that calls the API server-side.** `CONCIERGE_API_URL`
  has no `NEXT_PUBLIC_` prefix on purpose, so the browser talks to the Next route and the API
  address is not published to the client. The UI's job is to make Stage 1 and the three-valued
  constraints visible: the parsed query is shown back split into semantic text and hard
  filters, and each result lists its checks as **met** / **not met** / **no data**, each with
  its own icon and word rather than colour alone.

---

## 9. Failure modes this design is built to prevent

A summary of the structural choices, each stated as the bug it exists to make impossible. Every
one of these is a *silent* failure — none of them raises an exception on its own, and none is
visible in an aggregate metric.

| Failure | What it would look like | Structural prevention |
| --- | --- | --- |
| Leaked gold query | Every ablation number inflated; undetectable afterwards | Exact + embedding dedup that fails loudly; both trainers assert; removal count reported |
| In-fold cross-encoder feature | Stage 4 looks useless and gets deleted for the wrong reason | Hash-assigned disjoint folds, checksummed manifest, trainers raise |
| Paraphrase straddling the fold boundary | Same as above, but every query id looks clean | Folds assigned by near-duplicate cluster, not by query id; `synth.min_split_grouping` declares the floor |
| Checkpoint and manifest from different generations | Two artifacts that each pass their own checks and contradict each other | The cross-encoder's split-manifest checksum is recorded and compared at LambdaMART time |
| Train/serve feature skew | Offline metrics fine, served quality worse, no discrepancy visible | One pure feature function called by both paths |
| Permuted feature columns | Plausible predictions, no error | Feature order written twice and compared at load |
| Index/model mismatch | Confident garbage rankings, no error (384 dims is a common size) | `DenseIndexMeta` beside the index; loud failure on mismatch |
| Stale artifact reuse | A figure that does not correspond to the config that claims it | Every artifact path keyed by the config fingerprint |
| Silently ignored config key | An experiment that did not test what it claimed | `extra="forbid"` on every config model |
| Missing data read as a "no" | Good results discarded, or bad results presented as verified | Three-valued `ConstraintStatus` end to end, including in the UI |
| Hours crossing midnight | "Open late" queries wrong in both directions | `HoursInterval.covers()` as the only way to ask; unit-tested against each encoding |
| Mutated upstream score | The ablation becomes untraceable | Frozen dataclasses; stages return copies |
| Unstable tie-breaking | Ablation differences that are really sort noise | Every stage re-sorts stably, so ties keep the upstream order |
| An unmeasured number in a report | A fabricated result | `MetricCell` carries its own CI, *n* and exclusions; a missing measurement renders as an em dash with a reason, never `0` |
| Wrong scope indexed | Labeling effort spent on the wrong city | `data.metro` ships as `null`; `require_metro()` refuses to build |
| Operating point chosen on the test set | Headline NDCG@10 inflated with no model or code change to point at | `--sweep-on` must be stated; the gold curve carries a warning into the report saying its numbers must not pick depths |
| A smoke-test run quoted as a result | A plausible table computed over a handful of queries | `--max-queries` stamps PARTIAL RUN into the result object, and the report prints it on every table |

The common thread: every one of these is prevented by the *shape* of the code — a type, a
required field, an assertion the trainer calls — rather than by remembering to be careful. That
is the point. A convention survives until the evening someone is tired; a frozen dataclass and
a trainer that refuses to run do not need anyone to remember anything.

What this table deliberately does **not** claim is that the system is correct. It claims that a
specific list of silent failures has been converted into loud ones. The open defects that are
*not* in this table are in the README's limitations section and in `DECISIONS.md`, which is
where a known gap belongs — a failure-prevention table that quietly doubled as a status report
would be the most misleading page in the repository.
