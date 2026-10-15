# Concierge — Natural-Language Place Search

This file is the source of truth for this project. Read it fully at the start of every session before doing anything. If anything here conflicts with a request in chat, point out the conflict and ask before proceeding.

## 1. What this project is

Concierge is a natural-language search engine for local businesses. A user types a request the way they would ask a hotel concierge:

> "quiet café with wifi that's open past 10pm"
> "cheap ramen near downtown with outdoor seating"
> "somewhere good for a first date that isn't loud"

and gets the 10 best-matching places, ranked, with an explanation of which parts of the request each result satisfies.

It is built on the Yelp Open Dataset (~7M reviews, ~150K businesses, 11 metro areas, with business attributes such as hours, noise level, WiFi, and outdoor seating).

**One-line pitch:** a multi-stage search and ranking system — LLM query understanding, hybrid retrieval, a fine-tuned neural reranker, and a learning-to-rank model — served at low latency and evaluated rigorously against a hand-labeled test set.

## 2. Why this project exists (read this — it shapes every decision)

The owner is an undergraduate data science student building this as a flagship resume project to land data science and ML engineering internships at tech companies. Their experience so far is mostly finance (quant research, banking data science), so this project's job is to prove they can build and evaluate the kind of ML system tech companies run: search and ranking.

This has three consequences for how you work:

1. **The owner must be able to defend every decision in an interview.** Interviewers will ask "why reciprocal rank fusion?", "how did you mine hard negatives?", "how do you know your labels are good?" Code the owner cannot explain is a liability, not an asset. Prefer clear, readable, well-commented code over clever code. Explain non-obvious choices in plain English as you make them.
2. **Every number must be real and reproducible.** Resume bullets will cite these metrics. Never invent, estimate, or round optimistically. Every metric must come from code the owner can rerun with one command.
3. **Rigor is the differentiator.** Most student search projects embed text and call it done. This one wins on: measured ablations per stage, a hand-labeled gold test set, label-quality checks, leakage-free training, and latency measurements. Never cut corners on evaluation to move faster.

## 3. What "top tier" means — the bar for done

The finished project must have all of these:

- [ ] Four ranking stages, each with a measured contribution (ablation table).
- [ ] At least two models the owner trained (the cross-encoder reranker and the LambdaMART ranker) — not only pretrained models and API calls.
- [ ] A hand-labeled gold test set (~150 queries written by the owner, pooled graded relevance labels).
- [ ] Hard constraints (hours, attributes, price) handled correctly, not just semantic similarity.
- [ ] Label-quality validation: agreement between LLM-generated labels and the owner's labels (Cohen's kappa).
- [ ] Failure analysis by query type, with concrete examples of what breaks and why.
- [ ] A quality-vs-latency curve across reranker sizes and candidate counts.
- [ ] A served API with measured p50/p95 latency under load, plus a live demo frontend.
- [ ] A README that sells it: architecture diagram, ablation table, latency curve, failure analysis, demo GIF, honest limitations.

## 4. Architecture

```
query
  │
  ▼
[Stage 1] Query understanding (LLM → structured JSON)
  │        hard filters (open day/time, attributes, price, category) + semantic text
  ▼
[Stage 2] Hybrid retrieval over all businesses in scope        → top ~200
  │        BM25 (lexical)  +  dense embeddings with FAISS (semantic)
  │        fused with Reciprocal Rank Fusion (RRF), hard filters applied
  ▼
[Stage 3] Cross-encoder reranker (fine-tuned by us)            → top ~50
  │        reads (query, business document) jointly; slower, much more accurate
  ▼
[Stage 4] LambdaMART final ranker (LightGBM, trained by us)    → top 10
           features: stage scores + rating + popularity + constraint matches
  │
  ▼
results + "why this result" (which constraints each result satisfied)
```

**Why stages:** accurate models are slow. Cheap methods run over everything; expensive methods run only on the shortlist. Each stage trades recall for precision. Candidate counts (200 / 50 / 10) are tunable and part of the latency analysis.

### Stage 1 — Query understanding

- LLM: Gemini Flash via the free Gemini API (Google AI Studio key in `.env` as `GEMINI_API_KEY`). Groq is the fallback provider.
- Output: strict JSON validated against a schema, e.g. `{"semantic_text": "...", "open_at": {"day": "Fri", "time": "22:00"} | null, "attributes": {"WiFi": "free", "NoiseLevel": "quiet", ...}, "price_max": 2 | null, "categories": [...]}`.
- Requirements: response caching (identical queries never hit the API twice), timeouts, and a rule-based fallback for common constraints (open late, wifi, outdoor seating, cheap) when the API fails or rate-limits. Measure the latency cost of the LLM call and report the quality/latency tradeoff of LLM parsing vs. rules.

### Stage 2 — Hybrid retrieval

- BM25: `bm25s` library over business documents.
- Dense: a small pretrained embedding model from `sentence-transformers` (start with `BAAI/bge-small-en-v1.5`; follow its documented query/passage prefix conventions). Index with FAISS (start with exact `IndexFlatIP` on normalized vectors for correctness; move to HNSW only if latency requires it, and measure the recall cost).
- Fusion: Reciprocal Rank Fusion, `score = Σ 1/(k + rank)`, default `k = 60`. Document why RRF over weighted score sums (score scales differ between BM25 and cosine similarity).
- Hard filters from Stage 1 are applied here. Businesses with missing attribute data must be handled explicitly and consistently (decide: exclude, keep with penalty, or keep neutral — log the decision and measure its effect).

### Stage 3 — Cross-encoder reranker (trained)

- Initialize from `cross-encoder/ms-marco-MiniLM-L-6-v2` (pretrained on web search relevance). Fine-tune on our synthetic training data (Section 6).
- Train with PyTorch (via `sentence-transformers` CrossEncoder or plain `transformers` — prefer whichever keeps the training loop most readable). Training should run on free Google Colab GPU if the laptop is too slow; keep a CPU path for inference.
- Report NDCG@10 before vs. after fine-tuning on the gold set.

### Stage 4 — LambdaMART final ranker (trained)

- LightGBM with `objective="lambdarank"`, queries as groups.
- Features (start here, add only with evidence): BM25 score, dense score, RRF rank, cross-encoder score, star rating, log(review_count), number of constraints matched, fraction of constraints matched, whether constraint data was missing, category match.
- **Critical:** the cross-encoder score used as a feature for training LambdaMART must come from queries the cross-encoder was *not* trained on. Use a separate split of synthetic queries for LambdaMART training. Otherwise the feature is inflated and LambdaMART learns to over-trust it.
- Report feature importance and an ablation removing the non-text features (to show ratings/popularity actually matter).

### Serving

- FastAPI service; all indexes and models loaded once at startup.
- Export the cross-encoder to ONNX Runtime for faster CPU inference; verify scores match the PyTorch model within tolerance.
- Load-test and report p50/p95 latency per stage and end-to-end.
- Dockerized. Frontend in Next.js (owner's existing skill), showing results with highlighted matched constraints ("why this result").

## 5. Data

### Source and license — hard rules

- The Yelp Open Dataset is licensed for educational/non-commercial use. **Never commit any Yelp data files to git.** `data/` is gitignored. The README must tell readers to download the dataset themselves.
- Do not use "Yelp" in the project name or branding. Say "built on the Yelp Open Dataset."
- Before deploying a public demo, flag to the owner that they must review the Yelp Dataset Terms of Use for what a public demo may display.

### Files

`business.json`, `review.json` (~5GB), `tip.json`, `checkin.json`, `user.json`. Load with DuckDB or Polars (lazy/streaming). Never load the full review file into plain pandas.

### Known quirks — handle them deliberately

- Many attribute values are stringified Python literals, e.g. `"u'quiet'"`, `"'free'"`, and nested dicts as strings like `"{'romantic': False, 'casual': True}"`. Parse with `ast.literal_eval` plus normalization; never with `eval`.
- Hours are strings like `"7:0-20:0"`. Some ranges cross midnight (`"17:0-2:0"`) — handle overnight hours correctly and unit-test this.
- Attribute coverage is sparse — many businesses lack hours, NoiseLevel, WiFi, etc. Quantify missingness early; it drives Stage 1/2 design and is a key finding for the failure analysis.
- Filter permanently closed businesses (`is_open == 0`) from the searchable index.

### Scope

Start with one metro area chosen from the exploratory analysis (owner decides). Keep the pipeline metro-agnostic so expanding later is a config change.

### Business documents

One text document per business: name, categories, attributes rendered as readable phrases ("free WiFi, quiet, outdoor seating, $$"), plus a capped sample of tips and review snippets. Document the selection rule and length cap in DECISIONS.md, since the reranker's context window limits how much text fits.

## 6. Evaluation methodology (the heart of the project)

### Gold test set — owned by the owner

- `eval/gold_queries.csv` with columns `query_id, query, query_type`.
- `query_type` ∈ `vibe` ("cozy spot to read"), `constraint` ("open after 10pm with outdoor seating"), `specific` ("good pho"), `messy` (typos, vague, multi-intent).
- The owner writes all ~150 queries themselves. You may create the file with a handful of example rows and explain what makes a good test query. **Never write the gold queries for them, and never "improve" them.**

### Relevance labels — pooled, graded

- For each gold query, pool the top 20 results from every system being compared, deduplicate, and have the owner grade each with a labeling CLI tool: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant. Store in `eval/qrels.csv`.
- When a new system is added, label only newly surfaced unjudged results (incremental pooling). Report how many top-10 results were unjudged for each system.

### Metrics

- NDCG@10 (primary), Recall@100 (for retrieval stages), MRR.
- Report bootstrap 95% confidence intervals and paired bootstrap comparisons between consecutive systems. With ~150 queries, small gains may be noise — say so when they are.
- Break every metric down by `query_type`.
- Unit-test every metric against tiny hand-made examples with known answers.

### Required outputs (`make eval` produces all of them)

1. **Ablation table:** BM25 → +dense (RRF) → +cross-encoder (pretrained) → +cross-encoder (fine-tuned) → +query parsing → +LambdaMART, with NDCG@10, Recall@100, MRR, and p95 latency per row.
2. **Per-query-type breakdown.**
3. **Quality-vs-latency curve** (NDCG@10 vs. p95 latency) across candidate counts and reranker sizes.
4. **Failure analysis:** worst queries per type with concrete examples and diagnosed causes.

### Leakage rules — never violate

- Gold queries and gold labels are test-only. They must never appear in any training data, prompt examples for synthetic generation, hyperparameter tuning, or feature engineering.
- Use a separate validation split of synthetic queries for tuning.
- Deduplicate synthetic training queries against gold queries (exact and near-duplicate via embedding similarity) and report how many were removed.
- If you notice any leakage risk, stop and flag it.

## 7. Training data for the reranker and LambdaMART

- **Synthetic queries:** prompt the LLM with a business document and ask for several realistic queries that business satisfies, covering all four query types. Batch multiple queries per API call to stay within free-tier limits. Target a few thousand queries. Log the exact prompt in the repo.
- **Positives:** the source business.
- **Hard negatives:** high-ranking hybrid retrieval results for that query that are not the source business.
- **False-negative filtering (important):** some "negatives" are actually relevant (other quiet cafés exist). Use an LLM judge to grade top hard negatives and remove likely false negatives before training. Report how many were removed.
- **Label quality check:** run the LLM judge on a sample of gold-pooled pairs the owner labeled, and report agreement (Cohen's kappa, plus a confusion matrix). This justifies using LLM labels for training.
- **Splits:** synthetic queries split into: cross-encoder train / cross-encoder val / LambdaMART train / LambdaMART val. Fixed seeds; splits saved to disk.

## 8. Repository structure

```
concierge-search/
├── CLAUDE.md               # this file
├── DECISIONS.md            # every decision: options considered, choice, reason
├── README.md
├── Makefile                # make setup / data / index / train / eval / serve / test
├── pyproject.toml
├── .env.example            # GEMINI_API_KEY=, GROQ_API_KEY=
├── .gitignore              # data/, .env, model artifacts, caches
├── configs/                # YAML configs: metro, candidate counts, model names, seeds
├── data/                   # gitignored: raw Yelp files + processed parquet
├── concierge/
│   ├── data/               # loading, cleaning, attribute/hours parsing, documents
│   ├── query/              # Stage 1: LLM parser, schema, cache, rule fallback
│   ├── retrieval/          # Stage 2: BM25, dense + FAISS, RRF, filters
│   ├── rerank/             # Stage 3: cross-encoder training + inference + ONNX
│   ├── ltr/                # Stage 4: features + LambdaMART
│   ├── synth/              # synthetic queries, hard negatives, LLM judge
│   └── pipeline.py         # end-to-end search(query) -> results
├── eval/
│   ├── gold_queries.csv    # written by the owner
│   ├── qrels.csv           # labeled by the owner via the CLI tool
│   ├── label_cli.py
│   ├── metrics.py
│   └── run_eval.py
├── serve/                  # FastAPI app, load test, Dockerfile
├── web/                    # Next.js frontend
├── notebooks/              # exploration only; no logic lives here
└── tests/
```

## 9. Roadmap — work in phases, stop at every checkpoint

**Phase 1 — Foundations and evaluation**

1. Repo setup, Makefile, config, .gitignore, test scaffolding.
2. Data exploration: file summaries, attribute missingness, business counts per metro. **CHECKPOINT: owner picks the metro.**
3. Parsing (attributes, hours with overnight handling) + unit tests. Business document builder. **CHECKPOINT: owner reviews document design.**
4. Evaluation harness: gold query file (example rows only), labeling CLI, metrics + tests, `make eval`.
5. BM25 baseline. **CHECKPOINT: owner writes gold queries and labels the BM25 pool; first real numbers.**

**Phase 2 — Hybrid retrieval** Dense embeddings + FAISS, RRF fusion, filters, missing-data policy. Incremental labeling. Ablation rows 1–2. **CHECKPOINT.**

**Phase 3 — Reranker** Synthetic queries, dedup against gold, hard negatives, false-negative filtering, label-quality check (kappa), cross-encoder fine-tuning. Ablation rows for pretrained vs. fine-tuned. **CHECKPOINT.**

**Phase 4 — Query understanding and final ranking** LLM parser with schema, cache, fallback rules; LambdaMART with out-of-fold cross-encoder features; feature importance; quality-vs-latency curve; failure analysis. **CHECKPOINT.**

**Phase 5 — Ship** FastAPI, ONNX export with parity check, load test, Docker, Next.js frontend with "why this result," README with all figures, demo GIF.

At each checkpoint: summarize what was built, show results, list decisions made, and ask what to do next. Do not start the next phase without the owner's go-ahead.

## 10. How to work with the owner

- **Teach as you build.** The owner is learning this domain. Explain each non-obvious decision in plain English, with the alternatives you considered. Avoid unexplained jargon; define terms the first time.
- **DECISIONS.md is mandatory.** After every meaningful decision, append: the decision, alternatives, why this one, and how it could be tested. The owner studies this file for interviews.
- **Never fabricate.** No invented metrics, no placeholder numbers presented as results, no "this should improve NDCG by about X%." If something hasn't been measured, say so.
- **Ask before:** installing large dependencies, changing the architecture, changing evaluation methodology, or spending API quota on large batch jobs (state the expected number of calls first).
- **Owner-owned work — never do it for them:** writing gold queries, grading relevance labels, picking the metro, and approving each phase.
- **Reproducibility:** fixed random seeds, configs in YAML, artifacts versioned by config, every result regenerable with a `make` target.
- **Tests:** unit tests for parsing (especially hours and attributes), metrics, RRF, filters, and train/test split integrity. Run tests before every commit.
- **Commits:** small, working, clearly described. Commit after each completed step.
- **Code quality:** type hints, small functions, docstrings explaining why, no logic hidden in notebooks.
- **Constraints:** free tools only. Develop on a laptop CPU; use free Google Colab GPU for training if needed. Respect free-tier API rate limits with batching, caching, and backoff.

## 11. Target resume bullets (fill with real measured numbers only)

- Built a natural-language place search engine over [N]K businesses with hybrid BM25 + dense retrieval and LLM query parsing for hours, noise, and amenity constraints.
- Fine-tuned a PyTorch cross-encoder reranker with hard-negative mining and LLM-filtered false negatives, improving NDCG@10 [X]% over hybrid retrieval.
- Trained a LightGBM LambdaMART ranker blending relevance with quality signals, served end-to-end via FastAPI at [X] ms p95 latency.
