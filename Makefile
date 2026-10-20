# Every number in the README must be regenerable from these targets.
# Nothing here may silently succeed on missing inputs — fail loudly instead.

UV ?= uv
RUN := $(UV) run
CONFIG ?= configs/base.yaml

.DEFAULT_GOAL := help
# Every target in this file is phony — none of them is the name of a file they
# produce. A target left out of these lists stops working the moment a file or
# directory of the same name appears in the repo root (`label` was the one that
# was missing), and the failure looks like Make deciding the work is up to date.
.PHONY: help setup fix-openmp check-openmp data explore index train train-reranker train-ltr label eval serve test lint fmt typecheck clean
.PHONY: synth synth-generate synth-negatives synth-judge synth-splits synth-label-quality
.PHONY: demo

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install all dependencies
	$(UV) sync --all-groups
	@$(MAKE) --no-print-directory fix-openmp
	@echo "✓ env ready. Copy .env.example to .env and add your keys."

fix-openmp:  ## Consolidate duplicate OpenMP runtimes (macOS; rerun after any uv sync)
	@$(RUN) python scripts/fix_openmp.py

check-openmp:  ## Assert one OpenMP runtime AND that FAISS still matches brute force
	@$(RUN) python scripts/check_openmp.py

data:  ## Parse raw Yelp JSON into processed parquet (needs data/raw)
	$(RUN) python -m concierge.data.build --config $(CONFIG)

explore:  ## Dataset summary: metro counts, attribute missingness (Phase 1 step 2)
	$(RUN) python -m concierge.data.explore --config $(CONFIG)

index:  ## Build the BM25 and FAISS indexes over business documents
	$(RUN) python -m concierge.retrieval.build_index --config $(CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic training data (Phase 3). Four stages, run in this order.
#
# Each stage reads ONLY the previous stage's artifact out of
# artifacts/synth/<config fingerprint>/, which is what keeps the gold leakage
# guard on the training path: the only query file the later stages read is the
# deduplicated one that `synth-generate` writes. Every stage prints its quota
# plan, and the ones that would spend calls also ask first (CLAUDE.md §10); all
# four resume — generation replays cached responses, mining reuses examples
# already on disk, the judge caches a verdict per pair, and splitting is
# deterministic.
#
# `synth` chains them, as recipe lines rather than as prerequisites. Make orders
# prerequisites sequentially only in a serial build: under `make -j synth` (or an
# inherited MAKEFLAGS=-j) all four would start at once and stages 2-4 would read
# artifacts stage 1 has not written yet. Recipe lines run in order whatever -j
# says, so the one target a new reader is most likely to try cannot fail in a way
# that looks like a broken pipeline. Each stage stays individually runnable.
#
# Prerequisites: `make data` (business documents) and `make index` (hard negatives
# are mined from the real hybrid retriever).
# ─────────────────────────────────────────────────────────────────────────────

synth:  ## Build all synthetic training data (the four stages, in order)
	@$(MAKE) --no-print-directory synth-generate
	@$(MAKE) --no-print-directory synth-negatives
	@$(MAKE) --no-print-directory synth-judge
	@$(MAKE) --no-print-directory synth-splits

synth-generate:  ## Generate synthetic queries and dedup them against the gold set
	$(RUN) python -m concierge.synth.generate --config $(CONFIG)

synth-negatives:  ## Mine hard negatives from the hybrid retriever (no API calls)
	$(RUN) python -m concierge.synth.negatives --config $(CONFIG)

synth-judge:  ## LLM-judge the hardest negatives and drop the false ones
	$(RUN) python -m concierge.synth.judge --config $(CONFIG)

synth-splits:  ## Assign the four splits and write one training file per split
	$(RUN) python -m concierge.synth.splits --config $(CONFIG)

synth-label-quality:  ## Judge-vs-owner agreement (Cohen's kappa); needs eval/qrels.csv
	$(RUN) python -m concierge.synth.judge --config $(CONFIG) --job label-quality

# Sequential for the same reason `synth` is, and here the ordering is not merely
# tidy: one of LambdaMART's features is the *fine-tuned* cross-encoder's score,
# so its feature rows cannot be built until the reranker exists. Under `-j` these
# two would otherwise start together.
train:  ## Train both models we own (reranker first — LambdaMART needs its scores)
	@$(MAKE) --no-print-directory train-reranker
	@$(MAKE) --no-print-directory train-ltr

train-reranker:  ## Fine-tune the cross-encoder (Stage 3)
	$(RUN) python -m concierge.rerank.train --config $(CONFIG)

# Known gap, stated here because this is the target a reader runs: Stage 4 reads
# candidate-level feature rows (artifacts/synth/<fingerprint>/ltr_examples.jsonl)
# that nothing in the repo writes yet — building them needs retrieval plus the
# *fine-tuned* cross-encoder over each query's candidates, so the stage between
# train-reranker and here still has to be written. Until it is, this target stops
# at a FileNotFoundError that states the contract, which is the intended
# behaviour: no hand-assembled scores, no model, no number to retract.
train-ltr:  ## Train the LambdaMART final ranker (Stage 4; needs feature rows — see above)
	$(RUN) python -m concierge.ltr.train --config $(CONFIG)

label:  ## Grade pooled results for the gold queries (owner-only task)
	$(RUN) python -m eval.label_cli --config $(CONFIG)

eval:  ## Ablation table, per-query-type breakdown, latency curve, failure analysis
	$(RUN) python -m eval.run_eval --config $(CONFIG)

serve:  ## Run the FastAPI search service
	$(RUN) uvicorn serve.app:app --reload --port 8000

demo:  ## Run the API over tests/fixtures — no dataset needed (see scripts/demo_server.py)
	@echo "Fixture corpus, not the Yelp Open Dataset: a functional demonstration,"
	@echo "NOT a measurement. Every response says so. Frontend: cd web && \\"
	@echo "CONCIERGE_API_URL=http://127.0.0.1:8000 npm run dev"
	$(RUN) python scripts/demo_server.py --port 8000

test:  ## Run the unit tests
	$(RUN) pytest

lint:  ## Lint
	$(RUN) ruff check .

fmt:  ## Format
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck:  ## Static type check
	$(RUN) mypy concierge eval serve

clean:  ## Remove generated artifacts (keeps raw data and labels)
	rm -rf artifacts/* data/processed/* .pytest_cache .ruff_cache .mypy_cache
	@echo "✓ cleaned. data/raw and eval/*.csv untouched."
