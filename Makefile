# Every number in the README must be regenerable from these targets.
# Nothing here may silently succeed on missing inputs — fail loudly instead.

UV ?= uv
RUN := $(UV) run
CONFIG ?= configs/base.yaml

.DEFAULT_GOAL := help
.PHONY: help setup fix-openmp check-openmp data explore index train train-reranker train-ltr eval serve test lint fmt typecheck clean

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

train: train-reranker train-ltr  ## Train both models we own

train-reranker:  ## Fine-tune the cross-encoder (Stage 3)
	$(RUN) python -m concierge.rerank.train --config $(CONFIG)

train-ltr:  ## Train the LambdaMART final ranker (Stage 4)
	$(RUN) python -m concierge.ltr.train --config $(CONFIG)

label:  ## Grade pooled results for the gold queries (owner-only task)
	$(RUN) python -m eval.label_cli --config $(CONFIG)

eval:  ## Ablation table, per-query-type breakdown, latency curve, failure analysis
	$(RUN) python -m eval.run_eval --config $(CONFIG)

serve:  ## Run the FastAPI search service
	$(RUN) uvicorn serve.app:app --reload --port 8000

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
