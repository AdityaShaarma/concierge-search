# End-to-end verification — 2026-10-19

Measurements from running the system, not from reading it. Fixture corpus (8 businesses),
Apple Silicon laptop, CPU, default thread settings. These are functional verifications,
NOT quality measurements — none of them say anything about NDCG, which needs the dataset
and the owner's labels.

## Dense retrieval (Stage 2, semantic half) — first end-to-end run
- `BAAI/bge-small-en-v1.5` loads, dim 384, ~6s cold.
- Semantic retrieval works with no lexical overlap: "late night noodles" retrieves
  Midnight Ramen at 0.750, ahead of unrelated businesses.
- bge prefix asymmetry correct: `cos(passage(t), query(t)) = 0.938` for identical text t,
  i.e. the instruction prefix really is applied to queries and not to passages.
  `DenseIndex.build` uses `embed_passages`; `search_batch` uses `embed_queries`.
- FAISS id mapping survives save/load: identical ids AND identical scores.

## Cross-encoder (Stage 3) — first end-to-end run
- `cross-encoder/ms-marco-MiniLM-L-6-v2` loads in ~4.5s.
- Ranks sensibly from the joint (query, document) reading alone, no keyword filter:
  query "quiet place to work with wifi" -> quiet cafe 3.097, library cafe 1.626,
  taqueria -8.958, loud ramen -9.534.
- **22 ms/pair on CPU.** This is the empirical justification for the cascade: reranking
  200 candidates would cost ~4.4 s, which is why Stage 2 narrows to 50 first. Worth
  reporting alongside the quality-vs-latency curve.
- Document token budget for that query: 502 of 512.

## ONNX export (Stage 5 concern) — contradicts an assumption in CLAUDE.md
- Parity is good: over 12 probe pairs, max|Δ| = 4.768e-06 against a 1.0e-04 tolerance,
  mean|Δ| = 1.589e-06, ordering identical.
- **ONNX is SLOWER here: torch 20.46 ms/pair vs onnx 77.99 ms/pair -> 0.26x.**
- CLAUDE.md section 4 says to export to ONNX Runtime "for faster CPU inference". On this
  machine that is false by a factor of ~3.8. The likely reason is that PyTorch uses the
  optimized ARM backend on Apple Silicon while onnxruntime's default CPU provider does
  not; a Linux x86 serving host could easily measure the other way.
- The export tool already does the right thing: it measures, reports, and tells the
  caller to keep `serving.use_onnx=false` unless a measurement on the serving host says
  otherwise. Nothing was assumed.
- OPEN QUESTION FOR THE OWNER: this is a conflict with the brief. Options are to keep the
  export with the honest measurement attached (good interview material either way), or to
  measure again on the machine the demo will actually be served from.

## API
- `/health` with no artifacts -> 503, structured, naming the command to run.
- `/health` with a fixture pipeline -> 200, `ready=true`, `n_businesses=7`.
- `POST /search` -> 200, correct parse, correct result, per-stage timings in the payload.
- Malformed body -> 422. Unavailable pipeline -> 503 with an actionable message.
- Degradation warnings are returned on EVERY response and say explicitly that results are
  "NOT hybrid-retrieval results" / "NOT fine-tuned-reranker results" / "NOT final-ranker
  results" when those artifacts are missing.

## Frontend
- `npm run build` clean: TypeScript valid, 4 routes, 113 kB first load JS.
- Renders real results from the live API through the server-side proxy; the browser never
  talks to the API directly.
- Two bugs found by looking at it and fixed: the degradation banner rendered twice, and
  the parsed-query panel labelled three visible filter chips as "(2)".

## Not yet verified
- Anything involving the real dataset: no metro chosen, no parquet built, no real index.
- The fine-tuned reranker and LambdaMART: no trained models exist, because training data
  needs the dataset and an API key.
- Load testing under concurrency.
- Every quality metric in the project.
