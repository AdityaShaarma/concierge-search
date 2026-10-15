/**
 * TypeScript mirror of the wire contract in `serve/schemas.py`.
 *
 * Hand-written rather than generated from the OpenAPI schema. The reason is
 * the demo's failure mode: a codegen step means the frontend cannot be built
 * or typechecked unless the Python API is running, and this UI has to be
 * buildable (and reviewable) with no backend, no dataset and no API key
 * present. The contract is ~8 types and changes rarely, so the cost of
 * keeping it in sync by hand is lower than the cost of that coupling.
 *
 * If the Python schemas change, this file is the single place to follow.
 * Field names match `serve/schemas.py` exactly — snake_case included — so the
 * two files can be diffed side by side.
 */

/** Three-valued on purpose; see `ConstraintStatus` in `concierge/types.py`. */
export type ConstraintStatus = "satisfied" | "violated" | "unknown";

/** Where Stage 1's parse came from. `rules` means the LLM was not used. */
export type ParseSource = "llm" | "cache" | "rules";

/**
 * `requested` and `actual` are `Any` on the Python side: a constraint may be
 * about a time, a price band, a bool, or a nested attribute dict. `unknown`
 * (the TS type) is the honest mirror of that, and forces every read site to
 * narrow before rendering.
 */
export interface ConstraintView {
  name: string;
  requested: unknown;
  actual: unknown;
  status: ConstraintStatus;
  /** Pre-phrased by the backend so the "no data" wording cannot drift. */
  explanation: string;
}

/**
 * Already split into three buckets by the API. The UI keeps that split rather
 * than re-deriving it, which is the whole point of the server-side grouping:
 * an `unknown` must never be rendered through the same code path as a failure.
 */
export interface ConstraintGroups {
  matched: ConstraintView[];
  unmatched: ConstraintView[];
  unknown: ConstraintView[];
}

export interface BusinessView {
  business_id: string;
  name: string;
  city: string;
  state: string;
  latitude?: number | null;
  longitude?: number | null;
  stars: number;
  review_count: number;
  categories: string[];
  /**
   * Normalized attributes. An absent key means the dataset has no data for
   * it — never render a missing key as a negative.
   */
  attributes: Record<string, unknown>;
}

export interface ResultView {
  rank: number;
  score: number;
  business: BusinessView;
  constraints: ConstraintGroups;
  stage_scores: Record<string, number>;
}

/** `open_at` is a loose dict on the wire, so both fields are optional here. */
export interface OpenAtView {
  day?: string;
  minute_of_day?: number;
}

export interface GeoView {
  latitude?: number;
  longitude?: number;
  radius_km?: number;
  label?: string;
}

export interface ParsedQueryView {
  semantic_text: string;
  open_at: OpenAtView | null;
  attributes: Record<string, unknown>;
  price_max: number | null;
  categories: string[];
  geo: GeoView | null;
  source: ParseSource;
  parse_latency_ms: number;
  n_hard_constraints: number;
}

export interface StageTimingView {
  stage: string;
  ms: number;
}

export interface TimingsView {
  /** The pipeline's own end-to-end measurement, not the sum of the stages. */
  total_ms: number;
  stages: StageTimingView[];
}

export interface SearchResponseModel {
  query: string;
  parsed: ParsedQueryView;
  results: ResultView[];
  timings: TimingsView;
  candidate_counts: Record<string, number>;
  /**
   * Non-fatal problems with this deployment, e.g. "no fine-tuned
   * cross-encoder loaded". Surfaced prominently: a demo screenshot must never
   * silently be of a degraded system.
   */
  warnings: string[];
}

export type HealthStatus = "ok" | "degraded" | "unavailable";

export interface ArtifactStatus {
  name: string;
  loaded: boolean;
  detail: string;
}

export interface HealthResponse {
  status: HealthStatus;
  ready: boolean;
  artifacts: ArtifactStatus[];
  warnings: string[];
  n_businesses?: number | null;
  n_documents?: number | null;
  variant?: string | null;
  config_fingerprint?: string | null;
  error?: string | null;
}

/** Shape of every error body this UI is willing to display. */
export interface ErrorResponse {
  detail: string;
}
