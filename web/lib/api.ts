/**
 * Browser-side client for this app's own route handlers.
 *
 * Two jobs: turn any failure into a message worth showing, and normalise the
 * response so a React component never meets an `undefined` where it expects a
 * list. The normalisation never invents values — missing containers become
 * empty ones and nothing else — because a fabricated score or constraint in a
 * demo would misrepresent the system.
 */

import type { GeoView, HealthResponse, OpenAtView, SearchResponseModel } from "@/lib/types";

/** Thrown for every failed search; `message` is safe to render. */
export class SearchError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "SearchError";
    this.status = status;
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

/** Only the three documented values are trusted; anything else is "unknown". */
function asStatus(value: unknown): "satisfied" | "violated" | "unknown" {
  return value === "satisfied" || value === "violated" ? value : "unknown";
}

function normaliseConstraint(raw: unknown) {
  const record = asRecord(raw);
  const name = asString(record.name, "constraint");
  const status = asStatus(record.status);
  return {
    name,
    requested: record.requested ?? null,
    actual: record.actual ?? null,
    status,
    // The backend owns this wording. The fallback is only for a malformed
    // payload, and says nothing about whether the constraint passed.
    explanation: asString(record.explanation, name),
  };
}

function normaliseNumberMap(raw: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(asRecord(raw))) {
    if (typeof value === "number" && Number.isFinite(value)) out[key] = value;
  }
  return out;
}

/**
 * `open_at` and `geo` arrive as loose dicts (`dict[str, Any]` on the wire), so
 * each field is checked individually rather than cast. A missing half renders
 * as a shorter label; it never renders as "undefined".
 */
function normaliseOpenAt(raw: unknown): OpenAtView | null {
  if (raw === null || raw === undefined) return null;
  const record = asRecord(raw);
  const out: OpenAtView = {};
  if (typeof record.day === "string") out.day = record.day;
  if (typeof record.minute_of_day === "number") out.minute_of_day = record.minute_of_day;
  return out;
}

function normaliseGeo(raw: unknown): GeoView | null {
  if (raw === null || raw === undefined) return null;
  const record = asRecord(raw);
  const out: GeoView = {};
  if (typeof record.latitude === "number") out.latitude = record.latitude;
  if (typeof record.longitude === "number") out.longitude = record.longitude;
  if (typeof record.radius_km === "number") out.radius_km = record.radius_km;
  if (typeof record.label === "string") out.label = record.label;
  return out;
}

function normaliseResult(raw: unknown, index: number) {
  const record = asRecord(raw);
  const business = asRecord(record.business);
  const constraints = asRecord(record.constraints);

  return {
    // Falling back to position keeps the list numbered sensibly; it is a
    // display index, not a claim about the ranker.
    rank: asNumber(record.rank, index + 1),
    score: asNumber(record.score, Number.NaN),
    business: {
      business_id: asString(business.business_id, `unknown-${index}`),
      name: asString(business.name, "Unnamed business"),
      city: asString(business.city, ""),
      state: asString(business.state, ""),
      latitude: typeof business.latitude === "number" ? business.latitude : null,
      longitude: typeof business.longitude === "number" ? business.longitude : null,
      stars: asNumber(business.stars, Number.NaN),
      review_count: asNumber(business.review_count, 0),
      categories: asArray(business.categories).map((c) => asString(c, "")).filter(Boolean),
      attributes: asRecord(business.attributes),
    },
    constraints: {
      matched: asArray(constraints.matched).map(normaliseConstraint),
      unmatched: asArray(constraints.unmatched).map(normaliseConstraint),
      unknown: asArray(constraints.unknown).map(normaliseConstraint),
    },
    stage_scores: normaliseNumberMap(record.stage_scores),
  };
}

function normaliseResponse(raw: unknown, query: string): SearchResponseModel {
  const record = asRecord(raw);
  const parsed = asRecord(record.parsed);
  const timings = asRecord(record.timings);
  const source = parsed.source;

  return {
    query: asString(record.query, query),
    parsed: {
      semantic_text: asString(parsed.semantic_text, ""),
      open_at: normaliseOpenAt(parsed.open_at),
      attributes: asRecord(parsed.attributes),
      price_max: typeof parsed.price_max === "number" ? parsed.price_max : null,
      categories: asArray(parsed.categories).map((c) => asString(c, "")).filter(Boolean),
      geo: normaliseGeo(parsed.geo),
      source: source === "llm" || source === "cache" ? source : "rules",
      parse_latency_ms: asNumber(parsed.parse_latency_ms, Number.NaN),
      n_hard_constraints: asNumber(parsed.n_hard_constraints, 0),
    },
    results: asArray(record.results).map(normaliseResult),
    timings: {
      total_ms: asNumber(timings.total_ms, Number.NaN),
      stages: asArray(timings.stages)
        .map((stage) => {
          const s = asRecord(stage);
          return { stage: asString(s.stage, ""), ms: asNumber(s.ms, Number.NaN) };
        })
        .filter((s) => s.stage.length > 0),
    },
    candidate_counts: normaliseNumberMap(record.candidate_counts),
    warnings: asArray(record.warnings).map((w) => asString(w, "")).filter(Boolean),
  };
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    const detail = asRecord(body).detail;
    if (typeof detail === "string" && detail.trim().length > 0) return detail;
  } catch {
    // Fall through.
  }
  return `The search failed (HTTP ${response.status}).`;
}

export async function search(query: string, limit = 10): Promise<SearchResponseModel> {
  let response: Response;
  try {
    response = await fetch("/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, limit }),
    });
  } catch {
    // Same-origin request to this app's own handler: if even this throws, the
    // Next server itself is gone, so the backend's address is irrelevant here.
    throw new SearchError(
      "Lost the connection to the demo server. Check that `npm run dev` is still running.",
      0,
    );
  }

  if (!response.ok) {
    throw new SearchError(await readDetail(response), response.status);
  }

  try {
    return normaliseResponse(await response.json(), query);
  } catch {
    throw new SearchError("The search API returned a response this UI could not read.", 502);
  }
}

/**
 * Best-effort backend health. Returns `null` rather than throwing: the health
 * banner is a courtesy, and a failure to fetch it must not block searching.
 */
export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) return null;
    const record = asRecord(await response.json());
    const status = record.status;
    return {
      status: status === "ok" || status === "degraded" ? status : "unavailable",
      ready: record.ready === true,
      artifacts: asArray(record.artifacts).map((a) => {
        const artifact = asRecord(a);
        return {
          name: asString(artifact.name, "artifact"),
          loaded: artifact.loaded === true,
          detail: asString(artifact.detail, ""),
        };
      }),
      warnings: asArray(record.warnings).map((w) => asString(w, "")).filter(Boolean),
      n_businesses: typeof record.n_businesses === "number" ? record.n_businesses : null,
      n_documents: typeof record.n_documents === "number" ? record.n_documents : null,
      variant: typeof record.variant === "string" ? record.variant : null,
      config_fingerprint:
        typeof record.config_fingerprint === "string" ? record.config_fingerprint : null,
      error: typeof record.error === "string" ? record.error : null,
    };
  } catch {
    return null;
  }
}
