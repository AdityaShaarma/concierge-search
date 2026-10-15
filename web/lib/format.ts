/**
 * Display formatting. Pure functions, no React, so each one can be reasoned
 * about (and later unit-tested) on its own.
 *
 * Everything here is defensive about shape. The API carries several fields as
 * `Any` (constraint values, attributes, `open_at`), and this is a demo that
 * will be run live: one odd attribute value from the dataset must degrade to
 * plain text, never to a blank screen.
 */

/** Keys whose dataset spelling is worse than a hand-written label. */
const ATTRIBUTE_LABELS: Record<string, string> = {
  // Yelp's own name for the price band, with a trailing "2" that would look
  // like a typo in the UI.
  RestaurantsPriceRange2: "Price",
  WiFi: "WiFi",
  BusinessAcceptsCreditCards: "Accepts cards",
  RestaurantsTakeOut: "Takeout",
  RestaurantsDelivery: "Delivery",
  RestaurantsReservations: "Reservations",
  GoodForKids: "Good for kids",
  OutdoorSeating: "Outdoor seating",
  NoiseLevel: "Noise level",
  // Synthesised constraint names the pipeline may emit alongside attributes.
  open_at: "Open at",
  price_max: "Price",
  geo: "Location",
  categories: "Category",
};

/**
 * Turn a dataset key into something readable.
 *
 * Yelp attribute keys are PascalCase ("OutdoorSeating"), pipeline constraint
 * names are snake_case ("open_at"), and a few are neither. Rather than a
 * lookup table that has to be exhaustive, the table above covers the keys
 * worth hand-labelling and this falls back to mechanical splitting — so a new
 * attribute added upstream still renders acceptably instead of not at all.
 */
export function humanizeKey(key: string): string {
  const mapped = ATTRIBUTE_LABELS[key];
  if (mapped) return mapped;

  const spaced = key
    .replace(/_/g, " ")
    // Split PascalCase / camelCase without breaking runs of capitals.
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .trim();

  if (!spaced) return key;
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * Render an arbitrary JSON value as short display text.
 *
 * Mirrors the backend's `_jsonable` fallback-to-string philosophy: anything
 * unexpected becomes text rather than an error. Returns `null` for values
 * that carry no information worth a badge, so callers can skip them.
 */
export function formatValue(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (Array.isArray(value)) {
    const parts = value.map(formatValue).filter((p): p is string => p !== null);
    return parts.length > 0 ? parts.join(", ") : null;
  }
  if (typeof value === "object") {
    // Nested attribute dicts, e.g. Ambience {"romantic": false, "casual": true}.
    // Only the true keys are informative; listing the false ones would read as
    // a list of things the place is not.
    const entries = Object.entries(value as Record<string, unknown>);
    const truthy = entries.filter(([, v]) => v === true).map(([k]) => humanizeKey(k).toLowerCase());
    if (truthy.length > 0) return truthy.join(", ");
    const rendered = entries
      .map(([k, v]) => {
        const formatted = formatValue(v);
        return formatted === null ? null : `${humanizeKey(k).toLowerCase()}: ${formatted}`;
      })
      .filter((p): p is string => p !== null);
    return rendered.length > 0 ? rendered.join(", ") : null;
  }
  return String(value);
}

/** Minutes-from-midnight to "22:00". Values outside a day are clamped, not hidden. */
export function formatMinuteOfDay(minute: number): string {
  if (!Number.isFinite(minute)) return "unknown time";
  const clamped = Math.max(0, Math.min(24 * 60 - 1, Math.round(minute)));
  const hours = Math.floor(clamped / 60);
  const minutes = clamped % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

/**
 * A price band (1-4) as dollar signs, with the digit kept for screen readers
 * via the caller's label. Out-of-range values fall back to the number so a
 * surprising value is visible rather than silently dropped.
 */
export function formatPriceBand(band: unknown): string | null {
  const n = typeof band === "number" ? band : Number(band);
  if (!Number.isFinite(n)) return formatValue(band);
  if (n < 1 || n > 4) return String(n);
  return "$".repeat(Math.round(n));
}

/**
 * Milliseconds for display. Sub-millisecond stages (cache hits) would read as
 * "0 ms" at integer precision, which looks like a missing measurement, so
 * small values keep a decimal place.
 */
export function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return "–";
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

/** Stage keys from the pipeline, as short titles for the timing footer. */
export function humanizeStage(stage: string): string {
  const labels: Record<string, string> = {
    parse: "Query parsing",
    query_parse: "Query parsing",
    retrieve: "Hybrid retrieval",
    retrieval: "Hybrid retrieval",
    bm25: "BM25",
    dense: "Dense",
    fuse: "RRF fusion",
    fusion: "RRF fusion",
    filter: "Hard filters",
    filters: "Hard filters",
    rerank: "Cross-encoder",
    cross_encoder: "Cross-encoder",
    ltr: "LambdaMART",
    total: "Total",
  };
  return labels[stage] ?? humanizeKey(stage);
}

/** "4.5 stars from 182 reviews" — spelled out for screen readers. */
export function formatRatingLabel(stars: number, reviewCount: number): string {
  const plural = reviewCount === 1 ? "review" : "reviews";
  return `${stars} out of 5 stars from ${reviewCount.toLocaleString()} ${plural}`;
}

/** How the parse was produced, phrased for a viewer who has not read the code. */
export function describeParseSource(source: string): string {
  const labels: Record<string, string> = {
    llm: "parsed by the LLM",
    cache: "served from the parse cache",
    rules: "parsed by the rule-based fallback",
  };
  return labels[source] ?? `parse source: ${source}`;
}
