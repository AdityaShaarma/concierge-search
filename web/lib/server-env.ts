/**
 * Server-only configuration for reaching the FastAPI backend.
 *
 * Note what is *not* here: a `NEXT_PUBLIC_` variable. The browser never talks
 * to the Python API directly — it talks to this app's own route handlers,
 * which forward the request. Two reasons:
 *
 *   1. CORS. The demo is run on a laptop with the API on :8000 and the UI on
 *      :3000. Same-origin requests from the browser mean the API needs no CORS
 *      configuration to make the demo work.
 *   2. Failure shaping. A route handler can turn "connection refused" into a
 *      sentence a viewer can read. A direct browser fetch to a dead port
 *      cannot — see the comment in `next.config.ts`.
 */

/** Default matches `make serve`, so the demo works with no .env.local at all. */
export const DEFAULT_API_BASE_URL = "http://localhost:8000";

/**
 * How long to wait for the backend before giving up.
 *
 * Generous on purpose: the first search after startup can be slow (model
 * warm-up, lazy index load), and a demo that times out on its first query is
 * worse than one that takes a few seconds. This is a UI patience setting, not
 * a latency measurement — the numbers in the footer come from the pipeline's
 * own instrumentation, never from a timer here.
 */
export const API_TIMEOUT_MS = 30_000;

/** Trailing slashes would produce `//search`, which some servers 404. */
export function apiBaseUrl(): string {
  const configured = process.env.CONCIERGE_API_URL?.trim();
  const base = configured && configured.length > 0 ? configured : DEFAULT_API_BASE_URL;
  return base.replace(/\/+$/, "");
}

export function apiUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${apiBaseUrl()}${suffix}`;
}
