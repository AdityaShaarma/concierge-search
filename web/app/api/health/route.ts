import { NextResponse } from "next/server";

import { apiBaseUrl, apiUrl } from "@/lib/server-env";
import type { HealthResponse } from "@/lib/types";

/**
 * Server-side proxy for `GET /health`.
 *
 * The page calls this on load so it can say "the API is not running" before
 * the viewer types a query, rather than after. It also surfaces a *degraded*
 * backend — one serving retrieval order because no fine-tuned reranker is on
 * disk. CLAUDE.md is explicit that an unnoticed degradation is the worst
 * failure mode for this project, since it looks like success.
 */

export const dynamic = "force-dynamic";

/** Short timeout: this is a pre-flight check, not the search itself. */
const HEALTH_TIMEOUT_MS = 4_000;

export async function GET(): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);

  try {
    const upstream = await fetch(apiUrl("/health"), {
      signal: controller.signal,
      cache: "no-store",
    });

    // A /health that answers with an error status is still an answer: the body
    // carries which artifacts are missing, which is exactly what we want to
    // show. Only an unparseable body is treated as unreachable.
    const payload: unknown = await upstream.json();
    return NextResponse.json(payload, { status: 200 });
  } catch {
    // Synthesised, and labelled as such by `error`: this is the UI's own
    // conclusion that nothing answered, not a report from the server.
    const unreachable: HealthResponse = {
      status: "unavailable",
      ready: false,
      artifacts: [],
      warnings: [],
      error: `No response from the search API at ${apiBaseUrl()}.`,
    };
    return NextResponse.json(unreachable, { status: 200 });
  } finally {
    clearTimeout(timeout);
  }
}
