import { NextResponse } from "next/server";

import { API_TIMEOUT_MS, apiBaseUrl, apiUrl } from "@/lib/server-env";

/**
 * Server-side proxy for `POST /search`.
 *
 * The browser posts here; this forwards to the FastAPI service. The value it
 * adds over a Next `rewrites` proxy is error translation: every way the
 * backend can be unreachable is mapped to a short, actionable sentence, so the
 * demo degrades to a readable banner instead of a failed fetch in the console.
 *
 * Request validation is deliberately *not* duplicated here. `SearchRequest` in
 * `serve/schemas.py` already rejects an empty query, an over-long one, or an
 * out-of-range limit with a 422 naming the field. Re-implementing those bounds
 * in TypeScript would create two sources of truth that drift; the body is
 * passed through and the 422 is forwarded.
 */

/** Dynamic because a search result must never be cached as a static response. */
export const dynamic = "force-dynamic";

interface UpstreamFailure {
  status: number;
  detail: string;
}

/**
 * Turn a thrown fetch error into something a viewer can act on.
 *
 * Node's fetch reports a refused connection as an opaque `TypeError: fetch
 * failed` with the real reason on `.cause`, so the cause code is what actually
 * distinguishes "the API is not running" from "the hostname is wrong".
 */
function describeTransportFailure(error: unknown): UpstreamFailure {
  const base = apiBaseUrl();

  if (error instanceof DOMException && error.name === "AbortError") {
    return {
      status: 504,
      detail: `The search API did not respond within ${Math.round(
        API_TIMEOUT_MS / 1000,
      )}s. It may still be loading models — try again in a moment.`,
    };
  }

  const code =
    typeof error === "object" && error !== null && "cause" in error
      ? (error as { cause?: { code?: string } }).cause?.code
      : undefined;

  if (code === "ECONNREFUSED") {
    return {
      status: 503,
      // Mentioning 127.0.0.1 is not noise: Node resolves `localhost` to ::1
      // first on many Macs, while uvicorn binds 127.0.0.1 by default, which
      // produces exactly this error with the server running fine.
      detail:
        `Cannot reach the search API at ${base}. Start it with \`make serve\`. ` +
        `If the server is already running, set CONCIERGE_API_URL=http://127.0.0.1:8000 ` +
        `in web/.env.local — Node may be resolving localhost to IPv6.`,
    };
  }

  if (code === "ENOTFOUND" || code === "EAI_AGAIN") {
    return {
      status: 503,
      detail: `Cannot resolve the search API host in ${base}. Check CONCIERGE_API_URL in web/.env.local.`,
    };
  }

  return {
    status: 503,
    detail: `Cannot reach the search API at ${base}. Start it with \`make serve\`.`,
  };
}

/**
 * Read an error body without trusting it.
 *
 * FastAPI sends `{"detail": ...}`, but a 502 from something in between may
 * send HTML. Only a string `detail` is forwarded; anything else falls back to
 * a generic line, because dumping an HTML page into a UI banner is the
 * stack-trace-on-a-projector outcome this handler exists to prevent.
 */
async function readUpstreamDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail.trim().length > 0) return detail;
      // A 422 from pydantic carries a list of field errors.
      if (Array.isArray(detail)) {
        const messages = detail
          .map((item) =>
            typeof item === "object" && item !== null && "msg" in item
              ? String((item as { msg: unknown }).msg)
              : null,
          )
          .filter((m): m is string => m !== null);
        if (messages.length > 0) return messages.join("; ");
      }
    }
  } catch {
    // Fall through to the generic message below.
  }
  return `The search API returned ${response.status} ${response.statusText || "error"}.`;
}

export async function POST(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Expected a JSON body." }, { status: 400 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

  try {
    const upstream = await fetch(apiUrl("/search"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
      cache: "no-store",
    });

    if (!upstream.ok) {
      const detail = await readUpstreamDetail(upstream);
      return NextResponse.json({ detail }, { status: upstream.status });
    }

    // Forwarded verbatim: the response shape is the API's contract, and
    // reshaping it here would give the UI a second contract to track.
    const payload: unknown = await upstream.json();
    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    const failure = describeTransportFailure(error);
    return NextResponse.json({ detail: failure.detail }, { status: failure.status });
  } finally {
    clearTimeout(timeout);
  }
}
