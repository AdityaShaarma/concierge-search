"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ParsedQueryPanel } from "@/components/ParsedQueryPanel";
import { ResultCard } from "@/components/ResultCard";
import { SearchForm } from "@/components/SearchForm";
import {
  ErrorBanner,
  HealthBanner,
  IdleState,
  NoResultsState,
  ResultsSkeleton,
  WarningBanner,
} from "@/components/StateViews";
import { TimingsFooter } from "@/components/TimingsFooter";
import { fetchHealth, search, SearchError } from "@/lib/api";
import type { HealthResponse, SearchResponseModel } from "@/lib/types";

/**
 * The only stateful component in the app.
 *
 * Plain `useState` and `fetch` rather than a data-fetching library: there is
 * one endpoint, one in-flight request at a time, and no cache to invalidate.
 * A client-side cache would actively hurt the demo — the parse cache and the
 * latency figures are things we want to *observe*, and serving a repeat query
 * from React state would hide the backend's own cache hit.
 */

type Phase = "idle" | "searching" | "done" | "error";

export function SearchExperience() {
  const [query, setQuery] = useState("");
  /** The query the displayed response belongs to; needed for the empty state. */
  const [submitted, setSubmitted] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [response, setResponse] = useState<SearchResponseModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  /**
   * Monotonic request id. Clicking two example chips quickly can land the
   * responses out of order; without this the slower first response would
   * overwrite the faster second one and the page would show results for a
   * query the box no longer contains.
   */
  const requestId = useRef(0);

  useEffect(() => {
    let cancelled = false;
    // Best effort, once on mount: a dead or degraded backend should be visible
    // before the viewer types rather than after their first search fails.
    void fetchHealth().then((result) => {
      if (!cancelled) setHealth(result);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const runSearch = useCallback(async (raw: string) => {
    const trimmed = raw.trim();
    if (trimmed.length === 0) return;

    const id = requestId.current + 1;
    requestId.current = id;

    setSubmitted(trimmed);
    setPhase("searching");
    setError(null);

    try {
      const result = await search(trimmed);
      if (requestId.current !== id) return; // A newer search has started.
      setResponse(result);
      setPhase("done");
    } catch (caught) {
      if (requestId.current !== id) return;
      // Only `SearchError.message` is ever rendered; anything else becomes a
      // generic line, so no exception text can reach the screen.
      setError(
        caught instanceof SearchError
          ? caught.message
          : "Something went wrong running that search.",
      );
      setPhase("error");
    }
  }, []);

  const handleExample = useCallback(
    (example: string) => {
      setQuery(example);
      void runSearch(example);
    },
    [runSearch],
  );

  const results = response?.results ?? [];
  const showResults = phase === "done" && response !== null;

  /**
   * Warnings specific to this response, with the ones the standing health
   * banner already shows removed.
   *
   * The health check and every search report the same standing degradations, so
   * rendering both verbatim duplicated the banner on screen. Subtracting rather
   * than dropping the per-response banner entirely matters: a warning that
   * appears for one query and not another (a parse fallback, say) is genuinely
   * about that search and still has to surface.
   */
  const healthWarnings = new Set(health?.warnings ?? []);
  const newWarnings = (response?.warnings ?? []).filter((w) => !healthWarnings.has(w));

  /** One sentence for the live region, so the state change is announced once. */
  const statusMessage =
    phase === "searching"
      ? `Searching for ${submitted}`
      : phase === "error"
        ? "The search failed."
        : showResults
          ? `${results.length} ${results.length === 1 ? "result" : "results"} for ${submitted}`
          : "";

  return (
    <div className="space-y-6">
      <HealthBanner health={health} />

      <SearchForm
        value={query}
        onChange={setQuery}
        onSubmit={() => void runSearch(query)}
        onExample={handleExample}
        isSearching={phase === "searching"}
      />

      {/*
        Politeness matters here: `assertive` would interrupt a screen reader
        mid-sentence on every keystroke-triggered state change. The error
        banner itself is `role="alert"` for the case that genuinely warrants
        interrupting.
      */}
      <p role="status" aria-live="polite" className="sr-only-live">
        {statusMessage}
      </p>

      {phase === "idle" && <IdleState />}

      {phase === "searching" && (
        <div className="space-y-3">
          <p className="text-sm text-slate-500">Searching…</p>
          <ResultsSkeleton />
        </div>
      )}

      {phase === "error" && error !== null && (
        <ErrorBanner message={error} onRetry={() => void runSearch(submitted)} />
      )}

      {showResults && (
        <div className="space-y-4">
          {/*
            Only the warnings this response adds. The HealthBanner above already
            states the standing degradations (no FAISS index, no trained models),
            and they come through on every response too — rendering both put the
            same three bullets on screen twice, which reads as a layout bug and,
            worse, trains the viewer to scroll past the one banner that must
            never be ignored.
          */}
          <WarningBanner warnings={newWarnings} />

          {/* Stage 1 first: how the request was read explains everything below it. */}
          <ParsedQueryPanel parsed={response.parsed} />

          {results.length > 0 ? (
            <section aria-label="Results">
              <ul className="space-y-3">
                {results.map((result) => (
                  <ResultCard key={result.business.business_id} result={result} />
                ))}
              </ul>
            </section>
          ) : (
            <NoResultsState query={submitted} />
          )}

          <TimingsFooter response={response} />
        </div>
      )}
    </div>
  );
}
