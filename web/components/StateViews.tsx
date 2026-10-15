import type { HealthResponse } from "@/lib/types";

/**
 * The non-result states: loading, nothing searched yet, nothing found, a dead
 * backend, a degraded backend.
 *
 * These are in one file because they are one decision — what the page shows
 * when it cannot show results — and a demo is judged on these as much as on
 * the happy path. The requirement they all answer: never a stack trace, and
 * never a fabricated result.
 */

/**
 * Loading placeholder.
 *
 * Deliberately contains no text and no fake business names, ratings or
 * constraint rows. A skeleton with plausible sample data in it would put
 * numbers on screen that no measurement produced, and a screenshot taken a
 * moment too early would be a screenshot of fiction. Grey blocks cannot lie.
 *
 * `aria-hidden` because it carries no information; the "Searching…" status is
 * announced from the live region in `SearchExperience`.
 */
export function ResultsSkeleton({ count = 3 }: { count?: number }) {
  return (
    <ul aria-hidden="true" className="space-y-3">
      {Array.from({ length: count }, (_, index) => (
        <li
          key={index}
          className="animate-pulse rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
        >
          <div className="flex items-start gap-3">
            <div className="h-7 w-7 shrink-0 rounded-full bg-slate-200" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-1/2 rounded bg-slate-200" />
              <div className="h-3 w-1/3 rounded bg-slate-100" />
            </div>
          </div>
          <div className="mt-4 space-y-1.5">
            <div className="h-6 w-5/6 rounded bg-slate-100" />
            <div className="h-6 w-2/3 rounded bg-slate-100" />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Before the first search. Explains the one thing worth looking at. */
export function IdleState() {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white px-5 py-8 text-center">
      <p className="text-sm font-medium text-slate-900">No search yet</p>
      <p className="mx-auto mt-1.5 max-w-md text-sm leading-relaxed text-slate-500">
        Type a request or pick an example. Every result will show which parts of the request it
        satisfies, which it violates, and which the dataset has no data for.
      </p>
    </div>
  );
}

/**
 * A search that returned nothing.
 *
 * This is a legitimate outcome of hard filtering, not a failure, so it says
 * what to try instead of apologising. The actual filter drop counts are shown
 * by the timings footer, which the caller renders alongside this.
 */
export function NoResultsState({ query }: { query: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-5 py-8 text-center">
      <p className="text-sm font-medium text-slate-900">No places matched</p>
      <p className="mx-auto mt-1.5 max-w-md text-sm leading-relaxed text-slate-500">
        Nothing in the index satisfies every hard filter extracted from{" "}
        <span className="font-medium text-slate-700">&ldquo;{query}&rdquo;</span>. The filter
        detail below shows how many candidates each filter removed — loosening the most
        restrictive one (often a late opening time) is usually enough.
      </p>
    </div>
  );
}

/**
 * A failed search. `role="alert"` so it is announced immediately: the user is
 * waiting on this.
 *
 * The message always comes from the route handler, which only ever produces
 * short prose. Nothing here renders an exception or a response body.
 */
export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-unmet-border bg-unmet-bg px-4 py-3 text-sm text-unmet-text"
    >
      <p className="font-semibold">Search failed</p>
      <p className="mt-1 leading-relaxed">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 rounded border border-unmet-border bg-white px-2.5 py-1 text-xs font-medium text-unmet-text hover:bg-rose-50"
        >
          Try again
        </button>
      )}
    </div>
  );
}

/**
 * Warnings attached to a successful response, e.g. "no fine-tuned
 * cross-encoder loaded".
 *
 * Shown above the results, not hidden in a tooltip. CLAUDE.md's rule: a
 * deployment serving retrieval order while the README claims a fine-tuned
 * reranker is the worst failure for this project, because it looks like
 * success.
 */
export function WarningBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;

  return (
    <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
      <p className="font-semibold">This system is degraded</p>
      <ul className="mt-1 list-disc space-y-0.5 pl-5 leading-relaxed">
        {warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
      <p className="mt-1.5 text-xs">
        Results are real, but not from the full four-stage system.
      </p>
    </div>
  );
}

/**
 * Backend status, checked once on load so a dead API is visible before the
 * viewer types. Returns nothing when the backend is fully healthy — a green
 * "all good" bar would just be chrome.
 */
export function HealthBanner({ health }: { health: HealthResponse | null }) {
  if (health === null || health.status === "ok") return null;

  if (!health.ready) {
    return (
      <div
        role="status"
        className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"
      >
        <p className="font-semibold">The search API is not answering</p>
        <p className="mt-1 leading-relaxed">
          {health.error ?? "No pipeline is loaded, so searches will fail."} Start the API with{" "}
          <code className="rounded bg-white px-1 py-0.5 font-mono text-xs">make serve</code> and
          reload.
        </p>
      </div>
    );
  }

  return <WarningBanner warnings={health.warnings} />;
}
