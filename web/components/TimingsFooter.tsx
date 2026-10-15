import { formatMs, humanizeKey, humanizeStage } from "@/lib/format";
import type { SearchResponseModel } from "@/lib/types";

/**
 * Per-search latency and candidate counts, in small type under the results.
 *
 * Unobtrusive by design but always present: the pitch includes a latency
 * story, and a demo that only shows latency when it is good is not evidence.
 * Every number here is measured by the pipeline and passed through untouched —
 * nothing on this page is timed in the browser, because that would mix network
 * and render time into a figure the README attributes to the cascade.
 */

/**
 * `candidate_counts` mixes two kinds of key: survivor counts per stage, and
 * diagnostics the filter stage adds under a `dropped_` prefix (plus
 * `penalized_missing_data`). They are separated so the headline reads as the
 * cascade — 200 → 50 → 10 — with the reasons available but not in the way.
 */
function splitCounts(counts: Record<string, number>) {
  const survivors: [string, number][] = [];
  const diagnostics: [string, number][] = [];

  for (const entry of Object.entries(counts)) {
    const [key] = entry;
    if (key.startsWith("dropped_") || key === "penalized_missing_data") {
      diagnostics.push(entry);
    } else {
      survivors.push(entry);
    }
  }
  return { survivors, diagnostics };
}

export function TimingsFooter({ response }: { response: SearchResponseModel }) {
  const { timings, candidate_counts } = response;
  const { survivors, diagnostics } = splitCounts(candidate_counts);

  return (
    <section
      aria-label="Search timings and candidate counts"
      className="mt-4 border-t border-slate-200 pt-3 text-xs text-slate-500"
    >
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="font-medium text-slate-700">
          {`Total ${formatMs(timings.total_ms)}`}
        </span>
        {timings.stages.map((stage) => (
          <span key={stage.stage}>
            {humanizeStage(stage.stage)} <span className="font-mono">{formatMs(stage.ms)}</span>
          </span>
        ))}
      </div>

      {survivors.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <span className="text-slate-400">Candidates</span>
          {survivors.map(([stage, count]) => (
            <span key={stage}>
              {humanizeStage(stage)} <span className="font-mono">{count.toLocaleString()}</span>
            </span>
          ))}
        </div>
      )}

      {diagnostics.length > 0 && (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-slate-400 hover:text-slate-700">
            Filter detail
          </summary>
          <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
            {diagnostics.map(([key, count]) => (
              <li key={key}>
                {humanizeKey(key.replace(/^dropped_/, "dropped: "))}{" "}
                <span className="font-mono">{count.toLocaleString()}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <p className="mt-1.5 text-slate-400">
        Measured inside the pipeline; excludes browser and network time.
      </p>
    </section>
  );
}
