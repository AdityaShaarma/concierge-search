import {
  describeParseSource,
  formatMinuteOfDay,
  formatMs,
  formatPriceBand,
  formatValue,
  humanizeKey,
} from "@/lib/format";
import type { ParsedQueryView } from "@/lib/types";

/**
 * What Stage 1 understood, shown back to the user.
 *
 * This panel is the first thing a demo viewer should notice, because it is the
 * only visible evidence that query understanding happens at all: without it,
 * the system looks like a search box over an embedding index. The split it shows —
 * semantic text on one side, hard filters on the other — is the actual
 * architectural claim of Stage 1, that "open past 10pm" is arithmetic and
 * "quiet café" is similarity.
 */

interface Chip {
  key: string;
  label: string;
  value: string;
}

/**
 * Flatten the parsed query into display chips.
 *
 * Kept as a pure function above the component so the ordering decision is
 * visible in one place: time, then price, then location, then attributes, then
 * categories — most-restrictive first, which is roughly the order a person
 * reading the query would say them.
 */
function buildFilterChips(parsed: ParsedQueryView): Chip[] {
  const chips: Chip[] = [];

  if (parsed.open_at) {
    const day = parsed.open_at.day;
    const minute = parsed.open_at.minute_of_day;
    const time = typeof minute === "number" ? formatMinuteOfDay(minute) : null;
    // Either half may be absent on the wire, so the label degrades rather than
    // rendering "undefined".
    const value = [day, time].filter(Boolean).join(" ") || "unspecified time";
    chips.push({ key: "open_at", label: "Open", value });
  }

  if (parsed.price_max !== null) {
    const band = formatPriceBand(parsed.price_max);
    chips.push({
      key: "price_max",
      label: "Price",
      value: band ? `${band} or less` : "unspecified",
    });
  }

  if (parsed.geo) {
    const label = parsed.geo.label?.trim();
    const radius = parsed.geo.radius_km;
    const radiusText = typeof radius === "number" ? `within ${radius} km` : null;
    const value = [label || "a point", radiusText].filter(Boolean).join(", ");
    chips.push({ key: "geo", label: "Near", value });
  }

  for (const [key, raw] of Object.entries(parsed.attributes)) {
    const value = formatValue(raw);
    // A null here means the value carried no information (empty string, empty
    // dict). Rendering an empty chip would imply a filter that is not applied.
    if (value === null) continue;
    chips.push({ key: `attr-${key}`, label: humanizeKey(key), value });
  }

  if (parsed.categories.length > 0) {
    chips.push({
      key: "categories",
      label: parsed.categories.length === 1 ? "Category" : "Categories",
      value: parsed.categories.join(", "),
    });
  }

  return chips;
}

/** `rules` is the fallback path, so it is called out rather than shown plain. */
function sourceTone(source: ParsedQueryView["source"]): string {
  if (source === "rules") return "border-amber-300 bg-amber-50 text-amber-900";
  return "border-slate-300 bg-white text-slate-700";
}

export function ParsedQueryPanel({ parsed }: { parsed: ParsedQueryView }) {
  const chips = buildFilterChips(parsed);

  return (
    <section
      aria-labelledby="parsed-query-heading"
      className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="parsed-query-heading" className="text-sm font-semibold text-slate-900">
          How your request was read
        </h2>
        <p className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className={`rounded border px-1.5 py-0.5 font-medium ${sourceTone(parsed.source)}`}>
            {describeParseSource(parsed.source)}
          </span>
          <span>{formatMs(parsed.parse_latency_ms)}</span>
        </p>
      </div>

      <dl className="mt-3 space-y-3">
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Semantic text
          </dt>
          <dd className="mt-1 text-sm text-slate-900">
            {parsed.semantic_text.trim().length > 0 ? (
              <span className="rounded bg-sky-50 px-1.5 py-0.5 font-mono text-[13px] text-sky-900">
                {parsed.semantic_text}
              </span>
            ) : (
              <span className="text-slate-500">
                Empty — the whole request was expressed as hard filters.
              </span>
            )}
            <span className="mt-1 block text-xs text-slate-500">
              Embedded and scored by retrieval and the reranker.
            </span>
          </dd>
        </div>

        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Hard filters
            {/*
              Count the chips actually rendered, not parsed.n_hard_constraints.
              The backend figure deliberately excludes category (it counts
              open_at, attributes, price and geo — see concierge/types.py), while
              this panel does render a category chip. Using it here labelled
              three visible chips as "(2)", which reads as a bug in the parser
              rather than a difference in definition. The number next to a list
              should always describe that list.
            */}
            <span className="ml-1 font-normal normal-case tracking-normal text-slate-400">
              ({chips.length})
            </span>
          </dt>
          <dd className="mt-1">
            {chips.length > 0 ? (
              <ul className="flex flex-wrap gap-1.5">
                {chips.map((chip) => (
                  <li
                    key={chip.key}
                    className="inline-flex items-baseline gap-1 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-[13px]"
                  >
                    <span className="font-medium text-slate-900">{chip.label}:</span>
                    <span className="text-slate-700">{chip.value}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500">
                None extracted. This query is ranked on semantic relevance alone.
              </p>
            )}
            <span className="mt-1 block text-xs text-slate-500">
              Checked arithmetically against each business, not embedded.
            </span>
          </dd>
        </div>
      </dl>
    </section>
  );
}
