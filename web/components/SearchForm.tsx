"use client";

import { EXAMPLE_QUERIES } from "@/lib/examples";

/**
 * The search box and the one-click example chips.
 *
 * A real `<form>` with a real `<label>`, not a styled div: Enter submits, the
 * label is clickable, and the browser's own autofill and IME behaviour is left
 * alone. `maxLength` mirrors the API's 400-character limit so the field stops
 * the user rather than the server rejecting them after a round trip.
 */

/** Matches `SearchRequest.query` in serve/schemas.py. */
const MAX_QUERY_LENGTH = 400;

interface SearchFormProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  /** Chips both fill the box and run the search — one click, as specified. */
  onExample: (query: string) => void;
  isSearching: boolean;
}

export function SearchForm({ value, onChange, onSubmit, onExample, isSearching }: SearchFormProps) {
  const canSubmit = value.trim().length > 0 && !isSearching;

  return (
    <div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (canSubmit) onSubmit();
        }}
      >
        <label htmlFor="concierge-query" className="block text-sm font-medium text-slate-900">
          What are you looking for?
        </label>

        <div className="mt-2 flex flex-col gap-2 sm:flex-row">
          <input
            id="concierge-query"
            name="query"
            type="text"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            maxLength={MAX_QUERY_LENGTH}
            autoComplete="off"
            // Not `autoFocus`: it steals focus from a screen reader's reading
            // position on load, and the page's first line is the explanation of
            // what this is.
            placeholder="quiet café with wifi that's open past 10pm"
            aria-describedby="concierge-query-hint"
            className="w-full flex-1 rounded-md border border-slate-300 bg-white px-3 py-2.5 text-base text-slate-900 placeholder:text-slate-400 focus:border-sky-600"
          />
          <button
            type="submit"
            disabled={!canSubmit}
            className="shrink-0 rounded-md bg-slate-900 px-5 py-2.5 text-base font-medium text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {isSearching ? "Searching…" : "Search"}
          </button>
        </div>

        <p id="concierge-query-hint" className="mt-1.5 text-xs text-slate-500">
          Plain English. Times, prices, noise level and amenities are turned into hard filters;
          the rest is matched semantically.
        </p>
      </form>

      <div className="mt-4">
        <h2 className="text-xs font-medium uppercase tracking-wide text-slate-500">Try one</h2>
        <ul className="mt-2 flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map((example) => (
            <li key={example.query}>
              <button
                type="button"
                onClick={() => onExample(example.query)}
                disabled={isSearching}
                title={example.demonstrates}
                className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-left text-sm text-slate-700 transition-colors hover:border-slate-400 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {example.query}
                {/* Why this example is interesting — available to a screen
                    reader, and on hover for everyone else. */}
                <span className="sr-only-live">{`. Demonstrates: ${example.demonstrates}`}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
