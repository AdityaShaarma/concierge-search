import type { ReactNode } from "react";

import { humanizeKey } from "@/lib/format";
import type { ConstraintGroups, ConstraintStatus, ConstraintView } from "@/lib/types";

/**
 * "Why this result" — the feature this demo exists to show.
 *
 * The one rule in this file: the three constraint states must be
 * distinguishable without colour. Each badge therefore carries an icon *and* a
 * word ("met" / "not met" / "no data"), and the "no data" state is styled
 * neutrally rather than as a lesser shade of failure. Showing an unknown as a
 * failure would be a lie about the data — the dataset's attribute coverage is
 * sparse, and treating a missing NoiseLevel as "loud" is precisely the bug the
 * three-valued constraint logic was built to avoid.
 *
 * Icons are inline SVG rather than an icon library: three glyphs are not worth
 * a dependency, and inline paths cannot fail to load during a live demo.
 */

interface StateStyle {
  /** The word shown next to the icon. Never colour alone. */
  label: string;
  /** Longer phrasing for screen readers and tooltips. */
  description: string;
  container: string;
  badge: string;
  icon: ReactNode;
}

const CHECK_ICON = (
  <svg viewBox="0 0 20 20" className="h-3.5 w-3.5 shrink-0" aria-hidden="true" focusable="false">
    <path
      d="M4 10.5l4 4 8-9"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const CROSS_ICON = (
  <svg viewBox="0 0 20 20" className="h-3.5 w-3.5 shrink-0" aria-hidden="true" focusable="false">
    <path
      d="M5 5l10 10M15 5L5 15"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
    />
  </svg>
);

/**
 * A dash in a dashed circle, not a question mark in a warning triangle: this
 * state is an absence of information, not a problem with the result.
 */
const NO_DATA_ICON = (
  <svg viewBox="0 0 20 20" className="h-3.5 w-3.5 shrink-0" aria-hidden="true" focusable="false">
    <circle
      cx="10"
      cy="10"
      r="7"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeDasharray="2.6 2.4"
    />
    <path d="M6.5 10h7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);

const STATE_STYLES: Record<ConstraintStatus, StateStyle> = {
  satisfied: {
    label: "met",
    description: "This result satisfies the constraint",
    container: "border-met-border bg-met-bg",
    badge: "bg-white text-met-text border border-met-border",
    icon: CHECK_ICON,
  },
  violated: {
    label: "not met",
    description: "This result violates the constraint",
    container: "border-unmet-border bg-unmet-bg",
    badge: "bg-white text-unmet-text border border-unmet-border",
    icon: CROSS_ICON,
  },
  unknown: {
    label: "no data",
    description:
      "The dataset has no value for this constraint on this business — it is neither met nor violated",
    // Dashed border: a second non-colour cue that this row is a different kind
    // of thing from the two above it.
    container: "border-dashed border-nodata-border bg-nodata-bg",
    badge: "bg-white text-nodata-text border border-dashed border-nodata-border",
    icon: NO_DATA_ICON,
  },
};

function ConstraintRow({ constraint }: { constraint: ConstraintView }) {
  const style = STATE_STYLES[constraint.status];

  return (
    <li
      className={`flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border px-2.5 py-1.5 text-sm ${style.container}`}
    >
      <span
        className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${style.badge}`}
        title={style.description}
      >
        {style.icon}
        {style.label}
        <span className="sr-only-live">{`: ${style.description}.`}</span>
      </span>
      <span className="font-medium text-slate-900">{humanizeKey(constraint.name)}</span>
      {/*
        The backend's phrasing, not ours. `explanation` is built by the
        constraint logic so that "wanted $$, has $$$" and "no data" cannot
        drift between the ranker and the UI.
      */}
      <span className="text-slate-600">{constraint.explanation}</span>
    </li>
  );
}

/**
 * All three buckets, in a fixed order: met, not met, no data. Fixed because
 * position then becomes a fourth cue, and because the sparse-data rows belong
 * last — they are context, not findings.
 */
export function ConstraintList({ constraints }: { constraints: ConstraintGroups }) {
  const total =
    constraints.matched.length + constraints.unmatched.length + constraints.unknown.length;

  if (total === 0) {
    return (
      <p className="text-sm text-slate-500">
        No hard constraints were extracted from this query, so this result was ranked on semantic
        relevance alone.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <ul className="space-y-1.5">
        {/*
          Keys include the position as well as the name: nothing guarantees
          constraint names are unique within a bucket (two attribute checks can
          share a name), and a duplicate key warning during a live demo is the
          kind of thing that gets noticed.
        */}
        {constraints.matched.map((c, i) => (
          <ConstraintRow key={`met-${i}-${c.name}`} constraint={c} />
        ))}
        {constraints.unmatched.map((c, i) => (
          <ConstraintRow key={`unmet-${i}-${c.name}`} constraint={c} />
        ))}
        {constraints.unknown.map((c, i) => (
          <ConstraintRow key={`nodata-${i}-${c.name}`} constraint={c} />
        ))}
      </ul>

      {constraints.unknown.length > 0 && (
        <p className="text-xs leading-relaxed text-slate-500">
          &ldquo;No data&rdquo; means the dataset has no value for that attribute on this business.
          It is reported separately from a violation, and is not treated as one.
        </p>
      )}
    </div>
  );
}

/**
 * A one-line summary for the collapsed/header position: "2 met · 1 not met ·
 * 1 no data". Text, not a coloured bar, so the counts survive a screenshot in
 * greyscale and a screen reader.
 */
export function ConstraintSummary({ constraints }: { constraints: ConstraintGroups }) {
  const parts: string[] = [];
  if (constraints.matched.length > 0) parts.push(`${constraints.matched.length} met`);
  if (constraints.unmatched.length > 0) parts.push(`${constraints.unmatched.length} not met`);
  if (constraints.unknown.length > 0) parts.push(`${constraints.unknown.length} no data`);

  if (parts.length === 0) return null;

  return (
    <span className="text-xs font-medium text-slate-500">
      {parts.join(" · ")}
    </span>
  );
}
