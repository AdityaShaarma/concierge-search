import type { ReactNode } from "react";

import { humanizeKey } from "@/lib/format";
import type { ConstraintGroups, ConstraintStatus, ConstraintView } from "@/lib/types";

/**
 * "Why this result" — the feature this demo exists to show.
 *
 * The one rule in this file: the constraint states must be distinguishable
 * without colour. Each badge therefore carries an icon *and* a word, and the
 * undecided states are styled neutrally rather than as a lesser shade of
 * failure. Showing an unknown as a failure would be a lie about the data — the
 * dataset's attribute coverage is sparse, and treating a missing NoiseLevel as
 * "loud" is precisely the bug the three-valued constraint logic was built to
 * avoid.
 *
 * The backend's UNKNOWN covers two different situations, and this file renders
 * them with different words because one of them was being described falsely.
 * `actual === null` means the dataset holds no value at all — "no data". A
 * non-null `actual` means the dataset holds a value that did not settle the
 * request: a category list that does not contain the asked-for category
 * (`filters.check_categories` returns UNKNOWN on a lexical miss on purpose —
 * a miss is "we cannot tell this is the wrong kind of place", not "it is"), an
 * attribute value the matcher could not interpret, or an unparseable price
 * band. Calling those "no data" told the viewer the dataset was silent while
 * the explanation beside it read "no match for Cafes, has Bistros, French".
 * That contradiction sat on the one feature this page exists to demonstrate.
 *
 * One honest caveat about "not met": the served pipeline cannot produce it.
 * Stage 2's filter drops any candidate with a VIOLATED constraint
 * unconditionally (`concierge/retrieval/filters.py`, step 3 of `apply_filters`),
 * so every result that reaches this component has zero violations and
 * `constraints.unmatched` is always empty. The state is rendered anyway because
 * the API contract carries three statuses and the ranking features read all
 * three — but nobody should claim the demo *shows* a violation. It shows that a
 * violation is a reason to be absent.
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

/**
 * Which kind of UNKNOWN this is. Tested against `null`/`undefined` rather than
 * falsiness on purpose: an attribute whose value is `false` is a value, and
 * reading it as absent would put us straight back into the bug above.
 */
function isAbsentFromDataset(constraint: ConstraintView): boolean {
  return constraint.actual === null || constraint.actual === undefined;
}

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

/**
 * Same neutral styling as "no data" — both are context rather than findings,
 * and neither may read as a failure — but the words say which one it is.
 */
const UNDECIDED_STYLE: StateStyle = {
  label: "can't tell",
  description:
    "The dataset has a value here, but it does not settle the request — this is neither met nor violated",
  container: STATE_STYLES.unknown.container,
  badge: STATE_STYLES.unknown.badge,
  icon: STATE_STYLES.unknown.icon,
};

function styleFor(constraint: ConstraintView): StateStyle {
  if (constraint.status === "unknown" && !isAbsentFromDataset(constraint)) {
    return UNDECIDED_STYLE;
  }
  return STATE_STYLES[constraint.status];
}

function ConstraintRow({ constraint }: { constraint: ConstraintView }) {
  const style = styleFor(constraint);

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
          &ldquo;Can&rsquo;t tell&rdquo; means it has one that does not settle the request — a
          category list without the category asked for, say. Both are reported separately from a
          violation, and neither is treated as one.
        </p>
      )}
    </div>
  );
}

/**
 * A one-line summary for the collapsed/header position: "2 met · 1 not met ·
 * 1 no data". Text, not a coloured bar, so the counts survive a screenshot in
 * greyscale and a screen reader.
 *
 * The two undecided kinds are counted separately for the same reason they are
 * worded separately below: this line is what a reader skims, and "4 no data" on
 * a result whose rows say "no match for Cafes, has Bistros, French" is the
 * summary contradicting the detail.
 */
export function ConstraintSummary({ constraints }: { constraints: ConstraintGroups }) {
  const absent = constraints.unknown.filter(isAbsentFromDataset).length;
  const undecided = constraints.unknown.length - absent;

  const parts: string[] = [];
  if (constraints.matched.length > 0) parts.push(`${constraints.matched.length} met`);
  if (constraints.unmatched.length > 0) parts.push(`${constraints.unmatched.length} not met`);
  if (absent > 0) parts.push(`${absent} no data`);
  if (undecided > 0) parts.push(`${undecided} can't tell`);

  if (parts.length === 0) return null;

  return (
    <span className="text-xs font-medium text-slate-500">
      {parts.join(" · ")}
    </span>
  );
}
