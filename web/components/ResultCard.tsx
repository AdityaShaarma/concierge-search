import { ConstraintList, ConstraintSummary } from "@/components/ConstraintList";
import { formatPriceBand, formatRatingLabel, humanizeStage } from "@/lib/format";
import type { ResultView } from "@/lib/types";

/**
 * One result. Rank and identity at the top, the constraint explanation in the
 * body, and the cascade's own scores folded away in a `<details>`.
 *
 * The scores are included because the owner has to be able to answer "which
 * stage put this at position one?" in front of an interviewer, and the answer
 * is in `stage_scores` on every response. They are collapsed because a viewer
 * who has not read the architecture should see the explanation first.
 */

/** How many category tags to show before truncating. */
const MAX_CATEGORIES = 4;

function Stars({ stars, reviewCount }: { stars: number; reviewCount: number }) {
  // NaN only happens on a malformed payload; showing a dash is better than
  // "NaN ★", and better than inventing a rating.
  if (!Number.isFinite(stars)) {
    return <span className="text-sm text-slate-500">rating unavailable</span>;
  }

  return (
    <span className="inline-flex items-baseline gap-1 text-sm text-slate-700">
      <span aria-hidden="true" className="text-amber-500">
        ★
      </span>
      <span className="font-medium">{stars.toFixed(1)}</span>
      <span className="text-slate-500">
        ({reviewCount.toLocaleString()})
      </span>
      {/* The glyph is decorative; this is the value a screen reader reads. */}
      <span className="sr-only-live">{formatRatingLabel(stars, reviewCount)}</span>
    </span>
  );
}

function StageScores({ scores, finalScore }: { scores: Record<string, number>; finalScore: number }) {
  const entries = Object.entries(scores);
  if (entries.length === 0 && !Number.isFinite(finalScore)) return null;

  return (
    <details className="mt-3 border-t border-slate-100 pt-2">
      <summary className="cursor-pointer text-xs font-medium text-slate-500 hover:text-slate-800">
        Stage scores
      </summary>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-slate-600">
        {entries.map(([stage, value]) => (
          <li key={stage}>
            <span className="text-slate-500">{humanizeStage(stage)}:</span>{" "}
            {Number.isFinite(value) ? value.toFixed(4) : "–"}
          </li>
        ))}
        {Number.isFinite(finalScore) && (
          <li>
            <span className="text-slate-500">ranked by:</span> {finalScore.toFixed(4)}
          </li>
        )}
      </ul>
    </details>
  );
}

export function ResultCard({ result }: { result: ResultView }) {
  const { business, constraints } = result;
  const priceBand = formatPriceBand(business.attributes.RestaurantsPriceRange2);
  const categories = business.categories.slice(0, MAX_CATEGORIES);
  const hiddenCategories = business.categories.length - categories.length;
  const location = [business.city, business.state].filter(Boolean).join(", ");

  return (
    <li className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <article>
        <div className="flex items-start gap-3">
          {/*
            The rank is `aria-hidden` and restated in the heading's screen
            reader text, so a screen reader says "1. Nook Coffee" once instead
            of reading a bare number out of context.
          */}
          <span
            aria-hidden="true"
            className="mt-0.5 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white"
          >
            {result.rank}
          </span>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <h3 className="text-base font-semibold text-slate-900">
                <span className="sr-only-live">{`Result ${result.rank}: `}</span>
                {business.name}
              </h3>
              <ConstraintSummary constraints={constraints} />
            </div>

            <p className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm text-slate-600">
              <Stars stars={business.stars} reviewCount={business.review_count} />
              {location && <span>{location}</span>}
              {priceBand && (
                <span>
                  {/*
                    Dollar signs are unreadable aloud, so the band is spelled
                    out — but only when it really is a run of dollar signs. An
                    unexpected value from the dataset is read as-is rather than
                    described as a band it may not be.
                  */}
                  {/^\$+$/.test(priceBand) ? (
                    <>
                      <span aria-hidden="true">{priceBand}</span>
                      <span className="sr-only-live">{`price band ${priceBand.length} of 4`}</span>
                    </>
                  ) : (
                    <span>{`price: ${priceBand}`}</span>
                  )}
                </span>
              )}
            </p>

            {categories.length > 0 && (
              <ul className="mt-2 flex flex-wrap gap-1">
                {categories.map((category) => (
                  <li
                    key={category}
                    className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-600"
                  >
                    {category}
                  </li>
                ))}
                {hiddenCategories > 0 && (
                  <li className="px-1 py-0.5 text-[11px] text-slate-500">
                    {`+${hiddenCategories} more`}
                  </li>
                )}
              </ul>
            )}
          </div>
        </div>

        <div className="mt-3 border-t border-slate-100 pt-3">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Why this result
          </h4>
          <ConstraintList constraints={constraints} />
        </div>

        <StageScores scores={result.stage_scores} finalScore={result.score} />
      </article>
    </li>
  );
}
