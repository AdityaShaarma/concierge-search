/**
 * The three example queries from CLAUDE.md section 1, verbatim.
 *
 * They are not decoration: each one exercises a different part of the system,
 * which is what makes them worth putting one click away in a demo. The
 * `demonstrates` line is shown as the chip's accessible description so a
 * viewer knows what they are about to test.
 */
export interface ExampleQuery {
  query: string;
  /** What this query is meant to show off. Shown to screen readers and on hover. */
  demonstrates: string;
}

export const EXAMPLE_QUERIES: readonly ExampleQuery[] = [
  {
    query: "quiet café with wifi that's open past 10pm",
    demonstrates:
      "Hard constraints: opening hours (including past midnight), noise level and WiFi attributes.",
  },
  {
    query: "cheap ramen near downtown with outdoor seating",
    demonstrates:
      "Price band, a location constraint and an amenity, combined with a specific cuisine.",
  },
  {
    query: "somewhere good for a first date that isn't loud",
    demonstrates:
      "A vibe query with a negated constraint — mostly semantic, with one hard filter.",
  },
] as const;
