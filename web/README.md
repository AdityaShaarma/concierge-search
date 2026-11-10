# Concierge demo frontend

A single-page Next.js (App Router) demo for the Concierge search API. It exists to make three
things visible that a plain list of results would hide:

1. **How the query was understood.** The parsed query is shown back to the user, split into the
   semantic text (embedded and scored) and the hard filters (checked arithmetically). This is the
   only on-screen evidence that Stage 1 exists.
2. **Why each result is there.** Every result lists its constraint checks in four distinct
   states — **met**, **not met**, **no data** and **can't tell** — each with its own word, never
   colour alone. Three of them are reachable here: the served pipeline drops any candidate with
   a violated constraint (Stage 2's filter, step 3), so `unmatched` is always empty and **not
   met** is rendered for the contract's sake rather than ever being seen. A violation is a
   reason to be absent from the list, not a row inside it. The last two are styled identically (neutral, dashed border) because neither is
   a failure: the dataset's attribute coverage is sparse, and rendering a missing `NoiseLevel` as
   a failure would claim a quiet café is loud, which is exactly what the three-valued constraint
   logic exists to prevent.

   They are worded differently because the backend's single `UNKNOWN` covers two situations and
   calling both "no data" was false. `actual === null` is a genuine absence. A non-null `actual`
   means the dataset *has* a value that does not settle the request — most often a category list
   that does not contain the asked-for category, which `filters.check_categories` returns as
   UNKNOWN on purpose (a lexical miss is "we cannot tell this is the wrong kind of place", not
   "it is"). The page used to label that "no data" directly beside an explanation reading
   "no match for Cafes, has Bistros, French".
3. **What it cost.** Per-stage timings and candidate counts appear in a small footer under every
   search. Every number comes from the pipeline's own instrumentation and is passed through
   untouched; nothing on this page is timed in the browser.

## Run it

The API must be running first (`make serve` from the repo root, serving on
`http://localhost:8000`).

```bash
cd web
npm install          # or `npm ci`, which is what CI runs — see "Status" below
npm run dev          # http://localhost:3000
```

Other scripts: `npm run build`, `npm run start` (production), `npm run typecheck` (`tsc --noEmit`).

### Pointing at a different API

The backend address comes from `CONCIERGE_API_URL`, defaulting to `http://localhost:8000`, so the
demo works with no configuration at all. To override:

```bash
cp .env.local.example .env.local   # then edit CONCIERGE_API_URL
```

It is a server-side variable (no `NEXT_PUBLIC_` prefix) on purpose. The browser calls this app's
own `/api/search` and `/api/health` route handlers, which forward to the Python service. That
buys two things: the demo needs no CORS configuration, and a refused connection becomes a
sentence a viewer can read instead of a failed fetch in the devtools console.

**If searches fail with a connection error while uvicorn is clearly running**, set
`CONCIERGE_API_URL=http://127.0.0.1:8000`. Node resolves `localhost` to `::1` first on many Macs,
while uvicorn binds `127.0.0.1` by default.

## Layout

```
web/
├── app/
│   ├── page.tsx                 static chrome; renders the one client component
│   ├── layout.tsx               document shell and metadata
│   ├── globals.css              Tailwind entry + global focus/motion rules
│   └── api/
│       ├── search/route.ts      proxy for POST /search, with error translation
│       └── health/route.ts      proxy for GET /health, for the status banner
├── components/
│   ├── SearchExperience.tsx     the only stateful component
│   ├── SearchForm.tsx           input, submit, example-query chips
│   ├── ParsedQueryPanel.tsx     Stage 1 output, shown back to the user
│   ├── ResultCard.tsx           one result + collapsed stage scores
│   ├── ConstraintList.tsx       the three constraint states ("why this result")
│   ├── TimingsFooter.tsx        per-stage latency and candidate counts
│   └── StateViews.tsx           loading, idle, no-results, error, degraded banners
└── lib/
    ├── types.ts                 hand-written mirror of serve/schemas.py
    ├── api.ts                   browser-side client + response normalisation
    ├── format.ts                pure display formatting
    ├── examples.ts              the three example queries from CLAUDE.md §1
    └── server-env.ts            backend address and timeout (server-only)
```

## Deliberate choices

- **No data-fetching library.** One endpoint, one in-flight request. A client-side cache would
  also hide the backend's parse-cache hits and distort the latency story the footer is there to
  tell.
- **`lib/types.ts` is hand-written, not generated from the OpenAPI schema.** Codegen would mean
  the frontend could not be built or typechecked without the Python API running; this UI has to be
  reviewable with no backend, no dataset and no API key present. The contract is small and changes
  rarely. If `serve/schemas.py` changes, that file is the one place to follow.
- **Response normalisation fills missing containers only.** A malformed payload yields empty
  lists, never invented scores, names or constraint rows.
- **No fabricated sample data anywhere in the source.** The loading skeleton is grey blocks with
  no text, so a screenshot taken a moment too early cannot be a screenshot of fiction.
- **One light palette, not a system-following dark mode.** The three constraint colours are the
  only colours in this app that carry meaning; one known-contrast rendering is worth more than a
  second theme that has to be re-verified every time they change.
- **Accessibility is part of the feature, not a pass afterwards.** Real `<label>` and `<form>`
  (Enter submits), visible focus rings globally, a polite live region announcing search state,
  `role="alert"` on failures, star ratings and price bands spelled out for screen readers, and
  every constraint state carrying an icon and a word as well as a colour.

## Status

Stated in both directions, because an understated status claim is the same defect as an
inflated one: a reader who checks it and finds it false has no reason to trust the next one.

**Verified.**

- The install, `tsc --noEmit` and the production build all run clean. Locally
  (`.audit/verification-run.md` records a clean `npm run build`) and in CI, where
  `.github/workflows/ci.yml` runs `npm ci`, `npx tsc --noEmit` and `npm run build` in this
  directory on Node 20 on every push and pull request.
- The UI renders **real** responses from the API: the screenshots in the root README come from
  this app talking to the fixture-corpus demo server (`make demo`) through the server-side
  proxy. Its handling of live payload shapes is exercised, not assumed.

**Not verified.**

- It has never run against a dataset-backed index — no metro is chosen and no real index exists
  on disk — so nothing here says how the layout behaves with real names, long category lists,
  or a full page of ten results.
- There are no frontend tests. The pure functions in `lib/format.ts` are the obvious first
  candidates if that changes, and the build passing is not a substitute: `tsc` proves the
  shapes agree with `lib/types.ts`, not that `lib/types.ts` agrees with `serve/schemas.py`.

(This section previously said `npm install` had never been run and the UI had never rendered a
real response. `web/node_modules/`, `web/tsconfig.tsbuildinfo`, the CI job and the committed
verification record all said otherwise.)
