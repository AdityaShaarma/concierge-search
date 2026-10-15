import { SearchExperience } from "@/components/SearchExperience";

/**
 * The single page. A server component whose only job is the static chrome
 * (header, footer) so that the interactive part is one clearly-marked client
 * component — the boundary is visible in one file rather than inferred from
 * `"use client"` directives scattered through the tree.
 */
export default function Page() {
  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10 sm:px-6 sm:py-14">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
          Concierge
        </h1>
        <p className="mt-2 max-w-2xl text-base leading-relaxed text-slate-600">
          Ask for a place the way you would ask a hotel concierge. Each result shows which parts
          of your request it actually satisfies — and which the dataset has no data for.
        </p>
      </header>

      <SearchExperience />

      <footer className="mt-16 border-t border-slate-200 pt-6 text-xs leading-relaxed text-slate-500">
        <p>
          Four ranking stages: LLM query understanding, hybrid BM25 + dense retrieval fused with
          reciprocal rank fusion, a fine-tuned cross-encoder reranker, and a LambdaMART final
          ranker.
        </p>
        <p className="mt-2">Built on the Yelp Open Dataset, for educational use.</p>
      </footer>
    </main>
  );
}
