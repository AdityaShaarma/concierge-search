import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Concierge — natural-language place search",
  description:
    "Demo frontend for Concierge: LLM query understanding, hybrid retrieval, a fine-tuned cross-encoder reranker and a LambdaMART ranker, with per-result constraint explanations.",
};

/**
 * `lang` is set explicitly so a screen reader picks the right voice, and
 * `min-h-full` keeps the empty state's vertical centring honest on a short
 * page.
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
