import type { Config } from "tailwindcss";

/**
 * Tailwind 3 config (the version pinned in package.json; Tailwind 4 moves
 * configuration into CSS and would make this file dead weight).
 *
 * The theme additions are limited to the three constraint states. Those
 * colours are the one piece of visual design in this app that carries meaning
 * rather than taste, so they are named once here instead of being spelled as
 * raw palette classes at each use site — a stray `text-red-600` on a "no data"
 * badge would misrepresent the system's three-valued logic.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Constraint satisfied: green, the only state that is a win.
        met: {
          bg: "#ecfdf5",
          border: "#6ee7b7",
          text: "#065f46",
        },
        // Constraint violated: red. Reserved strictly for a measured failure.
        unmet: {
          bg: "#fef2f2",
          border: "#fca5a5",
          text: "#991b1b",
        },
        // No data in the dataset: deliberately neutral slate, never red. A
        // missing NoiseLevel is not a loud café.
        nodata: {
          bg: "#f8fafc",
          border: "#cbd5e1",
          text: "#475569",
        },
      },
      fontFamily: {
        // System stack: no webfont fetch, so the demo renders identically with
        // no network (a conference-wifi failure mode).
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
