import type { NextConfig } from "next";

/**
 * Deliberately almost empty.
 *
 * The API lives behind a route handler (`app/api/search/route.ts`) rather than
 * a Next `rewrites` proxy, because the handler can also translate a dead
 * backend into a friendly error payload. A rewrite would surface a raw
 * connection failure to the browser, which is exactly the stack-trace-on-a-
 * projector outcome the demo must avoid.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
