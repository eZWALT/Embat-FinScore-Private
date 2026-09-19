import type { NextConfig } from "next";

// Keep local and Vercel builds on the same default Next.js configuration.
const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/api/ask": ["./src/lib/agent/prompts/**/*"],
    "/api/watcher/reply": ["./src/lib/agent/prompts/**/*"],
  },
};

export default nextConfig;
