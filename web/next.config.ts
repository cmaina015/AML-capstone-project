import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// The site is fully static (web/out): model info is public/info.json and prediction runs in the browser,
// so neither production nor `next dev` needs a backend.
const shared: NextConfig = {
  images: { unoptimized: true },
  // Emit /pipeline/index.html etc. so any static host (Hugging Face static Space, FastAPI StaticFiles) serves every page.
  trailingSlash: true,
  poweredByHeader: false,
  turbopack: {
    rules: {
      "*.css": { loaders: ["@tailwindcss/turbopack"], as: "*.css" },
    },
  },
};

export default function config(phase: string): NextConfig {
  return phase === PHASE_DEVELOPMENT_SERVER ? shared : { ...shared, output: "export" };
}
