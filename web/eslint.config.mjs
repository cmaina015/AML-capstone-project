import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    // The site is a static export served by FastAPI. Client-side <Link> navigation requests
    // per-segment RSC files that the export names differently, so plain <a> links are used.
    rules: { "@next/next/no-html-link-for-pages": "off" },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Vendored onnxruntime-web build, copied by scripts/copy-ort.mjs.
    "public/ort/**",
  ]),
]);

export default eslintConfig;
