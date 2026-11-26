// Copy the onnxruntime-web runtime (WebAssembly build) into public/ort so the site is self-contained.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(web, "node_modules/onnxruntime-web/dist");
const dst = join(web, "public/ort");
mkdirSync(dst, { recursive: true });
for (const f of ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(join(src, f), join(dst, f));
}
console.log("copied onnxruntime-web to public/ort");
