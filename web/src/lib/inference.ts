/**
 * In-browser prediction: the same steps as app/inference.py:Predictor.predict, run with onnxruntime-web.
 *
 *  1. decode          -> lib/decode.ts           (same pixels as PIL .convert("RGB"))
 *  2. resize 224x224  -> lib/pilResize.ts         (bit-identical to Pillow bilinear, as in training)
 *  3. ToTensor + ImageNet normalise, NCHW float32
 *  4. ONNX model      -> logits and the raw 7x7 Grad-CAM for both classes (scripts/export_onnx.py)
 *  5. softmax, threshold from model_config.json
 *  6. Grad-CAM post-processing exactly as pytorch-grad-cam 1.5.7 + show_cam_on_image + cv2 JET
 */
import type * as Ort from "onnxruntime-web";
import { decodeUpload, InputError } from "./decode";
import { JET_LUT } from "./jetLut";
import { pilResizeBilinear } from "./pilResize";
import type { Info, Prediction } from "./types";

export { InputError };

declare global {
  interface Window {
    ort?: typeof Ort;
  }
}

export type LoadProgress = { loaded: number; total: number; stage: "runtime" | "download" | "verify" | "init" };

const ORT_DIR = "/ort/";
const CACHE_NAME = "breast-histo-model";

let sessionPromise: Promise<Ort.InferenceSession> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (window.ort) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Could not load the inference runtime."));
    document.head.appendChild(s);
  });
}

async function hex(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchModel(info: Info, onProgress: (p: LoadProgress) => void): Promise<ArrayBuffer> {
  const { url, sha256, bytes } = info.browser_model;
  const key = `${url}?sha256=${sha256}`;
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(key);
    if (hit) {
      const buf = await hit.arrayBuffer();
      if (buf.byteLength === bytes) return buf;
    }
  } catch {
    cache = null; // Cache Storage unavailable (private mode, http): download every visit
  }

  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download the model (HTTP ${res.status}).`);
  const total = Number(res.headers.get("content-length")) || bytes;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress({ loaded, total, stage: "download" });
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }

  onProgress({ loaded, total, stage: "verify" });
  const actual = await hex(buf.buffer);
  if (actual !== sha256) throw new Error("The downloaded model failed its checksum; reload the page to retry.");
  if (cache) {
    try {
      // drop older model versions, keep this one
      for (const req of await cache.keys()) if (req.url.split("?")[0].endsWith(url) && !req.url.endsWith(key)) await cache.delete(req);
      await cache.put(key, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
    } catch { /* quota: fine, just not cached */ }
  }
  return buf.buffer;
}

export function loadModel(info: Info, onProgress: (p: LoadProgress) => void = () => {}): Promise<Ort.InferenceSession> {
  sessionPromise ??= (async () => {
    onProgress({ loaded: 0, total: info.browser_model.bytes, stage: "runtime" });
    await loadScript(`${ORT_DIR}ort.wasm.min.js`);
    const ort = window.ort!;
    ort.env.wasm.wasmPaths = ORT_DIR;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.wasm.proxy = true; // run in a worker so the page stays responsive
    const model = await fetchModel(info, onProgress);
    onProgress({ loaded: info.browser_model.bytes, total: info.browser_model.bytes, stage: "init" });
    return ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  })().catch((e) => {
    sessionPromise = null;
    throw e;
  });
  return sessionPromise;
}

/* ---------------- Grad-CAM post-processing (pytorch-grad-cam 1.5.7) ---------------- */

const f32 = Math.fround;

/** scale_cam_image without resize: x - min, / (1e-7 + max), in float32 like numpy. */
function minMax(a: Float32Array): Float32Array {
  let mn = Infinity;
  for (const v of a) if (v < mn) mn = v;
  const out = new Float32Array(a.length);
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) { out[i] = a[i] - mn; if (out[i] > mx) mx = out[i]; }
  const d = f32(1e-7 + mx);
  for (let i = 0; i < a.length; i++) out[i] = out[i] / d;
  return out;
}

/** cv2.resize(src, (dw, dh)) with INTER_LINEAR on float32 (half-pixel centres, edge clamped). */
function cvResizeLinear(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  const tab = (s: number, d: number) => {
    const idx = new Int32Array(d), fr = new Float32Array(d), scale = s / d;
    for (let i = 0; i < d; i++) {
      let fx = (i + 0.5) * scale - 0.5;
      let sx = Math.floor(fx);
      fx -= sx;
      if (sx < 0) { fx = 0; sx = 0; }
      if (sx >= s - 1) { fx = 0; sx = s - 1; }
      idx[i] = sx; fr[i] = fx;
    }
    return { idx, fr };
  };
  const X = tab(sw, dw), Y = tab(sh, dh);
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const y0 = Y.idx[y], y1 = Math.min(y0 + 1, sh - 1), fy = Y.fr[y];
    for (let x = 0; x < dw; x++) {
      const x0 = X.idx[x], x1 = Math.min(x0 + 1, sw - 1), fx = X.fr[x];
      const top = src[y0 * sw + x0] * (1 - fx) + src[y0 * sw + x1] * fx;
      const bot = src[y1 * sw + x0] * (1 - fx) + src[y1 * sw + x1] * fx;
      out[y * dw + x] = top * (1 - fy) + bot * fy;
    }
  }
  return out;
}

function toPngB64(rgb: Uint8Array, w: number, h: number): string {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d")!;
  const im = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    im.data[i * 4] = rgb[i * 3]; im.data[i * 4 + 1] = rgb[i * 3 + 1]; im.data[i * 4 + 2] = rgb[i * 3 + 2]; im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  return c.toDataURL("image/png").split(",")[1];
}

function renderCam(info: Info, input: Uint8Array, cam7: Float32Array, targetIndex: number, gh: number, gw: number, S: number) {
  // compute_cam_per_layer: relu -> scale(resize) ; aggregate_multi_layers: relu -> mean -> scale
  let cam = minMax(cam7.map((v) => Math.max(v, 0)));
  cam = cvResizeLinear(cam, gw, gh, S, S);
  cam = minMax(cam.map((v) => Math.max(v, 0)));
  for (let i = 0; i < cam.length; i++) cam[i] = Math.min(1, Math.max(0, cam[i]));

  // show_cam_on_image(rgb/255, cam, use_rgb=True, image_weight=w) and cv2 JET heatmap
  const n = S * S;
  const heat = new Uint8Array(n * 3);
  const blend = new Float32Array(n * 3);
  const wImg = f32(info.decision.overlay_image_weight);
  const wHeat = f32(1 - info.decision.overlay_image_weight);
  let mx = 0;
  for (let i = 0; i < n; i++) {
    const k = Math.trunc(f32(255 * cam[i])) * 3;
    for (let c = 0; c < 3; c++) {
      const hv = JET_LUT[k + c];
      heat[i * 3 + c] = hv;
      const v = f32(f32(wHeat * f32(hv / 255)) + f32(wImg * f32(input[i * 3 + c] / 255)));
      blend[i * 3 + c] = v;
      if (v > mx) mx = v;
    }
  }
  const overlay = new Uint8Array(n * 3);
  for (let i = 0; i < n * 3; i++) overlay[i] = Math.trunc(f32(255 * f32(blend[i] / mx)));

  // cv2.resize(cam, (7, 7), INTER_AREA): 224 / 7 = 32, so plain block means
  const bh = S / gh, bw = S / gw;
  const grid: number[][] = [];
  let camMax = 0;
  for (const v of cam) if (v > camMax) camMax = v;
  for (let r = 0; r < gh; r++) {
    const row: number[] = [];
    for (let c = 0; c < gw; c++) {
      let s = 0;
      for (let y = r * bh; y < (r + 1) * bh; y++) for (let x = c * bw; x < (c + 1) * bw; x++) s += cam[y * S + x];
      row.push(Math.round((s / (bh * bw)) * 1e4) / 1e4);
    }
    grid.push(row);
  }
  const classes = info.model.classes as Record<string, Prediction["label"]>;
  return {
    explanation: {
      method: "Grad-CAM",
      target_class: classes[String(targetIndex)],
      target_layer: info.explainability.target_layer,
      grid_hw: [gh, gw] as [number, number],
      grid,
      flat: camMax < 1e-6,
      note: info.explainability.note,
    },
    overlay: toPngB64(overlay, S, S),
    heatmap: toPngB64(heat, S, S),
  };
}

/* ---------------- predict ---------------- */

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export async function predictInBrowser(info: Info, file: File, explain: boolean): Promise<Prediction> {
  const session = await loadModel(info);
  const ort = window.ort!;
  const [S, S2] = info.decision.size_hw;
  const t0 = performance.now();
  const decoded = await decodeUpload(file, {
    maxBytes: info.limits.max_upload_mb * 1024 * 1024,
    maxPixels: info.decision.max_pixels,
    aspectTolerance: info.decision.aspect_tolerance,
    sizeHW: [S, S2],
  });
  const input = pilResizeBilinear(decoded.rgb, decoded.width, decoded.height, S2, S);

  const pp = info.model.preprocessing.details.normalize;
  const mean = pp?.mean ?? MEAN, std = pp?.std ?? STD;
  const n = S * S2;
  const x = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    const m = f32(mean[c]), s = f32(std[c]);
    for (let i = 0; i < n; i++) x[c * n + i] = f32(f32(f32(input[i * 3 + c]) / 255) - m) / s;
  }
  const t1 = performance.now();

  const out = await session.run({ input: new ort.Tensor("float32", x, [1, 3, S, S2]) });
  const logits = out.logits.data as Float32Array;
  const cams = out.cams.data as Float32Array;
  const t2 = performance.now();

  const classes = info.model.classes as Record<string, Prediction["label"]>;
  const positive = Number(Object.keys(classes).find((k) => classes[k] === info.model.positive_class));
  const mx = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - mx), e1 = Math.exp(logits[1] - mx);
  const probs = [e0 / (e0 + e1), e1 / (e0 + e1)];
  const p = probs[positive];
  const threshold = info.model.threshold;
  const predIndex = p >= threshold ? positive : 1 - positive;

  const notes = [...decoded.warnings];
  const margin = info.decision.near_threshold_margin;
  if (Math.abs(p - threshold) < margin)
    notes.push(`The probability is within ${margin} of the decision threshold, so this result is borderline.`);

  const result: Prediction = {
    label: classes[String(predIndex)],
    probabilities: { [classes["0"]]: probs[0], [classes["1"]]: probs[1] } as Prediction["probabilities"],
    p_malignant: p,
    threshold,
    decision_rule: info.model.decision_rule,
    input: decoded.info,
    warnings: notes,
    images: { model_input: toPngB64(input, S2, S) },
    explanation: null,
    timing_ms: { preprocess: 0, predict: 0, explain: null, total: 0 },
  };
  let t3 = t2;
  if (explain) {
    const [gh, gw] = info.explainability.cam_grid_hw as [number, number];
    const cam7 = cams.slice(predIndex * gh * gw, (predIndex + 1) * gh * gw);
    const r = renderCam(info, input, cam7, predIndex, gh, gw, S);
    result.explanation = r.explanation;
    result.images.overlay = r.overlay;
    result.images.heatmap = r.heatmap;
    t3 = performance.now();
  }
  const ms = (a: number, b: number) => Math.round((b - a) * 10) / 10;
  result.timing_ms = { preprocess: ms(t0, t1), predict: ms(t1, t2), explain: explain ? ms(t2, t3) : null, total: ms(t0, t3) };
  return result;
}
