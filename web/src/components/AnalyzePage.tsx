"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { predict } from "@/lib/api";
import { jet, pct, probPct } from "@/lib/format";
import { SERVER_MODE } from "@/lib/mode";
import type { Info, Prediction } from "@/lib/types";
import { useInfo } from "@/lib/useInfo";
import { Note, PageHeader } from "./common";

const ACCEPT = ".png,.jpg,.jpeg,.tif,.tiff,.bmp";

type ModelState =
  | { status: "idle" }
  | { status: "loading"; stage: "runtime" | "download" | "verify" | "init"; loaded: number; total: number }
  | { status: "ready" }
  | { status: "error"; message: string };

const mb = (b: number) => (b / 1024 / 1024).toFixed(0);

/** One line under the upload controls: model download progress, then where the model runs. */
function ModelStatus({ model, bytes }: { model: ModelState; bytes?: number }) {
  if (model.status === "error")
    return (
      <p className="mt-3 rounded-md border border-danger-line bg-danger px-3 py-2 text-sm text-danger-fg" role="alert">
        The model could not be loaded: {model.message}
      </p>
    );
  if (model.status === "ready" && SERVER_MODE)
    return (
      <p className="mt-3 flex flex-wrap items-center gap-x-2 text-xs text-fg-muted">
        <span className="h-2 w-2 rounded-full bg-brand" aria-hidden />
        Predictions run on the FastAPI server (PyTorch). Images are sent for analysis and are not stored.
        <a href="/docs" className="text-brand-fg hover:underline">API docs →</a>
      </p>
    );
  if (model.status === "ready")
    return (
      <p className="mt-3 flex items-center gap-2 text-xs text-fg-muted">
        <span className="h-2 w-2 rounded-full bg-brand" aria-hidden />
        Model ready. It runs in your browser: images are analyzed on this device and never uploaded.
      </p>
    );
  if (SERVER_MODE)
    return <p className="mt-3 text-xs text-fg-muted" role="status">Connecting to the server…</p>;
  const loading = model.status === "loading" ? model : null;
  const frac = loading && loading.total ? loading.loaded / loading.total : 0;
  const text = !loading || loading.stage === "runtime"
    ? `Preparing the model (${bytes ? mb(bytes) : "…"} MB download, first visit only)…`
    : loading.stage === "download"
      ? `Downloading the model: ${mb(loading.loaded)} of ${mb(loading.total)} MB (first visit only)`
      : loading.stage === "verify" ? "Checking the model file…" : "Starting the model…";
  return (
    <div className="mt-3" role="status" aria-live="polite">
      <p className="text-xs text-fg-muted">{text}</p>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-brand transition-[width]" style={{ width: `${Math.round(frac * 100)}%` }} />
      </div>
    </div>
  );
}
const BREAKHIS_URL = "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/";

/* ------------------------------------------------------------------ icons */

const Icon = {
  upload: <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />,
  image: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="10" r="2" />
      <path d="m21 16-5-5-9 9" />
    </>
  ),
  heat: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
    </>
  ),
  chip: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5m0-8h.01" />
    </>
  ),
  eye: (
    <>
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
};

function Svg({ children, className = "h-5 w-5" }: { children: ReactNode; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
      {children}
    </svg>
  );
}

function Panel({ title, icon, children, className = "" }: { title: string; icon?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-5 shadow-sm ${className}`}>
      <h2 className="mb-4 flex items-center gap-2 text-base font-semibold text-fg">
        {icon && (
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-light text-brand-fg">
            <Svg className="h-4 w-4">{icon}</Svg>
          </span>
        )}
        {title}
      </h2>
      {children}
    </section>
  );
}

/* ------------------------------------------------------------------ pieces */

function StatTiles({ info }: { info: Info | null }) {
  const h = info?.performance.headline.at_selected_threshold;
  const tiles = [
    { label: "ROC-AUC", value: h ? h.roc_auc.mean.toFixed(3) : "—", sub: h ? `± ${h.roc_auc.std.toFixed(3)} across 5 folds` : "" },
    { label: "Malignant recall", value: h ? pct(h.malignant_recall.mean) : "—", sub: "malignant images detected" },
    { label: "Benign recall", value: h ? pct(h.benign_recall.mean) : "—", sub: "benign images correctly cleared" },
    { label: "Decision threshold", value: info ? pct(info.model.threshold, 0) : "—", sub: "malignant probability needed to flag" },
  ];
  return (
    <div className="mb-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {tiles.map((t) => (
          <div key={t.label} className="rounded-xl border border-line bg-surface px-4 py-3 shadow-sm">
            <p className="text-xs font-medium text-fg-muted">{t.label}</p>
            <p className="mt-1 text-2xl font-semibold text-fg tabular">{t.value}</p>
            <p className="mt-0.5 text-xs text-fg-muted">{t.sub}</p>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-fg-muted">
        Patient-level 5-fold cross-validation: every score comes from patients the model did not see during training.
      </p>
    </div>
  );
}

function Gauge({ p, t, label }: { p: number | null; t: number; label?: "benign" | "malignant" }) {
  const cx = 110;
  const cy = 104;
  const r = 84;
  const pt = (v: number, rad = r) => [cx - rad * Math.cos(Math.PI * v), cy - rad * Math.sin(Math.PI * v)] as const;
  const arc = (a: number, b: number) => {
    const [x0, y0] = pt(a);
    const [x1, y1] = pt(b);
    return `M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`;
  };
  const [tx0, ty0] = pt(t, r - 13);
  const [tx1, ty1] = pt(t, r + 13);
  const color = label === "malignant" ? "var(--color-malignant)" : "var(--color-benign)";
  return (
    <svg
      viewBox="0 0 220 122"
      className="mx-auto w-full max-w-[260px]"
      role="img"
      aria-label={p == null ? "No prediction yet" : `Probability of malignant ${probPct(p)}, threshold ${pct(t, 0)}`}
    >
      <path d={arc(0, t)} stroke="var(--color-benign-light)" strokeWidth="14" fill="none" />
      <path d={arc(t, 1)} stroke="var(--color-malignant-light)" strokeWidth="14" fill="none" />
      {p != null && p > 0.002 && <path d={arc(0, Math.min(p, 0.999))} stroke={color} strokeWidth="14" fill="none" strokeLinecap="round" />}
      <line x1={tx0} y1={ty0} x2={tx1} y2={ty1} stroke="var(--color-fg)" strokeWidth="2.5" />
      <text x={cx} y={cy - 18} textAnchor="middle" fill="var(--color-fg)" fontSize="26" fontWeight="600">
        {p == null ? "—" : probPct(p)}
      </text>
      <text x={cx} y={cy} textAnchor="middle" fill="var(--color-fg-muted)" fontSize="10.5">
        probability of malignant
      </text>
      <text x={cx - r} y={cy + 15} textAnchor="middle" fill="var(--color-fg-faint)" fontSize="9.5">
        0%
      </text>
      <text x={cx + r} y={cy + 15} textAnchor="middle" fill="var(--color-fg-faint)" fontSize="9.5">
        100%
      </text>
    </svg>
  );
}

function GridOverlay({ grid, opacity }: { grid: number[][]; opacity: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const n = grid.length;
    const s = cv.width / n;
    grid.forEach((row, i) =>
      row.forEach((v, j) => {
        ctx.fillStyle = jet(v);
        ctx.fillRect(j * s, i * s, s, s);
      }),
    );
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    for (let k = 1; k < n; k++) {
      ctx.beginPath();
      ctx.moveTo(k * s + 0.5, 0);
      ctx.lineTo(k * s + 0.5, cv.height);
      ctx.moveTo(0, k * s + 0.5);
      ctx.lineTo(cv.width, k * s + 0.5);
      ctx.stroke();
    }
  }, [grid]);
  return <canvas ref={ref} width={224} height={224} className="absolute inset-0 h-full w-full" style={{ opacity }} />;
}

function Pane({
  caption,
  empty,
  busy,
  children,
  onMove,
  onLeave,
}: {
  caption: ReactNode;
  empty: { icon: ReactNode; text: string };
  busy: boolean;
  children?: ReactNode;
  onMove?: (e: React.MouseEvent<HTMLDivElement>) => void;
  onLeave?: () => void;
}) {
  const filled = Boolean(children);
  return (
    <figure className="min-w-0">
      <div
        className={`relative aspect-square overflow-hidden rounded-lg ${filled ? "bg-gray-900" : "border-2 border-dashed border-line bg-subtle"}`}
        onMouseMove={onMove}
        onMouseLeave={onLeave}
      >
        {filled ? (
          children
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-fg-faint">
            <Svg className="h-9 w-9">{empty.icon}</Svg>
            <span className="text-sm">{empty.text}</span>
          </div>
        )}
        {busy && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/70">
            <span className="h-8 w-8 animate-spin rounded-full border-[3px] border-line-strong border-t-brand" />
          </div>
        )}
      </div>
      <figcaption className="mt-2 text-xs text-fg-muted">{caption}</figcaption>
    </figure>
  );
}

function ResultPanel({ r, busy, threshold }: { r: Prediction | null; busy: boolean; threshold: number }) {
  if (!r) {
    return (
      <Panel title="Result" icon={Icon.chip} className="flex h-full flex-col">
        <Gauge p={null} t={threshold} />
        <p className="mt-1 text-center text-sm text-fg-muted">{busy ? "Analyzing image…" : "No image analyzed yet"}</p>
        <div className="mt-6 border-t border-line-soft pt-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wide text-fg-muted">How it works</p>
          <ol className="space-y-3 text-sm text-fg-2">
            {[
              ["Upload", "Choose, drag or paste an H&E image."],
              ["Predict", `Swin Tiny estimates the probability that the image is malignant. At ${pct(threshold, 0)} or above it is labelled malignant.`],
              ["Explain", "Grad-CAM marks the regions that drove the prediction."],
            ].map(([t, d], i) => (
              <li key={t} className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand text-xs font-semibold text-white">{i + 1}</span>
                <span>
                  <span className="font-medium text-fg">{t}.</span> {d}
                </span>
              </li>
            ))}
          </ol>
        </div>
      </Panel>
    );
  }

  const malignant = r.label === "malignant";
  const margin = r.p_malignant - r.threshold;
  return (
    <Panel title="Result" icon={Icon.chip} className="flex h-full flex-col">
      <div
        className={`flex items-center justify-between rounded-lg border-l-4 px-4 py-3 ${
          malignant ? "border-malignant bg-malignant-light" : "border-benign bg-benign-light"
        }`}
      >
        <div>
          <p className="text-xs text-fg-muted">Prediction</p>
          <p className={`text-2xl font-semibold ${malignant ? "text-malignant-fg" : "text-benign-fg"}`}>{malignant ? "Malignant" : "Benign"}</p>
        </div>
        {Math.abs(margin) < 0.05 && <span className="rounded-full bg-warn-strong px-2.5 py-0.5 text-xs font-medium text-warn-fg">Borderline</span>}
      </div>

      <div className="mt-4">
        <Gauge p={r.p_malignant} t={r.threshold} label={r.label} />
        <div className="mt-1 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-fg-muted">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-benign-light ring-1 ring-benign" /> benign
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-0.5 bg-fg" /> threshold {pct(r.threshold, 0)}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-malignant-light ring-1 ring-malignant" /> malignant
          </span>
        </div>
      </div>

      <p className="mt-4 rounded-lg bg-subtle px-3 py-2.5 text-sm leading-relaxed text-fg-2">
        The model gives this image a <strong className="text-fg">{probPct(r.p_malignant)}</strong> probability of being
        malignant. Images at <strong className="text-fg">{pct(r.threshold, 0)}</strong> or above are labelled
        malignant, so this one is labelled <strong className={malignant ? "text-malignant-fg" : "text-benign-fg"}>{r.label}</strong>
        {Math.abs(margin) < 0.05 ? ", but it is close to the threshold and should be treated as uncertain." : "."}
      </p>

      <dl className="mt-3 divide-y divide-line-soft text-sm">
        {[
          ["Probability of benign", probPct(r.probabilities.benign)],
          ["Distance from threshold", `${margin >= 0 ? "+" : "−"}${(Math.abs(margin) * 100).toFixed(1)} points`],
          ["Uploaded image", `${r.input.width} × ${r.input.height} ${r.input.format}`],
          [SERVER_MODE ? "Processing time (server)" : "Processing time (this device)",`${(r.timing_ms.total / 1000).toFixed(1)} s`],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 py-2">
            <dt className="text-fg-muted">{k}</dt>
            <dd className="tabular text-right text-fg">{v}</dd>
          </div>
        ))}
      </dl>

      {r.warnings.length > 0 && (
        <div className="mt-3 space-y-2">
          {r.warnings.map((w) => (
            <Note key={w} tone="yellow">
              {w}
            </Note>
          ))}
        </div>
      )}
      <p className="mt-auto pt-4 text-xs text-fg-faint">Not a diagnosis. For research and education only.</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ page */

export function AnalyzePage() {
  const { info } = useInfo();
  const [file, setFile] = useState<File | null>(null);
  const [explain, setExplain] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [opacity, setOpacity] = useState(45);
  const [showGrid, setShowGrid] = useState(false);
  const [cell, setCell] = useState<{ r: number; c: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [browserModel, setModel] = useState<ModelState>({ status: "idle" });
  // On the server there is nothing to download: ready as soon as the model information has loaded.
  const model: ModelState = SERVER_MODE ? { status: info ? "ready" : "idle" } : browserModel;
  const maxMb = info?.limits.max_upload_mb ?? 10;
  const threshold = info?.model.threshold ?? 0.5;

  // Start fetching the model as soon as the page has its config (cached by the browser after the first visit).
  useEffect(() => {
    if (!info || SERVER_MODE) return;
    let live = true;
    import("@/lib/inference")
      .then(({ loadModel }) =>
        loadModel(info, (p) => live && setModel({ status: "loading", ...p })),
      )
      .then(() => live && setModel({ status: "ready" }))
      .catch((e: Error) => live && setModel({ status: "error", message: e.message }));
    return () => {
      live = false;
    };
  }, [info]);

  async function analyze(f: File | null = file) {
    if (!f || !info) return;
    if (f.size > maxMb * 1024 * 1024) {
      setError(`File is larger than ${maxMb} MB.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await predict(info, f, explain);
      // exposed for scripts/check_browser_parity.py, which compares it with the Python pipeline
      (window as unknown as { __lastPrediction?: Prediction }).__lastPrediction = r;
      setResult(r);
      setCell(null);
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  function choose(f: File | undefined | null) {
    if (!f) return;
    setFile(f);
    setResult(null);
    analyze(f);
  }

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) choose(new File([f], f.name || "pasted.png", { type: f.type }));
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  const exp = result?.explanation;
  const n = exp?.grid_hw[0] ?? 7;
  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!exp) return;
    const b = e.currentTarget.getBoundingClientRect();
    setCell({
      c: Math.min(n - 1, Math.floor(((e.clientX - b.left) / b.width) * n)),
      r: Math.min(n - 1, Math.floor(((e.clientY - b.top) / b.height) * n)),
    });
  };
  const cellBox = cell && exp && (
    <div
      className="pointer-events-none absolute border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.4)]"
      style={{ left: `${(cell.c / n) * 100}%`, top: `${(cell.r / n) * 100}%`, width: `${100 / n}%`, height: `${100 / n}%` }}
    />
  );
  const inputSrc = result ? `data:image/png;base64,${result.images.model_input}` : null;

  return (
    <>
      <PageHeader title="Analyze an image">
        Upload an H&amp;E-stained breast tissue image at 100X magnification. The model predicts whether it is benign or
        malignant and can show a Grad-CAM heatmap of the regions that influenced the prediction.
      </PageHeader>

      <StatTiles info={info} />

      <div className="grid items-stretch gap-6 lg:grid-cols-3">
        <section className="rounded-xl border border-line bg-surface p-5 shadow-sm lg:col-span-2">
          <label
            htmlFor="file"
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              choose(e.dataTransfer.files?.[0]);
            }}
            className={`flex cursor-pointer items-center gap-4 rounded-lg border-2 border-dashed px-4 py-4 transition-colors ${
              dragOver ? "border-brand bg-brand-light" : "border-line bg-subtle hover:border-line-strong hover:bg-muted"
            }`}
          >
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-surface text-brand-fg shadow-sm ring-1 ring-line">
              <Svg>{Icon.upload}</Svg>
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium text-fg">{file ? file.name : "Drop an image here, or click to browse"}</span>
              <span className="block text-xs text-fg-muted">PNG, JPEG, TIFF or BMP · up to {maxMb} MB · paste with Ctrl+V</span>
            </span>
            <span className="hidden shrink-0 rounded-md border border-line-strong bg-surface px-3 py-1.5 text-sm font-medium text-fg-2 sm:block">Browse</span>
            <input id="file" type="file" accept={ACCEPT} className="sr-only" onChange={(e) => choose(e.target.files?.[0])} />
          </label>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <label className="flex items-center gap-2 text-sm text-fg-2">
              <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} className="h-4 w-4 accent-brand" />
              Generate Grad-CAM heatmap
            </label>
            <button
              type="button"
              onClick={() => analyze()}
              disabled={!file || busy || !info}
              className="rounded-md bg-brand px-5 py-2 text-sm font-medium text-white hover:bg-brand-dark disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? (model.status === "ready" ? "Analyzing…" : "Loading model…") : "Analyze"}
            </button>
          </div>
          <ModelStatus model={model} bytes={info?.browser_model.bytes} />
          {error && (
            <p className="mt-3 rounded-md border border-danger-line bg-danger px-3 py-2 text-sm text-danger-fg" role="alert">
              {error}
            </p>
          )}

          <div className="mt-5 grid grid-cols-2 gap-3 sm:gap-4">
            <Pane caption="Input image (resized to 224 × 224)" empty={{ icon: Icon.image, text: "Input image" }} busy={busy} onMove={onMove} onLeave={() => setCell(null)}>
              {inputSrc && (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={inputSrc} alt="Input image resized to 224 by 224" className="h-full w-full" />
                  {cellBox}
                </>
              )}
            </Pane>
            <Pane
              caption={exp ? `Grad-CAM for the predicted class (${exp.target_class})` : "Grad-CAM heatmap"}
              empty={{ icon: Icon.heat, text: result ? "Heatmap not generated" : "Grad-CAM heatmap" }}
              busy={busy && explain}
              onMove={onMove}
              onLeave={() => setCell(null)}
            >
              {exp && inputSrc && result?.images.heatmap && (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={inputSrc} alt="" className="absolute inset-0 h-full w-full" />
                  {showGrid ? (
                    <GridOverlay grid={exp.grid} opacity={opacity / 100} />
                  ) : (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={`data:image/png;base64,${result.images.heatmap}`} alt="Grad-CAM heatmap" className="absolute inset-0 h-full w-full" style={{ opacity: opacity / 100 }} />
                  )}
                  {cellBox}
                </>
              )}
            </Pane>
          </div>

          <div className={`mt-4 rounded-lg bg-subtle px-4 py-3 ${exp ? "" : "opacity-50"}`}>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-fg-2">
              <label className="flex items-center gap-2">
                Opacity
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={opacity}
                  disabled={!exp}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="w-28 accent-brand"
                />
                <span className="w-9 tabular text-fg-muted">{opacity}%</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showGrid} disabled={!exp} onChange={(e) => setShowGrid(e.target.checked)} className="h-4 w-4 accent-brand" />
                Show raw 7 × 7 grid
              </label>
              <span className="flex items-center gap-2 text-xs text-fg-muted">
                low
                <span className="h-2 w-24 rounded" style={{ background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map(jet).join(",")})` }} />
                high
              </span>
              <span className="text-xs text-fg-muted tabular sm:ml-auto">
                {exp && cell ? `Cell (${cell.r + 1}, ${cell.c + 1}): ${exp.grid[cell.r][cell.c].toFixed(2)}` : "Hover an image to read a cell"}
              </span>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-fg-muted">
              {exp?.flat
                ? "For this image no region had positive evidence for the predicted class, so the heatmap is empty."
                : "The heatmap shows which regions pushed the model towards its prediction. It is computed on a 7 × 7 grid, so it marks areas of roughly 32 × 32 pixels rather than individual cells."}
            </p>
          </div>
        </section>

        <ResultPanel r={result} busy={busy} threshold={threshold} />
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-3">
        <Panel title="About the model" icon={Icon.chip}>
          <dl className="divide-y divide-line-soft text-sm">
            {[
              ["Architecture", "Swin Tiny"],
              ["Pretraining", "ImageNet"],
              ["Dataset", info ? `BreakHis 100X, ${info.dataset.n_patients} patients` : "…"],
              ["Images", info ? info.dataset.n_images.toLocaleString() : "…"],
              ["Input size", "224 × 224 RGB"],
              ["Evaluation", "5-fold CV by patient"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 py-1.5">
                <dt className="text-fg-muted">{k}</dt>
                <dd className="text-right text-fg">{v}</dd>
              </div>
            ))}
          </dl>
          <a href="/pipeline/index.html" className="mt-3 inline-block text-sm text-brand-fg hover:underline">
            How it was built and tested →
          </a>
        </Panel>

        <Panel title="About the input image" icon={Icon.info}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-fg-2">
            <li>H&amp;E-stained breast tissue at 100X magnification.</li>
            <li>Best as a 224 × 224 patch. Other sizes are resized without keeping the aspect ratio.</li>
            <li>Grayscale images, other stains or magnifications are outside the training data.</li>
            <li>
              {SERVER_MODE
                ? "Images are sent to the server for analysis and are not stored."
                : "Images are analyzed in your browser and never uploaded."}
            </li>
          </ul>
          <a href={BREAKHIS_URL} target="_blank" rel="noopener" className="mt-3 inline-block text-sm text-brand-fg hover:underline">
            Get example images from BreakHis →
          </a>
        </Panel>

        <Panel title="Reading the heatmap" icon={Icon.eye}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-fg-2">
            <li>Red areas pushed the model most towards its prediction; blue areas had little effect.</li>
            <li>The map is computed on a 7 × 7 grid (32 × 32 pixels per cell), so it shows regions, not single cells.</li>
            <li>A highlighted area is where the model looked, not proof that the area is diagnostic.</li>
          </ul>
          <a href="/pipeline/index.html#gradcam" className="mt-3 inline-block text-sm text-brand-fg hover:underline">
            More about Grad-CAM →
          </a>
        </Panel>
      </div>
    </>
  );
}
