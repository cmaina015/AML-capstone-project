"use client";
import { useCallback, useEffect, useRef, useSyncExternalStore, type ReactNode } from "react";
import type { Info } from "@/lib/types";
import { useInfo } from "@/lib/useInfo";

/* Six-slide presentation in a Beamer style (frame-title band, blocks, footer bar), following the
   proposal format of the DSA 8401 brief: problem statement, data source, planned approach, serving.
   The frame stays light in both themes, like a projected deck. Sizes are in em; the base font size
   scales with the frame width (globals.css). */

const AUTHOR = "Catherine Maina 110872";
const SHORT_TITLE = "Breast Histopathology Classification";
const DATE = "November 2026";
const API_URL = "https://cmaina015-breast-histopathology-api.hf.space/";
const DOCS_URL = "https://cmaina015-breast-histopathology-api.hf.space/docs";
const STATIC_URL = "https://cmaina015-breast-histopathology.static.hf.space";

const NAVY = "#0f2b5b";
const BLUE = "#1f4fd6";
const ROSE = "#c2410c"; // one warm accent, used sparingly for the key finding

/* ------------------------------------------------------------------ building blocks */

function Balls({ items }: { items: ReactNode[] }) {
  return (
    <ul className="bm-list space-y-[0.3em]">
      {items.map((it, i) => (
        <li key={i}>{it}</li>
      ))}
    </ul>
  );
}

function Block({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <div className="overflow-hidden rounded-[0.35em] shadow-[0_0.15em_0.4em_rgba(15,43,91,0.18)]">
      <div className="flex items-center gap-[0.45em] bg-[#0f2b5b] px-[0.7em] py-[0.3em] text-[1.02em] text-white">
        {icon}
        <span>{title}</span>
      </div>
      <div className="h-full bg-[#edf3ff] px-[0.8em] py-[0.55em] text-[0.95em] text-[#14213d]">{children}</div>
    </div>
  );
}

function Stat({ value, label, accent = false }: { value: string; label: ReactNode; accent?: boolean }) {
  return (
    <div className="border-l-[0.22em] pl-[0.6em]" style={{ borderColor: accent ? ROSE : BLUE }}>
      <p className="text-[2.2em] font-bold leading-none" style={{ color: accent ? ROSE : NAVY }}>{value}</p>
      <p className="mt-[0.25em] text-[0.88em] leading-snug text-[#1e293b]">{label}</p>
    </div>
  );
}

function Node({ title, accent = false, children }: { title: string; accent?: boolean; children: ReactNode }) {
  return (
    <div className={`flex-1 rounded-[0.4em] border-[0.12em] bg-white px-[0.6em] py-[0.5em] text-center ${accent ? "border-[#b91c1c]" : "border-[#0f2b5b]"}`}>
      <p className={`text-[1.02em] font-bold leading-tight ${accent ? "text-[#b91c1c]" : "text-[#0f2b5b]"}`}>{title}</p>
      <p className="mt-[0.3em] text-[0.8em] leading-snug text-[#14213d]">{children}</p>
    </div>
  );
}

const Arrow = () => (
  <span className="self-center px-[0.15em] text-[1.4em] leading-none text-[#1f4fd6] max-md:rotate-90" aria-hidden>→</span>
);

/** Small line icons (stroke = currentColor), sized to the text. */
function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-[1em] w-[1em] shrink-0" fill="none" stroke="currentColor" strokeWidth={2}
         strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}
const I = {
  users: "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75",
  scale: "M12 3v18M5 7h14M5 7l-3 7a4 4 0 0 0 6 0zM19 7l-3 7a4 4 0 0 0 6 0z",
  crop: "M6 2v14a2 2 0 0 0 2 2h14M18 22V8a2 2 0 0 0-2-2H2",
  input: "M4 4h16v16H4zM4 15l5-5 4 4 3-3 4 4",
  output: "M9 11l3 3 8-8M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9",
  target: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12zM12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
};

const n = (v: number) => v.toLocaleString("en-US");

/* ------------------------------------------------------------------ the slides */

type Slide =
  | { id: string; kind: "title" }
  | { id: string; kind: "frame"; title: string; body: ReactNode };

function slides(info: Info | null): Slide[] {
  const d = info?.dataset;
  const leak = info?.performance.leakage_prone_baseline;
  const perPatient = d ? Math.round(d.n_images / d.n_patients) : 26;
  const malPct = d ? Math.round(d.classes.malignant.image_fraction * 100) : 69;
  const leaked = leak ? `${leak.leakage.test_patients_also_in_train} of ${leak.leakage.test_patients}` : "78 of 78";

  return [
    { id: "title", kind: "title" },
    {
      id: "problem",
      kind: "frame",
      title: "Problem Statement",
      body: (
        <div className="grid gap-[1em] md:grid-cols-[1.55fr_1fr]">
          <div className="flex flex-col gap-[0.8em]">
            <p className="text-[1.02em] leading-relaxed text-[#14213d]">
              Every breast biopsy ends up as a glass slide that a pathologist reads by eye. It is careful, slow work,
              and a second opinion is not always at hand. I wanted to see whether a model could look at one of these
              tissue images and say <b>benign</b> or <b>malignant</b>, and still be right for patients it has
              <b> never seen</b>, not just the ones it learned from.
            </p>
            <div className="rounded-[0.35em] border-l-[0.3em] border-[#1f4fd6] bg-[#edf3ff] px-[0.8em] py-[0.5em] text-[0.95em] text-[#14213d]">
              <b>The question:</b> can a fine-tuned vision transformer tell benign from malignant BreakHis 100× images
              for patients it has not seen, and show where in the image it looked?
            </div>
            <div className="grid grid-cols-1 gap-[0.6em] text-[0.9em] text-[#14213d] sm:grid-cols-3">
              <div><p className="flex items-center gap-[0.35em] font-bold text-[#0f2b5b]"><Icon d={I.input} />Input</p><p>One H&amp;E tissue image at 100×.</p></div>
              <div><p className="flex items-center gap-[0.35em] font-bold text-[#0f2b5b]"><Icon d={I.output} />Output</p><p>Benign or malignant, a probability, and a heatmap.</p></div>
              <div><p className="flex items-center gap-[0.35em] font-bold text-[#0f2b5b]"><Icon d={I.target} />Success</p><p>Catches malignant cases (target 95%) in unseen patients.</p></div>
            </div>
          </div>
          <div className="flex flex-col justify-center gap-[1em] rounded-[0.45em] bg-[#f6f8fd] p-[0.9em]">
            <p className="text-[0.85em] font-bold uppercase tracking-wide text-[#1d3f9e]">Why it matters</p>
            <Stat accent value="26.5%" label="of new cancer cases in Kenyan women are breast cancer (GLOBOCAN 2024)" />
            <Stat value="694,000" label="breast cancer deaths worldwide in 2024 (WHO, 2026)" />
            <p className="text-[0.85em] italic text-[#334155]">A research tool for triage, not a diagnosis.</p>
          </div>
        </div>
      ),
    },
    {
      id: "data",
      kind: "frame",
      title: "Data Source and What Makes It Messy",
      body: (
        <div className="flex flex-col gap-[0.9em]">
          <p className="text-[0.95em] text-[#14213d]">
            <b>BreakHis</b> (Spanhol et al., 2016): 7,909 H&amp;E images from 82 patients, collected in Brazil at four
            magnifications. I work with the 100× set, 700 × 460 px images resized to 224 × 224.
          </p>
          <div className="grid grid-cols-2 gap-[0.9em] md:grid-cols-4">
            <Stat value={d ? n(d.n_images) : "2,081"} label="images at 100×" />
            <Stat value={String(d ? d.n_patients : 81)} label={`patients (${d ? d.classes.benign.patients : 24} benign, ${d ? d.classes.malignant.patients : 57} malignant)`} />
            <Stat value={`${malPct}%`} label="of images are malignant" />
            <Stat value={`~${perPatient}`} label="near-identical images per patient" />
          </div>
          <div className="grid gap-[0.8em] md:grid-cols-3">
            <Block title="Patient leakage" icon={<Icon d={I.users} />}>
              A random 70/15/15 split put <b style={{ color: ROSE }}>{leaked}</b> test patients in training too. I split by patient instead.
            </Block>
            <Block title="Class imbalance" icon={<Icon d={I.scale} />}>
              Only {d ? d.classes.benign.patients : 24} benign patients. Folds are stratified and recall is reported per class.
            </Block>
            <Block title="Narrow scope" icon={<Icon d={I.crop} />}>
              One lab, one stain, images squashed to 224 × 224. No claims beyond BreakHis 100×.
            </Block>
          </div>
        </div>
      ),
    },
    {
      id: "approach",
      kind: "frame",
      title: "Planned Approach (1/3): ML Techniques",
      body: (
        <div className="flex flex-col gap-[0.9em]">
          <div className="grid gap-[0.8em] md:grid-cols-3">
            <Block title="Classical ML (4 Models)">
              <Balls items={["Logistic Regression", "Support Vector Machine", "Random Forest", "XGBoost"]} />
            </Block>
            <Block title="Pretrained CNNs (3 Models)">
              <Balls items={["MobileNetV2", "ResNet50", "EfficientNetB0"]} />
            </Block>
            <Block title="Transformers (3 Models)">
              <Balls items={["Vision Transformer Base (ViT Base)", "Shifted Window Transformer Tiny (Swin Tiny)", "Data-efficient Image Transformer Small (DeiT Small)"]} />
            </Block>
          </div>
          <div className="flex flex-col items-stretch gap-[0.5em] md:flex-row">
            <Node title="Train and compare">All 10 models on the same patient-level splits</Node>
            <Arrow />
            <Node title="Evaluate">5-fold CV by patient; malignant recall ≥ 95%</Node>
            <Arrow />
            <Node title="Explain">Grad-CAM heatmap on Swin Tiny</Node>
            <Arrow />
            <Node title="Serve" accent>Swin Tiny goes to the backend →</Node>
          </div>
        </div>
      ),
    },
    {
      id: "backend",
      kind: "frame",
      title: "Planned Approach (2/3): Backend",
      body: (
        <div className="flex flex-col gap-[1em]">
          <div className="flex flex-col items-stretch gap-[0.6em] md:flex-row">
            <div className="flex md:w-[20%]">
              <Node title="Frontend">Sends the image to POST /api/predict</Node>
            </div>
            <Arrow />
            <div className="flex-1 rounded-[0.5em] border-[0.12em] border-dashed border-[#1f4fd6] bg-[#f3f7ff] p-[0.6em]">
              <p className="mb-[0.5em] text-[0.88em] font-bold text-[#0f2b5b]">Backend: one Docker container on Hugging Face Spaces</p>
              <div className="flex flex-col items-stretch gap-[0.4em] md:flex-row">
                <Node title="1. FastAPI">Checks the upload, resizes to 224 × 224</Node>
                <Arrow />
                <Node title="2. Swin Tiny (CPU)" accent>PyTorch inference with checked weights</Node>
                <Arrow />
                <Node title="3. Grad-CAM">Returns label, probability and heatmap as JSON</Node>
              </div>
            </div>
          </div>
          <div className="grid gap-[0.8em] text-[0.92em] text-[#14213d] md:grid-cols-3">
            <p><b className="text-[#0f2b5b]">Private by design:</b> images are processed in memory and never saved.</p>
            <p><b className="text-[#0f2b5b]">Same everywhere:</b> <code className="text-[0.92em]">docker compose up</code> runs the same image locally.</p>
            <p><b className="text-[#0f2b5b]">Fast enough:</b> about 0.4 s per image on a CPU, heatmap included.</p>
          </div>
        </div>
      ),
    },
    {
      id: "frontend",
      kind: "frame",
      title: "Planned Approach (3/3): Frontend and Live App",
      body: (
        <div className="grid items-center gap-[1.2em] md:grid-cols-[1fr_auto]">
          <div className="flex flex-col gap-[0.8em]">
            <Block title="Frontend: Next.js web app">
              <Balls items={[
                <><b>Analyze:</b> upload a tissue image, get benign or malignant with a probability.</>,
                <><b>Explain:</b> the Grad-CAM heatmap is shown over the image.</>,
                <><b>Model and pipeline pages:</b> results, data and how it was built.</>,
                <><b>Two modes:</b> calls the FastAPI backend, or runs the model in the browser (ONNX).</>,
              ]} />
            </Block>
            <div>
              <a href={API_URL} className="break-all font-mono text-[1em] text-[#1f4fd6] underline">{API_URL}</a>
              <p className="mt-[0.3em] text-[0.82em] text-[#334155]">
                API docs: <a href={DOCS_URL} className="break-all font-mono text-[#1f4fd6] underline">{DOCS_URL}</a>
              </p>
              <p className="mt-[0.2em] text-[0.82em] text-[#334155]">
                In-browser version: <a href={STATIC_URL} className="break-all font-mono text-[#1f4fd6] underline">{STATIC_URL}</a>
              </p>
            </div>
            <p className="text-[1.25em] text-[#0f2b5b]">Thank you. Questions?</p>
          </div>
          <div className="flex flex-col items-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/qr-live.svg" alt="QR code linking to the live app" className="h-[8em] w-[8em] rounded-[0.3em] border border-[#c9d6f2] bg-white p-[0.3em]" />
            <p className="mt-[0.3em] text-[0.82em] text-[#334155]">Scan to open the app</p>
          </div>
        </div>
      ),
    },
  ];
}

/* ------------------------------------------------------------------ frame rendering */

/** Deterministic 7 x 7 "heatmap" motif for the title slide (echoes the Grad-CAM grid). */
const MOTIF = [
  [1, 1, 2, 2, 1, 1, 1], [1, 2, 3, 3, 2, 1, 1], [2, 3, 4, 5, 3, 2, 1], [2, 3, 5, 5, 4, 2, 1],
  [1, 2, 3, 4, 3, 2, 1], [1, 1, 2, 2, 2, 1, 0], [0, 1, 1, 1, 1, 0, 0],
];
const SHADES = ["#16346b", "#1d4185", "#2a56a8", "#3f74d6", "#7aa2f0", "#c9dafc"];

function TitleSlide() {
  return (
    <div className="relative flex h-full flex-col justify-center overflow-hidden bg-[linear-gradient(135deg,#0b1f45_0%,#0f2b5b_55%,#173c7a_100%)] px-[6%] py-[1.5em] text-white">
      <div className="pointer-events-none absolute right-[5%] top-1/2 hidden -translate-y-1/2 grid-cols-7 gap-[0.35em] opacity-90 md:grid" aria-hidden>
        {MOTIF.flat().map((v, k) => (
          <span key={k} className="h-[1.7em] w-[1.7em] rounded-[0.3em]" style={{ background: SHADES[v] }} />
        ))}
      </div>
      <div className="relative md:max-w-[58%]">
        <p className="text-[0.85em] uppercase tracking-[0.2em] text-[#9db8ff]">DSA 8401 · Capstone Project</p>
        <h2 className="mt-[0.4em] text-[2.3em] font-semibold leading-tight">Benign and Malignant Breast Histopathology Image Classification</h2>
        <p className="mt-[0.5em] text-[1.1em] text-[#c9dafc]">Comparative Study of Handcrafted Features and Fine-Tuned Vision Transformers</p>
        <div className="my-[1.2em] h-[0.15em] w-[5em] bg-[#7aa2f0]" />
        <p className="text-[1.2em]">{AUTHOR}</p>
        <p className="mt-[0.2em] text-[0.9em] text-[#c9dafc]">Institute of Mathematical Sciences, Strathmore University, Nairobi, Kenya</p>
        <p className="mt-[0.2em] text-[0.9em] text-[#c9dafc]">{DATE}</p>
      </div>
    </div>
  );
}

function Frame({ title, page, total, children }: { title: string; page: number; total: number; children: ReactNode }) {
  return (
    <div className="flex h-full flex-col">
      <div className="border-b-[0.2em] border-[#1f4fd6] bg-[#edf3ff] px-[4%] py-[0.5em]">
        <h2 className="text-[1.6em] leading-tight text-[#0f2b5b]">{title}</h2>
      </div>
      {/* Beamer centres a frame's content vertically */}
      <div className="flex min-h-0 flex-1 flex-col justify-center px-[3.5%] py-[0.8em]">{children}</div>
      <div className="grid grid-cols-[1fr_auto] text-[0.62em] text-white md:grid-cols-3">
        <span className="truncate bg-[#0f2b5b] px-[1.2em] py-[0.35em]">{AUTHOR} (Strathmore University)</span>
        <span className="hidden truncate bg-[#1f4fd6] px-[1.2em] py-[0.35em] text-center md:block">{SHORT_TITLE}</span>
        <span className="flex justify-between gap-[1em] bg-[#0f2b5b] px-[1.2em] py-[0.35em]">
          <span className="hidden md:inline">{DATE}</span>
          <span className="tabular">{page} / {total}</span>
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ the viewer */

function subscribeHash(onChange: () => void) {
  window.addEventListener("hashchange", onChange);
  return () => window.removeEventListener("hashchange", onChange);
}

export function SlidesPage() {
  const { info } = useInfo();
  const deck = slides(info);
  const total = deck.length;
  // The current slide lives in the address (#N), so any slide can be linked to directly.
  const hash = useSyncExternalStore(subscribeHash, () => window.location.hash, () => "");
  const i = Math.max(0, Math.min(total - 1, (parseInt(hash.slice(1), 10) || 1) - 1));
  const shell = useRef<HTMLDivElement>(null);
  const touchX = useRef<number | null>(null);

  const go = useCallback(
    (k: number) => window.location.replace(`#${Math.max(0, Math.min(total - 1, k)) + 1}`),
    [total],
  );
  const fullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen();
    else shell.current?.requestFullscreen?.();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select") || e.altKey || e.ctrlKey || e.metaKey) return;
      if (["ArrowRight", "PageDown", " "].includes(e.key)) go(i + 1);
      else if (["ArrowLeft", "PageUp"].includes(e.key)) go(i - 1);
      else if (e.key === "Home") go(0);
      else if (e.key === "End") go(total - 1);
      else if (e.key === "f" || e.key === "F") fullscreen();
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [i, total, go, fullscreen]);

  const s = deck[i];
  const label = s.kind === "title" ? "Title" : s.title;
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-2xl font-semibold text-fg">Presentation</h1>
        <p className="text-xs text-fg-muted">Arrow keys or swipe to move · F for full screen</p>
      </div>

      <div ref={shell} className="slide-shell">
        <div
          className="slide-stage relative overflow-hidden rounded-lg border border-line bg-white text-[#14213d] shadow-sm"
          onTouchStart={(e) => (touchX.current = e.touches[0].clientX)}
          onTouchEnd={(e) => {
            if (touchX.current === null) return;
            const dx = e.changedTouches[0].clientX - touchX.current;
            if (Math.abs(dx) > 50) go(dx < 0 ? i + 1 : i - 1);
            touchX.current = null;
          }}
        >
          <article key={s.id} className="slide-in h-full" aria-roledescription="slide" aria-label={`${i + 1} of ${total}: ${label}`}>
            {s.kind === "title" ? <TitleSlide /> : <Frame title={s.title} page={i + 1} total={total}>{s.body}</Frame>}
          </article>
        </div>

        <div className="slide-controls mt-3 flex items-center gap-3">
          <button type="button" onClick={() => go(i - 1)} disabled={i === 0} aria-label="Previous slide"
                  className="rounded-md border border-line-strong bg-surface px-3 py-1.5 text-sm text-fg-2 hover:bg-subtle disabled:opacity-40">
            ← Previous
          </button>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted" aria-hidden>
            <div className="h-full rounded-full bg-brand transition-[width]" style={{ width: `${((i + 1) / total) * 100}%` }} />
          </div>
          <span className="w-14 text-center text-sm tabular text-fg-muted">{i + 1} / {total}</span>
          <button type="button" onClick={() => go(i + 1)} disabled={i === total - 1} aria-label="Next slide"
                  className="rounded-md bg-brand px-3 py-1.5 text-sm font-medium text-white hover:bg-brand-dark disabled:opacity-40">
            Next →
          </button>
          <button type="button" onClick={fullscreen} aria-label="Full screen" title="Full screen (F)"
                  className="hidden rounded-md border border-line-strong bg-surface px-3 py-1.5 text-sm text-fg-2 hover:bg-subtle sm:block">
            ⛶
          </button>
        </div>
      </div>

      <nav className="mt-4 flex flex-wrap gap-1.5" aria-label="All slides">
        {deck.map((d, k) => {
          const t = d.kind === "title" ? "Title" : d.title;
          return (
            <button key={d.id} type="button" onClick={() => go(k)} title={t}
                    className={`rounded px-2 py-1 text-xs ${k === i ? "bg-brand-light font-medium text-brand-fg" : "text-fg-muted hover:bg-subtle"}`}>
              {k + 1}. {t.length > 34 ? `${t.slice(0, 32)}…` : t}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
