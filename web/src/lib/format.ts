export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export const pm = (m: { mean: number; std: number }, digits = 1) =>
  `${(m.mean * 100).toFixed(digits)} ± ${(m.std * 100).toFixed(digits)}`;
export const num = (n: number) => n.toLocaleString("en-US");
export const fixed = (x: number, d = 3) => x.toFixed(d);

/** OpenCV COLORMAP_JET approximation for the colour-scale legend and the raw 7x7 grid view. */
export function jet(v: number): string {
  const c = (x: number) => Math.round(Math.min(Math.max(x, 0), 1) * 255);
  return `rgb(${c(1.5 - Math.abs(4 * v - 3))},${c(1.5 - Math.abs(4 * v - 2))},${c(1.5 - Math.abs(4 * v - 1))})`;
}

/** Deterministic PRNG (mulberry32) so procedural art renders identically on server and client. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Probability as a percentage that never claims certainty: 0.99997 -> ">99.9%", 0.00002 -> "<0.1%". */
export function probPct(p: number): string {
  if (p >= 0.9995) return ">99.9%";
  if (p < 0.0005) return "<0.1%";
  return `${(p * 100).toFixed(1)}%`;
}
