/**
 * Port of Pillow's Image.resize(size, Image.BILINEAR) for 8-bit RGB (libImaging/Resample.c), which is what
 * torchvision's Resize does on a PIL image and therefore what the model was trained with.
 * Same coefficients, same 22-bit fixed point, same horizontal-then-vertical passes, so the output is
 * bit-identical to Pillow (checked by scripts/check_resize.mjs). A browser canvas resize is NOT.
 */
const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);

type Coeffs = { ksize: number; bounds: Int32Array; kk: Float64Array };

function bilinear(x: number): number {
  if (x < 0) x = -x;
  return x < 1 ? 1 - x : 0;
}

function precompute(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    const base = xx * ksize;
    for (let x = 0; x < xmax; x++) {
      const w = bilinear((x + xmin - center + 0.5) * ss);
      kk[base + x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) if (ww !== 0) kk[base + x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  // normalize_coeffs_8bpc: C (int) casts truncate toward zero
  for (let i = 0; i < kk.length; i++) kk[i] = Math.trunc(kk[i] < 0 ? -0.5 + kk[i] * ONE : 0.5 + kk[i] * ONE);
  return { ksize, bounds, kk };
}

function clip8(v: number): number {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

/** src: packed RGB (3 bytes/pixel), w x h. Returns packed RGB outW x outH. */
export function pilResizeBilinear(src: Uint8Array, w: number, h: number, outW: number, outH: number): Uint8Array {
  if (w === outW && h === outH) return src.slice();
  let img = src;
  let iw = w;
  let ih = h;

  const vert = precompute(h, outH);
  const horiz = precompute(w, outW);
  const needH = outW !== w;
  const needV = outH !== h;

  if (needH) {
    // only the source rows the vertical pass will read
    const yFirst = vert.bounds[0];
    const yLast = vert.bounds[outH * 2 - 2] + vert.bounds[outH * 2 - 1];
    const rows = needV ? yLast - yFirst : h;
    const y0 = needV ? yFirst : 0;
    if (needV) for (let i = 0; i < outH; i++) vert.bounds[i * 2] -= yFirst;
    const tmp = new Uint8Array(outW * rows * 3);
    const { ksize, bounds, kk } = horiz;
    for (let yy = 0; yy < rows; yy++) {
      const rowIn = (yy + y0) * iw * 3;
      const rowOut = yy * outW * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const k = xx * ksize;
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let x = 0; x < xmax; x++) {
          const p = rowIn + (x + xmin) * 3;
          const c = kk[k + x];
          s0 += img[p] * c;
          s1 += img[p + 1] * c;
          s2 += img[p + 2] * c;
        }
        const o = rowOut + xx * 3;
        tmp[o] = clip8(s0);
        tmp[o + 1] = clip8(s1);
        tmp[o + 2] = clip8(s2);
      }
    }
    img = tmp;
    iw = outW;
    ih = rows;
  }

  if (needV) {
    const out = new Uint8Array(outW * outH * 3);
    const { ksize, bounds, kk } = vert;
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const k = yy * ksize;
      for (let xx = 0; xx < iw; xx++) {
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let y = 0; y < ymax; y++) {
          const p = ((y + ymin) * iw + xx) * 3;
          const c = kk[k + y];
          s0 += img[p] * c;
          s1 += img[p + 1] * c;
          s2 += img[p + 2] * c;
        }
        const o = (yy * outW + xx) * 3;
        out[o] = clip8(s0);
        out[o + 1] = clip8(s1);
        out[o + 2] = clip8(s2);
      }
    }
    return out;
  }
  void ih;
  return img;
}
