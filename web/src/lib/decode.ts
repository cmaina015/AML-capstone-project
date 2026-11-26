/**
 * Browser version of app/inference.py:decode_upload. Produces the same packed RGB pixels that
 * PIL.Image.open(f).convert("RGB") gives, plus the same info and warnings.
 *
 *  PNG  - decoded in JS (fast-png), so alpha is dropped exactly as PIL does (a canvas would premultiply).
 *  TIFF - decoded in JS (utif2); browsers cannot open TIFF.
 *  JPEG / BMP - browser decoder with colour management and EXIF rotation turned off, like PIL.
 */
import { convertIndexedToRgb, decode as decodePng } from "fast-png";
import * as UTIF from "utif2";

export class InputError extends Error {}

export type Decoded = {
  rgb: Uint8Array; // packed RGB, width * height * 3
  width: number;
  height: number;
  info: { format: string; mode: string; width: number; height: number; aspect_ratio: number; frames: number };
  warnings: string[];
};

type Limits = { maxBytes: number; maxPixels: number; aspectTolerance: number; sizeHW: [number, number] };

function sniff(b: Uint8Array): string | null {
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return "PNG";
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "JPEG";
  if (b[0] === 0x42 && b[1] === 0x4d) return "BMP";
  if ((b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a && b[3] === 0) ||
      (b[0] === 0x4d && b[1] === 0x4d && b[2] === 0 && b[3] === 0x2a)) return "TIFF";
  if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46) return "GIF";
  if (b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46) return "WEBP";
  return null;
}

/** Pixel size from the file header, so huge images are rejected before decoding (like MAX_PIXELS). */
function headerSize(fmt: string, b: Uint8Array): [number, number] | null {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (fmt === "PNG" && b.length >= 24) return [dv.getUint32(16), dv.getUint32(20)];
  if (fmt === "BMP" && b.length >= 26) return [Math.abs(dv.getInt32(18, true)), Math.abs(dv.getInt32(22, true))];
  if (fmt === "JPEG") {
    let i = 2;
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) { i++; continue; }
      const m = b[i + 1];
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return [dv.getUint16(i + 7), dv.getUint16(i + 5)];
      if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
      i += 2 + dv.getUint16(i + 2);
    }
  }
  return null;
}

function jpegMode(b: Uint8Array): string {
  for (let i = 2; i + 9 < b.length; ) {
    if (b[i] !== 0xff) { i++; continue; }
    const m = b[i + 1];
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { 1: "L", 3: "RGB", 4: "CMYK" }[b[i + 9]] ?? "RGB";
    if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
    i += 2 + ((b[i + 2] << 8) | b[i + 3]);
  }
  return "RGB";
}

function bmpMode(b: Uint8Array): string {
  const bpp = b[28] | (b[29] << 8);
  return bpp <= 8 ? "P" : "RGB";
}

function decodePngRgb(bytes: Uint8Array) {
  const png = decodePng(bytes);
  const { width: w, height: h, depth, channels } = png;
  const n = w * h;
  const rgb = new Uint8Array(n * 3);
  if (png.palette) {
    const full = convertIndexedToRgb(png);
    const c = png.palette[0].length;
    for (let i = 0; i < n; i++) {
      rgb[i * 3] = full[i * c]; rgb[i * 3 + 1] = full[i * c + 1]; rgb[i * 3 + 2] = full[i * c + 2];
    }
    const alpha = c === 4 || (png.transparency?.length ?? 0) > 0;
    return { rgb, w, h, mode: "P", alpha };
  }
  const d = png.data;
  // PIL: 16-bit colour keeps the high byte; 16-bit grey ("I;16") is clipped to 255 on convert("RGB").
  const sample = (v: number) => (depth === 16 ? v >> 8 : v);
  if (channels >= 3) {
    for (let i = 0; i < n; i++) {
      rgb[i * 3] = sample(d[i * channels]); rgb[i * 3 + 1] = sample(d[i * channels + 1]); rgb[i * 3 + 2] = sample(d[i * channels + 2]);
    }
    return { rgb, w, h, mode: channels === 4 ? "RGBA" : "RGB", alpha: channels === 4 };
  }
  // grey or grey+alpha; depths 1/2/4 are still bit-packed per row
  const grey = new Uint8Array(n);
  if (depth >= 8) {
    for (let i = 0; i < n; i++) {
      const v = d[i * channels];
      grey[i] = depth === 16 ? (channels === 2 ? v >> 8 : Math.min(255, v)) : v;
    }
  } else {
    const rowBytes = Math.ceil((w * depth) / 8);
    const maxv = (1 << depth) - 1;
    const scale = 255 / maxv;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const bit = x * depth;
        const v = (d[y * rowBytes + (bit >> 3)] >> (8 - depth - (bit & 7))) & maxv;
        grey[y * w + x] = Math.round(v * scale);
      }
  }
  for (let i = 0; i < n; i++) rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = grey[i];
  const mode = channels === 2 ? "LA" : depth === 1 ? "1" : depth === 16 ? "I;16" : "L";
  return { rgb, w, h, mode, alpha: channels === 2 };
}

function decodeTiffRgb(bytes: Uint8Array) {
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const ifds = UTIF.decode(buf);
  if (!ifds.length) throw new InputError("Could not read the file as an image (no TIFF pages).");
  const ifd = ifds[0];
  UTIF.decodeImage(buf, ifd);
  const rgba = UTIF.toRGBA8(ifd);
  const w = ifd.width as number;
  const h = ifd.height as number;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  const photometric = (ifd.t262 as number[] | undefined)?.[0] ?? 2;
  const spp = (ifd.t277 as number[] | undefined)?.[0] ?? 1;
  const bps = (ifd.t258 as number[] | undefined)?.[0] ?? 1;
  let mode = "RGB";
  if (photometric <= 1) mode = bps === 1 ? "1" : spp >= 2 ? "LA" : bps === 16 ? "I;16" : "L";
  else if (photometric === 3) mode = "P";
  else if (photometric === 5) mode = "CMYK";
  else if (spp >= 4) mode = "RGBA";
  return { rgb, w, h, mode, alpha: mode === "RGBA" || mode === "LA", frames: ifds.length };
}

/**
 * PIL ignores the EXIF Orientation tag, but browsers apply it even with imageOrientation "none".
 * Return a copy of the JPEG with Orientation set to 1 (as stored), so the pixels match PIL.
 */
function neutraliseJpegOrientation(b: Uint8Array): Uint8Array {
  const out = b.slice();
  const dv = new DataView(out.buffer);
  for (let i = 2; i + 4 < out.length; ) {
    if (out[i] !== 0xff) break;
    const m = out[i + 1];
    if (m === 0xda || m === 0xd9) break; // start of scan: no more metadata
    const len = dv.getUint16(i + 2);
    if (m === 0xe1 && dv.getUint32(i + 4) === 0x45786966 && dv.getUint16(i + 8) === 0) {
      const t = i + 10; // TIFF header
      const le = dv.getUint16(t) === 0x4949;
      const ifd = t + dv.getUint32(t + 4, le);
      const n = dv.getUint16(ifd, le);
      for (let e = 0; e < n; e++) {
        const p = ifd + 2 + e * 12;
        if (p + 12 > i + 2 + len) break;
        if (dv.getUint16(p, le) === 0x0112) dv.setUint16(p + 8, 1, le);
      }
    }
    i += 2 + len;
  }
  return out;
}

async function decodeBrowserRgb(file: Blob) {
  const bmp = await createImageBitmap(file, {
    colorSpaceConversion: "none", premultiplyAlpha: "none", imageOrientation: "none",
  } as ImageBitmapOptions);
  const w = bmp.width;
  const h = bmp.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d", { colorSpace: "srgb", willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return { rgb, w, h };
}

export async function decodeUpload(file: File, lim: Limits): Promise<Decoded> {
  if (file.size === 0) throw new InputError("The uploaded file is empty.");
  if (file.size > lim.maxBytes) throw new InputError(`File is larger than ${lim.maxBytes / 1024 / 1024} MB.`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const fmt = sniff(bytes);
  if (fmt === null) throw new InputError("Could not read the file as an image.");
  if (!["PNG", "JPEG", "TIFF", "BMP"].includes(fmt)) throw new InputError(`Unsupported image format ${fmt}. Use PNG, JPEG, TIFF or BMP.`);

  const hs = headerSize(fmt, bytes);
  if (hs && hs[0] * hs[1] > lim.maxPixels) throw new InputError(`Image is too large (${hs[0]}x${hs[1]} pixels).`);

  let out: { rgb: Uint8Array; w: number; h: number; mode: string; alpha: boolean; frames: number };
  try {
    if (fmt === "PNG") out = { ...decodePngRgb(bytes), frames: 1 };
    else if (fmt === "TIFF") out = decodeTiffRgb(bytes);
    else {
      const blob = fmt === "JPEG" ? new Blob([neutraliseJpegOrientation(bytes) as BlobPart], { type: "image/jpeg" }) : file;
      const r = await decodeBrowserRgb(blob);
      out = { ...r, mode: fmt === "JPEG" ? jpegMode(bytes) : bmpMode(bytes), alpha: false, frames: 1 };
    }
  } catch (e) {
    if (e instanceof InputError) throw e;
    throw new InputError(`Could not read the file as an image (${(e as Error)?.name ?? "error"}).`);
  }
  const { w, h, mode } = out;
  if (w * h > lim.maxPixels) throw new InputError(`Image is too large (${w}x${h} pixels).`);

  const [th, tw] = lim.sizeHW;
  const notes: string[] = [];
  if (out.frames > 1) notes.push(`The file has ${out.frames} frames; only the first was used.`);
  if (["1", "L", "LA", "I", "I;16", "F"].includes(mode))
    notes.push("The image is not in colour. The model was trained on colour H&E images, so the result is unreliable.");
  if (out.alpha) notes.push("Transparency was removed (alpha channel dropped).");
  const aspect = w / h;
  if (Math.abs(aspect - tw / th) > lim.aspectTolerance)
    notes.push(`The image is ${w}x${h} (aspect ratio ${aspect.toFixed(2)}). It is squashed to ${tw}x${th}, ` +
      `which distorts tissue shapes; the training images were ${tw}x${th} squares.`);
  else if (w !== tw || h !== th) notes.push(`Resized from ${w}x${h} to ${tw}x${th}.`);

  return {
    rgb: out.rgb, width: w, height: h, warnings: notes,
    info: { format: fmt, mode, width: w, height: h, aspect_ratio: Math.round(aspect * 1e4) / 1e4, frames: out.frames },
  };
}
