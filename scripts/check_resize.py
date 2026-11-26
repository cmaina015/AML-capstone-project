"""Check that web/src/lib/pilResize.ts is bit-identical to Pillow's bilinear resize (what the model saw).

Runs random images of many sizes (up- and down-scaling, non-square, tiny, 700x460 like BreakHis) through
torchvision's Resize and through the TypeScript port under Node, then compares every byte.
Usage: python scripts/check_resize.py      (needs Node >= 23.6 for .ts imports)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZES = [(700, 460), (460, 700), (224, 224), (225, 223), (1000, 1000), (50, 80), (3, 2), (2048, 1536),
         (701, 459), (224, 600), (128, 128), (1, 1)]

NODE = r"""
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [mod, dir] = process.argv.slice(2);
const { pilResizeBilinear } = await import(pathToFileURL(mod).href);
const cases = JSON.parse(readFileSync(dir + "/cases.json", "utf8"));
for (const c of cases) {
  const src = new Uint8Array(readFileSync(`${dir}/${c.name}.in`));
  writeFileSync(`${dir}/${c.name}.js`, pilResizeBilinear(src, c.w, c.h, 224, 224));
}
"""


def main():
    rng = np.random.default_rng(1)
    resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cases = []
        for i, (w, h) in enumerate(SIZES):
            # mix of noise and smooth gradients so both sharp and flat regions are exercised
            noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            if i % 2:
                yy, xx = np.mgrid[0:h, 0:w]
                noise[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
            img = Image.fromarray(noise)
            name = f"c{i}"
            (d / f"{name}.in").write_bytes(noise.tobytes())
            (d / f"{name}.pil").write_bytes(np.asarray(resize(img)).tobytes())
            cases.append({"name": name, "w": w, "h": h})
        (d / "cases.json").write_text(json.dumps(cases))
        (d / "run.mjs").write_text(NODE)
        subprocess.run(["node", str(d / "run.mjs"), str(ROOT / "web/src/lib/pilResize.ts"), str(d)], check=True)
        bad = 0
        for c in cases:
            a = np.frombuffer((d / f"{c['name']}.pil").read_bytes(), np.uint8).astype(int)
            b = np.frombuffer((d / f"{c['name']}.js").read_bytes(), np.uint8).astype(int)
            diff = int(np.abs(a - b).max()) if a.shape == b.shape else -1
            print(f"{c['w']:>5}x{c['h']:<5} max |PIL - TS| = {diff}  identical={diff == 0}")
            bad += diff != 0
    print("ALL IDENTICAL" if not bad else f"{bad} case(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
