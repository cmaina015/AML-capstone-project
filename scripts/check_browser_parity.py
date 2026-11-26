"""Compare the in-browser pipeline (ONNX Runtime Web) with the Python/PyTorch pipeline on the same files.

Needs the FastAPI app running (it serves both the static site and POST /api/predict):
    uvicorn app.main:app --host 127.0.0.1 --port 7860
and Playwright with Chrome:  pip install playwright  (uses the installed Chrome via channel="chrome")

    python scripts/check_browser_parity.py IMAGE [IMAGE ...] [--url http://127.0.0.1:7860]

For every file it uploads the image in the browser and to the API, then reports the difference in
p(malignant), whether label and warnings agree, and pixel differences of the resized input, the Grad-CAM
7x7 grid, the overlay and the heatmap.
"""
import argparse
import base64
import io
import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def api_predict(url: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{url}/api/predict?explain=true", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def png(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    args = ap.parse_args()

    rows, fails = [], 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(args.url + "/")
        page.wait_for_selector("text=Model ready", timeout=300_000)
        for path in args.images:
            server = api_predict(args.url, path)
            page.evaluate("window.__lastPrediction = null")
            page.set_input_files("#file", str(path))
            page.wait_for_function("window.__lastPrediction !== null", timeout=120_000)
            web = page.evaluate("window.__lastPrediction")

            d_p = abs(web["p_malignant"] - server["p_malignant"])
            same_label = web["label"] == server["label"]
            same_warn = web["warnings"] == server["warnings"]
            d_in = int(np.abs(png(web["images"]["model_input"]) - png(server["images"]["model_input"])).max())
            d_grid = float(np.abs(np.array(web["explanation"]["grid"]) - np.array(server["explanation"]["grid"])).max())
            ov = np.abs(png(web["images"]["overlay"]) - png(server["images"]["overlay"]))
            hm = np.abs(png(web["images"]["heatmap"]) - png(server["images"]["heatmap"]))
            ok = d_p < 1e-4 and same_label and same_warn and d_in <= 1 and d_grid < 1e-3
            fails += not ok
            rows.append({
                "file": path.name, "server_p": round(server["p_malignant"], 6), "browser_p": round(web["p_malignant"], 6),
                "abs_diff_p": f"{d_p:.1e}", "label": f"{server['label']}/{web['label']}", "same_warnings": same_warn,
                "input_px_max": d_in, "cam_grid_max": f"{d_grid:.1e}",
                "overlay_px_max": int(ov.max()), "overlay_px_differing_%": round(100 * float((ov > 0).any(-1).mean()), 3),
                "heatmap_px_max": int(hm.max()), "browser_ms": web["timing_ms"]["total"], "ok": ok,
            })
            if not same_warn:
                print("  warnings differ:", server["warnings"], "vs", web["warnings"])
        browser.close()
    for r in rows:
        print(json.dumps(r))
    print("ALL MATCH" if not fails else f"{fails} file(s) outside tolerance")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
