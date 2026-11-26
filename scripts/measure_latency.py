"""Measure end-to-end latency of a running server (local uvicorn or Docker container).

Usage:  python scripts/measure_latency.py [--url http://localhost:7860] [--n 20] [--image path.png]
Without --image, a synthetic 224x224 H&E-like patch is used.
"""
import argparse
import io
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def synthetic_png() -> bytes:
    rng = np.random.RandomState(0)
    arr = np.clip(np.array([215, 160, 200]) + rng.normal(0, 30, (224, 224, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def post(url: str, data: bytes, filename: str) -> float:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
    return (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--image", type=Path)
    args = ap.parse_args()

    data = args.image.read_bytes() if args.image else synthetic_png()
    name = args.image.name if args.image else "synthetic.png"
    for explain in (False, True):
        url = f"{args.url}/api/predict?explain={str(explain).lower()}"
        post(url, data, name)  # warm-up
        times = sorted(post(url, data, name) for _ in range(args.n))
        p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
        print(f"explain={str(explain):<5}  n={args.n}  median={statistics.median(times):7.0f} ms  "
              f"p95={p95:7.0f} ms  min={times[0]:7.0f} ms  max={times[-1]:7.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
