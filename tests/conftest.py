import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "deployment_artifacts"


def _weights_present() -> bool:
    cfg = ARTIFACTS / "model_config.json"
    if not cfg.is_file():
        return False
    return (ARTIFACTS / json.loads(cfg.read_text())["model"]["weights_file"]).is_file()


requires_weights = pytest.mark.skipif(not _weights_present(), reason="deployment_artifacts weights not present")


@pytest.fixture(scope="session")
def client():
    if not _weights_present():
        pytest.skip("deployment_artifacts weights not present")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def he_like_image(size=(224, 224), seed=0, mode="RGB") -> Image.Image:
    """Synthetic pink/purple texture loosely resembling an H&E patch."""
    rng = np.random.RandomState(seed)
    w, h = size
    base = np.array([215, 160, 200], dtype=np.float32)
    noise = rng.normal(0, 25, (h, w, 3))
    blobs = np.zeros((h, w), dtype=np.float32)
    for _ in range(40):
        cy, cx, r = rng.randint(0, h), rng.randint(0, w), rng.randint(3, 12)
        yy, xx = np.ogrid[:h, :w]
        blobs[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 1
    arr = base + noise - blobs[..., None] * np.array([90, 110, 40])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.convert(mode) if mode != "RGB" else img


def to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
