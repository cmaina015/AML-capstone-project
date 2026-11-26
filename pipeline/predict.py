"""Stage 5, predict: classify image files with exported artifacts, from the command line.

Uses the same Predictor as the API (app/inference.py), so results are identical to POST /api/predict.

    python -m pipeline predict IMAGE [IMAGE ...] [--artifacts deployment_artifacts] [--heatmap-dir out/]
"""
import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.artifacts import load_artifacts
from app.inference import InputError, Predictor, decode_upload
from app.settings import Settings

log = logging.getLogger(__name__)


def run(images: List[str], artifact_dir: str = "deployment_artifacts", heatmap_dir: Optional[str] = None) -> List[Dict]:
    settings = Settings.from_env()
    settings = Settings(**{**settings.__dict__, "artifact_dir": Path(artifact_dir)})
    art = load_artifacts(settings)
    predictor = Predictor(art.config, art.weights_path)
    results = []
    for path in images:
        try:
            decoded = decode_upload(Path(path).read_bytes(), predictor.size_hw)
        except InputError as exc:
            results.append({"file": path, "error": str(exc)})
            continue
        r = predictor.predict(decoded, explain=heatmap_dir is not None)
        row = {"file": path, "label": r["label"], "p_malignant": round(r["p_malignant"], 6),
               "threshold": round(r["threshold"], 4), "warnings": r.get("warnings", [])}
        if heatmap_dir and r.get("images", {}).get("heatmap"):
            out = Path(heatmap_dir)
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"{Path(path).stem}_gradcam.png"
            target.write_bytes(base64.b64decode(r["images"]["overlay"] if "overlay" in r["images"]
                                                else r["images"]["heatmap"]))
            row["heatmap"] = str(target)
        results.append(row)
    print(json.dumps(results, indent=2))
    return results
