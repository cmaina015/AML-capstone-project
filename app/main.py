"""FastAPI app: data description, image upload + prediction, Grad-CAM explanation."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .artifacts import load_artifacts
from .inference import InputError, Predictor, decode_upload
from .settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# The Next.js frontend is compiled to static files (web/out) and served by this app.
WEB_DIR = Path(os.environ.get("WEB_DIR", Path(__file__).resolve().parent.parent / "web" / "out"))
SUMMARY_METRICS = ["malignant_recall", "benign_recall", "balanced_accuracy", "accuracy",
                   "malignant_precision", "weighted_f1", "roc_auc"]


def _mean_std(block: Dict) -> Dict:
    return {m: {"mean": block[m]["mean"], "std": block[m]["std"]} for m in SUMMARY_METRICS}


def _fold_summary(f: Dict) -> Dict:
    keep = SUMMARY_METRICS + ["threshold", "tn", "fp", "fn", "tp", "n"]
    return {
        "fold": f["fold"],
        "test_patients": f["test_patients"],
        "threshold": f["threshold_selection"]["threshold"],
        "test_at_selected": {k: f["test_at_selected"][k] for k in keep},
        "test_at_0.5": {k: f["test_at_0.5"][k] for k in keep},
    }


def _extra_limitations(cfg: Dict, metrics: Dict, stats: Dict) -> List[str]:
    """Limitations found when reviewing the training run, computed from the artifacts."""
    sel = cfg["decision"]["selection"]
    fold_t = sel.get("cv_fold_thresholds", [])
    val_benign = stats["splits"]["deployment_model"]["benign"]["val_patients"]
    items = [
        f"The decision threshold ({cfg['decision']['threshold']:.2f}) was chosen on a validation set "
        f"with only {val_benign} benign patients."
        + (f" Thresholds chosen the same way in the {len(fold_t)} cross-validation folds ranged from "
           f"{min(fold_t):.2f} to {max(fold_t):.2f}, so the exact cut-off is uncertain." if fold_t else ""),
    ]
    head = metrics["headline"]
    if head["at_0.5"]["balanced_accuracy"]["mean"] >= head["at_selected_threshold"]["balanced_accuracy"]["mean"]:
        items.append("In cross-validation, a fixed 0.5 threshold did at least as well on average as the "
                     "validation-selected thresholds (balanced accuracy "
                     f"{head['at_0.5']['balanced_accuracy']['mean']:.3f} vs "
                     f"{head['at_selected_threshold']['balanced_accuracy']['mean']:.3f}).")
    items += [
        "Grad-CAM sometimes highlights structures that are not diagnostic, such as adipose (fat) tissue. "
        "A highlighted region is where the model looked, not necessarily where a lesion is.",
        "The model was trained on 224x224 patches. Larger or non-square images (for example the "
        "original 700x460 BreakHis images) are squashed to 224x224, which changes tissue geometry.",
    ]
    return items


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    if settings.torch_num_threads:
        torch.set_num_threads(settings.torch_num_threads)
    art = load_artifacts(settings)
    predictor = Predictor(art.config, art.weights_path)
    app.state.settings = settings
    app.state.artifacts = art
    app.state.predictor = predictor
    app.state.info = build_info(art)
    app.state.info["limits"] = {"max_upload_mb": settings.max_upload_mb,
                                "formats": ["PNG", "JPEG", "TIFF", "BMP"]}
    log.info("Model loaded: %s, threshold=%.4f, torch threads=%d, CAM layer output %s",
             art.config["model"]["architecture"], predictor.threshold, torch.get_num_threads(),
             predictor.layer_output_shape)
    yield


def build_info(art) -> Dict:
    cfg, metrics, stats = art.config, art.metrics, art.dataset_stats
    head = metrics["headline"]
    base = metrics.get("leakage_prone_baseline")
    pp = cfg["preprocessing"]
    return {
        "intended_use": cfg["intended_use"],
        "model": {
            "architecture": cfg["model"]["architecture"],
            "weights_sha256": cfg["model"]["weights_sha256"],
            "classes": cfg["classes"],
            "positive_class": cfg["positive_class"],
            "threshold": cfg["decision"]["threshold"],
            "reference_threshold": cfg["decision"]["reference_threshold"],
            "decision_rule": cfg["decision"]["rule"],
            "threshold_selection": cfg["decision"]["selection"],
            "preprocessing": {
                "resize": f"{pp['resize']['method']} to {pp['resize']['size_hw'][1]}x{pp['resize']['size_hw'][0]} "
                          f"({pp['resize']['interpolation']}, aspect ratio not preserved, no crop)",
                "normalize": f"ImageNet mean {pp['normalize']['mean']} / std {pp['normalize']['std']}",
                "details": pp,
            },
            "training": cfg["training"],
            "versions": cfg["versions"],
        },
        "dataset": stats,
        "performance": {
            "headline": {
                "label": "Patient-level cross-validation",
                "description": head["description"],
                "protocol": head["protocol"],
                "n_folds": len(head["folds"]),
                "at_selected_threshold": _mean_std(head["at_selected_threshold"]),
                "at_0.5": _mean_std(head["at_0.5"]),
                "fold_thresholds": head["fold_thresholds"],
                "pooled_confusion_matrix": head["pooled_confusion_matrix"],
                "folds": [_fold_summary(f) for f in head["folds"]],
            },
            "leakage_prone_baseline": None if base is None else {
                "label": base["label"],
                "description": base["description"],
                "leakage": base["leakage"],
                "test_at_selected": {m: base["test_at_selected"][m] for m in SUMMARY_METRICS},
            },
            "deployed_model": metrics["deployed_model"],
        },
        "explainability": cfg["explainability"],
        "limitations": metrics["limitations"] + _extra_limitations(cfg, metrics, stats),
        "figures": sorted(art.figures),
    }


app = FastAPI(title="Breast Histopathology Classifier", version="1.0.0", lifespan=lifespan,
              description="Swin Tiny trained on BreakHis 100X with Grad-CAM explanations. "
                          "Research and education demo; not for clinical use.")
@app.get("/health")
def health(request: Request) -> Dict:
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "predictor")}


@app.get("/api/info")
def info(request: Request) -> Dict:
    return request.app.state.info


@app.get("/api/figures/{name}")
def figure(name: str, request: Request) -> FileResponse:
    figures = request.app.state.artifacts.figures
    if name not in figures:                       # allow-list: only exported figures
        raise HTTPException(status_code=404, detail="Unknown figure.")
    return FileResponse(figures[name], media_type="image/png")


@app.post("/api/predict")
def predict(request: Request,
            file: UploadFile = File(..., description="PNG, JPEG, TIFF or BMP image of an H&E patch"),
            explain: bool = Query(True, description="Also compute a Grad-CAM explanation (slower)")) -> Dict:
    settings: Settings = request.app.state.settings
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than {settings.max_upload_mb:g} MB.")
    predictor: Predictor = request.app.state.predictor
    try:
        decoded = decode_upload(data, predictor.size_hw)
    except InputError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    result = predictor.predict(decoded, explain=explain)
    log.info("predict explain=%s %sx%s -> %s p=%.4f in %.0f ms", explain, decoded.info["width"],
             decoded.info["height"], result["label"], result["p_malignant"], result["timing_ms"]["total"])
    return result


# Mounted last so /api/*, /health and /docs take precedence.
if (WEB_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    log.warning("Frontend not built (%s missing); run `npm run build` in web/. API only.", WEB_DIR)

    @app.get("/", include_in_schema=False)
    def frontend_missing() -> Dict:
        return {"detail": "Frontend not built. Run `npm ci && npm run build` in web/, or see /docs for the API."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
