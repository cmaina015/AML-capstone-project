"""Stage 4, export: write the deployable artifacts and check that they reload correctly.

Writes, to cfg.out_dir, the same files as deployment_artifacts/:
  swin_tiny_breakhis.safetensors, model_config.json, metrics.json, dataset_stats.json,
  split_assignments.csv, requirements-pinned.txt and the figures used by the app.
Then reloads the model from those files only (as the API does) and checks that CPU probabilities and
Grad-CAM match the trained model.
"""
import hashlib
import json
import logging
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from app.inference import get_cam_target_layer, make_swin_reshape_transform  # noqa: E402

from .config import CLASSES, LABEL_MAP, WEIGHTS_FILENAME, BREAKHIS_URL, Config  # noqa: E402
from .data import AUGMENTATION_DESCRIPTIONS, build_eval_transform, load_rgb, make_loader, preprocessing_spec  # noqa: E402
from .train import load_final_model, predict_proba, summarise  # noqa: E402

log = logging.getLogger(__name__)
CAM_TARGET_LAYER_PATH = "layers[-1].blocks[-1]"


def _dist_version(*names: str) -> Optional[str]:
    for n in names:
        try:
            return pkg_version(n)
        except PackageNotFoundError:
            continue
    return None


def versions() -> Dict:
    return {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "grad-cam": _dist_version("grad-cam"), "safetensors": _dist_version("safetensors"),
        "numpy": np.__version__, "pillow": _dist_version("pillow", "Pillow"), "opencv": cv2.__version__,
        "opencv_distribution": next((f"{n}=={_dist_version(n)}" for n in
                                     ("opencv-python-headless", "opencv-python", "opencv-contrib-python")
                                     if _dist_version(n)), None),
        "scikit-learn": _dist_version("scikit-learn"), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")


# ------------------------------------------------------------------ figures
def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_class_distribution(stats: Dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in zip(axes, ["images", "patients"]):
        vals = [stats["classes"][c][key] for c in CLASSES]
        ax.bar(CLASSES, vals, color=["#4C72B0", "#C44E52"])
        ax.set_title(f"{key.capitalize()} per class ({stats['magnification']})")
        ax.set_ylabel(key)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom")
    _save(fig, out / "class_distribution.png")


def figure_cv(cv: List[Dict], out: Path) -> np.ndarray:
    y = np.concatenate([r["y_test"] for r in cv])
    pred = np.concatenate([(r["p_test"] >= r["threshold_selection"]["threshold"]).astype(int) for r in cv])
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Patient-level CV, pooled test folds\n(each fold at its selected threshold)")
    _save(fig, out / "cv_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4.4))
    for r in cv:
        fpr, tpr, _ = roc_curve(r["y_test"], r["p_test"])
        ax.plot(fpr, tpr, alpha=0.8, label=f"fold {r['fold'] + 1} (AUC {r['test_at_selected']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate (1 - benign recall)")
    ax.set_ylabel("True positive rate (malignant recall)")
    ax.set_title("Patient-level CV - ROC per test fold")
    ax.legend(fontsize=8)
    _save(fig, out / "cv_roc_curve.png")
    return cm


def figure_history(history: Dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ep = range(1, len(history["train_loss"]) + 1)
    for ax, key in zip(axes, ["loss", "acc"]):
        ax.plot(ep, history[f"train_{key}"], marker="o", label="train")
        ax.plot(ep, history[f"val_{key}"], marker="o", label="validation")
        ax.set_title(f"Deployment model - {key}")
        ax.set_xlabel("epoch")
        ax.legend()
    _save(fig, out / "training_history.png")


# ------------------------------------------------------------------ Grad-CAM shapes and reload check
def cam_shapes(model: torch.nn.Module, image_size: int) -> Dict:
    target = get_cam_target_layer(model)
    shapes = {}
    hooks = [target.register_forward_hook(lambda m, i, o: shapes.__setitem__("block output", tuple(o.shape))),
             target.norm2.register_forward_hook(lambda m, i, o: shapes.__setitem__("block.norm2 output", tuple(o.shape))),
             model.norm.register_forward_hook(lambda m, i, o: shapes.__setitem__("model.norm output", tuple(o.shape)))]
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    grid = tuple(make_swin_reshape_transform(model.num_features)(torch.zeros(shapes["block output"])).shape[-2:])
    assert grid == (image_size // 32, image_size // 32), f"Unexpected CAM grid {grid}"
    return {"shapes": shapes, "grid": grid}


def reload_check(out: Path, model: torch.nn.Module, val_df: pd.DataFrame, cfg: Config) -> Dict:
    """Load only what the API loads (model_config.json + weights) and compare CPU probabilities."""
    from app.inference import Predictor
    mc = json.loads((out / "model_config.json").read_text())
    w = out / mc["model"]["weights_file"]
    assert sha256_file(w) == mc["model"]["weights_sha256"], "Weights checksum mismatch"
    reloaded = timm.create_model(mc["model"]["architecture"], pretrained=False, num_classes=mc["model"]["num_classes"])
    reloaded.load_state_dict(load_file(str(w)), strict=True)
    tf = build_eval_transform(mc["preprocessing"])
    cpu = torch.device("cpu")
    loader = make_loader(val_df, tf, False, cfg.seed, cfg.batch_size, 0, cpu)
    _, p_orig = predict_proba(model.to(cpu), loader, cpu)
    _, p_reload = predict_proba(reloaded, loader, cpu)
    diff = float(np.abs(p_orig - p_reload).max())
    assert diff < 1e-5, f"Reloaded model differs from the trained model (max diff {diff})"
    # The API's own Predictor must agree as well.
    predictor = Predictor(mc, w)
    from app.inference import decode_upload
    data = Path(val_df["path"].iloc[0]).read_bytes()
    p_api = predictor.predict(decode_upload(data, predictor.size_hw), explain=True)["p_malignant"]
    assert abs(p_api - p_orig[0]) < 1e-5, "API Predictor disagrees with the trained model"
    return {"max_abs_prob_diff_cpu": diff, "api_predictor_diff": float(abs(p_api - p_orig[0]))}


# ------------------------------------------------------------------ main
def run(cfg: Config, trained: Dict) -> Path:
    out = cfg.out
    cv, baseline, final, splits = trained["cv"], trained["baseline"], trained["final"], trained["splits"]
    model = load_final_model(cfg)
    final_train_df, final_val_df = splits["deployment"]
    threshold = final["threshold_selection"]["threshold"]
    pp = preprocessing_spec(cfg.image_size)
    ver = versions()
    cam = cam_shapes(model, cfg.image_size)
    hp = cfg.hp

    # 1. weights
    shutil.copyfile(cfg.work / "runs" / cfg.run_name / "final.safetensors", out / WEIGHTS_FILENAME)
    weights_sha = sha256_file(out / WEIGHTS_FILENAME)

    # 2. model_config.json (schema_version 1, read by app/ and the browser export)
    fold_thresholds = [r["threshold_selection"]["threshold"] for r in cv]
    stats = json.loads((cfg.work / "dataset_stats.json").read_text())
    model_config = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "intended_use": "Research and education demo. Not a medical device; not for clinical diagnosis.",
        "model": {"library": "timm", "architecture": cfg.model_name, "num_classes": len(CLASSES),
                  "weights_file": WEIGHTS_FILENAME, "weights_format": "safetensors", "weights_sha256": weights_sha,
                  "load_recipe": "timm.create_model(architecture, pretrained=False, num_classes=2); "
                                 "model.load_state_dict(safetensors.torch.load_file(weights_file), strict=True); "
                                 "model.eval()"},
        "classes": {str(i): c for c, i in LABEL_MAP.items()},
        "positive_class": "malignant",
        "decision": {
            "score": "p_malignant = softmax(logits.double(), dim=1)[:, 1]",
            "rule": "predict malignant if p_malignant >= threshold, else benign",
            "threshold": threshold,
            "reference_threshold": cfg.reference_threshold,
            "selection": {
                "data": "deployment validation holdout (patient-level, ~15% of patients)",
                "criterion": f"maximise validation benign recall s.t. malignant recall >= "
                             f"{cfg.target_malignant_recall}; ties -> higher malignant recall; t midway "
                             "between adjacent validation scores",
                **final["threshold_selection"],
                "n_val_patients": int(final_val_df["patient_id"].nunique()),
                "n_val_images": int(len(final_val_df)),
                "cv_fold_thresholds": fold_thresholds,
            },
        },
        "preprocessing": pp,
        "training_input": {"image_sizes_seen": stats["image_sizes"],
                           "note": "Uploads with a different size or aspect ratio are squashed to 224x224, which "
                                   "may distort tissue geometry compared with the training images."},
        "training": {
            "pretrained_weights": f"timm default for {cfg.model_name} (ImageNet)",
            "optimizer": "AdamW", "learning_rate": hp["lr"], "weight_decay": hp["weight_decay"],
            "backbone_lr_mult": hp["backbone_lr_mult"],
            "scheduler": (f"CosineAnnealingLR(T_max={hp['epochs']})" if hp["warmup_epochs"] == 0 else
                          f"linear warm-up {hp['warmup_epochs']} epoch(s), then cosine over {hp['epochs']} epochs"),
            "loss": (f"CrossEntropyLoss(label_smoothing={hp['label_smoothing']}"
                     + (", inverse-frequency class weights" if hp["class_weights"] else "") + ")"
                     + (f", mixup alpha={hp['mixup_alpha']}" if hp["mixup_alpha"] > 0 else "")),
            "drop_path_rate": hp["drop_path"],
            "batch_size": cfg.batch_size, "max_epochs": hp["epochs"], "mixed_precision": "fp16 autocast (train only)",
            "early_stopping": {"monitor": "val_loss (plain cross-entropy)", "patience": hp["patience"],
                               "min_delta": cfg.early_stopping_min_delta, "restore": "min val_loss weights"},
            "augmentation": AUGMENTATION_DESCRIPTIONS[hp["augment"]],
            "seed": cfg.seed,
            "best_epoch": final["training"]["best_epoch"],
            "epochs_run": final["training"]["epochs_run"],
            "n_train_patients": int(final_train_df["patient_id"].nunique()),
            "n_train_images": int(len(final_train_df)),
            "pipeline": {"command": "python -m pipeline run", "run_name": cfg.run_name, "hyperparameters": hp},
        },
        "explainability": {
            "method": "Grad-CAM",
            "library": f"grad-cam=={ver['grad-cam']} (import pytorch_grad_cam)",
            "target_layer": CAM_TARGET_LAYER_PATH,
            "observed_shapes": {k: list(v) for k, v in cam["shapes"].items()},
            "reshape_transform": "(B,H,W,C)->permute(0,3,1,2); (B,N,C)->reshape to sqrt(N) grid; (B,C,H,W) unchanged",
            "cam_grid_hw": list(cam["grid"]),
            "target_class": "predicted class (after threshold)",
            "note": f"Coarse: computed on a {cam['grid'][0]}x{cam['grid'][1]} grid (one cell per ~32x32 px) and "
                    "upsampled to 224x224. Shows influential regions, not individual cells.",
            "overlay": "pytorch_grad_cam.utils.image.show_cam_on_image(rgb_224_float01, cam, use_rgb=True, image_weight=0.6)",
        },
        "data": {"dataset": "BreakHis", "magnification": cfg.magnification, "stain": "H&E", "url": BREAKHIS_URL},
        "versions": ver,
    }
    write_json(model_config, out / "model_config.json")

    # 3. metrics.json
    cm = figure_cv(cv, out)
    ref = f"test_at_{cfg.reference_threshold}"
    shift = {}
    for mag in cfg.shift_magnifications:
        per_fold = [r["shift"][mag] for r in cv if mag in r.get("shift", {})]
        if not per_fold:
            continue
        table = pd.DataFrame([f["metrics"] for f in per_fold])
        shift[mag] = {m: {"mean": float(table[m].mean()), "std": float(table[m].std(ddof=1))} for m in
                      ["balanced_accuracy", "malignant_recall", "benign_recall", "roc_auc"]}
        shift[mag]["n_images"] = int(table["n"].sum())
    metrics = {
        "headline": {
            "description": f"Patient-level {cfg.n_folds}-fold StratifiedGroupKFold (test folds), each fold scored at "
                           "its own validation-selected threshold. This is the performance estimate shown in the app.",
            "protocol": {"outer": f"StratifiedGroupKFold(n_splits={cfg.n_folds}, shuffle=True, random_state={cfg.seed})",
                         "groups": "patient_id parsed from filename",
                         "inner_validation": f"patient-grouped holdout ~{cfg.inner_val_fraction:.0%} of training patients",
                         "threshold_rule": f"max validation benign recall s.t. malignant recall >= {cfg.target_malignant_recall}"},
            "at_selected_threshold": summarise(cv, "test_at_selected"),
            "at_0.5": summarise(cv, ref),
            "fold_thresholds": fold_thresholds,
            "pooled_confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist(), "rows": "true", "cols": "predicted"},
            "folds": [{"fold": r["fold"] + 1, "test_patients": r["test_patients"],
                       "best_epoch": r["training"]["best_epoch"], "threshold_selection": r["threshold_selection"],
                       "test_at_selected": r["test_at_selected"], "test_at_0.5": r[ref]} for r in cv],
        },
        "magnification_shift": {
            "description": f"Each CV fold's model scored on its own unseen test patients at other magnifications "
                           f"(trained on {cfg.magnification}), at the fold's selected threshold.",
            "results": shift,
        } if shift else None,
        "leakage_prone_baseline": None if baseline is None else {
            "label": "leakage-prone baseline",
            "description": "Original image-level stratified 70/15/15 split (seed 42). Images from the same patient can "
                           "appear in train and test, so these numbers are optimistic. Not shown as the app headline.",
            "leakage": baseline["leakage"], "threshold_selection": baseline["threshold_selection"],
            "test_at_selected": baseline["test_at_selected"], "test_at_0.5": baseline[ref],
        },
        "deployed_model": {
            "description": "Trained on all patients except a patient-level validation holdout. It has no held-out test "
                           "set of its own; use the headline CV estimate for expected performance.",
            "threshold": threshold, "validation_at_threshold": final["val_at_selected"],
            "best_epoch": final["training"]["best_epoch"],
        },
        "limitations": [
            "Validation data was used for both early stopping and threshold selection, so validation metrics are "
            "optimistic and the threshold is fitted to a few patients.",
            f"Only {stats['n_patients']} patients from a single institution (BreakHis); no external validation.",
            f"Single magnification ({cfg.magnification}), H&E staining; other magnifications, stains or scanners are out of scope.",
            "Image-level predictions; no slide- or patient-level aggregation.",
            "Grad-CAM is computed on a coarse 7x7 grid and highlights influential regions, not diagnostic features.",
            "Research and education demo only; not for clinical use.",
        ],
    }
    write_json(metrics, out / "metrics.json")

    # 4. dataset_stats.json with split summaries
    stats["splits"] = {
        "headline_cv": f"{cfg.n_folds}-fold patient-level StratifiedGroupKFold",
        "deployment_model": {c: {"train_patients": int(final_train_df.loc[final_train_df["label"] == c, "patient_id"].nunique()),
                                 "train_images": int((final_train_df["label"] == c).sum()),
                                 "val_patients": int(final_val_df.loc[final_val_df["label"] == c, "patient_id"].nunique()),
                                 "val_images": int((final_val_df["label"] == c).sum())} for c in CLASSES},
    }
    write_json(stats, out / "dataset_stats.json")

    # 5. splits, pinned requirements, figures
    shutil.copyfile(cfg.work / "splits.csv", out / "split_assignments.csv")
    base = lambda v: v.split("+")[0] if v else v  # noqa: E731
    (out / "requirements-pinned.txt").write_text("\n".join([
        "# Generated by python -m pipeline: versions of the training environment.",
        f"torch=={base(ver['torch'])}", f"torchvision=={base(ver['torchvision'])}", f"timm=={ver['timm']}",
        f"grad-cam=={ver['grad-cam']}", f"safetensors=={ver['safetensors']}", f"numpy=={ver['numpy']}",
        f"pillow=={ver['pillow']}",
        (f"opencv-python-headless=={ver['opencv_distribution'].split('==')[1]}" if ver["opencv_distribution"]
         else "# opencv-python-headless: version not detected"),
    ]) + "\n")
    figure_class_distribution(stats, out)
    figure_history(final["training"]["history"], out)
    write_json({"config": cfg.to_dict(), "versions": ver}, out / "run_config.json")

    # 6. reload check (what the API will load)
    check = reload_check(out, model, final_val_df, cfg)
    log.info("Reload check passed: %s", check)
    log.info("Artifacts written to %s", out.resolve())
    return out
