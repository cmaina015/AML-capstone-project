"""Stage 3, train: patient-level cross-validation, leakage baseline, magnification-shift test, final model.

* Each CV fold trains on its training patients, early-stops and chooses the threshold on its validation
  patients, and is scored once on its test patients. Finished folds are saved to
  work/runs/<run_name>/ and reloaded on a re-run (resumable on Colab).
* Shift test: each fold's model also scores ITS OWN test patients at 40X/200X/400X. The patients are
  unseen, so this measures the effect of a different magnification without patient leakage.
* The final (deployed) model trains on all patients except a patient-level validation holdout.
"""
import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from tqdm.auto import tqdm

from . import split as split_stage
from .config import CLASSES, LABEL_MAP, Config, get_device, seed_everything
from .data import build_eval_transform, build_train_transform, make_loader, preprocessing_spec
from .ingest import load_index

log = logging.getLogger(__name__)
METRIC_KEYS = ["accuracy", "balanced_accuracy", "malignant_recall", "benign_recall",
               "malignant_precision", "malignant_f1", "weighted_f1", "roc_auc"]


# ------------------------------------------------------------------ model and training loop
class ValidationLossEarlyStopping:
    """Patience counts significant improvements; weights are kept at the absolute minimum loss."""

    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta = patience, min_delta
        self.reference_loss = self.best_loss = float("inf")
        self.best_state, self.best_epoch, self.bad_epochs = None, None, 0

    def update(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss is non-finite.")
        if val_loss < self.best_loss:
            self.best_loss, self.best_epoch = val_loss, epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.bad_epochs = val_loss, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self.best_state)


def build_model(cfg: Config, pretrained: Optional[bool] = None) -> nn.Module:
    return timm.create_model(cfg.model_name, pretrained=cfg.pretrained if pretrained is None else pretrained,
                             num_classes=len(CLASSES), drop_path_rate=cfg.hp["drop_path"])


def lr_factor(epochs: int, warmup: int) -> Callable[[int], float]:
    """Linear warm-up, then cosine decay to 0, stepped once per epoch (= CosineAnnealingLR when warmup=0)."""
    def f(e: int) -> float:
        if e < warmup:
            return (e + 1) / (warmup + 1)
        return 0.5 * (1 + math.cos(math.pi * (e - warmup) / max(1, epochs - warmup)))
    return f


def train_model(model: nn.Module, train_loader, val_loader, run_name: str, cfg: Config,
                y_train: np.ndarray, device: torch.device) -> Tuple[nn.Module, Dict]:
    hp = cfg.hp
    model = model.to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([{"params": body, "lr": hp["lr"] * hp["backbone_lr_mult"]},
                                   {"params": head, "lr": hp["lr"]}], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor(hp["epochs"], hp["warmup_epochs"]))
    weight = None
    if hp["class_weights"]:
        counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float64)
        weight = torch.tensor(len(y_train) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=hp["label_smoothing"])
    val_criterion = nn.CrossEntropyLoss()           # plain CE: comparable across configurations
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = ValidationLossEarlyStopping(hp["patience"], cfg.early_stopping_min_delta)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    stopped_early = False

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        tot_loss, tot_correct, n = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"[{run_name}] epoch {epoch}/{hp['epochs']}", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            mix = hp["mixup_alpha"] > 0
            if mix:
                lam = float(np.random.beta(hp["mixup_alpha"], hp["mixup_alpha"]))
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1 - lam) * imgs[perm]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(imgs)
                loss = (lam * criterion(out, labels) + (1 - lam) * criterion(out, labels[perm])
                        if mix else criterion(out, labels))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item() * imgs.size(0)
            tot_correct += (out.argmax(1) == labels).sum().item()
            n += imgs.size(0)

        model.eval()
        v_loss, v_correct, vn, v_y, v_p = 0.0, 0, 0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                v_loss += val_criterion(out, labels).item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                vn += imgs.size(0)
                v_y.append(labels.cpu().numpy())
                v_p.append(torch.softmax(out.double(), 1)[:, 1].cpu().numpy())
        scheduler.step()
        v_y, v_p = np.concatenate(v_y), np.concatenate(v_p)
        v_auc = float(roc_auc_score(v_y, v_p)) if len(np.unique(v_y)) == 2 else float("nan")
        for k, v in (("train_loss", tot_loss / n), ("train_acc", tot_correct / n), ("val_loss", v_loss / vn),
                     ("val_acc", v_correct / vn), ("val_auc", v_auc)):
            history[k].append(v)
        log.info("[%s] epoch %d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_auc=%.4f",
                 run_name, epoch, tot_loss / n, tot_correct / n, v_loss / vn, v_correct / vn, v_auc)
        if stopper.update(v_loss / vn, model, epoch):
            stopped_early = True
            log.info("[%s] early stopping at epoch %d", run_name, epoch)
            break

    stopper.restore(model)
    b = stopper.best_epoch - 1
    summary = {"best_epoch": stopper.best_epoch, "best_val_loss": float(stopper.best_loss),
               "epochs_run": len(history["val_loss"]), "stopped_early": stopped_early,
               "train_val_acc_gap": float(history["train_acc"][b] - history["val_acc"][b]),
               "history": history}
    log.info("[%s] restored epoch %d (val_loss=%.5f)", run_name, stopper.best_epoch, stopper.best_loss)
    return model, summary


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """(y_true, p_malignant): float32 forward pass, softmax in float64 (what the CPU app computes)."""
    model.eval()
    ys, ps = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device)).double()
        ps.append(torch.softmax(logits, dim=1)[:, LABEL_MAP["malignant"]].cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ------------------------------------------------------------------ threshold and metrics
def select_threshold_for_recall(y_true: np.ndarray, p: np.ndarray, target_recall: float) -> Dict:
    """Maximise validation benign recall subject to malignant recall >= target_recall.

    Ties in benign recall go to the higher malignant recall; t is placed midway between the winning
    interval's lowest validation score and the next score below it.
    """
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        raise ValueError("Validation set must contain both classes.")
    scores = np.unique(p)
    pos, neg = p[y_true == 1], p[y_true == 0]
    rec = np.array([(pos >= s).mean() for s in scores])
    spec = np.array([(neg < s).mean() for s in scores])
    ok = rec >= target_recall - 1e-12
    ok &= spec == spec[ok].max()
    ok &= rec == rec[ok].max()
    i = int(np.flatnonzero(ok)[0])
    t = float(scores[i]) if i == 0 else float((scores[i - 1] + scores[i]) / 2)
    pred = (p >= t).astype(int)
    return {"threshold": t, "target_malignant_recall": target_recall,
            "val_malignant_recall": float(recall_score(y_true, pred, pos_label=1)),
            "val_benign_recall": float(recall_score(y_true, pred, pos_label=0)),
            "val_n_malignant": int(len(pos)), "val_n_benign": int((y_true == 0).sum())}


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if both else float("nan"),
        "malignant_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "benign_recall": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "malignant_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "malignant_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)) if both else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y_true)),
    }


def summarise(results: List[Dict], key: str) -> Dict:
    table = pd.DataFrame([r[key] for r in results])
    return {m: {"mean": float(table[m].mean()), "std": float(table[m].std(ddof=1)),
                "per_fold": [float(v) for v in table[m]]} for m in METRIC_KEYS}


# ------------------------------------------------------------------ experiments
def run_experiment(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: Optional[pd.DataFrame],
                   run_name: str, seed: int, device: torch.device) -> Dict:
    seed_everything(seed)
    pp = preprocessing_spec(cfg.image_size)
    eval_tf = build_eval_transform(pp)

    def loader(frame, train=False):
        tf = build_train_transform(pp, cfg.hp["augment"]) if train else eval_tf
        return make_loader(frame, tf, train, seed, cfg.batch_size, cfg.num_workers, device)

    model, summary = train_model(build_model(cfg), loader(train_df, True), loader(val_df), run_name, cfg,
                                 train_df["y"].to_numpy(), device)
    y_val, p_val = predict_proba(model, loader(val_df), device)
    thr = select_threshold_for_recall(y_val, p_val, cfg.target_malignant_recall)
    out = {"run": run_name, "hp": dict(cfg.hp), "training": summary, "threshold_selection": thr,
           "y_val": y_val, "p_val": p_val,
           "val_log_loss": float(log_loss(y_val, np.clip(p_val, 1e-7, 1 - 1e-7), labels=[0, 1])),
           "val_at_selected": compute_metrics(y_val, p_val, thr["threshold"]), "model": model}
    if test_df is not None:
        y_te, p_te = predict_proba(model, loader(test_df), device)
        out.update({"y_test": y_te, "p_test": p_te,
                    "test_at_selected": compute_metrics(y_te, p_te, thr["threshold"]),
                    f"test_at_{cfg.reference_threshold}": compute_metrics(y_te, p_te, cfg.reference_threshold)})
        s = out["test_at_selected"]
        log.info("[%s] t=%.4f | test: malignant recall %.3f, benign recall %.3f, balanced acc %.3f, AUC %.4f",
                 run_name, thr["threshold"], s["malignant_recall"], s["benign_recall"],
                 s["balanced_accuracy"], s["roc_auc"])
    return out


def shift_eval(cfg: Config, model: nn.Module, test_patients: List[str], threshold: float,
               index: pd.DataFrame, seed: int, device: torch.device) -> Dict:
    """Score the fold's own (unseen) test patients at the other magnifications."""
    eval_tf = build_eval_transform(preprocessing_spec(cfg.image_size))
    out = {}
    for mag in cfg.shift_magnifications:
        sub = index[(index["magnification"] == mag) & index["patient_id"].isin(test_patients)]
        if sub.empty or sub["y"].nunique() < 2:
            continue
        y, p = predict_proba(model, make_loader(sub.reset_index(drop=True), eval_tf, False, seed,
                                                cfg.batch_size, cfg.num_workers, device), device)
        out[mag] = {"y": y, "p": p, "metrics": compute_metrics(y, p, threshold)}
    return out


def _run_dir(cfg: Config) -> Path:
    d = cfg.work / "runs" / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _release(res: Dict) -> None:
    res.pop("model", None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_cv(cfg: Config, splits: Dict, device: torch.device) -> List[Dict]:
    index = load_index(cfg) if cfg.run_shift_eval else None
    results = []
    for f in splits["folds"]:
        path = _run_dir(cfg) / f"fold{f['fold'] + 1}.pkl"
        if path.is_file():
            with open(path, "rb") as fh:
                saved = pickle.load(fh)
            if saved["hp"] == cfg.hp and (not cfg.run_shift_eval or "shift" in saved):
                log.info("fold %d loaded from %s", f["fold"] + 1, path)
                results.append(saved)
                continue
        t0 = time.time()
        res = run_experiment(cfg, f["train"], f["val"], f["test"], f"fold{f['fold'] + 1}",
                             cfg.seed + f["fold"], device)
        res["fold"] = f["fold"]
        res["test_patients"] = sorted(f["test"]["patient_id"].unique().tolist())
        if cfg.run_shift_eval:
            res["shift"] = shift_eval(cfg, res["model"], res["test_patients"],
                                      res["threshold_selection"]["threshold"], index, cfg.seed, device)
        res["minutes"] = (time.time() - t0) / 60
        _release(res)
        with open(path, "wb") as fh:
            pickle.dump(res, fh)
        results.append(res)
    return results


def run_baseline(cfg: Config, splits: Dict, device: torch.device) -> Dict:
    path = _run_dir(cfg) / "image_level_baseline.pkl"
    if path.is_file():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    tr, va, te = splits["image_level"]
    res = run_experiment(cfg, tr, va, te, "image-level baseline", cfg.seed, device)
    res["leakage"] = split_stage.leakage_audit(tr, te)
    _release(res)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return res


def run_final(cfg: Config, splits: Dict, device: torch.device) -> Dict:
    """Train the deployed model; its weights go to work/runs/<run>/final.safetensors for the export stage."""
    d = _run_dir(cfg)
    if (d / "final.pkl").is_file() and (d / "final.safetensors").is_file():
        with open(d / "final.pkl", "rb") as fh:
            saved = pickle.load(fh)
        if saved["hp"] == cfg.hp:
            log.info("final model loaded from %s", d)
            return saved
    tr, va = splits["deployment"]
    res = run_experiment(cfg, tr, va, None, "deployment", cfg.seed, device)
    state = {k: v.detach().cpu().contiguous() for k, v in res["model"].state_dict().items()}
    save_file(state, str(d / "final.safetensors"), metadata={"architecture": cfg.model_name, "format": "pt"})
    _release(res)
    with open(d / "final.pkl", "wb") as fh:
        pickle.dump(res, fh)
    return res


def load_final_model(cfg: Config) -> nn.Module:
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(load_file(str(_run_dir(cfg) / "final.safetensors")), strict=True)
    return model.eval()


def run(cfg: Config) -> Dict:
    device = get_device()
    if device.type != "cuda":
        log.warning("No GPU found: training on CPU will be very slow.")
    torch.backends.cudnn.benchmark = True
    splits = split_stage.load(cfg)
    cv = run_cv(cfg, splits, device)
    baseline = run_baseline(cfg, splits, device) if cfg.run_leakage_baseline else None
    final = run_final(cfg, splits, device)
    sel = summarise(cv, "test_at_selected")
    log.info("Patient-level CV: balanced accuracy %.3f ± %.3f, ROC-AUC %.3f ± %.3f",
             sel["balanced_accuracy"]["mean"], sel["balanced_accuracy"]["std"],
             sel["roc_auc"]["mean"], sel["roc_auc"]["std"])
    return {"cv": cv, "baseline": baseline, "final": final, "splits": splits}
