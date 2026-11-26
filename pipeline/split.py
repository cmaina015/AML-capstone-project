"""Stage 2, split: patient-level cross-validation folds, the deployment holdout, and the leakage audit.

Every split groups images by patient, so no patient is in more than one of train / validation / test.
The image-level split of the original study is rebuilt only to measure how much it leaks.
Output: work/splits.csv (same columns as deployment_artifacts/split_assignments.csv) and
work/leakage_audit.json.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import CLASSES, Config
from .ingest import load_index

log = logging.getLogger(__name__)


def stratified_group_holdout(frame: pd.DataFrame, fraction: float, seed: int,
                             min_per_class: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-grouped, class-stratified holdout: the first fold with enough patients of each class."""
    n_splits = max(2, int(round(1 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for keep_idx, hold_idx in sgkf.split(frame, frame["y"], frame["patient_id"]):
        hold = frame.iloc[hold_idx]
        per_class = hold.groupby("y")["patient_id"].nunique()
        if len(per_class) == 2 and per_class.min() >= min_per_class:
            return frame.iloc[keep_idx].reset_index(drop=True), hold.reset_index(drop=True)
    raise RuntimeError("No holdout fold had enough patients of both classes.")


def assert_disjoint_patients(**parts: pd.DataFrame) -> None:
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(parts[names[i]]["patient_id"]) & set(parts[names[j]]["patient_id"])
            if shared:
                raise AssertionError(f"Patients shared by {names[i]} and {names[j]}: {sorted(shared)}")


def cv_folds(df: pd.DataFrame, cfg: Config) -> List[Dict]:
    outer = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    folds = []
    for k, (trv_idx, te_idx) in enumerate(outer.split(df, df["y"], df["patient_id"])):
        trv = df.iloc[trv_idx].reset_index(drop=True)
        te = df.iloc[te_idx].reset_index(drop=True)
        tr, va = stratified_group_holdout(trv, cfg.inner_val_fraction, cfg.seed + k, cfg.min_val_patients_per_class)
        assert_disjoint_patients(train=tr, val=va, test=te)
        folds.append({"fold": k, "train": tr, "val": va, "test": te})
    return folds


def deployment_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, va = stratified_group_holdout(df, cfg.inner_val_fraction, cfg.seed, cfg.min_val_patients_per_class)
    assert_disjoint_patients(train=tr, val=va)
    return tr, va


def image_level_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The original study's 70/15/15 split by image (leakage-prone; for comparison only)."""
    trv, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=cfg.seed)
    tr, va = train_test_split(trv, test_size=0.15 / 0.85, stratify=trv["label"], random_state=cfg.seed)
    return tuple(x.reset_index(drop=True) for x in (tr, va, te))


def leakage_audit(tr: pd.DataFrame, te: pd.DataFrame) -> Dict:
    test_patients = set(te["patient_id"])
    leaked = test_patients & set(tr["patient_id"])
    return {"test_patients": len(test_patients), "test_patients_also_in_train": len(leaked),
            "test_images_from_leaked_patients": int(te["patient_id"].isin(tr["patient_id"]).sum()),
            "test_images": int(len(te))}


def _rows(protocol: str, fold: int, split: str, part: pd.DataFrame) -> List[Dict]:
    return [{"protocol": protocol, "fold": fold, "split": split, "filename": f, "patient_id": p, "label": l}
            for f, p, l in zip(part["filename"], part["patient_id"], part["label"])]


SPLIT_SKLEARN_VERSION = "1.6.1"   # version that produced deployment_artifacts/split_assignments.csv


def build(df: pd.DataFrame, cfg: Config) -> Dict:
    import sklearn
    if sklearn.__version__ != SPLIT_SKLEARN_VERSION:
        log.warning("scikit-learn %s: StratifiedGroupKFold shuffles differently from %s, so the folds will not "
                    "match the published ones. Install requirements-train.txt, or pass --splits-csv "
                    "deployment_artifacts/split_assignments.csv.", sklearn.__version__, SPLIT_SKLEARN_VERSION)
    folds = cv_folds(df, cfg)
    dep_tr, dep_va = deployment_split(df, cfg)
    img_tr, img_va, img_te = image_level_split(df, cfg)
    return {"folds": folds, "deployment": (dep_tr, dep_va), "image_level": (img_tr, img_va, img_te)}


def splits_from_csv(df: pd.DataFrame, path: Path) -> Dict:
    """Rebuild every split from a saved split_assignments.csv, so a re-run uses identical patients."""
    rows = pd.read_csv(path)
    by = {}
    known = set(df["filename"])
    for (protocol, fold, split), g in rows.groupby(["protocol", "fold", "split"]):
        missing = set(g["filename"]) - known
        if missing:
            raise ValueError(f"{len(missing)} files in {path} are not in the dataset, e.g. {sorted(missing)[:3]}")
        by[(protocol, int(fold), split)] = df[df["filename"].isin(set(g["filename"]))].reset_index(drop=True)
    folds = [{"fold": k, "train": by[("patient_cv", k, "train")], "val": by[("patient_cv", k, "val")],
              "test": by[("patient_cv", k, "test")]}
             for k in sorted({f for (p, f, _) in by if p == "patient_cv"})]
    out = {"folds": folds, "deployment": (by[("deployment", -1, "train")], by[("deployment", -1, "val")])}
    if ("image_level_baseline", -1, "test") in by:
        out["image_level"] = tuple(by[("image_level_baseline", -1, s)] for s in ("train", "val", "test"))
    return out


def save(splits: Dict, cfg: Config) -> None:
    rows = []
    for f in splits["folds"]:
        for s in ("train", "val", "test"):
            rows += _rows("patient_cv", f["fold"], s, f[s])
    if "image_level" in splits:
        for s, part in zip(("train", "val", "test"), splits["image_level"]):
            rows += _rows("image_level_baseline", -1, s, part)
    for s, part in zip(("train", "val"), splits["deployment"]):
        rows += _rows("deployment", -1, s, part)
    pd.DataFrame(rows).to_csv(cfg.work / "splits.csv", index=False)
    audit = {"image_level_split": leakage_audit(splits["image_level"][0], splits["image_level"][2])
             if "image_level" in splits else None,
             "patient_level_folds": [leakage_audit(f["train"], f["test"]) for f in splits["folds"]]}
    (cfg.work / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def describe(splits: Dict) -> str:
    lines = []
    for f in splits["folds"]:
        counts = {s: {c: f[s].loc[f[s]["label"] == c, "patient_id"].nunique() for c in CLASSES}
                  for s in ("train", "val", "test")}
        lines.append(f"fold {f['fold'] + 1}: patients (benign/malignant) " +
                     ", ".join(f"{s} {v['benign']}/{v['malignant']}" for s, v in counts.items()))
    return "\n".join(lines)


def run(cfg: Config) -> Dict:
    df = load_index(cfg, cfg.magnification)
    splits = splits_from_csv(df, Path(cfg.splits_csv)) if cfg.splits_csv else build(df, cfg)
    if "image_level" not in splits:
        splits["image_level"] = image_level_split(df, cfg)
    all_test = pd.concat([f["test"] for f in splits["folds"]])
    assert set(all_test["filename"]) == set(df["filename"]) and len(all_test) == len(df), \
        "Every image must be in exactly one test fold"
    save(splits, cfg)
    log.info("Splits:\n%s", describe(splits))
    audit = json.loads((cfg.work / "leakage_audit.json").read_text())
    if audit["image_level_split"]:
        a = audit["image_level_split"]
        log.info("Leakage audit: image-level split puts %d of %d test patients in training too; "
                 "patient-level folds: 0", a["test_patients_also_in_train"], a["test_patients"])
    return splits


def load(cfg: Config) -> Dict:
    df = load_index(cfg, cfg.magnification)
    path = cfg.work / "splits.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the split stage first")
    return splits_from_csv(df, path)
