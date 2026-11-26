"""Stage 1, ingest: locate BreakHis, index every image, parse patient IDs, run integrity checks.

Output: work/index.csv (all magnifications) and work/dataset_stats.json (the training magnification).
Integrity problems that would make a patient-level split invalid stop the pipeline.
"""
import hashlib
import json
import logging
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .config import BREAKHIS_URL, CLASSES, LABEL_MAP, MAGNIFICATIONS, Config

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
BREAKHIS_PATTERN = re.compile(
    r"^SOB_(?P<cls>[BM])_(?P<subtype>[A-Z]+)-(?P<year>\d+)-(?P<slide>[0-9A-Za-z]+)"
    r"-(?P<mag>\d+)-(?P<seq>\d+)\.(?P<ext>[A-Za-z]+)$"
)
SUBTYPE_NAMES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes tumour", "TA": "tubular adenoma",
    "DC": "ductal carcinoma", "LC": "lobular carcinoma", "MC": "mucinous carcinoma",
    "PC": "papillary carcinoma",
}


class IntegrityError(ValueError):
    pass


def parse_breakhis_filename(name: str) -> Optional[Dict[str, str]]:
    """'SOB_B_A-14-22549AB-100-001.png' -> class, subtype, patient_id '14-22549AB', magnification '100X'."""
    m = BREAKHIS_PATTERN.match(name)
    if m is None:
        return None
    g = m.groupdict()
    return {
        "file_class": "benign" if g["cls"] == "B" else "malignant",
        "subtype": g["subtype"],
        "patient_id": f"{g['year']}-{g['slide']}",
        "file_magnification": f"{g['mag']}X",
    }


def find_data_root(search_dir: Path, magnification: str) -> Optional[Path]:
    """The folder that contains <magnification>/benign and <magnification>/malignant."""
    for root, dirs, _ in os.walk(search_dir):
        if magnification in dirs:
            sub = Path(root) / magnification
            names = {d.lower() for d in os.listdir(sub) if (sub / d).is_dir()}
            if all(c in names for c in CLASSES):
                return Path(root)
    return None


def resolve_data_root(data_dir: str, magnification: str, extract_to: Path) -> Path:
    """data_dir may be the extracted dataset or the zip file (extracted once into work/).

    Returns the folder holding <mag>/<class>/ (the layout used for training), or, for the original
    BreakHis download (patient folders), the top folder: index_images then labels by file name.
    """
    p = Path(data_dir)
    if p.is_file() and p.suffix.lower() == ".zip":
        if not extract_to.exists() or not any(extract_to.iterdir()):
            log.info("Extracting %s to %s", p, extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(extract_to)
        p = extract_to
    elif not p.is_dir():
        raise FileNotFoundError(f"{data_dir} is neither a folder nor a .zip file")
    return find_data_root(p, magnification) or p


def _class_dir(mag_dir: Path, class_name: str) -> Optional[Path]:
    for d in mag_dir.iterdir():
        if d.is_dir() and d.name.lower() == class_name:
            return d
    return None


def _find_images(folder: Path) -> List[Path]:
    found = set()
    for ext in IMAGE_EXTENSIONS:
        found.update(folder.rglob(f"*.{ext}"))
        found.update(folder.rglob(f"*.{ext.upper()}"))
    return sorted(found)          # sorted, so the row order (and therefore the folds) is reproducible


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _index_by_filename(data_root: Path) -> List[Dict]:
    """Original BreakHis layout (benign/SOB/<subtype>/<patient>/<mag>/...): class and magnification
    come from the file name, which check_integrity later cross-checks against the folder names."""
    records = []
    for p in _find_images(data_root):
        parsed = parse_breakhis_filename(p.name)
        if parsed is None:
            continue
        folder_cls = next((c for c in CLASSES if c in (s.lower() for s in p.parts)), parsed["file_class"])
        records.append({"path": str(p), "filename": p.name, "label": folder_cls,
                        "magnification": parsed["file_magnification"], **parsed})
    return records


def index_images(data_root: Path) -> pd.DataFrame:
    records = []
    if not any((data_root / m).is_dir() for m in MAGNIFICATIONS):
        records = _index_by_filename(data_root)
        if not records:
            raise FileNotFoundError(f"No BreakHis images found under {data_root}")
    for mag in (MAGNIFICATIONS if not records else []):
        mag_dir = data_root / mag
        if not mag_dir.is_dir():
            log.warning("Magnification folder %s not found; skipped", mag_dir)
            continue
        for cls in CLASSES:
            cls_dir = _class_dir(mag_dir, cls)
            if cls_dir is None:
                raise FileNotFoundError(f'No "{cls}" folder under {mag_dir}')
            for p in _find_images(cls_dir):
                rec = {"path": str(p), "filename": p.name, "label": cls, "magnification": mag}
                parsed = parse_breakhis_filename(p.name)
                rec.update(parsed or {"file_class": None, "subtype": None, "patient_id": None,
                                      "file_magnification": None})
                records.append(rec)
    df = pd.DataFrame(records)
    df["y"] = df["label"].map(LABEL_MAP).astype(int)
    return df


def check_integrity(df: pd.DataFrame) -> Dict:
    """Fatal: unparseable names, class letter vs folder mismatch, a patient in both classes."""
    unparsed = df[df["patient_id"].isna()]
    if len(unparsed):
        raise IntegrityError(f"{len(unparsed)} filenames do not follow the BreakHis pattern, e.g. "
                             f"{unparsed['filename'].head(5).tolist()}")
    mismatch = df[df["file_class"] != df["label"]]
    if len(mismatch):
        raise IntegrityError(f"{len(mismatch)} files whose B/M letter disagrees with their folder: "
                             f"{mismatch['filename'].head(5).tolist()}")
    n_cls = df.groupby("patient_id")["label"].nunique()
    both = n_cls[n_cls > 1].index.tolist()
    if both:
        raise IntegrityError(f"Patients found in both classes: {both}")
    subtypes = df.groupby("patient_id")["subtype"].agg(lambda s: sorted(set(s)))
    return {
        "unparsed_filenames": 0,
        "patients_in_both_classes": 0,
        "patients_spanning_subtypes": {k: v for k, v in subtypes.items() if len(v) > 1},
        "files_with_other_magnification_in_name": int((df["file_magnification"] != df["magnification"]).sum()),
    }


def _image_stats(path: str) -> Dict:
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
        arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    flat = arr.reshape(-1, 3)
    out = {"width": w, "height": h, "mode": mode, "n_px": flat.shape[0]}
    for i, c in enumerate("rgb"):
        out[f"sum_{c}"] = flat[:, i].sum()
        out[f"sq_{c}"] = (flat[:, i] ** 2).sum()
    return out


def _pooled_rgb(sub: pd.DataFrame) -> Dict:
    n = sub["n_px"].sum()
    mean = [sub[f"sum_{c}"].sum() / n for c in "rgb"]
    std = [math.sqrt(max(sub[f"sq_{c}"].sum() / n - m ** 2, 0.0)) for c, m in zip("rgb", mean)]
    return {"mean": [round(v, 4) for v in mean], "std": [round(v, 4) for v in std]}


def dataset_statistics(df: pd.DataFrame, magnification: str, integrity: Dict) -> Dict:
    """Display statistics for the app's Data page (not used for normalisation)."""
    stats = pd.DataFrame([_image_stats(p) for p in df["path"]])
    stats["label"] = df["label"].values
    sizes = stats.groupby(["width", "height"]).size().sort_values(ascending=False)
    sub = (df.groupby(["label", "subtype"]).agg(images=("path", "size"), patients=("patient_id", "nunique"))
           .reset_index())
    return {
        "dataset": "BreakHis (Breast Cancer Histopathological Database)",
        "dataset_url": BREAKHIS_URL,
        "magnification": magnification,
        "stain": "H&E",
        "note": "Descriptive statistics for display only. Model input normalisation uses ImageNet "
                "mean/std - see model_config.json -> preprocessing.normalize.",
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "classes": {
            c: {"images": int((df["label"] == c).sum()),
                "patients": int(df.loc[df["label"] == c, "patient_id"].nunique()),
                "image_fraction": round(float((df["label"] == c).mean()), 4)}
            for c in CLASSES
        },
        "subtypes": [{"class": r.label, "code": r.subtype, "name": SUBTYPE_NAMES.get(r.subtype),
                      "images": int(r.images), "patients": int(r.patients)} for r in sub.itertuples()],
        "image_sizes": [{"width": int(w), "height": int(h), "count": int(c)} for (w, h), c in sizes.items()],
        "colour_modes": {k: int(v) for k, v in stats["mode"].value_counts().items()},
        "rgb_0_1": {"all": _pooled_rgb(stats), **{c: _pooled_rgb(stats[stats["label"] == c]) for c in CLASSES}},
        "integrity": integrity,
    }


def run(cfg: Config) -> pd.DataFrame:
    root = resolve_data_root(cfg.data_dir, cfg.magnification, cfg.work / "extracted")
    log.info("Data root: %s", root)
    df_all = index_images(root)
    df_all["md5"] = [file_md5(p) for p in df_all["path"]]
    df_all.to_csv(cfg.work / "index.csv", index=False)

    df = df_all[df_all["magnification"] == cfg.magnification].reset_index(drop=True)
    integrity = check_integrity(df)
    if df["filename"].duplicated().any():
        raise IntegrityError("Duplicate filenames at the training magnification")
    dup = df[df["md5"].duplicated(keep=False)]
    cross = [g for _, g in dup.groupby("md5") if g["patient_id"].nunique() > 1]
    integrity["identical_content_images"] = int(len(dup))
    integrity["identical_content_groups_across_patients"] = len(cross)
    if cross:
        log.warning("%d groups of identical images span more than one patient", len(cross))

    stats = dataset_statistics(df, cfg.magnification, integrity)
    (cfg.work / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Indexed %d images (%d at %s) from %d patients", len(df_all), len(df), cfg.magnification,
             df["patient_id"].nunique())
    return df_all


def load_index(cfg: Config, magnification: Optional[str] = None) -> pd.DataFrame:
    path = cfg.work / "index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the ingest stage first")
    df = pd.read_csv(path)
    if magnification:
        df = df[df["magnification"] == magnification].reset_index(drop=True)
    return df
