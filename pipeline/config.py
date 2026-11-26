"""Run configuration: defaults reproduce the deployed model; a JSON file and CLI flags override them."""
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

CLASSES = ["benign", "malignant"]
LABEL_MAP = {"benign": 0, "malignant": 1}          # index 1 = positive class
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]
BREAKHIS_URL = "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
WEIGHTS_FILENAME = "swin_tiny_breakhis.safetensors"

# Training hyperparameters. These defaults are the settings of the deployed model.
BASE_HP: Dict = {
    "lr": 1e-4,                 # head learning rate (and backbone, times backbone_lr_mult)
    "backbone_lr_mult": 1.0,
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
    "drop_path": 0.1,           # timm's default stochastic depth for Swin Tiny
    "augment": "base",          # "base" or "strong" (see pipeline/data.py)
    "class_weights": False,     # inverse-frequency weights in the training loss
    "mixup_alpha": 0.0,         # 0 = off
    "epochs": 10,
    "warmup_epochs": 0,
    "patience": 3,
}


@dataclass
class Config:
    data_dir: str = ""                     # BreakHis folder (with 40X/100X/200X/400X) or the .zip
    out_dir: str = "artifacts"             # exported model + JSON + figures (same layout as deployment_artifacts/)
    work_dir: str = "work"                 # index, splits and per-fold results (resumable)
    magnification: str = "100X"
    shift_magnifications: List[str] = field(default_factory=lambda: ["40X", "200X", "400X"])
    model_name: str = "swin_tiny_patch4_window7_224"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    n_folds: int = 5
    inner_val_fraction: float = 0.15
    min_val_patients_per_class: int = 2
    target_malignant_recall: float = 0.95
    reference_threshold: float = 0.5
    early_stopping_min_delta: float = 1e-4
    run_leakage_baseline: bool = True      # also train on the leaky image-level split, for comparison
    run_shift_eval: bool = True            # score each CV fold's test patients at other magnifications
    pretrained: bool = True
    splits_csv: Optional[str] = None       # reuse an existing split_assignments.csv (identical folds)
    run_name: str = "baseline"
    hp: Dict = field(default_factory=lambda: dict(BASE_HP))

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides) -> "Config":
        data = {}
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        hp = {**BASE_HP, **data.pop("hp", {})}
        unknown = set(hp) - set(BASE_HP)
        if unknown:
            raise ValueError(f"Unknown hyperparameters in {path}: {sorted(unknown)}")
        cfg = cls(**{**data, "hp": hp})
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def work(self) -> Path:
        p = Path(self.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
