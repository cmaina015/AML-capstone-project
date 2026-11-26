"""Locate the trained artifacts (written by `python -m pipeline export`), downloading from the HF Hub if needed."""
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .settings import Settings

log = logging.getLogger(__name__)

CONFIG_FILE = "model_config.json"
METRICS_FILE = "metrics.json"
STATS_FILE = "dataset_stats.json"
FIGURES = ("class_distribution.png", "cv_confusion_matrix.png", "cv_roc_curve.png", "training_history.png")


@dataclass
class Artifacts:
    config: Dict
    metrics: Dict
    dataset_stats: Dict
    weights_path: Path
    figures: Dict[str, Path]  # only figures that are actually available


def _resolve(name: str, settings: Settings, required: bool = True) -> Path | None:
    """Local file in ARTIFACT_DIR, else hf_hub_download from HF_MODEL_REPO (cached under HF_HOME)."""
    local = settings.artifact_dir / name
    if local.is_file():
        return local
    if settings.hf_model_repo:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        log.info("Downloading %s from %s@%s", name, settings.hf_model_repo, settings.hf_model_revision)
        try:
            return Path(hf_hub_download(repo_id=settings.hf_model_repo, filename=name,
                                        revision=settings.hf_model_revision, token=settings.hf_token))
        except EntryNotFoundError:
            if required:
                raise
            return None
    if required:
        raise FileNotFoundError(
            f"{name} not found in {settings.artifact_dir} and HF_MODEL_REPO is not set.")
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifacts(settings: Settings) -> Artifacts:
    config = json.loads(_resolve(CONFIG_FILE, settings).read_text(encoding="utf-8"))
    metrics = json.loads(_resolve(METRICS_FILE, settings).read_text(encoding="utf-8"))
    stats = json.loads(_resolve(STATS_FILE, settings).read_text(encoding="utf-8"))

    weights_path = _resolve(config["model"]["weights_file"], settings)
    expected = config["model"]["weights_sha256"]
    actual = sha256_file(weights_path)
    if actual != expected:
        raise RuntimeError(f"Weights checksum mismatch for {weights_path}: {actual} != {expected}")

    figures = {}
    for name in FIGURES:
        path = _resolve(name, settings, required=False)
        if path is not None:
            figures[name] = path
    return Artifacts(config, metrics, stats, weights_path, figures)
