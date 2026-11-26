"""Runtime settings, read from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _optional(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    # Folder holding model_config.json, metrics.json, dataset_stats.json and the figures.
    artifact_dir: Path
    # If the weights (or any artifact) are missing locally, download them from this HF model repo.
    hf_model_repo: Optional[str]
    hf_model_revision: str
    # Read from the HF_TOKEN secret; only needed when the model repo is private.
    hf_token: Optional[str]
    max_upload_mb: float
    torch_num_threads: Optional[int]

    @classmethod
    def from_env(cls) -> "Settings":
        threads = _optional("TORCH_NUM_THREADS")
        return cls(
            artifact_dir=Path(os.environ.get("ARTIFACT_DIR", PROJECT_ROOT / "deployment_artifacts")),
            hf_model_repo=_optional("HF_MODEL_REPO"),
            hf_model_revision=os.environ.get("HF_MODEL_REVISION", "main"),
            hf_token=_optional("HF_TOKEN"),
            max_upload_mb=float(os.environ.get("MAX_UPLOAD_MB", "10")),
            torch_num_threads=int(threads) if threads else None,
        )
