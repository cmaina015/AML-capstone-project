"""Download the trained weights (110 MB, not stored in git) into deployment_artifacts/.

The file is taken from the public Hugging Face Space that serves the model, and its SHA-256 is
checked against deployment_artifacts/model_config.json, so the API refuses nothing it would not
have produced itself. To create the weights yourself instead, run the pipeline:
    python -m pipeline run --data-dir <BreakHis> --out-dir deployment_artifacts

    python scripts/fetch_weights.py [--repo CMaina015/breast-histopathology-api]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "deployment_artifacts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="CMaina015/breast-histopathology-api", help="Hugging Face Space id")
    args = ap.parse_args()
    cfg = json.loads((ART / "model_config.json").read_text(encoding="utf-8"))
    name, expected = cfg["model"]["weights_file"], cfg["model"]["weights_sha256"]
    target = ART / name
    if target.is_file() and sha256(target) == expected:
        print(f"{target} already present and verified.")
        return
    src = Path(hf_hub_download(repo_id=args.repo, repo_type="space", filename=f"deployment_artifacts/{name}"))
    actual = sha256(src)
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: downloaded {actual}, model_config.json expects {expected}")
    shutil.copyfile(src, target)
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB), SHA-256 verified.")


if __name__ == "__main__":
    main()
