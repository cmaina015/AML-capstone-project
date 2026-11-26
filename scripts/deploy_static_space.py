"""Publish web/out (the static site + ONNX model) to a free Hugging Face *static* Space.

Prerequisites: `hf auth login` with a write token, then
    python scripts/export_onnx.py && python scripts/export_static.py
    (cd web && npm ci && npm run build)
    python scripts/deploy_static_space.py --space USER/breast-histopathology
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/out"

CARD = """---
title: Breast Histopathology Classifier
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: static
pinned: false
short_description: Swin Tiny + Grad-CAM on BreakHis, runs in your browser
# Cross-origin isolation lets ONNX Runtime Web use several CPU threads (about 2x faster).
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Breast Histopathology Classifier

Swin Tiny (`swin_tiny_patch4_window7_224`) fine-tuned on BreakHis 100X H&E images to classify
benign vs malignant tissue, with Grad-CAM heatmaps. Evaluated with patient-level 5-fold
cross-validation.

The model runs entirely in the visitor's browser (ONNX Runtime Web): uploaded images are not sent
to any server. The browser pipeline was checked against the original PyTorch / pytorch-grad-cam
pipeline (same resized pixels, probabilities within 1e-7).

For research and education only. Not intended for clinical diagnosis.
Data: [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="USER/space-name")
    args = ap.parse_args()
    if not (OUT / "index.html").is_file() or not (OUT / "model").is_dir():
        raise SystemExit("web/out is missing or has no model; build it first (see docstring).")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.space,
                    repo_type="space", commit_message="Space card")
    api.upload_folder(folder_path=str(OUT), repo_id=args.space, repo_type="space",
                      commit_message="Deploy static site", delete_patterns=["_next/**"])
    user, name = args.space.split("/")
    print(f"https://huggingface.co/spaces/{args.space}")
    print(f"https://{user}-{name}.static.hf.space".lower().replace("_", "-"))


if __name__ == "__main__":
    main()
