"""Publish the FastAPI + PyTorch version to a Hugging Face *Docker* Space (needs a PRO account).

Hugging Face builds the image from the Dockerfile: the Next.js site is built in server mode
(NEXT_PUBLIC_INFERENCE=server), so the Analyze page sends images to POST /api/predict.

Prerequisites: `hf auth login` with a write token of a PRO account, and deployment_artifacts/ with the
safetensors weights. Run with classic LFS uploads if Xet stalls on your network:
    set HF_HUB_DISABLE_XET=1
    python scripts/deploy_docker_space.py --space USER/breast-histopathology-api
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

CARD = """---
title: Breast Histopathology Classifier API
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI + PyTorch Swin Tiny with Grad-CAM on BreakHis
---

# Breast Histopathology Classifier (FastAPI)

Swin Tiny (`swin_tiny_patch4_window7_224`) fine-tuned on BreakHis 100X H&E images to classify benign
vs malignant tissue, with Grad-CAM heatmaps. Evaluated with patient-level 5-fold cross-validation.

This Space runs the **FastAPI + PyTorch** server:

* the web app at `/` (Analyze, Pipeline and Data pages), whose predictions go to the server;
* interactive API docs at [`/docs`](./docs);
* `POST /api/predict?explain=true`, which takes an image file and returns the probability, the label,
  and the Grad-CAM heatmap;
* `GET /api/info` for the model card and metrics, and `GET /health`.

Uploaded images are processed in memory and are not stored.

For research and education only. Not intended for clinical diagnosis.
Data: [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).
"""

ALLOW = [
    "Dockerfile", ".dockerignore", "requirements.txt",
    "app/*.py",
    "deployment_artifacts/*.json", "deployment_artifacts/*.png", "deployment_artifacts/*.safetensors",
    "web/package.json", "web/package-lock.json", "web/next.config.ts", "web/tsconfig.json",
    "web/eslint.config.mjs", "web/next-env.d.ts", "web/.gitignore",
    "web/scripts/**", "web/src/**", "web/public/**",
]
IGNORE = ["**/__pycache__/**", "*.pyc", "web/public/model/*.onnx", "web/public/ort/**"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="USER/space-name")
    args = ap.parse_args()
    if not any((ROOT / "deployment_artifacts").glob("*.safetensors")):
        raise SystemExit("deployment_artifacts/ has no .safetensors weights.")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.space,
                    repo_type="space", commit_message="Space card")
    api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space",
                      allow_patterns=ALLOW, ignore_patterns=IGNORE, commit_message="Deploy FastAPI app")
    files = api.list_repo_files(args.space, repo_type="space")
    print(len(files), "files in the Space")
    user, name = args.space.split("/")
    print(f"https://huggingface.co/spaces/{args.space}")
    print(f"https://{user}-{name}.hf.space".lower().replace("_", "-"))


if __name__ == "__main__":
    main()
