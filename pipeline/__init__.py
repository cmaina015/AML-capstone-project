"""Scripted, reproducible training pipeline: ingest -> split -> train (CV + final model) -> export -> predict.

Run everything with `python -m pipeline run --data-dir <BreakHis folder or zip>`; see README.md.
The code is a script version of swin_tiny_train_for_deployment.ipynb: same parsing, splits, training,
threshold rule and artifact format, so its output can replace deployment_artifacts/ directly.
"""
