"""Unit tests for the training pipeline's data handling (no GPU or dataset needed)."""
import math

import numpy as np
import pandas as pd
import pytest

from pipeline.config import BASE_HP, Config
from pipeline.ingest import IntegrityError, check_integrity, parse_breakhis_filename
from pipeline.split import cv_folds, deployment_split, image_level_split, leakage_audit
from pipeline.train import compute_metrics, lr_factor, select_threshold_for_recall


def fake_index(n_benign=12, n_malignant=20, per_patient=6, seed=0) -> pd.DataFrame:
    rows = []
    for cls, letter, n in [("benign", "B", n_benign), ("malignant", "M", n_malignant)]:
        for p in range(n):
            for k in range(per_patient):
                name = f"SOB_{letter}_A-14-{p}{letter}-100-{k + 1:03d}.png"
                rows.append({"path": name, "filename": name, "label": cls, "magnification": "100X",
                             **parse_breakhis_filename(name)})
    df = pd.DataFrame(rows)
    df["y"] = (df["label"] == "malignant").astype(int)
    return df


def test_parse_filename():
    r = parse_breakhis_filename("SOB_B_A-14-22549AB-100-001.png")
    assert r == {"file_class": "benign", "subtype": "A", "patient_id": "14-22549AB", "file_magnification": "100X"}
    assert parse_breakhis_filename("SOB_M_DC-14-2523-400-024.png")["patient_id"] == "14-2523"
    assert parse_breakhis_filename("random.png") is None


def test_integrity_rejects_patient_in_both_classes():
    df = fake_index()
    df.loc[0, ["label", "file_class"]] = "malignant"           # same patient now in both classes
    with pytest.raises(IntegrityError):
        check_integrity(df)


def test_integrity_rejects_folder_mismatch():
    df = fake_index()
    df.loc[0, "label"] = "malignant"                           # folder says malignant, name says benign
    with pytest.raises(IntegrityError):
        check_integrity(df)


def test_cv_folds_are_patient_disjoint_and_cover_every_image():
    df = fake_index()
    folds = cv_folds(df, Config())
    assert len(folds) == 5
    tests = pd.concat([f["test"] for f in folds])
    assert sorted(tests["filename"]) == sorted(df["filename"])     # each image in exactly one test fold
    for f in folds:
        tr, va, te = (set(f[s]["patient_id"]) for s in ("train", "val", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert f["val"].groupby("y")["patient_id"].nunique().min() >= 2


def test_deployment_holdout_is_patient_disjoint():
    tr, va = deployment_split(fake_index(), Config())
    assert not set(tr["patient_id"]) & set(va["patient_id"])


def test_leakage_audit_detects_image_level_leak():
    df = fake_index()
    tr, _, te = image_level_split(df, Config())
    audit = leakage_audit(tr, te)
    assert audit["test_patients_also_in_train"] > 0             # the random split leaks patients
    for f in cv_folds(df, Config()):
        assert leakage_audit(f["train"], f["test"])["test_patients_also_in_train"] == 0


def test_threshold_meets_recall_target():
    rng = np.random.RandomState(0)
    y = np.r_[np.zeros(40), np.ones(60)].astype(int)
    p = np.clip(np.r_[rng.beta(2, 5, 40), rng.beta(5, 2, 60)], 0, 1)
    sel = select_threshold_for_recall(y, p, 0.95)
    m = compute_metrics(y, p, sel["threshold"])
    assert m["malignant_recall"] >= 0.95
    assert sel["val_malignant_recall"] == pytest.approx(m["malignant_recall"])


def test_lr_schedule_matches_cosine_annealing():
    f = lr_factor(10, 0)
    for e in range(10):
        assert f(e) == pytest.approx(0.5 * (1 + math.cos(math.pi * e / 10)))
    w = lr_factor(15, 1)
    assert w(0) == pytest.approx(0.5) and w(1) == pytest.approx(1.0)


def test_config_rejects_unknown_hyperparameter(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hp": {"learning_rate": 0.1}}')
    with pytest.raises(ValueError):
        Config.load(str(bad))
    good = tmp_path / "good.json"
    good.write_text('{"hp": {"label_smoothing": 0.1}}')
    cfg = Config.load(str(good))
    assert cfg.hp["label_smoothing"] == 0.1 and cfg.hp["lr"] == BASE_HP["lr"]
