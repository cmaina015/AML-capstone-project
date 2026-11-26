import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import load_file

from app.inference import decode_upload, make_swin_reshape_transform
from conftest import ARTIFACTS, he_like_image, requires_weights, to_bytes


def post(client, img_bytes, explain=True, name="x.png", ctype="image/png"):
    return client.post(f"/api/predict?explain={str(explain).lower()}",
                       files={"file": (name, img_bytes, ctype)})


def png_size(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


# ---------------------------------------------------------------- basic routes
def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    from app.main import WEB_DIR
    if (WEB_DIR / "index.html").is_file():          # Next.js static export present
        assert "Breast Histopathology" in r.text and "/_next/" in r.text
    else:
        assert "Frontend not built" in r.text


def test_info_shows_patient_level_headline(client):
    info = client.get("/api/info").json()
    head = info["performance"]["headline"]
    assert head["label"] == "Patient-level cross-validation"
    assert head["n_folds"] == 5
    assert 0 < head["at_selected_threshold"]["roc_auc"]["mean"] < 1
    assert info["performance"]["leakage_prone_baseline"]["label"] == "leakage-prone baseline"
    text = " ".join(info["limitations"])
    assert "7x7" in text and "early stopping and threshold selection" in text
    assert info["dataset"]["n_patients"] == 81
    folds = info["performance"]["headline"]["folds"]
    assert len(folds) == 5 and sum(len(f["test_patients"]) for f in folds) == 81
    assert info["limits"]["max_upload_mb"] == 10


def test_figures_allow_list(client):
    assert client.get("/api/figures/class_distribution.png").status_code == 200
    assert client.get("/api/figures/model_config.json").status_code == 404
    assert client.get("/api/figures/..%2Fmodel_config.json").status_code == 404


# ---------------------------------------------------------------- prediction
def test_predict_with_explanation(client):
    r = post(client, to_bytes(he_like_image(seed=1)))
    assert r.status_code == 200, r.text
    d = r.json()
    p, thr = d["p_malignant"], d["threshold"]
    assert 0.0 <= p <= 1.0
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-9
    assert d["label"] == ("malignant" if p >= thr else "benign")
    assert set(d["images"]) == {"model_input", "overlay", "heatmap"}
    assert all(png_size(v) == (224, 224) for v in d["images"].values())
    e = d["explanation"]
    assert e["target_class"] == d["label"] and e["grid_hw"] == [7, 7]
    assert np.array(e["grid"]).shape == (7, 7)
    assert d["warnings"] == [] or all("borderline" in w for w in d["warnings"])


def test_predict_without_explanation(client):
    d = post(client, to_bytes(he_like_image(seed=1)), explain=False).json()
    assert d["explanation"] is None
    assert set(d["images"]) == {"model_input"}
    assert d["timing_ms"]["explain"] is None


def test_explain_flag_does_not_change_prediction(client):
    img = to_bytes(he_like_image(seed=2))
    a, b = post(client, img, explain=True).json(), post(client, img, explain=False).json()
    assert a["p_malignant"] == b["p_malignant"] and a["label"] == b["label"]


@requires_weights
def test_preprocessing_matches_training_pipeline(client):
    """App prediction == an independently written copy of the notebook's eval pipeline."""
    cfg = json.loads((ARTIFACTS / "model_config.json").read_text())
    reference = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = timm.create_model(cfg["model"]["architecture"], pretrained=False, num_classes=2)
    model.load_state_dict(load_file(str(ARTIFACTS / cfg["model"]["weights_file"])), strict=True)
    model.eval()
    for seed, size in [(3, (224, 224)), (4, (700, 460))]:
        img = he_like_image(size=size, seed=seed)
        with torch.no_grad():
            p_ref = float(torch.softmax(model(reference(img).unsqueeze(0)).double(), 1)[0, 1])
        p_app = post(client, to_bytes(img), explain=False).json()["p_malignant"]
        assert abs(p_ref - p_app) < 1e-6, (size, p_ref, p_app)


# ---------------------------------------------------------------- input validation
def test_non_square_upload_warns(client):
    d = post(client, to_bytes(he_like_image(size=(700, 460), seed=5)), explain=False).json()
    assert any("squashed" in w for w in d["warnings"])
    assert d["input"]["width"] == 700 and d["input"]["height"] == 460


def test_grayscale_upload_warns(client):
    d = post(client, to_bytes(he_like_image(seed=6, mode="L")), explain=False).json()
    assert any("not in colour" in w for w in d["warnings"])


def test_jpeg_and_tiff_accepted(client):
    for fmt, ctype in [("JPEG", "image/jpeg"), ("TIFF", "image/tiff")]:
        r = post(client, to_bytes(he_like_image(seed=7), fmt), explain=False, name=f"x.{fmt.lower()}", ctype=ctype)
        assert r.status_code == 200, (fmt, r.text)


@pytest.mark.parametrize("payload,status", [
    (b"", 400),
    (b"this is not an image", 415),
])
def test_rejects_bad_files(client, payload, status):
    assert post(client, payload).status_code == status


def test_rejects_unsupported_format(client):
    r = post(client, to_bytes(he_like_image(seed=8), "GIF"), name="x.gif", ctype="image/gif")
    assert r.status_code == 415 and "Unsupported image format" in r.json()["detail"]


def test_rejects_oversized_upload(client):
    big = b"\x00" * (int(client.app.state.settings.max_upload_mb * 1024 * 1024) + 10)
    assert post(client, big).status_code == 413


# ---------------------------------------------------------------- Grad-CAM internals
def test_reshape_transform_layouts():
    rt = make_swin_reshape_transform(768)
    bhwc = torch.randn(2, 7, 7, 768)
    expected = bhwc.permute(0, 3, 1, 2)
    assert torch.equal(rt(bhwc), expected)
    assert torch.equal(rt(bhwc.reshape(2, 49, 768)), expected)
    assert torch.equal(rt(expected), expected)
    with pytest.raises(ValueError):
        rt(torch.randn(2, 50, 768))
    with pytest.raises(ValueError):
        rt(torch.randn(2, 7, 7, 512))


def test_concurrent_gradcam_matches_sequential(client):
    """Hooks on the shared model must not leak between simultaneous requests."""
    predictor = client.app.state.predictor
    decoded = [decode_upload(to_bytes(he_like_image(seed=s)), predictor.size_hw) for s in (10, 11, 12)]
    sequential = [predictor.predict(d, explain=True) for d in decoded]
    jobs = decoded * 3
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent = list(pool.map(lambda d: predictor.predict(d, explain=True), jobs))
    for i, res in enumerate(concurrent):
        ref = sequential[i % 3]
        assert res["p_malignant"] == ref["p_malignant"]
        np.testing.assert_allclose(res["explanation"]["grid"], ref["explanation"]["grid"], atol=1e-6)
