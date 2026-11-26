"""Export the deployed Swin Tiny to ONNX for in-browser inference, with Grad-CAM built into the graph.

Browsers (onnxruntime-web) cannot back-propagate, so the Grad-CAM gradient is written out in closed
form. The CAM layer is layers[-1].blocks[-1] (output A, shape 1x7x7x768); after it the network is only

    logit_c = fc_w[c] . mean_hw(LayerNorm(A)) + fc_b[c]

so dlogit_c/dA is the LayerNorm backward of (fc_w[c] / 49), which is a few tensor ops. The graph outputs

    logits : (1, 2)      same as model(x)
    cams   : (2, 7, 7)   ReLU(sum_k w_k A_k) for each class, w_k = mean_hw(dlogit_c/dA_k)
                          i.e. Grad-CAM *before* pytorch-grad-cam's min-max scaling and resize, which the
                          browser does (web/src/lib/inference.ts).

Every output is checked against the PyTorch model and pytorch-grad-cam before the file is written.

Usage:  python scripts/export_onnx.py  [--out web/public/model/swin_tiny_breakhis.onnx]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.artifacts import load_artifacts  # noqa: E402
from app.inference import Predictor  # noqa: E402
from app.settings import Settings  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402


class SwinWithCam(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model
        n_tokens = 7 * 7
        ln = model.norm
        # dlogit_c / d(LayerNorm output) for every token, times the LayerNorm gain: (C, 768)
        self.register_buffer("g_xhat", (model.head.fc.weight * ln.weight) / n_tokens)
        self.eps = ln.eps

    def forward(self, x):
        m = self.m
        a = m.layers(m.patch_embed(x))                     # (1, 7, 7, 768) = blocks[-1] output
        logits = m.forward_head(m.norm(a))                 # exactly the library head

        mu = a.mean(-1, keepdim=True)
        sigma = torch.sqrt(((a - mu) ** 2).mean(-1, keepdim=True) + self.eps)
        xhat = (a - mu) / sigma                            # (1, 7, 7, 768)
        g = self.g_xhat[:, None, None, :]                  # (C, 1, 1, 768)
        grad = (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True)) / sigma  # (C,7,7,768)
        weights = grad.mean(dim=(1, 2))                    # (C, 768)
        cams = torch.relu((a * weights[:, None, None, :]).sum(-1))  # (C, 7, 7)
        return logits, cams


def scale(cam):  # pytorch_grad_cam.utils.image.scale_cam_image without resize
    cam = cam - cam.min()
    return cam / (1e-7 + cam.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "web/public/model/swin_tiny_breakhis.onnx"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    art = load_artifacts(Settings.from_env())
    pred = Predictor(art.config, art.weights_path)
    model = pred.model.eval()
    wrapped = SwinWithCam(model).eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(wrapped, (x,), str(out), input_names=["input"], output_names=["logits", "cams"],
                      opset_version=17, dynamo=False, do_constant_folding=True)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    import cv2
    worst = {"logit": 0.0, "prob": 0.0, "cam_raw": 0.0, "cam_224": 0.0}
    rng = np.random.default_rng(0)
    # random tensors plus real-looking colour images run through the real preprocessing
    inputs = [torch.randn(1, 3, 224, 224) for _ in range(4)]
    from PIL import Image
    for _ in range(4):
        img = Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)).resize((460, 700))
        inputs.append(pred.transform(img).unsqueeze(0))
    for xi in inputs:
        with torch.no_grad():
            ref_logits = model(xi)
            w_logits, w_cams = wrapped(xi)
        o_logits, o_cams = sess.run(None, {"input": xi.numpy()})
        ref_p = torch.softmax(ref_logits.double(), 1).numpy()
        o_p = torch.softmax(torch.from_numpy(o_logits).double(), 1).numpy()
        worst["logit"] = max(worst["logit"], float(np.abs(o_logits - ref_logits.numpy()).max()))
        worst["prob"] = max(worst["prob"], float(np.abs(o_p - ref_p).max()))
        for c in (0, 1):
            with GradCAM(model=model, target_layers=[pred.target_layer],
                         reshape_transform=pred.reshape_transform) as gc:
                ref_cam = gc(input_tensor=xi, targets=[ClassifierOutputTarget(c)])[0]   # (224, 224)
            mine = scale(cv2.resize(np.float32(scale(o_cams[c])), (224, 224)))
            mine = scale(np.maximum(mine, 0))
            worst["cam_224"] = max(worst["cam_224"], float(np.abs(mine - ref_cam).max()))
            worst["cam_raw"] = max(worst["cam_raw"], float(np.abs(o_cams[c] - w_cams[c].numpy()).max()))
    print("max abs differences vs PyTorch / pytorch-grad-cam:", json.dumps(worst))
    assert worst["prob"] < 1e-4, worst
    assert worst["cam_224"] < 1e-3, worst

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"file": out.name, "sha256": sha, "bytes": out.stat().st_size,
            "source_weights_sha256": art.config["model"]["weights_sha256"],
            "opset": 17, "outputs": {"logits": [1, 2], "cams": [2, 7, 7]},
            "parity": worst}
    (out.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) sha256={sha[:12]}…")


if __name__ == "__main__":
    main()
