"""Model loading, preprocessing (rebuilt from model_config.json), prediction and Grad-CAM.

The preprocessing and Grad-CAM functions mirror swin_tiny_train_for_deployment.ipynb exactly.
"""
import base64
import io
import math
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Preprocessing (identical to the notebook's build_resize / build_eval_transform)
# ---------------------------------------------------------------------------
_INTERPOLATION = {"bilinear": T.InterpolationMode.BILINEAR,
                  "bicubic": T.InterpolationMode.BICUBIC,
                  "nearest": T.InterpolationMode.NEAREST}


def build_resize(pp: Dict) -> T.Resize:
    r = pp["resize"]
    if r["method"] != "squash" or pp.get("center_crop") is not None:
        raise ValueError(f'Unsupported preprocessing: {r["method"]}, crop={pp.get("center_crop")}')
    return T.Resize(tuple(r["size_hw"]), interpolation=_INTERPOLATION[r["interpolation"]],
                    antialias=r["antialias"])


def build_eval_transform(pp: Dict) -> T.Compose:
    return T.Compose([build_resize(pp), T.ToTensor(),
                      T.Normalize(pp["normalize"]["mean"], pp["normalize"]["std"])])


# ---------------------------------------------------------------------------
# Grad-CAM helpers (identical to the notebook)
# ---------------------------------------------------------------------------
SUPPORTED_TARGET_LAYER = "layers[-1].blocks[-1]"


def get_cam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.layers[-1].blocks[-1]


def make_swin_reshape_transform(num_channels: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Convert a Swin activation/gradient to (B, C, H, W) for Grad-CAM."""
    def reshape_transform(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4 and t.shape[-1] == num_channels:      # (B, H, W, C) - timm >= 0.9 block output
            return t.permute(0, 3, 1, 2).contiguous()
        if t.ndim == 4 and t.shape[1] == num_channels:       # already (B, C, H, W)
            return t
        if t.ndim == 3 and t.shape[-1] == num_channels:      # (B, N, C) - flattened tokens
            b, n, c = t.shape
            side = int(round(math.sqrt(n)))
            if side * side != n:
                raise ValueError(f"Token count {n} is not a square grid.")
            return t.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unexpected activation shape {tuple(t.shape)} for C={num_channels}.")
    return reshape_transform


# ---------------------------------------------------------------------------
# Input decoding and validation
# ---------------------------------------------------------------------------
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF", "BMP"}
MAX_PIXELS = 40_000_000          # reject decompression bombs well before PIL's own limit
ASPECT_TOLERANCE = 0.02
NEAR_THRESHOLD_MARGIN = 0.05


class InputError(ValueError):
    """Raised for uploads that cannot be used. `status` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class DecodedImage:
    image: Image.Image          # RGB
    info: Dict
    warnings: List[str]


def decode_upload(data: bytes, training_size_hw: List[int]) -> DecodedImage:
    if not data:
        raise InputError("The uploaded file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                w, h = probe.size
                if w * h > MAX_PIXELS:
                    raise InputError(f"Image is too large ({w}x{h} pixels).", status=413)
                probe.verify()
            im = Image.open(io.BytesIO(data))
            n_frames = getattr(im, "n_frames", 1)
            im.seek(0)
            im.load()
    except InputError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning,
            OSError, SyntaxError, ValueError) as exc:
        raise InputError(f"Could not read the file as an image ({type(exc).__name__}).", status=415)

    if fmt not in ALLOWED_FORMATS:
        raise InputError(f"Unsupported image format {fmt}. Use PNG, JPEG, TIFF or BMP.", status=415)

    mode = im.mode
    th, tw = training_size_hw
    notes: List[str] = []
    if n_frames > 1:
        notes.append(f"The file has {n_frames} frames; only the first was used.")
    if mode in ("1", "L", "LA", "I", "I;16", "F"):
        notes.append("The image is not in colour. The model was trained on colour H&E images, "
                     "so the result is unreliable.")
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in im.info):
        notes.append("Transparency was removed (alpha channel dropped).")
    aspect = w / h
    if abs(aspect - tw / th) > ASPECT_TOLERANCE:
        notes.append(f"The image is {w}x{h} (aspect ratio {aspect:.2f}). It is squashed to {tw}x{th}, "
                     f"which distorts tissue shapes; the training images were {tw}x{th} squares.")
    elif (w, h) != (tw, th):
        notes.append(f"Resized from {w}x{h} to {tw}x{th}.")

    rgb = im.convert("RGB")
    im.close()
    info = {"format": fmt, "mode": mode, "width": w, "height": h,
            "aspect_ratio": round(aspect, 4), "frames": n_frames}
    return DecodedImage(rgb, info, notes)


def _png_b64(arr_or_img) -> str:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(self, config: Dict, weights_path: Path):
        self.config = config
        m = config["model"]
        self.classes = {int(k): v for k, v in config["classes"].items()}
        self.positive_index = next(k for k, v in self.classes.items() if v == config["positive_class"])
        self.threshold = float(config["decision"]["threshold"])
        self.preprocessing = config["preprocessing"]
        self.size_hw = list(self.preprocessing["resize"]["size_hw"])
        self.resize = build_resize(self.preprocessing)
        self.transform = build_eval_transform(self.preprocessing)

        model = timm.create_model(m["architecture"], pretrained=False, num_classes=m["num_classes"])
        model.load_state_dict(load_file(str(weights_path)), strict=True)
        self.model = model.eval()

        cam_cfg = config["explainability"]
        if cam_cfg["target_layer"] != SUPPORTED_TARGET_LAYER:
            raise ValueError(f"Unsupported Grad-CAM target layer {cam_cfg['target_layer']}")
        self.target_layer = get_cam_target_layer(self.model)
        self.reshape_transform = make_swin_reshape_transform(self.model.num_features)
        self.cam_grid_hw = list(cam_cfg["cam_grid_hw"])
        self.overlay_image_weight = 0.6

        # One lock for every forward pass. Grad-CAM hooks attach to the shared model, so a concurrent
        # forward pass (plain or CAM) would otherwise leak activations into another request's CAM.
        self._lock = threading.Lock()
        self.layer_output_shape = self._check_layer_shape()

    def _check_layer_shape(self) -> List[int]:
        """Confirm the CAM layer still produces the shape recorded at training time."""
        captured = {}
        handle = self.target_layer.register_forward_hook(
            lambda mod, inp, out: captured.__setitem__("shape", list(out.shape)))
        try:
            with torch.no_grad():
                self.model(torch.zeros(1, 3, *self.size_hw))
        finally:
            handle.remove()
        expected = self.config["explainability"]["observed_shapes"]["block output"]
        if captured["shape"] != expected:
            raise RuntimeError(f"Grad-CAM layer output {captured['shape']} != training-time {expected}")
        grid = list(self.reshape_transform(torch.zeros(captured["shape"])).shape[-2:])
        if grid != self.cam_grid_hw:
            raise RuntimeError(f"Grad-CAM grid {grid} != training-time {self.cam_grid_hw}")
        return captured["shape"]

    def predict(self, decoded: DecodedImage, explain: bool) -> Dict:
        t0 = time.perf_counter()
        x = self.transform(decoded.image).unsqueeze(0)
        model_input = self.resize(decoded.image)
        t_pre = time.perf_counter()

        cam = None
        with self._lock:
            with torch.no_grad():
                logits = self.model(x)
            probs = torch.softmax(logits.double(), dim=1)[0]
            p_pos = float(probs[self.positive_index])
            pred_index = self.positive_index if p_pos >= self.threshold else 1 - self.positive_index
            t_pred = time.perf_counter()
            if explain:
                # Gradients are required here: no torch.no_grad(). Fresh GradCAM per call; the
                # context manager removes its hooks before the lock is released.
                with GradCAM(model=self.model, target_layers=[self.target_layer],
                             reshape_transform=self.reshape_transform) as gradcam:
                    cam = gradcam(input_tensor=x, targets=[ClassifierOutputTarget(pred_index)])[0]
        t_cam = time.perf_counter()

        notes = list(decoded.warnings)
        if abs(p_pos - self.threshold) < NEAR_THRESHOLD_MARGIN:
            notes.append(f"The probability is within {NEAR_THRESHOLD_MARGIN} of the decision threshold, "
                         "so this result is borderline.")

        result = {
            "label": self.classes[pred_index],
            "probabilities": {self.classes[i]: float(probs[i]) for i in self.classes},
            "p_malignant": p_pos,
            "threshold": self.threshold,
            "decision_rule": self.config["decision"]["rule"],
            "input": decoded.info,
            "warnings": notes,
            "images": {"model_input": _png_b64(model_input)},
            "explanation": None,
            "timing_ms": {"preprocess": round((t_pre - t0) * 1000, 1),
                          "predict": round((t_pred - t_pre) * 1000, 1),
                          "explain": round((t_cam - t_pred) * 1000, 1) if explain else None,
                          "total": round((t_cam - t0) * 1000, 1)},
        }
        if explain:
            result["explanation"] = self._render_cam(model_input, cam, pred_index)
            result["images"].update(result["explanation"].pop("images"))
        return result

    def _render_cam(self, model_input: Image.Image, cam: np.ndarray, target_index: int) -> Dict:
        rgb = np.asarray(model_input, dtype=np.float32) / 255.0
        cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
        overlay = show_cam_on_image(rgb, cam, use_rgb=True, image_weight=self.overlay_image_weight)
        heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        gh, gw = self.cam_grid_hw
        grid = cv2.resize(cam, (gw, gh), interpolation=cv2.INTER_AREA)
        flat = float(cam.max()) < 1e-6
        return {
            "method": "Grad-CAM",
            "target_class": self.classes[target_index],
            "target_layer": SUPPORTED_TARGET_LAYER,
            "grid_hw": self.cam_grid_hw,
            "grid": np.round(grid, 4).tolist(),
            "flat": flat,
            "note": self.config["explainability"]["note"],
            "images": {"overlay": _png_b64(overlay), "heatmap": _png_b64(heatmap)},
        }
