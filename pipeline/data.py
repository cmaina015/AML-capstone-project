"""Preprocessing and data loading.

The evaluation transform is the app's own (app/inference.py, rebuilt from the PREPROCESSING dict that
is saved in model_config.json), so training and serving cannot drift apart.
"""
import random
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset

from app.inference import _INTERPOLATION, build_eval_transform, build_resize  # noqa: F401  (re-exported)


def preprocessing_spec(image_size: int) -> Dict:
    """Saved verbatim in model_config.json; the API and the browser rebuild preprocessing from it."""
    return {
        "decode": "PIL.Image.open(file).convert('RGB')  (RGBA/LA/P/L/CMYK -> RGB; alpha is dropped, not composited)",
        "resize": {"method": "squash", "op": "torchvision.transforms.Resize(size=(H, W))",
                   "size_hw": [image_size, image_size], "interpolation": "bilinear", "antialias": True,
                   "preserves_aspect_ratio": False},
        "center_crop": None,
        "to_tensor": "torchvision.transforms.ToTensor(): HWC uint8 [0,255] -> CHW float32 [0,1]",
        "normalize": {"mean": [float(v) for v in IMAGENET_DEFAULT_MEAN], "std": [float(v) for v in IMAGENET_DEFAULT_STD],
                      "source": "ImageNet (timm IMAGENET_DEFAULT_MEAN / IMAGENET_DEFAULT_STD)",
                      "note": "Dataset statistics in dataset_stats.json are for display only and are NOT used here."},
        "channel_order": "RGB",
        "tensor_layout": "NCHW",
        "dtype": "float32",
    }


class RandomRot90:
    """Rotate by 0, 90, 180 or 270 degrees (no interpolation, no black corners)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randrange(4)
        return img if k == 0 else img.transpose(
            [Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270][k - 1])

    def __repr__(self) -> str:
        return "RandomRot90()"


AUGMENTATION_DESCRIPTIONS = {
    "base": ["RandomHorizontalFlip(p=0.5)", "RandomVerticalFlip(p=0.5)", "RandomRotation(20)",
             "ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)"],
    "strong": ["RandomResizedCrop(224x224, scale=(0.6, 1.0), ratio=(1.3, 1.75))",
               "RandomHorizontalFlip(p=0.5)", "RandomVerticalFlip(p=0.5)", "RandomRot90",
               "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04)",
               "GaussianBlur(3, sigma=0.1-1.0) with p=0.2"],
}


def build_train_transform(pp: Dict, augment: str = "base") -> T.Compose:
    norm = [T.ToTensor(), T.Normalize(pp["normalize"]["mean"], pp["normalize"]["std"])]
    if augment == "base":
        return T.Compose([build_resize(pp), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.RandomRotation(20), T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                          *norm])
    if augment == "strong":
        r = pp["resize"]
        return T.Compose([
            T.RandomResizedCrop(tuple(r["size_hw"]), scale=(0.6, 1.0), ratio=(1.3, 1.75),
                                interpolation=_INTERPOLATION[r["interpolation"]], antialias=r["antialias"]),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(), RandomRot90(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
            *norm,
        ])
    raise ValueError(f"Unknown augmentation {augment!r}")


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class HistopathologyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable):
        self.paths = frame["path"].tolist()
        self.labels = frame["y"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.transform(load_rgb(self.paths[idx])), self.labels[idx]




def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(frame: pd.DataFrame, transform: Callable, shuffle: bool, seed: int, batch_size: int,
                num_workers: int, device: torch.device) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(HistopathologyDataset(frame, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(device.type == "cuda"),
                      worker_init_fn=_seed_worker, generator=g)
