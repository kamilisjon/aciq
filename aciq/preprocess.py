"""ImageNet preprocessing: PIL image files → tinygrad `Tensor` batch.

Mirrors `torchvision.transforms._presets.ImageClassification(crop_size=224)`:
BILINEAR resize with short-edge == 256 (antialiased), center crop to 224, uint8 → [0,1]
float32, HWC → CHW transpose, ImageNet mean/std normalisation, then stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tinygrad import Tensor


_RESIZE_SIZE = 256
_CROP_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_one(img: Image.Image) -> np.ndarray:
  w, h = img.size
  if w <= h:
    new_w, new_h = _RESIZE_SIZE, int(_RESIZE_SIZE * h / w)
  else:
    new_h, new_w = _RESIZE_SIZE, int(_RESIZE_SIZE * w / h)
  img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

  left = (new_w - _CROP_SIZE) // 2
  top = (new_h - _CROP_SIZE) // 2
  img = img.crop((left, top, left + _CROP_SIZE, top + _CROP_SIZE))

  arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
  arr = arr.transpose(2, 0, 1)  # CHW
  return (arr - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]


def load_and_preprocess(image_paths: list[Path]) -> Tensor:
  batch = np.stack([_preprocess_one(Image.open(p).convert("RGB")) for p in image_paths])
  return Tensor(batch)
