from __future__ import annotations

import numpy as np
from PIL import Image
from matplotlib import cm
from tinygrad import Tensor

from aciq.datasets.imagenet import normalize_to_chw, resize_and_center_crop
from aciq.models.resnet import ResNet


# Source: https://ieeexplore.ieee.org/document/10348813


_OVERLAY_ALPHA = 0.45


def _cam_for_class(feat: np.ndarray, fc_weight: np.ndarray, class_idx: int) -> np.ndarray:
  return np.einsum("chw,c->hw", feat, fc_weight[class_idx])


def _predict_with_features(model: ResNet, x: Tensor) -> tuple[np.ndarray, int, float]:
  activations = model.get_activations(x)
  feat = activations["layer4.1.activation_2"].numpy()
  gap = feat.mean(axis=(2, 3))
  fc_w = model.fc.weight.numpy()
  fc_b = model.fc.bias.numpy() if model.fc.bias is not None else np.zeros(fc_w.shape[0], dtype=np.float32)
  logits = gap @ fc_w.T + fc_b
  shifted = logits - logits.max(axis=1, keepdims=True)
  exps = np.exp(shifted)
  probs = exps / exps.sum(axis=1, keepdims=True)
  class_idx = int(np.argmax(probs[0]))
  return feat[0], class_idx, float(probs[0, class_idx])


def _upsample_bilinear(cam: np.ndarray, size: int) -> np.ndarray:
  img = Image.fromarray(cam.astype(np.float32), mode="F")
  return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def compute_cam(model: ResNet, img: Image.Image) -> tuple[np.ndarray, int, float]:
  cropped = resize_and_center_crop(img.convert("RGB"))
  rgb = np.asarray(cropped)
  x = Tensor(normalize_to_chw(cropped)[None, ...])
  feat, class_idx, prob = _predict_with_features(model, x)
  fc_weight = model.fc.weight.numpy()
  cam = _cam_for_class(feat, fc_weight, class_idx)
  cam_up = _upsample_bilinear(cam, size=rgb.shape[0])
  cam_norm = (cam_up - cam_up.min()) / max(cam_up.max() - cam_up.min(), 1e-9)
  heatmap = (cm.jet(cam_norm)[..., :3] * 255).astype(np.float32)
  composite = (1 - _OVERLAY_ALPHA) * rgb.astype(np.float32) + _OVERLAY_ALPHA * heatmap
  return composite.astype(np.uint8), class_idx, prob
