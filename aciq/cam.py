from __future__ import annotations

import copy

import numpy as np
from tinygrad import Tensor

from aciq.distributions import fit_distributions
from aciq.quantization.clipping import bound_symmetric_aciq_mae, bound_symmetric_minmax, quantize_symmetric
from aciq.resnet import ResNet, _weight_modules


_METHODS = {"per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"}


def cam_for_class(feat: np.ndarray, fc_weight: np.ndarray, class_idx: int) -> np.ndarray:
  return np.einsum("chw,c->hw", feat, fc_weight[class_idx])


def predict_batch_with_features(model: ResNet, x: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  ResNet.clear_jit_caches()
  activations = model.get_activations(x)
  feat = activations["layer4.1.activation_2"].numpy()
  gap = feat.mean(axis=(2, 3))
  fc_w = model.fc.weight.numpy()
  fc_b = model.fc.bias.numpy() if model.fc.bias is not None else np.zeros(fc_w.shape[0], dtype=np.float32)
  logits = gap @ fc_w.T + fc_b
  shifted = logits - logits.max(axis=1, keepdims=True)
  exps = np.exp(shifted)
  probs = exps / exps.sum(axis=1, keepdims=True)
  class_idx = np.argmax(probs, axis=1)
  pred_probs = probs[np.arange(len(class_idx)), class_idx]
  return feat, class_idx, pred_probs


def _alpha_aciq(vec: np.ndarray, bits: int) -> float:
  best_dist = fit_distributions(vec)[0]
  alpha_max = bound_symmetric_minmax(vec)
  return float(bound_symmetric_aciq_mae(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_max))


def _quantize_per_tensor(weight: np.ndarray, bits: int, clip: str) -> np.ndarray:
  vec = weight.flatten().astype(np.float64)
  alpha = bound_symmetric_minmax(vec) if clip == "minmax" else _alpha_aciq(vec, bits)
  return quantize_symmetric(vec, alpha, bits).reshape(weight.shape)


def _quantize_per_channel(weight: np.ndarray, bits: int, clip: str) -> np.ndarray:
  out = np.empty_like(weight, dtype=np.float64)
  for c in range(weight.shape[0]):
    ch = weight[c].flatten().astype(np.float64)
    alpha = bound_symmetric_minmax(ch) if clip == "minmax" else _alpha_aciq(ch, bits)
    out[c] = quantize_symmetric(ch, alpha, bits).reshape(weight[c].shape)
  return out


def build_quantized_variant(base: ResNet, method: str, bits: int) -> ResNet:
  assert method in _METHODS, f"unknown method {method!r}; expected one of {sorted(_METHODS)}"
  q_model = copy.deepcopy(base)
  fp_mods = dict(_weight_modules(base))
  q_mods = dict(_weight_modules(q_model))
  for weight_name, q_module in q_mods.items():
    fp_weight = fp_mods[weight_name].weight.numpy().astype(np.float32)
    if method.startswith("per_tensor_"):
      q_weight = _quantize_per_tensor(fp_weight, bits, method.removeprefix("per_tensor_"))
    else:
      q_weight = _quantize_per_channel(fp_weight, bits, method.removeprefix("per_channel_"))
    q_module.weight = Tensor(q_weight.astype(np.float32))
  return q_model
