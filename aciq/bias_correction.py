from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear


MATH_DTYPE = PARAMETERS_DTYPE = np.float32

# -----------------------------------------------------------------------------
# Mean-shift measurement
# -----------------------------------------------------------------------------


@dataclass
class MeanShift:
  method: str
  layer: str
  mean_shift: float


def compute_shift(fp32_outputs: dict[str, np.ndarray], quant_outputs: dict[str, np.ndarray], method: str) -> list[MeanShift]:
  return [MeanShift(method=method, layer=name, mean_shift=float(np.mean(np.abs(fp32_outputs[name] - quant_outputs[name])))) for name in fp32_outputs]


class MeanShiftAccumulator:
  def __init__(self) -> None:
    self._sums: dict[str, np.ndarray] = {}
    self._counts: dict[str, int] = {}

  def update(self, name: str, activation: np.ndarray) -> None:
    channel_sum = activation.sum(axis=(0, 2, 3)).astype(MATH_DTYPE)
    b, _, h, w = activation.shape
    if name not in self._sums:
      self._sums[name] = np.zeros_like(channel_sum)
      self._counts[name] = 0
    self._sums[name] += channel_sum
    self._counts[name] += b * h * w

  def get_per_channel_means(self) -> dict[str, np.ndarray]:
    return {name: (self._sums[name] / self._counts[name]).astype(MATH_DTYPE) for name in self._sums}


# -----------------------------------------------------------------------------
# Bias correction
# -----------------------------------------------------------------------------


def _quantization_error_epsilon(w_fp: np.ndarray, w_dequant: np.ndarray) -> np.ndarray:
  eps = (w_dequant - w_fp).astype(MATH_DTYPE)
  if eps.ndim == 4:
    return eps.sum(axis=(2, 3))
  if eps.ndim == 2:
    return eps
  raise ValueError(f"unsupported weight rank {eps.ndim}")


def apply_bias_correction(module: Conv2d | Linear, W_fp: np.ndarray, b_orig: np.ndarray, E_x: np.ndarray) -> None:
  delta_b = _quantization_error_epsilon(W_fp, module.weight.numpy().astype(MATH_DTYPE)) @ E_x.astype(MATH_DTYPE)
  module.bias = Tensor((b_orig.astype(MATH_DTYPE) - delta_b).astype(PARAMETERS_DTYPE))
