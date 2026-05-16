from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear


# -----------------------------------------------------------------------------
# Mean-shift measurement
# -----------------------------------------------------------------------------


@dataclass
class MeanShift:
  method: str
  layer: str
  mean_shift: float


class ChannelMeansAccumulator:
  def __init__(self) -> None:
    # Holds 1D vectors, which holds running sum of channel params. One value per layer channel.
    self.channels_sums: dict[str, np.ndarray] = {}
    # Holds parameters counts. It is enough to hold parameters count of a single channel as other channels have same parameter counts.
    self.param_counts: dict[str, int] = {}

  def update(self, layer_name: str, activation: np.ndarray) -> None:
    channel_sum = activation.sum(axis=(0, 2, 3)).astype(np.float32)
    b, _, h, w = activation.shape
    if layer_name not in self.channels_sums:
      self.channels_sums[layer_name] = np.zeros_like(channel_sum)
      self.param_counts[layer_name] = 0
    self.channels_sums[layer_name] += channel_sum
    self.param_counts[layer_name] += b * h * w

  def _get_per_layer_means(self) -> dict[str, np.ndarray]:
    return {layer_name: (self.channels_sums[layer_name] / self.param_counts[layer_name]).astype(np.float32) for layer_name in self.channels_sums}

  def layers_means_shifts(self, other: "ChannelMeansAccumulator", method: str) -> list[MeanShift]:
    self_means, other_means = self._get_per_layer_means(), other._get_per_layer_means()
    return [MeanShift(method, layer_name, float(np.mean(np.abs(self_means[layer_name] - other_means[layer_name])))) for layer_name in self_means]


# -----------------------------------------------------------------------------
# Bias correction
# -----------------------------------------------------------------------------


def _quantization_error_epsilon(w_fp: np.ndarray, w_dequant: np.ndarray) -> np.ndarray:
  eps = (w_dequant - w_fp).astype(np.float32)
  if eps.ndim == 4:
    return eps.sum(axis=(2, 3))
  if eps.ndim == 2:
    return eps
  raise ValueError(f"unsupported weight rank {eps.ndim}")


def apply_bias_correction(module: Conv2d | Linear, W_fp: np.ndarray, b_orig: np.ndarray, E_x: np.ndarray) -> None:
  delta_b = _quantization_error_epsilon(W_fp, module.weight.numpy().astype(np.float32)) @ E_x.astype(np.float32)
  module.bias = Tensor((b_orig.astype(np.float32) - delta_b).astype(np.float32))
