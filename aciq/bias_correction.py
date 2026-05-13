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


def compute_shift(fp32_outputs: dict[str, np.ndarray], quant_outputs: dict[str, np.ndarray], method: str) -> list[MeanShift]:
  return [MeanShift(method=method, layer=name, mean_shift=float(np.mean(np.abs(fp32_outputs[name] - quant_outputs[name])))) for name in fp32_outputs]


class StatsAccumulator:
  def __init__(self) -> None:
    self._sums: dict[str, np.ndarray] = {}
    self._counts: dict[str, int] = {}

  def update(self, name: str, activation: np.ndarray) -> None:
    channel_sum = activation.sum(axis=(0, 2, 3)).astype(np.float64)
    b, _, h, w = activation.shape
    if name not in self._sums:
      self._sums[name] = np.zeros_like(channel_sum)
      self._counts[name] = 0
    self._sums[name] += channel_sum
    self._counts[name] += b * h * w

  def get_per_channel_means(self) -> dict[str, np.ndarray]:
    return {name: (self._sums[name] / self._counts[name]).astype(np.float32) for name in self._sums}

# -----------------------------------------------------------------------------
# Bias correction
# -----------------------------------------------------------------------------

@dataclass
class LayerInputStats:
  E_x: np.ndarray  # (C_in,)
  Var_x: np.ndarray  # (C_in,)


def _epsilon_input_contracted(W_fp: np.ndarray, W_q: np.ndarray) -> np.ndarray:
  """Returns ε_summed of shape (C_out, C_in): sum of ε over kernel spatial dims for conv,
  identity for linear."""
  eps = (W_q - W_fp).astype(np.float64)
  if eps.ndim == 4:
    return eps.sum(axis=(2, 3))
  if eps.ndim == 2:
    return eps
  raise ValueError(f"unsupported weight rank {eps.ndim}")


def bias_correction_delta(W_fp: np.ndarray, W_q: np.ndarray, E_x: np.ndarray) -> np.ndarray:
  """Per-output-channel bias error introduced by quantization: Δb = ε_sum @ E[x].

  Subtract the returned vector from the layer's bias to absorb the mean shift.
  """
  return _epsilon_input_contracted(W_fp, W_q) @ E_x.astype(np.float64)


def apply_correction(module: Conv2d | Linear, W_fp: np.ndarray, b_orig: np.ndarray, stats: LayerInputStats) -> None:
  W_q = module.weight.numpy().astype(np.float64)
  delta_b = bias_correction_delta(W_fp, W_q, stats.E_x)
  new_b = b_orig.astype(np.float64) - delta_b
  module.bias = Tensor(new_b.astype(np.float32))