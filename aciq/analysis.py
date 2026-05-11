from dataclasses import dataclass

import numpy as np


@dataclass
class LayerStats:
  mean: np.ndarray  # per-channel means, shape (C,)
  var: np.ndarray  # per-channel variances, shape (C,)


@dataclass
class ShiftResult:
  mean_shift: dict[str, float]  # mean of |per-channel mean difference|
  var_shift: dict[str, float]  # mean of |per-channel variance difference|


@dataclass
class ShiftRow:
  method: str
  layer: str
  mean_shift: float
  var_shift: float


def compute_shift(fp32_outputs: dict[str, LayerStats], quant_outputs: dict[str, LayerStats]) -> ShiftResult:
  mean_shift = {name: float(np.mean(np.abs(fp32_outputs[name].mean - quant_outputs[name].mean))) for name in fp32_outputs}
  var_shift = {name: float(np.mean(np.abs(fp32_outputs[name].var - quant_outputs[name].var))) for name in fp32_outputs}
  return ShiftResult(mean_shift=mean_shift, var_shift=var_shift)


class StatsAccumulator:
  def __init__(self) -> None:
    self._sums: dict[str, np.ndarray] = {}
    self._sq_sums: dict[str, np.ndarray] = {}
    self._counts: dict[str, int] = {}

  def update(self, name: str, activation: np.ndarray) -> None:
    ch_sum = activation.sum(axis=(0, 2, 3)).astype(np.float64)
    ch_sq_sum = (activation.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
    b, _, h, w = activation.shape
    n = b * h * w

    if name not in self._sums:
      self._sums[name] = np.zeros_like(ch_sum)
      self._sq_sums[name] = np.zeros_like(ch_sq_sum)
      self._counts[name] = 0

    self._sums[name] += ch_sum
    self._sq_sums[name] += ch_sq_sum
    self._counts[name] += n

  def finalize(self) -> dict[str, LayerStats]:
    result: dict[str, LayerStats] = {}
    for name in self._sums:
      n = self._counts[name]
      mean = (self._sums[name] / n).astype(np.float32)
      var = ((self._sq_sums[name] / n) - (self._sums[name] / n) ** 2).astype(np.float32)
      result[name] = LayerStats(mean=mean, var=var)
    return result
