from dataclasses import dataclass

import numpy as np


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
