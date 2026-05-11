from dataclasses import dataclass

import numpy as np


@dataclass
class LayerStats:
  mean: np.ndarray  # per-channel means, shape (C,)


@dataclass
class ShiftRow:
  method: str
  layer: str
  mean_shift: float


def compute_shift(fp32_outputs: dict[str, LayerStats], quant_outputs: dict[str, LayerStats], method: str) -> list[ShiftRow]:
  return [
    ShiftRow(
      method=method,
      layer=name,
      mean_shift=float(np.mean(np.abs(fp32_outputs[name].mean - quant_outputs[name].mean))),
    )
    for name in fp32_outputs
  ]


class StatsAccumulator:
  def __init__(self) -> None:
    self._sums: dict[str, np.ndarray] = {}
    self._counts: dict[str, int] = {}

  def update(self, name: str, activation: np.ndarray) -> None:
    ch_sum = activation.sum(axis=(0, 2, 3)).astype(np.float64)
    b, _, h, w = activation.shape
    n = b * h * w

    if name not in self._sums:
      self._sums[name] = np.zeros_like(ch_sum)
      self._counts[name] = 0

    self._sums[name] += ch_sum
    self._counts[name] += n

  def finalize(self) -> dict[str, LayerStats]:
    return {name: LayerStats(mean=(self._sums[name] / self._counts[name]).astype(np.float32)) for name in self._sums}
