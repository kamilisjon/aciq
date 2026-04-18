import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LayerStats:
  mean: np.ndarray  # per-channel means, shape (C,)
  var: np.ndarray  # per-channel variances, shape (C,)


@dataclass
class ShiftResult:
  mean_shift: dict[str, float]  # mean of |per-channel mean difference|
  var_shift: dict[str, float]  # mean of |per-channel variance difference|


def compute_shift(fp32_outputs: dict[str, LayerStats], quant_outputs: dict[str, LayerStats]) -> ShiftResult:
  """Mean of absolute per-channel mean and variance shifts."""
  mean_shift = {name: float(np.mean(np.abs(fp32_outputs[name].mean - quant_outputs[name].mean))) for name in fp32_outputs}
  var_shift = {name: float(np.mean(np.abs(fp32_outputs[name].var - quant_outputs[name].var))) for name in fp32_outputs}
  return ShiftResult(mean_shift=mean_shift, var_shift=var_shift)


class StatsAccumulator:
  """Online accumulation of per-channel mean and variance from (B, C, H, W) activations."""

  def __init__(self) -> None:
    self._sums: dict[str, np.ndarray] = {}
    self._sq_sums: dict[str, np.ndarray] = {}
    self._counts: dict[str, int] = {}

  def update(self, name: str, activation: np.ndarray) -> None:
    """Accumulate statistics from a (B, C, H, W) activation tensor."""
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


def save_shifts_csv(shifts: dict[str, ShiftResult], layer_names: list[str], save_path: Path) -> None:
  save_path.parent.mkdir(parents=True, exist_ok=True)
  header = ["method"]
  for name in layer_names:
    header += [f"{name}__mean_shift", f"{name}__var_shift"]

  with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for method, shift in shifts.items():
      row: list[str | float] = [method]
      for name in layer_names:
        row += [shift.mean_shift[name], shift.var_shift[name]]
      writer.writerow(row)


def load_shifts_csv(path: Path) -> tuple[dict[str, ShiftResult], list[str]]:
  with open(path) as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    assert columns is not None

    layer_names: list[str] = []
    for col in columns:
      if col.endswith("__mean_shift"):
        layer_names.append(col.removesuffix("__mean_shift"))

    shifts: dict[str, ShiftResult] = {}
    for row in reader:
      method = row["method"]
      mean_shift = {name: float(row[f"{name}__mean_shift"]) for name in layer_names}
      var_shift = {name: float(row[f"{name}__var_shift"]) for name in layer_names}
      shifts[method] = ShiftResult(mean_shift=mean_shift, var_shift=var_shift)

  return shifts, layer_names
