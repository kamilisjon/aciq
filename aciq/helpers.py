import csv
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import BatchNorm, Conv2d


PATH_DATETIME_FORMAT = "%Y%m%d_%H%M%S"
RESULTS_DIR: Path = Path("results")
CSV_SEPARATOR = ","


def get_output_dir(root: Path, prefix: str = "") -> Path:
  timestamp = datetime.now().strftime(PATH_DATETIME_FORMAT)
  return root / (f"{prefix}_{timestamp}" if prefix else timestamp)


def save_csv(rows: list[Any], path: Path) -> None:
  assert rows, "save_csv requires a non-empty list (dataclass schema is read from rows[0])"
  assert is_dataclass(rows[0]), f"save_csv expects dataclass instances, got {type(rows[0]).__name__}"
  field_names = [f.name for f in fields(rows[0])]
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=field_names, delimiter=CSV_SEPARATOR)
    writer.writeheader()
    for row in rows:
      writer.writerow(asdict(row))


def _parse_csv_value(t: type, v: str) -> Any:
  if t is bool:
    return v == "True"
  return t(v)


def load_csv(path: Path, row_type: type[Any]) -> list[Any]:
  assert is_dataclass(row_type), f"load_csv expects a dataclass type, got {row_type.__name__}"
  hints = get_type_hints(row_type)
  with open(path) as f:
    reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
    return [row_type(**{k: _parse_csv_value(hints[k], v) for k, v in raw.items()}) for raw in reader]


def fuse_conv_bn(conv: Conv2d, bn: BatchNorm) -> tuple[Tensor, Tensor]:
  scale = bn.weight / (bn.running_var + bn.eps).sqrt()
  fused_w = conv.weight * scale.reshape(-1, 1, 1, 1)
  conv_b = conv.bias if conv.bias is not None else Tensor.zeros(conv.weight.shape[0])
  fused_b = scale * (conv_b - bn.running_mean) + bn.bias
  return fused_w, fused_b


def fuse_conv_bn_inplace(conv: Conv2d, bn: BatchNorm) -> None:
  conv.weight, conv.bias = fuse_conv_bn(conv, bn)


def mean_absolute_error(data_array_1: np.ndarray, data_array_2: np.ndarray) -> float:
  return float(np.mean(np.abs(data_array_1 - data_array_2)))
