from typing import Callable

import numpy as np

from scipy.optimize import root_scalar


def bound_symmetric_minmax(data: np.ndarray) -> float:
  return float(np.max(np.abs(data)))


def _bound_symmetric_aciq_mae(alpha: float, cdf: Callable[[float], float], b: int) -> float:
  return cdf(alpha) - cdf(-alpha) - 1 + 1 / (2 ** (b + 1))


def bound_symmetric_aciq_mae(cdf: Callable[[float], float], b: int, alpha_max: float) -> float:
  def g(alpha: float) -> float:
    return _bound_symmetric_aciq_mae(alpha, cdf, b)

  lo, hi = 0.0, alpha_max
  if g(hi) <= 0:
    return alpha_max
  return root_scalar(g, bracket=(lo, hi), method="brentq").root


def quantize_symmetric(data: np.ndarray, alpha: float, bits: int, return_dequantized: bool = True) -> np.ndarray:
  qmax = 2 ** (bits - 1) - 1
  scale = alpha / qmax
  quantized = np.clip(np.round(data / scale), -qmax, qmax)
  if return_dequantized:
    return quantized * scale
  else:
    return quantized