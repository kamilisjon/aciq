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

  if g(alpha_max) <= 0:
    return alpha_max

  return root_scalar(g, bracket=(0.0, alpha_max), method="brentq").root


def quantize_symmetric(data: np.ndarray, alpha: float, bits: int) -> np.ndarray:
  # NOTE: returns dequantized weights
  # TODO: write this in terms of thesis formulation and test against this implementation
  qmax = 2 ** (bits - 1) - 1
  scale = alpha / qmax
  quantized = np.clip(np.round(data / scale), -qmax, qmax)
  return quantized * scale
