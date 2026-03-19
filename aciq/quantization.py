from typing import Callable

import numpy as np

from scipy.optimize import root_scalar


def minmax_alpha(data: np.ndarray) -> float:
  return float(np.max(np.abs(data)))


def solve_symmetric_mae_alpha(cdf: Callable[[float], float], b: int, alpha_max: float) -> float:
  def g(alpha):
    return cdf(alpha) - cdf(-alpha) - 1 + 1 / (2 ** (b + 1))

  lo, hi = 0.0, alpha_max

  if g(hi) <= 0:
    print("Using alpha_max")
    return alpha_max

  return root_scalar(g, bracket=(lo, hi), method="brentq").root


def percentile_alpha(data: np.ndarray, percentile: float = 99.99) -> float:
  return float(np.percentile(np.abs(data), percentile))


def quantize(data: np.ndarray, alpha: float, bits: int) -> np.ndarray:
  """Symmetric uniform quantization with range [-alpha, alpha]. Returns dequantized values."""
  qmax = 2 ** (bits - 1) - 1
  scale = alpha / qmax
  quantized = np.clip(np.round(data / scale), -qmax, qmax)
  return quantized * scale
