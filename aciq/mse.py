from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from aciq.distributions import Distribution
from aciq.quantization import minmax_alpha


def clipping_mse(dist: Distribution, alpha: float) -> float:
  lower, _ = quad(lambda x: (x + alpha) ** 2 * dist.pdf_at(np.asarray(x)), -np.inf, -alpha, limit=100)
  upper, _ = quad(lambda x: (x - alpha) ** 2 * dist.pdf_at(np.asarray(x)), alpha, np.inf, limit=100)
  return float(lower + upper)


def granular_mse(alpha: float, bits: int) -> float:
  return alpha**2 / (3.0 * 4**bits)


def total_mse(dist: Distribution, alpha: float, bits: int) -> float:
  return clipping_mse(dist, alpha) + granular_mse(alpha, bits)


def optimal_alpha(dist: Distribution, bits: int) -> float:
  alpha_max = minmax_alpha(dist._data)
  result = minimize_scalar(
    lambda a: total_mse(dist, a, bits),
    bounds=(1e-8, alpha_max),
    method="bounded",
  )
  return float(result.x)
