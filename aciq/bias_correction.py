from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear

# Clipped normal distribution


def clipped_normal_mean(beta: np.ndarray, gamma: np.ndarray, a: float = 0.0, b: float = np.inf) -> np.ndarray:
  # Source: https://arxiv.org/pdf/1906.04721
  beta = np.asarray(beta, dtype=np.float64)
  gamma = np.asarray(gamma, dtype=np.float64)
  sigma = np.abs(gamma)
  alpha = (a - beta) / sigma
  beta_n = np.full_like(beta, np.inf) if not np.isfinite(b) else (b - beta) / sigma

  Phi_alpha = norm.cdf(alpha)
  Phi_beta = np.ones_like(beta) if not np.isfinite(b) else norm.cdf(beta_n)
  phi_alpha = norm.pdf(alpha)
  phi_beta = np.zeros_like(beta) if not np.isfinite(b) else norm.pdf(beta_n)

  term_low = a * Phi_alpha
  term_high = 0.0 if not np.isfinite(b) else b * (1.0 - Phi_beta)
  return term_low + term_high + sigma * (phi_alpha - phi_beta) + beta * (Phi_beta - Phi_alpha)


def clipped_normal_var(beta: np.ndarray, gamma: np.ndarray, a: float = 0.0, b: float = np.inf) -> np.ndarray:
  """Variance of N(beta, gamma**2) clipped to [a, b]. ReLU corresponds to a=0, b=inf."""
  beta = np.asarray(beta, dtype=np.float64)
  gamma = np.asarray(gamma, dtype=np.float64)
  sigma = np.abs(gamma)
  alpha = (a - beta) / sigma
  beta_n = np.full_like(beta, np.inf) if not np.isfinite(b) else (b - beta) / sigma

  Phi_alpha = norm.cdf(alpha)
  Phi_beta = np.ones_like(beta) if not np.isfinite(b) else norm.cdf(beta_n)
  phi_alpha = norm.pdf(alpha)
  phi_beta = np.zeros_like(beta) if not np.isfinite(b) else norm.pdf(beta_n)

  mu_clip = clipped_normal_mean(beta, gamma, a, b)
  Z = Phi_beta - Phi_alpha
  b_phi_beta = 0.0 if not np.isfinite(b) else b * phi_beta
  term_high = 0.0 if not np.isfinite(b) else (b - mu_clip) ** 2 * (1.0 - Phi_beta)

  var = (
    Z * (beta**2 + sigma**2 + mu_clip**2 - 2.0 * mu_clip * beta)
    + sigma * (a * phi_alpha - b_phi_beta)
    + sigma * (beta - 2.0 * mu_clip) * (phi_alpha - phi_beta)
    + (a - mu_clip) ** 2 * Phi_alpha
    + term_high
  )
  return np.maximum(var, 0.0)


# -----------------------------------------------------------------------------
# Per-layer input statistics
# -----------------------------------------------------------------------------


@dataclass
class LayerInputStats:
  E_x: np.ndarray  # (C_in,)
  Var_x: np.ndarray  # (C_in,)


# -----------------------------------------------------------------------------
# Bias correction
# -----------------------------------------------------------------------------


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


def apply_correction(
  module: Conv2d | Linear,
  W_fp: np.ndarray,
  b_orig: np.ndarray,
  stats: LayerInputStats,
) -> None:
  W_q = module.weight.numpy().astype(np.float64)
  delta_b = bias_correction_delta(W_fp, W_q, stats.E_x)
  new_b = b_orig.astype(np.float64) - delta_b
  module.bias = Tensor(new_b.astype(np.float32))
