from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
# Bias and variance correction
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


def _w_squared_contracted(W: np.ndarray) -> np.ndarray:
  """Returns sum_over_kernel(W**2) of shape (C_out, C_in) for conv, W**2 for linear."""
  W2 = (W.astype(np.float64)) ** 2
  if W2.ndim == 4:
    return W2.sum(axis=(2, 3))
  if W2.ndim == 2:
    return W2
  raise ValueError(f"unsupported weight rank {W2.ndim}")


def _w_contracted(W: np.ndarray) -> np.ndarray:
  """Returns sum_over_kernel(W) of shape (C_out, C_in) for conv, W for linear (used for E[y])."""
  Wd = W.astype(np.float64)
  if Wd.ndim == 4:
    return Wd.sum(axis=(2, 3))
  if Wd.ndim == 2:
    return Wd
  raise ValueError(f"unsupported weight rank {Wd.ndim}")


def bias_correction_delta(W_fp: np.ndarray, W_q: np.ndarray, E_x: np.ndarray) -> np.ndarray:
  """Per-output-channel bias error introduced by quantization: Δb = ε_sum @ E[x].

  Subtract the returned vector from the layer's bias to absorb the mean shift.
  """
  return _epsilon_input_contracted(W_fp, W_q) @ E_x.astype(np.float64)


def output_mean(W: np.ndarray, b: np.ndarray, E_x: np.ndarray) -> np.ndarray:
  """E[y] per output channel = W_sum @ E[x] + b. Spatial stationarity assumed for conv."""
  return _w_contracted(W) @ E_x.astype(np.float64) + b.astype(np.float64)


def output_variance(W: np.ndarray, Var_x: np.ndarray) -> np.ndarray:
  """Var[y] per output channel = W²_sum @ Var[x] under per-channel input independence
  and per-channel spatial i.i.d."""
  return _w_squared_contracted(W) @ Var_x.astype(np.float64)


def variance_correction_scale(
  W_fp: np.ndarray, W_q: np.ndarray, Var_x: np.ndarray, eps: float = 1e-8, clip: tuple[float, float] = (0.5, 2.0)
) -> np.ndarray:
  """Per-output-channel scale s_c = sqrt(Var[y_fp] / Var[y_q]). Clipped for stability."""
  var_fp = output_variance(W_fp, Var_x)
  var_q = output_variance(W_q, Var_x)
  s = np.sqrt(np.maximum(var_fp, 0.0) / np.maximum(var_q, eps))
  return np.clip(s, clip[0], clip[1])


class CorrectionMode(StrEnum):
  BIAS = "bias"
  VARIANCE = "variance"
  JOINT = "joint"


def apply_correction(
  module: Conv2d | Linear,
  W_fp: np.ndarray,
  b_orig: np.ndarray,
  mode: CorrectionMode,
  stats: LayerInputStats,
  var_clip: tuple[float, float] = (0.5, 2.0),
) -> None:
  W_q = module.weight.numpy().astype(np.float64)
  E_x = stats.E_x
  Var_x = stats.Var_x
  s = variance_correction_scale(W_fp, W_q, Var_x, clip=var_clip)
  s_w = s.reshape((-1, 1, 1, 1)) if W_q.ndim == 4 else s.reshape((-1, 1))

  match mode:
    case CorrectionMode.BIAS:
      delta_b = bias_correction_delta(W_fp, W_q, E_x)
      new_b = b_orig.astype(np.float64) - delta_b
      module.bias = Tensor(new_b.astype(np.float32))
    case CorrectionMode.VARIANCE:
      E_y_q = output_mean(W_q, b_orig, E_x)
      module.weight = Tensor((s_w * W_q).astype(np.float32))
      module.bias = Tensor((s * b_orig.astype(np.float64) + (1.0 - s) * E_y_q).astype(np.float32))
    case CorrectionMode.JOINT:
      delta_b = bias_correction_delta(W_fp, W_q, E_x)
      b_corrected = b_orig.astype(np.float64) - delta_b
      E_y_fp = output_mean(W_fp, b_orig, E_x)
      module.weight = Tensor((s_w * W_q).astype(np.float32))
      module.bias = Tensor((s * b_corrected + (1.0 - s) * E_y_fp).astype(np.float32))
