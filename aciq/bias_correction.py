from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear

from aciq.resnet import Bottleneck, ResNet


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
# Per-layer input statistics (recursive forward propagation)
# -----------------------------------------------------------------------------


@dataclass
class LayerInputStats:
  E_x: np.ndarray  # (C_in,)
  Var_x: np.ndarray  # (C_in,)


def _post_relu_stats_from_bn(beta: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """ReLU(N(beta, gamma**2)) per channel."""
  return clipped_normal_mean(beta, gamma), clipped_normal_var(beta, gamma)


def _post_residual_stats(
  beta_main: np.ndarray, gamma_main: np.ndarray, mu_skip: np.ndarray, var_skip: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Statistics of ReLU(N(beta_main, gamma_main**2) + skip).

  Treats skip as approximately Gaussian, independent of the main branch. Combined
  pre-ReLU per channel is N(beta_main + mu_skip, gamma_main**2 + var_skip).
  """
  mu = beta_main + mu_skip
  var = gamma_main**2 + var_skip
  sigma = np.sqrt(np.maximum(var, 0.0))
  return clipped_normal_mean(mu, sigma), clipped_normal_var(mu, sigma)


def compute_input_stats(model: ResNet, bn_params: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, LayerInputStats]:
  """Walk the network forward, propagating per-channel (E[x], Var[x]) for each weight module's input.

  - The stem layer has no preceding BN/ReLU; its entry is omitted (caller treats stem as no-op).
  - Block first conv (`conv1`) input = previous block's post-residual activation stats.
  - Within-block convs (`conv2`, `conv3`) inputs = ReLU(BN_prev) stats.
  - Downsample conv input = block input (same as `conv1`).
  - FC input = layer4 last activation averaged over H*W; mean unchanged, variance / (H*W).
  """
  out: dict[str, LayerInputStats] = {}

  # After stem: ReLU(BN_stem). This is the input to the first block's conv1.
  gamma_stem, beta_stem = bn_params["stem"]
  current_mu, current_var = _post_relu_stats_from_bn(beta_stem, gamma_stem)
  # Spatial size after stem (stride 2 conv + stride 2 maxpool from 224 input) = 56x56.
  # layer1 keeps the same size (stride 1); layer2/3/4 each halve it via the first block's stride-2 conv.
  spatial_hw = 56 * 56

  for li, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
    if li >= 2:
      spatial_hw = max(1, spatial_hw // 4)
    for bi, block in enumerate(layer):
      prefix = f"layer{li}.{bi}"

      # conv1 input = previous block's post-residual activation
      out[f"{prefix}.conv1"] = LayerInputStats(E_x=current_mu.copy(), Var_x=current_var.copy())
      if block.downsample:
        out[f"{prefix}.downsample"] = LayerInputStats(E_x=current_mu.copy(), Var_x=current_var.copy())

      # conv2 input = ReLU(BN of conv1)
      gamma_bn1, beta_bn1 = bn_params[f"{prefix}.conv1"]
      mu_h1, var_h1 = _post_relu_stats_from_bn(beta_bn1, gamma_bn1)
      out[f"{prefix}.conv2"] = LayerInputStats(E_x=mu_h1, Var_x=var_h1)

      if isinstance(block, Bottleneck):
        # conv3 input = ReLU(BN of conv2)
        gamma_bn2, beta_bn2 = bn_params[f"{prefix}.conv2"]
        mu_h2, var_h2 = _post_relu_stats_from_bn(beta_bn2, gamma_bn2)
        out[f"{prefix}.conv3"] = LayerInputStats(E_x=mu_h2, Var_x=var_h2)
        last_bn_key = f"{prefix}.conv3"
      else:
        last_bn_key = f"{prefix}.conv2"

      # Update post-residual stats after this block
      gamma_last, beta_last = bn_params[last_bn_key]
      if block.downsample:
        gamma_ds, beta_ds = bn_params[f"{prefix}.downsample"]
        mu_skip = beta_ds
        var_skip = gamma_ds**2
      else:
        mu_skip = current_mu
        var_skip = current_var
      current_mu, current_var = _post_residual_stats(beta_last, gamma_last, mu_skip, var_skip)

  # FC input = global average pool of layer4 last activation. Mean is preserved;
  # variance shrinks by 1/(H*W) under spatial i.i.d. assumption.
  out["fc"] = LayerInputStats(E_x=current_mu, Var_x=current_var / float(spatial_hw))
  return out


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


def apply_correction(
  module: Conv2d | Linear,
  W_fp: np.ndarray,
  b_orig: np.ndarray,
  mode: str,
  stats: LayerInputStats,
  var_clip: tuple[float, float] = (0.5, 2.0),
) -> None:
  W_q = module.weight.numpy().astype(np.float64)
  E_x = stats.E_x
  Var_x = stats.Var_x

  if mode == "bias":
    delta_b = bias_correction_delta(W_fp, W_q, E_x)
    new_b = b_orig.astype(np.float64) - delta_b
    module.bias = Tensor(new_b.astype(np.float32))
    return

  s = variance_correction_scale(W_fp, W_q, Var_x, clip=var_clip)
  s_w = s.reshape((-1, 1, 1, 1)) if W_q.ndim == 4 else s.reshape((-1, 1))

  if mode == "variance":
    E_y_q = output_mean(W_q, b_orig, E_x)
    new_W = (s_w * W_q).astype(np.float32)
    new_b = (s * b_orig.astype(np.float64) + (1.0 - s) * E_y_q).astype(np.float32)
  elif mode == "joint":
    delta_b = bias_correction_delta(W_fp, W_q, E_x)
    b_corrected = b_orig.astype(np.float64) - delta_b
    E_y_fp = output_mean(W_fp, b_orig, E_x)
    new_W = (s_w * W_q).astype(np.float32)
    new_b = (s * b_corrected + (1.0 - s) * E_y_fp).astype(np.float32)
  else:
    raise ValueError(f"unknown correction mode {mode!r}")

  module.weight = Tensor(new_W)
  module.bias = Tensor(new_b)


CORRECTION_MODES = ("bias", "variance", "joint")
