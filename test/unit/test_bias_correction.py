import unittest

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear

from aciq.bias_correction import (
  CORRECTION_MODES,
  apply_correction,
  bias_correction_delta,
  capture_bn_params,
  clipped_normal_mean,
  clipped_normal_var,
  compute_input_stats,
  output_mean,
  output_variance,
  variance_correction_scale,
  LayerInputStats,
  _post_residual_stats,
)
from aciq.models.resnet import ResNet
from aciq.quantization import quantize


def _quantize_weight(w: np.ndarray, bits: int = 4) -> np.ndarray:
  alpha = float(np.max(np.abs(w))) or 1.0
  return quantize(w.flatten(), alpha, bits).reshape(w.shape).astype(np.float32)


class TestClippedNormal(unittest.TestCase):
  def test_relu_mean_matches_monte_carlo(self):
    rng = np.random.default_rng(0)
    beta = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    gamma = np.array([0.5, 1.0, 1.0, 1.5, 2.0])
    samples = rng.normal(loc=beta, scale=gamma, size=(2_000_000, 5))
    relu = np.maximum(samples, 0.0)
    expected = relu.mean(axis=0)
    got = clipped_normal_mean(beta, gamma)
    np.testing.assert_allclose(got, expected, atol=5e-3)

  def test_relu_var_matches_monte_carlo(self):
    rng = np.random.default_rng(1)
    beta = np.array([-1.0, 0.0, 1.5])
    gamma = np.array([1.0, 1.0, 0.7])
    samples = rng.normal(loc=beta, scale=gamma, size=(2_000_000, 3))
    relu = np.maximum(samples, 0.0)
    expected = relu.var(axis=0)
    got = clipped_normal_var(beta, gamma)
    np.testing.assert_allclose(got, expected, atol=5e-3)

  def test_gamma_sign_invariant(self):
    # |gamma| should be used internally so a negative gamma doesn't flip the mean
    np.testing.assert_allclose(
      clipped_normal_mean(np.array([0.5]), np.array([-1.0])),
      clipped_normal_mean(np.array([0.5]), np.array([1.0])),
    )


class TestBiasCorrectionMath(unittest.TestCase):
  def test_linear_eliminates_mean_shift(self):
    rng = np.random.default_rng(2)
    C_in, C_out = 32, 16
    W_fp = rng.normal(scale=0.1, size=(C_out, C_in)).astype(np.float32)
    b = rng.normal(scale=0.05, size=(C_out,)).astype(np.float32)
    W_q = _quantize_weight(W_fp, bits=3)

    x = rng.normal(loc=0.4, scale=0.6, size=(20_000, C_in)).astype(np.float32)
    y_fp = x @ W_fp.T + b
    y_q = x @ W_q.T + b
    observed_shift = (y_q - y_fp).mean(axis=0)

    delta = bias_correction_delta(W_fp, W_q, x.mean(axis=0))
    np.testing.assert_allclose(delta, observed_shift, atol=1e-3)

    b_corr = b - delta
    y_q_corr = x @ W_q.T + b_corr
    np.testing.assert_allclose(y_q_corr.mean(axis=0), y_fp.mean(axis=0), atol=2e-3)

  def test_conv_eliminates_mean_shift(self):
    rng = np.random.default_rng(3)
    C_in, C_out, K = 8, 4, 3
    W_fp = rng.normal(scale=0.2, size=(C_out, C_in, K, K)).astype(np.float32)
    W_q = _quantize_weight(W_fp, bits=3)

    E_x = np.array([0.3, -0.1, 0.2, 0.0, 0.5, 0.1, 0.0, -0.2])
    delta = bias_correction_delta(W_fp, W_q, E_x)

    expected = np.zeros(C_out)
    for c in range(C_out):
      for ci in range(C_in):
        expected[c] += (W_q[c, ci] - W_fp[c, ci]).sum() * E_x[ci]
    np.testing.assert_allclose(delta, expected, atol=1e-6)


class TestVarianceCorrection(unittest.TestCase):
  def test_variance_scale_brings_var_close(self):
    rng = np.random.default_rng(4)
    C_in, C_out = 64, 16
    W_fp = rng.normal(scale=0.1, size=(C_out, C_in)).astype(np.float32)
    W_q = _quantize_weight(W_fp, bits=3)
    Var_x = rng.uniform(0.1, 1.5, size=(C_in,))
    s = variance_correction_scale(W_fp, W_q, Var_x, clip=(0.1, 5.0))
    var_fp = output_variance(W_fp, Var_x)
    var_q_scaled = (s**2) * output_variance(W_q, Var_x)
    np.testing.assert_allclose(var_q_scaled, var_fp, rtol=1e-6)

  def test_variance_scale_clipped(self):
    rng = np.random.default_rng(5)
    C_in, C_out = 4, 2
    W_fp = rng.normal(size=(C_out, C_in)).astype(np.float32)
    W_q = (W_fp * 1e-12).astype(np.float32)  # near-zero quantized weight
    Var_x = np.ones(C_in)
    s = variance_correction_scale(W_fp, W_q, Var_x, clip=(0.5, 2.0))
    assert np.all(s == 2.0)


class TestApplyCorrection(unittest.TestCase):
  def _make_linear(self, W: np.ndarray, b: np.ndarray) -> Linear:
    layer = Linear(W.shape[1], W.shape[0], bias=True)
    layer.weight = Tensor(W.astype(np.float32))
    layer.bias = Tensor(b.astype(np.float32))
    return layer

  def _make_conv(self, W: np.ndarray, b: np.ndarray) -> Conv2d:
    C_out, C_in, K, _ = W.shape
    layer = Conv2d(C_in, C_out, kernel_size=K, padding=K // 2, bias=True)
    layer.weight = Tensor(W.astype(np.float32))
    layer.bias = Tensor(b.astype(np.float32))
    return layer

  def test_bias_mode_modifies_only_bias(self):
    rng = np.random.default_rng(6)
    W_fp = rng.normal(scale=0.1, size=(8, 16)).astype(np.float32)
    b = rng.normal(scale=0.05, size=(8,)).astype(np.float32)
    W_q = _quantize_weight(W_fp, bits=3)
    layer = self._make_linear(W_q, b)
    stats = LayerInputStats(E_x=rng.normal(size=(16,)), Var_x=np.ones(16))
    apply_correction(layer, W_fp, b, "bias", stats)
    np.testing.assert_allclose(layer.weight.numpy(), W_q)
    assert not np.allclose(layer.bias.numpy(), b)

  def test_joint_preserves_mean_and_variance(self):
    rng = np.random.default_rng(7)
    C_in, C_out = 32, 8
    W_fp = rng.normal(scale=0.2, size=(C_out, C_in)).astype(np.float32)
    b = rng.normal(scale=0.1, size=(C_out,)).astype(np.float32)
    W_q = _quantize_weight(W_fp, bits=3)
    x = rng.normal(loc=0.5, scale=0.7, size=(100_000, C_in)).astype(np.float32)
    E_x = x.mean(axis=0)
    Var_x = x.var(axis=0)

    layer = self._make_linear(W_q, b)
    apply_correction(layer, W_fp, b, "joint", LayerInputStats(E_x=E_x, Var_x=Var_x), var_clip=(0.01, 100.0))
    W_out = layer.weight.numpy()
    b_out = layer.bias.numpy()
    y_fp = x @ W_fp.T + b
    y_corr = x @ W_out.T + b_out
    np.testing.assert_allclose(y_corr.mean(axis=0), y_fp.mean(axis=0), atol=5e-3)
    np.testing.assert_allclose(y_corr.var(axis=0), y_fp.var(axis=0), rtol=2e-2, atol=5e-3)

  def test_none_mode_is_noop(self):
    rng = np.random.default_rng(8)
    W = rng.normal(size=(4, 8)).astype(np.float32)
    b = rng.normal(size=(4,)).astype(np.float32)
    layer = self._make_linear(W, b)
    apply_correction(layer, W, b, "none", LayerInputStats(E_x=np.zeros(8), Var_x=np.ones(8)))
    np.testing.assert_array_equal(layer.weight.numpy(), W)
    np.testing.assert_array_equal(layer.bias.numpy(), b)

  def test_modes_constant_lists_all(self):
    assert set(CORRECTION_MODES) == {"none", "bias", "variance", "joint"}


class TestResidualPropagation(unittest.TestCase):
  def test_post_residual_matches_monte_carlo(self):
    """ReLU(N(beta_main, gamma_main**2) + skip) per-channel stats vs Monte Carlo."""
    rng = np.random.default_rng(9)
    beta_main = np.array([0.5, -0.3, 1.2])
    gamma_main = np.array([1.0, 0.8, 1.5])
    mu_skip = np.array([0.2, 0.1, -0.4])
    var_skip = np.array([0.3, 0.5, 0.7])
    main = rng.normal(loc=beta_main, scale=gamma_main, size=(1_000_000, 3))
    skip = rng.normal(loc=mu_skip, scale=np.sqrt(var_skip), size=(1_000_000, 3))
    relu = np.maximum(main + skip, 0.0)
    expected_mu = relu.mean(axis=0)
    expected_var = relu.var(axis=0)
    got_mu, got_var = _post_residual_stats(beta_main, gamma_main, mu_skip, var_skip)
    np.testing.assert_allclose(got_mu, expected_mu, atol=5e-3)
    np.testing.assert_allclose(got_var, expected_var, atol=8e-3)


class TestResnetIntegration(unittest.TestCase):
  def test_capture_and_input_stats_resnet18(self):
    model = ResNet(18)
    bn_params = capture_bn_params(model)
    stats = compute_input_stats(model, bn_params)
    # Every weight module except stem should have a stats entry with C_in matching weight shape
    expected_keys = {"fc"}
    for li in range(1, 5):
      layer = (model.layer1, model.layer2, model.layer3, model.layer4)[li - 1]
      for bi in range(len(layer)):
        prefix = f"layer{li}.{bi}"
        expected_keys |= {f"{prefix}.conv1", f"{prefix}.conv2"}
        if layer[bi].downsample:
          expected_keys.add(f"{prefix}.downsample")
    assert set(stats.keys()) == expected_keys
    # Shape check on a known layer
    s = stats["layer1.0.conv1"]
    assert s.E_x.shape == (model.layer1[0].conv1.weight.shape[1],)
    assert s.Var_x.shape == (model.layer1[0].conv1.weight.shape[1],)
    # All variances finite and non-negative
    assert np.all(np.isfinite(s.Var_x))
    assert np.all(s.Var_x >= 0.0)


if __name__ == "__main__":
  unittest.main()
