import unittest

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear

from aciq.bias_correction import apply_bias_correction, bias_correction_delta, LayerInputStats
from aciq.distributions import ClippedGaussian
from aciq.resnet import compute_input_stats, _post_residual_stats
from aciq.resnet import ResNet
from aciq.quantization import quantize_symmetric


def _quantize_weight(w: np.ndarray, bits: int = 4) -> np.ndarray:
  alpha = float(np.max(np.abs(w))) or 1.0
  return quantize_symmetric(w.flatten(), alpha, bits).reshape(w.shape).astype(np.float32)


class TestClippedNormal(unittest.TestCase):
  def test_relu_mean_matches_monte_carlo(self):
    rng = np.random.default_rng(0)
    beta = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    gamma = np.array([0.5, 1.0, 1.0, 1.5, 2.0])
    samples = rng.normal(loc=beta, scale=gamma, size=(2_000_000, 5))
    relu = np.maximum(samples, 0.0)
    expected = relu.mean(axis=0)
    got = ClippedGaussian.mean(beta, gamma)
    np.testing.assert_allclose(got, expected, atol=5e-3)

  def test_relu_var_matches_monte_carlo(self):
    rng = np.random.default_rng(1)
    beta = np.array([-1.0, 0.0, 1.5])
    gamma = np.array([1.0, 1.0, 0.7])
    samples = rng.normal(loc=beta, scale=gamma, size=(2_000_000, 3))
    relu = np.maximum(samples, 0.0)
    expected = relu.var(axis=0)
    got = ClippedGaussian.variance(beta, gamma)
    np.testing.assert_allclose(got, expected, atol=5e-3)

  def test_gamma_sign_invariant(self):
    # |gamma| should be used internally so a negative gamma doesn't flip the mean
    np.testing.assert_allclose(
      ClippedGaussian.mean(np.array([0.5]), np.array([-1.0])),
      ClippedGaussian.mean(np.array([0.5]), np.array([1.0])),
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
    apply_bias_correction(layer, W_fp, b, stats)
    np.testing.assert_allclose(layer.weight.numpy(), W_q)
    assert not np.allclose(layer.bias.numpy(), b)


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
    stats = compute_input_stats(model)
    # Every weight module should have a stats entry with C_in matching weight shape
    expected_keys = {"stem", "fc"}
    for li in range(1, 5):
      layer = (model.layer1, model.layer2, model.layer3, model.layer4)[li - 1]
      for bi in range(len(layer)):
        prefix = f"layer{li}.{bi}"
        expected_keys |= {f"{prefix}.conv1", f"{prefix}.conv2"}
        if layer[bi].downsample_conv is not None:
          expected_keys.add(f"{prefix}.downsample")
    assert set(stats.keys()) == expected_keys
    # Stem stats: N(0, 1) per input channel, shape derived from conv1
    stem = stats["stem"]
    assert stem.E_x.shape == (model.conv1.weight.shape[1],)
    assert np.allclose(stem.E_x, 0.0)
    assert np.allclose(stem.Var_x, 1.0)
    # Shape check on a known layer
    s = stats["layer1.0.conv1"]
    assert s.E_x.shape == (model.layer1[0].conv1.weight.shape[1],)
    assert s.Var_x.shape == (model.layer1[0].conv1.weight.shape[1],)
    # All variances finite and non-negative
    assert np.all(np.isfinite(s.Var_x))
    assert np.all(s.Var_x >= 0.0)


if __name__ == "__main__":
  unittest.main()
