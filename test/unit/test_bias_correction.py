import unittest

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear

from aciq.bias_correction import apply_bias_correction
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
    E_x = rng.normal(size=(16,))
    apply_bias_correction(layer, W_fp, b, E_x)
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
    # Stem E[x]: zeros per input channel, shape derived from conv1
    stem = stats["stem"]
    assert stem.shape == (model.conv1.weight.shape[1],)
    assert np.allclose(stem, 0.0)
    # Shape check on a known layer
    s = stats["layer1.0.conv1"]
    assert s.shape == (model.layer1[0].conv1.weight.shape[1],)


if __name__ == "__main__":
  unittest.main()
