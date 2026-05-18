from __future__ import annotations

import copy
from enum import StrEnum

import numpy as np
import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit, function
from tinygrad.nn.optim import AdamW

from aciq.distributions import ClippedGaussian
from aciq.helpers import fuse_conv_bn_inplace
from aciq.models.resnet import BasicBlock
from aciq.quantization.bias_correction import apply_bias_correction


class BlockName(StrEnum):
  STEM = "stem"
  BLOCK1_A1 = "block1_activation_1"
  BLOCK1_A2 = "block1_activation_2"


class BlockName2(StrEnum):
  STEM = "Įvadinis sluoksnis"
  BLOCK1_A1 = "Bloko 1 vid. aktivacija"
  BLOCK1_A2 = "Bloko 1 išėjimas"


class ResNet4:
  def __init__(self) -> None:
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.block1 = BasicBlock(64, 64, stride=1)
    self.fc = nn.Linear(64, 10)
    self.fused = False

  @function
  def __call__(self, x: Tensor) -> Tensor:
    out = self.conv1(x)
    if not self.fused:
      out = self.bn1(out)
    self.stem_activation = out.relu()
    out = self.stem_activation.pad([1, 1, 1, 1]).max_pool2d((3, 3), 2)
    out = self.block1(out)
    return self.fc(out.mean((2, 3)))

  @TinyJit
  @Tensor.train()
  def train_step(self, X_train: Tensor, Y_train: Tensor, opt: AdamW, batch_size: int) -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    loss = self(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())

  @TinyJit
  def test_loss(self, X_test: Tensor, Y_test: Tensor) -> Tensor:
    return self(X_test).sparse_categorical_crossentropy(Y_test).realize()

  @TinyJit
  def test_acc(self, X_test: Tensor, Y_test: Tensor) -> Tensor:
    return (self(X_test).argmax(axis=1) == Y_test).mean()

  @TinyJit
  def get_activations(self, X: Tensor) -> dict[str, Tensor]:
    self(X)
    return {
      str(BlockName.STEM): self.stem_activation.realize(),
      str(BlockName.BLOCK1_A1): self.block1.activation_1.realize(),
      str(BlockName.BLOCK1_A2): self.block1.activation_2.realize(),
    }

  def fuse(self) -> None:
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    self.block1.fuse()
    self.fused = True

  @property
  def weight_modules(self) -> list[nn.Conv2d | nn.Linear]:
    return [m for _, m in self.named_weight_modules]

  @property
  def named_weight_modules(self) -> list[tuple[str, nn.Conv2d | nn.Linear]]:
    return [
      ("stem", self.conv1),
      ("block1.conv1", self.block1.conv1),
      ("block1.conv2", self.block1.conv2),
      ("classifier", self.fc),
    ]

  @classmethod
  def clear_jit_caches(cls) -> None:
    for name in ("train_step", "test_loss", "test_acc", "get_activations"):
      cls.__dict__[name].reset()


# -----------------------------------------------------------------------------
# Bias correction
# -----------------------------------------------------------------------------


def _bn_effective_params(bn: nn.BatchNorm) -> tuple[np.ndarray, np.ndarray]:
  assert bn.weight is not None and bn.bias is not None, "expected affine BatchNorm"
  gamma_eff = np.abs(bn.weight.numpy().astype(np.float64))
  beta_eff = bn.bias.numpy().astype(np.float64)
  return gamma_eff, beta_eff


def _post_residual_stats(beta_main: np.ndarray, gamma_main: np.ndarray, mu_skip: np.ndarray, var_skip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  mu = beta_main + mu_skip
  var = gamma_main**2 + var_skip
  sigma = np.sqrt(np.maximum(var, 0.0))
  return ClippedGaussian.mean(mu, sigma), ClippedGaussian.variance(mu, sigma)


def compute_input_stats(model: ResNet4) -> dict[str, np.ndarray]:
  """Per-channel E[x] at each weight-bearing module, derived analytically from the BN parameters.

  Mirrors the construction in aciq/models/resnet.py:275 for the ResNet18 model, simplified to
  ResNet4's single BasicBlock without a downsample branch.
  """
  out: dict[str, np.ndarray] = {}

  # CIFAR-10 inputs are per-channel zero-mean unit-variance after `_load_normalized`.
  c_in_stem = int(model.conv1.weight.shape[1])
  out["stem"] = np.zeros(c_in_stem, dtype=np.float64)

  # ReLU(BN(stem)) is the input to block1.conv1. The interim max-pool is treated as a no-op
  # on the per-channel mean, matching the simplification used by `aciq/models/resnet.py`.
  gamma_stem, beta_stem = _bn_effective_params(model.bn1)
  mu_after_stem = ClippedGaussian.mean(beta_stem, gamma_stem)
  var_after_stem = ClippedGaussian.variance(beta_stem, gamma_stem)
  out["block1.conv1"] = mu_after_stem.copy()

  # ReLU(BN(block1.conv1)) is the input to block1.conv2.
  gamma_b1, beta_b1 = _bn_effective_params(model.block1.bn1)
  out["block1.conv2"] = ClippedGaussian.mean(beta_b1, gamma_b1)

  # Post-residual ReLU of the BasicBlock feeds the classifier (via GAP, which preserves the mean).
  # BasicBlock has stride=1 and matching channels, so the skip branch is the block input itself.
  gamma_b2, beta_b2 = _bn_effective_params(model.block1.bn2)
  post_mu, _ = _post_residual_stats(beta_b2, gamma_b2, mu_after_stem, var_after_stem)
  out["classifier"] = post_mu
  return out


def bias_correct_model(base: ResNet4, fp_modules: dict[str, nn.Conv2d | nn.Linear], input_stats: dict[str, np.ndarray]) -> ResNet4:
  """Deep-copy `base` and subtract the analytic per-channel mean shift from each module's bias."""
  m = copy.deepcopy(base)
  for name, module in m.named_weight_modules:
    W_fp = fp_modules[name].weight.numpy()
    b_orig = module.bias.numpy() if module.bias is not None else np.zeros(module.weight.shape[0], dtype=np.float32)
    apply_bias_correction(module, W_fp, b_orig, input_stats[name])
  return m
