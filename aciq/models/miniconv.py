# Reference: https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist.py
from __future__ import annotations

import copy
from enum import StrEnum

import numpy as np
import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit, function
from tinygrad.nn.optim import Optimizer

from aciq.distributions import ClippedGaussian
from aciq.helpers import fuse_conv_bn_inplace
from aciq.quantization.bias_correction import apply_bias_correction


class BlockName(StrEnum):
  BLOCK1 = "block1"
  BLOCK2 = "block2"
  BLOCK3 = "block3"
  BLOCK4 = "block4"


class BlockName2(StrEnum):
  BLOCK1 = "Block 1"
  BLOCK2 = "Block 2"
  BLOCK3 = "Block 3"
  BLOCK4 = "Block 4"


class MiniConv:
  def __init__(self) -> None:
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False)
    self.bn4 = nn.BatchNorm2d(128)
    self.fc = nn.Linear(128, 10)
    self.fused = False

  def _block(self, x: Tensor, conv: nn.Conv2d, bn: nn.BatchNorm) -> Tensor:
    out = conv(x)
    if not self.fused:
      out = bn(out)
    return out.relu()

  @function
  def __call__(self, x: Tensor) -> Tensor:
    self.block1 = self._block(x, self.conv1, self.bn1)
    self.block2 = self._block(self.block1, self.conv2, self.bn2)
    self.block3 = self._block(self.block2, self.conv3, self.bn3)
    self.block4 = self._block(self.block3, self.conv4, self.bn4)
    return self.fc(self.block4.mean((2, 3)))

  @TinyJit
  @Tensor.train()
  def train_step(self, X_train: Tensor, Y_train: Tensor, opt: Optimizer, batch_size: int) -> Tensor:
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
    return {str(name): v.realize() for name, v in zip(BlockName, [self.block1, self.block2, self.block3, self.block4])}

  def fuse(self) -> None:
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    fuse_conv_bn_inplace(self.conv2, self.bn2)
    fuse_conv_bn_inplace(self.conv3, self.bn3)
    fuse_conv_bn_inplace(self.conv4, self.bn4)
    self.fused = True

  @property
  def weight_modules(self) -> list[nn.Conv2d | nn.Linear]:
    return [m for _, m in self.named_weight_modules]

  @property
  def named_weight_modules(self) -> list[tuple[str, nn.Conv2d | nn.Linear]]:
    return [
      ("block1", self.conv1),
      ("block2", self.conv2),
      ("block3", self.conv3),
      ("block4", self.conv4),
      ("classifier", self.fc),
    ]

  @classmethod
  def clear_jit_caches(cls) -> None:
    for name in ("train_step", "test_loss", "test_acc", "get_activations"):
      cls.__dict__[name].reset()


# -----------------------------------------------------------------------------------------------------------
# Bias correction. Source: https://arxiv.org/abs/1906.04721
# -----------------------------------------------------------------------------------------------------------


def _bn_effective_params(bn: nn.BatchNorm) -> tuple[np.ndarray, np.ndarray]:
  assert bn.weight is not None and bn.bias is not None, "expected affine BatchNorm"
  gamma_eff = np.abs(bn.weight.numpy().astype(np.float64))
  beta_eff = bn.bias.numpy().astype(np.float64)
  return gamma_eff, beta_eff


def compute_input_stats(model: MiniConv) -> dict[str, np.ndarray]:
  out: dict[str, np.ndarray] = {}
  out["block1"] = np.zeros(int(model.conv1.weight.shape[1]), dtype=np.float64)
  gamma_b1, beta_b1 = _bn_effective_params(model.bn1)
  out["block2"] = ClippedGaussian.mean(beta_b1, gamma_b1)
  gamma_b2, beta_b2 = _bn_effective_params(model.bn2)
  out["block3"] = ClippedGaussian.mean(beta_b2, gamma_b2)
  gamma_b3, beta_b3 = _bn_effective_params(model.bn3)
  out["block4"] = ClippedGaussian.mean(beta_b3, gamma_b3)
  gamma_b4, beta_b4 = _bn_effective_params(model.bn4)
  out["classifier"] = ClippedGaussian.mean(beta_b4, gamma_b4)
  return out


def bias_correct_model(base: MiniConv, fp_modules: dict[str, nn.Conv2d | nn.Linear], input_stats: dict[str, np.ndarray]) -> MiniConv:
  m = copy.deepcopy(base)
  for name, module in m.named_weight_modules:
    W_fp = fp_modules[name].weight.numpy()
    b_orig = module.bias.numpy() if module.bias is not None else np.zeros(module.weight.shape[0], dtype=np.float32)
    apply_bias_correction(module, W_fp, b_orig, input_stats[name])
  return m
