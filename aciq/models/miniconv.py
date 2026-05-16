# Reference: https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist.py
from __future__ import annotations

from enum import StrEnum

import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit, function
from tinygrad.nn.optim import AdamW

from aciq.helpers import fuse_conv_bn_inplace


class BlockName(StrEnum):
  BLOCK1 = "block1"
  BLOCK2 = "block2"
  BLOCK3 = "block3"
  BLOCK4 = "block4"


class BlockName2(StrEnum):
  BLOCK1 = "Sluoksnis 1"
  BLOCK2 = "Sluoksnis 2"
  BLOCK3 = "Sluoksnis 3"
  BLOCK4 = "Sluoksnis 4"


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
    self.classifier = nn.Linear(128, 10)
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
    return self.classifier(self.block4.mean((2, 3)))

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
      ("classifier", self.classifier),
    ]

  @classmethod
  def clear_jit_caches(cls) -> None:
    for name in ("train_step", "test_loss", "test_acc", "get_activations"):
      cls.__dict__[name].reset()
