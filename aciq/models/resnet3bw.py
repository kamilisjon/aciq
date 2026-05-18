from __future__ import annotations

from enum import StrEnum

import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit, function
from tinygrad.nn.optim import AdamW

from aciq.helpers import fuse_conv_bn_inplace
from aciq.models.resnet import BasicBlock


class BlockName(StrEnum):
  STEM = "stem"
  BLOCK1_A1 = "block1_activation_1"
  BLOCK1_A2 = "block1_activation_2"


class BlockName2(StrEnum):
  STEM = "Įvadinis sluoksnis"
  BLOCK1_A1 = "Bloko 1 vid. aktivacija"
  BLOCK1_A2 = "Bloko 1 išėjimas"


class ResNet3BW:
  def __init__(self, in_channels: int = 3) -> None:
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
