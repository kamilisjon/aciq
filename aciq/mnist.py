# Reference: https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist.py
from __future__ import annotations

import numpy as np
import tinygrad.nn as nn
from tinygrad import GlobalCounters, Tensor, TinyJit, function
from tinygrad.helpers import tqdm
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

from aciq.bn_fusion import fuse_conv_bn_inplace


_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def _load_normalized() -> tuple[Tensor, Tensor, Tensor, Tensor]:
  x_train, y_train, x_test, y_test = mnist()
  x_train = (x_train.float() / 255.0 - _MNIST_MEAN) / _MNIST_STD
  x_test = (x_test.float() / 255.0 - _MNIST_MEAN) / _MNIST_STD
  return x_train, y_train, x_test, y_test


class MNISTModel:
  def __init__(self) -> None:
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(64)
    self.classifier = nn.Linear(64, 10)
    self.fused = False
    self.batch_size = 0

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
    return self.classifier(self.block3.mean((2, 3)))

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

  def fuse(self) -> None:
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    fuse_conv_bn_inplace(self.conv2, self.bn2)
    fuse_conv_bn_inplace(self.conv3, self.bn3)
    self.fused = True

  @property
  def activations(self) -> dict[str, Tensor]:
    return {
      "block1": self.block1,
      "block2": self.block2,
      "block3": self.block3,
    }


def train_model(seed: int, steps: int = 70, lr: float = 1e-3, batch_size: int = 512) -> tuple[MNISTModel, float, list[float], list[float]]:
  Tensor.manual_seed(seed)
  np.random.seed(seed)

  x_train, y_train, x_test, y_test = _load_normalized()
  model = MNISTModel()
  opt = AdamW(get_parameters(model), lr=lr)
  train_losses: list[float] = []
  test_losses: list[float] = []
  for _ in (t := tqdm(range(steps), desc="train")):
    GlobalCounters.reset()
    train_loss = float(model.train_step(x_train, y_train, opt, batch_size).item())
    test_loss = float(model.test_loss(x_test, y_test).item())
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    t.set_description(f"train_loss: {train_loss:6.5f} test_loss: {test_loss:6.5f}")
  return model, evaluate_model(model, x_test, y_test), train_losses, test_losses


def evaluate_model(model: MNISTModel, x_test: Tensor, y_test: Tensor) -> float:
  return float(model.test_acc(x_test, y_test).item())
