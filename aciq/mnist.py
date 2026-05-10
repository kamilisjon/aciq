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
    self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False)
    self.bn4 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
    self.bn5 = nn.BatchNorm2d(128)
    self.classifier = nn.Linear(128, 10)
    self.fused = False
    self.opt = None
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
    self.block4 = self._block(self.block3, self.conv4, self.bn4)
    self.block5 = self._block(self.block4, self.conv5, self.bn5)
    return self.classifier(self.block5.mean((2, 3)))

  @TinyJit
  @Tensor.train()
  def train_step(self, X: Tensor, Y: Tensor) -> Tensor:
    self.opt.zero_grad()
    samples = Tensor.randint(self.batch_size, high=X.shape[0])
    loss = self(X[samples]).sparse_categorical_crossentropy(Y[samples]).backward()
    return loss.realize(*self.opt.schedule_step())

  @TinyJit
  def test_loss_step(self, X: Tensor, Y: Tensor) -> Tensor:
    return self(X).sparse_categorical_crossentropy(Y).realize()

  @TinyJit
  def get_test_acc(self, X: Tensor, Y: Tensor) -> Tensor:
    return (self(X).argmax(axis=1) == Y).mean()

  def fuse(self) -> None:
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    fuse_conv_bn_inplace(self.conv2, self.bn2)
    fuse_conv_bn_inplace(self.conv3, self.bn3)
    fuse_conv_bn_inplace(self.conv4, self.bn4)
    fuse_conv_bn_inplace(self.conv5, self.bn5)
    self.fused = True

  @property
  def activations(self) -> dict[str, Tensor]:
    return {
      "block1": self.block1,
      "block2": self.block2,
      "block3": self.block3,
      "block4": self.block4,
      "block5": self.block5,
    }


def train_model(seed: int, epochs: int = 10, lr: float = 1e-3, batch_size: int = 512) -> tuple[MNISTModel, float, list[float], list[float]]:
  Tensor.manual_seed(seed)
  np.random.seed(seed)

  x_train, y_train, x_test, y_test = _load_normalized()
  model = MNISTModel()
  model.opt = AdamW(get_parameters(model), lr=lr)
  model.batch_size = batch_size

  steps_per_epoch = int(x_train.shape[0]) // batch_size
  train_losses: list[float] = []
  test_losses: list[float] = []
  for _ in tqdm(range(epochs), desc="train"):
    epoch_loss_sum = 0.0
    for _ in range(steps_per_epoch):
      GlobalCounters.reset()
      epoch_loss_sum += float(model.train_step(x_train, y_train).item())
    train_losses.append(epoch_loss_sum / steps_per_epoch)
    test_losses.append(float(model.test_loss_step(x_test, y_test).item()))

  return model, evaluate_model(model, x_test, y_test), train_losses, test_losses


def evaluate_model(model: MNISTModel, x_test: Tensor, y_test: Tensor) -> float:
  model.get_test_acc(x_test, y_test).item()  # warmup
  return float(model.get_test_acc(x_test, y_test).item())
