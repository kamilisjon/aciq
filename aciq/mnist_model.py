from __future__ import annotations

import numpy as np
import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import tqdm
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

from aciq.fusion import fuse_inplace


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

  def _block(self, x: Tensor, conv: nn.Conv2d, bn: nn.BatchNorm) -> Tensor:
    out = conv(x)
    if not self.fused:
      out = bn(out)
    return out.relu()

  def __call__(self, x: Tensor) -> Tensor:
    self.block1 = self._block(x, self.conv1, self.bn1)
    self.block2 = self._block(self.block1, self.conv2, self.bn2)
    self.block3 = self._block(self.block2, self.conv3, self.bn3)
    self.block4 = self._block(self.block3, self.conv4, self.bn4)
    self.block5 = self._block(self.block4, self.conv5, self.bn5)
    return self.classifier(self.block5.mean((2, 3)))

  def fuse(self) -> None:
    fuse_inplace(self.conv1, self.bn1)
    fuse_inplace(self.conv2, self.bn2)
    fuse_inplace(self.conv3, self.bn3)
    fuse_inplace(self.conv4, self.bn4)
    fuse_inplace(self.conv5, self.bn5)
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


def train_model(seed: int, epochs: int = 10, lr: float = 1e-3, batch_size: int = 512) -> tuple[MNISTModel, float]:
  Tensor.manual_seed(seed)
  np.random.seed(seed)

  x_train, y_train, x_test, y_test = _load_normalized()
  model = MNISTModel()
  opt = Adam(get_parameters(model), lr=lr)

  @TinyJit
  @Tensor.train()
  def train_step(X: Tensor, Y: Tensor) -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(batch_size, high=X.shape[0])
    loss = model(X[samples]).sparse_categorical_crossentropy(Y[samples]).backward()
    return loss.realize(*opt.schedule_step())

  steps_per_epoch = int(x_train.shape[0]) // batch_size
  total_steps = epochs * steps_per_epoch
  for _ in tqdm(range(total_steps), desc="train"):
    train_step(x_train, y_train)

  return model, evaluate_model(model, x_test, y_test)


def evaluate_model(model: MNISTModel, x_test: Tensor, y_test: Tensor) -> float:
  @TinyJit
  def get_test_acc(X: Tensor, Y: Tensor) -> Tensor:
    return (model(X).argmax(axis=1) == Y).mean()

  get_test_acc(x_test, y_test).item()
  return float(get_test_acc(x_test, y_test).item())
