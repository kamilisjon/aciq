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
TEST_CHUNK_SIZE = 1000


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
  def train_step(self, X: Tensor, Y: Tensor, opt: AdamW) -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(self.batch_size, high=X.shape[0])
    loss = self(X[samples]).sparse_categorical_crossentropy(Y[samples]).backward()
    return loss.realize(*opt.schedule_step())

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


def train_model(seed: int, steps: int = 1170, lr: float = 1e-3, batch_size: int = 512, eval_every: int = 10) -> tuple[MNISTModel, float, list[float], list[float]]:
  Tensor.manual_seed(seed)
  np.random.seed(seed)

  x_train, y_train, x_test, y_test = _load_normalized()
  model = MNISTModel()
  opt = AdamW(get_parameters(model), lr=lr)
  model.batch_size = batch_size

  train_losses: list[float] = []
  test_losses: list[float] = []
  window_sum = 0.0
  for i in tqdm(range(steps), desc="train"):
    GlobalCounters.reset()
    window_sum += float(model.train_step(x_train, y_train, opt).item())
    if (i + 1) % eval_every == 0:
      train_losses.append(window_sum / eval_every)
      assert x_test.shape[0] % TEST_CHUNK_SIZE == 0
      chunk_loss_sum = 0.0
      for j in range(0, x_test.shape[0], TEST_CHUNK_SIZE):
        x_chunk = x_test[j:j + TEST_CHUNK_SIZE].contiguous()
        y_chunk = y_test[j:j + TEST_CHUNK_SIZE].contiguous()
        chunk_loss_sum += float(model.test_loss_step(x_chunk, y_chunk).item())
      test_losses.append(chunk_loss_sum / (x_test.shape[0] // TEST_CHUNK_SIZE))
      window_sum = 0.0

  return model, evaluate_model(model, x_test, y_test), train_losses, test_losses


def evaluate_model(model: MNISTModel, x_test: Tensor, y_test: Tensor) -> float:
  assert x_test.shape[0] % TEST_CHUNK_SIZE == 0
  chunk_acc_sum = 0.0
  for j in range(0, x_test.shape[0], TEST_CHUNK_SIZE):
    x_chunk = x_test[j:j + TEST_CHUNK_SIZE].contiguous()
    y_chunk = y_test[j:j + TEST_CHUNK_SIZE].contiguous()
    chunk_acc_sum += float(model.get_test_acc(x_chunk, y_chunk).item())
  return chunk_acc_sum / (x_test.shape[0] // TEST_CHUNK_SIZE)
