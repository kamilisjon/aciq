# Reference: https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist.py
from __future__ import annotations

import numpy as np
from tinygrad import GlobalCounters, Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

from aciq.models.miniconv import MiniConv


_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def _load_normalized() -> tuple[Tensor, Tensor, Tensor, Tensor]:
  x_train, y_train, x_test, y_test = mnist()
  x_train = (x_train.float() / 255.0 - _MNIST_MEAN) / _MNIST_STD
  x_test = (x_test.float() / 255.0 - _MNIST_MEAN) / _MNIST_STD
  return x_train, y_train, x_test, y_test


def train_model(
  model_cls: type[MiniConv],
  seed: int = 0,
  steps: int = 100,
  batch_size: int = 4096,
  gather_losses: bool = False,
) -> tuple[MiniConv, float, list[float], list[float]]:
  Tensor.manual_seed(seed)
  np.random.seed(seed)

  x_train, y_train, x_test, y_test = _load_normalized()
  model = model_cls()
  opt = AdamW(get_parameters(model), lr=1e-3)
  train_losses: list[float] = []
  test_losses: list[float] = []
  for _ in (t := tqdm(range(steps), desc="train")):
    GlobalCounters.reset()
    train_loss = float(model.train_step(x_train, y_train, opt, batch_size).item())
    if gather_losses:
      test_loss = float(model.test_loss(x_test, y_test).item())
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      t.set_description(f"train_loss: {train_loss:6.5f} test_loss: {test_loss:6.5f}")
    else:
      t.set_description(f"train_loss: {train_loss:6.5f}")
  test_acc = float(model.test_acc(x_test, y_test).item())
  return model, test_acc, train_losses, test_losses
