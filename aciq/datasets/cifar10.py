from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from tinygrad import GlobalCounters, Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn.datasets import cifar
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters


class _Trainable(Protocol):
  def train_step(self, X: Tensor, Y: Tensor, opt: AdamW, batch_size: int) -> Tensor: ...
  def test_loss(self, X: Tensor, Y: Tensor) -> Tensor: ...
  def test_acc(self, X: Tensor, Y: Tensor) -> Tensor: ...


_M = TypeVar("_M", bound=_Trainable)


_CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _load_normalized() -> tuple[Tensor, Tensor, Tensor, Tensor]:
  x_train, y_train, x_test, y_test = cifar()
  mean = Tensor(_CIFAR_MEAN).reshape(1, 3, 1, 1)
  std = Tensor(_CIFAR_STD).reshape(1, 3, 1, 1)
  x_train = (x_train.float() / 255.0 - mean) / std
  x_test = (x_test.float() / 255.0 - mean) / std
  return x_train, y_train, x_test, y_test


def train_model(
  model_cls: type[_M],
  seed: int = 0,
  steps: int = 500,
  batch_size: int = 4096,
  gather_losses: bool = False,
) -> tuple[_M, float, list[float], list[float]]:
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
