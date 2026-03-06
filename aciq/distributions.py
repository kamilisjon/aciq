from __future__ import annotations
import functools
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np


def skewness(data: np.ndarray) -> np.floating[Any]:
  d = data - np.mean(data)
  return np.mean(d**3) / np.mean(d**2) ** 1.5


def kurtosis(data: np.ndarray) -> np.floating[Any]:
  d = data - np.mean(data)
  return np.mean(d**4) / np.mean(d**2) ** 2 - 3.0


def _ged_kurtosis(beta: float) -> float:
  return math.exp(math.lgamma(5 / beta) + math.lgamma(1 / beta) - 2 * math.lgamma(3 / beta)) - 3


class DistributionType(Enum):
  GAUSSIAN = auto()
  LAPLACE = auto()
  STUDENT_T = auto()
  GENERALIZED_GAUSSIAN = auto()


class Distribution(ABC):
  def __init__(self, data: np.ndarray):
    self._data = data

  @abstractmethod
  def pdf_at(self, x: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def __repr__(self) -> str: ...

  def pdf(self) -> np.ndarray:
    return self.pdf_at(self._data)

  def logpdf(self) -> np.ndarray:
    return np.log(self.pdf())

  @functools.cached_property
  def log_likelihood(self) -> float:
    return float(np.sum(self.logpdf()))

  @staticmethod
  def fit(data: np.ndarray, dist_type: DistributionType) -> Distribution:
    match dist_type:
      case DistributionType.GAUSSIAN:
        return Gaussian(data)
      case DistributionType.LAPLACE:
        return Laplace(data)
      case DistributionType.STUDENT_T:
        return StudentT(data)
      case DistributionType.GENERALIZED_GAUSSIAN:
        return GeneralizedGaussian(data)
      case _:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


class Gaussian(Distribution):
  def __repr__(self) -> str:
    return f"Gaussian({self.mu:.1f}, {self.sigma:.1f})"

  @property
  def mu(self) -> np.floating[Any]:
    return np.mean(self._data)

  @functools.cached_property
  def sigma(self) -> np.floating[Any]:
    return np.std(self._data)

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return np.exp(-(((x - self.mu) / self.sigma) ** 2) / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma


class Laplace(Distribution):
  def __repr__(self) -> str:
    return f"Laplace({self.mu:.1f}, {self.b:.1f})"

  @property
  def mu(self) -> np.floating[Any]:
    return np.median(self._data)

  @functools.cached_property
  def b(self) -> np.floating[Any]:
    return np.mean(np.abs(self._data - self.mu))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(x - self.mu) / self.b) / (2.0 * self.b)


class StudentT(Distribution):
  def __repr__(self) -> str:
    return f"Student-t({self.df:.1f}, {self.loc:.1f}, {self.scale:.1f})"

  @functools.cached_property
  def df(self) -> np.floating[Any]:
    k = kurtosis(self._data)
    if k <= 0:
      return self._data.dtype.type(np.inf)
    return np.maximum(6.0 / k + 4.0, 2.01)

  @property
  def loc(self) -> np.floating[Any]:
    return np.mean(self._data)

  @functools.cached_property
  def scale(self) -> np.floating[Any]:
    if np.isinf(self.df):
      return np.sqrt(np.var(self._data))
    return np.sqrt(np.var(self._data) * (self.df - 2.0) / self.df)

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    coeff = math.exp(math.lgamma((self.df + 1) / 2) - math.lgamma(self.df / 2)) / (math.sqrt(self.df * math.pi) * self.scale)
    return coeff * (1 + ((x - self.loc) / self.scale) ** 2 / self.df) ** (-(self.df + 1) / 2)


class GeneralizedGaussian(Distribution):
  def __repr__(self) -> str:
    return f"GED({self.beta:.1f}, {self.loc:.1f}, {self.scale:.1f})"

  @functools.cached_property
  def beta(self) -> np.floating[Any]:
    k = kurtosis(self._data)
    lo, hi = 0.01, 100.0
    k_at_hi = _ged_kurtosis(hi)
    k_at_lo = _ged_kurtosis(lo)
    if k >= k_at_lo:
      return self._data.dtype.type(lo)
    if k <= k_at_hi:
      return self._data.dtype.type(hi)
    for _ in range(100):
      mid = (lo + hi) / 2
      if _ged_kurtosis(mid) > k:
        lo = mid
      else:
        hi = mid
    return self._data.dtype.type((lo + hi) / 2)

  @property
  def loc(self) -> np.floating[Any]:
    return np.mean(self._data)

  @functools.cached_property
  def scale(self) -> np.floating[Any]:
    return np.sqrt(np.var(self._data) * math.exp(math.lgamma(1 / self.beta) - math.lgamma(3 / self.beta)))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return self.beta / (2 * self.scale * math.exp(math.lgamma(1 / self.beta))) * np.exp(-(np.abs((x - self.loc) / self.scale) ** self.beta))
