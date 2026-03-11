from __future__ import annotations
import functools
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

from scipy import stats
import numpy as np


def skewness(data: np.ndarray) -> np.floating[Any]:
  d = data - np.mean(data)
  return np.mean(d**3) / np.mean(d**2) ** 1.5


def kurtosis(data: np.ndarray) -> np.floating[Any]:
  d = data - np.mean(data)
  return np.mean(d**4) / np.mean(d**2) ** 2 - 3.0


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
  def cdf_at(self, x: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def __repr__(self) -> str: ...

  def pdf(self) -> np.ndarray:
    return self.pdf_at(self._data)

  def logpdf(self) -> np.ndarray:
    p = self.pdf()
    return np.log(np.maximum(p, np.finfo(p.dtype).tiny))

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
    return f"Gaussian({self.mu:.2f}, {self.sigma:.2f})"

  @property
  def mu(self) -> np.floating[Any]:
    return np.mean(self._data)

  @functools.cached_property
  def sigma(self) -> np.floating[Any]:
    return np.std(self._data)

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return np.exp(-(((x - self.mu) / self.sigma) ** 2) / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma

  def cdf_at(self, x: np.ndarray) -> np.ndarray:
    return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)


class Laplace(Distribution):
  def __repr__(self) -> str:
    return f"Laplace({self.mu:.2f}, {self.b:.2f})"

  @property
  def mu(self) -> np.floating[Any]:
    return np.median(self._data)

  @functools.cached_property
  def b(self) -> np.floating[Any]:
    return np.mean(np.abs(self._data - self.mu))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(x - self.mu) / self.b) / (2.0 * self.b)

  def cdf_at(self, x: np.ndarray) -> np.ndarray:
    return stats.laplace.cdf(x, loc=self.mu, scale=self.b)


class StudentT(Distribution):
  def __repr__(self) -> str:
    return f"Student-t({self.df:.2f}, {self.loc:.2f}, {self.scale:.2f})"

  @functools.cached_property
  def _fit(self) -> tuple[float, float, float]:
    return stats.t.fit(self._data)

  @functools.cached_property
  def df(self) -> float:
    return self._fit[0]

  @functools.cached_property
  def loc(self) -> float:
    return self._fit[1]

  @functools.cached_property
  def scale(self) -> float:
    return self._fit[2]

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    coeff = math.exp(math.lgamma((self.df + 1) / 2) - math.lgamma(self.df / 2)) / (math.sqrt(self.df * math.pi) * self.scale)
    return coeff * (1 + ((x - self.loc) / self.scale) ** 2 / self.df) ** (-(self.df + 1) / 2)

  def cdf_at(self, x: np.ndarray) -> np.ndarray:
    return stats.t.cdf(x, self.df, loc=self.loc, scale=self.scale)


class GeneralizedGaussian(Distribution):
  def __repr__(self) -> str:
    return f"GED({self.beta:.2f}, {self.loc:.2f}, {self.scale:.2f})"

  @functools.cached_property
  def _fit(self) -> tuple[float, float, float]:
    return stats.gennorm.fit(self._data)

  @functools.cached_property
  def beta(self) -> float:
    return self._fit[0]

  @functools.cached_property
  def loc(self) -> float:
    return self._fit[1]

  @functools.cached_property
  def scale(self) -> float:
    return self._fit[2]

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return self.beta / (2 * self.scale * math.exp(math.lgamma(1 / self.beta))) * np.exp(-(np.abs((x - self.loc) / self.scale) ** self.beta))

  def cdf_at(self, x: np.ndarray) -> np.ndarray:
    return stats.gennorm.cdf(x, self.beta, loc=self.loc, scale=self.scale)
