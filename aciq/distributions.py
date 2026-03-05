from __future__ import annotations
import functools
import math
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy import stats


class DistributionType(str, Enum):
  GAUSSIAN = "Gaussian"
  LAPLACE = "Laplace"
  STUDENT_T = "Student-t"


class Distribution:
  def __init__(self, data: np.ndarray):
    self._data = data

  @staticmethod
  def _mean(data: np.ndarray) -> float:
    return float(np.mean(data))

  @staticmethod
  def _variance(data: np.ndarray) -> float:
    return float(np.var(data, ddof=1))

  @staticmethod
  def _std(data: np.ndarray) -> float:
    return float(np.std(data, ddof=1))

  @staticmethod
  def _skewness(data: np.ndarray) -> float:
    return float(stats.skew(data))

  # TODO: What does kurtosis mean? How it is calculated?
  @staticmethod
  def _kurtosis(data: np.ndarray) -> float:
    return float(stats.kurtosis(data))

  @staticmethod
  def _min(data: np.ndarray) -> float:
    return float(data.min())

  @staticmethod
  def _max(data: np.ndarray) -> float:
    return float(data.max())

  @staticmethod
  def _median(data: np.ndarray) -> float:
    return float(np.median(data))

  @staticmethod
  def _n(data: np.ndarray) -> int:
    return len(data)

  @functools.cached_property
  def n(self) -> int:
    return self._n(self._data)

  @functools.cached_property
  def mean(self) -> float:
    return self._mean(self._data)

  @functools.cached_property
  def variance(self) -> float:
    return self._variance(self._data)

  @functools.cached_property
  def std(self) -> float:
    return self._std(self._data)

  @functools.cached_property
  def skewness(self) -> float:
    return self._skewness(self._data)

  @functools.cached_property
  def kurtosis(self) -> float:
    return self._kurtosis(self._data)

  @functools.cached_property
  def min(self) -> float:
    return self._min(self._data)

  @functools.cached_property
  def median(self) -> float:
    return self._median(self._data)

  @functools.cached_property
  def max(self) -> float:
    return self._max(self._data)

  @functools.cache
  def fit(self, dist_type: DistributionType) -> FittedDistribution:
    match dist_type:
      case DistributionType.GAUSSIAN:
        return Gaussian(self._data)
      case DistributionType.LAPLACE:
        return Laplace(self._data)
      case DistributionType.STUDENT_T:
        return StudentT(self._data)
      case _:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


# TODO: test against R. Gaussian and Laplace.
# TODO: find scientific article for distribution scientific backing. Need to be citable, so could be included into report. Gaussian and Laplace.


class FittedDistribution(ABC):
  def __init__(self, data: np.ndarray):
    self._data = data

  @abstractmethod
  def pdf_at(self, x: np.ndarray) -> np.ndarray: ...

  def pdf(self) -> np.ndarray:
    return self.pdf_at(self._data)

  def logpdf(self) -> np.ndarray:
    return np.log(self.pdf())

  @functools.cached_property
  def log_likelihood(self) -> float:
    return float(np.sum(self.logpdf()))


class Gaussian(FittedDistribution):
  @property
  def mu(self) -> float:
    return Distribution._mean(self._data)

  # TODO: why cant I directly use self.std? Why does ddof differ?
  @functools.cached_property
  def sigma(self) -> float:
    return float(np.std(self._data, ddof=0))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    z = (x - self.mu) / self.sigma
    return np.exp(-(z**2) / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma


class Laplace(FittedDistribution):
  @property
  def mu(self) -> float:
    return Distribution._median(self._data)

  @functools.cached_property
  def b(self) -> float:
    return float(np.mean(np.abs(self._data - self.mu)))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(x - self.mu) / self.b) / (2.0 * self.b)


class StudentT(FittedDistribution):
  @functools.cached_property
  def df(self) -> float:
    k = Distribution._kurtosis(self._data)
    # TODO: why df is inf at kurtosis <= 0?
    if k <= 0:
      return float("inf")
    return max(6.0 / k + 4.0, 2.01)

  @property
  def loc(self) -> float:
    return Distribution._mean(self._data)

  @functools.cached_property
  def scale(self) -> float:
    var = float(np.var(self._data, ddof=0))
    # TODO: why this formula for non positive kurtosis / inf df?
    if np.isinf(self.df):
      return float(np.sqrt(var))
    return float(np.sqrt(var * (self.df - 2.0) / self.df))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    coeff = math.exp(math.lgamma((self.df + 1) / 2) - math.lgamma(self.df / 2)) / (math.sqrt(self.df * math.pi) * self.scale)
    return coeff * (1 + ((x - self.loc) / self.scale) ** 2 / self.df) ** (-(self.df + 1) / 2)
