from __future__ import annotations
import functools
import math
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


# TODO: What does skewness mean?
# TODO: should we return same type as would numpy return for kurtosis and skewness?
def skewness(data: np.ndarray) -> float:
  d = data - np.mean(data)
  return float(np.mean(d**3) / np.mean(d**2) ** 1.5)


# TODO: What does kurtosis mean? What variants of kurtosis exist as scipy has bias and fisher parameters?
def kurtosis(data: np.ndarray) -> float:
  d = data - np.mean(data)
  return float(np.mean(d**4) / np.mean(d**2) ** 2 - 3.0)


# TODO: test against R. all distributions
# TODO: find scientific article for distribution scientific backing. Need to be citable, so could be included into report. all distributions


class DistributionType(str, Enum):
  GAUSSIAN = "Gaussian"
  LAPLACE = "Laplace"
  STUDENT_T = "Student-t"


class Distribution(ABC):
  def __init__(self, data: np.ndarray):
    self._data = data

  @abstractmethod
  def pdf_at(self, x: np.ndarray) -> np.ndarray: ...

  def pdf(self) -> np.ndarray:
    return self.pdf_at(self._data)

  def logpdf(self) -> np.ndarray:
    return np.log(self.pdf())

  @functools.cached_property
  # TODO: how does Log-Likelihood informs about how well data fits the distribution?
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
      case _:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


class Gaussian(Distribution):
  @property
  def mu(self) -> float:
    return np.mean(self._data)

  @functools.cached_property
  def sigma(self) -> float:
    return np.std(self._data)

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    z = (x - self.mu) / self.sigma
    return np.exp(-(z**2) / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma


class Laplace(Distribution):
  @property
  def mu(self) -> float:
    return np.median(self._data)

  @functools.cached_property
  def b(self) -> float:
    return float(np.mean(np.abs(self._data - self.mu)))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(x - self.mu) / self.b) / (2.0 * self.b)


class StudentT(Distribution):
  @functools.cached_property
  def df(self) -> float:
    k = kurtosis(self._data)
    # TODO: why df is inf at kurtosis <= 0?
    if k <= 0:
      return float("inf")
    return max(6.0 / k + 4.0, 2.01)

  @property
  def loc(self) -> float:
    return np.mean(self._data)

  @functools.cached_property
  def scale(self) -> float:
    var = np.var(self._data)
    # TODO: why this formula for non positive kurtosis / inf df?
    if np.isinf(self.df):
      return float(np.sqrt(var))
    return float(np.sqrt(var * (self.df - 2.0) / self.df))

  def pdf_at(self, x: np.ndarray) -> np.ndarray:
    coeff = math.exp(math.lgamma((self.df + 1) / 2) - math.lgamma(self.df / 2)) / (math.sqrt(self.df * math.pi) * self.scale)
    return coeff * (1 + ((x - self.loc) / self.scale) ** 2 / self.df) ** (-(self.df + 1) / 2)
