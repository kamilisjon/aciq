from __future__ import annotations
import functools
import math
from abc import ABC, abstractmethod
from typing import Any

from scipy import stats
import numpy as np


def skewness(data: np.ndarray) -> np.floating[Any]:
  d = data - np.mean(data)
  return np.mean(d**3) / np.mean(d**2) ** 1.5


def kurtosis(data: np.ndarray) -> np.floating[Any]:
  d = data - np.mean(data)
  return np.mean(d**4) / np.mean(d**2) ** 2 - 3.0


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


class Gaussian(Distribution):
  def __repr__(self) -> str:
    return f"Gaussian({self.mu:.4f}, {self.sigma:.4f})"

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
    return f"Laplace({self.mu:.4f}, {self.b:.4f})"

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
    return f"Student-t({self.df:.4f}, {self.loc:.4f}, {self.scale:.4f})"

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
    return f"GED({self.beta:.4f}, {self.loc:.4f}, {self.scale:.4f})"

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


Distributions: set[type[Distribution]] = {Gaussian, Laplace, StudentT, GeneralizedGaussian}


def fit_distributions(data: np.ndarray) -> dict[type[Distribution], Distribution]:
  return {distribution: distribution(data) for distribution in Distributions}


class ClippedGaussian:
  """Analytical mean / variance for N(beta, gamma**2) clipped to [a, b].

  Source: arXiv:1906.04721. ReLU is the special case (a=0, b=+inf)."""

  @classmethod
  def mean(cls, beta: np.ndarray, gamma: np.ndarray, a: float = 0.0, b: float = np.inf) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    sigma = np.abs(gamma)
    alpha_n = (a - beta) / sigma
    beta_n = np.full_like(beta, np.inf) if not np.isfinite(b) else (b - beta) / sigma

    Phi_alpha = stats.norm.cdf(alpha_n)
    Phi_beta = np.ones_like(beta) if not np.isfinite(b) else stats.norm.cdf(beta_n)
    phi_alpha = stats.norm.pdf(alpha_n)
    phi_beta = np.zeros_like(beta) if not np.isfinite(b) else stats.norm.pdf(beta_n)

    term_low = a * Phi_alpha
    term_high = 0.0 if not np.isfinite(b) else b * (1.0 - Phi_beta)
    return term_low + term_high + sigma * (phi_alpha - phi_beta) + beta * (Phi_beta - Phi_alpha)

  @classmethod
  def variance(cls, beta: np.ndarray, gamma: np.ndarray, a: float = 0.0, b: float = np.inf) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    sigma = np.abs(gamma)
    alpha_n = (a - beta) / sigma
    beta_n = np.full_like(beta, np.inf) if not np.isfinite(b) else (b - beta) / sigma

    Phi_alpha = stats.norm.cdf(alpha_n)
    Phi_beta = np.ones_like(beta) if not np.isfinite(b) else stats.norm.cdf(beta_n)
    phi_alpha = stats.norm.pdf(alpha_n)
    phi_beta = np.zeros_like(beta) if not np.isfinite(b) else stats.norm.pdf(beta_n)

    mu_clip = cls.mean(beta, gamma, a, b)
    Z = Phi_beta - Phi_alpha
    b_phi_beta = 0.0 if not np.isfinite(b) else b * phi_beta
    term_high = 0.0 if not np.isfinite(b) else (b - mu_clip) ** 2 * (1.0 - Phi_beta)

    var = (
      Z * (beta**2 + sigma**2 + mu_clip**2 - 2.0 * mu_clip * beta)
      + sigma * (a * phi_alpha - b_phi_beta)
      + sigma * (beta - 2.0 * mu_clip) * (phi_alpha - phi_beta)
      + (a - mu_clip) ** 2 * Phi_alpha
      + term_high
    )
    return np.maximum(var, 0.0)
