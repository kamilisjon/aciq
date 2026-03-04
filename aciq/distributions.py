from __future__ import annotations
import functools

import numpy as np
from scipy import stats

Q1_PERCENTILE: int = 25
Q3_PERCENTILE: int = 75


class Distribution:
    def __init__(self, data: np.ndarray):
        self._data = data

    @functools.cached_property
    def mean(self) -> float: return float(np.mean(self._data))

    @functools.cached_property
    def variance(self) -> float: return float(np.var(self._data, ddof=1))

    @functools.cached_property
    def std(self) -> float: return float(np.std(self._data, ddof=1))

    @functools.cached_property
    def skewness(self) -> float: return float(stats.skew(self._data))

    #TODO: What does kurtosis mean? How it is calculated?
    @functools.cached_property
    def kurtosis(self) -> float: return float(stats.kurtosis(self._data))

    @functools.cached_property
    def minimum(self) -> float: return float(self._data.min())

    @functools.cached_property
    def maximum(self) -> float: return float(self._data.max())

    @functools.cached_property
    def median(self) -> float: return float(np.median(self._data))

    @functools.cached_property
    def q1(self) -> float: return float(np.percentile(self._data, Q1_PERCENTILE))

    @functools.cached_property
    def q3(self) -> float: return float(np.percentile(self._data, Q3_PERCENTILE))

    @functools.cached_property
    def n(self) -> float: return len(self._data)


class Gaussian(Distribution):
    def __init__(self, data):
        super().__init__(data)

    @property
    def mu(self) -> float: return self.mean

    # TODO: why cant I directly use self.std? Why does ddof differ?
    @functools.cached_property
    def sigma(self) -> float: return float(np.std(self._data, ddof=0))

    def pdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma

    # TODO: make np.log(self.pdf()) pass scipy tests
    def logpdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return -z**2 / 2.0 - np.log(np.sqrt(2.0 * np.pi)) - np.log(self.sigma)

    @functools.cached_property
    def log_likelihood(self) -> float:
        return float(np.sum(self.logpdf()))


class Laplace(Distribution):
    def __init__(self, data):
        super().__init__(data)

    @property
    def mu(self) -> float: return self.median

    @functools.cached_property
    def b(self) -> float: return float(np.mean(np.abs(self._data - self.mu)))

    def pdf(self) -> np.ndarray:
        return np.exp(-np.abs(self._data - self.mu) / self.b) / (2.0 * self.b)

    # TODO: make np.log(self.pdf()) pass scipy tests
    def logpdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.b
        return np.log(0.5 * np.exp(-np.abs(z))) - np.log(self.b)

    @functools.cached_property
    def log_likelihood(self) -> float:
        return float(np.sum(self.logpdf()))
