from __future__ import annotations
from functools import lru_cache

import numpy as np
from scipy import stats

Q1_PERCENTILE: int = 25
Q3_PERCENTILE: int = 75


class Distribution:
    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    @lru_cache(maxsize=None)
    def mean(self) -> float: return float(np.mean(self._data))

    @property
    @lru_cache(maxsize=None)
    def variance(self) -> float: return float(np.var(self._data, ddof=1))

    @property
    @lru_cache(maxsize=None)
    def std(self) -> float: return float(np.std(self._data, ddof=1))

    @property
    @lru_cache(maxsize=None)
    def skewness(self) -> float: return float(stats.skew(self._data))

    #TODO: What does kurtosis mean? How it is calculated?
    @property
    @lru_cache(maxsize=None)
    def kurtosis(self) -> float: return float(stats.kurtosis(self._data))

    @property
    @lru_cache(maxsize=None)
    def minimum(self) -> float: return float(self._data.min())

    @property
    @lru_cache(maxsize=None)
    def maximum(self) -> float: return float(self._data.max())

    @property
    @lru_cache(maxsize=None)
    def q1(self) -> float: return float(np.percentile(self._data, Q1_PERCENTILE))

    @property
    @lru_cache(maxsize=None)
    def q3(self) -> float: return float(np.percentile(self._data, Q3_PERCENTILE))

    @property
    @lru_cache(maxsize=None)
    def n(self) -> float: return len(self._data)


class Gaussian(Distribution):
    def __init__(self, data):
        super().__init__(data)

    @property
    @lru_cache(maxsize=None)
    def mu(self) -> float: return self.mean

    # TODO: why cant I directly use self.std? Why does ddof differ?
    @property
    @lru_cache(maxsize=None)
    def sigma(self) -> float: return float(np.std(self._data, ddof=0))

    def pdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma

    def logpdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return -z**2 / 2.0 - np.log(np.sqrt(2.0 * np.pi)) - np.log(self.sigma)

    @property
    @lru_cache(maxsize=None)
    def log_likelihood(self) -> float:
        return float(np.sum(self.logpdf()))
