from __future__ import annotations
from functools import cached_property

import numpy as np
from scipy import stats

Q1_PERCENTILE: int = 25
Q3_PERCENTILE: int = 75


class Distribution:
    def __init__(self, data: np.ndarray):
        self._data = data

    @cached_property
    def mean(self) -> float: return float(np.mean(self._data))

    @cached_property
    def variance(self) -> float: return float(np.var(self._data, ddof=1))

    @cached_property
    def std(self) -> float: return float(np.std(self._data, ddof=1))

    @cached_property
    def skewness(self) -> float: return float(stats.skew(self._data))

    #TODO: What does kurtosis mean? How it is calculated?
    @cached_property
    def kurtosis(self) -> float: return float(stats.kurtosis(self._data))

    @cached_property
    def minimum(self) -> float: return float(self._data.min())

    @cached_property
    def maximum(self) -> float: return float(self._data.max())

    @cached_property
    def q1(self) -> float: return float(np.percentile(self._data, Q1_PERCENTILE))

    @cached_property
    def q3(self) -> float: return float(np.percentile(self._data, Q3_PERCENTILE))

    @cached_property
    def n(self) -> float: return len(self._data)


class Gaussian(Distribution):
    def __init__(self, data):
        super().__init__(data)

    @cached_property
    def mu(self) -> float: return self.mean

    # TODO: why cant I directly use self.std? Why does ddof differ?
    @cached_property
    def sigma(self) -> float: return float(np.std(self._data, ddof=0))

    def pdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma

    def logpdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return -z**2 / 2.0 - np.log(np.sqrt(2.0 * np.pi)) - np.log(self.sigma)

    @cached_property
    def log_likelihood(self) -> float:
        return float(np.sum(self.logpdf()))
