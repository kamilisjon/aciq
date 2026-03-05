from __future__ import annotations
import functools

import numpy as np
from scipy import stats


class Distribution:
    def __init__(self, data: np.ndarray):
        self._data = data

    @functools.cached_property
    def n(self) -> int: return self._n(self._data)

    @functools.cached_property
    def mean(self) -> float: return self._mean(self._data)

    @functools.cached_property
    def variance(self) -> float: return self._variance(self._data)

    @functools.cached_property
    def std(self) -> float: return self._std(self._data)

    @functools.cached_property
    def skewness(self) -> float: return self._skewness(self._data)

    @functools.cached_property
    def kurtosis(self) -> float: return self._kurtosis(self._data)

    @functools.cached_property
    def min(self) -> float: return self._min(self._data)

    @functools.cached_property
    def median(self) -> float: return self._median(self._data)

    @functools.cached_property
    def max(self) -> float: return self._max(self._data)

    @staticmethod
    def _mean(data: np.ndarray) -> float: return float(np.mean(data))

    @staticmethod
    def _variance(data: np.ndarray) -> float: return float(np.var(data, ddof=1))

    @staticmethod
    def _std(data: np.ndarray) -> float: return float(np.std(data, ddof=1))

    @staticmethod
    def _skewness(data: np.ndarray) -> float: return float(stats.skew(data))

    #TODO: What does kurtosis mean? How it is calculated?
    @staticmethod
    def _kurtosis(data: np.ndarray) -> float: return float(stats.kurtosis(data))

    @staticmethod
    def _min(data: np.ndarray) -> float: return float(data.min())

    @staticmethod
    def _max(data: np.ndarray) -> float: return float(data.max())

    @staticmethod
    def _median(data: np.ndarray) -> float: return float(np.median(data))

    @staticmethod
    def _n(data: np.ndarray) -> int: return len(data)

    @functools.cached_property
    def gaussian(self) -> Laplace:
        return Laplace(self._data)

    @functools.cached_property
    def laplace(self) -> Laplace:
        return Gaussian(self._data)


class Gaussian:
    def __init__(self, data):
        self._data = data

    @property
    def mu(self) -> float: return Distribution._mean(self._data)

    # TODO: why cant I directly use self.std? Why does ddof differ?
    @functools.cached_property
    def sigma(self) -> float: return float(np.std(self._data, ddof=0))

    def pdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma

    def pdf_at(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sigma
        return np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi) / self.sigma

    # TODO: make np.log(self.pdf()) pass scipy tests
    def logpdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.sigma
        return -z**2 / 2.0 - np.log(np.sqrt(2.0 * np.pi)) - np.log(self.sigma)

    @functools.cached_property
    def log_likelihood(self) -> float:
        return float(np.sum(self.logpdf()))


class Laplace:
    def __init__(self, data):
        self._data = data

    @property
    def mu(self) -> float: return Distribution._median(self._data)

    @functools.cached_property
    def b(self) -> float: return float(np.mean(np.abs(self._data - self.mu)))

    def pdf(self) -> np.ndarray:
        return np.exp(-np.abs(self._data - self.mu) / self.b) / (2.0 * self.b)

    def pdf_at(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-np.abs(x - self.mu) / self.b) / (2.0 * self.b)

    # TODO: make np.log(self.pdf()) pass scipy tests
    def logpdf(self) -> np.ndarray:
        z = (self._data - self.mu) / self.b
        return np.log(0.5 * np.exp(-np.abs(z))) - np.log(self.b)

    @functools.cached_property
    def log_likelihood(self) -> float:
        return float(np.sum(self.logpdf()))
