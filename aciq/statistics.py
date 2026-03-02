from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import stats


#TODO: What does kurtosis mean? How it is calculated?

@dataclass
class Moments:
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    n: int

    @classmethod
    def from_array(cls, vec: np.ndarray) -> "Moments":
        return cls(
            mean=float(np.mean(vec)),
            variance=float(np.var(vec, ddof=1)),
            std=float(np.std(vec, ddof=1)),
            skewness=float(stats.skew(vec)),
            kurtosis=float(stats.kurtosis(vec)),
            minimum=float(vec.min()),
            maximum=float(vec.max()),
            q1=float(np.percentile(vec, 25)),
            q3=float(np.percentile(vec, 75)),
            n=len(vec),
        )

class Distribution(StrEnum):
    GAUSSIAN = "norm"
    LAPLACE = "laplace"
    STUDENT_T = 't'

@dataclass
class DistributionFit:
    distribution: Distribution
    ks_statistic: float
    ks_pvalue: float

def fit_distribution(data: np.ndarray, dist: Distribution) -> DistributionFit:
    match dist:
        case Distribution.GAUSSIAN:
            params = stats.norm.fit(data)
            # ll = np.sum(stats.norm.logpdf(data, *params))
        case Distribution.LAPLACE:
            params = stats.laplace.fit(data)
            # ll = np.sum(stats.laplace.logpdf(data, *params))
        case Distribution.STUDENT_T:
            params = stats.t.fit(data)
            # ll = np.sum(stats.t.logpdf(data, *params))
    ks_stat, ks_p = stats.kstest(data, dist, args=params)
    return DistributionFit(dist, float(ks_stat), float(ks_p))