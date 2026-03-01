from dataclasses import dataclass

import numpy as np
from scipy import stats


#TODO: What does kurtosis mean? How it is calculated?

@dataclass
class Moments:
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float  # excess kurtosis (Gaussian=0)
    min: float
    max: float
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
            min=float(vec.min()),
            max=float(vec.max()),
            q1=float(np.percentile(vec, 25)),
            q3=float(np.percentile(vec, 75)),
            n=len(vec),
        )
