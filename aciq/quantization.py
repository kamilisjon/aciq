"""Phase 3: Optimal clipping threshold derivation.

Implements the MSE cost function from Eq. (4) of the thesis plan:

    L(α) = ∫_{|x|>α} (x − α·sgn(x))² p(x) dx  +  (1/3) · α² / 4^b

and finds α* = argmin L(α) via Golden Section Search (scipy minimize_scalar).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from aciq.distributions import Distribution, Gaussian, Laplace


def clipping_mse(dist: Gaussian | Laplace, alpha: float) -> float:
    """Clipping distortion D_clip(α): squared error in the saturation regions.

    D_clip(α) = ∫_{x<-α} (x+α)² p(x) dx  +  ∫_{x>α} (x-α)² p(x) dx
    """
    lower, _ = quad(lambda x: (x + alpha) ** 2 * dist.pdf_at(np.asarray(x)),
                    -np.inf, -alpha, limit=100)
    upper, _ = quad(lambda x: (x - alpha) ** 2 * dist.pdf_at(np.asarray(x)),
                    alpha, np.inf, limit=100)
    return float(lower + upper)


def granular_mse(alpha: float, bits: int) -> float:
    """Granular distortion under Bennett's approximation (Eq. 3–4).

    D_gran = (1/3) · (α / 2^b)² · (2^b)² / (2^b)²  =  α² / (3 · 4^b)
    """
    # TODO: investigate what this Bennett`s approximation means. Perhaps we may use integrals directly as in 2.1 section of the plan?
    return alpha ** 2 / (3.0 * 4 ** bits)


def total_mse(dist: Gaussian | Laplace, alpha: float, bits: int) -> float:
    """Total quantization MSE: L(α) = D_clip(α) + D_gran(α)."""
    return clipping_mse(dist, alpha) + granular_mse(alpha, bits)


def optimal_alpha(dist: Gaussian | Laplace, bits: int) -> tuple[float, float]:
    """Find α* = argmin L(α) via Golden Section Search.

    Returns (alpha_star, mse_star).
    """
    alpha_max = max(abs(dist.minimum), abs(dist.maximum))
    result = minimize_scalar(
        lambda a: total_mse(dist, a, bits),
        bounds=(1e-8, alpha_max * 1.5),
        method="bounded",
    )
    return float(result.x), float(result.fun)


def minmax_alpha(dist: Distribution) -> float:
    # TODO: should we focus on asymetric quantization?
    #        https://arxiv.org/pdf/2004.09602
    #        https://arxiv.org/pdf/2103.13630
    """Naive min-max clipping threshold: α = max(|min|, |max|)."""
    return float(max(abs(dist.minimum), abs(dist.maximum)))
