import numpy as np

SEED: int = 42
DISTRIBUTION_SAMPLE_SIZE: int = 50_000
RELATIVE_TOLERANCE = 1e-10


def make_gaussian_data(mu: float = 3.0, sigma: float = 1.5, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
  return np.random.default_rng(seed).normal(loc=mu, scale=sigma, size=n)


def make_laplace_data(mu: float, b: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
  return np.random.default_rng(seed).laplace(loc=mu, scale=b, size=n)


def make_student_t_data(df: float, loc: float, scale: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.standard_t(df, size=n) * scale + loc
