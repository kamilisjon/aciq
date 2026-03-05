import numpy as np

DISTRIBUTION_SAMPLE_SIZE: int = 100_000


def make_gaussian_data(mu: float = 3.0, sigma: float = 1.5, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int | None = None) -> np.ndarray:
  return np.random.default_rng(seed).normal(loc=mu, scale=sigma, size=n)


def make_nonpositive_kurtosis_data(n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int | None = None) -> np.ndarray:
  # Uniform distribution has excess kurtosis = -1.2 TODO: why so?
  return np.random.default_rng(seed).uniform(size=n)


def make_laplace_data(mu: float, b: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int | None = None) -> np.ndarray:
  return np.random.default_rng(seed).laplace(loc=mu, scale=b, size=n)


def make_student_t_data(df: float, loc: float, scale: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int | None = None) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.standard_t(df, size=n) * scale + loc
