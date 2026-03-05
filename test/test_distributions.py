import numpy as np
from scipy import stats

from aciq.distributions import Gaussian, Laplace, StudentT

SEED: int = 42
GAUSSIAN_TEST_MU_SIGMA: list[tuple[float, float]] = [(-3.0, 0.1), (0.0, 1.0), (100.0, 50.0)]
LAPLACE_TEST_MU_B: list[tuple[float, float]] = GAUSSIAN_TEST_MU_SIGMA
STUDENT_T_TEST_DF_LOC_SCALE: list[tuple[float, float, float]] = [(5.0, 0.0, 1.0), (10.0, -3.0, 0.5), (3.0, 100.0, 50.0)]
DISTRIBUTION_SAMPLE_SIZE: int = 50_000
RELATIVE_TOLERANCE = 1e-10


def make_gaussian_data(mu: float, sigma: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
  return np.random.default_rng(seed).normal(loc=mu, scale=sigma, size=n)


def make_laplace_data(mu: float, b: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
  return np.random.default_rng(seed).laplace(loc=mu, scale=b, size=n)


def make_student_t_data(df: float, loc: float, scale: float, n: int = DISTRIBUTION_SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.standard_t(df, size=n) * scale + loc


class TestGaussian:
  def test_fit_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma, n=DISTRIBUTION_SAMPLE_SIZE)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      np.testing.assert_equal(g.mu, scipy_mu)
      np.testing.assert_equal(g.sigma, scipy_sigma)

  def test_pdf_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma, n=DISTRIBUTION_SAMPLE_SIZE)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      np.testing.assert_allclose(g.pdf(), stats.norm.pdf(data, loc=scipy_mu, scale=scipy_sigma), rtol=RELATIVE_TOLERANCE)

  def test_logpdf_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma, n=DISTRIBUTION_SAMPLE_SIZE)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      expected = stats.norm.logpdf(data, loc=scipy_mu, scale=scipy_sigma)
      np.testing.assert_allclose(g.logpdf(), expected, rtol=RELATIVE_TOLERANCE)


class TestLaplace:
  def test_fit_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b, n=DISTRIBUTION_SAMPLE_SIZE)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      np.testing.assert_equal(g.mu, scipy_mu)
      np.testing.assert_equal(g.b, scipy_b)

  def test_pdf_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b, n=DISTRIBUTION_SAMPLE_SIZE)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      np.testing.assert_allclose(g.pdf(), stats.laplace.pdf(data, loc=scipy_mu, scale=scipy_b), rtol=RELATIVE_TOLERANCE)

  def test_logpdf_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b, n=DISTRIBUTION_SAMPLE_SIZE)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      expected = stats.laplace.logpdf(data, loc=scipy_mu, scale=scipy_b)
      np.testing.assert_allclose(g.logpdf(), expected, rtol=RELATIVE_TOLERANCE)


class TestStudentT:
  def test_logpdf_formula_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      expected = stats.t.logpdf(data, t.df, loc=t.loc, scale=t.scale)
      np.testing.assert_allclose(t.logpdf(), expected, rtol=RELATIVE_TOLERANCE)

  def test_pdf_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      expected = stats.t.pdf(data, t.df, loc=t.loc, scale=t.scale)
      np.testing.assert_allclose(t.pdf(), expected, rtol=RELATIVE_TOLERANCE)
