import numpy as np
from scipy import stats

from aciq.distributions import Distribution, DistributionType, Gaussian, Laplace, StudentT
from test.helpers import make_gaussian_data, make_laplace_data, make_student_t_data, RELATIVE_TOLERANCE


GAUSSIAN_TEST_MU_SIGMA: list[tuple[float, float]] = [(-3.0, 0.1), (0.0, 1.0), (100.0, 50.0)]
LAPLACE_TEST_MU_B: list[tuple[float, float]] = GAUSSIAN_TEST_MU_SIGMA
STUDENT_T_TEST_DF_LOC_SCALE: list[tuple[float, float, float]] = [(5.0, 0.0, 1.0), (10.0, -3.0, 0.5), (3.0, 100.0, 50.0)]


class TestGaussian:
  def test_fit_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      np.testing.assert_equal(g.mu, scipy_mu)
      np.testing.assert_equal(g.sigma, scipy_sigma)

  def test_pdf_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      np.testing.assert_allclose(g.pdf(), stats.norm.pdf(data, loc=scipy_mu, scale=scipy_sigma), rtol=RELATIVE_TOLERANCE)

  def test_logpdf_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      expected = stats.norm.logpdf(data, loc=scipy_mu, scale=scipy_sigma)
      np.testing.assert_allclose(g.logpdf(), expected, rtol=RELATIVE_TOLERANCE)

  def test_log_likelihood_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      expected = float(np.sum(stats.norm.logpdf(data, loc=scipy_mu, scale=scipy_sigma)))
      np.testing.assert_allclose(g.log_likelihood, expected, rtol=RELATIVE_TOLERANCE)


class TestLaplace:
  def test_fit_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      np.testing.assert_equal(g.mu, scipy_mu)
      np.testing.assert_equal(g.b, scipy_b)

  def test_pdf_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      np.testing.assert_allclose(g.pdf(), stats.laplace.pdf(data, loc=scipy_mu, scale=scipy_b), rtol=RELATIVE_TOLERANCE)

  def test_logpdf_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      expected = stats.laplace.logpdf(data, loc=scipy_mu, scale=scipy_b)
      np.testing.assert_allclose(g.logpdf(), expected, rtol=RELATIVE_TOLERANCE)

  def test_log_likelihood_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      expected = float(np.sum(stats.laplace.logpdf(data, loc=scipy_mu, scale=scipy_b)))
      np.testing.assert_allclose(g.log_likelihood, expected, rtol=RELATIVE_TOLERANCE)


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

  def test_log_likelihood_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      expected = float(np.sum(stats.t.logpdf(data, t.df, loc=t.loc, scale=t.scale)))
      np.testing.assert_allclose(t.log_likelihood, expected, rtol=RELATIVE_TOLERANCE)


class TestDistributionFit:
  def test_fit_gaussian_returns_gaussian(self):
    data = make_gaussian_data()
    assert isinstance(Distribution(data).fit(DistributionType.GAUSSIAN), Gaussian)

  def test_fit_laplace_returns_laplace(self):
    data = make_gaussian_data()
    assert isinstance(Distribution(data).fit(DistributionType.LAPLACE), Laplace)

  def test_fit_student_t_returns_student_t(self):
    data = make_gaussian_data()
    assert isinstance(Distribution(data).fit(DistributionType.STUDENT_T), StudentT)

  def test_fit_is_cached(self):
    data = make_gaussian_data()
    d = Distribution(data)
    assert d.fit(DistributionType.GAUSSIAN) is d.fit(DistributionType.GAUSSIAN)

  def test_fit_gaussian_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution(data).fit(DistributionType.GAUSSIAN)
    assert isinstance(fitted, Gaussian)
    direct = Gaussian(data)
    np.testing.assert_equal(fitted.mu, direct.mu)
    np.testing.assert_equal(fitted.sigma, direct.sigma)

  def test_fit_laplace_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution(data).fit(DistributionType.LAPLACE)
    assert isinstance(fitted, Laplace)
    direct = Laplace(data)
    np.testing.assert_equal(fitted.mu, direct.mu)
    np.testing.assert_equal(fitted.b, direct.b)

  def test_fit_student_t_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution(data).fit(DistributionType.STUDENT_T)
    assert isinstance(fitted, StudentT)
    direct = StudentT(data)
    np.testing.assert_equal(fitted.df, direct.df)
    np.testing.assert_equal(fitted.loc, direct.loc)
    np.testing.assert_equal(fitted.scale, direct.scale)
