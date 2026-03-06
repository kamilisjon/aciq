import unittest

import numpy as np
import pytest
from scipy import stats

from aciq.distributions import Distribution, DistributionType, Gaussian, GeneralizedGaussian, Laplace, StudentT, skewness, kurtosis
from test.helpers import make_gaussian_data, make_ged_data, make_laplace_data, make_student_t_data, make_nonpositive_kurtosis_data


GAUSSIAN_TEST_MU_SIGMA: list[tuple[float, float]] = [(-3.0, 0.1), (0.0, 1.0), (100.0, 50.0)]
LAPLACE_TEST_MU_B: list[tuple[float, float]] = GAUSSIAN_TEST_MU_SIGMA
STUDENT_T_TEST_DF_LOC_SCALE: list[tuple[float, float, float]] = [(5.0, 0.0, 1.0), (10.0, -3.0, 0.5), (3.0, 100.0, 50.0)]
GED_TEST_BETA_LOC_SCALE: list[tuple[float, float, float]] = [(0.5, 0.0, 1.0), (1.5, -3.0, 0.5), (3.0, 100.0, 50.0)]


# TODO: test all distributions against R with https://rpy2.github.io/


class TestCustomStatistics(unittest.TestCase):
  def test_skewness_matches_scipy(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(skewness(data), stats.skew(data))

  def test_kurtosis_matches_scipy(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(kurtosis(data), stats.kurtosis(data))


class TestGaussian(unittest.TestCase):
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
      np.testing.assert_allclose(g.pdf(), stats.norm.pdf(data, loc=scipy_mu, scale=scipy_sigma))

  def test_logpdf_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      expected = stats.norm.logpdf(data, loc=scipy_mu, scale=scipy_sigma)
      np.testing.assert_allclose(g.logpdf(), expected)

  def test_log_likelihood_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      expected = float(np.sum(stats.norm.logpdf(data, loc=scipy_mu, scale=scipy_sigma)))
      np.testing.assert_allclose(g.log_likelihood, expected)

  def test_pdf_at_arbitrary_x_matches_scipy(self):
    for mu, sigma in GAUSSIAN_TEST_MU_SIGMA:
      data = make_gaussian_data(mu=mu, sigma=sigma)
      g = Gaussian(data)
      x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
      scipy_mu, scipy_sigma = stats.norm.fit(data)
      np.testing.assert_allclose(g.pdf_at(x), stats.norm.pdf(x, loc=scipy_mu, scale=scipy_sigma))


class TestLaplace(unittest.TestCase):
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
      np.testing.assert_allclose(g.pdf(), stats.laplace.pdf(data, loc=scipy_mu, scale=scipy_b))

  def test_logpdf_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      expected = stats.laplace.logpdf(data, loc=scipy_mu, scale=scipy_b)
      np.testing.assert_allclose(g.logpdf(), expected)

  def test_log_likelihood_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      expected = np.sum(stats.laplace.logpdf(data, loc=scipy_mu, scale=scipy_b))
      np.testing.assert_allclose(g.log_likelihood, expected)

  def test_pdf_at_arbitrary_x_matches_scipy(self):
    for mu, b in LAPLACE_TEST_MU_B:
      data = make_laplace_data(mu=mu, b=b)
      g = Laplace(data)
      x = np.linspace(mu - 3 * b, mu + 3 * b, 100)
      scipy_mu, scipy_b = stats.laplace.fit(data)
      np.testing.assert_allclose(g.pdf_at(x), stats.laplace.pdf(x, loc=scipy_mu, scale=scipy_b))


class TestStudentT(unittest.TestCase):
  def test_fit_matches_scipy(self):
    # Only df > 4 cases: StudentT fit uses kurtosis which is only finite for df > 4
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      scipy_df, scipy_loc, scipy_scale = stats.t.fit(data)
      np.testing.assert_allclose(t.df, scipy_df)
      np.testing.assert_allclose(t.loc, scipy_loc)
      np.testing.assert_allclose(t.scale, scipy_scale)

  def test_logpdf_formula_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      expected = stats.t.logpdf(data, t.df, loc=t.loc, scale=t.scale)
      np.testing.assert_allclose(t.logpdf(), expected)

  def test_pdf_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      expected = stats.t.pdf(data, t.df, loc=t.loc, scale=t.scale)
      np.testing.assert_allclose(t.pdf(), expected)

  def test_log_likelihood_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      expected = np.sum(stats.t.logpdf(data, t.df, loc=t.loc, scale=t.scale))
      np.testing.assert_allclose(t.log_likelihood, expected)

  def test_df_is_inf_when_kurtosis_nonpositive(self):
    data = make_nonpositive_kurtosis_data()
    assert kurtosis(data) <= 0
    assert StudentT(data).df == float("inf")
    # Test if does not fail with inf df
    StudentT(data).scale
    StudentT(data).pdf()
    StudentT(data).log_likelihood

  def test_pdf_at_arbitrary_x_matches_scipy(self):
    for df, loc, scale in STUDENT_T_TEST_DF_LOC_SCALE:
      data = make_student_t_data(df=df, loc=loc, scale=scale)
      t = StudentT(data)
      x = np.linspace(loc - 3 * scale, loc + 3 * scale, 100)
      expected = stats.t.pdf(x, t.df, loc=t.loc, scale=t.scale)
      np.testing.assert_allclose(t.pdf_at(x), expected)


class TestGeneralizedGaussian(unittest.TestCase):
  def test_fit_matches_scipy(self):
    for beta, loc, scale in GED_TEST_BETA_LOC_SCALE:
      data = make_ged_data(beta=beta, loc=loc, scale=scale)
      g = GeneralizedGaussian(data)
      scipy_beta, scipy_loc, scipy_scale = stats.gennorm.fit(data)
      np.testing.assert_allclose(g.beta, scipy_beta)
      np.testing.assert_allclose(g.loc, scipy_loc)
      np.testing.assert_allclose(g.scale, scipy_scale)

  def test_logpdf_formula_matches_scipy(self):
    for beta, loc, scale in GED_TEST_BETA_LOC_SCALE:
      data = make_ged_data(beta=beta, loc=loc, scale=scale)
      g = GeneralizedGaussian(data)
      expected = stats.gennorm.logpdf(data, g.beta, loc=g.loc, scale=g.scale)
      np.testing.assert_allclose(g.logpdf(), expected)

  def test_pdf_matches_scipy(self):
    for beta, loc, scale in GED_TEST_BETA_LOC_SCALE:
      data = make_ged_data(beta=beta, loc=loc, scale=scale)
      g = GeneralizedGaussian(data)
      expected = stats.gennorm.pdf(data, g.beta, loc=g.loc, scale=g.scale)
      np.testing.assert_allclose(g.pdf(), expected)

  def test_log_likelihood_matches_scipy(self):
    for beta, loc, scale in GED_TEST_BETA_LOC_SCALE:
      data = make_ged_data(beta=beta, loc=loc, scale=scale)
      g = GeneralizedGaussian(data)
      expected = np.sum(stats.gennorm.logpdf(data, g.beta, loc=g.loc, scale=g.scale))
      np.testing.assert_allclose(g.log_likelihood, expected)

  def test_pdf_at_arbitrary_x_matches_scipy(self):
    for beta, loc, scale in GED_TEST_BETA_LOC_SCALE:
      data = make_ged_data(beta=beta, loc=loc, scale=scale)
      g = GeneralizedGaussian(data)
      x = np.linspace(loc - 3 * scale, loc + 3 * scale, 100)
      expected = stats.gennorm.pdf(x, g.beta, loc=g.loc, scale=g.scale)
      np.testing.assert_allclose(g.pdf_at(x), expected)


class TestDistributionFit(unittest.TestCase):
  def test_fit_gaussian_returns_gaussian(self):
    data = make_gaussian_data()
    assert isinstance(Distribution.fit(data, DistributionType.GAUSSIAN), Gaussian)

  def test_fit_laplace_returns_laplace(self):
    data = make_gaussian_data()
    assert isinstance(Distribution.fit(data, DistributionType.LAPLACE), Laplace)

  def test_fit_student_t_returns_student_t(self):
    data = make_gaussian_data()
    assert isinstance(Distribution.fit(data, DistributionType.STUDENT_T), StudentT)

  def test_fit_gaussian_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution.fit(data, DistributionType.GAUSSIAN)
    assert isinstance(fitted, Gaussian)
    direct = Gaussian(data)
    assert fitted.mu == direct.mu
    assert fitted.sigma == direct.sigma

  def test_fit_laplace_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution.fit(data, DistributionType.LAPLACE)
    assert isinstance(fitted, Laplace)
    direct = Laplace(data)
    assert fitted.mu == direct.mu
    assert fitted.b == direct.b

  def test_fit_student_t_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution.fit(data, DistributionType.STUDENT_T)
    assert isinstance(fitted, StudentT)
    direct = StudentT(data)
    assert fitted.df == direct.df
    assert fitted.loc == direct.loc
    assert fitted.scale == direct.scale

  def test_all_distribution_types_have_fit(self):
    data = make_gaussian_data()
    for dist_type in DistributionType:
      Distribution.fit(data, dist_type)

  def test_fit_generalized_gaussian_returns_generalized_gaussian(self):
    data = make_gaussian_data()
    assert isinstance(Distribution.fit(data, DistributionType.GENERALIZED_GAUSSIAN), GeneralizedGaussian)

  def test_fit_generalized_gaussian_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution.fit(data, DistributionType.GENERALIZED_GAUSSIAN)
    assert isinstance(fitted, GeneralizedGaussian)
    direct = GeneralizedGaussian(data)
    assert fitted.beta == direct.beta
    assert fitted.loc == direct.loc
    assert fitted.scale == direct.scale

  def test_fit_raises_for_unsupported_type(self):
    with pytest.raises(ValueError):
      Distribution.fit(make_gaussian_data(), "unsupported")  # type: ignore[arg-type]


class TestCaching(unittest.TestCase):
  def test_gaussian_sigma_is_cached(self):
    g = Gaussian(make_gaussian_data())
    assert g.sigma is g.sigma

  def test_gaussian_log_likelihood_is_cached(self):
    g = Gaussian(make_gaussian_data())
    assert g.log_likelihood is g.log_likelihood

  def test_laplace_b_is_cached(self):
    g = Laplace(make_gaussian_data())
    assert g.b is g.b

  def test_laplace_log_likelihood_is_cached(self):
    g = Laplace(make_gaussian_data())
    assert g.log_likelihood is g.log_likelihood

  def test_student_t_df_is_cached(self):
    t = StudentT(make_gaussian_data())
    assert t.df is t.df

  def test_student_t_scale_is_cached(self):
    t = StudentT(make_gaussian_data())
    assert t.scale is t.scale

  def test_student_t_log_likelihood_is_cached(self):
    t = StudentT(make_gaussian_data())
    assert t.log_likelihood is t.log_likelihood

  def test_generalized_gaussian_beta_is_cached(self):
    g = GeneralizedGaussian(make_gaussian_data())
    assert g.beta is g.beta

  def test_generalized_gaussian_scale_is_cached(self):
    g = GeneralizedGaussian(make_gaussian_data())
    assert g.scale is g.scale

  def test_generalized_gaussian_log_likelihood_is_cached(self):
    g = GeneralizedGaussian(make_gaussian_data())
    assert g.log_likelihood is g.log_likelihood


if __name__ == "__main__":
  unittest.main()
