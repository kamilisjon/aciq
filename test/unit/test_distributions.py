import statistics
import unittest

import numpy as np
import pytest
from scipy import stats

from aciq.distributions import Distribution, DistributionType, Gaussian, Laplace, StudentT
from test.helpers import make_gaussian_data, make_laplace_data, make_student_t_data, make_nonpositive_kurtosis_data


GAUSSIAN_TEST_MU_SIGMA: list[tuple[float, float]] = [(-3.0, 0.1), (0.0, 1.0), (100.0, 50.0)]
LAPLACE_TEST_MU_B: list[tuple[float, float]] = GAUSSIAN_TEST_MU_SIGMA
STUDENT_T_TEST_DF_LOC_SCALE: list[tuple[float, float, float]] = [(5.0, 0.0, 1.0), (10.0, -3.0, 0.5), (3.0, 100.0, 50.0)]


class TestDistributionStatistics(unittest.TestCase):
  def test_mean(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._mean(data), statistics.mean(data.tolist()))

  def test_variance(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._variance(data), statistics.variance(data.tolist()))

  def test_stdev(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._stdev(data), statistics.stdev(data.tolist()))

  def test_median(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._median(data), statistics.median(data.tolist()))

  def test_min(self):
    data = make_gaussian_data()
    assert Distribution._min(data) == min(data.tolist())

  def test_max(self):
    data = make_gaussian_data()
    assert Distribution._max(data) == max(data.tolist())

  def test_n(self):
    data = make_gaussian_data()
    assert Distribution._n(data) == len(data.tolist())


class TestDistributionCachedProperties(unittest.TestCase):
  def test_mean(self):
    data = make_gaussian_data()
    assert Distribution(data).mean == Distribution._mean(data)

  def test_variance(self):
    data = make_gaussian_data()
    assert Distribution(data).variance == Distribution._variance(data)

  def test_std(self):
    data = make_gaussian_data()
    assert Distribution(data).stdev == Distribution._stdev(data)

  def test_median(self):
    data = make_gaussian_data()
    assert Distribution(data).median == Distribution._median(data)

  def test_min(self):
    data = make_gaussian_data()
    assert Distribution(data).min == Distribution._min(data)

  def test_max(self):
    data = make_gaussian_data()
    assert Distribution(data).max == Distribution._max(data)

  def test_n(self):
    data = make_gaussian_data()
    assert Distribution(data).n == len(data.tolist())


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
    assert Distribution._kurtosis(data) <= 0
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


class TestDistributionFit(unittest.TestCase):
  def test_fit_gaussian_returns_gaussian(self):
    data = make_gaussian_data()
    assert isinstance(Distribution(data).fit(DistributionType.GAUSSIAN), Gaussian)

  def test_fit_laplace_returns_laplace(self):
    data = make_gaussian_data()
    assert isinstance(Distribution(data).fit(DistributionType.LAPLACE), Laplace)

  def test_fit_student_t_returns_student_t(self):
    data = make_gaussian_data()
    assert isinstance(Distribution(data).fit(DistributionType.STUDENT_T), StudentT)

  def test_fit_gaussian_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution(data).fit(DistributionType.GAUSSIAN)
    assert isinstance(fitted, Gaussian)
    direct = Gaussian(data)
    assert fitted.mu == direct.mu
    assert fitted.sigma == direct.sigma

  def test_fit_laplace_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution(data).fit(DistributionType.LAPLACE)
    assert isinstance(fitted, Laplace)
    direct = Laplace(data)
    assert fitted.mu == direct.mu
    assert fitted.b == direct.b

  def test_fit_student_t_matches_direct(self):
    data = make_gaussian_data()
    fitted = Distribution(data).fit(DistributionType.STUDENT_T)
    assert isinstance(fitted, StudentT)
    direct = StudentT(data)
    assert fitted.df == direct.df
    assert fitted.loc == direct.loc
    assert fitted.scale == direct.scale

  def test_all_distribution_types_have_fit(self):
    d = Distribution(make_gaussian_data())
    for dist_type in DistributionType:
      d.fit(dist_type)

  def test_fit_raises_for_unsupported_type(self):
    with pytest.raises(ValueError):
      Distribution(make_gaussian_data()).fit("unsupported")


class TestCaching(unittest.TestCase):
  def test_distribution_mean_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.mean is d.mean

  def test_distribution_variance_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.variance is d.variance

  def test_distribution_std_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.stdev is d.stdev

  def test_distribution_skewness_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.skewness is d.skewness

  def test_distribution_kurtosis_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.kurtosis is d.kurtosis

  def test_distribution_min_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.min is d.min

  def test_distribution_max_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.max is d.max

  def test_distribution_median_is_cached(self):
    d = Distribution(make_gaussian_data())
    assert d.median is d.median

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

  def test_fit_is_cached(self):
    data = make_gaussian_data()
    d = Distribution(data)
    assert d.fit(DistributionType.GAUSSIAN) is d.fit(DistributionType.GAUSSIAN)


if __name__ == "__main__":
  unittest.main()
