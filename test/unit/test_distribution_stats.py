import statistics

import numpy as np

from aciq.distributions import Distribution, Gaussian, Laplace, StudentT
from test.helpers import make_gaussian_data, RELATIVE_TOLERANCE


class TestDistributionStats:
  def test_mean(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._mean(data), statistics.mean(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_variance(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._variance(data), statistics.variance(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_std(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._stdev(data), statistics.stdev(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_median(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution._median(data), statistics.median(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_min(self):
    data = make_gaussian_data()
    assert Distribution._min(data) == min(data.tolist())

  def test_max(self):
    data = make_gaussian_data()
    assert Distribution._max(data) == max(data.tolist())

  def test_n(self):
    data = make_gaussian_data()
    assert Distribution._n(data) == len(data.tolist())


class TestDistributionCachedProperties:
  def test_mean(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution(data).mean, statistics.mean(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_variance(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution(data).variance, statistics.variance(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_std(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution(data).stdev, statistics.stdev(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_median(self):
    data = make_gaussian_data()
    np.testing.assert_allclose(Distribution(data).median, statistics.median(data.tolist()), rtol=RELATIVE_TOLERANCE)

  def test_min(self):
    data = make_gaussian_data()
    assert Distribution(data).min == min(data.tolist())

  def test_max(self):
    data = make_gaussian_data()
    assert Distribution(data).max == max(data.tolist())

  def test_n(self):
    data = make_gaussian_data()
    assert Distribution(data).n == len(data.tolist())


class TestCachedProperties:
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
