import statistics

import numpy as np

from aciq.distributions import Distribution
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
    np.testing.assert_allclose(Distribution._std(data), statistics.stdev(data.tolist()), rtol=RELATIVE_TOLERANCE)

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
