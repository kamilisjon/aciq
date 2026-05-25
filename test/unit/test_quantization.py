import unittest

import numpy as np

from aciq.quantization.clipping import bound_symmetric_minmax, quantize_symmetric


class TestMinmaxRange(unittest.TestCase):
  def test_symmetric_data(self):
    data = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert bound_symmetric_minmax(data) == 3.0

  def test_positive_only(self):
    data = np.array([0.5, 1.0, 2.0, 4.0])
    assert bound_symmetric_minmax(data) == 4.0

  def test_negative_only(self):
    data = np.array([-5.0, -2.0, -0.1])
    assert bound_symmetric_minmax(data) == 5.0

  def test_single_element(self):
    assert bound_symmetric_minmax(np.array([7.0])) == 7.0


class TestQuantize(unittest.TestCase):
  def test_zero_is_preserved(self):
    data = np.array([-1.0, 0.0, 1.0])
    result = quantize_symmetric(data, alpha=1.0, bits=8)
    assert result[1] == 0.0

  def test_output_shape_matches_input(self):
    data = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    result = quantize_symmetric(data, alpha=bound_symmetric_minmax(data), bits=4)
    assert result.shape == data.shape

  def test_values_within_alpha(self):
    data = np.random.default_rng(0).normal(size=10_000).astype(np.float32)
    alpha = bound_symmetric_minmax(data)
    result = quantize_symmetric(data, alpha, bits=8)
    assert np.all(np.abs(result) <= alpha)

  def test_number_of_unique_levels(self):
    data = np.random.default_rng(0).normal(size=100_000).astype(np.float32)
    alpha = bound_symmetric_minmax(data)
    result_4 = quantize_symmetric(data, alpha, bits=4)
    result_8 = quantize_symmetric(data, alpha, bits=8)
    # 4bit signed symmetric: -7..+7 = 15 levels
    assert len(np.unique(result_4)) <= 2**4 - 1
    # 8bit signed symmetric: -127..+127 = 255 levels
    assert len(np.unique(result_8)) <= 2**8 - 1


if __name__ == "__main__":
  unittest.main()
