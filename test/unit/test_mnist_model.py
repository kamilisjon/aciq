import unittest

import numpy as np
from tinygrad import Tensor

from aciq.mnist import MNISTModel


class TestMNISTModel(unittest.TestCase):
  def test_forward_shape(self):
    m = MNISTModel()
    out = m(Tensor.rand(4, 1, 28, 28))
    self.assertEqual(out.shape, (4, 10))

  def test_activation_keys(self):
    m = MNISTModel()
    m(Tensor.rand(2, 1, 28, 28))
    self.assertEqual(list(m.activations.keys()), ["block1", "block2", "block3", "block4", "block5"])
    for name, t in m.activations.items():
      self.assertEqual(t.shape[0], 2, f"batch dim mismatch at {name}")

  def test_fuse_keeps_logits_close(self):
    m = MNISTModel()
    np.random.seed(0)
    x = Tensor(np.random.randn(4, 1, 28, 28).astype(np.float32))
    pre = m(x).numpy()
    m.fuse()
    post = m(x).numpy()
    np.testing.assert_allclose(post, pre, atol=1e-4, rtol=1e-4)
    self.assertTrue(m.fused)


if __name__ == "__main__":
  unittest.main()
