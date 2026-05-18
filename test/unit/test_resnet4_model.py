import unittest

import numpy as np
from tinygrad import Tensor

from aciq.models import ResNet4


class TestResNet4(unittest.TestCase):
  def test_forward_shape(self):
    m = ResNet4()
    out = m(Tensor.rand(4, 3, 32, 32))
    self.assertEqual(out.shape, (4, 10))

  def test_fuse_keeps_logits_close(self):
    m = ResNet4()
    np.random.seed(0)
    x = Tensor(np.random.randn(4, 3, 32, 32).astype(np.float32))
    pre = m(x).numpy()
    m.fuse()
    post = m(x).numpy()
    np.testing.assert_allclose(post, pre, atol=1e-4, rtol=1e-4)
    self.assertTrue(m.fused)


if __name__ == "__main__":
  unittest.main()
