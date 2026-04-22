import unittest

import numpy as np
from tinygrad import Tensor

from aciq.models.resnet import ResNet


class TestResNetFuse(unittest.TestCase):
  model: ResNet
  x_np: np.ndarray
  vanilla_out: np.ndarray

  @classmethod
  def setUpClass(cls):
    cls.model = ResNet(18)
    cls.model.load_from_pretrained()
    np.random.seed(0)
    cls.x_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    cls.vanilla_out = cls.model(Tensor(cls.x_np)).numpy()

  def test_fused_forward_matches_vanilla(self):
    """Forward output before and after `fuse()` must match to fp32 tolerance."""
    model = self.model
    model.fuse()
    fused_out = model(Tensor(self.x_np)).numpy()
    np.testing.assert_allclose(fused_out, self.vanilla_out, atol=1e-4, rtol=1e-4)
    self.assertTrue(np.array_equal(fused_out.argmax(axis=1), self.vanilla_out.argmax(axis=1)))

  def test_fused_flag_propagates(self):
    """After `model.fuse()`, every block plus the ResNet itself must have `fused=True`."""
    model = ResNet(18)
    model.load_from_pretrained()
    self.assertFalse(model.fused)
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
      for block in layer:
        self.assertFalse(block.fused)
    model.fuse()
    self.assertTrue(model.fused)
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
      for block in layer:
        self.assertTrue(block.fused)


if __name__ == "__main__":
  unittest.main()
