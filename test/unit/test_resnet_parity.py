import unittest

import numpy as np
import torch
import torchvision
from tinygrad import Tensor

from aciq.resnet import ResNet


class TestResNetParity(unittest.TestCase):
  def _assert_parity(self, tg_model: ResNet, tv_factory) -> None:
    tg_model.load_from_pretrained()

    state = torch.hub.load_state_dict_from_url(tg_model.url, map_location="cpu", progress=False)
    tv = tv_factory(weights=None)
    tv.load_state_dict(state, strict=False)
    tv.eval()

    np.random.seed(0)
    x = np.random.randn(2, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
      tv_out = tv(torch.from_numpy(x)).numpy()
    tg_out = tg_model(Tensor(x)).numpy()

    np.testing.assert_allclose(tg_out, tv_out, atol=1e-4, rtol=1e-4)
    self.assertTrue(np.array_equal(tg_out.argmax(axis=1), tv_out.argmax(axis=1)))

  def test_resnet18(self):
    self._assert_parity(ResNet(18, num_classes=1000), torchvision.models.resnet18)

  def test_resnet34(self):
    self._assert_parity(ResNet(34, num_classes=1000), torchvision.models.resnet34)

  def test_resnet50(self):
    self._assert_parity(ResNet(50, num_classes=1000), torchvision.models.resnet50)

  def test_resnet101(self):
    self._assert_parity(ResNet(101, num_classes=1000), torchvision.models.resnet101)

  def test_resnet152(self):
    self._assert_parity(ResNet(152, num_classes=1000), torchvision.models.resnet152)


if __name__ == "__main__":
  unittest.main()
