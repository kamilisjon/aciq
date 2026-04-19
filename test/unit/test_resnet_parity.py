import unittest

import numpy as np
import torch
import torchvision
from tinygrad import Tensor

from aciq.models.resnet import ResNet


def _force_v1(tv_model: torchvision.models.ResNet) -> None:
  """Mutate each torchvision Bottleneck so the downsampling stride moves from conv2 → conv1
  (v1.5 → v1). No-op for BasicBlock blocks."""
  for layer in (tv_model.layer1, tv_model.layer2, tv_model.layer3, tv_model.layer4):
    for block in layer:
      if hasattr(block, "conv3"):  # Bottleneck has conv3; BasicBlock does not.
        s = block.conv2.stride
        block.conv1.stride = s
        block.conv2.stride = (1, 1)


class TestResNetParity(unittest.TestCase):
  def _assert_parity(self, tg_model: ResNet, tv_factory) -> None:
    tg_model.load_from_pretrained()

    state = torch.hub.load_state_dict_from_url(tg_model.url, map_location="cpu", progress=False)
    tv = tv_factory(weights=None)
    tv.load_state_dict(state, strict=False)
    tv.eval()
    _force_v1(tv)

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
