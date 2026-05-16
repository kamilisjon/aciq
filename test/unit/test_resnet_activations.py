import unittest

import numpy as np
import torch
import torchvision
from tinygrad import Tensor

from aciq.models.resnet import ResNet


RESNET18_EXPECTED_KEYS = [
  "stem",
  *[f"layer{i}.{j}.activation_{k}" for i in range(1, 5) for j in range(2) for k in (1, 2)],
]


class TestResNetActivations(unittest.TestCase):
  model: ResNet
  x_np: np.ndarray

  @classmethod
  def setUpClass(cls):
    cls.model = ResNet(18)
    cls.model.load_from_pretrained()
    np.random.seed(0)
    cls.x_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    _ = cls.model(Tensor(cls.x_np))  # populate activations

  def test_activation_keys_and_shapes_resnet18(self):
    acts = self.model.activations
    self.assertEqual(list(acts.keys()), RESNET18_EXPECTED_KEYS)
    self.assertEqual(len(acts), 17)
    expected_channels = {"stem": 64}
    for i, c in [(1, 64), (2, 128), (3, 256), (4, 512)]:
      for j in range(2):
        for k in (1, 2):
          expected_channels[f"layer{i}.{j}.activation_{k}"] = c
    for name, t in acts.items():
      self.assertEqual(t.shape[0], 2, f"batch dim mismatch at {name}")
      self.assertEqual(t.shape[1], expected_channels[name], f"channel dim mismatch at {name}")

  def test_activation_values_match_torch_relu_hooks(self):
    state = torch.hub.load_state_dict_from_url(self.model.url, map_location="cpu", progress=False)
    tv = torchvision.models.resnet18(weights=None)
    tv.load_state_dict(state, strict=False)
    tv.eval()

    torch_taps: dict[str, np.ndarray] = {}
    block_calls: dict[tuple[int, int], list[np.ndarray]] = {}

    def stem_hook(_m, _i, out):
      torch_taps["stem"] = out.detach().numpy().copy()

    tv.relu.register_forward_hook(stem_hook)

    for i, layer in enumerate((tv.layer1, tv.layer2, tv.layer3, tv.layer4), 1):
      for j, block in enumerate(layer):
        bucket: list[np.ndarray] = []
        block_calls[(i, j)] = bucket

        def make_hook(b):
          def hook(_m, _i, out):
            b.append(out.detach().numpy().copy())

          return hook

        block.relu.register_forward_hook(make_hook(bucket))

    with torch.no_grad():
      tv(torch.from_numpy(self.x_np))

    for (i, j), calls in block_calls.items():
      self.assertEqual(len(calls), 2, f"expected 2 ReLU calls per BasicBlock at layer{i}.{j}")
      torch_taps[f"layer{i}.{j}.activation_1"] = calls[0]
      torch_taps[f"layer{i}.{j}.activation_2"] = calls[1]

    acts = self.model.activations
    for name, torch_arr in torch_taps.items():
      np.testing.assert_allclose(acts[name].numpy(), torch_arr, atol=1e-4, rtol=1e-4, err_msg=f"mismatch at {name}")


if __name__ == "__main__":
  unittest.main()
