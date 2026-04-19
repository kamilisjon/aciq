import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tinygrad import Tensor
from torchvision.transforms._presets import ImageClassification

from aciq.models.resnet import ResNet18
from aciq.preprocess import load_and_preprocess


RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"


class TestEndToEnd(unittest.TestCase):
  def test_preprocess_and_inference_matches_torch(self):
    """Full pipeline (preprocess + pretrained ResNet18 inference) must match torch+torchvision
    on the same synthetic JPEGs."""
    np.random.seed(0)
    tmpdir = Path(tempfile.mkdtemp())
    paths: list[Path] = []
    for i, (w, h) in enumerate([(300, 400), (500, 500), (800, 600), (224, 224)]):
      arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
      p = tmpdir / f"{i}.jpg"
      Image.fromarray(arr).save(p, "JPEG", quality=95)
      paths.append(p)

    # torch pipeline
    state = torch.hub.load_state_dict_from_url(RESNET18_URL, map_location="cpu", progress=False)
    tv = torchvision.models.resnet18(weights=None)
    tv.load_state_dict(state, strict=False)
    tv.eval()
    tv_preset = ImageClassification(crop_size=224)
    tv_batch = torch.stack([tv_preset(Image.open(p).convert("RGB")) for p in paths])
    with torch.no_grad():
      tv_logits = tv(tv_batch).numpy()

    # tinygrad pipeline
    tg = ResNet18()
    tg.load_from_pretrained()
    tg_logits = tg(load_and_preprocess(paths)).numpy()

    np.testing.assert_allclose(tg_logits, tv_logits, atol=1e-4, rtol=1e-4)
    self.assertTrue(np.array_equal(tg_logits.argmax(axis=1), tv_logits.argmax(axis=1)))


if __name__ == "__main__":
  unittest.main()
