import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms._presets import ImageClassification

from aciq.datasets.imagenet import load_and_preprocess


class TestPreprocess(unittest.TestCase):
  def test_matches_torchvision_image_classification(self):
    """Hand-rolled preprocess must match torchvision's ImageClassification(crop_size=224)."""
    np.random.seed(0)
    tmpdir = Path(tempfile.mkdtemp())
    paths: list[Path] = []
    for i, (w, h) in enumerate([(300, 400), (500, 500), (800, 600), (224, 224)]):
      arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
      p = tmpdir / f"{i}.jpg"
      Image.fromarray(arr).save(p, "JPEG", quality=95)
      paths.append(p)

    tv_preset = ImageClassification(crop_size=224)
    ref = np.stack([tv_preset(Image.open(p).convert("RGB")).numpy() for p in paths])

    ours = load_and_preprocess(paths).numpy()

    np.testing.assert_allclose(ours, ref, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
