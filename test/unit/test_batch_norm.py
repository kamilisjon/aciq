import unittest

import torch.nn as nn
import torchvision

from aciq.batch_norm import collect_conv_bn_pairs


class TestCollectConvBnPairs(unittest.TestCase):
  def test_pair_names_match(self):
    model = torchvision.models.resnet18(weights=None)
    pairs = collect_conv_bn_pairs(model)
    for conv_name, conv, bn_name, bn in pairs:
      self.assertIsInstance(conv, nn.Conv2d)
      self.assertIsInstance(bn, nn.BatchNorm2d)
      self.assertEqual(conv.out_channels, bn.num_features)

  def test_mismatch_raises(self):
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 16, 3), nn.BatchNorm2d(16))
    with self.assertRaises(AssertionError):
      collect_conv_bn_pairs(model)


if __name__ == "__main__":
  unittest.main()
