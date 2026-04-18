import copy
import unittest

import numpy as np
import torch.nn as nn
import torchvision
from torch.nn.utils.fusion import fuse_conv_bn_eval

from aciq.batch_norm import collect_conv_bn_pairs, fuse_bn_into_bias, fuse_bn_into_conv


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


class TestFuseBnIntoConv(unittest.TestCase):
  def test_fused_weights_and_bias_match_torch_native(self):
    """Manual BN fusion must match torch.nn.utils.fusion.fuse_conv_bn_eval output."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    pairs = collect_conv_bn_pairs(model)

    for conv_name, conv, bn_name, bn in pairs:
      pre_weight = conv.weight.data.numpy()
      pre_bias = conv.bias.data.numpy() if conv.bias is not None else None
      manual_fused_weight = fuse_bn_into_conv(pre_weight, bn)
      manual_fused_bias = fuse_bn_into_bias(pre_bias, bn)

      torch_fused = fuse_conv_bn_eval(copy.deepcopy(conv), copy.deepcopy(bn))
      torch_fused_weight = torch_fused.weight.data.numpy()
      assert torch_fused.bias is not None, f"Expected fused Conv to have bias at {conv_name}"
      torch_fused_bias = torch_fused.bias.data.numpy()

      np.testing.assert_allclose(manual_fused_weight, torch_fused_weight, atol=1e-6, err_msg=f"Fused weight mismatch at {conv_name}")
      np.testing.assert_allclose(manual_fused_bias, torch_fused_bias, atol=1e-6, err_msg=f"Fused bias mismatch at {conv_name}")


if __name__ == "__main__":
  unittest.main()
