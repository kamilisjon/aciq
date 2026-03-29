import unittest

import numpy as np
import onnx
import onnx.numpy_helper
import torch
import torch.nn as nn
import torchvision

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
  def _export_conv_params(self, model: nn.Module, path: str, do_constant_folding: bool) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Export to ONNX and return (weight, bias) per Conv node in graph order."""
    model.eval()
    dummy = (torch.randn(1, 3, 32, 32),)
    torch.onnx.export(model, dummy, path, opset_version=18, do_constant_folding=do_constant_folding)
    onnx_model = onnx.load(path)
    tensors = {t.name: onnx.numpy_helper.to_array(t) for t in onnx_model.graph.initializer}
    params: list[tuple[np.ndarray, np.ndarray | None]] = []
    for n in onnx_model.graph.node:
      if n.op_type == "Conv":
        weight = tensors[n.input[1]]
        # ResNet Conv layers have bias=False, so unfused ONNX has 2 inputs.
        # After BN constant folding, ONNX fuses BN into Conv and adds a bias (3 inputs).
        bias = tensors.get(n.input[2]) if len(n.input) > 2 else None
        params.append((weight, bias))
    return params

  def test_fused_weights_and_bias_match_onnx(self):
    """Manual BN fusion must match ONNX constant-folded Conv weights and biases."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    onnx_fused = self._export_conv_params(model, "/tmp/test_bn_fused.onnx", do_constant_folding=True)
    onnx_unfused = self._export_conv_params(model, "/tmp/test_bn_unfused.onnx", do_constant_folding=False)

    pairs = collect_conv_bn_pairs(model)
    self.assertEqual(len(pairs), len(onnx_fused))

    for i, (conv_name, conv, bn_name, bn) in enumerate(pairs):
      pre_weight = conv.weight.data.numpy()
      pre_bias = conv.bias.data.numpy() if conv.bias is not None else None
      manual_fused_weight = fuse_bn_into_conv(pre_weight, bn)
      manual_fused_bias = fuse_bn_into_bias(pre_bias, bn)

      onnx_fused_weight, onnx_fused_bias = onnx_fused[i]
      onnx_unfused_weight, onnx_unfused_bias = onnx_unfused[i]

      assert onnx_unfused_bias is None, f"Expected unfused ONNX Conv to have no bias at {conv_name}"
      assert onnx_fused_bias is not None, f"Expected fused ONNX Conv to have bias at {conv_name}"
      np.testing.assert_allclose(onnx_unfused_weight, pre_weight, atol=1e-7, err_msg=f"Unfused weight mismatch at {conv_name}")
      np.testing.assert_allclose(manual_fused_weight, onnx_fused_weight, atol=1e-7, err_msg=f"Fused weight mismatch at {conv_name}")
      np.testing.assert_allclose(manual_fused_bias, onnx_fused_bias, atol=1e-7, err_msg=f"Fused bias mismatch at {conv_name}")


if __name__ == "__main__":
  unittest.main()
