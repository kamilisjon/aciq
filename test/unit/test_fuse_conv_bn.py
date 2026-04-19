import unittest

import numpy as np
import torch
import torch.nn as torch_nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from tinygrad import Tensor
from tinygrad.nn import BatchNorm2d as TGBatchNorm2d, Conv2d as TGConv2d

from aciq.fusion import fuse_conv_bn


def _matched_pair(bias: bool):
  torch.manual_seed(0)
  tc = torch_nn.Conv2d(3, 8, kernel_size=3, bias=bias)
  tb = torch_nn.BatchNorm2d(8)
  assert tb.running_mean is not None and tb.running_var is not None
  tb.running_mean.normal_()
  tb.running_var.abs_().add_(0.1)
  tb.weight.data.normal_()
  tb.bias.data.normal_()
  tc.eval()
  tb.eval()

  tgc = TGConv2d(3, 8, kernel_size=3, bias=bias)
  tgc.weight.assign(Tensor(tc.weight.data.numpy()))
  if bias:
    assert tc.bias is not None and tgc.bias is not None
    tgc.bias.assign(Tensor(tc.bias.data.numpy()))

  tgb = TGBatchNorm2d(8)
  assert tgb.weight is not None and tgb.bias is not None
  assert tgb.running_mean is not None and tgb.running_var is not None
  tgb.weight.assign(Tensor(tb.weight.data.numpy()))
  tgb.bias.assign(Tensor(tb.bias.data.numpy()))
  tgb.running_mean.assign(Tensor(tb.running_mean.data.numpy()))
  tgb.running_var.assign(Tensor(tb.running_var.data.numpy()))
  return tc, tb, tgc, tgb


class TestFuseConvBn(unittest.TestCase):
  def _assert_matches_torch(self, bias: bool) -> None:
    tc, tb, tgc, tgb = _matched_pair(bias)
    ref = fuse_conv_bn_eval(tc, tb)
    our_w, our_b = fuse_conv_bn(tgc, tgb)
    np.testing.assert_allclose(our_w.numpy(), ref.weight.data.numpy(), atol=1e-6)
    np.testing.assert_allclose(our_b.numpy(), ref.bias.data.numpy(), atol=1e-6)

  def test_matches_torch_fuse_conv_bn_eval_no_bias(self):
    self._assert_matches_torch(bias=False)

  def test_matches_torch_fuse_conv_bn_eval_with_bias(self):
    self._assert_matches_torch(bias=True)


if __name__ == "__main__":
  unittest.main()
