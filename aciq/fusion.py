from __future__ import annotations

from tinygrad import Tensor
from tinygrad.nn import BatchNorm, Conv2d


def fuse_conv_bn(conv: Conv2d, bn: BatchNorm) -> tuple[Tensor, Tensor]:
  scale = bn.weight / (bn.running_var + bn.eps).sqrt()
  fused_w = conv.weight * scale.reshape(-1, 1, 1, 1)
  conv_b = conv.bias if conv.bias is not None else Tensor.zeros(conv.weight.shape[0])
  fused_b = scale * (conv_b - bn.running_mean) + bn.bias
  return fused_w, fused_b


def fuse_inplace(conv: Conv2d, bn: BatchNorm) -> None:
  conv.weight, conv.bias = fuse_conv_bn(conv, bn)
