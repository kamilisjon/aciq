"""Conv+BN fusion for tinygrad tensors.

Folds a trailing BatchNorm2d into the preceding Conv2d, producing the equivalent fused
(weight, bias) pair. Numerically equivalent to `torch.nn.utils.fusion.fuse_conv_bn_eval`.
"""

from __future__ import annotations

from tinygrad import Tensor
from tinygrad.nn import BatchNorm, Conv2d


def fuse_conv_bn(conv: Conv2d, bn: BatchNorm) -> tuple[Tensor, Tensor]:
  """Return the (weight, bias) that a Conv2d would need to represent `Conv2d(x); BN(·)`.

  Math: `scale = γ / √(σ² + ε); fused_w = w * scale; fused_b = scale * (b - μ) + β`.
  When `conv.bias is None`, the initial bias `b` is treated as zero.
  """
  scale = bn.weight / (bn.running_var + bn.eps).sqrt()
  fused_w = conv.weight * scale.reshape(-1, 1, 1, 1)
  conv_b = conv.bias if conv.bias is not None else Tensor.zeros(conv.weight.shape[0])
  fused_b = scale * (conv_b - bn.running_mean) + bn.bias
  return fused_w, fused_b


def fuse_inplace(conv: Conv2d, bn: BatchNorm) -> None:
  """Fold `bn` into `conv` in place: overwrites `conv.weight` and attaches `conv.bias` (even
  when the Conv was built with `bias=False`)."""
  conv.weight, conv.bias = fuse_conv_bn(conv, bn)
