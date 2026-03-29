import numpy as np
import torch.nn as nn


def collect_conv_bn_pairs(model: nn.Module) -> list[tuple[str, nn.Conv2d, str, nn.BatchNorm2d]]:
  convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
  bns = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
  assert len(convs) == len(bns), f"Conv/BN count mismatch: {len(convs)} vs {len(bns)}"
  return [(cn, cm, bn, bm) for (cn, cm), (bn, bm) in zip(convs, bns)]


def _bn_scale(bn: nn.BatchNorm2d) -> np.ndarray:
  assert bn.running_var is not None
  return bn.weight.data.numpy() / np.sqrt(bn.running_var.data.numpy() + bn.eps)


def fuse_bn_into_conv(conv_weight: np.ndarray, bn: nn.BatchNorm2d) -> np.ndarray:
  return conv_weight * _bn_scale(bn)[:, None, None, None]


def fuse_bn_into_bias(conv_bias: np.ndarray | None, bn: nn.BatchNorm2d) -> np.ndarray:
  assert bn.running_mean is not None
  scale = _bn_scale(bn)
  bias = conv_bias if conv_bias is not None else np.zeros_like(scale)
  return scale * (bias - bn.running_mean.data.numpy()) + bn.bias.data.numpy()
