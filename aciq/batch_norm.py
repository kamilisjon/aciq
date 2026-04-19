import torch.nn as nn


def collect_conv_bn_pairs(model: nn.Module) -> list[tuple[str, nn.Conv2d, str, nn.BatchNorm2d]]:
  convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
  bns = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
  assert len(convs) == len(bns), f"Conv/BN count mismatch: {len(convs)} vs {len(bns)}"
  return [(cn, cm, bn, bm) for (cn, cm), (bn, bm) in zip(convs, bns)]
