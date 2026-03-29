from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tinygrad.helpers import ContextVar, tqdm


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/bn_fusion_effects")
OPSET_VERSION = 18
BATCH_SIZE = ContextVar("BATCH_SIZE", 16)
IMAGE_H_W = ContextVar("IMAGE_H_W", 224)


def collect_conv_bn_pairs(model: nn.Module) -> list[tuple[str, nn.Conv2d, str, nn.BatchNorm2d]]:
  convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
  bns = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
  assert len(convs) == len(bns), f"Conv/BN count mismatch: {len(convs)} vs {len(bns)}"
  return [(cn, cm, bn, bm) for (cn, cm), (bn, bm) in zip(convs, bns)]


def fuse_bn_into_conv(conv_weight: np.ndarray, bn: nn.BatchNorm2d) -> np.ndarray:
  gamma = bn.weight.data.numpy()
  var = bn.running_var.data.numpy()
  eps = bn.eps
  scale = gamma / np.sqrt(var + eps)
  return conv_weight * scale[:, None, None, None]


def plot_channel_ranges(layer_idx: int, conv_name: str, pre_weight: np.ndarray, post_weight: np.ndarray, save_dir: Path):
  out_ch = pre_weight.shape[0]
  pre_flat = pre_weight.reshape(out_ch, -1)
  post_flat = post_weight.reshape(out_ch, -1)

  # Symmetric per-channel alpha: max(|w|) per channel -> range is [-alpha, alpha]
  pre_alpha = np.abs(pre_flat).max(axis=1)
  post_alpha = np.abs(post_flat).max(axis=1)

  # Symmetric per-tensor alpha
  pre_tensor_alpha = float(np.abs(pre_weight).max())
  post_tensor_alpha = float(np.abs(post_weight).max())

  channels = np.arange(out_ch)

  fig, ax = plt.subplots(figsize=(12, 5))

  ax.vlines(channels - 0.15, -pre_alpha, pre_alpha, colors="steelblue", linewidth=0.8, alpha=0.7, label="Before BN fusion")
  ax.vlines(channels + 0.15, -post_alpha, post_alpha, colors="firebrick", linewidth=0.8, alpha=0.7, label="After BN fusion")

  ax.axhline(y=-pre_tensor_alpha, color="steelblue", linestyle="--", linewidth=1, label=f"Pre-fusion per-tensor α={pre_tensor_alpha:.4f}")
  ax.axhline(y=pre_tensor_alpha, color="steelblue", linestyle="--", linewidth=1)
  ax.axhline(y=-post_tensor_alpha, color="firebrick", linestyle="--", linewidth=1, label=f"Post-fusion per-tensor α={post_tensor_alpha:.4f}")
  ax.axhline(y=post_tensor_alpha, color="firebrick", linestyle="--", linewidth=1)
  ax.axhline(y=0, color="black", linewidth=0.5)

  ax.set_title(f"Layer {layer_idx}: {conv_name} ({out_ch} channels)", fontsize=10)
  ax.set_xlabel("Output channel")
  ax.set_ylabel("Weight value")
  ax.legend(fontsize=7.5, loc="upper left", prop={"family": "monospace", "size": 7.5})
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  safe = conv_name.replace("/", "_").replace(":", "_").replace(".", "_")[:60]
  save_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(save_dir / f"layer_{layer_idx:03d}_{safe}.png", dpi=500)
  plt.close(fig)


def main():
  MODELS_DIR.mkdir(exist_ok=True)

  model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
  model.eval()

  dummy_input = (torch.randn(BATCH_SIZE.value, 3, IMAGE_H_W.value, IMAGE_H_W.value),)
  for name, fold in [("not_fused", False), ("fused", True)]:
    save_path = MODELS_DIR / f"resnet50_v1_Opset{OPSET_VERSION}_{name}.onnx"
    torch.onnx.export(model, dummy_input, str(save_path), opset_version=OPSET_VERSION, do_constant_folding=fold)
    print(f"Saved {save_path}")

  pairs = collect_conv_bn_pairs(model)
  for idx, (conv_name, conv, bn_name, bn) in tqdm(enumerate(pairs)):
    pre_weight = conv.weight.data.numpy()
    post_weight = fuse_bn_into_conv(pre_weight, bn)
    plot_channel_ranges(idx, conv_name, pre_weight, post_weight, RESULTS_DIR)


if __name__ == "__main__":
  main()
