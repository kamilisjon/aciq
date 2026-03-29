from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tinygrad.helpers import ContextVar, tqdm

from aciq.batch_norm import collect_conv_bn_pairs, fuse_bn_into_conv


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/bn_fusion_effects")
OPSET_VERSION = 18
BATCH_SIZE = ContextVar("BATCH_SIZE", 16)
IMAGE_H_W = ContextVar("IMAGE_H_W", 224)
MODELS = {
  "resnet18": torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT),
  "resnet50": torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT),
}


def plot_channel_ranges(layer_idx: int, conv_name: str, pre_weight: np.ndarray, post_weight: np.ndarray, save_dir: Path):
  out_ch = pre_weight.shape[0]
  pre_flat = pre_weight.reshape(out_ch, -1)
  post_flat = post_weight.reshape(out_ch, -1)

  pre_min, pre_max = pre_flat.min(axis=1), pre_flat.max(axis=1)
  post_min, post_max = post_flat.min(axis=1), post_flat.max(axis=1)

  # Symmetric per-tensor alpha for quantization clip
  pre_tensor_alpha = float(np.abs(pre_weight).max())
  post_tensor_alpha = float(np.abs(post_weight).max())

  channels = np.arange(out_ch)

  fig, ax = plt.subplots(figsize=(12, 5))

  ax.vlines(channels - 0.15, pre_min, pre_max, colors="steelblue", linewidth=0.8, alpha=0.7, label="Per-channel [min,max] before BN fusion")
  ax.vlines(channels + 0.15, post_min, post_max, colors="firebrick", linewidth=0.8, alpha=0.7, label="Per-channel [min,max] after BN fusion")

  ax.axhline(y=-pre_tensor_alpha, color="steelblue", linestyle="--", linewidth=1, label=f"Per-tensor clip α={pre_tensor_alpha:.4f} before BN fusion")
  ax.axhline(y=pre_tensor_alpha, color="steelblue", linestyle="--", linewidth=1)
  ax.axhline(y=-post_tensor_alpha, color="firebrick", linestyle="--", linewidth=1, label=f"Per-tensor clip α={post_tensor_alpha:.4f} after BN fusion")
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


def analyze_model(model_name: str, model: torch.nn.Module):
  model.eval()

  dummy_input = (torch.randn(BATCH_SIZE.value, 3, IMAGE_H_W.value, IMAGE_H_W.value),)
  for name, fold in [("not_fused", False), ("fused", True)]:
    save_path = MODELS_DIR / f"{model_name}_Opset{OPSET_VERSION}_{name}.onnx"
    torch.onnx.export(model, dummy_input, str(save_path), opset_version=OPSET_VERSION, do_constant_folding=fold)
    print(f"Saved {save_path}")

  save_dir = RESULTS_DIR / model_name
  pairs = collect_conv_bn_pairs(model)
  for idx, (conv_name, conv, bn_name, bn) in tqdm(enumerate(pairs)):
    pre_weight = conv.weight.data.numpy()
    post_weight = fuse_bn_into_conv(pre_weight, bn)
    plot_channel_ranges(idx, conv_name, pre_weight, post_weight, save_dir)


if __name__ == "__main__":
  MODELS_DIR.mkdir(exist_ok=True)
  for model_name, model in MODELS.items():
    print(f"=== {model_name} ===")
    analyze_model(model_name, model)
