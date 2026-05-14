import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad.helpers import tqdm

from aciq.helpers import RESULTS_DIR, fuse_conv_bn, get_output_dir
from aciq.plotting_style import TailwindColor
from aciq.resnet import ResNet, _conv_bn_pairs


def plot_channel_ranges(layer_idx: int, conv_name: str, pre_weight: np.ndarray, post_weight: np.ndarray, save_dir: Path) -> None:
  out_ch = pre_weight.shape[0]
  pre_flat = pre_weight.reshape(out_ch, -1)
  post_flat = post_weight.reshape(out_ch, -1)

  pre_min, pre_max = pre_flat.min(axis=1), pre_flat.max(axis=1)
  post_min, post_max = post_flat.min(axis=1), post_flat.max(axis=1)

  pre_tensor_alpha = float(np.abs(pre_weight).max())
  post_tensor_alpha = float(np.abs(post_weight).max())

  channels = np.arange(out_ch)

  fig, ax = plt.subplots(figsize=(12, 5))

  ax.vlines(channels - 0.15, pre_min, pre_max, colors=TailwindColor.BLUE, linewidth=0.8, alpha=0.7, label="Kanalų [min,max] prieš BN suliejimą")
  ax.vlines(channels + 0.15, post_min, post_max, colors=TailwindColor.ROSE, linewidth=0.8, alpha=0.7, label="Kanalų [min,max] po BN suliejimo")

  ax.axhline(y=-pre_tensor_alpha, color=TailwindColor.BLUE, linestyle="--", linewidth=1, label=f"Tensoriaus α={pre_tensor_alpha:.4f} prieš BN suliejimą")
  ax.axhline(y=pre_tensor_alpha, color=TailwindColor.BLUE, linestyle="--", linewidth=1)
  ax.axhline(y=-post_tensor_alpha, color=TailwindColor.ROSE, linestyle="--", linewidth=1, label=f"Tensoriaus α={post_tensor_alpha:.4f} po BN suliejimo")
  ax.axhline(y=post_tensor_alpha, color=TailwindColor.ROSE, linestyle="--", linewidth=1)
  ax.axhline(y=0, color="black", linewidth=0.5)

  ax.set_xlabel("Išėjimo kanalas")
  ax.set_ylabel("Svorio reikšmė")
  ax.legend(loc="upper left")
  fig.tight_layout()

  safe = conv_name.replace("/", "_").replace(":", "_").replace(".", "_")[:60]
  save_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(save_dir / f"layer_{layer_idx:03d}_{safe}.png")
  plt.close(fig)


def run(model: ResNet, save_dir: Path) -> None:
  for idx, (name, conv, bn) in enumerate(tqdm(_conv_bn_pairs(model))):
    pre_weight = conv.weight.numpy()
    post_weight, _ = fuse_conv_bn(conv, bn)
    plot_channel_ranges(idx, name, pre_weight, post_weight.numpy(), save_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-channel weight range visualisation before/after BN fusion")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  args = parser.parse_args()

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_bn_fusion")
  print(f"Output directory: {save_dir}")

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()

  run(model, save_dir)
  print(f"Done. Plots in {save_dir}")
