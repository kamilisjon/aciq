import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tinygrad.nn import Conv2d, Linear

from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.plotting_style import HIST_BINS, LINE_WIDTH, NEUTRAL_COLOR, TailwindColor
from aciq.resnet import ResNet


def _resolve_layer(model: ResNet, path: str) -> Conv2d | Linear:
  obj: Any = model
  for token in path.split("."):
    obj = obj[int(token)] if token.isdigit() else getattr(obj, token)
  assert isinstance(obj, (Conv2d, Linear)), f"{path} resolved to {type(obj).__name__}, expected Conv2d or Linear"
  return obj


def plot_grid(weight: np.ndarray, channels: list[int], rows: int, cols: int, out_path: Path, quantile: float) -> None:
  c_out = weight.shape[0]
  flat_all = weight.flatten().astype(np.float64)
  kde = gaussian_kde(flat_all)
  tail = (1.0 - quantile / 100.0) / 2.0
  w_low, w_high = np.quantile(flat_all, [tail, 1.0 - tail])
  pad = 0.05 * (w_high - w_low)
  x_lo, x_hi = float(w_low - pad), float(w_high + pad)
  x_grid = np.linspace(x_lo, x_hi, 200)
  kde_y = kde(x_grid)

  fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(cols * 1.4, rows * 1.4))
  for ax in axes.flat:
    ax.set_box_aspect(1)
    ax.set_visible(False)
  for slot, c in enumerate(channels):
    if c >= c_out:
      continue
    ax = axes.flat[slot]
    ax.set_visible(True)
    ax.hist(weight[c].flatten(), bins=HIST_BINS, density=True, color=NEUTRAL_COLOR, alpha=0.5)
    ax.plot(x_grid, kde_y, color=TailwindColor.BLUE, linewidth=LINE_WIDTH)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

  axes.flat[0].set_xlim(x_lo, x_hi)
  fig.tight_layout(pad=0.2)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path)
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-channel weight distribution grid (one subplot per output channel; overlay per-layer KDE).")
  parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--layer", type=str, default="layer3.0.conv2", help="Dot-path to the weight-bearing module, e.g. layer3.0.conv2.")
  parser.add_argument("--rows", type=int, default=5)
  parser.add_argument("--cols", type=int, default=5)
  parser.add_argument("--quantile", type=float, default=99.9, help="Central per-layer percentile used to clip the shared x-axis. 99.9 keeps the central 99.9 percent.")
  args = parser.parse_args()

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  module = _resolve_layer(model, args.layer)
  weight = module.weight.numpy()
  c_out = weight.shape[0]

  rows, cols = args.rows, args.cols
  channels = list(range(min(rows * cols, c_out)))

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_weights_per_channel")
  out_path = save_dir / f"{args.layer}_{rows}x{cols}.png"
  print(f"Rendering {len(channels)} channels into {rows}x{cols} grid -> {out_path}")
  plot_grid(weight, channels, rows, cols, out_path, args.quantile)
  print("Done.")
