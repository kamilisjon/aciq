import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad.nn import Conv2d, Linear

from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.plotting_style import HIST_BINS, TailwindColor, capped_savefig_dpi
from aciq.resnet import ResNet, _weight_modules


def plot_layer_grid(modules: list[tuple[str, Conv2d | Linear]], rows: int, cols: int, out_path: Path, quantile: float) -> None:
  fig_w, fig_h = cols * 1.4, rows * 1.4
  fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
  for ax in axes.flat:
    ax.set_box_aspect(1)
    ax.set_visible(False)
  tail = (1.0 - quantile / 100.0) / 2.0
  for slot, (_, module) in enumerate(modules):
    if slot >= rows * cols:
      break
    ax = axes.flat[slot]
    ax.set_visible(True)
    w = module.weight.numpy().flatten().astype(np.float64)
    lo, hi = np.quantile(w, [tail, 1.0 - tail])
    pad = 0.05 * (hi - lo)
    ax.hist(w, bins=HIST_BINS, density=True, color=TailwindColor.TEAL)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
  fig.tight_layout(pad=0.2)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  dpi = capped_savefig_dpi(fig_w, fig_h)
  fig.savefig(out_path, dpi=dpi)
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-layer weight distribution grid (one bar histogram per weight-bearing module).")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--rows", type=int, default=5)
  parser.add_argument("--cols", type=int, default=5)
  parser.add_argument("--quantile", type=float, default=99.9, help="Per-layer percentile clip applied to each subplot's x-axis.")
  args = parser.parse_args()

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()
  modules = _weight_modules(model)

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_weights_per_layer")
  out_path = save_dir / f"{args.rows}x{args.cols}.png"
  print(f"Rendering {min(len(modules), args.rows * args.cols)} layers into {args.rows}x{args.cols} grid -> {out_path}")
  plot_layer_grid(modules, args.rows, args.cols, out_path, args.quantile)
  print("Done.")
