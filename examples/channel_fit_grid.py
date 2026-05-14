import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tinygrad.nn import Conv2d, Linear

from aciq.distributions import Distribution, fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.plotting_style import DIST_COLORS, HIST_BINS, LINE_WIDTH, NEUTRAL_COLOR, capped_savefig_dpi
from aciq.resnet import ResNet, _weight_modules


def _best_fit(vec: np.ndarray) -> tuple[type[Distribution], Distribution]:
  fits = fit_distributions(vec)
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  return best_type, fits[best_type]


def plot_channel_grid(name: str, module: Conv2d | Linear, rows: int, cols: int, quantile: float, out_path: Path) -> None:
  weight = module.weight.numpy()
  num_channels = int(weight.shape[0])
  selected = min(rows * cols, num_channels)

  global_best_type, global_best_dist = _best_fit(weight.flatten().astype(np.float64))

  fig_w, fig_h = cols * 1.8, rows * 1.8 + 0.7
  fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
  for ax in axes.flat:
    ax.set_box_aspect(1)
    ax.set_visible(False)
  tail = (1.0 - quantile / 100.0) / 2.0

  for slot in range(selected):
    ax = axes.flat[slot]
    ax.set_visible(True)
    chan = weight[slot].flatten().astype(np.float64)
    chan_best_type, chan_best_dist = _best_fit(chan)
    lo, hi = np.quantile(chan, [tail, 1.0 - tail])
    pad = 0.05 * (hi - lo)
    x_lo, x_hi = lo - pad, hi + pad
    ax.hist(chan, bins=HIST_BINS, density=True, color=NEUTRAL_COLOR, alpha=0.5)
    grid = np.linspace(x_lo, x_hi, 400)
    ax.plot(grid, global_best_dist.pdf_at(grid), color=DIST_COLORS[global_best_type], linewidth=LINE_WIDTH, linestyle="--")
    ax.plot(grid, chan_best_dist.pdf_at(grid), color=DIST_COLORS[chan_best_type], linewidth=LINE_WIDTH, linestyle="-")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"ch {slot}", fontsize=8, pad=2)

  dist_handles = [Line2D([0], [0], color=col, lw=LINE_WIDTH, label=cls.name) for cls, col in DIST_COLORS.items()]
  style_handles = [
    Line2D([0], [0], color="black", lw=LINE_WIDTH, linestyle="--", label="layer fit"),
    Line2D([0], [0], color="black", lw=LINE_WIDTH, linestyle="-", label="channel fit"),
  ]
  fig.legend(
    handles=dist_handles + style_handles,
    loc="lower center",
    ncol=len(dist_handles) + len(style_handles),
    bbox_to_anchor=(0.5, 0.0),
    frameon=False,
    fontsize=9,
  )
  fig.tight_layout(pad=0.3, rect=(0.0, 0.06, 1.0, 1.0))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  dpi = capped_savefig_dpi(fig_w, fig_h)
  fig.savefig(out_path, dpi=dpi)
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-channel distribution overlay grid for a single layer.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--layer", type=str, default="layer3.0.conv1")
  parser.add_argument("--rows", type=int, default=4)
  parser.add_argument("--cols", type=int, default=4)
  parser.add_argument("--quantile", type=float, default=99.9, help="Percent of data inside the per-channel visualization x-range.")
  args = parser.parse_args()

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()

  mods = dict(_weight_modules(model))
  if args.layer not in mods:
    raise SystemExit(f"Layer '{args.layer}' not found. Available: {sorted(mods)}")

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_channel_fit_grid")
  safe_name = args.layer.replace(".", "_")
  out_path = save_dir / f"channel_fit_grid_{safe_name}.png"
  print(f"Rendering {args.rows}x{args.cols} channel grid for {args.layer} -> {out_path}")
  plot_channel_grid(args.layer, mods[args.layer], args.rows, args.cols, args.quantile, out_path)
  print("Done.")
