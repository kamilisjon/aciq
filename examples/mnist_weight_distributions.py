import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tinygrad.nn import Conv2d

from aciq.distributions import fit_distributions, kurtosis, skewness
from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.mnist import MNISTModel, train_model
from aciq.plotting_style import DIST_COLORS, HIST_BINS, LINE_WIDTH, NEUTRAL_COLOR, STATS_TEXT_KW, capped_savefig_dpi
from aciq.quantization import bound_symmetric_aciq_mae, bound_symmetric_minmax


def plot_layer(ax: Axes, weights: np.ndarray, layer_name: str, bits: int) -> None:
  vec = weights.flatten().astype(np.float64)
  vec_sorted = np.sort(vec)
  fits = fit_distributions(vec_sorted)

  ax.hist(vec, bins=HIST_BINS, density=True, alpha=0.5, color=NEUTRAL_COLOR, label="Empirinė")
  for dist_type, fitted in sorted(fits.items(), key=lambda kv: kv[1].log_likelihood, reverse=True):
    ax.plot(vec_sorted, fitted.pdf(), color=DIST_COLORS[dist_type], linewidth=LINE_WIDTH, linestyle="--", label=f"{repr(fitted)}")

  alpha_mm = float(bound_symmetric_minmax(vec))
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  best = fits[best_type]
  alpha_aciq = float(bound_symmetric_aciq_mae(cdf=lambda x: float(best.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_mm))

  ax.axvline(-alpha_mm, color=NEUTRAL_COLOR, linestyle=":", linewidth=LINE_WIDTH, label=f"MinMax α={alpha_mm:.4f}")
  ax.axvline(alpha_mm, color=NEUTRAL_COLOR, linestyle=":", linewidth=LINE_WIDTH)
  if alpha_aciq != alpha_mm:
    ax.axvline(-alpha_aciq, color=DIST_COLORS[best_type], linestyle="-", linewidth=LINE_WIDTH, label=f"ACIQ {best.name} α={alpha_aciq:.4f}")
    ax.axvline(alpha_aciq, color=DIST_COLORS[best_type], linestyle="-", linewidth=LINE_WIDTH)

  eda_lines = [
    f"n {vec.size:,}",
    f"Min {float(np.min(vec)):.5f}",
    f"Max {float(np.max(vec)):.5f}",
    f"Mean {float(np.mean(vec)):.5f}",
    f"Variance {float(np.var(vec)):.6f}",
    f"Skewness {float(skewness(vec)):.4f}",
    f"Kurtosis {float(kurtosis(vec)):.4f}",
  ]
  ax.text(0.98, 0.96, "\n".join(eda_lines), transform=ax.transAxes, **STATS_TEXT_KW)
  ax.set_title(layer_name)
  ax.set_xlabel("Svorio reikšmė")
  ax.set_ylabel("Tankis")
  ax.legend(loc="upper left", fontsize=7)
  ax.grid(False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Qualitative MNIST conv weight distributions after one training run.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--bits", type=int, default=4)
  args = parser.parse_args()

  print(f"Training seed={args.seed} for {args.steps} steps")
  model, fp32_acc, _, _ = train_model(seed=args.seed, steps=args.steps, gather_losses=False)
  print(f"FP32 test acc = {fp32_acc:.4f}")
  model.fuse()

  conv_modules: list[tuple[str, Conv2d]] = [
    ("block1", model.conv1),
    ("block2", model.conv2),
    ("block3", model.conv3),
    ("block4", model.conv4),
  ]

  fig_w, fig_h = 12.0, 8.0
  fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
  for ax, (name, mod) in zip(axes.flat, conv_modules):
    plot_layer(ax, mod.weight.numpy(), name, args.bits)
  fig.tight_layout()

  save_dir = get_output_dir(RESULTS_DIR, "mnist_weight_distributions")
  save_dir.mkdir(parents=True, exist_ok=True)
  out_path = save_dir / "distributions.png"
  fig.savefig(out_path, dpi=capped_savefig_dpi(fig_w, fig_h))
  plt.close(fig)
  print(f"Saved {out_path}")
