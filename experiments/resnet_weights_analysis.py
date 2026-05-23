import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tinygrad.nn import Conv2d, Linear

from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT, fit_distributions, kurtosis, skewness
from aciq.helpers import RESULTS_DIR, get_output_dir, mean_absolute_error, save_csv
from aciq.plotting_style import DIST_COLORS, HIST_BINS, LINE_WIDTH, NEUTRAL_COLOR, TailwindColor, capped_savefig_dpi
from aciq.quantization.clipping import bound_symmetric_aciq_mae, bound_symmetric_minmax, quantize_symmetric
from aciq.models.resnet import ResNet, _weight_modules


BIT_WIDTHS: tuple[int, ...] = (4, 8)

PER_LAYER_GRID: dict[str, tuple[int, int]] = {
  "resnet18": (3, 7),    # 21 weight layers
  "resnet34": (5, 8),    # 37 weight layers
  "resnet50": (6, 9),    # 54 weight layers
  "resnet101": (11, 9),  # 105 weight layers — appendix layout from README
  "resnet152": (13, 12), # 156 weight layers
}

CHANNEL_GRID: tuple[int, int] = (2, 4)


@dataclass
class LayerFits:
  idx: int
  name: str
  module: Conv2d | Linear
  flat: np.ndarray
  layer_best: Distribution
  channel_bests: list[Distribution]


@dataclass
class WeightMomentsRow:
  layer_idx: int
  layer_name: str
  n: int
  mean: float
  variance: float
  skewness: float
  kurtosis: float


@dataclass
class FitSummaryRow:
  layer_idx: int
  layer_name: str
  bits: int
  best_fit: str
  alpha_minmax: float
  alpha_aciq: float
  mae_minmax: float
  mae_aciq: float


@dataclass
class ChannelFitCountsRow:
  layer_idx: int
  layer_name: str
  num_channels: int
  global_best: str
  count_gaussian: int
  count_laplace: int
  count_studentt: int
  count_ged: int


def build_layer_fits(model: ResNet) -> list[LayerFits]:
  fits: list[LayerFits] = []
  for layer_idx, (name, module) in enumerate(_weight_modules(model), 1):
    weight = module.weight.numpy()
    flat = weight.flatten().astype(np.float64)
    layer_best = fit_distributions(np.sort(flat))[0]
    channel_bests = [fit_distributions(weight[c].flatten().astype(np.float64))[0] for c in range(int(weight.shape[0]))]
    fits.append(LayerFits(idx=layer_idx, name=name, module=module, flat=flat, layer_best=layer_best, channel_bests=channel_bests))
    print(f"  [{layer_idx:>3}] {name:40s}  layer_best={type(layer_best).name:10s}  channels={len(channel_bests):>4}")
  return fits


def plot_per_layer_grid(fits: list[LayerFits], rows: int, cols: int, quantile: float, out_path: Path) -> None:
  fig_w, fig_h = cols * 1.4, rows * 1.4
  fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
  for ax in axes.flat:
    ax.set_box_aspect(1)
    ax.set_visible(False)
  tail = (1.0 - quantile / 100.0) / 2.0
  for slot, f in enumerate(fits):
    if slot >= rows * cols:
      break
    ax = axes.flat[slot]
    ax.set_visible(True)
    lo, hi = np.quantile(f.flat, [tail, 1.0 - tail])
    pad = 0.05 * (hi - lo)
    ax.hist(f.flat, bins=HIST_BINS, density=True, color=TailwindColor.TEAL)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
  fig.tight_layout(pad=0.2)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  dpi = capped_savefig_dpi(fig_w, fig_h)
  fig.savefig(out_path, dpi=dpi)
  plt.close(fig)


def compute_moments(fits: list[LayerFits]) -> list[WeightMomentsRow]:
  return [
    WeightMomentsRow(
      layer_idx=f.idx,
      layer_name=f.name,
      n=int(f.flat.size),
      mean=float(np.mean(f.flat)),
      variance=float(np.var(f.flat)),
      skewness=float(skewness(f.flat)),
      kurtosis=float(kurtosis(f.flat)) + 3.0,
    )
    for f in fits
  ]


def plot_moments(rows: list[WeightMomentsRow], out_path: Path) -> None:
  idx = np.array([r.layer_idx for r in rows])
  mean = np.array([r.mean for r in rows])
  variance = np.array([r.variance for r in rows])
  skew = np.array([r.skewness for r in rows])
  kurt = np.array([r.kurtosis for r in rows])

  fig, axes = plt.subplots(2, 2, figsize=(11, 7))
  for ax, values, ylabel in (
    (axes[0, 0], mean, "Mean"),
    (axes[0, 1], variance, "Variance"),
    (axes[1, 0], skew, "Skewness"),
    (axes[1, 1], kurt, "Kurtosis"),
  ):
    ax.bar(idx, values, color=TailwindColor.TEAL)
    ax.set_ylabel(ylabel)

  for ax in (axes[0, 0], axes[1, 0]):
    ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=LINE_WIDTH)
  axes[1, 1].axhline(3.0, color=TailwindColor.RED, linewidth=2.0, label="Gaussian")
  axes[1, 1].legend(loc="upper left")

  for ax in axes[1]:
    ax.set_xlabel("Layer index (forward order)")

  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path)
  plt.close(fig)


def compute_fit_summary(fits: list[LayerFits], bit_widths: tuple[int, ...]) -> list[FitSummaryRow]:
  rows: list[FitSummaryRow] = []
  for f in fits:
    alpha_minmax = float(bound_symmetric_minmax(f.flat))
    cdf = lambda x, dist=f.layer_best: float(dist.cdf_at(np.asarray(x)))
    for bits in bit_widths:
      alpha_aciq = float(bound_symmetric_aciq_mae(cdf=cdf, b=bits, alpha_max=alpha_minmax))
      mae_minmax = float(mean_absolute_error(f.flat, quantize_symmetric(f.flat, alpha_minmax, bits)))
      mae_aciq = float(mean_absolute_error(f.flat, quantize_symmetric(f.flat, alpha_aciq, bits)))
      rows.append(
        FitSummaryRow(
          layer_idx=f.idx,
          layer_name=f.name,
          bits=bits,
          best_fit=type(f.layer_best).name,
          alpha_minmax=alpha_minmax,
          alpha_aciq=alpha_aciq,
          mae_minmax=mae_minmax,
          mae_aciq=mae_aciq,
        )
      )
      print(
        f"  [{f.idx:>3}] {f.name:40s}  bits={bits}  best={type(f.layer_best).name:10s}  "
        f"amm={alpha_minmax:.4f}  aaciq={alpha_aciq:.4f}  mm={mae_minmax:.3e}  aciq={mae_aciq:.3e}"
      )
  return rows


def plot_channel_grid(f: LayerFits, rows: int, cols: int, quantile: float, out_path: Path) -> None:
  weight = f.module.weight.numpy()
  num_channels = int(weight.shape[0])
  selected = min(rows * cols, num_channels)

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
    chan_best_dist = f.channel_bests[slot]
    lo, hi = np.quantile(chan, [tail, 1.0 - tail])
    pad = 0.05 * (hi - lo)
    x_lo, x_hi = lo - pad, hi + pad
    ax.hist(chan, bins=HIST_BINS, density=True, color=NEUTRAL_COLOR, alpha=0.5)
    grid = np.linspace(x_lo, x_hi, 400)
    ax.plot(grid, f.layer_best.pdf_at(grid), color=DIST_COLORS[type(f.layer_best)], linewidth=LINE_WIDTH, linestyle="--")
    ax.plot(grid, chan_best_dist.pdf_at(grid), color=DIST_COLORS[type(chan_best_dist)], linewidth=LINE_WIDTH, linestyle="-")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

  dist_handles = [Line2D([0], [0], color=col, lw=LINE_WIDTH, label=cls.name) for cls, col in DIST_COLORS.items()]
  style_handles = [
    Line2D([0], [0], color="black", lw=LINE_WIDTH, linestyle="--", label="Layer fit"),
    Line2D([0], [0], color="black", lw=LINE_WIDTH, linestyle="-", label="Channel fit"),
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


def compute_channel_fit_counts(fits: list[LayerFits]) -> list[ChannelFitCountsRow]:
  rows: list[ChannelFitCountsRow] = []
  for f in fits:
    counts: dict[type[Distribution], int] = {Gaussian: 0, Laplace: 0, StudentT: 0, GeneralizedGaussian: 0}
    for dist in f.channel_bests:
      counts[type(dist)] += 1
    rows.append(
      ChannelFitCountsRow(
        layer_idx=f.idx,
        layer_name=f.name,
        num_channels=len(f.channel_bests),
        global_best=type(f.layer_best).name,
        count_gaussian=counts[Gaussian],
        count_laplace=counts[Laplace],
        count_studentt=counts[StudentT],
        count_ged=counts[GeneralizedGaussian],
      )
    )
  return rows


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Full weight-distribution analysis for a pretrained ResNet: per-layer histograms, moments, MinMax/ACIQ fit summary at int4 and int8, per-channel fit grids, and per-channel fit counts.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--per-layer-quantile", type=float, default=99.9, help="Per-layer percentile clip applied to each subplot's x-axis.")
  parser.add_argument("--channel-grid-layers", type=str, nargs="+", default=["layer3.0.conv1"], help="Layer names to render per-channel distribution grids for.")
  parser.add_argument("--channel-grid-quantile", type=float, default=99.9, help="Percent of data inside each per-channel subplot's x-range.")
  args = parser.parse_args()

  per_layer_rows, per_layer_cols = PER_LAYER_GRID[args.model]
  channel_grid_rows, channel_grid_cols = CHANNEL_GRID

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()

  available = {name for name, _ in _weight_modules(model)}
  unknown = [layer for layer in args.channel_grid_layers if layer not in available]
  if unknown:
    raise SystemExit(f"Unknown layer(s) {unknown}. Available: {sorted(available)}")

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_weights_analysis")
  print(f"Output directory: {save_dir}")

  print("Building per-layer + per-channel distribution fits")
  fits = build_layer_fits(model)
  fits_by_name = {f.name: f for f in fits}

  print("Rendering per-layer weight histogram grid")
  plot_per_layer_grid(fits, per_layer_rows, per_layer_cols, args.per_layer_quantile, save_dir / "per_layer" / f"{per_layer_rows}x{per_layer_cols}.png")

  print("Computing per-layer moments")
  moment_rows = compute_moments(fits)
  save_csv(moment_rows, save_dir / "statistics" / "moments.csv")
  plot_moments(moment_rows, save_dir / "statistics" / "moments.png")
  skew_arr = np.array([r.skewness for r in moment_rows])
  kurt_arr = np.array([r.kurtosis for r in moment_rows])
  print(f"  Skewness range: [{skew_arr.min():+.3f}, {skew_arr.max():+.3f}]")
  print(f"  Kurtosis range: [{kurt_arr.min():+.3f}, {kurt_arr.max():+.3f}]")

  print(f"Computing MinMax/ACIQ fit summary at bits={list(BIT_WIDTHS)}")
  fit_rows = compute_fit_summary(fits, BIT_WIDTHS)
  save_csv(fit_rows, save_dir / "fit_summary" / "fit_summary.csv")

  for layer_name in args.channel_grid_layers:
    safe_name = layer_name.replace(".", "_")
    out_path = save_dir / "channel_grid" / f"channel_fit_grid_{safe_name}.png"
    print(f"Rendering {channel_grid_rows}x{channel_grid_cols} channel grid for {layer_name}")
    plot_channel_grid(fits_by_name[layer_name], channel_grid_rows, channel_grid_cols, args.channel_grid_quantile, out_path)

  print("Computing per-channel fit counts")
  count_rows = compute_channel_fit_counts(fits)
  save_csv(count_rows, save_dir / "channel_counts" / "channel_fit_counts.csv")

  print(f"Done. All artifacts written under {save_dir}")
