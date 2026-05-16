import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from scipy.stats import spearmanr

from aciq.quantization.bias_correction import ChannelMeansAccumulator
from aciq.distributions import fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, save_csv
from aciq.datasets.mnist import _load_normalized, train_model
from aciq.models import MiniConv
from aciq.models.miniconv import BlockName, BlockName2
from aciq.plotting_style import NEUTRAL_COLOR, TailwindColor
from aciq.quantization.clipping import bound_symmetric_minmax, quantize_symmetric, bound_symmetric_aciq_mae


BITS = 4


@dataclass
class MnistResultRow:
  seed: int
  fp32_acc: float
  minmax_acc: float
  aciq_acc: float
  minmax_block1_mean_shift: float
  minmax_block2_mean_shift: float
  minmax_block3_mean_shift: float
  minmax_block4_mean_shift: float
  aciq_block1_mean_shift: float
  aciq_block2_mean_shift: float
  aciq_block3_mean_shift: float
  aciq_block4_mean_shift: float


@dataclass
class QuantStatsRow:
  seed: int
  layer_name: str
  channel: int
  channel_size: int
  minmax_alpha: float
  aciq_alpha: float
  aciq_best_fit: str
  minmax_weight_err: float
  aciq_weight_err: float


@dataclass
class DistributionCountsRow:
  layer_name: str
  distribution_name: str
  count: int


@dataclass
class MaeSummaryRow:
  seed: int
  layer_name: str
  num_channels: int
  total_weights: int
  minmax_total_err: float
  aciq_total_err: float
  minmax_mae: float
  aciq_mae: float


def build_distribution_counts(rows: list[QuantStatsRow]) -> list[DistributionCountsRow]:
  counts: dict[tuple[str, str], int] = defaultdict(int)
  for r in rows:
    counts[(r.layer_name, r.aciq_best_fit)] += 1
  return [DistributionCountsRow(layer_name=l, distribution_name=d, count=n) for (l, d), n in sorted(counts.items())]


def build_mae_summary(rows: list[QuantStatsRow]) -> list[MaeSummaryRow]:
  agg: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: {"n_ch": 0.0, "tot_w": 0.0, "mm": 0.0, "aq": 0.0})
  for r in rows:
    a = agg[(r.seed, r.layer_name)]
    a["n_ch"] += 1
    a["tot_w"] += r.channel_size
    a["mm"] += r.minmax_weight_err
    a["aq"] += r.aciq_weight_err
  out: list[MaeSummaryRow] = []
  for (seed, layer), a in sorted(agg.items()):
    tot_w = int(a["tot_w"])
    out.append(MaeSummaryRow(
      seed=seed,
      layer_name=layer,
      num_channels=int(a["n_ch"]),
      total_weights=tot_w,
      minmax_total_err=a["mm"],
      aciq_total_err=a["aq"],
      minmax_mae=a["mm"] / tot_w,
      aciq_mae=a["aq"] / tot_w,
    ))
  return out


@dataclass
class MaePerNetworkRow:
  seed: int
  total_weights: int
  minmax_total_err: float
  aciq_total_err: float
  minmax_mae: float
  aciq_mae: float


@dataclass
class MaeAverageRow:
  n_seeds: int
  total_weights_per_network: int
  mean_fp32_acc: float
  std_fp32_acc: float
  mean_minmax_acc: float
  std_minmax_acc: float
  mean_aciq_acc: float
  std_aciq_acc: float
  mean_paired_acc_diff: float
  std_paired_acc_diff: float
  mean_minmax_mae: float
  std_minmax_mae: float
  mean_aciq_mae: float
  std_aciq_mae: float


def build_mae_per_network(rows: list[QuantStatsRow]) -> list[MaePerNetworkRow]:
  agg: dict[int, dict[str, float]] = defaultdict(lambda: {"tot_w": 0.0, "mm": 0.0, "aq": 0.0})
  for r in rows:
    a = agg[r.seed]
    a["tot_w"] += r.channel_size
    a["mm"] += r.minmax_weight_err
    a["aq"] += r.aciq_weight_err
  out: list[MaePerNetworkRow] = []
  for seed, a in sorted(agg.items()):
    tot_w = int(a["tot_w"])
    out.append(MaePerNetworkRow(
      seed=seed,
      total_weights=tot_w,
      minmax_total_err=a["mm"],
      aciq_total_err=a["aq"],
      minmax_mae=a["mm"] / tot_w,
      aciq_mae=a["aq"] / tot_w,
    ))
  return out


def build_mae_average(per_network: list[MaePerNetworkRow], result_rows: list[MnistResultRow]) -> list[MaeAverageRow]:
  assert per_network, "build_mae_average requires at least one per-network row"
  fp = np.array([r.fp32_acc for r in result_rows], dtype=np.float64)
  mm_acc = np.array([r.minmax_acc for r in result_rows], dtype=np.float64)
  aq_acc = np.array([r.aciq_acc for r in result_rows], dtype=np.float64)
  d_acc = aq_acc - mm_acc
  mm_mae = np.array([r.minmax_mae for r in per_network], dtype=np.float64)
  aq_mae = np.array([r.aciq_mae for r in per_network], dtype=np.float64)
  std = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0
  return [MaeAverageRow(
    n_seeds=len(per_network),
    total_weights_per_network=per_network[0].total_weights,
    mean_fp32_acc=float(fp.mean()),
    std_fp32_acc=std(fp),
    mean_minmax_acc=float(mm_acc.mean()),
    std_minmax_acc=std(mm_acc),
    mean_aciq_acc=float(aq_acc.mean()),
    std_aciq_acc=std(aq_acc),
    mean_paired_acc_diff=float(d_acc.mean()),
    std_paired_acc_diff=std(d_acc),
    mean_minmax_mae=float(mm_mae.mean()),
    std_minmax_mae=std(mm_mae),
    mean_aciq_mae=float(aq_mae.mean()),
    std_aciq_mae=std(aq_mae),
  )]


def _shifts(rows: list[MnistResultRow], method: str, block: str) -> np.ndarray:
  return np.array([getattr(r, f"{method}_{block}_mean_shift") for r in rows])


# --- Quantization ---


class QuantMethod(StrEnum):
  MINMAX = "minmax"
  ACIQ = "aciq"


def _aciq_alpha(vec: np.ndarray) -> tuple[float, str]:
  alpha_mm = bound_symmetric_minmax(vec)
  best = fit_distributions(np.sort(vec))[0]
  alpha = float(bound_symmetric_aciq_mae(cdf=lambda x: float(best.cdf_at(np.asarray(x))), b=BITS, alpha_max=alpha_mm))
  return alpha, type(best).name


@dataclass
class LayerQuantStats:
  layer_name: str
  channel_size: int
  alpha_per_channel: list[float]
  best_fit_per_channel: list[str]
  total_err_per_channel: list[float]


def quantize_model(model: MiniConv, method: QuantMethod) -> tuple[MiniConv, list[LayerQuantStats]]:
  qmodel = copy.deepcopy(model)
  qmodel.fuse()
  stats: list[LayerQuantStats] = []
  for name, mod in qmodel.named_weight_modules:
    w = mod.weight.numpy()
    ch_size = int(np.prod(w.shape[1:]))
    q_buf = np.empty_like(w, dtype=np.float32)
    alphas: list[float] = []
    fits: list[str] = []
    errs: list[float] = []
    for c in range(w.shape[0]):
      ch_vec = w[c].flatten()
      if method == QuantMethod.MINMAX:
        alpha = float(bound_symmetric_minmax(ch_vec))
        fit_name = ""
      else:
        alpha, fit_name = _aciq_alpha(ch_vec)
      q_vec = quantize_symmetric(ch_vec, alpha, BITS)
      q_buf[c] = q_vec.reshape(w[c].shape).astype(np.float32)
      alphas.append(alpha)
      fits.append(fit_name)
      errs.append(float(np.sum(np.abs(ch_vec - q_vec))))
    mod.weight = Tensor(q_buf)
    stats.append(LayerQuantStats(
      layer_name=name,
      channel_size=ch_size,
      alpha_per_channel=alphas,
      best_fit_per_channel=fits,
      total_err_per_channel=errs,
    ))
  MiniConv.clear_jit_caches()
  return qmodel, stats


# --- Distribution shift measurement ---


def collect_layer_outputs(model: MiniConv, x_test: Tensor, batch_size: int = 100) -> ChannelMeansAccumulator:
  acc = ChannelMeansAccumulator()
  n = x_test.shape[0]
  assert n % batch_size == 0, f"test set size {n} must be divisible by batch_size {batch_size}"
  for start in tqdm(range(0, n, batch_size), desc="  activations"):
    idx = Tensor.arange(start, start + batch_size)
    for name, act in model.get_activations(x_test[idx]).items():
      acc.update(name, act.numpy())
  return acc


# --- Training + measurement ---


def run_training(n_models: int, steps: int) -> tuple[list[MnistResultRow], list[QuantStatsRow]]:
  _, _, x_test, y_test = _load_normalized()

  result_rows: list[MnistResultRow] = []
  quant_stats_rows: list[QuantStatsRow] = []
  for seed in range(n_models):
    print(f"[{seed + 1}/{n_models}] Training model (seed={seed})...")
    model, fp32_acc, _, _ = train_model(MiniConv, seed=seed, steps=steps)
    print("Model trained")

    fp32_outputs = collect_layer_outputs(model, x_test)
    print("Collected activations")

    accs: dict[QuantMethod, float] = {}
    shifts: dict[QuantMethod, dict[str, float]] = {}
    layer_stats: dict[QuantMethod, list[LayerQuantStats]] = {}
    for method in QuantMethod:
      qmodel, stats = quantize_model(model, method)
      accs[method] = float(qmodel.test_acc(x_test, y_test).item())
      q_outputs = collect_layer_outputs(qmodel, x_test)
      shifts[method] = {r.layer: r.mean_shift for r in fp32_outputs.layers_means_shifts(q_outputs, method)}
      layer_stats[method] = stats
      print(f"{method} quantization done")

    for mm_layer, aciq_layer in zip(layer_stats[QuantMethod.MINMAX], layer_stats[QuantMethod.ACIQ]):
      assert mm_layer.layer_name == aciq_layer.layer_name
      assert mm_layer.channel_size == aciq_layer.channel_size
      for ch, (mm_alpha, aciq_alpha, aciq_fit, mm_err, aciq_err) in enumerate(
        zip(
          mm_layer.alpha_per_channel,
          aciq_layer.alpha_per_channel,
          aciq_layer.best_fit_per_channel,
          mm_layer.total_err_per_channel,
          aciq_layer.total_err_per_channel,
        )
      ):
        quant_stats_rows.append(
          QuantStatsRow(
            seed=seed,
            layer_name=mm_layer.layer_name,
            channel=ch,
            channel_size=mm_layer.channel_size,
            minmax_alpha=mm_alpha,
            aciq_alpha=aciq_alpha,
            aciq_best_fit=aciq_fit,
            minmax_weight_err=mm_err,
            aciq_weight_err=aciq_err,
          )
        )

    print(f"  FP32={fp32_acc:.4f}  MinMax={accs[QuantMethod.MINMAX]:.4f}  ACIQ={accs[QuantMethod.ACIQ]:.4f}")
    result_rows.append(
      MnistResultRow(
        seed=seed,
        fp32_acc=fp32_acc,
        minmax_acc=accs[QuantMethod.MINMAX],
        aciq_acc=accs[QuantMethod.ACIQ],
        **{f"{m}_{b}_mean_shift": shifts[m][b] for m in QuantMethod for b in BlockName},
      )
    )
    MiniConv.clear_jit_caches()
  return result_rows, quant_stats_rows


# --- Analysis ---


def plot_scatter(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
  for col_idx, (method, color) in enumerate([(QuantMethod.MINMAX, TailwindColor.TEAL), (QuantMethod.ACIQ, TailwindColor.ORANGE)]):
    acc_drops = np.array([r.fp32_acc - getattr(r, f"{method}_acc") for r in rows])
    total_shifts = np.sum([_shifts(rows, method, b) for b in BlockName], axis=0)
    ax = axes[col_idx]
    ax.scatter(total_shifts, acc_drops, color=color, alpha=0.6, s=20)
    rho, p = spearmanr(total_shifts, acc_drops)
    ax.set_title(f"{method.upper()} rho={rho:.3f} p={p:.3g}")
    ax.set_xlabel("Bendras vidurkio poslinkis")
    if col_idx == 0:
      ax.set_ylabel("Tikslumo kritimas")
    ax.grid(False)
  fig.suptitle("Mean shift vs accuracy drop (Spearman correlation)", y=1.02)
  fig.tight_layout()
  fig.savefig(save_dir / "scatter_mean_shift_vs_accuracy.png")
  plt.close(fig)


def plot_minmax_vs_aciq_accuracy(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fp32 = np.array([r.fp32_acc for r in rows])
  mm = np.array([r.minmax_acc for r in rows])
  aciq = np.array([r.aciq_acc for r in rows])
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(mm, aciq, color=TailwindColor.VIOLET, alpha=0.6, s=24)
  lo = float(min(mm.min(), aciq.min())) - 0.02
  hi = float(max(mm.max(), aciq.max(), fp32.max())) + 0.02
  ax.plot([lo, hi], [lo, hi], color=NEUTRAL_COLOR, linestyle="--", linewidth=1.0, label="y = x")
  ax.axvline(float(fp32.mean()), color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8)
  ax.axhline(float(fp32.mean()), color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8)
  ax.set_xlim(lo, hi)
  ax.set_ylim(lo, hi)
  ax.set_xlabel("MINMAX tikslumas")
  ax.set_ylabel("ACIQ tikslumas")
  wins_aciq = int(np.sum(aciq > mm))
  ax.set_title(
    f"MINMAX prieš ACIQ tikslumas\n"
    f"vidurkis: MINMAX={mm.mean():.3f}  ACIQ={aciq.mean():.3f}  "
    f"ACIQ > MINMAX: {wins_aciq}/{len(rows)}"
  )
  ax.grid(False)
  ax.legend(loc="lower right")
  fig.tight_layout()
  fig.savefig(save_dir / "minmax_vs_aciq_accuracy.png")
  plt.close(fig)


def plot_per_layer_shift(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10, 5))
  for color, method in [(TailwindColor.TEAL, QuantMethod.MINMAX), (TailwindColor.ORANGE, QuantMethod.ACIQ)]:
    per_layer_means = [_shifts(rows, method, b).mean() for b in BlockName]
    per_layer_stds = [_shifts(rows, method, b).std() for b in BlockName]

    x_pos = np.arange(len(BlockName))
    ax.bar(
      x_pos + (-0.2 if method == QuantMethod.MINMAX else 0.2),
      per_layer_means,
      width=0.35,
      color=color,
      label=method.upper(),
      yerr=per_layer_stds,
      capsize=3,
    )
  ax.set_xticks(np.arange(len(BlockName)))
  ax.set_xticklabels(BlockName2)
  ax.set_xlabel("Sluoksnis")
  ax.set_ylabel("Išvesties vidurkio poslinkis")
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_dir / "per_layer_mean_shift.png")
  plt.close(fig)


# --- Main ---


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MNIST quantization distribution shift analysis")
  parser.add_argument("--n-models", type=int, default=100, help="Number of models to train")
  parser.add_argument("--steps", type=int, default=100, help="Training steps per model")
  parser.add_argument(
    "--from-dir",
    type=Path,
    default=None,
    help="Load `results.csv` from this experiment directory and re-render plots only (no training).",
  )
  args = parser.parse_args()
  save_dir = get_output_dir(RESULTS_DIR, "mnist")

  quant_stats_rows: list[QuantStatsRow] = []
  if args.from_dir:
    rows = load_csv(args.from_dir / "results.csv", MnistResultRow)
    print(f"Loaded {len(rows)} models from {args.from_dir / 'results.csv'}")
    quant_stats_path = args.from_dir / "quant_stats.csv"
    if quant_stats_path.exists():
      quant_stats_rows = load_csv(quant_stats_path, QuantStatsRow)
  else:
    print(f"Running training with {args.n_models} models, {args.steps} steps each...")
    rows, quant_stats_rows = run_training(args.n_models, args.steps)
    save_csv(rows, save_dir / "results.csv")
    if quant_stats_rows:
      save_csv(quant_stats_rows, save_dir / "quant_stats.csv")

  if quant_stats_rows:
    save_csv(build_distribution_counts(quant_stats_rows), save_dir / "distribution_counts.csv")
    save_csv(build_mae_summary(quant_stats_rows), save_dir / "mae_summary.csv")
    per_network = build_mae_per_network(quant_stats_rows)
    save_csv(per_network, save_dir / "mae_per_network.csv")
    save_csv(build_mae_average(per_network, rows), save_dir / "mae_average.csv")
    print(f"Emitted derived summaries to {save_dir}/")

  plot_scatter(rows, save_dir)
  plot_per_layer_shift(rows, save_dir)
  plot_minmax_vs_aciq_accuracy(rows, save_dir)
  print(f"Plots saved to {save_dir}/")
