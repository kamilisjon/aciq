import argparse
import copy
import time
from collections import defaultdict
from dataclasses import dataclass, make_dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from scipy.stats import rankdata, spearmanr, wilcoxon

from aciq.datasets.cifar10 import _load_normalized, train_model
from aciq.distributions import fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, save_csv
from aciq.models import ResNet4
from aciq.models.resnet4 import BlockName, BlockName2, bias_correct_model, compute_input_stats
from aciq.plotting_style import NEUTRAL_COLOR, TailwindColor
from aciq.quantization.bias_correction import ChannelMeansAccumulator
from aciq.quantization.clipping import bound_symmetric_aciq_mae, bound_symmetric_minmax, quantize_symmetric


BITS = 4
VARIANT_LABELS: tuple[str, ...] = ("minmax", "minmax_bias", "aciq", "aciq_bias")


class QuantMethod(StrEnum):
  MINMAX = "minmax"
  ACIQ = "aciq"


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
  mean_minmax_bias_acc: float
  std_minmax_bias_acc: float
  mean_aciq_acc: float
  std_aciq_acc: float
  mean_aciq_bias_acc: float
  std_aciq_bias_acc: float
  mean_paired_acc_diff: float
  std_paired_acc_diff: float
  mean_minmax_bias_gain: float
  std_minmax_bias_gain: float
  mean_aciq_bias_gain: float
  std_aciq_bias_gain: float
  mean_minmax_mae: float
  std_minmax_mae: float
  mean_aciq_mae: float
  std_aciq_mae: float


@dataclass
class WilcoxonRow:
  metric: str
  n_pairs: int
  median_diff: float
  statistic: float
  p_value: float
  rank_biserial: float


@dataclass
class LayerQuantStats:
  layer_name: str
  channel_size: int
  alpha_per_channel: list[float]
  best_fit_per_channel: list[str]
  total_err_per_channel: list[float]


@dataclass
class AccuracyRow:
  seed: int
  fp32_acc: float
  minmax_acc: float
  minmax_bias_acc: float
  aciq_acc: float
  aciq_bias_acc: float


def _make_shifts_row_cls(model_name: str, BlockName: type[StrEnum]) -> type:
  fields = [
    ("seed", int),
    *[(f"{v}_{b}_mean_shift", float) for v in VARIANT_LABELS for b in BlockName],
  ]
  return make_dataclass(f"{model_name}ShiftsRow", fields)


def _shifts(rows: list, method: str, block: str) -> np.ndarray:
  return np.array([getattr(r, f"{method}_{block}_mean_shift") for r in rows])


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


def _wilcoxon(diff: np.ndarray, metric: str) -> WilcoxonRow:
  nonzero = diff[diff != 0]
  if len(nonzero) == 0:
    return WilcoxonRow(metric=metric, n_pairs=0, median_diff=0.0, statistic=0.0, p_value=1.0, rank_biserial=0.0)
  abs_ranks = rankdata(np.abs(nonzero))
  T_plus = float(np.sum(abs_ranks[nonzero > 0]))
  T_minus = float(np.sum(abs_ranks[nonzero < 0]))
  res = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
  return WilcoxonRow(
    metric=metric,
    n_pairs=len(nonzero),
    median_diff=float(np.median(diff)),
    statistic=float(res.statistic),
    p_value=float(res.pvalue),
    rank_biserial=(T_plus - T_minus) / (T_plus + T_minus),
  )


def build_wilcoxon(accuracy_rows: list[AccuracyRow], per_network: list[MaePerNetworkRow]) -> list[WilcoxonRow]:
  assert len(accuracy_rows) == len(per_network), "accuracy and per-network rows must align by seed"
  acc_diff = np.array([r.aciq_acc - r.minmax_acc for r in accuracy_rows], dtype=np.float64)
  acc_bias_diff = np.array([r.aciq_bias_acc - r.minmax_bias_acc for r in accuracy_rows], dtype=np.float64)
  mae_diff = np.array([r.aciq_mae - r.minmax_mae for r in per_network], dtype=np.float64)
  minmax_bias_gain = np.array([r.minmax_bias_acc - r.minmax_acc for r in accuracy_rows], dtype=np.float64)
  aciq_bias_gain = np.array([r.aciq_bias_acc - r.aciq_acc for r in accuracy_rows], dtype=np.float64)
  return [
    _wilcoxon(acc_diff, "accuracy_aciq_vs_minmax"),
    _wilcoxon(acc_bias_diff, "accuracy_aciq_vs_minmax_after_bias"),
    _wilcoxon(mae_diff, "mae_aciq_vs_minmax"),
    _wilcoxon(minmax_bias_gain, "minmax_acc_bias_vs_none"),
    _wilcoxon(aciq_bias_gain, "aciq_acc_bias_vs_none"),
  ]


def build_mae_average(per_network: list[MaePerNetworkRow], accuracy_rows: list[AccuracyRow]) -> list[MaeAverageRow]:
  assert per_network, "build_mae_average requires at least one per-network row"
  fp = np.array([r.fp32_acc for r in accuracy_rows], dtype=np.float64)
  mm_acc = np.array([r.minmax_acc for r in accuracy_rows], dtype=np.float64)
  mm_bias_acc = np.array([r.minmax_bias_acc for r in accuracy_rows], dtype=np.float64)
  aq_acc = np.array([r.aciq_acc for r in accuracy_rows], dtype=np.float64)
  aq_bias_acc = np.array([r.aciq_bias_acc for r in accuracy_rows], dtype=np.float64)
  d_acc = aq_acc - mm_acc
  mm_bias_gain = mm_bias_acc - mm_acc
  aq_bias_gain = aq_bias_acc - aq_acc
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
    mean_minmax_bias_acc=float(mm_bias_acc.mean()),
    std_minmax_bias_acc=std(mm_bias_acc),
    mean_aciq_acc=float(aq_acc.mean()),
    std_aciq_acc=std(aq_acc),
    mean_aciq_bias_acc=float(aq_bias_acc.mean()),
    std_aciq_bias_acc=std(aq_bias_acc),
    mean_paired_acc_diff=float(d_acc.mean()),
    std_paired_acc_diff=std(d_acc),
    mean_minmax_bias_gain=float(mm_bias_gain.mean()),
    std_minmax_bias_gain=std(mm_bias_gain),
    mean_aciq_bias_gain=float(aq_bias_gain.mean()),
    std_aciq_bias_gain=std(aq_bias_gain),
    mean_minmax_mae=float(mm_mae.mean()),
    std_minmax_mae=std(mm_mae),
    mean_aciq_mae=float(aq_mae.mean()),
    std_aciq_mae=std(aq_mae),
  )]


def _aciq_alpha(vec: np.ndarray) -> tuple[float, str]:
  alpha_mm = bound_symmetric_minmax(vec)
  best = fit_distributions(np.sort(vec))[0]
  alpha = float(bound_symmetric_aciq_mae(cdf=lambda x: float(best.cdf_at(np.asarray(x))), b=BITS, alpha_max=alpha_mm))
  return alpha, type(best).name


def quantize_model(model: ResNet4, method: QuantMethod) -> tuple[ResNet4, list[LayerQuantStats]]:
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
  type(qmodel).clear_jit_caches()
  return qmodel, stats


def collect_layer_outputs(model, x_test: Tensor, batch_size: int = 100) -> ChannelMeansAccumulator:
  acc = ChannelMeansAccumulator()
  n = x_test.shape[0]
  assert n % batch_size == 0, f"test set size {n} must be divisible by batch_size {batch_size}"
  for start in tqdm(range(0, n, batch_size), desc="  activations"):
    idx = Tensor.arange(start, start + batch_size)
    for name, act in model.get_activations(x_test[idx]).items():
      acc.update(name, act.numpy())
  return acc


def run_training(shifts_row_cls: type, n_models: int, steps: int) -> tuple[list[AccuracyRow], list, list[QuantStatsRow]]:
  _, _, x_test, y_test = _load_normalized()

  accuracy_rows: list[AccuracyRow] = []
  shifts_rows: list = []
  quant_stats_rows: list[QuantStatsRow] = []
  t_start = time.perf_counter()
  for seed in range(n_models):
    timings: dict[str, float] = {}
    t = time.perf_counter()
    def lap(name: str) -> None:
      nonlocal t
      now = time.perf_counter()
      timings[name] = now - t
      t = now

    model, fp32_acc, _, _ = train_model(ResNet4, seed=seed, steps=steps)
    lap("train")
    fp32_outputs = collect_layer_outputs(model, x_test)
    lap("fp32_act")
    # Capture FP32 weights in the **fused** frame so they line up with the quantized variants
    # (which are fused-then-quantized inside quantize_model). Bias correction requires
    # W_fp and W_q to be in the same frame; otherwise the BN-fold offset leaks into the
    # `eps = W_q - W_fp` term and the correction over-shoots.
    fp_fused = copy.deepcopy(model)
    fp_fused.fuse()
    fp_modules = dict(fp_fused.named_weight_modules)
    input_stats = compute_input_stats(model)  # BN params (gamma, beta) are unchanged by fusion
    lap("input_stats")

    variants: dict[str, ResNet4] = {}
    layer_stats: dict[QuantMethod, list[LayerQuantStats]] = {}
    for method in QuantMethod:
      ResNet4.clear_jit_caches()
      qmodel, stats = quantize_model(model, method)
      variants[str(method)] = qmodel
      layer_stats[method] = stats
    lap("quantize")

    variants["minmax_bias"] = bias_correct_model(variants["minmax"], fp_modules, input_stats)
    variants["aciq_bias"] = bias_correct_model(variants["aciq"], fp_modules, input_stats)
    lap("bias_correct")

    accs: dict[str, float] = {}
    shifts: dict[str, dict[str, float]] = {}
    for label in VARIANT_LABELS:
      ResNet4.clear_jit_caches()
      qmodel = variants[label]
      accs[label] = float(qmodel.test_acc(x_test, y_test).item())
      q_outputs = collect_layer_outputs(qmodel, x_test)
      shifts[label] = {r.layer: r.mean_shift for r in fp32_outputs.layers_means_shifts(q_outputs, label)}
    lap("measure")

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

    accuracy_rows.append(
      AccuracyRow(
        seed=seed,
        fp32_acc=fp32_acc,
        minmax_acc=accs["minmax"],
        minmax_bias_acc=accs["minmax_bias"],
        aciq_acc=accs["aciq"],
        aciq_bias_acc=accs["aciq_bias"],
      )
    )
    shifts_rows.append(
      shifts_row_cls(
        seed=seed,
        **{f"{v}_{b}_mean_shift": shifts[v][b] for v in VARIANT_LABELS for b in BlockName},
      )
    )
    seed_total = sum(timings.values())
    timing_str = " ".join(f"{k}={v:.1f}s" for k, v in timings.items())
    print(
      f"[{seed + 1}/{n_models}] seed={seed}  total={seed_total:.1f}s  ({timing_str})  "
      f"fp32={fp32_acc:.3f} mm={accs['minmax']:.3f} mm+b={accs['minmax_bias']:.3f} aq={accs['aciq']:.3f} aq+b={accs['aciq_bias']:.3f}"
    )
  elapsed = time.perf_counter() - t_start
  per_seed = elapsed / n_models if n_models else 0.0
  print(f"\nrun_training complete: {n_models} seeds in {elapsed:.1f}s ({per_seed:.1f}s/seed)")
  return accuracy_rows, shifts_rows, quant_stats_rows


def plot_scatter(accuracy_rows: list[AccuracyRow], shifts_rows: list, BlockName: type[StrEnum], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
  for col_idx, (method, color) in enumerate([(QuantMethod.MINMAX, TailwindColor.TEAL), (QuantMethod.ACIQ, TailwindColor.ORANGE)]):
    acc_drops = np.array([r.fp32_acc - getattr(r, f"{method}_acc") for r in accuracy_rows])
    total_shifts = np.sum([_shifts(shifts_rows, method, b) for b in BlockName], axis=0)
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


def plot_paired_diff_histograms(accuracy_rows: list[AccuracyRow], per_network: list[MaePerNetworkRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  acc_diff = np.array([r.aciq_acc - r.minmax_acc for r in accuracy_rows], dtype=np.float64)
  acc_bias_diff = np.array([r.aciq_bias_acc - r.minmax_bias_acc for r in accuracy_rows], dtype=np.float64)
  mae_diff = np.array([r.aciq_mae - r.minmax_mae for r in per_network], dtype=np.float64)
  minmax_bias_gain = np.array([r.minmax_bias_acc - r.minmax_acc for r in accuracy_rows], dtype=np.float64)
  aciq_bias_gain = np.array([r.aciq_bias_acc - r.aciq_acc for r in accuracy_rows], dtype=np.float64)

  panels: list[tuple[np.ndarray, str, str]] = [
    (acc_diff, "ACIQ − MinMax (be korekcijos)", "Tikslumo skirtumas"),
    (acc_bias_diff, "ACIQ − MinMax (po bias korek.)", "Tikslumo skirtumas"),
    (mae_diff, "ACIQ − MinMax MAE", "MAE skirtumas"),
    (minmax_bias_gain, "MinMax: bias prieaugis", "Tikslumo skirtumas"),
    (aciq_bias_gain, "ACIQ: bias prieaugis", "Tikslumo skirtumas"),
  ]

  fig, axes = plt.subplots(2, 3, figsize=(15, 8))
  for ax, (diff, title, xlabel) in zip(axes.flat, panels):
    ax.hist(diff, bins=30, color=TailwindColor.VIOLET, alpha=0.75)
    ax.axvline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.0, label="0")
    median = float(np.median(diff))
    ax.axvline(median, color=TailwindColor.AMBER, linestyle="-", linewidth=1.2, label=f"mediana={median:.3e}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tinklų skaičius")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)
  for ax in axes.flat[len(panels):]:
    ax.set_visible(False)
  fig.tight_layout()
  fig.savefig(save_dir / "paired_diff_histograms.png")
  plt.close(fig)


def plot_minmax_vs_aciq_accuracy(accuracy_rows: list[AccuracyRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fp32 = np.array([r.fp32_acc for r in accuracy_rows])
  mm = np.array([r.minmax_acc for r in accuracy_rows])
  aciq = np.array([r.aciq_acc for r in accuracy_rows])
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
  ax.set_title(f"MINMAX prieš ACIQ tikslumas\nvidurkis: MINMAX={mm.mean():.3f}  ACIQ={aciq.mean():.3f}  ACIQ > MINMAX: {wins_aciq}/{len(accuracy_rows)}")
  ax.grid(False)
  ax.legend(loc="lower right")
  fig.tight_layout()
  fig.savefig(save_dir / "minmax_vs_aciq_accuracy.png")
  plt.close(fig)


def plot_per_layer_shift(shifts_rows: list, BlockName: type[StrEnum], BlockName2: type[StrEnum], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10, 5))
  for color, method in [(TailwindColor.TEAL, QuantMethod.MINMAX), (TailwindColor.ORANGE, QuantMethod.ACIQ)]:
    per_layer_means = [_shifts(shifts_rows, method, b).mean() for b in BlockName]
    per_layer_stds = [_shifts(shifts_rows, method, b).std() for b in BlockName]

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
  ax.set_xticklabels(list(BlockName2))
  ax.set_xlabel("Sluoksnis")
  ax.set_ylabel("Išvesties vidurkio poslinkis")
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_dir / "per_layer_mean_shift.png")
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Quantization distribution-shift analysis on CIFAR-10 with ResNet4")
  parser.add_argument("--n-models", type=int, default=100, help="Number of models to train")
  parser.add_argument("--steps", type=int, default=500, help="Training steps per model")
  parser.add_argument(
    "--from-dir",
    type=Path,
    default=None,
    help="Load `results.csv` from this experiment directory and re-render plots only (no training).",
  )
  args = parser.parse_args()

  ShiftsRow = _make_shifts_row_cls(ResNet4.__name__, BlockName)
  save_dir = get_output_dir(RESULTS_DIR, "cifar10_resnet4_quant_shift")

  quant_stats_rows: list[QuantStatsRow] = []
  if args.from_dir:
    accuracy_rows = load_csv(args.from_dir / "accuracy.csv", AccuracyRow)
    shifts_rows = load_csv(args.from_dir / "shifts.csv", ShiftsRow)
    print(f"Loaded {len(accuracy_rows)} models from {args.from_dir}/")
    quant_stats_path = args.from_dir / "quant_stats.csv"
    if quant_stats_path.exists():
      quant_stats_rows = load_csv(quant_stats_path, QuantStatsRow)
  else:
    print(f"Running training with {args.n_models} models, {args.steps} steps each...")
    accuracy_rows, shifts_rows, quant_stats_rows = run_training(ShiftsRow, args.n_models, args.steps)
    save_csv(accuracy_rows, save_dir / "accuracy.csv")
    save_csv(shifts_rows, save_dir / "shifts.csv")
    if quant_stats_rows:
      save_csv(quant_stats_rows, save_dir / "quant_stats.csv")

  if quant_stats_rows:
    save_csv(build_distribution_counts(quant_stats_rows), save_dir / "distribution_counts.csv")
    save_csv(build_mae_summary(quant_stats_rows), save_dir / "mae_summary.csv")
    per_network = build_mae_per_network(quant_stats_rows)
    save_csv(per_network, save_dir / "mae_per_network.csv")
    save_csv(build_mae_average(per_network, accuracy_rows), save_dir / "mae_average.csv")
    save_csv(build_wilcoxon(accuracy_rows, per_network), save_dir / "wilcoxon.csv")
    plot_paired_diff_histograms(accuracy_rows, per_network, save_dir)
    print(f"Emitted derived summaries to {save_dir}/")

  plot_scatter(accuracy_rows, shifts_rows, BlockName, save_dir)
  plot_per_layer_shift(shifts_rows, BlockName, BlockName2, save_dir)
  plot_minmax_vs_aciq_accuracy(accuracy_rows, save_dir)
  print(f"Plots saved to {save_dir}/")
