import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, save_csv
from aciq.datasets.mnist import _load_normalized, train_model
from aciq.models import ResNet3BW
from aciq.models.resnet3bw import BlockName, BlockName2
from aciq.plotting_style import NEUTRAL_COLOR, TailwindColor

from experiments.mnist_pipeline import (
  QuantMethod,
  QuantStatsRow,
  build_distribution_counts,
  build_mae_average,
  build_mae_per_network,
  build_mae_summary,
  collect_layer_outputs,
  plot_minmax_vs_aciq_accuracy,
  quantize_model,
)


@dataclass
class ResNet3ResultRow:
  seed: int
  fp32_acc: float
  minmax_acc: float
  aciq_acc: float
  minmax_stem_mean_shift: float
  minmax_block1_activation_1_mean_shift: float
  minmax_block1_activation_2_mean_shift: float
  aciq_stem_mean_shift: float
  aciq_block1_activation_1_mean_shift: float
  aciq_block1_activation_2_mean_shift: float


def _shifts(rows: list[ResNet3ResultRow], method: str, block: str) -> np.ndarray:
  return np.array([getattr(r, f"{method}_{block}_mean_shift") for r in rows])


def run_training(n_models: int, steps: int) -> tuple[list[ResNet3ResultRow], list[QuantStatsRow]]:
  _, _, x_test, y_test = _load_normalized()

  result_rows: list[ResNet3ResultRow] = []
  quant_stats_rows: list[QuantStatsRow] = []
  for seed in range(n_models):
    print(f"[{seed + 1}/{n_models}] Training model (seed={seed})...")
    model, fp32_acc, _, _ = train_model(ResNet3BW, seed=seed, steps=steps)
    print("Model trained")

    fp32_outputs = collect_layer_outputs(model, x_test)
    print("Collected activations")

    accs: dict[QuantMethod, float] = {}
    shifts: dict[QuantMethod, dict[str, float]] = {}
    layer_stats = {}
    for method in QuantMethod:
      ResNet3BW.clear_jit_caches()
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
      ResNet3ResultRow(
        seed=seed,
        fp32_acc=fp32_acc,
        minmax_acc=accs[QuantMethod.MINMAX],
        aciq_acc=accs[QuantMethod.ACIQ],
        **{f"{m}_{b}_mean_shift": shifts[m][b] for m in QuantMethod for b in BlockName},
      )
    )
  return result_rows, quant_stats_rows


def plot_scatter(rows: list[ResNet3ResultRow], save_dir: Path) -> None:
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


def plot_per_layer_shift(rows: list[ResNet3ResultRow], save_dir: Path) -> None:
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
  ax.set_xticklabels(list(BlockName2))
  ax.set_xlabel("Sluoksnis")
  ax.set_ylabel("Išvesties vidurkio poslinkis")
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_dir / "per_layer_mean_shift.png")
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ResNet3BW quantization distribution shift analysis")
  parser.add_argument("--n-models", type=int, default=100, help="Number of models to train")
  parser.add_argument("--steps", type=int, default=100, help="Training steps per model")
  parser.add_argument(
    "--from-dir",
    type=Path,
    default=None,
    help="Load `results.csv` from this experiment directory and re-render plots only (no training).",
  )
  args = parser.parse_args()
  save_dir = get_output_dir(RESULTS_DIR, "resnet3bw")

  quant_stats_rows: list[QuantStatsRow] = []
  if args.from_dir:
    rows = load_csv(args.from_dir / "results.csv", ResNet3ResultRow)
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
