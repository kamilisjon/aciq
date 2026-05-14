import argparse
import copy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from scipy.stats import spearmanr

from aciq.bias_correction import ChannelMeansAccumulator
from aciq.distributions import fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, save_csv
from aciq.plotting_style import NEUTRAL_COLOR, SERIES_COLORS, TailwindColor
from aciq.mnist import BlockName, MNISTModel, _load_normalized, train_model, BlockName2
from aciq.quantization import bound_symmetric_minmax, quantize_symmetric, bound_symmetric_aciq_mae


BITS = 4
LOSS_TRACKED_SEEDS = 5


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
class MnistLossRow:
  seed: int
  step: int
  train_loss: float
  test_loss: float


@dataclass
class QuantStatsRow:
  seed: int
  layer_name: str
  channel: int
  minmax_alpha: float
  aciq_alpha: float
  aciq_best_fit: str
  minmax_weight_mae: float
  aciq_weight_mae: float


def _shifts(rows: list[MnistResultRow], method: str, block: str) -> np.ndarray:
  return np.array([getattr(r, f"{method}_{block}_mean_shift") for r in rows])


# --- Quantization ---


class QuantMethod(StrEnum):
  MINMAX = "minmax"
  ACIQ = "aciq"


def _aciq_alpha(vec: np.ndarray) -> tuple[float, str]:
  alpha_mm = bound_symmetric_minmax(vec)
  fits = fit_distributions(np.sort(vec))
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  best = fits[best_type]
  alpha = float(bound_symmetric_aciq_mae(cdf=lambda x: float(best.cdf_at(np.asarray(x))), b=BITS, alpha_max=alpha_mm))
  return alpha, best_type.name


@dataclass
class LayerQuantStats:
  layer_name: str
  alpha_per_channel: list[float]
  best_fit_per_channel: list[str]
  mae_per_channel: list[float]


def quantize_model(model: MNISTModel, method: QuantMethod) -> tuple[MNISTModel, list[LayerQuantStats]]:
  qmodel = copy.deepcopy(model)
  qmodel.fuse()
  stats: list[LayerQuantStats] = []
  for name, mod in qmodel.named_weight_modules:
    w = mod.weight.numpy()
    q_buf = np.empty_like(w, dtype=np.float32)
    alphas: list[float] = []
    fits: list[str] = []
    maes: list[float] = []
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
      maes.append(float(np.mean(np.abs(ch_vec - q_vec))))
    mod.weight = Tensor(q_buf)
    stats.append(LayerQuantStats(layer_name=name, alpha_per_channel=alphas, best_fit_per_channel=fits, mae_per_channel=maes))
  MNISTModel.clear_jit_caches()
  return qmodel, stats


# --- Distribution shift measurement ---


def collect_layer_outputs(model: MNISTModel, x_test: Tensor, batch_size: int = 100) -> ChannelMeansAccumulator:
  acc = ChannelMeansAccumulator()
  n = x_test.shape[0]
  assert n % batch_size == 0, f"test set size {n} must be divisible by batch_size {batch_size}"
  for start in tqdm(range(0, n, batch_size), desc="  activations"):
    idx = Tensor.arange(start, start + batch_size)
    for name, act in model.get_activations(x_test[idx]).items():
      acc.update(name, act.numpy())
  return acc


# --- Training + measurement ---


def run_training(n_models: int, steps: int) -> tuple[list[MnistResultRow], list[MnistLossRow], list[QuantStatsRow]]:
  _, _, x_test, y_test = _load_normalized()

  result_rows: list[MnistResultRow] = []
  loss_rows: list[MnistLossRow] = []
  quant_stats_rows: list[QuantStatsRow] = []
  for seed in range(n_models):
    print(f"[{seed + 1}/{n_models}] Training model (seed={seed})...")
    model, fp32_acc, train_losses, test_losses = train_model(seed=seed, steps=steps, gather_losses=seed < LOSS_TRACKED_SEEDS)
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
      for ch, (mm_alpha, aciq_alpha, aciq_fit, mm_mae, aciq_mae) in enumerate(
        zip(
          mm_layer.alpha_per_channel,
          aciq_layer.alpha_per_channel,
          aciq_layer.best_fit_per_channel,
          mm_layer.mae_per_channel,
          aciq_layer.mae_per_channel,
        )
      ):
        quant_stats_rows.append(
          QuantStatsRow(
            seed=seed,
            layer_name=mm_layer.layer_name,
            channel=ch,
            minmax_alpha=mm_alpha,
            aciq_alpha=aciq_alpha,
            aciq_best_fit=aciq_fit,
            minmax_weight_mae=mm_mae,
            aciq_weight_mae=aciq_mae,
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
    loss_rows.extend(
      MnistLossRow(seed=seed, step=step, train_loss=tl, test_loss=el) for step, (tl, el) in enumerate(zip(train_losses, test_losses), start=1)
    )
    MNISTModel.clear_jit_caches()
  return result_rows, loss_rows, quant_stats_rows


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


def plot_loss_curves(loss_rows: list[MnistLossRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  by_seed: dict[int, list[MnistLossRow]] = {}
  for r in loss_rows:
    by_seed.setdefault(r.seed, []).append(r)

  fig, ax = plt.subplots(figsize=(8, 5))
  for i, seed in enumerate(sorted(by_seed)):
    seed_rows = sorted(by_seed[seed], key=lambda r: r.step)
    xs = [r.step for r in seed_rows]
    color = SERIES_COLORS[i % len(SERIES_COLORS)]
    ax.plot(xs, [r.train_loss for r in seed_rows], color=color, linestyle="--", alpha=0.8)
    ax.plot(xs, [r.test_loss for r in seed_rows], color=color, linestyle="-", alpha=0.8)

  ax.set_xlabel("Žingsnis")
  ax.set_ylabel("Nuostolis")
  ax.legend(
    handles=[
      Line2D([0], [0], color="black", linestyle="-", label="Validacijos nuostolis"),
      Line2D([0], [0], color="black", linestyle="--", label="Mokymo nuostolis"),
    ]
  )
  fig.tight_layout()
  fig.savefig(save_dir / "loss_curves.png")
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
    help="Load `results.csv` and `losses.csv` from this experiment directory and re-render plots only (no training).",
  )
  args = parser.parse_args()
  save_dir = get_output_dir(RESULTS_DIR, "mnist")

  if args.from_dir:
    rows = load_csv(args.from_dir / "results.csv", MnistResultRow)
    print(f"Loaded {len(rows)} models from {args.from_dir / 'results.csv'}")
    losses_path = args.from_dir / "losses.csv"
    loss_rows = load_csv(losses_path, MnistLossRow) if losses_path.exists() else []
  else:
    print(f"Running training with {args.n_models} models, {args.steps} steps each...")
    rows, loss_rows, quant_stats_rows = run_training(args.n_models, args.steps)
    save_csv(rows, save_dir / "results.csv")
    if loss_rows:
      save_csv(loss_rows, save_dir / "losses.csv")
    if quant_stats_rows:
      save_csv(quant_stats_rows, save_dir / "quant_stats.csv")

  plot_scatter(rows, save_dir)
  plot_per_layer_shift(rows, save_dir)
  plot_minmax_vs_aciq_accuracy(rows, save_dir)
  if loss_rows:
    plot_loss_curves(loss_rows, save_dir)
  print(f"Plots saved to {save_dir}/")
