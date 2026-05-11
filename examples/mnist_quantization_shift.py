import argparse
import copy
import gc
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from scipy.stats import spearmanr

from aciq.mean_shift import StatsAccumulator, compute_shift
from aciq.distributions import Distribution, DistributionType
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, save_csv
from aciq.mnist import BlockName, MNISTModel, _load_normalized, train_model
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha


BITS = 4
LOSS_TRACKED_SEEDS = 10


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


def _shifts(rows: list[MnistResultRow], method: str, block: str) -> np.ndarray:
  return np.array([getattr(r, f"{method}_{block}_mean_shift") for r in rows])


# --- Quantization ---


class QuantMethod(StrEnum):
  MINMAX = "minmax"
  ACIQ = "aciq"


def _aciq_alpha(vec: np.ndarray) -> float:
  alpha_mm = minmax_alpha(vec)
  sorted_vec = np.sort(vec)
  fits = {dt: Distribution.fit(sorted_vec, dt) for dt in DistributionType}
  best_dist = fits[max(fits, key=lambda dt: fits[dt].log_likelihood)]
  return solve_symmetric_mae_alpha(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=BITS, alpha_max=alpha_mm)


def quantize_model(model: MNISTModel, method: QuantMethod) -> MNISTModel:
  qmodel = copy.deepcopy(model)
  qmodel.fuse()
  alpha_fn = minmax_alpha if method == QuantMethod.MINMAX else _aciq_alpha
  for mod in qmodel.weight_modules:
    w = mod.weight.numpy()
    alpha = alpha_fn(w.flatten())
    mod.weight = Tensor(quantize(w.flatten(), alpha, BITS).reshape(w.shape).astype(np.float32))
  MNISTModel.clear_jit_caches()
  return qmodel


# --- Distribution shift measurement ---


def collect_layer_outputs(model: MNISTModel, x_test: Tensor, batch_size: int = 100) -> dict[str, np.ndarray]:
  acc = StatsAccumulator()
  n = x_test.shape[0]
  assert n % batch_size == 0, f"test set size {n} must be divisible by batch_size {batch_size}"
  for start in tqdm(range(0, n, batch_size), desc="  activations"):
    idx = Tensor.arange(start, start + batch_size)
    for name, act in model.get_activations(x_test[idx]).items():
      acc.update(name, act.numpy())
  return acc.get_per_channel_means()


# --- Training + measurement ---


def run_training(n_models: int, steps: int) -> tuple[list[MnistResultRow], list[MnistLossRow]]:
  _, _, x_test, y_test = _load_normalized()

  result_rows: list[MnistResultRow] = []
  loss_rows: list[MnistLossRow] = []
  for seed in range(n_models):
    print(f"[{seed + 1}/{n_models}] Training model (seed={seed})...")
    model, fp32_acc, train_losses, test_losses = train_model(seed=seed, steps=steps, gather_losses=seed < LOSS_TRACKED_SEEDS)
    print("Model trained")

    fp32_outputs = collect_layer_outputs(model, x_test)
    print("Collected activations")

    accs: dict[QuantMethod, float] = {}
    shifts: dict[QuantMethod, dict[str, float]] = {}
    for method in QuantMethod:
      qmodel = quantize_model(model, method)
      accs[method] = float(qmodel.test_acc(x_test, y_test).item())
      q_outputs = collect_layer_outputs(qmodel, x_test)
      shifts[method] = {r.layer: r.mean_shift for r in compute_shift(fp32_outputs, q_outputs, method)}
      print(f"{method} quantization done")
      # del qmodel, q_outputs

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
    # del model, fp32_outputs
    MNISTModel.clear_jit_caches()
    # gc.collect()
  return result_rows, loss_rows


# --- Analysis ---


def plot_scatter(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(2, len(BlockName) + 1, figsize=(4 * (len(BlockName) + 1), 8))
  for row_idx, (method, color) in enumerate([(QuantMethod.MINMAX, "steelblue"), (QuantMethod.ACIQ, "indianred")]):
    acc_drops = np.array([r.fp32_acc - getattr(r, f"{method}_acc") for r in rows])

    for col_idx, block in enumerate(BlockName):
      ax = axes[row_idx, col_idx]
      shifts = _shifts(rows, method, block)
      ax.scatter(shifts, acc_drops, color=color, alpha=0.6, s=20)
      rho, p = spearmanr(shifts, acc_drops)
      ax.set_title(f"{method.upper()} {block}\nrho={rho:.3f} p={p:.3g}", fontsize=9)
      ax.set_xlabel("Mean shift", fontsize=8)
      ax.set_ylabel("Accuracy drop", fontsize=8)
      ax.grid(True, alpha=0.3)

    ax = axes[row_idx, len(BlockName)]
    total_shifts = sum(_shifts(rows, method, b) for b in BlockName)
    ax.scatter(total_shifts, acc_drops, color=color, alpha=0.6, s=20)
    rho, p = spearmanr(total_shifts, acc_drops)
    ax.set_title(f"{method.upper()} total\nrho={rho:.3f} p={p:.3g}", fontsize=9)
    ax.set_xlabel("Total mean shift", fontsize=8)
    ax.set_ylabel("Accuracy drop", fontsize=8)
    ax.grid(True, alpha=0.3)

  fig.suptitle("Mean shift vs accuracy drop (Spearman correlation)", fontsize=12, y=1.02)
  fig.tight_layout()
  fig.savefig(save_dir / "scatter_mean_shift_vs_accuracy.png", dpi=700, bbox_inches="tight")
  plt.close(fig)


def plot_shift_accumulation(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10, 5))
  for color, method in [("steelblue", QuantMethod.MINMAX), ("indianred", QuantMethod.ACIQ)]:
    per_layer_means = [_shifts(rows, method, b).mean() for b in BlockName]
    per_layer_stds = [_shifts(rows, method, b).std() for b in BlockName]
    cumulative = np.cumsum(per_layer_means)

    x_pos = np.arange(len(BlockName))
    label = method.upper()
    ax.bar(
      x_pos + (-0.2 if method == QuantMethod.MINMAX else 0.2),
      per_layer_means,
      width=0.35,
      color=color,
      alpha=0.5,
      label=f"{label} per-layer shift",
      yerr=per_layer_stds,
      capsize=3,
    )
    ax.plot(x_pos, cumulative, color=color, marker="o", linewidth=2, linestyle="--", label=f"{label} cumulative shift")
  ax.set_xticks(np.arange(len(BlockName)))
  ax.set_xticklabels(BlockName)
  ax.set_xlabel("Layer")
  ax.set_ylabel("Output mean shift")
  ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(save_dir / "mean_shift_accumulation.png", dpi=700)
  plt.close(fig)


def plot_loss_curves(loss_rows: list[MnistLossRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  by_seed: dict[int, list[MnistLossRow]] = {}
  for r in loss_rows:
    by_seed.setdefault(r.seed, []).append(r)

  cmap = plt.get_cmap("tab10")
  fig, ax = plt.subplots(figsize=(8, 5))
  for i, seed in enumerate(sorted(by_seed)):
    seed_rows = sorted(by_seed[seed], key=lambda r: r.step)
    xs = [r.step for r in seed_rows]
    color = cmap(i % cmap.N)
    ax.plot(xs, [r.train_loss for r in seed_rows], color=color, linestyle="--", alpha=0.8)
    ax.plot(xs, [r.test_loss for r in seed_rows], color=color, linestyle="-", alpha=0.8)

  ax.set_xlabel("Step")
  ax.set_ylabel("Loss")
  ax.legend(
    handles=[
      Line2D([0], [0], color="black", linestyle="-", label="Validation loss"),
      Line2D([0], [0], color="black", linestyle="--", label="Train loss"),
    ]
  )
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(save_dir / "loss_curves.png", dpi=700)
  plt.close(fig)


# --- Main ---


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MNIST quantization distribution shift analysis")
  parser.add_argument("--n-models", type=int, default=100, help="Number of models to train")
  parser.add_argument("--steps", type=int, default=100, help="Training steps per model")
  parser.add_argument("--from-csv", type=Path, default=None, help="Load results from CSV instead of training")
  args = parser.parse_args()
  save_dir = get_output_dir(RESULTS_DIR, "mnist")

  if args.from_csv:
    rows = load_csv(args.from_csv, MnistResultRow)
    print(f"Loaded {len(rows)} models from {args.from_csv}\n")
    losses_path = args.from_csv.parent / "losses.csv"
    loss_rows = load_csv(losses_path, MnistLossRow) if losses_path.exists() else []
  else:
    print(f"Running training with {args.n_models} models, {args.steps} steps each...")
    rows, loss_rows = run_training(args.n_models, args.steps)
    save_csv(rows, save_dir / "results.csv")
    if loss_rows:
      save_csv(loss_rows, save_dir / "losses.csv")

  plot_scatter(rows, save_dir)
  plot_shift_accumulation(rows, save_dir)
  if loss_rows:
    plot_loss_curves(loss_rows, save_dir)
  print(f"Plots saved to {save_dir}/")
