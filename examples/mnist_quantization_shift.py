import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn import Conv2d, Linear
from scipy.stats import spearmanr

from aciq.mean_shift import LayerStats, StatsAccumulator, compute_shift
from aciq.distributions import Distribution, DistributionType
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, save_csv
from aciq.mnist import BlockName, MNISTModel, _load_normalized, evaluate_model, train_model
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


# --- Quantization ---


def _weight_modules(model: MNISTModel) -> list[Conv2d | Linear]:
  return [ model.conv1,model.conv2,model.conv3, model.conv4, model.classifier]


def quantize_model(model: MNISTModel, method: str) -> MNISTModel:
  qmodel = copy.deepcopy(model)
  qmodel.fuse()
  for mod in _weight_modules(qmodel):
    w = mod.weight.numpy()
    alpha = _compute_alpha(w.flatten(), method)
    mod.weight = Tensor(quantize(w.flatten(), alpha, BITS).reshape(w.shape).astype(np.float32))
  return qmodel


def _compute_alpha(vec: np.ndarray, method: str) -> float:
  alpha_mm = minmax_alpha(vec)
  if method == "minmax":
    return alpha_mm
  sorted_vec = np.sort(vec)
  fits = {dt: Distribution.fit(sorted_vec, dt) for dt in DistributionType}
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  best_dist = fits[best_type]
  return solve_symmetric_mae_alpha(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=BITS, alpha_max=alpha_mm)


# --- Distribution shift measurement ---


def collect_layer_outputs(model: MNISTModel, x_test: Tensor, batch_size: int = 100) -> dict[str, LayerStats]:
  acc = StatsAccumulator()
  n = x_test.shape[0]
  assert n % batch_size == 0, f"test set size {n} must be divisible by batch_size {batch_size}"
  for start in tqdm(range(0, n, batch_size), desc="  activations"):
    idx = Tensor.arange(start, start + batch_size)
    for name, act in model.get_activations(x_test[idx]).items():
      acc.update(name, act.numpy())
  return acc.finalize()


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

    # MinMax quantization
    mm_model = quantize_model(model, "minmax")
    mm_acc = evaluate_model(mm_model, x_test, y_test)
    mm_outputs = collect_layer_outputs(mm_model, x_test)
    mm_shift = {r.layer: r.mean_shift for r in compute_shift(fp32_outputs, mm_outputs, "minmax")}
    print("MinMax quantization done")

    # ACIQ quantization
    aciq_model = quantize_model(model, "aciq")
    aciq_acc = evaluate_model(aciq_model, x_test, y_test)
    aciq_outputs = collect_layer_outputs(aciq_model, x_test)
    aciq_shift = {r.layer: r.mean_shift for r in compute_shift(fp32_outputs, aciq_outputs, "aciq")}
    print("ACIQ quantization done")

    print(f"  FP32={fp32_acc:.4f}  MinMax={mm_acc:.4f}  ACIQ={aciq_acc:.4f}")
    result_rows.append(MnistResultRow(
      seed=seed,
      fp32_acc=fp32_acc,
      minmax_acc=mm_acc,
      aciq_acc=aciq_acc,
      **{f"minmax_{b}_mean_shift": mm_shift[b] for b in BlockName},
      **{f"aciq_{b}_mean_shift": aciq_shift[b] for b in BlockName},
    ))
    loss_rows.extend(
      MnistLossRow(seed=seed, step=step, train_loss=tl, test_loss=el)
      for step, (tl, el) in enumerate(zip(train_losses, test_losses), start=1)
    )
  return result_rows, loss_rows


# --- Analysis ---


def plot_scatter(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(2, len(BlockName) + 1, figsize=(4 * (len(BlockName) + 1), 8))
  for row_idx, (method, color) in enumerate([("minmax", "steelblue"), ("aciq", "indianred")]):
    acc_drops = np.array([r.fp32_acc - getattr(r, f"{method}_acc") for r in rows])

    for col_idx, block in enumerate(BlockName):
      ax = axes[row_idx, col_idx]
      shifts = np.array([getattr(r, f"{method}_{block}_mean_shift") for r in rows])
      ax.scatter(shifts, acc_drops, color=color, alpha=0.6, s=20)
      rho, p = spearmanr(shifts, acc_drops)
      ax.set_title(f"{method.upper()} {block}\nrho={rho:.3f} p={p:.3g}", fontsize=9)
      ax.set_xlabel("Mean shift", fontsize=8)
      ax.set_ylabel("Accuracy drop", fontsize=8)
      ax.grid(True, alpha=0.3)

    ax = axes[row_idx, len(BlockName)]
    total_shifts = np.array([sum(getattr(r, f"{method}_{b}_mean_shift") for b in BlockName) for r in rows])
    ax.scatter(total_shifts, acc_drops, color=color, alpha=0.6, s=20)
    rho, p = spearmanr(total_shifts, acc_drops)
    ax.set_title(f"{method.upper()} total\nrho={rho:.3f} p={p:.3g}", fontsize=9)
    ax.set_xlabel(f"Total mean shift", fontsize=8)
    ax.set_ylabel("Accuracy drop", fontsize=8)
    ax.grid(True, alpha=0.3)

  fig.suptitle(f"Mean shift vs accuracy drop (Spearman correlation)", fontsize=12, y=1.02)
  fig.tight_layout()
  fig.savefig(save_dir / "scatter_mean_shift_vs_accuracy.png", dpi=700, bbox_inches="tight")
  plt.close(fig)


def plot_shift_accumulation(rows: list[MnistResultRow], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10, 5))
  for color, method in [("steelblue", "minmax"), ("indianred", "aciq")]:
    per_layer_means = [np.mean([getattr(r, f"{method}_{b}_mean_shift") for r in rows]) for b in BlockName]
    per_layer_stds = [np.std([getattr(r, f"{method}_{b}_mean_shift") for r in rows]) for b in BlockName]
    cumulative = np.cumsum(per_layer_means)

    x_pos = np.arange(len(BlockName))
    label = method.upper()
    ax.bar(
      x_pos + (-0.2 if method == "minmax" else 0.2),
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

  fig, ax = plt.subplots(figsize=(8, 5))
  for i, seed in enumerate(sorted(by_seed)):
    seed_rows = sorted(by_seed[seed], key=lambda r: r.step)
    xs = [r.step for r in seed_rows]
    train_losses = [r.train_loss for r in seed_rows]
    test_losses = [r.test_loss for r in seed_rows]
    ax.plot(xs, train_losses, color="steelblue", alpha=0.6, label="Training loss" if i == 0 else None)
    ax.plot(xs, test_losses, color="indianred", alpha=0.6, label="Testing loss" if i == 0 else None)

  ax.set_xlabel("Step")
  ax.set_ylabel("Loss")
  ax.legend()
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
