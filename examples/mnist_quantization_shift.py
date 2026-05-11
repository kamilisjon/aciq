import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor, TinyJit
from tinygrad.nn import Conv2d, Linear
from scipy.stats import spearmanr

from aciq.analysis import LayerStats, ShiftResult, StatsAccumulator, compute_shift
from aciq.distributions import Distribution, DistributionType
from aciq.helpers import get_output_dir
from aciq.mnist import MNISTModel, _load_normalized, evaluate_model, train_model
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha


RESULTS_DIR = Path("results")
BITS = 4
BLOCK_NAMES = ["block1", "block2", "block3", "block4", "block5"]
TEST_CHUNK_SIZE = 1000


# --- Quantization ---


def _weight_modules(model: MNISTModel) -> list[tuple[str, Conv2d | Linear]]:
  return [
    ("conv1", model.conv1),
    ("conv2", model.conv2),
    ("conv3", model.conv3),
    ("conv4", model.conv4),
    ("conv5", model.conv5),
    ("classifier", model.classifier),
  ]


def quantize_model(model: MNISTModel, method: str) -> MNISTModel:
  qmodel = copy.deepcopy(model)
  qmodel.fuse()

  for _, mod in _weight_modules(qmodel):
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


def collect_layer_outputs(model: MNISTModel, x_test: Tensor) -> dict[str, LayerStats]:
  """Collect per-channel output means and variances for each block over the test set.

  Iterates the test set in chunks of TEST_CHUNK_SIZE so each captured JIT graph holds only
  one chunk's activation memory. StatsAccumulator combines per-chunk channel sums into the
  global per-channel mean and variance in finalize(). Explicit return of
  model.activations[k].realize() avoids the stale-attribute pitfall of reading
  model.activations after JIT replay."""

  @TinyJit
  def get_activations(X: Tensor) -> tuple[Tensor, ...]:
    model(X)
    return tuple(model.activations[k].realize() for k in BLOCK_NAMES)

  assert x_test.shape[0] % TEST_CHUNK_SIZE == 0
  acc = StatsAccumulator()
  for i in range(0, x_test.shape[0], TEST_CHUNK_SIZE):
    acts = get_activations(x_test[i:i + TEST_CHUNK_SIZE])
    for name, act in zip(BLOCK_NAMES, acts):
      acc.update(name, act.numpy())
  return acc.finalize()


# --- Training + measurement ---


@dataclass
class ModelResult:
  seed: int
  fp32_accuracy: float
  minmax_accuracy: float
  aciq_accuracy: float
  minmax_shifts: ShiftResult
  aciq_shifts: ShiftResult
  train_losses: list[float]
  test_losses: list[float]


def run_training(n_models: int, steps: int, eval_every: int) -> list[ModelResult]:
  _, _, x_test, y_test = _load_normalized()

  results: list[ModelResult] = []
  for seed in range(n_models):
    print(f"[{seed + 1}/{n_models}] Training model (seed={seed})...")
    model, fp32_acc, train_losses, test_losses = train_model(seed=seed, steps=steps, eval_every=eval_every)
    print("Model trained")

    fp32_outputs = collect_layer_outputs(model, x_test)
    print("Collected activations")

    # MinMax quantization
    mm_model = quantize_model(model, "minmax")
    mm_acc = evaluate_model(mm_model, x_test, y_test)
    mm_outputs = collect_layer_outputs(mm_model, x_test)
    mm_shift = compute_shift(fp32_outputs, mm_outputs)
    print("MinMax quantization done")

    # ACIQ quantization
    aciq_model = quantize_model(model, "aciq")
    aciq_acc = evaluate_model(aciq_model, x_test, y_test)
    aciq_outputs = collect_layer_outputs(aciq_model, x_test)
    aciq_shift = compute_shift(fp32_outputs, aciq_outputs)
    print("ACIQ quantization done")

    print(f"  FP32={fp32_acc:.4f}  MinMax={mm_acc:.4f}  ACIQ={aciq_acc:.4f}")
    results.append(ModelResult(seed, fp32_acc, mm_acc, aciq_acc, mm_shift, aciq_shift, train_losses, test_losses))
  return results


# --- CSV I/O ---


def save_results_csv(results: list[ModelResult], save_path: Path) -> None:
  save_path.parent.mkdir(parents=True, exist_ok=True)
  header = ["seed", "fp32_acc", "minmax_acc", "aciq_acc"]
  for layer in BLOCK_NAMES:
    header += [f"minmax_{layer}_mean_shift", f"aciq_{layer}_mean_shift"]

  with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in results:
      row: list[float | int] = [r.seed, r.fp32_accuracy, r.minmax_accuracy, r.aciq_accuracy]
      for layer in BLOCK_NAMES:
        row += [r.minmax_shifts.mean_shift[layer], r.aciq_shifts.mean_shift[layer]]
      writer.writerow(row)


def load_results_csv(path: Path) -> list[dict[str, float]]:
  with open(path) as f:
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def save_losses_csv(results: list[ModelResult], save_path: Path, eval_every: int) -> None:
  save_path.parent.mkdir(parents=True, exist_ok=True)
  with open(save_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", "step", "train_loss", "test_loss"])
    writer.writeheader()
    for r in results:
      for idx, (tr, te) in enumerate(zip(r.train_losses, r.test_losses), start=1):
        writer.writerow({"seed": r.seed, "step": idx * eval_every, "train_loss": tr, "test_loss": te})


def load_losses_csv(path: Path) -> list[dict[str, float | int]]:
  with open(path) as f:
    return [
      {
        "seed": int(r["seed"]),
        "step": int(r["step"]),
        "train_loss": float(r["train_loss"]),
        "test_loss": float(r["test_loss"]),
      }
      for r in csv.DictReader(f)
    ]


# --- Analysis ---


def _plot_scatter_grid(rows: list[dict[str, float]], shift_key: str, shift_label: str, save_dir: Path, filename: str) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(2, len(BLOCK_NAMES) + 1, figsize=(4 * (len(BLOCK_NAMES) + 1), 8))
  for row_idx, (method, color) in enumerate([("minmax", "steelblue"), ("aciq", "indianred")]):
    acc_drops = np.array([r["fp32_acc"] - r[f"{method}_acc"] for r in rows])

    for col_idx, block in enumerate(BLOCK_NAMES):
      ax = axes[row_idx, col_idx]
      shifts = np.array([r[f"{method}_{block}_{shift_key}"] for r in rows])
      ax.scatter(shifts, acc_drops, color=color, alpha=0.6, s=20)
      rho, p = spearmanr(shifts, acc_drops)
      ax.set_title(f"{method.upper()} {block}\nrho={rho:.3f} p={p:.3g}", fontsize=9)
      ax.set_xlabel(shift_label, fontsize=8)
      ax.set_ylabel("Accuracy drop", fontsize=8)
      ax.grid(True, alpha=0.3)

    ax = axes[row_idx, len(BLOCK_NAMES)]
    total_shifts = np.array([sum(r[f"{method}_{b}_{shift_key}"] for b in BLOCK_NAMES) for r in rows])
    ax.scatter(total_shifts, acc_drops, color=color, alpha=0.6, s=20)
    rho, p = spearmanr(total_shifts, acc_drops)
    ax.set_title(f"{method.upper()} total\nrho={rho:.3f} p={p:.3g}", fontsize=9)
    ax.set_xlabel(f"Total {shift_label.lower()}", fontsize=8)
    ax.set_ylabel("Accuracy drop", fontsize=8)
    ax.grid(True, alpha=0.3)

  fig.suptitle(f"{shift_label} vs accuracy drop (Spearman correlation)", fontsize=12, y=1.02)
  fig.tight_layout()
  fig.savefig(save_dir / filename, dpi=700, bbox_inches="tight")
  plt.close(fig)


def plot_scatter(rows: list[dict[str, float]], save_dir: Path) -> None:
  _plot_scatter_grid(rows, "mean_shift", "Mean shift", save_dir, "scatter_mean_shift_vs_accuracy.png")


def _plot_accumulation(rows: list[dict[str, float]], shift_key: str, ylabel: str, title: str, save_dir: Path, filename: str) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10, 5))
  for color, method in [("steelblue", "minmax"), ("indianred", "aciq")]:
    per_layer_means = [np.mean([r[f"{method}_{b}_{shift_key}"] for r in rows]) for b in BLOCK_NAMES]
    per_layer_stds = [np.std([r[f"{method}_{b}_{shift_key}"] for r in rows]) for b in BLOCK_NAMES]
    cumulative = np.cumsum(per_layer_means)

    x_pos = np.arange(len(BLOCK_NAMES))
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

  ax.set_xticks(np.arange(len(BLOCK_NAMES)))
  ax.set_xticklabels(BLOCK_NAMES)
  ax.set_xlabel("Layer")
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(save_dir / filename, dpi=700)
  plt.close(fig)


def plot_shift_accumulation(rows: list[dict[str, float]], save_dir: Path) -> None:
  _plot_accumulation(
    rows,
    "mean_shift",
    ylabel="Output mean shift |E[fp32] - E[quant]|",
    title="Mean shift accumulation across layers",
    save_dir=save_dir,
    filename="mean_shift_accumulation.png",
  )


def plot_loss_curves(loss_rows: list[dict[str, float | int]], save_dir: Path, max_lines: int = 10) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  by_seed: dict[int, list[dict[str, float | int]]] = {}
  for r in loss_rows:
    by_seed.setdefault(int(r["seed"]), []).append(r)
  selected_seeds = sorted(by_seed)[:max_lines]

  fig, ax = plt.subplots(figsize=(8, 5))
  for i, seed in enumerate(selected_seeds):
    seed_rows = sorted(by_seed[seed], key=lambda r: r["step"])
    xs = [r["step"] for r in seed_rows]
    train_losses = [r["train_loss"] for r in seed_rows]
    test_losses = [r["test_loss"] for r in seed_rows]
    ax.plot(xs, train_losses, color="steelblue", alpha=0.6, label="Training loss" if i == 0 else None)
    ax.plot(xs, test_losses, color="indianred", alpha=0.6, label="Testing set loss" if i == 0 else None)

  ax.set_xlabel("Step")
  ax.set_ylabel("Cross-entropy loss")
  ax.set_title(f"MNIST training and testing set loss across {len(selected_seeds)} runs")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(save_dir / "loss_curves.png", dpi=700)
  plt.close(fig)


# --- Main ---


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MNIST quantization distribution shift analysis")
  parser.add_argument("--n-models", type=int, default=30, help="Number of models to train")
  parser.add_argument("--steps", type=int, default=1170, help="Training steps per model")
  parser.add_argument("--eval-every", type=int, default=10, help="Test-loss eval cadence (steps)")
  parser.add_argument("--from-csv", type=Path, default=None, help="Load results from CSV instead of training")
  args = parser.parse_args()
  save_dir = get_output_dir(RESULTS_DIR, "mnist")

  if args.from_csv:
    rows = load_results_csv(args.from_csv)
    print(f"Loaded {len(rows)} models from {args.from_csv}\n")
    losses_path = args.from_csv.parent / "losses.csv"
    loss_rows = load_losses_csv(losses_path) if losses_path.exists() else []
  else:
    print(f"Running training with {args.n_models} models, {args.steps} steps each (eval every {args.eval_every})...")
    results = run_training(args.n_models, args.steps, args.eval_every)
    save_results_csv(results, save_dir / "results.csv")
    save_losses_csv(results, save_dir / "losses.csv", args.eval_every)
    rows = load_results_csv(save_dir / "results.csv")
    loss_rows = [
      {"seed": r.seed, "step": (idx + 1) * args.eval_every, "train_loss": r.train_losses[idx], "test_loss": r.test_losses[idx]}
      for r in results
      for idx in range(len(r.train_losses))
    ]

  plot_scatter(rows, save_dir)
  plot_shift_accumulation(rows, save_dir)
  if loss_rows:
    plot_loss_curves(loss_rows, save_dir)
  print(f"Plots saved to {save_dir}/")
