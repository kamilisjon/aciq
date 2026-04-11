import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from aciq.analysis import LayerStats, ShiftResult, compute_shift
from aciq.batch_norm import collect_conv_bn_pairs, fuse_bn_into_conv, fuse_bn_into_bias
from aciq.distributions import Distribution, DistributionType
from aciq.mnist_model import MNISTModel, get_mnist_loaders, train_model, evaluate_model
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha


RESULTS_DIR = Path("results/mnist_quantization_shift")
BITS = 4
BLOCK_NAMES = ["block1", "block2", "block3", "block4", "block5"]


# --- Quantization ---


def quantize_model(model: MNISTModel, method: str) -> MNISTModel:
  device = next(model.parameters()).device
  qmodel = copy.deepcopy(model).cpu()
  qmodel.eval()

  for conv_name, conv, bn_name, bn in collect_conv_bn_pairs(qmodel):
    weight = conv.weight.data.numpy()
    fused_w = fuse_bn_into_conv(weight, bn)
    fused_b = fuse_bn_into_bias(conv.bias.data.numpy() if conv.bias is not None else None, bn)

    vec = fused_w.flatten()
    alpha = _compute_alpha(vec, method)
    conv.weight.data = torch.from_numpy(quantize(fused_w, alpha, BITS))
    conv.bias = nn.Parameter(torch.from_numpy(fused_b))

    # BN is fused into conv — remove it
    parent, idx = bn_name.rsplit(".", 1)
    del getattr(qmodel, parent)[int(idx)]

  # Quantize classifier
  w = qmodel.classifier.weight.data.numpy()
  alpha = _compute_alpha(w.flatten(), method)
  qmodel.classifier.weight.data = torch.from_numpy(quantize(w, alpha, BITS))

  return qmodel.to(device)


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


def collect_layer_outputs(model: MNISTModel, test_loader: torch.utils.data.DataLoader, device: str) -> dict[str, LayerStats]:
  """Collect per-channel output means and variances for each block, over all test images."""
  model.eval()
  sums: dict[str, torch.Tensor] = {}
  sq_sums: dict[str, torch.Tensor] = {}
  counts: dict[str, int] = {name: 0 for name in BLOCK_NAMES}
  hooks = []

  def make_hook(name: str):
    def hook_fn(module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
      # out: (B, C, H, W) — sum over B, H, W for per-channel totals
      batch_sum = out.detach().sum(dim=[0, 2, 3])  # (C,)
      batch_sq_sum = (out.detach() ** 2).sum(dim=[0, 2, 3])  # (C,)
      if name not in sums:
        sums[name] = torch.zeros_like(batch_sum, device="cpu")
        sq_sums[name] = torch.zeros_like(batch_sq_sum, device="cpu")
      sums[name] += batch_sum.cpu()
      sq_sums[name] += batch_sq_sum.cpu()
      b, _, h, w = out.shape
      counts[name] += b * h * w

    return hook_fn

  for name in BLOCK_NAMES:
    block = getattr(model, name)
    hooks.append(block.register_forward_hook(make_hook(name)))

  with torch.no_grad():
    for images, _ in test_loader:
      model(images.to(device))

  for h in hooks:
    h.remove()

  result: dict[str, LayerStats] = {}
  for name in BLOCK_NAMES:
    n = counts[name]
    mean = (sums[name] / n).numpy()
    var = ((sq_sums[name] / n) - (sums[name] / n) ** 2).numpy()
    result[name] = LayerStats(mean=mean, var=var)
  return result


# --- Training + measurement ---


@dataclass
class ModelResult:
  seed: int
  fp32_accuracy: float
  minmax_accuracy: float
  aciq_accuracy: float
  minmax_shifts: ShiftResult
  aciq_shifts: ShiftResult


def run_training(n_models: int, epochs: int, device: str) -> list[ModelResult]:
  _, test_loader = get_mnist_loaders()

  results: list[ModelResult] = []
  for seed in range(n_models):
    print(f"[{seed + 1}/{n_models}] Training model (seed={seed})...")
    model, fp32_acc = train_model(seed=seed, epochs=epochs, device=device)
    model.eval()

    fp32_outputs = collect_layer_outputs(model, test_loader, device)

    # MinMax quantization
    mm_model = quantize_model(model, "minmax")
    mm_acc = evaluate_model(mm_model, test_loader, device)
    mm_outputs = collect_layer_outputs(mm_model, test_loader, device)
    mm_shift = compute_shift(fp32_outputs, mm_outputs)

    # ACIQ quantization
    aciq_model = quantize_model(model, "aciq")
    aciq_acc = evaluate_model(aciq_model, test_loader, device)
    aciq_outputs = collect_layer_outputs(aciq_model, test_loader, device)
    aciq_shift = compute_shift(fp32_outputs, aciq_outputs)

    print(f"  FP32={fp32_acc:.4f}  MinMax={mm_acc:.4f}  ACIQ={aciq_acc:.4f}")
    results.append(ModelResult(seed, fp32_acc, mm_acc, aciq_acc, mm_shift, aciq_shift))
  return results


# --- CSV I/O ---


def save_results_csv(results: list[ModelResult], save_path: Path) -> None:
  save_path.parent.mkdir(parents=True, exist_ok=True)
  header = ["seed", "fp32_acc", "minmax_acc", "aciq_acc"]
  for layer in BLOCK_NAMES:
    header += [f"minmax_{layer}_mean_shift", f"aciq_{layer}_mean_shift"]
  for layer in BLOCK_NAMES:
    header += [f"minmax_{layer}_var_shift", f"aciq_{layer}_var_shift"]

  with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in results:
      row: list[float | int] = [r.seed, r.fp32_accuracy, r.minmax_accuracy, r.aciq_accuracy]
      for layer in BLOCK_NAMES:
        row += [r.minmax_shifts.mean_shift[layer], r.aciq_shifts.mean_shift[layer]]
      for layer in BLOCK_NAMES:
        row += [r.minmax_shifts.var_shift[layer], r.aciq_shifts.var_shift[layer]]
      writer.writerow(row)


def load_results_csv(path: Path) -> list[dict[str, float]]:
  with open(path) as f:
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


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
  _plot_scatter_grid(rows, "var_shift", "Variance shift", save_dir, "scatter_var_shift_vs_accuracy.png")


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
  _plot_accumulation(
    rows,
    "var_shift",
    ylabel="Output variance shift |Var[fp32] - Var[quant]|",
    title="Variance shift accumulation across layers",
    save_dir=save_dir,
    filename="var_shift_accumulation.png",
  )


# --- Main ---


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MNIST quantization distribution shift analysis")
  parser.add_argument("--n-models", type=int, default=30, help="Number of models to train")
  parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--from-csv", type=Path, default=None, help="Load results from CSV instead of training")
  args = parser.parse_args()

  save_dir = RESULTS_DIR

  if args.from_csv:
    rows = load_results_csv(args.from_csv)
    print(f"Loaded {len(rows)} models from {args.from_csv}\n")
  else:
    print(f"Running training with {args.n_models} models, {args.epochs} epochs each...")
    results = run_training(args.n_models, args.epochs, args.device)
    csv_path = save_dir / "results.csv"
    save_results_csv(results, csv_path)
    rows = load_results_csv(csv_path)

  plot_scatter(rows, save_dir)
  plot_shift_accumulation(rows, save_dir)
  print(f"Plots saved to {save_dir}/")
