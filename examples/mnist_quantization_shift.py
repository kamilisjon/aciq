import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from aciq.batch_norm import collect_conv_bn_pairs, fuse_bn_into_conv, fuse_bn_into_bias
from aciq.distributions import Distribution, DistributionType
from aciq.mnist_model import MNISTModel, get_mnist_loaders, train_model, evaluate_model
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha


RESULTS_DIR = Path("results/mnist_quantization_shift")
BITS = 4
BLOCK_NAMES = ["block1", "block2", "block3", "block4", "block5"]


# --- Quantization ---


def quantize_model(model: MNISTModel, method: str) -> MNISTModel:
  """Quantize model weights (INT8). Method: 'minmax' or 'aciq'."""
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
  # ACIQ: find best distribution, solve for optimal alpha
  sorted_vec = np.sort(vec)
  fits = {dt: Distribution.fit(sorted_vec, dt) for dt in DistributionType}
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  best_dist = fits[best_type]
  return solve_symmetric_mae_alpha(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=BITS, alpha_max=alpha_mm)


# --- Distribution shift measurement ---


def collect_layer_outputs(model: MNISTModel, test_loader: torch.utils.data.DataLoader, device: str) -> dict[str, np.ndarray]:
  """Run all test images through model, capture mean output of each block per batch, then average."""
  model.eval()
  sums: dict[str, float] = {name: 0.0 for name in BLOCK_NAMES}
  n_batches = 0
  hooks = []

  def make_hook(name: str):
    def hook_fn(module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
      sums[name] += float(out.detach().mean().cpu())
    return hook_fn

  for name in BLOCK_NAMES:
    block = getattr(model, name)
    hooks.append(block.register_forward_hook(make_hook(name)))

  with torch.no_grad():
    for images, _ in test_loader:
      model(images.to(device))
      n_batches += 1

  for h in hooks:
    h.remove()
  return {name: np.asarray(sums[name] / n_batches) for name in BLOCK_NAMES}


def compute_shift(fp32_outputs: dict[str, np.ndarray], quant_outputs: dict[str, np.ndarray]) -> dict[str, float]:
  """Compute output mean shift |E[out_fp32] - E[out_quant]| per layer."""
  return {name: float(np.abs(fp32_outputs[name] - quant_outputs[name])) for name in fp32_outputs}


# --- Correlation analysis ---


@dataclass
class ModelResult:
  seed: int
  fp32_accuracy: float
  minmax_accuracy: float
  aciq_accuracy: float
  minmax_shifts: dict[str, float]
  aciq_shifts: dict[str, float]


def run_analysis(n_models: int, epochs: int, device: str) -> list[ModelResult]:
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
    mm_shifts = compute_shift(fp32_outputs, mm_outputs)

    # ACIQ quantization
    aciq_model = quantize_model(model, "aciq")
    aciq_acc = evaluate_model(aciq_model, test_loader, device)
    aciq_outputs = collect_layer_outputs(aciq_model, test_loader, device)
    aciq_shifts = compute_shift(fp32_outputs, aciq_outputs)

    print(f"  FP32={fp32_acc:.4f}  MinMax={mm_acc:.4f}  ACIQ={aciq_acc:.4f}")
    results.append(ModelResult(seed, fp32_acc, mm_acc, aciq_acc, mm_shifts, aciq_shifts))
  return results


# --- Plotting ---


def plot_shift_vs_accuracy(results: list[ModelResult], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  layer_names = list(results[0].minmax_shifts.keys())

  for layer_name in layer_names:
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, label, get_shifts, get_acc in [
      ("steelblue", "MinMax", lambda r: r.minmax_shifts, lambda r: r.fp32_accuracy - r.minmax_accuracy),
      ("indianred", "ACIQ", lambda r: r.aciq_shifts, lambda r: r.fp32_accuracy - r.aciq_accuracy),
    ]:
      xs = [get_shifts(r)[layer_name] for r in results]
      ys = [get_acc(r) for r in results]
      ax.scatter(xs, ys, color=color, alpha=0.7, label=label, s=30)
      if len(set(xs)) > 1:
        r_val, p_val = pearsonr(xs, ys)
        ax.annotate(f"{label}: r={r_val:.3f} p={p_val:.3f}", xy=(0.02, 0.98 if label == "MinMax" else 0.92),
                    xycoords="axes fraction", fontsize=8, va="top", family="monospace")

    ax.set_xlabel("Output mean shift |E[fp32] - E[quant]|")
    ax.set_ylabel("Accuracy drop (FP32 - quantized)")
    ax.set_title(f"{layer_name}: output mean shift vs accuracy drop")
    ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"shift_vs_accuracy_{layer_name}.png", dpi=500)
    plt.close(fig)

  # Aggregate plot (sum of mean shifts across layers)
  fig, ax = plt.subplots(figsize=(8, 5))
  for color, label, get_shifts, get_acc in [
    ("steelblue", "MinMax", lambda r: r.minmax_shifts, lambda r: r.fp32_accuracy - r.minmax_accuracy),
    ("indianred", "ACIQ", lambda r: r.aciq_shifts, lambda r: r.fp32_accuracy - r.aciq_accuracy),
  ]:
    xs = [sum(get_shifts(r).values()) for r in results]
    ys = [get_acc(r) for r in results]
    ax.scatter(xs, ys, color=color, alpha=0.7, label=label, s=30)
    if len(set(xs)) > 1:
      r_val, p_val = pearsonr(xs, ys)
      ax.annotate(f"{label}: r={r_val:.3f} p={p_val:.3f}", xy=(0.02, 0.98 if label == "MinMax" else 0.92),
                  xycoords="axes fraction", fontsize=8, va="top", family="monospace")

  ax.set_xlabel("Total output mean shift (sum across layers)")
  ax.set_ylabel("Accuracy drop (FP32 - quantized)")
  ax.set_title("Aggregate output mean shift vs accuracy drop")
  ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(save_dir / "shift_vs_accuracy_aggregate.png", dpi=500)
  plt.close(fig)


def plot_shift_accumulation(results: list[ModelResult], save_dir: Path) -> None:
  """Plot how distribution shifts accumulate across layers (mean over all models)."""
  save_dir.mkdir(parents=True, exist_ok=True)
  layer_names = list(results[0].minmax_shifts.keys())

  fig, ax = plt.subplots(figsize=(10, 5))
  for color, label, get_shifts in [
    ("steelblue", "MinMax", lambda r: r.minmax_shifts),
    ("indianred", "ACIQ", lambda r: r.aciq_shifts),
  ]:
    per_layer_means = [np.mean([get_shifts(r)[name] for r in results]) for name in layer_names]
    per_layer_stds = [np.std([get_shifts(r)[name] for r in results]) for name in layer_names]
    cumulative = np.cumsum(per_layer_means)

    x_pos = np.arange(len(layer_names))
    ax.bar(x_pos + (-0.2 if label == "MinMax" else 0.2), per_layer_means, width=0.35,
           color=color, alpha=0.5, label=f"{label} per-layer shift", yerr=per_layer_stds, capsize=3)
    ax.plot(x_pos, cumulative, color=color, marker="o", linewidth=2, linestyle="--", label=f"{label} cumulative shift")

  ax.set_xticks(np.arange(len(layer_names)))
  ax.set_xticklabels(layer_names)
  ax.set_xlabel("Layer")
  ax.set_ylabel("Output mean shift |E[fp32] - E[quant]|")
  ax.set_title("Distribution shift accumulation across layers")
  ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(save_dir / "shift_accumulation.png", dpi=500)
  plt.close(fig)


def plot_layer_correlation(results: list[ModelResult], save_dir: Path) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  layer_names = list(results[0].minmax_shifts.keys())
  first_layer, last_layer = layer_names[0], layer_names[-1]

  fig, ax = plt.subplots(figsize=(8, 5))
  for color, label, get_shifts in [
    ("steelblue", "MinMax", lambda r: r.minmax_shifts),
    ("indianred", "ACIQ", lambda r: r.aciq_shifts),
  ]:
    xs = [get_shifts(r)[first_layer] for r in results]
    ys = [get_shifts(r)[last_layer] for r in results]
    ax.scatter(xs, ys, color=color, alpha=0.7, label=label, s=30)
    if len(set(xs)) > 1:
      r_val, p_val = pearsonr(xs, ys)
      ax.annotate(f"{label}: r={r_val:.3f} p={p_val:.3f}", xy=(0.02, 0.98 if label == "MinMax" else 0.92),
                  xycoords="axes fraction", fontsize=8, va="top", family="monospace")

  ax.set_xlabel(f"{first_layer} output mean shift")
  ax.set_ylabel(f"{last_layer} output mean shift")
  ax.set_title(f"Layer correlation: {first_layer} vs {last_layer}")
  ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(save_dir / "layer_correlation.png", dpi=500)
  plt.close(fig)


# --- CSV export ---


def save_results_csv(results: list[ModelResult], save_path: Path) -> None:
  save_path.parent.mkdir(parents=True, exist_ok=True)
  layer_names = list(results[0].minmax_shifts.keys())
  header = ["seed", "fp32_acc", "minmax_acc", "aciq_acc"]
  for layer in layer_names:
    header += [f"minmax_{layer}_mean_shift", f"aciq_{layer}_mean_shift"]

  with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in results:
      row: list[float | int] = [r.seed, r.fp32_accuracy, r.minmax_accuracy, r.aciq_accuracy]
      for layer in layer_names:
        row += [r.minmax_shifts[layer], r.aciq_shifts[layer]]
      writer.writerow(row)


# --- Main ---

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MNIST quantization distribution shift analysis")
  parser.add_argument("--n-models", type=int, default=30, help="Number of models to train")
  parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  save_dir = RESULTS_DIR
  print(f"Running correlation analysis with {args.n_models} models, {args.epochs} epochs each...")
  results = run_analysis(args.n_models, args.epochs, args.device)
  save_results_csv(results, save_dir / "results.csv")
  plot_shift_vs_accuracy(results, save_dir)
  plot_shift_accumulation(results, save_dir)
  plot_layer_correlation(results, save_dir)
  print(f"Results saved to {save_dir}/")
