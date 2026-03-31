import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


RESULTS_CSV = Path("results/mnist_quantization_shift/results.csv")
BLOCK_NAMES = ["block1", "block2", "block3", "block4", "block5"]


def load_results(path: Path) -> list[dict[str, float]]:
  with open(path) as f:
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main() -> None:
  path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_CSV
  rows = load_results(path)
  print(f"Loaded {len(rows)} models from {path}\n")

  for method in ["minmax", "aciq"]:
    acc_drops = [r["fp32_acc"] - r[f"{method}_acc"] for r in rows]

    print(f"=== {method.upper()} ===")
    print(f"{'layer':<12} {'Spearman rho':>14} {'p-value':>12}")
    print("-" * 40)

    for block in BLOCK_NAMES:
      shifts = [r[f"{method}_{block}_mean_shift"] for r in rows]
      rho, p = spearmanr(shifts, acc_drops)
      print(f"{block:<12} {rho:>14.4f} {p:>12.4g}")

    # Aggregate: sum of shifts across all layers
    total_shifts = [sum(r[f"{method}_{b}_mean_shift"] for b in BLOCK_NAMES) for r in rows]
    rho, p = spearmanr(total_shifts, acc_drops)
    print(f"{'total':<12} {rho:>14.4f} {p:>12.4g}")
    print()

  # Scatter plots
  fig, axes = plt.subplots(2, len(BLOCK_NAMES) + 1, figsize=(4 * (len(BLOCK_NAMES) + 1), 8))
  for row_idx, (method, color) in enumerate([("minmax", "steelblue"), ("aciq", "indianred")]):
    acc_drops = np.array([r["fp32_acc"] - r[f"{method}_acc"] for r in rows])

    for col_idx, block in enumerate(BLOCK_NAMES):
      ax = axes[row_idx, col_idx]
      shifts = np.array([r[f"{method}_{block}_mean_shift"] for r in rows])
      ax.scatter(shifts, acc_drops, color=color, alpha=0.6, s=20)
      rho, p = spearmanr(shifts, acc_drops)
      ax.set_title(f"{method.upper()} {block}\nrho={rho:.3f} p={p:.3g}", fontsize=9)
      ax.set_xlabel("Mean shift", fontsize=8)
      ax.set_ylabel("Accuracy drop", fontsize=8)
      ax.grid(True, alpha=0.3)

    # Aggregate column
    ax = axes[row_idx, len(BLOCK_NAMES)]
    total_shifts = np.array([sum(r[f"{method}_{b}_mean_shift"] for b in BLOCK_NAMES) for r in rows])
    ax.scatter(total_shifts, acc_drops, color=color, alpha=0.6, s=20)
    rho, p = spearmanr(total_shifts, acc_drops)
    ax.set_title(f"{method.upper()} total\nrho={rho:.3f} p={p:.3g}", fontsize=9)
    ax.set_xlabel("Total mean shift", fontsize=8)
    ax.set_ylabel("Accuracy drop", fontsize=8)
    ax.grid(True, alpha=0.3)

  fig.tight_layout()
  save_path = path.parent / "scatter_shift_vs_accuracy.png"
  fig.savefig(save_path, dpi=300)
  plt.close(fig)
  print(f"Scatter plot saved to {save_path}")


if __name__ == "__main__":
  main()
