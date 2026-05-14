import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aciq.distributions import kurtosis, skewness
from aciq.helpers import RESULTS_DIR, get_output_dir, save_csv
from aciq.plotting_style import LINE_WIDTH, NEUTRAL_COLOR, TailwindColor
from aciq.resnet import ResNet, _weight_modules


@dataclass
class WeightMomentsRow:
  layer_idx: int
  layer_name: str
  n: int
  mean: float
  variance: float
  skewness: float
  kurtosis: float


def compute_moments(model: ResNet) -> list[WeightMomentsRow]:
  rows: list[WeightMomentsRow] = []
  for layer_idx, (name, module) in enumerate(_weight_modules(model), 1):
    w = module.weight.numpy().flatten().astype(np.float64)
    rows.append(
      WeightMomentsRow(
        layer_idx=layer_idx,
        layer_name=name,
        n=int(w.size),
        mean=float(np.mean(w)),
        variance=float(np.var(w)),
        skewness=float(skewness(w)),
        kurtosis=float(kurtosis(w)) + 3.0,
      )
    )
  return rows


def plot_moments(rows: list[WeightMomentsRow], out_path: Path) -> None:
  idx = np.array([r.layer_idx for r in rows])
  mean = np.array([r.mean for r in rows])
  variance = np.array([r.variance for r in rows])
  skew = np.array([r.skewness for r in rows])
  kurt = np.array([r.kurtosis for r in rows])

  fig, axes = plt.subplots(2, 2, figsize=(11, 7))
  for ax, values, ylabel in (
    (axes[0, 0], mean, "Mean"),
    (axes[0, 1], variance, "Variance"),
    (axes[1, 0], skew, "Skewness"),
    (axes[1, 1], kurt, "Kurtosis"),
  ):
    ax.bar(idx, values, color=TailwindColor.TEAL)
    ax.set_ylabel(ylabel)

  for ax in (axes[0, 0], axes[1, 0]):
    ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=LINE_WIDTH)
  axes[1, 1].axhline(3.0, color=TailwindColor.RED, linewidth=2.0, label="Gaussian")
  axes[1, 1].legend(loc="upper left")

  for ax in axes[1]:
    ax.set_xlabel("Layer index (forward order)")

  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path)
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-layer weight moments (mean, variance, skewness, kurtosis) for a pre-trained ResNet.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  args = parser.parse_args()

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()
  rows = compute_moments(model)

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_weight_statistics")
  save_csv(rows, save_dir / "moments.csv")
  plot_moments(rows, save_dir / "moments.png")
  print(f"Wrote {len(rows)} rows to {save_dir}/")

  kurt = np.array([r.kurtosis for r in rows])
  skew = np.array([r.skewness for r in rows])
  print(f"  Skewness range: [{skew.min():+.3f}, {skew.max():+.3f}]")
  print(f"  Kurtosis range: [{kurt.min():+.3f}, {kurt.max():+.3f}]")
