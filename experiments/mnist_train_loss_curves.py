import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt

from aciq.helpers import RESULTS_DIR, get_output_dir, save_csv
from aciq.datasets.mnist import train_model
from aciq.models import MiniConv
from aciq.plotting_style import SERIES_COLORS


@dataclass
class AccuracyRow:
  seed: int
  name: str
  test_accuracy: float


def _parse_run(spec: str) -> tuple[int, str]:
  seed, _, name = spec.partition(":")
  if not seed or not name:
    raise argparse.ArgumentTypeError(f"bad --runs entry {spec!r}; expected SEED:NAME")
  return int(seed), name


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train MiniConv on MNIST for one or more seeds and plot per-step train/test losses.")
  parser.add_argument("--steps", type=int, default=100, help="Training steps per run")
  parser.add_argument(
    "--runs", type=_parse_run, nargs="+", required=True, metavar="SEED:NAME", help="One or more SEED:NAME pairs (e.g. 0:fast 1:slow)"
  )
  args = parser.parse_args()

  save_dir = get_output_dir(RESULTS_DIR, "mnist_miniconv_loss_curves")
  save_dir.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(8, 5))
  accuracy_rows: list[AccuracyRow] = []
  for i, (seed, name) in enumerate(args.runs):
    print(f"[{i + 1}/{len(args.runs)}] Training MiniConv on MNIST (seed={seed}, name={name})...")
    _, test_acc, train_losses, test_losses = train_model(MiniConv, seed=seed, steps=args.steps, gather_losses=True)
    print(f"  final test accuracy: {test_acc:.4f}")
    accuracy_rows.append(AccuracyRow(seed=seed, name=name, test_accuracy=float(test_acc)))
    color = SERIES_COLORS[i % len(SERIES_COLORS)]
    xs = list(range(1, len(train_losses) + 1))
    ax.plot(xs, train_losses, color=color, linestyle="--", alpha=0.8, label=f"{name} (train)")
    ax.plot(xs, test_losses, color=color, linestyle="-", alpha=0.8, label=f"{name} (validation)")
    MiniConv.clear_jit_caches()

  ax.set_xlabel("Step")
  ax.set_ylabel("Loss")
  ax.legend()
  fig.tight_layout()
  out_path = save_dir / "loss_curves.png"
  fig.savefig(out_path)
  plt.close(fig)
  save_csv(accuracy_rows, save_dir / "accuracy.csv")
  print(f"Plot saved to {out_path}")
  print(f"Accuracy saved to {save_dir / 'accuracy.csv'}")
