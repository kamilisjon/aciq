import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt

from aciq.helpers import RESULTS_DIR, get_output_dir, save_csv
from aciq.datasets.mnist import train_model as train_mnist
from aciq.datasets.cifar10 import train_model as train_cifar10
from aciq.models import MiniConv, ResNet3BW
from aciq.plotting_style import SERIES_COLORS


@dataclass
class AccuracyRow:
  seed: int
  name: str
  test_accuracy: float


MODEL_MAP = {"miniconv": MiniConv, "resnet3bw": ResNet3BW}
DATASET_MAP = {"mnist": train_mnist, "cifar10": train_cifar10}


def _parse_run(spec: str) -> tuple[int, str]:
  seed, _, name = spec.partition(":")
  if not seed or not name:
    raise argparse.ArgumentTypeError(f"bad --runs entry {spec!r}; expected SEED:NAME")
  return int(seed), name


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train one or more model seeds and plot per-step train/test losses.")
  parser.add_argument("--dataset", choices=list(DATASET_MAP), required=True)
  parser.add_argument("--modeltype", choices=list(MODEL_MAP), required=True)
  parser.add_argument("--steps", type=int, default=100, help="Training steps per run")
  parser.add_argument(
    "--runs", type=_parse_run, nargs="+", required=True, metavar="SEED:NAME", help="One or more SEED:NAME pairs (e.g. 0:fast 1:slow)"
  )
  args = parser.parse_args()

  model_cls = MODEL_MAP[args.modeltype]
  train_fn = DATASET_MAP[args.dataset]
  save_dir = get_output_dir(RESULTS_DIR, f"{args.dataset}_{args.modeltype}_loss_curves")
  save_dir.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(8, 5))
  for i, (seed, name) in enumerate(args.runs):
    print(f"[{i + 1}/{len(args.runs)}] Training {args.modeltype} on {args.dataset} (seed={seed}, name={name})...")
    _, _, train_losses, test_losses = train_fn(model_cls, seed=seed, steps=args.steps, gather_losses=True)
    color = SERIES_COLORS[i % len(SERIES_COLORS)]
    xs = list(range(1, len(train_losses) + 1))
    ax.plot(xs, train_losses, color=color, linestyle="--", alpha=0.8, label=f"{name} (mokymo)")
    ax.plot(xs, test_losses, color=color, linestyle="-", alpha=0.8, label=f"{name} (validacijos)")
    model_cls.clear_jit_caches()

  ax.set_xlabel("Žingsnis")
  ax.set_ylabel("Nuostolis")
  ax.legend()
  fig.tight_layout()
  out_path = save_dir / "loss_curves.png"
  fig.savefig(out_path)
  plt.close(fig)
  print(f"Plot saved to {out_path}")
