import argparse

import numpy as np
import matplotlib.pyplot as plt

from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.mnist import train_model
from aciq.plotting_style import TailwindColor, capped_savefig_dpi


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train one MNIST model and plot its loss curve.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--steps", type=int, default=300)
  args = parser.parse_args()

  print(f"Training (seed={args.seed}, steps={args.steps})...")
  _, accuracy, train_losses, test_losses = train_model(seed=args.seed, steps=args.steps, gather_losses=True)
  print(f"FP32 test accuracy: {accuracy:.4f}")

  save_dir = get_output_dir(RESULTS_DIR, "mnist_single_train")
  save_dir.mkdir(parents=True, exist_ok=True)

  steps = np.arange(1, len(train_losses) + 1)
  fig_w, fig_h = 8.0, 5.0
  fig, ax = plt.subplots(figsize=(fig_w, fig_h))
  ax.plot(steps, train_losses, color=TailwindColor.BLUE, linestyle="--", label="Mokymo nuostolis")
  ax.plot(steps, test_losses, color=TailwindColor.ORANGE, linestyle="-", label="Validacijos nuostolis")
  ax.set_xlabel("Žingsnis")
  ax.set_ylabel("Nuostolis")
  ax.legend(loc="upper right")
  fig.tight_layout()
  fig.savefig(save_dir / "loss_curve.png", dpi=capped_savefig_dpi(fig_w, fig_h))
  plt.close(fig)
  print(f"Saved {save_dir}/loss_curve.png")
