import argparse

import matplotlib.pyplot as plt
import numpy as np

from typing import Any

from aciq.distributions import DIST_COLORS, DistributionType
from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.resnet import ResNet


def _resolve_layer_weight(model: ResNet, dotted: str) -> np.ndarray:
  obj: Any = model
  for part in dotted.split("."):
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj.weight.numpy().flatten().astype(np.float32)


def main() -> None:
  parser = argparse.ArgumentParser(description="Fit all candidate distributions to a single ResNet weight tensor and plot.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--layer", type=str, default="layer4.1.conv2", help="Dotted path to a weight-bearing module (e.g. `layer3.0.conv1`, `fc`).")
  args = parser.parse_args()

  depth = int(args.model.removeprefix("resnet"))
  model = ResNet(depth)
  model.load_from_pretrained()

  weights = _resolve_layer_weight(model, args.layer)
  sorted_weights = np.sort(weights)
  fits = {cls: cls(sorted_weights) for cls in DistributionType}

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.hist(weights, bins=300, density=True, alpha=0.45, color="#94A3B8", edgecolor="white", linewidth=0.3, label="Empirical")
  for cls in sorted(DistributionType, key=lambda c: c.__name__):
    fitted = fits[cls]
    ax.plot(sorted_weights, fitted.pdf(), color=DIST_COLORS[cls], linewidth=1.6, label=f"{cls.__name__}    ll={fitted.log_likelihood:.4g}")
  ax.set_xlabel("Weight value")
  ax.set_ylabel("Density")
  ax.set_title(f"{args.model} · {args.layer} · n={weights.size:,}", fontsize=11)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.grid(True, alpha=0.25, linewidth=0.6)
  ax.legend(frameon=False, fontsize=9, prop={"family": "monospace", "size": 8})
  fig.tight_layout()

  out_dir = get_output_dir(RESULTS_DIR, "distribution_fits")
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"{args.model}_{args.layer.replace('.', '_')}.png"
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved to {out_path}")


if __name__ == "__main__":
  main()
