import argparse
from typing import Any

import numpy as np

from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.models.resnet import ResNet
from examples.resnet_pipeline import analyze_layer


def _resolve_layer_weight(model: ResNet, dotted: str) -> np.ndarray:
  obj: Any = model
  for part in dotted.split("."):
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj.weight.numpy().flatten().astype(np.float32)


def main() -> None:
  parser = argparse.ArgumentParser(description="Fit all candidate distributions to a single ResNet weight tensor and plot.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--layer", type=str, default="layer3.0.conv1", help="Dotted path to a weight-bearing module (e.g. `layer3.0.conv1`, `fc`).")
  parser.add_argument("--bits", type=int, default=8, help="Bit-width for the MinMax/ACIQ clip overlay drawn alongside the fits.")
  args = parser.parse_args()

  print(f"Loading {args.model}...")
  depth = int(args.model.removeprefix("resnet"))
  model = ResNet(depth)
  model.load_from_pretrained()

  weights = _resolve_layer_weight(model, args.layer)
  print(f"Loaded {weights.size:,} weights from {args.layer}")

  out_dir = get_output_dir(RESULTS_DIR, "distribution_fits")
  print(f"Fitting four distributions and rendering {args.bits}-bit overlay...")
  analyze_layer(weights, layer_name=args.layer, layer_idx=0, bits=args.bits, save_path=out_dir)
  print(f"Saved to {out_dir}/")


if __name__ == "__main__":
  main()
