import argparse
from dataclasses import dataclass

import numpy as np

from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT, fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir, save_csv
from aciq.models.resnet import ResNet, _weight_modules


@dataclass
class ChannelFitCountsRow:
  layer_idx: int
  layer_name: str
  num_channels: int
  global_best: str
  count_gaussian: int
  count_laplace: int
  count_studentt: int
  count_ged: int


def summarize(model: ResNet) -> list[ChannelFitCountsRow]:
  rows: list[ChannelFitCountsRow] = []
  for layer_idx, (name, module) in enumerate(_weight_modules(model), 1):
    weight = module.weight.numpy()
    num_channels = int(weight.shape[0])
    global_best = type(fit_distributions(weight.flatten().astype(np.float64))[0])
    counts: dict[type[Distribution], int] = {Gaussian: 0, Laplace: 0, StudentT: 0, GeneralizedGaussian: 0}
    for c in range(num_channels):
      chan = weight[c].flatten().astype(np.float64)
      counts[type(fit_distributions(chan)[0])] += 1
    rows.append(
      ChannelFitCountsRow(
        layer_idx=layer_idx,
        layer_name=name,
        num_channels=num_channels,
        global_best=global_best.name,
        count_gaussian=counts[Gaussian],
        count_laplace=counts[Laplace],
        count_studentt=counts[StudentT],
        count_ged=counts[GeneralizedGaussian],
      )
    )
    print(
      f"  [{layer_idx:>3}] {name:40s}  n={num_channels:>4}  global={global_best.name:10s}"
      f"  G={counts[Gaussian]:>3} L={counts[Laplace]:>3} T={counts[StudentT]:>3} GED={counts[GeneralizedGaussian]:>3}"
    )
  return rows


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-channel best-fit distribution counts vs the layer-level best fit.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  args = parser.parse_args()

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()
  rows = summarize(model)

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_channel_fit_counts")
  out_path = save_dir / "channel_fit_counts.csv"
  save_csv(rows, out_path)
  print(f"Wrote {len(rows)} rows to {out_path}")
