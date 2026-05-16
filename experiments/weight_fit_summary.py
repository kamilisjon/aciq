import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aciq.distributions import fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir, mean_absolute_error, save_csv
from aciq.quantization.clipping import bound_symmetric_aciq_mae, bound_symmetric_minmax, quantize_symmetric
from aciq.resnet import ResNet, _weight_modules


@dataclass
class FitSummaryRow:
  layer_idx: int
  layer_name: str
  best_fit: str
  alpha_minmax: float
  alpha_aciq: float
  mae_minmax: float
  mae_aciq: float


def summarize(model: ResNet, bits: int) -> list[FitSummaryRow]:
  rows: list[FitSummaryRow] = []
  for layer_idx, (name, module) in enumerate(_weight_modules(model), 1):
    vec = module.weight.numpy().flatten().astype(np.float64)
    best_dist = fit_distributions(np.sort(vec))[0]
    alpha_minmax = float(bound_symmetric_minmax(vec))
    alpha_aciq = float(
      bound_symmetric_aciq_mae(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_minmax)
    )
    mae_minmax = mean_absolute_error(vec, quantize_symmetric(vec, alpha_minmax, bits))
    mae_aciq = mean_absolute_error(vec, quantize_symmetric(vec, alpha_aciq, bits))
    rows.append(
      FitSummaryRow(
        layer_idx=layer_idx,
        layer_name=name,
        best_fit=type(best_dist).name,
        alpha_minmax=alpha_minmax,
        alpha_aciq=alpha_aciq,
        mae_minmax=float(mae_minmax),
        mae_aciq=float(mae_aciq),
      )
    )
    print(f"  [{layer_idx:>3}] {name:40s}  best={type(best_dist).name:10s}  amm={alpha_minmax:.4f}  aaciq={alpha_aciq:.4f}  mm={mae_minmax:.3e}  aciq={mae_aciq:.3e}")
  return rows


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Per-layer distribution fit + per-tensor MinMax/ACIQ clipping summary.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--bits", type=int, default=8)
  args = parser.parse_args()

  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()
  rows = summarize(model, args.bits)

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_fit_summary")
  out_path = save_dir / "fit_summary.csv"
  save_csv(rows, out_path)
  print(f"Wrote {len(rows)} rows to {out_path}")
