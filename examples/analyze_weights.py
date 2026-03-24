import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import onnx

from aciq.onnx_io import extract_tensors
from aciq.distributions import Distribution, DistributionType, kurtosis, skewness
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha


RESULTS_DIR = Path("results")
BITS = 8

# TODO: should group layers by which model block that are in. What blocks does ResNet have?
#       Perhaps should group by what activation function is applied?

models: dict[str, Path] = {"resnet50": Path("models/resnet50_Opset18.onnx")}

DIST_COLORS = {
  DistributionType.GAUSSIAN: "red",
  DistributionType.LAPLACE: "green",
  DistributionType.STUDENT_T: "orange",
  DistributionType.GENERALIZED_GAUSSIAN: "blue",
}


def analyze_layer(vec: np.ndarray, layer_name: str, layer_idx: int, bits: int, save_path: Path | None = None) -> tuple[float, float]:
  vec_sorted = np.sort(vec)

  # Distribution fits
  fits: dict[DistributionType, Distribution] = {}
  for dist_type in DistributionType:
    fits[dist_type] = Distribution.fit(vec_sorted, dist_type)

  # MinMax quantization
  alpha = minmax_alpha(vec)
  mae_minmax = float(np.mean(np.abs(vec - quantize(vec, alpha, bits))))

  # Optimal alpha*
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  best_dist = fits[best_type]
  alpha_aciq = solve_symmetric_mae_alpha(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha)
  mae_aciq = float(np.mean(np.abs(vec - quantize(vec, alpha_aciq, bits))))

  if save_path is not None:
    save_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vec, bins=300, density=True, alpha=0.5, color="steelblue", label="Empirical")

    for dist_type, fitted in fits.items():
      ax.plot(vec_sorted, fitted.pdf(), color=DIST_COLORS[dist_type], linewidth=0.7, linestyle="--",
              label=f"{repr(fitted):30s} ll={fitted.log_likelihood:.3g}")

    ax.axvline(-alpha, color="grey", linestyle=":", linewidth=1.2, label=f"MinMax α={alpha:.2f} MAE={mae_minmax:.2e}")
    ax.axvline(alpha, color="grey", linestyle=":", linewidth=1.2)

    if alpha_aciq != alpha:
      ax.axvline(-alpha_aciq, color=DIST_COLORS[best_type], linestyle="-", linewidth=0.7,
                 label=f"CLIP {repr(best_dist)} α={alpha_aciq:.2f} MAE={mae_aciq:.2e}")
      ax.axvline(alpha_aciq, color=DIST_COLORS[best_type], linestyle="-", linewidth=0.7)

    eda_lines = [
      f"n        = {vec.size:,}",
      f"Min      = {float(np.min(vec)):.5f}",
      f"Max      = {float(np.max(vec)):.5f}",
      f"Mean     = {float(np.mean(vec)):.5f}",
      f"Variance = {float(np.var(vec)):.6f}",
      f"Skewness = {float(skewness(vec)):.4f}",
      f"Kurtosis = {float(kurtosis(vec)):.4f}",
    ]
    ax.text(0.98, 0.96, "\n".join(eda_lines), transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right", multialignment="left", bbox=dict(facecolor="lightgrey"), family="monospace")
    safe = layer_name.replace("/", "_").replace(":", "_")
    ax.set_title(f"Layer {layer_idx}: {layer_name} ({bits}bit)", fontsize=10)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7.5, loc="upper left", prop={"family": "monospace", "size": 7.5})
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / f"layer_{layer_idx:03d}_{safe[:60]}.png", dpi=500)
    plt.close(fig)

  return mae_minmax, mae_aciq


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--plot", action="store_true", help="Generate distribution plots")
  args = parser.parse_args()

  if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)
  for model_name, model_path in models.items():
    if model_name == "bert":
      continue
    results_dir = RESULTS_DIR / model_name
    model = onnx.load(str(model_path))
    nodes, tensors = list(model.graph.node), extract_tensors(model)
    print(f"[{model_name}] Total nodes: {len(nodes)}")

    csv_path = results_dir / "mae_comparison.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["layer_idx", "op_type", "name", "n", "mae_per_layer", "mae_aciq", "mae_per_channel", "mae_per_channel_aciq"])

    layer_idx = 0
    for node in nodes:
      match node.op_type:
        case "Conv":
          weight_name = node.input[1]
          weight_arr = onnx.numpy_helper.to_array(tensors[weight_name]).astype(np.float32)
          layer_idx += 1
          safe_name = weight_name.replace("/", "_").replace(":", "_")[:60]

          # Per-tensor
          vec = weight_arr.flatten()
          plot_dir = results_dir / "per_tensor" if args.plot else None
          mae_mm, mae_ac = analyze_layer(vec, weight_name, layer_idx, BITS, plot_dir)

          # Per-channel (axis 0 = output channels)
          ch_plot_dir = results_dir / "per_channel" / f"{layer_idx:03d}_{safe_name}" if args.plot else None
          total_err_mm = 0.0
          total_err_ac = 0.0
          total_n = 0
          for ch in range(weight_arr.shape[0]):
            ch_vec = weight_arr[ch].flatten()
            ch_mae_mm, ch_mae_ac = analyze_layer(ch_vec, f"{weight_name}/ch{ch}", ch, BITS, ch_plot_dir)
            total_err_mm += ch_mae_mm * ch_vec.size
            total_err_ac += ch_mae_ac * ch_vec.size
            total_n += ch_vec.size
          mae_ch = total_err_mm / total_n
          mae_ch_ac = total_err_ac / total_n

          print(f"[{layer_idx:>3}] Conv   {weight_name:50} n={len(vec):,}  MAE: layer={mae_mm:.2e}  aciq={mae_ac:.2e}  per_ch={mae_ch:.2e}  per_ch_aciq={mae_ch_ac:.2e}")
          writer.writerow([layer_idx, "Conv", weight_name, len(vec), mae_mm, mae_ac, mae_ch, mae_ch_ac])

        case "Gemm":
          weight_name = node.input[1]
          vec = onnx.numpy_helper.to_array(tensors[weight_name]).flatten().astype(np.float32)
          layer_idx += 1
          plot_dir = results_dir / "per_tensor" if args.plot else None
          mae_mm, mae_ac = analyze_layer(vec, weight_name, layer_idx, BITS, plot_dir)

          print(f"[{layer_idx:>3}] Gemm   {weight_name:50} n={len(vec):,}  MAE: layer={mae_mm:.2e}  aciq={mae_ac:.2e}")
          writer.writerow([layer_idx, "Gemm", weight_name, len(vec), mae_mm, mae_ac, "", ""])

        case _:
          continue

    csv_file.close()
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
  main()
