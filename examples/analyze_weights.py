import argparse
import copy
import csv
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import onnx

from aciq.onnx_io import extract_tensors
from aciq.distributions import Distribution, DistributionType, kurtosis, skewness
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha
from aciq.benchmark import run_benchmark


DATASET_PATH = Path("/home/kamilis/Downloads/imagenet-object-localization-challenge")
RESULTS_DIR = Path("results")
BITS = 8

# TODO: should group layers by which model block that are in. What blocks does ResNet have?
#       Perhaps should group by what activation function is applied?

models: dict[str, Path] = {"resnet18": Path("models/resnet18_Opset18.onnx"), "resnet50": Path("models/resnet50_Opset18.onnx")}

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
  alpha_minmax = minmax_alpha(vec)
  mae_minmax = float(np.mean(np.abs(vec - quantize(vec, alpha_minmax, bits))))

  # Optimal alpha*
  best_type = max(fits, key=lambda dt: fits[dt].log_likelihood)
  best_dist = fits[best_type]
  alpha_aciq = solve_symmetric_mae_alpha(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_minmax)

  if save_path is not None:
    save_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vec, bins=300, density=True, alpha=0.5, color="steelblue", label="Empirical")

    for dist_type, fitted in fits.items():
      ax.plot(
        vec_sorted,
        fitted.pdf(),
        color=DIST_COLORS[dist_type],
        linewidth=0.7,
        linestyle="--",
        label=f"{repr(fitted):30s} ll={fitted.log_likelihood:.3g}",
      )

    ax.axvline(-alpha_minmax, color="grey", linestyle=":", linewidth=1.2, label=f"MinMax α={alpha_minmax:.2f} MAE={mae_minmax:.2e}")
    ax.axvline(alpha_minmax, color="grey", linestyle=":", linewidth=1.2)

    if alpha_aciq != alpha_minmax:
      mae_aciq = float(np.mean(np.abs(vec - quantize(vec, alpha_aciq, bits))))
      ax.axvline(
        -alpha_aciq, color=DIST_COLORS[best_type], linestyle="-", linewidth=0.7, label=f"CLIP {repr(best_dist)} α={alpha_aciq:.2f} MAE={mae_aciq:.2e}"
      )
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
    ax.text(
      0.98,
      0.96,
      "\n".join(eda_lines),
      transform=ax.transAxes,
      fontsize=7.5,
      va="top",
      ha="right",
      multialignment="left",
      bbox=dict(facecolor="lightgrey"),
      family="monospace",
    )
    safe = layer_name.replace("/", "_").replace(":", "_")
    ax.set_title(f"Layer {layer_idx}: {layer_name} ({bits}bit)", fontsize=10)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7.5, loc="upper left", prop={"family": "monospace", "size": 7.5})
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / f"layer_{layer_idx:03d}_{safe[:60]}.png", dpi=500)
    plt.close(fig)

  return alpha_minmax, alpha_aciq


def replace_weight(model: onnx.ModelProto, name: str, new_data: np.ndarray):
  for i, init in enumerate(model.graph.initializer):
    if init.name == name:
      model.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(new_data, name=name))
      return


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--plot", action="store_true", help="Generate distribution plots")
  parser.add_argument("--benchmark", action="store_true", help="Benchmark onnx models")
  args = parser.parse_args()

  if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)
  for model_name, model_path in models.items():
    results_dir = RESULTS_DIR / model_name
    model = onnx.load(str(model_path))
    nodes, tensors = list(model.graph.node), extract_tensors(model)
    print(f"[{model_name}] Total nodes: {len(nodes)}")

    fq_models = {name: copy.deepcopy(model) for name in ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]}

    csv_path = results_dir / "mae_comparison.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["layer_idx", "op_type", "name", "n", "n_ch", "err", "err_aciq", "err_channel", "err_channel_aciq"])

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
          alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, BITS, plot_dir)
          quant_weight_minmax = quantize(vec, alpha_minmax, BITS)
          quant_weight_aciq = quantize(vec, alpha_aciq, BITS)

          # Per-channel (axis 0 = output channels)
          ch_plot_dir = results_dir / "per_channel" / f"{layer_idx:03d}_{safe_name}" if args.plot else None
          total_err_minmax = 0.0
          total_err_aciq = 0.0
          fq_ch_mm = np.empty_like(weight_arr)
          fq_ch_ac = np.empty_like(weight_arr)
          for ch in range(weight_arr.shape[0]):
            ch_vec = weight_arr[ch].flatten()
            ch_alpha_minmax, ch_alpha_aciq = analyze_layer(ch_vec, f"{weight_name}/ch{ch}", ch, BITS, ch_plot_dir)
            quant_weight_minmax_ch = quantize(ch_vec, ch_alpha_minmax, BITS)
            quant_weight_aciq_ch = quantize(ch_vec, ch_alpha_aciq, BITS)
            total_err_minmax += np.sum(np.abs(ch_vec - quant_weight_minmax_ch))
            total_err_aciq += np.sum(np.abs(ch_vec - quant_weight_aciq_ch))
            fq_ch_mm[ch] = quant_weight_minmax_ch.reshape(weight_arr[ch].shape)
            fq_ch_ac[ch] = quant_weight_aciq_ch.reshape(weight_arr[ch].shape)

          replace_weight(fq_models["per_tensor_minmax"], weight_name, quant_weight_minmax.reshape(weight_arr.shape))
          replace_weight(fq_models["per_tensor_aciq"], weight_name, quant_weight_aciq.reshape(weight_arr.shape))
          replace_weight(fq_models["per_channel_minmax"], weight_name, fq_ch_mm)
          replace_weight(fq_models["per_channel_aciq"], weight_name, fq_ch_ac)

          print(f"[{layer_idx:>3}] Conv   {weight_name:50} n={len(vec):,}")
          writer.writerow([
            layer_idx,
            "Conv",
            weight_name,
            len(vec),
            len(ch_vec),
            np.sum(np.abs(vec - quant_weight_minmax)),
            np.sum(np.abs(vec - quant_weight_aciq)),
            total_err_minmax,
            total_err_aciq,
          ])

        case "Gemm":
          weight_name = node.input[1]
          weight_arr = onnx.numpy_helper.to_array(tensors[weight_name]).astype(np.float32)
          vec = weight_arr.flatten()
          layer_idx += 1
          plot_dir = results_dir / "per_tensor" if args.plot else None
          alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, BITS, plot_dir)
          quant_weight_minmax = quantize(vec, alpha_minmax, BITS)
          quant_weight_aciq = quantize(vec, alpha_aciq, BITS)

          replace_weight(fq_models["per_tensor_minmax"], weight_name, quant_weight_minmax.reshape(weight_arr.shape))
          replace_weight(fq_models["per_tensor_aciq"], weight_name, quant_weight_aciq.reshape(weight_arr.shape))
          total_err_minmax = 0.0
          total_err_aciq = 0.0
          fq_ch_mm = np.empty_like(weight_arr)
          fq_ch_ac = np.empty_like(weight_arr)
          for ch in range(weight_arr.shape[0]):
            ch_vec = weight_arr[ch].flatten()
            ch_alpha_minmax, ch_alpha_aciq = analyze_layer(ch_vec, f"{weight_name}/ch{ch}", ch, BITS)
            quant_weight_minmax_ch = quantize(ch_vec, ch_alpha_minmax, BITS)
            quant_weight_aciq_ch = quantize(ch_vec, ch_alpha_aciq, BITS)
            total_err_minmax += np.sum(np.abs(ch_vec - quant_weight_minmax_ch))
            total_err_aciq += np.sum(np.abs(ch_vec - quant_weight_aciq_ch))
            fq_ch_mm[ch] = quant_weight_minmax_ch.reshape(weight_arr[ch].shape)
            fq_ch_ac[ch] = quant_weight_aciq_ch.reshape(weight_arr[ch].shape)
          replace_weight(fq_models["per_channel_minmax"], weight_name, fq_ch_mm)
          replace_weight(fq_models["per_channel_aciq"], weight_name, fq_ch_ac)

          print(f"[{layer_idx:>3}] Gemm   {weight_name:50} n={len(vec):,}")
          writer.writerow([
            layer_idx,
            "Gemm",
            weight_name,
            len(vec),
            len(ch_vec),
            np.sum(np.abs(vec - quant_weight_minmax)),
            np.sum(np.abs(vec - quant_weight_aciq)),
            total_err_minmax,
            total_err_aciq,
          ])

        case _:
          continue

    csv_file.close()
    print(f"CSV written to {csv_path}")

    if args.benchmark:
      print(run_benchmark(model_path, DATASET_PATH, batch_size=1))
    for name, fq_model in fq_models.items():
      save_path = results_dir / f"{model_name}_{name}_{BITS}bit.onnx"
      onnx.save(fq_model, str(save_path))
      print(f"Saved {save_path}.")
      if args.benchmark:
        print(run_benchmark(save_path, DATASET_PATH, batch_size=1))


if __name__ == "__main__":
  main()
