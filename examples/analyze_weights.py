import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import onnx

from aciq.onnx_io import load_onnx, extract_layers
from aciq.distributions import Distribution, DistributionType, kurtosis, skewness
from aciq.quantization import minmax_alpha, quantize


RESULTS_DIR = Path("results")
BITS = [4, 8]

# TODO: should group layers by which model block that are in. What blocks does ResNet have?
#       Perhaps should group by what activation function is applied?
# TODO: why some layers of Bert have Add with input weights and other have encapsulated weights? How to unify?
# TODO: add this file testing to CI. Will need to add models auto download and will need to study GitHub caching.

models: dict[str, Path] = {"resnet50": Path("models/resnet50_Opset18.onnx"), "bert": Path("models/bert_Opset18.onnx")}

DIST_COLORS = {
  DistributionType.GAUSSIAN: "red",
  DistributionType.LAPLACE: "green",
  DistributionType.STUDENT_T: "orange",
  DistributionType.GENERALIZED_GAUSSIAN: "blue",
}


def plot_layer(vec: np.ndarray, layer_name: str, layer_idx: int, bits: int, save_path: Path) -> None:
  save_path.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.hist(vec, bins=300, density=True, alpha=0.5, color="steelblue", label="Empirical")

  # Distribution fits
  vec_sorted = np.sort(vec)
  for dist_type in DistributionType:
    fitted = Distribution.fit(vec_sorted, dist_type)
    ll = fitted.log_likelihood
    ax.plot(vec_sorted, fitted.pdf(), color=DIST_COLORS[dist_type], linewidth=0.7, linestyle="--", label=f"{repr(fitted):30s} ll={ll:.3g}")

  # MinMax quantization
  alpha = minmax_alpha(vec)
  vec_q = quantize(vec, alpha, bits)
  mse = float(np.mean((vec - vec_q) ** 2))
  ax.axvline(-alpha, color="grey", linestyle=":", linewidth=1.2, label=f"MinMax α={alpha:.2f} MSE={mse:.2e}")
  ax.axvline(alpha, color="grey", linestyle=":", linewidth=1.2)

  eda_lines = [
    f"n        = {vec.size:,}",
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


def main():
  if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)
  for model_name, model_path in models.items():
    results_dir = RESULTS_DIR / model_name
    model = load_onnx(model_path)
    layers = extract_layers(model)
    print(f"[{model_name}] Total layers: {len(layers)}")

    for idx, layer in enumerate(layers, 1):
      vec = onnx.numpy_helper.to_array(layer.tensor).flatten().astype(np.float32)
      n = len(vec)
      print(f"[{idx:>3}/{len(layers)}] {layer.op_type:20} {layer.tensor.name:50} n={n:,}")
      for bits in BITS:
        plot_layer(vec, layer.tensor.name, idx, bits, results_dir / f"{bits}bit")


if __name__ == "__main__":
  main()
