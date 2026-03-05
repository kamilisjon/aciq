from pathlib import Path
from enum import StrEnum

import numpy as np
import matplotlib.pyplot as plt
import onnx

from aciq.onnx_io import load_onnx, extract_layers
from aciq.distributions import Distribution


RESULTS_DIR = Path("results")

# TODO: should group layers by which model block that are in. What blocks does ResNet have? Perhaps should group by what activation function is applied?
models: dict[str, Path] = {
    'resnet50': Path("models/resnet50_Opset18.onnx"),
    'bert': Path("models/bert_Opset18.onnx")  # TODO: why some layers of Bert have Add with input weights and other have encapsulated weights? How to unify?
} 

class Distributions(StrEnum):
    GAUSSIAN = "norm"
    LAPLACE = "laplace"


DIST_COLORS = {
    Distributions.GAUSSIAN:  "red",
    Distributions.LAPLACE:   "green",
}

def plot_layer_fit(vec: np.ndarray, layer_name: str, layer_idx: int, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vec, bins=200, density=True, alpha=0.5, color="steelblue", label="Empirical")

    x_sorted = np.sort(vec)
    distribution = Distribution(x_sorted)

    fit_lines = []
    for dist in [Distributions.GAUSSIAN, Distributions.LAPLACE]:
        match dist:
            case Distributions.GAUSSIAN:
                ll = distribution.gaussian.log_likelihood
                pdf = distribution.gaussian.pdf()
            case Distributions.LAPLACE:
                ll = distribution.laplace.log_likelihood
                pdf = distribution.laplace.pdf()

        fit_lines.append(f"{dist:10s} ll={ll:.3g}")
        ax.plot(x_sorted, pdf, color=DIST_COLORS[dist], linewidth=1.2, linestyle="--", label=dist)

    eda_lines = [
        f"n         = {distribution.n:,}",
        f"Mean      = {distribution.mean:.5f}",
        f"Variance  = {distribution.variance:.6f}",
        f"Skewness  = {distribution.skewness:.4f}",
        f"Kurtosis  = {distribution.kurtosis:.4f}",
    ]

    textstr = "\n".join(eda_lines + [""] + fit_lines)
    ax.text(0.98, 0.96, textstr, transform=ax.transAxes, fontsize=7.5,va="top", ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), family="monospace")
    safe = layer_name.replace("/", "_").replace(":", "_")
    ax.set_title(f"Layer {layer_idx}: {layer_name}", fontsize=10)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / f"layer_{layer_idx:03d}_{safe[:60]}.png", dpi=150)
    plt.close(fig)

def main():
    for model_name, model_path in models.items():
        results_dir = RESULTS_DIR / model_name
        model = load_onnx(model_path)
        layers = extract_layers(model)
        print(f"[{model_name}] Total layers: {len(layers)}")

        for idx, layer in enumerate(layers, 1):
            vec = onnx.numpy_helper.to_array(layer.tensor).flatten().astype(np.float32)
            n = len(vec)
            print(f"[{idx:>3}/{len(layers)}] {layer.op_type:20} {layer.tensor.name:50} n={n:,}")
            plot_layer_fit(vec, layer.tensor.name, idx, results_dir)

if __name__ == "__main__":
    main()
