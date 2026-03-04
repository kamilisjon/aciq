from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import onnx
from enum import StrEnum

from aciq.onnx_io import load_onnx, extract_layers
from aciq.distributions import Distribution, Gaussian, Laplace


RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/resnet50_Opset18.onnx")

class Distributions(StrEnum):
    GAUSSIAN = "norm"
    LAPLACE = "laplace"


DIST_COLORS = {
    Distributions.GAUSSIAN:  "red",
    Distributions.LAPLACE:   "green",
}

def plot_layer_fit(vec: np.ndarray, layer_name: str, layer_idx: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    n_bins = min(200, max(50, len(vec) // 500))
    ax.hist(vec, bins=n_bins, density=True, alpha=0.5, color="steelblue", label="Empirical")


    x_sorted = np.sort(vec)

    fit_lines = []
    for dist in [Distributions.GAUSSIAN, Distributions.LAPLACE]:
        match dist:
            case Distributions.GAUSSIAN:
                dist_fit = Gaussian(x_sorted)
                ll = dist_fit.log_likelihood
                pdf = dist_fit.pdf()
            case Distributions.LAPLACE:
                dist_fit = Laplace(vec)
                ll = dist_fit.log_likelihood
                pdf = Laplace(x_sorted).pdf()

        fit_lines.append(f"{dist:10s} ll={ll:.3g}")
        ax.plot(x_sorted, pdf, color=DIST_COLORS[dist], linewidth=1.2, linestyle="--", label=dist)


    distribution = Distribution(vec)

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_onnx(MODEL_PATH)
    layers = extract_layers(model)
    print(f"Total layers: {len(layers)}")

    for idx, layer in enumerate(layers, 1):
        vec = onnx.numpy_helper.to_array(layer.tensor).flatten().astype(np.float32)
        n = len(vec)
        print(f"[{idx:>3}/{len(layers)}] {layer.op_type:20} {layer.tensor.name:50} n={n:,}")
        if n > 2000_000:
            continue
        plot_layer_fit(vec, layer.tensor.name, idx, RESULTS_DIR)


if __name__ == "__main__":
    main()
