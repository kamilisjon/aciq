from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from aciq.onnx_io import load_onnx, extract_layers
from aciq.statistics import Moments

RESULTS_DIR = Path("results/eda")
MODEL_PATH = Path("models/resnet50_Opset18.onnx")


def plot_layer_histogram(vec: np.ndarray, moments: Moments, layer_name: str, layer_idx: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # Empirical histogram (normalised to density)
    n_bins = min(200, max(50, len(vec) // 500))
    ax.hist(vec, bins=n_bins, density=True, alpha=0.6, color="steelblue", label="Empirical weights")

    # Fitted Gaussian overlay
    x = np.linspace(vec.min(), vec.max(), 500)
    gauss = stats.norm.pdf(x, moments.mean, moments.std)
    ax.plot(x, gauss, "r-", linewidth=2, label="Gaussian fit")

    # Vertical lines for key statistics
    ax.axvline(moments.mean, color="red",    linestyle="--", linewidth=1.2, label=f'Mean = {moments.mean:.4f}')
    ax.axvline(moments.q1,   color="orange", linestyle=":",  linewidth=1.2, label=f'Q1 = {moments.q1:.4f}')
    ax.axvline(moments.q3,   color="orange", linestyle=":",  linewidth=1.2, label=f'Q3 = {moments.q3:.4f}')

    # Annotation box with all moments
    textstr = (
        f'n = {moments.n:,}\n'
        f'Mean     = {moments.mean:.5f}\n'
        f'Variance = {moments.variance:.6f}\n'
        f'Skewness = {moments.skewness:.4f}\n'
        f'Kurtosis = {moments.kurtosis:.4f}'
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.98, 0.96, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="right", bbox=props)
    safe_name = layer_name.replace("/", "_").replace(":", "_")
    ax.set_title(f"Layer {layer_idx}: {layer_name}", fontsize=11)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / f"layer_{layer_idx:03d}_{safe_name[:60]}.png", dpi=150)
    plt.close(fig)


def plot_summary_dashboard(layer_names: list, all_moments: list[Moments], save_path: Path) -> None:
    metrics = ["mean", "variance", "skewness", "kurtosis"]
    colors  = ["steelblue", "seagreen", "darkorange", "crimson"]
    x = np.arange(1, len(layer_names) + 1)

    fig, axes = plt.subplots(4, 1, figsize=(max(14, len(layer_names) // 2), 16),sharex=True)
    for ax, metric, color in zip(axes, metrics, colors):
        values = [getattr(m, metric) for m in all_moments]
        ax.plot(x, values, marker="o", markersize=3, linewidth=1, color=color, label=metric.capitalize())
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Layer index", fontsize=12)
    axes[-1].set_xticks(x[::max(1, len(x) // 30)])
    fig.suptitle("ResNet50 — Statistical Moments per Weight Layer", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path / "summary_moments.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary dashboard → {save_path / 'summary_moments.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_dir = RESULTS_DIR / "histograms"
    hist_dir.mkdir(exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = load_onnx(MODEL_PATH)
    layers = extract_layers(model)
    print(f"Total layers: {len(layers)}")

    # Filter: only tensors large enough to be weight matrices/conv kernels
    layer_items = [(name, arr.flatten().astype(np.float32)) for name, arr in layers.items()]

    layer_names, all_moments = [], []

    for idx, (name, vec) in enumerate(layer_items, 1):
        moments = Moments.from_array(vec)
        layer_names.append(name)
        all_moments.append(moments)

        print(f"[{idx:>3}/{len(layer_items)}] {name}")
        print(f"         n={moments.n:,}  mean={moments.mean:.5f}  "
              f"var={moments.variance:.6f}  "
              f"skew={moments.skewness:.4f}  kurt={moments.kurtosis:.4f}")

        plot_layer_histogram(vec, moments, name, idx, hist_dir)

    plot_summary_dashboard(layer_names, all_moments, RESULTS_DIR)

if __name__ == "__main__":
    main()
