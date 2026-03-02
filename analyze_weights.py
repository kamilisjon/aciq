from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from aciq.onnx_io import load_onnx, extract_layers
from aciq.statistics import Distribution, DistributionFit, Moments, fit_distribution

RESULTS_DIR = Path("results/phase2")
MODEL_PATH = Path("models/resnet50_Opset18.onnx")

DIST_COLORS = {
    Distribution.GAUSSIAN:  "red",
    Distribution.LAPLACE:   "green",
    Distribution.STUDENT_T: "purple",
}


def fit_all(data: np.ndarray) -> list[DistributionFit]:
    results = [fit_distribution(data, dist) for dist in Distribution]
    results.sort(key=lambda r: r.ks_statistic)
    return results

def _pdf_values(dist: Distribution, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    if dist == Distribution.GAUSSIAN:
        return stats.norm.pdf(x, *stats.norm.fit(data))
    if dist == Distribution.LAPLACE:
        return stats.laplace.pdf(x, *stats.laplace.fit(data))
    return stats.t.pdf(x, *stats.t.fit(data))


def plot_layer_fit(
    vec: np.ndarray,
    fits: list[DistributionFit],
    moments: Moments,
    layer_name: str,
    layer_idx: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    n_bins = min(200, max(50, len(vec) // 500))
    ax.hist(vec, bins=n_bins, density=True, alpha=0.5, color="steelblue", label="Empirical")

    x = np.linspace(vec.min(), vec.max(), 500)
    for fit in fits:
        lw, ls = (1.2, "--")
        label = f"{fit.distribution.name.capitalize()}  KS={fit.ks_statistic:.4f}  p={fit.ks_pvalue:.3g}"
        ax.plot(x, _pdf_values(fit.distribution, vec, x),
                color=DIST_COLORS[fit.distribution], linewidth=lw, linestyle=ls, label=label)

    eda_lines = [
        f"n         = {moments.n:,}",
        f"Mean      = {moments.mean:.5f}",
        f"Variance  = {moments.variance:.6f}",
        f"Skewness  = {moments.skewness:.4f}",
        f"Kurtosis  = {moments.kurtosis:.4f}",
    ]
    fit_lines = [
        f"{f.distribution.name:10s} KS={f.ks_statistic:.4f}  p={f.ks_pvalue:.3g} ll={f.log_likelyhood:.3g}" for f in fits
    ]
    textstr = "\n".join(eda_lines + [""] + fit_lines)
    ax.text(0.98, 0.96, textstr, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            family="monospace")

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
    hist_dir = RESULTS_DIR / "histograms"
    hist_dir.mkdir(exist_ok=True)

    model = load_onnx(MODEL_PATH)
    layers = extract_layers(model)
    print(f"Total layers: {len(layers)}")

    for idx, (name, arr) in enumerate(layers.items(), 1):
        vec = arr.flatten().astype(np.float32)
        fits = fit_all(vec)
        moments = Moments.from_array(vec)
        print(f"\n[{idx:>3}/{len(layers)}] {name} n={len(vec):,}")
        plot_layer_fit(vec, fits, moments, name, idx, hist_dir)


if __name__ == "__main__":
    main()
