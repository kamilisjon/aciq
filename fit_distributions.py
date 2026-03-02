from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from aciq.onnx_io import load_onnx, extract_layers
from aciq.statistics import Distribution, DistributionFit, fit_distribution

RESULTS_DIR = Path("results/phase2")
MODEL_PATH = Path("models/bert_Opset18.onnx")
KS_ALPHA = 0.05

DIST_COLORS = {
    Distribution.GAUSSIAN:  "red",
    Distribution.LAPLACE:   "green",
    Distribution.STUDENT_T: "purple",
}


def fit_all(data: np.ndarray) -> list[DistributionFit]:
    results = [fit_distribution(data, dist) for dist in Distribution]
    results.sort(key=lambda r: r.ks_statistic)
    return results


def select_best(fits: list[DistributionFit]) -> DistributionFit | None:
    passing = [f for f in fits if f.ks_pvalue >= KS_ALPHA]
    return min(passing, key=lambda f: f.ks_statistic) if passing else None


def _pdf_values(dist: Distribution, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    if dist == Distribution.GAUSSIAN:
        return stats.norm.pdf(x, *stats.norm.fit(data))
    if dist == Distribution.LAPLACE:
        return stats.laplace.pdf(x, *stats.laplace.fit(data))
    return stats.t.pdf(x, *stats.t.fit(data))


def plot_layer_fit(
    vec: np.ndarray,
    fits: list[DistributionFit],
    layer_name: str,
    layer_idx: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    n_bins = min(200, max(50, len(vec) // 500))
    ax.hist(vec, bins=n_bins, density=True, alpha=0.5, color="steelblue", label="Empirical")

    x = np.linspace(vec.min(), vec.max(), 500)
    best = select_best(fits)
    for fit in fits:
        is_best = best is not None and fit.distribution == best.distribution
        lw, ls = (2.5, "-") if is_best else (1.2, "--")
        label = f"{fit.distribution.name.capitalize()}  KS={fit.ks_statistic:.4f}  p={fit.ks_pvalue:.3g}"
        ax.plot(x, _pdf_values(fit.distribution, vec, x),
                color=DIST_COLORS[fit.distribution], linewidth=lw, linestyle=ls, label=label)

    textstr = "\n".join(
        f"{f.distribution.name:10s} KS={f.ks_statistic:.4f}  p={f.ks_pvalue:.3g}" for f in fits
    )
    ax.text(0.98, 0.96, textstr, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            family="monospace")

    best_label = best.distribution.name.capitalize() if best else "Other (no fit passed KS p>0.05)"
    safe = layer_name.replace("/", "_").replace(":", "_")
    ax.set_title(f"Layer {layer_idx}: {layer_name}\nBest fit: {best_label}", fontsize=10)
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
        best = select_best(fits)

        print(f"\n[{idx:>3}/{len(layers)}] {name} n={len(vec):,}")
        for f in fits:
            marker = " <--" if best and f.distribution == best.distribution else ""
            print(f"    {f.distribution.name:10s} KS={f.ks_statistic:.4f}  p={f.ks_pvalue:.4g}{marker}")
        if best is None:
            print(f"    --> Other (no fit passed KS p>{KS_ALPHA})")

        plot_layer_fit(vec, fits, name, idx, hist_dir)


if __name__ == "__main__":
    main()
