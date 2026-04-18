from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aciq.distributions import Distribution, DistributionType, kurtosis, skewness
from aciq.quantization import minmax_alpha, quantize, solve_symmetric_mae_alpha


DIST_COLORS = {
  DistributionType.GAUSSIAN: "red",
  DistributionType.LAPLACE: "green",
  DistributionType.STUDENT_T: "orange",
  DistributionType.GENERALIZED_GAUSSIAN: "blue",
}


def fit_distributions(data_array: np.ndarray) -> dict[DistributionType, Distribution]:
  return {dist_type: Distribution.fit(data_array, dist_type) for dist_type in DistributionType}


def mae(data_array_1: np.ndarray, data_array_2: np.ndarray) -> float:
  return float(np.mean(np.abs(data_array_1 - data_array_2)))


def analyze_layer(vec: np.ndarray, layer_name: str, layer_idx: int, bits: int, save_path: Path | None = None) -> tuple[float, float]:
  vec_sorted = np.sort(vec)

  fits = fit_distributions(vec_sorted)

  # MinMax quantization
  alpha_minmax = minmax_alpha(vec)
  mae_minmax = mae(vec, quantize(vec, alpha_minmax, bits))

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
      mae_aciq = mae(vec, quantize(vec, alpha_aciq, bits))
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
