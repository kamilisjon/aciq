from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import onnx

from aciq.onnx_io import extract_tensors


MODEL_PATH = Path("models/resnet50_v1_Opset18_fused.onnx")
LAYER_NAME = "onnx::Conv_650"
RESULTS_DIR = Path("results/illustrate_quantization")
BITS = 4
BINS = 1000


def load_weights() -> np.ndarray:
  model = onnx.load(str(MODEL_PATH))
  tensors = extract_tensors(model)
  return onnx.numpy_helper.to_array(tensors[LAYER_NAME]).astype(np.float32).flatten()


def setup_plot(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  counts, edges = np.histogram(weights, bins=BINS, density=True)
  centers = (edges[:-1] + edges[1:]) / 2
  return centers, counts


def draw_grid_lines(ax: plt.Axes, scale: float, y_max: float) -> None:
  qmax = 2 ** (BITS - 1) - 1  # 7
  tag_offset = scale * 0.12  # shift tags to the right so they don't sit on the line

  for level in range(-qmax, qmax + 1):
    x = level * scale
    is_boundary = abs(level) == qmax
    ax.axvline(x, color="dimgrey", linestyle="-" if is_boundary else "--", linewidth=2 if is_boundary else 0.8, alpha=0.9 if is_boundary else 0.7)
    ax.text(x + tag_offset, y_max, str(level), ha="left", va="bottom", fontsize=12, color="dimgrey", fontweight="bold" if is_boundary else "normal")


def draw_axes(ax: plt.Axes) -> None:
  ax.set_xlabel("Svorių reikšmės", fontsize=14)
  ax.set_ylabel("Tankis", fontsize=14)
  ax.set_title("ResNet50 sluoksnio svorių pasiskirstymas", fontsize=16)


def main() -> None:
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  weights = load_weights()

  alpha = float(np.percentile(np.abs(weights), 99.99  ))
  qmax = 2 ** (BITS - 1) - 1
  scale = alpha / qmax

  centers, counts = setup_plot(weights)
  y_max = float(counts.max())

  # xlim: data min/max with 5% margin
  w_min, w_max = float(weights.min()), float(weights.max())
  margin = (w_max - w_min) * 0.05
  xlim = (w_min - margin, w_max + margin)

  # --- Plot 1: Quantization grid ---
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(centers, counts, color="steelblue", linewidth=1.5)
  draw_grid_lines(ax, scale, y_max)
  draw_axes(ax)
  ax.set_xlim(xlim)
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / "1_quantization_grid.png", dpi=700)
  plt.close(fig)

  # --- Plot 2: Clipping error ---
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(centers, counts, color="steelblue", linewidth=1.5)
  # Fill the area under the curve at the tails beyond ±alpha
  mask_left = centers < -alpha
  mask_right = centers > alpha
  ax.fill_between(centers[mask_left], counts[mask_left], color="indianred", alpha=0.7, label="Ribojimo paklaida")
  ax.fill_between(centers[mask_right], counts[mask_right], color="indianred", alpha=0.7)
  # Also color the tail lines red
  ax.plot(centers[mask_left], counts[mask_left], color="indianred", linewidth=1.5)
  ax.plot(centers[mask_right], counts[mask_right], color="indianred", linewidth=1.5)
  draw_grid_lines(ax, scale, y_max)
  draw_axes(ax)
  ax.legend(fontsize=12)
  ax.set_xlim(xlim)
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / "2_clipping_error.png", dpi=700)
  plt.close(fig)

  # --- Plot 3: Rounding error ---
  fig, ax = plt.subplots(figsize=(10, 5))
  mask_inside = (centers >= -alpha) & (centers <= alpha)
  ax.plot(centers, counts, color="steelblue", linewidth=1.5)
  ax.plot(centers[mask_inside], counts[mask_inside], color="indianred", linewidth=1.5, label="Apvalinimo paklaida")
  draw_grid_lines(ax, scale, y_max)
  draw_axes(ax)
  ax.legend(fontsize=12)
  ax.set_xlim(xlim)
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / "3_rounding_error.png", dpi=700)
  plt.close(fig)

  print(f"Saved 3 plots to {RESULTS_DIR}/")


if __name__ == "__main__":
  main()
