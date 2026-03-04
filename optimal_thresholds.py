"""Phase 3 deliverable: table of optimal clipping thresholds α* vs Min-Max baseline.

For each layer of the model, fits the best distribution (Gaussian vs Laplace by
log-likelihood) and finds the MSE-minimising clipping threshold α* for each
bit-width in BITS_LIST.  Prints a comparison table and saves it to a CSV file.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnx

from aciq.distributions import Gaussian, Laplace
from aciq.onnx_io import extract_layers, load_onnx
from aciq.quantization import minmax_alpha, optimal_alpha, total_mse


MODEL_PATH = Path("models/resnet50_Opset18.onnx")
RESULTS_DIR = Path("results")
BITS_LIST: list[int] = [4, 8]
MIN_WEIGHTS: int = 64  # skip constant/bias tensors with too few elements


def best_fit(vec: np.ndarray) -> tuple[Gaussian | Laplace, str]:
    g = Gaussian(vec)
    lap = Laplace(vec)
    if g.log_likelihood >= lap.log_likelihood:
        return g, "Gaussian"
    return lap, "Laplace"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_onnx(MODEL_PATH)
    layers = extract_layers(model)
    print(f"Model: {MODEL_PATH}  |  Total tensors: {len(layers)}\n")

    rows: list[dict] = []

    for idx, layer in enumerate(layers, 1):
        vec = onnx.numpy_helper.to_array(layer.tensor).flatten().astype(np.float64)
        if len(vec) < MIN_WEIGHTS:
            continue

        dist, dist_name = best_fit(vec)
        mm = minmax_alpha(dist)

        row: dict = {
            "idx": idx,
            "layer": layer.tensor.name,
            "op_type": layer.op_type,
            "n": len(vec),
            "best_dist": dist_name,
            "minmax_alpha": mm,
        }

        bit_parts: list[str] = []
        for bits in BITS_LIST:
            a_star, mse_star = optimal_alpha(dist, bits)
            # TODO: how does ONNX calculate error? I remember they have something in onnxruntime Python library.
            mse_mm = total_mse(dist, mm, bits)
            ratio = mse_mm / mse_star if mse_star > 0 else float("nan")
            row[f"alpha_star_{bits}b"] = a_star
            row[f"mse_star_{bits}b"] = mse_star
            row[f"mse_minmax_{bits}b"] = mse_mm
            row[f"mse_ratio_{bits}b"] = ratio
            bit_parts.append(f"{bits}b: α*={a_star:.4f}  ratio={ratio:.3f}x")

        rows.append(row)
        print(
            f"[{idx:>3}/{len(layers)}] {layer.op_type:12} {dist_name:8} "
            f"n={len(vec):>8,}  MinMax={mm:.4f}  " + "  |  ".join(bit_parts)
        )

    # ── Summary table ────────────────────────────────────────────────────────
    sep = "─" * 130
    bits_header = "".join(
        f"  {'α*('+str(b)+'b)':>9}  {'MM α':>8}  {'ratio':>7}" for b in BITS_LIST
    )
    print(f"\n{sep}")
    print(f"{'#':>4}  {'Op':12}  {'Dist':8}  {'n':>8}  {'MinMax α':>9}{bits_header}")
    print(sep)
    for row in rows:
        bits_cols = "".join(
            f"  {row[f'alpha_star_{b}b']:>9.4f}  {row['minmax_alpha']:>8.4f}  {row[f'mse_ratio_{b}b']:>6.2f}x"
            for b in BITS_LIST
        )
        print(
            f"{row['idx']:>4}  {row['op_type']:12}  {row['best_dist']:8}  "
            f"{row['n']:>8,}  {row['minmax_alpha']:>9.4f}{bits_cols}"
        )
    print(sep)
    print("MSE ratio = L(α_MinMax) / L(α*)  — higher means more gain from optimal clipping.\n")

    # ── Per-bit summary statistics ───────────────────────────────────────────
    for bits in BITS_LIST:
        ratios = [r[f"mse_ratio_{bits}b"] for r in rows if not np.isnan(r[f"mse_ratio_{bits}b"])]
        print(
            f"{bits}-bit  median ratio={np.median(ratios):.3f}x  "
            f"mean={np.mean(ratios):.3f}x  max={np.max(ratios):.3f}x"
        )

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "optimal_thresholds.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
