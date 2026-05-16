import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn import Conv2d, Linear

from aciq.datasets.imagenet import benchmark_accuracy, load_and_preprocess, sample_imagenet_val
from aciq.quantization.bias_correction import ChannelMeansAccumulator, MeanShift
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, mean_absolute_error, save_csv
from aciq.models.resnet import ResNet, compute_input_stats, _weight_modules, _bias_correct_model
from aciq.quantization.clipping import quantize_symmetric, bound_symmetric_minmax, bound_symmetric_aciq_mae
from aciq.distributions import fit_distributions, kurtosis, skewness
from aciq.plotting_style import DIST_COLORS, HIST_BINS, LINE_WIDTH, NEUTRAL_COLOR, SERIES_COLORS, STATS_TEXT_KW


METHOD_NAMES = ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]
PER_CHANNEL_METHODS = {"per_channel_minmax", "per_channel_aciq"}


@dataclass
class MaeComparisonRow:
  layer_idx: int
  op_type: str
  name: str
  n: int
  n_per_ch: int
  ch_count: int
  err: float
  err_aciq: float
  err_channel: float
  err_channel_aciq: float


@dataclass
class BenchmarkRow:
  method: str
  correction_mode: str
  top1: float
  top5: float


def analyze_layer(vec: np.ndarray, layer_name: str, layer_idx: int, bits: int, save_path: Path | None = None) -> tuple[float, float]:
  vec_sorted = np.sort(vec)

  fits = fit_distributions(vec_sorted)

  # MinMax quantization
  alpha_minmax = bound_symmetric_minmax(vec)
  mae_minmax = mean_absolute_error(vec, quantize_symmetric(vec, alpha_minmax, bits))

  # Optimal alpha*
  best_dist = fits[0]
  alpha_aciq = bound_symmetric_aciq_mae(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_minmax)

  if save_path is not None:
    save_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vec, bins=HIST_BINS, density=True, alpha=0.5, color=NEUTRAL_COLOR, label="Empirinis")

    for fitted in fits:
      ax.plot(
        vec_sorted,
        fitted.pdf(),
        color=DIST_COLORS[type(fitted)],
        linewidth=LINE_WIDTH ,
        linestyle="--",
        label=f"{repr(fitted)}",
      )

    ax.axvline(-alpha_minmax, color=NEUTRAL_COLOR, linestyle=":", linewidth=LINE_WIDTH , label=f"MinMax α={alpha_minmax:.4f} MAE={mae_minmax:.2e}")
    ax.axvline(alpha_minmax, color=NEUTRAL_COLOR, linestyle=":", linewidth=LINE_WIDTH )

    if alpha_aciq != alpha_minmax:
      mae_aciq = mean_absolute_error(vec, quantize_symmetric(vec, alpha_aciq, bits))
      ax.axvline(
        -alpha_aciq, color=DIST_COLORS[type(best_dist)], linestyle="-", linewidth=LINE_WIDTH , label=f"ACIQ {best_dist.name} α={alpha_aciq:.4f} MAE={mae_aciq:.2e}"
      )
      ax.axvline(alpha_aciq, color=DIST_COLORS[type(best_dist)], linestyle="-", linewidth=LINE_WIDTH )

    eda_lines = [
      f"n {vec.size:,}",
      f"Minimumas {float(np.min(vec)):.5f}",
      f"Maksimumas {float(np.max(vec)):.5f}",
      f"Vidurkis {float(np.mean(vec)):.5f}",
      f"Dispersija {float(np.var(vec)):.6f}",
      f"Asimetrija {float(skewness(vec)):.4f}",
      f"Ekscesas {float(kurtosis(vec)):.4f}",
    ]
    ax.text(0.98, 0.96, "\n".join(eda_lines), transform=ax.transAxes, **STATS_TEXT_KW)
    safe = layer_name.replace("/", "_").replace(":", "_")
    ax.set_xlabel("Svorių reikšmės")
    ax.set_ylabel("Tankis")
    ax.legend(loc="upper left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path / f"layer_{layer_idx:03d}_{safe[:60]}.png")
    plt.close(fig)

  return alpha_minmax, alpha_aciq


def plot_shift(
  rows: list[MeanShift],
  layer_names: list[str],
  save_dir: Path,
  filename: str = "mean_shift.png",
) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)

  by_method: dict[str, dict[str, float]] = {}
  for r in rows:
    by_method.setdefault(r.method, {})[r.layer] = r.mean_shift

  methods = list(by_method.keys())
  n_methods = len(methods)
  x_pos = np.arange(len(layer_names))
  bar_width = 0.7 / n_methods
  offsets = np.linspace(-0.35 + bar_width / 2, 0.35 - bar_width / 2, n_methods)

  fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.5), 5))
  for i, method in enumerate(methods):
    per_layer = [by_method[method][name] for name in layer_names]
    label = "Poslinkio korekcija" if method.endswith("::bias") else "Be korekcijos"
    ax.bar(x_pos + offsets[i], per_layer, width=bar_width, color=SERIES_COLORS[i % len(SERIES_COLORS)], label=label)

  ax.set_xticks(x_pos)
  ax.set_xticklabels(layer_names, rotation=45, ha="right")
  ax.set_xlabel("Sluoksnis")
  ax.set_ylabel("Išėjimo vidurkio poslinkis")
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_dir / filename)
  plt.close(fig)


@dataclass
class PipelineConfig:
  model_depth: int
  bits: int
  dataset_path: Path
  plot_per_channel: bool
  output_dir: Path
  n_per_class: int | None = None

  @property
  def model_name(self) -> str:
    return f"resnet{self.model_depth}"

  @property
  def weight_results_dir(self) -> Path:
    return self.output_dir / "weight_analysis"

  @property
  def shift_results_dir(self) -> Path:
    return self.output_dir / "quantization_shift"

  @property
  def correction_results_dir(self) -> Path:
    return self.output_dir / "bias_variance_correction"


# ---------------------------------------------------------------------------
# Stage 2: Weight Distribution Analysis
# ---------------------------------------------------------------------------


def stage_weight_analysis(config: PipelineConfig, fused_model: ResNet, fq_models: dict[str, ResNet]) -> None:
  weight_modules = _weight_modules(fused_model)
  print(f"  Total Conv/Linear layers: {len(weight_modules)}")
  fq_lookups: dict[str, dict[str, Conv2d | Linear]] = {m: dict(_weight_modules(fq_models[m])) for m in METHOD_NAMES}

  config.weight_results_dir.mkdir(parents=True, exist_ok=True)
  csv_path = config.weight_results_dir / "mae_comparison.csv"
  rows: list[MaeComparisonRow] = []

  for layer_idx, (weight_name, module) in enumerate(weight_modules, 1):
    op_type = "Linear" if isinstance(module, Linear) else "Conv"
    weight_arr = module.weight.numpy().astype(np.float32)
    vec = weight_arr.flatten()
    safe_name = weight_name.replace("/", "_").replace(":", "_")[:60]

    # Per-tensor
    plot_dir = config.weight_results_dir / "per_tensor"
    alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, config.bits, plot_dir)
    q_mm = quantize_symmetric(vec, alpha_minmax, config.bits)
    q_ac = quantize_symmetric(vec, alpha_aciq, config.bits)

    fq_lookups["per_tensor_minmax"][weight_name].weight = Tensor(q_mm.reshape(weight_arr.shape).astype(np.float32))
    fq_lookups["per_tensor_aciq"][weight_name].weight = Tensor(q_ac.reshape(weight_arr.shape).astype(np.float32))

    # Per-channel (axis 0 = output channels/features)
    ch_plot_dir = config.weight_results_dir / "per_channel" / f"{layer_idx:03d}_{safe_name}" if config.plot_per_channel else None
    total_err_minmax = 0.0
    total_err_aciq = 0.0
    fq_ch_mm = np.empty_like(weight_arr)
    fq_ch_ac = np.empty_like(weight_arr)
    for ch in range(weight_arr.shape[0]):
      ch_vec = weight_arr[ch].flatten()
      ch_alpha_minmax, ch_alpha_aciq = analyze_layer(ch_vec, f"{weight_name}/ch{ch}", ch, config.bits, ch_plot_dir)
      q_mm_ch = quantize_symmetric(ch_vec, ch_alpha_minmax, config.bits)
      q_ac_ch = quantize_symmetric(ch_vec, ch_alpha_aciq, config.bits)
      total_err_minmax += float(np.sum(np.abs(ch_vec - q_mm_ch)))
      total_err_aciq += float(np.sum(np.abs(ch_vec - q_ac_ch)))
      fq_ch_mm[ch] = q_mm_ch.reshape(weight_arr[ch].shape)
      fq_ch_ac[ch] = q_ac_ch.reshape(weight_arr[ch].shape)

    fq_lookups["per_channel_minmax"][weight_name].weight = Tensor(fq_ch_mm.astype(np.float32))
    fq_lookups["per_channel_aciq"][weight_name].weight = Tensor(fq_ch_ac.astype(np.float32))

    print(f"  [{layer_idx:>3}] {op_type:6s} {weight_name:40} n={len(vec):,}")
    rows.append(
      MaeComparisonRow(
        layer_idx=layer_idx,
        op_type=op_type,
        name=weight_name,
        n=len(vec),
        n_per_ch=len(ch_vec),
        ch_count=int(len(vec) / len(ch_vec)),
        err=float(np.sum(np.abs(vec - q_mm))),
        err_aciq=float(np.sum(np.abs(vec - q_ac))),
        err_channel=total_err_minmax,
        err_channel_aciq=total_err_aciq,
      )
    )

  save_csv(rows, csv_path)
  print(f"  CSV written to {csv_path}")


def _collect_activations(model: ResNet, image_paths: list[Path], batch_size: int = 32) -> ChannelMeansAccumulator:
  acc = ChannelMeansAccumulator()
  for start in tqdm(range(0, len(image_paths), batch_size), desc="  activations"):
    batch_paths = image_paths[start : start + batch_size]
    real_n = len(batch_paths)
    activations = model.get_activations(load_and_preprocess(batch_paths, pad_to_batch_size=batch_size))
    for name, act in activations.items():
      # slice off the zero-padded tail so accumulated stats only reflect real images
      acc.update(name, (act[:real_n] if real_n < batch_size else act).numpy())
  return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description="Full ACIQ quantization analysis pipeline")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--bits", type=int, default=8)
  parser.add_argument("--dataset-path", type=Path, default=None, help="Path to ImageNet dataset root (required unless --from-dir is set).")
  parser.add_argument("--plot-per-channel", action="store_true", help="Generate per-channel weight distribution plots (slow)")
  parser.add_argument("--n-per-class", type=int, default=None, help="Sample N ImageNet val images per class for shift analysis and benchmarking. Default: use the full validation set.")
  parser.add_argument(
    "--from-dir",
    type=Path,
    default=None,
    help="Load `quantization_shift/shifts.csv` from this experiment directory and re-render the shift plot only (no model loading, no inference).",
  )
  args = parser.parse_args()

  save_dir = get_output_dir(RESULTS_DIR, args.model)
  print(f"Output directory: {save_dir}")

  if args.from_dir:
    shifts_path = args.from_dir / "quantization_shift" / "shifts.csv"
    loaded_rows: list[MeanShift] = load_csv(shifts_path, MeanShift)
    print(f"Loaded {len(loaded_rows)} shift rows from {shifts_path}")
    layer_names = list(dict.fromkeys(r.layer for r in loaded_rows))
    minmax_rows = [r for r in loaded_rows if r.method.startswith("per_channel_minmax")]
    aciq_rows = [r for r in loaded_rows if r.method.startswith("per_channel_aciq")]
    plot_shift(minmax_rows, layer_names, save_dir / "quantization_shift", filename="mean_shift_minmax.png")
    plot_shift(aciq_rows, layer_names, save_dir / "quantization_shift", filename="mean_shift_aciq.png")
    print(f"Plots saved to {save_dir / 'quantization_shift'}")
    return

  assert args.dataset_path is not None, "--dataset-path is required unless --from-dir is set"

  config = PipelineConfig(
    model_depth=int(args.model.removeprefix("resnet")),
    bits=args.bits,
    dataset_path=args.dataset_path,
    plot_per_channel=args.plot_per_channel,
    output_dir=save_dir,
    n_per_class=args.n_per_class,
  )

  model = ResNet(config.model_depth)
  model.load_from_pretrained()
  model.fuse()
  fq_models = {m: copy.deepcopy(model) for m in METHOD_NAMES}

  print(f"\n=== Stage 1: Weight Distribution Analysis ({config.model_name}) ===")
  stage_weight_analysis(config, model, fq_models)

  print(f"\n=== Stage 2: Bias and Variance Correction ({config.model_name}) ===")
  input_stats = compute_input_stats(model)
  fp_modules = dict(_weight_modules(model))
  variants: dict[tuple[str, str], ResNet] = {}
  for method in METHOD_NAMES:
    variants[(method, "none")] = fq_models[method]
    if method in PER_CHANNEL_METHODS:
      variants[(method, "bias")] = _bias_correct_model(fq_models[method], fp_modules, input_stats)
      print(f"  applied 'bias' correction to {method}")

  print(f"\n=== Stage 3: Quantization Shift Analysis ({config.model_name}) ===")
  image_paths = sample_imagenet_val(config.dataset_path, config.n_per_class)
  print(f"  Using {len(image_paths)} images")
  print("  Collecting FP32 stats...")
  ResNet.clear_jit_caches()
  fp32_acc = _collect_activations(model, image_paths)
  shift_rows: list[MeanShift] = []
  for (method, mode), m in variants.items():
    label = f"{method}::{mode}"
    print(f"  Collecting {label} stats...")
    ResNet.clear_jit_caches()
    q_acc = _collect_activations(m, image_paths)
    shift_rows.extend(fp32_acc.layers_means_shifts(q_acc, label))
  shifts_csv = config.shift_results_dir / "shifts.csv"
  save_csv(shift_rows, shifts_csv)
  print(f"  CSV saved to {shifts_csv}")
  layer_names = list(fp32_acc.channels_sums.keys())
  minmax_rows = [r for r in shift_rows if r.method.startswith("per_channel_minmax")]
  aciq_rows = [r for r in shift_rows if r.method.startswith("per_channel_aciq")]
  plot_shift(minmax_rows, layer_names, config.shift_results_dir, filename="mean_shift_minmax.png")
  plot_shift(aciq_rows, layer_names, config.shift_results_dir, filename="mean_shift_aciq.png")
  print(f"  Plots saved to {config.shift_results_dir}/")

  print(f"\n=== Stage 4: Benchmarking ({config.model_name}) ===")
  config.correction_results_dir.mkdir(parents=True, exist_ok=True)
  bench_rows: list[BenchmarkRow] = []
  print("  benchmarking FP32")
  ResNet.clear_jit_caches()
  top1, top5 = benchmark_accuracy(model.infer, config.dataset_path, n_per_class=config.n_per_class)
  bench_rows.append(BenchmarkRow(method="fp32", correction_mode="none", top1=float(top1), top5=float(top5)))
  print(f"  FP32: top1={top1:.2f}  top5={top5:.2f}")
  for (method, mode), m in variants.items():
    print(f"  benchmarking {method}::{mode}")
    ResNet.clear_jit_caches()
    top1, top5 = benchmark_accuracy(m.infer, config.dataset_path, n_per_class=config.n_per_class)
    bench_rows.append(BenchmarkRow(method=method, correction_mode=mode, top1=float(top1), top5=float(top5)))
    print(f"  {method}::{mode}: top1={top1:.2f}  top5={top5:.2f}")
  bench_csv = config.correction_results_dir / "benchmark_results.csv"
  save_csv(bench_rows, bench_csv)
  print(f"  CSV saved to {bench_csv}")

  print(f"\nDone. All results in {config.output_dir}")


if __name__ == "__main__":
  main()
