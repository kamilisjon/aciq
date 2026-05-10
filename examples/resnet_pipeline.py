import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tinygrad import GlobalCounters, Tensor, TinyJit
from tinygrad.helpers import tqdm
from tinygrad.nn import BatchNorm, Conv2d, Linear

from aciq.analysis import LayerStats, ShiftResult, StatsAccumulator, compute_shift, save_shifts_csv
from aciq.benchmark import benchmark_accuracy, sample_imagenet_val
from aciq.bias_correction import (
  CORRECTION_MODES,
  LayerInputStats,
  apply_correction,
  capture_bn_params,
  compute_input_stats,
)
from aciq.fusion import fuse_conv_bn
from aciq.helpers import get_output_dir
from aciq.resnet import Bottleneck, ResNet
from aciq.plotting import plot_channel_ranges, plot_shift
from aciq.preprocess import load_and_preprocess
from aciq.quantization import quantize
from aciq.weight_analysis import analyze_layer


METHOD_NAMES = ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]
RESULTS_DIR = Path("results")
PER_CHANNEL_METHODS = {"per_channel_minmax", "per_channel_aciq"}


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
  def bn_results_dir(self) -> Path:
    return self.output_dir / "bn_fusion_effects"

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
# ResNet structural walkers
# ---------------------------------------------------------------------------


def _conv_bn_pairs(model: ResNet) -> list[tuple[str, Conv2d, BatchNorm]]:
  """Every (conv, bn) pair in forward order with a qualified name."""
  pairs: list[tuple[str, Conv2d, BatchNorm]] = [("stem", model.conv1, model.bn1)]
  for li, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
    for bi, block in enumerate(layer):
      prefix = f"layer{li}.{bi}"
      pairs.append((f"{prefix}.conv1", block.conv1, block.bn1))
      pairs.append((f"{prefix}.conv2", block.conv2, block.bn2))
      if isinstance(block, Bottleneck):
        pairs.append((f"{prefix}.conv3", block.conv3, block.bn3))
      if block.downsample:
        pairs.append((f"{prefix}.downsample", block.downsample[0], block.downsample[1]))
  return pairs


def _weight_modules(model: ResNet) -> list[tuple[str, Conv2d | Linear]]:
  """Every weight-bearing module (Conv + fc) in forward order."""
  mods: list[tuple[str, Conv2d | Linear]] = [("stem", model.conv1)]
  for li, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
    for bi, block in enumerate(layer):
      prefix = f"layer{li}.{bi}"
      mods.append((f"{prefix}.conv1", block.conv1))
      mods.append((f"{prefix}.conv2", block.conv2))
      if isinstance(block, Bottleneck):
        mods.append((f"{prefix}.conv3", block.conv3))
      if block.downsample:
        mods.append((f"{prefix}.downsample", block.downsample[0]))
  mods.append(("fc", model.fc))
  return mods


# ---------------------------------------------------------------------------
# Stage 1: BN Fusion Analysis
# ---------------------------------------------------------------------------


def stage_bn_analysis(config: PipelineConfig, model: ResNet) -> None:
  for idx, (name, conv, bn) in enumerate(tqdm(_conv_bn_pairs(model))):
    pre_weight = conv.weight.numpy()
    post_weight, _ = fuse_conv_bn(conv, bn)
    plot_channel_ranges(idx, name, pre_weight, post_weight.numpy(), config.bn_results_dir)


# ---------------------------------------------------------------------------
# Stage 2: Weight Distribution Analysis
# ---------------------------------------------------------------------------


def stage_weight_analysis(config: PipelineConfig, fused_model: ResNet, fq_models: dict[str, ResNet]) -> None:
  weight_modules = _weight_modules(fused_model)
  print(f"  Total Conv/Linear layers: {len(weight_modules)}")
  fq_lookups: dict[str, dict[str, Conv2d | Linear]] = {m: dict(_weight_modules(fq_models[m])) for m in METHOD_NAMES}

  config.weight_results_dir.mkdir(parents=True, exist_ok=True)
  csv_path = config.weight_results_dir / "mae_comparison.csv"
  with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["layer_idx", "op_type", "name", "n", "n_per_ch", "ch_count", "err", "err_aciq", "err_channel", "err_channel_aciq"])

    for layer_idx, (weight_name, module) in enumerate(weight_modules, 1):
      op_type = "Linear" if isinstance(module, Linear) else "Conv"
      weight_arr = module.weight.numpy().astype(np.float32)
      vec = weight_arr.flatten()
      safe_name = weight_name.replace("/", "_").replace(":", "_")[:60]

      # Per-tensor
      plot_dir = config.weight_results_dir / "per_tensor"
      alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, config.bits, plot_dir)
      q_mm = quantize(vec, alpha_minmax, config.bits)
      q_ac = quantize(vec, alpha_aciq, config.bits)

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
        q_mm_ch = quantize(ch_vec, ch_alpha_minmax, config.bits)
        q_ac_ch = quantize(ch_vec, ch_alpha_aciq, config.bits)
        total_err_minmax += float(np.sum(np.abs(ch_vec - q_mm_ch)))
        total_err_aciq += float(np.sum(np.abs(ch_vec - q_ac_ch)))
        fq_ch_mm[ch] = q_mm_ch.reshape(weight_arr[ch].shape)
        fq_ch_ac[ch] = q_ac_ch.reshape(weight_arr[ch].shape)

      fq_lookups["per_channel_minmax"][weight_name].weight = Tensor(fq_ch_mm.astype(np.float32))
      fq_lookups["per_channel_aciq"][weight_name].weight = Tensor(fq_ch_ac.astype(np.float32))

      print(f"  [{layer_idx:>3}] {op_type:6s} {weight_name:40} n={len(vec):,}")
      writer.writerow([
        layer_idx,
        op_type,
        weight_name,
        len(vec),
        len(ch_vec),
        int(len(vec) / len(ch_vec)),
        float(np.sum(np.abs(vec - q_mm))),
        float(np.sum(np.abs(vec - q_ac))),
        total_err_minmax,
        total_err_aciq,
      ])

  print(f"  CSV written to {csv_path}")


# ---------------------------------------------------------------------------
# Stage 2.5: Bias and Variance Correction (analytical, BN-derived)
# ---------------------------------------------------------------------------


def stage_corrections(
  fp_model: ResNet,
  fq_models: dict[str, ResNet],
  input_stats: dict[str, LayerInputStats],
) -> dict[tuple[str, str], ResNet]:
  """Build the (method × correction_mode) variant matrix.

  Per-tensor methods get only the uncorrected baseline because variance correction
  is not hardware-realizable in a per-tensor weight scheme. Per-channel methods
  get the full {none, bias, variance, joint} set, with non-"none" modes deep-copied
  off the post-Stage-2 model and corrected per layer (stem is left untouched —
  no preceding BN to derive analytical input stats from).
  """
  fp_modules = dict(_weight_modules(fp_model))
  variants: dict[tuple[str, str], ResNet] = {}
  for method in METHOD_NAMES:
    base = fq_models[method]
    variants[(method, "none")] = base
    if method not in PER_CHANNEL_METHODS:
      continue
    for mode in CORRECTION_MODES:
      m = copy.deepcopy(base)
      mods = dict(_weight_modules(m))
      for name, module in mods.items():
        if name == "stem":
          continue  # no preceding BN; correction is undefined analytically
        stats = input_stats[name]
        W_fp = fp_modules[name].weight.numpy()
        b_orig = module.bias.numpy() if module.bias is not None else np.zeros(module.weight.shape[0], dtype=np.float32)
        apply_correction(module, W_fp, b_orig, mode, stats)
      variants[(method, mode)] = m
      print(f"  applied '{mode}' correction to {method}")
  return variants


# ---------------------------------------------------------------------------
# Stage 3: Quantization Shift Analysis
# ---------------------------------------------------------------------------


def _collect_activations(model: ResNet, image_paths: list[Path], batch_size: int = 32) -> dict[str, LayerStats]:
  jmodel = TinyJit(model)
  jmodel(Tensor.rand(batch_size, 3, 224, 224)).realize()
  GlobalCounters.reset()
  jmodel(Tensor.rand(batch_size, 3, 224, 224)).realize()

  acc = StatsAccumulator()
  for start in tqdm(range(0, len(image_paths), batch_size), desc="  activations"):
    batch_paths = image_paths[start : start + batch_size]
    jmodel(load_and_preprocess(batch_paths, pad_to_batch_size=batch_size))
    real_n = len(batch_paths)
    for name, act in model.activations.items():
      # slice off the zero-padded tail so accumulated stats only reflect real images
      acc.update(name, (act[:real_n] if real_n < batch_size else act).numpy())
  return acc.finalize()


def stage_shift_analysis(
  config: PipelineConfig, fp32_model: ResNet, variants: dict[tuple[str, str], ResNet]
) -> None:
  image_paths = sample_imagenet_val(config.dataset_path, config.n_per_class)
  print(f"  Using {len(image_paths)} images")

  print("  Collecting FP32 stats...")
  fp32_stats = _collect_activations(fp32_model, image_paths)

  shifts: dict[str, ShiftResult] = {}
  for (method, mode), model in variants.items():
    label = f"{method}::{mode}"
    print(f"  Collecting {label} stats...")
    quant_stats = _collect_activations(model, image_paths)
    shifts[label] = compute_shift(fp32_stats, quant_stats)

  layer_names = list(fp32_stats.keys())
  csv_path = config.shift_results_dir / "shifts.csv"
  save_shifts_csv(shifts, layer_names, csv_path)
  print(f"  CSV saved to {csv_path}")

  plot_shift(shifts, layer_names, config.shift_results_dir, model_name=config.model_name)
  print(f"  Plots saved to {config.shift_results_dir}/")


# ---------------------------------------------------------------------------
# Stage 4: Benchmarking
# ---------------------------------------------------------------------------


def stage_benchmark(
  config: PipelineConfig, fp_model: ResNet, variants: dict[tuple[str, str], ResNet]
) -> None:
  config.correction_results_dir.mkdir(parents=True, exist_ok=True)
  csv_path = config.correction_results_dir / "benchmark_results.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "correction_mode", "top1", "top5"])

    print(f"  benchmarking FP32")
    top1, top5 = benchmark_accuracy(fp_model, config.dataset_path)
    writer.writerow(["fp32", "none", top1, top5])
    print(f"  FP32: top1={top1:.2f}  top5={top5:.2f}")

    for (method, mode), model in variants.items():
      print(f"  benchmarking {method}::{mode}")
      top1, top5 = benchmark_accuracy(model, config.dataset_path)
      writer.writerow([method, mode, top1, top5])
      print(f"  {method}::{mode}: top1={top1:.2f}  top5={top5:.2f}")
  print(f"  CSV saved to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description="Full ACIQ quantization analysis pipeline")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--bits", type=int, default=8)
  parser.add_argument("--dataset-path", type=Path, required=True, help="Path to ImageNet dataset root")
  parser.add_argument("--plot-per-channel", action="store_true", help="Generate per-channel weight distribution plots (slow)")
  parser.add_argument("--n-per-class", type=int, default=5, help="Sample N ImageNet val images per class for shift analysis.")
  parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Root results directory")
  args = parser.parse_args()

  config = PipelineConfig(
    model_depth=int(args.model.removeprefix("resnet")),
    bits=args.bits,
    dataset_path=args.dataset_path,
    plot_per_channel=args.plot_per_channel,
    output_dir=get_output_dir(args.output_dir, args.model),
    n_per_class=args.n_per_class,
  )
  print(f"Output directory: {config.output_dir}")

  model = ResNet(config.model_depth)
  model.load_from_pretrained()

  print(f"\n=== Stage 1: BN Fusion Analysis ({config.model_name}) ===")
  stage_bn_analysis(config, model)

  print("\n=== Capturing BN parameters (analytical bias/variance correction) ===")
  bn_params = capture_bn_params(model)
  model.fuse()
  input_stats = compute_input_stats(model, bn_params)
  print(f"  captured BN params for {len(bn_params)} layers; input stats for {len(input_stats)} weight modules")

  fq_models = {m: copy.deepcopy(model) for m in METHOD_NAMES}

  print(f"\n=== Stage 2: Weight Distribution Analysis ({config.model_name}) ===")
  stage_weight_analysis(config, model, fq_models)

  print(f"\n=== Stage 2.5: Bias and Variance Correction ({config.model_name}) ===")
  variants = stage_corrections(model, fq_models, input_stats)
  print(f"  built {len(variants)} quantized variants (per-tensor: none only; per-channel: none/bias/variance/joint)")

  print(f"\n=== Stage 3: Quantization Shift Analysis ({config.model_name}) ===")
  stage_shift_analysis(config, model, variants)

  print(f"\n=== Stage 4: Benchmarking ({config.model_name}) ===")
  stage_benchmark(config, model, variants)

  print(f"\nDone. All results in {config.output_dir}")


if __name__ == "__main__":
  main()
