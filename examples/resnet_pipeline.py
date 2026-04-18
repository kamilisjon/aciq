import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision
from tinygrad.helpers import tqdm

from aciq.analysis import ShiftResult, collect_layer_stats, compute_shift, save_shifts_csv
from aciq.batch_norm import collect_conv_bn_pairs, fuse_bn_into_conv
from aciq.benchmark import run_benchmark
from aciq.onnx_io import extract_tensors, replace_weight
from aciq.onnx_session import get_block_output_names
from aciq.plotting import plot_channel_ranges, plot_shift
from aciq.quantization import quantize
from aciq.weight_analysis import analyze_layer


OPSET_VERSION = 18
METHOD_NAMES = ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]


@dataclass
class PipelineConfig:
  model_name: str
  bits: int
  dataset_path: Path
  n_images: int | None
  plot: bool
  benchmark: bool
  cuda: bool
  output_dir: Path

  @property
  def models_dir(self) -> Path:
    return self.output_dir / "models"

  @property
  def onnx_fused_path(self) -> Path:
    return self.models_dir / f"{self.model_name}_Opset{OPSET_VERSION}_fused.onnx"

  @property
  def onnx_not_fused_path(self) -> Path:
    return self.models_dir / f"{self.model_name}_Opset{OPSET_VERSION}_not_fused.onnx"

  @property
  def bn_results_dir(self) -> Path:
    return self.output_dir / "bn_fusion_effects"

  @property
  def weight_results_dir(self) -> Path:
    return self.output_dir / "weight_analysis"

  @property
  def shift_results_dir(self) -> Path:
    return self.output_dir / "quantization_shift"

  def quantized_model_path(self, method: str) -> Path:
    return self.models_dir / f"{self.model_name}_{method}_{self.bits}bit.onnx"


def load_pytorch_model(model_name: str) -> torch.nn.Module:
  match model_name:
    case "resnet18":
      return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    case "resnet50":
      return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    case _:
      raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Stage 1: BN Fusion Analysis
# ---------------------------------------------------------------------------


def stage_bn_analysis(config: PipelineConfig, model: torch.nn.Module) -> None:
  model.eval()
  config.models_dir.mkdir(parents=True, exist_ok=True)

  dummy_input = (torch.randn(1, 3, 224, 224),)
  for name, fold in [("not_fused", False), ("fused", True)]:
    save_path = config.models_dir / f"{config.model_name}_Opset{OPSET_VERSION}_{name}.onnx"
    torch.onnx.export(model, dummy_input, str(save_path), opset_version=OPSET_VERSION, do_constant_folding=fold)
    print(f"  Saved {save_path}")

  if config.plot:
    pairs = collect_conv_bn_pairs(model)
    for idx, (conv_name, conv, bn_name, bn) in tqdm(enumerate(pairs)):
      pre_weight = conv.weight.data.numpy()
      post_weight = fuse_bn_into_conv(pre_weight, bn)
      plot_channel_ranges(idx, conv_name, pre_weight, post_weight, config.bn_results_dir)


# ---------------------------------------------------------------------------
# Stage 2: Weight Distribution Analysis
# ---------------------------------------------------------------------------


def stage_weight_analysis(config: PipelineConfig) -> None:
  model = onnx.load(str(config.onnx_fused_path))
  nodes, tensors = list(model.graph.node), extract_tensors(model)
  print(f"  Total nodes: {len(nodes)}")

  fq_models = {name: copy.deepcopy(model) for name in METHOD_NAMES}

  csv_path = config.weight_results_dir / "mae_comparison.csv"
  config.weight_results_dir.mkdir(parents=True, exist_ok=True)
  csv_file = open(csv_path, "w", newline="")
  writer = csv.writer(csv_file)
  writer.writerow(["layer_idx", "op_type", "name", "n", "n_per_ch", "ch_count", "err", "err_aciq", "err_channel", "err_channel_aciq"])

  layer_idx = 0
  for node in nodes:
    if node.op_type not in ("Conv", "Gemm"):
      continue

    weight_name = node.input[1]
    weight_arr = onnx.numpy_helper.to_array(tensors[weight_name]).astype(np.float32)
    vec = weight_arr.flatten()
    layer_idx += 1
    safe_name = weight_name.replace("/", "_").replace(":", "_")[:60]

    # Per-tensor
    plot_dir = config.weight_results_dir / "per_tensor" if config.plot else None
    alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, config.bits, plot_dir)
    quant_weight_minmax = quantize(vec, alpha_minmax, config.bits)
    quant_weight_aciq = quantize(vec, alpha_aciq, config.bits)

    replace_weight(fq_models["per_tensor_minmax"], weight_name, quant_weight_minmax.reshape(weight_arr.shape))
    replace_weight(fq_models["per_tensor_aciq"], weight_name, quant_weight_aciq.reshape(weight_arr.shape))

    # Per-channel (axis 0 = output channels)
    ch_plot_dir = config.weight_results_dir / "per_channel" / f"{layer_idx:03d}_{safe_name}" if config.plot else None
    total_err_minmax = 0.0
    total_err_aciq = 0.0
    fq_ch_mm = np.empty_like(weight_arr)
    fq_ch_ac = np.empty_like(weight_arr)
    for ch in range(weight_arr.shape[0]):
      ch_vec = weight_arr[ch].flatten()
      ch_alpha_minmax, ch_alpha_aciq = analyze_layer(ch_vec, f"{weight_name}/ch{ch}", ch, config.bits, ch_plot_dir)
      quant_weight_minmax_ch = quantize(ch_vec, ch_alpha_minmax, config.bits)
      quant_weight_aciq_ch = quantize(ch_vec, ch_alpha_aciq, config.bits)
      total_err_minmax += np.sum(np.abs(ch_vec - quant_weight_minmax_ch))
      total_err_aciq += np.sum(np.abs(ch_vec - quant_weight_aciq_ch))
      fq_ch_mm[ch] = quant_weight_minmax_ch.reshape(weight_arr[ch].shape)
      fq_ch_ac[ch] = quant_weight_aciq_ch.reshape(weight_arr[ch].shape)

    replace_weight(fq_models["per_channel_minmax"], weight_name, fq_ch_mm)
    replace_weight(fq_models["per_channel_aciq"], weight_name, fq_ch_ac)

    print(f"  [{layer_idx:>3}] {node.op_type:6s} {weight_name:50} n={len(vec):,}")
    writer.writerow([
      layer_idx,
      node.op_type,
      weight_name,
      len(vec),
      len(ch_vec),
      int(len(vec) / len(ch_vec)),
      np.sum(np.abs(vec - quant_weight_minmax)),
      np.sum(np.abs(vec - quant_weight_aciq)),
      total_err_minmax,
      total_err_aciq,
    ])

  csv_file.close()
  print(f"  CSV written to {csv_path}")

  for name, fq_model in fq_models.items():
    save_path = config.quantized_model_path(name)
    onnx.save(fq_model, str(save_path))
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Stage 3: Quantization Shift Analysis
# ---------------------------------------------------------------------------


def stage_shift_analysis(config: PipelineConfig) -> None:
  fp32_model = onnx.load(str(config.onnx_fused_path))
  layer_names = get_block_output_names(fp32_model)
  print(f"  Tracking {len(layer_names)} layers")

  val_dir = config.dataset_path / "ILSVRC" / "Data" / "CLS-LOC" / "val"
  image_paths = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == ".JPEG"])
  if config.n_images is not None:
    image_paths = image_paths[: config.n_images]
  print(f"  Using {len(image_paths)} images from {val_dir}")

  # Collect FP32 stats
  print("  Collecting FP32 stats...")
  fp32_stats = collect_layer_stats(config.onnx_fused_path, layer_names, image_paths, cuda=config.cuda)

  # Collect quantized stats and compute shifts
  shifts: dict[str, ShiftResult] = {}
  for method in METHOD_NAMES:
    print(f"  Collecting {method} stats...")
    quant_stats = collect_layer_stats(config.quantized_model_path(method), layer_names, image_paths, cuda=config.cuda)
    shifts[method] = compute_shift(fp32_stats, quant_stats)

  # Save CSV
  csv_path = config.shift_results_dir / "shifts.csv"
  save_shifts_csv(shifts, layer_names, csv_path)
  print(f"  CSV saved to {csv_path}")

  # Plot
  plot_shift(shifts, layer_names, config.shift_results_dir, model_name=config.model_name)
  print(f"  Plots saved to {config.shift_results_dir}/")


# ---------------------------------------------------------------------------
# Stage 4: Benchmarking
# ---------------------------------------------------------------------------


def stage_benchmark(config: PipelineConfig) -> None:
  print(f"  FP32: {run_benchmark(config.onnx_fused_path, config.dataset_path, batch_size=1)}")
  for method in METHOD_NAMES:
    path = config.quantized_model_path(method)
    print(f"  {method}: {run_benchmark(path, config.dataset_path, batch_size=1)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description="Full ACIQ quantization analysis pipeline")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
  parser.add_argument("--bits", type=int, default=8)
  parser.add_argument("--dataset-path", type=Path, required=True, help="Path to ImageNet dataset root")
  parser.add_argument("--n-images", type=int, default=None, help="Limit validation images for shift analysis (default: all)")
  parser.add_argument("--plot", action="store_true", help="Generate distribution and BN fusion plots")
  parser.add_argument("--benchmark", action="store_true", help="Run accuracy/speed benchmarks")
  parser.add_argument("--cuda", action="store_true", help="Use CUDA for inference")
  parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Root results directory")
  args = parser.parse_args()

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  config = PipelineConfig(
    model_name=args.model,
    bits=args.bits,
    dataset_path=args.dataset_path,
    n_images=args.n_images,
    plot=args.plot,
    benchmark=args.benchmark,
    cuda=args.cuda,
    output_dir=args.output_dir / f"{args.model}_{timestamp}",
  )
  config.output_dir.mkdir(parents=True, exist_ok=True)
  print(f"Output directory: {config.output_dir}")

  pytorch_model = load_pytorch_model(config.model_name)

  print(f"\n=== Stage 1: BN Fusion Analysis ({config.model_name}) ===")
  stage_bn_analysis(config, pytorch_model)

  print(f"\n=== Stage 2: Weight Distribution Analysis ({config.model_name}) ===")
  stage_weight_analysis(config)

  print(f"\n=== Stage 3: Quantization Shift Analysis ({config.model_name}) ===")
  stage_shift_analysis(config)

  if config.benchmark:
    print(f"\n=== Stage 4: Benchmarking ({config.model_name}) ===")
    stage_benchmark(config)

  print(f"\nDone. All results in {config.output_dir}")


if __name__ == "__main__":
  main()
