import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.fusion import fuse_conv_bn_eval
from tinygrad.helpers import tqdm

from aciq.analysis import ShiftResult, compute_shift, save_shifts_csv
from aciq.batch_norm import collect_conv_bn_pairs
from aciq.benchmark import benchmark_accuracy
from aciq.helpers import get_output_dir
from aciq.plotting import plot_channel_ranges, plot_shift
from aciq.quantization import quantize
from aciq.torch_hooks import collect_activations, get_resnet_block_modules
from aciq.weight_analysis import analyze_layer


METHOD_NAMES = ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]
RESULTS_DIR = Path("results")


@dataclass
class PipelineConfig:
  model_name: str
  bits: int
  dataset_path: Path
  plot_per_channel: bool
  device: str
  output_dir: Path

  @property
  def models_dir(self) -> Path:
    return self.output_dir / "models"

  @property
  def fused_model_path(self) -> Path:
    return self.models_dir / f"{self.model_name}_fused.pt"

  @property
  def not_fused_model_path(self) -> Path:
    return self.models_dir / f"{self.model_name}_not_fused.pt"

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
    return self.models_dir / f"{self.model_name}_{method}_{self.bits}bit.pt"


def load_pytorch_model(model_name: str) -> nn.Module:
  match model_name:
    case "resnet18":
      return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    case "resnet50":
      return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    case _:
      raise ValueError(f"Unknown model: {model_name}")


def _set_submodule(model: nn.Module, path: str, new_module: nn.Module) -> None:
  parent_name, _, child_name = path.rpartition(".")
  parent = model.get_submodule(parent_name) if parent_name else model
  parent._modules[child_name] = new_module


def build_bn_fused_model(model: nn.Module) -> nn.Module:
  """Return a deep copy of model with each (Conv, BN) pair fused into one Conv; BN replaced by Identity."""
  fused = copy.deepcopy(model)
  fused.eval()
  for conv_name, conv, bn_name, bn in collect_conv_bn_pairs(fused):
    _set_submodule(fused, conv_name, fuse_conv_bn_eval(conv, bn))
    _set_submodule(fused, bn_name, nn.Identity())
  return fused


def iter_weight_modules(model: nn.Module) -> list[tuple[str, nn.Conv2d | nn.Linear]]:
  return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _assign_weight(model: nn.Module, name: str, new_weight: np.ndarray) -> None:
  target = model.get_submodule(name)
  assert isinstance(target, (nn.Conv2d, nn.Linear))
  target.weight.data = torch.from_numpy(new_weight).float()


# ---------------------------------------------------------------------------
# Stage 1: BN Fusion Analysis
# ---------------------------------------------------------------------------


def stage_bn_analysis(config: PipelineConfig, model: nn.Module) -> None:
  model.eval()
  config.models_dir.mkdir(parents=True, exist_ok=True)

  torch.save(model, config.not_fused_model_path)
  print(f"  Saved {config.not_fused_model_path}")

  fused_model = build_bn_fused_model(model)
  torch.save(fused_model, config.fused_model_path)
  print(f"  Saved {config.fused_model_path}")

  pairs = collect_conv_bn_pairs(model)
  for idx, (conv_name, conv, bn_name, bn) in tqdm(enumerate(pairs)):
    pre_weight = conv.weight.data.numpy()
    post_weight = fuse_conv_bn_eval(conv, bn).weight.data.numpy()
    plot_channel_ranges(idx, conv_name, pre_weight, post_weight, config.bn_results_dir)


# ---------------------------------------------------------------------------
# Stage 2: Weight Distribution Analysis
# ---------------------------------------------------------------------------


def stage_weight_analysis(config: PipelineConfig) -> None:
  model = torch.load(config.fused_model_path, weights_only=False)
  weight_modules = iter_weight_modules(model)
  print(f"  Total Conv/Linear layers: {len(weight_modules)}")

  fq_models = {name: copy.deepcopy(model) for name in METHOD_NAMES}

  csv_path = config.weight_results_dir / "mae_comparison.csv"
  config.weight_results_dir.mkdir(parents=True, exist_ok=True)
  csv_file = open(csv_path, "w", newline="")
  writer = csv.writer(csv_file)
  writer.writerow(["layer_idx", "op_type", "name", "n", "n_per_ch", "ch_count", "err", "err_aciq", "err_channel", "err_channel_aciq"])

  for layer_idx, (weight_name, module) in enumerate(weight_modules, 1):
    op_type = "Conv" if isinstance(module, nn.Conv2d) else "Linear"
    weight_arr = module.weight.data.numpy().astype(np.float32)
    vec = weight_arr.flatten()
    safe_name = weight_name.replace("/", "_").replace(":", "_")[:60]

    # Per-tensor
    plot_dir = config.weight_results_dir / "per_tensor"
    alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, config.bits, plot_dir)
    quant_weight_minmax = quantize(vec, alpha_minmax, config.bits)
    quant_weight_aciq = quantize(vec, alpha_aciq, config.bits)

    _assign_weight(fq_models["per_tensor_minmax"], weight_name, quant_weight_minmax.reshape(weight_arr.shape))
    _assign_weight(fq_models["per_tensor_aciq"], weight_name, quant_weight_aciq.reshape(weight_arr.shape))

    # Per-channel (axis 0 = output channels/features)
    ch_plot_dir = config.weight_results_dir / "per_channel" / f"{layer_idx:03d}_{safe_name}" if config.plot_per_channel else None
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

    _assign_weight(fq_models["per_channel_minmax"], weight_name, fq_ch_mm)
    _assign_weight(fq_models["per_channel_aciq"], weight_name, fq_ch_ac)

    print(f"  [{layer_idx:>3}] {op_type:6s} {weight_name:50} n={len(vec):,}")
    writer.writerow([
      layer_idx,
      op_type,
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
    torch.save(fq_model, save_path)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Stage 3: Quantization Shift Analysis
# ---------------------------------------------------------------------------


def stage_shift_analysis(config: PipelineConfig) -> None:
  device = config.device
  fp32_model = torch.load(config.fused_model_path, weights_only=False).to(device)

  block_modules = get_resnet_block_modules(fp32_model)
  layer_names = [n for n, _ in block_modules]
  print(f"  Tracking {len(layer_names)} layers")

  val_dir = config.dataset_path / "ILSVRC" / "Data" / "CLS-LOC" / "val"
  image_paths = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == ".JPEG"])
  print(f"  Using {len(image_paths)} images from {val_dir}")

  print("  Collecting FP32 stats...")
  fp32_stats = collect_activations(fp32_model, block_modules, image_paths, device=device)

  shifts: dict[str, ShiftResult] = {}
  for method in METHOD_NAMES:
    print(f"  Collecting {method} stats...")
    quant_model = torch.load(config.quantized_model_path(method), weights_only=False).to(device)
    quant_modules = get_resnet_block_modules(quant_model)
    quant_stats = collect_activations(quant_model, quant_modules, image_paths, device=device)
    shifts[method] = compute_shift(fp32_stats, quant_stats)

  csv_path = config.shift_results_dir / "shifts.csv"
  save_shifts_csv(shifts, layer_names, csv_path)
  print(f"  CSV saved to {csv_path}")

  plot_shift(shifts, layer_names, config.shift_results_dir, model_name=config.model_name)
  print(f"  Plots saved to {config.shift_results_dir}/")


# ---------------------------------------------------------------------------
# Stage 4: Benchmarking
# ---------------------------------------------------------------------------


def stage_benchmark(config: PipelineConfig) -> None:
  device = config.device
  fp32_model = torch.load(config.fused_model_path, weights_only=False).to(device)
  print(f"  FP32: {benchmark_accuracy(fp32_model, device, config.dataset_path, batch_size=32)}")
  for method in METHOD_NAMES:
    model = torch.load(config.quantized_model_path(method), weights_only=False).to(device)
    print(f"  {method}: {benchmark_accuracy(model, device, config.dataset_path, batch_size=32)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description="Full ACIQ quantization analysis pipeline")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
  parser.add_argument("--bits", type=int, default=8)
  parser.add_argument("--dataset-path", type=Path, required=True, help="Path to ImageNet dataset root")
  parser.add_argument("--plot-per-channel", action="store_true", help="Generate per-channel weight distribution plots (slow)")
  parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Root results directory")
  args = parser.parse_args()

  config = PipelineConfig(
    model_name=args.model,
    bits=args.bits,
    dataset_path=args.dataset_path,
    plot_per_channel=args.plot_per_channel,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=get_output_dir(RESULTS_DIR, args.model),
  )
  print(f"Output directory: {config.output_dir}")

  pytorch_model = load_pytorch_model(config.model_name)

  print(f"\n=== Stage 1: BN Fusion Analysis ({config.model_name}) ===")
  stage_bn_analysis(config, pytorch_model)

  print(f"\n=== Stage 2: Weight Distribution Analysis ({config.model_name}) ===")
  stage_weight_analysis(config)

  print(f"\n=== Stage 3: Quantization Shift Analysis ({config.model_name}) ===")
  stage_shift_analysis(config)

  print(f"\n=== Stage 4: Benchmarking ({config.model_name}) ===")
  stage_benchmark(config)

  print(f"\nDone. All results in {config.output_dir}")


if __name__ == "__main__":
  main()
