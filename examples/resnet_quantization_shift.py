import argparse
import csv
from pathlib import Path

import numpy as np
import onnx
from tinygrad.helpers import tqdm

from aciq.analysis import ShiftResult, StatsAccumulator, compute_shift
from aciq.benchmark import load_and_preprocess
from aciq.onnx_session import create_session_with_intermediates, get_block_output_names
from aciq.plotting import plot_shift


RESULTS_DIR = Path("results/resnet_quantization_shift")
DATASET_PATH = Path("/home/kamilis/Downloads/imagenet-object-localization-challenge")

MODELS: dict[str, dict[str, Path]] = {
  "resnet18": {
    "fp32": Path("models/resnet18_Opset18.onnx"),
    "per_tensor_minmax": Path("results/resnet18/resnet18_per_tensor_minmax_8bit.onnx"),
    "per_tensor_aciq": Path("results/resnet18/resnet18_per_tensor_aciq_8bit.onnx"),
    "per_channel_minmax": Path("results/resnet18/resnet18_per_channel_minmax_8bit.onnx"),
    "per_channel_aciq": Path("results/resnet18/resnet18_per_channel_aciq_8bit.onnx"),
  },
  "resnet50": {
    "fp32": Path("models/resnet50_Opset18.onnx"),
    "per_tensor_minmax": Path("results/resnet50/resnet50_per_tensor_minmax_8bit.onnx"),
    "per_tensor_aciq": Path("results/resnet50/resnet50_per_tensor_aciq_8bit.onnx"),
    "per_channel_minmax": Path("results/resnet50/resnet50_per_channel_minmax_8bit.onnx"),
    "per_channel_aciq": Path("results/resnet50/resnet50_per_channel_aciq_8bit.onnx"),
  },
}

METHOD_NAMES = ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]


def collect_layer_stats(model_path: Path, layer_names: list[str], image_paths: list[Path], batch_size: int = 1, cuda: bool = False) -> dict[str, np.ndarray]:
  """Run inference and accumulate per-channel statistics for tracked layers."""
  session, all_output_names = create_session_with_intermediates(model_path, layer_names, cuda=cuda)
  input_name = session.get_inputs()[0].name

  acc = StatsAccumulator()
  for start in tqdm(range(0, len(image_paths), batch_size), desc=f"  {model_path.name}"):
    batch_paths = image_paths[start : start + batch_size]
    batch = load_and_preprocess(batch_paths)
    results = session.run(all_output_names, {input_name: batch})

    # results[0] is the model logits, results[1:] are the intermediate outputs
    for i, name in enumerate(layer_names):
      acc.update(name, results[i + 1])

  return acc.finalize()


def save_shifts_csv(shifts: dict[str, ShiftResult], layer_names: list[str], save_path: Path) -> None:
  save_path.parent.mkdir(parents=True, exist_ok=True)
  header = ["method"]
  for name in layer_names:
    header += [f"{name}__mean_shift", f"{name}__var_shift"]

  with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for method, shift in shifts.items():
      row: list[str | float] = [method]
      for name in layer_names:
        row += [shift.mean_shift[name], shift.var_shift[name]]
      writer.writerow(row)


def load_shifts_csv(path: Path) -> tuple[dict[str, ShiftResult], list[str]]:
  with open(path) as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    assert columns is not None

    # Extract layer names from column headers
    layer_names: list[str] = []
    for col in columns:
      if col.endswith("__mean_shift"):
        layer_names.append(col.removesuffix("__mean_shift"))

    shifts: dict[str, ShiftResult] = {}
    for row in reader:
      method = row["method"]
      mean_shift = {name: float(row[f"{name}__mean_shift"]) for name in layer_names}
      var_shift = {name: float(row[f"{name}__var_shift"]) for name in layer_names}
      shifts[method] = ShiftResult(mean_shift=mean_shift, var_shift=var_shift)

  return shifts, layer_names


def run_analysis(model_name: str, dataset_path: Path, n_images: int | None = None, cuda: bool = False) -> None:
  paths = MODELS[model_name]
  save_dir = RESULTS_DIR / model_name

  # Determine tracked layers
  fp32_model = onnx.load(str(paths["fp32"]))
  layer_names = get_block_output_names(fp32_model)
  print(f"[{model_name}] Tracking {len(layer_names)} layers")

  # Get ImageNet val images
  val_dir = dataset_path / "ILSVRC" / "Data" / "CLS-LOC" / "val"
  image_paths = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == ".JPEG"])
  if n_images is not None:
    image_paths = image_paths[:n_images]
  print(f"[{model_name}] Using {len(image_paths)} images from {val_dir}")

  # Collect FP32 stats
  print(f"[{model_name}] Collecting FP32 stats...")
  fp32_stats = collect_layer_stats(paths["fp32"], layer_names, image_paths, cuda=cuda)

  # Collect quantized stats and compute shifts
  shifts: dict[str, ShiftResult] = {}
  for method in METHOD_NAMES:
    print(f"[{model_name}] Collecting {method} stats...")
    quant_stats = collect_layer_stats(paths[method], layer_names, image_paths, cuda=cuda)
    shifts[method] = compute_shift(fp32_stats, quant_stats)

  # Save CSV
  csv_path = save_dir / "shifts.csv"
  save_shifts_csv(shifts, layer_names, csv_path)
  print(f"[{model_name}] CSV saved to {csv_path}")

  # Plot
  plot_shift(shifts, layer_names, save_dir, model_name=model_name)
  print(f"[{model_name}] Plots saved to {save_dir}/")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ResNet quantization mean/variance shift analysis")
  parser.add_argument("--model", type=str, default="resnet18", choices=list(MODELS.keys()))
  parser.add_argument("--n-images", type=int, default=None, help="Use first N images (default: all 50k)")
  parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
  parser.add_argument("--cuda", action="store_true", help="Use CUDA for inference")
  parser.add_argument("--from-csv", type=Path, default=None, help="Load from CSV instead of running inference")
  args = parser.parse_args()

  if args.from_csv:
    shifts, layer_names = load_shifts_csv(args.from_csv)
    save_dir = RESULTS_DIR / args.model
    plot_shift(shifts, layer_names, save_dir, model_name=args.model)
    print(f"Plots saved to {save_dir}/")
  else:
    run_analysis(args.model, args.dataset_path, n_images=args.n_images, cuda=args.cuda)
