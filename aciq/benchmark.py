import time
from pathlib import Path
from enum import Enum
import csv
import json

import onnxruntime
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import PIL.Image as pil_image
from tinygrad.helpers import tqdm


WARMUP_RUNS_COUNT = 300
BENCHMARK_RUNS_COUNT = 100
IMAGENET_LABELS_FILEPATH = "aciq/imagenet_class_index.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_preprocess(image_paths: list[Path], pad_to_batch_size: int | None = None):
  # timm style ImageNet pre-process
  images = [Image.open(p).convert("RGB") for p in image_paths]
  transform = transforms.Compose([
    transforms.Resize(256, interpolation=pil_image.Resampling.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  processed = [transform(img) for img in images]

  # Handle batch padding logic
  if pad_to_batch_size is not None:
    n = len(processed)
    if n > pad_to_batch_size:
      raise ValueError(f"Received {n} images, but batch size is {pad_to_batch_size}. Cannot exceed batch size.")

    if n < pad_to_batch_size:
      c, h, w = processed[0].shape
      num_pad = pad_to_batch_size - n
      pad_images = [torch.zeros((c, h, w)) for _ in range(num_pad)]
      processed.extend(pad_images)

  return torch.stack(processed).numpy()


class ExecProvider(Enum):
  CPU = 0
  CUDA = 1

  def __repr__(self):
    return self.name


def setup_session(model_path: Path, exec_provider: ExecProvider) -> onnxruntime.InferenceSession:
  session_options = onnxruntime.SessionOptions()
  session_options.log_severity_level = 0
  session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

  match exec_provider:
    case ExecProvider.CPU:
      provider = ["CPUExecutionProvider"]
    case ExecProvider.CUDA:
      provider = ["CUDAExecutionProvider"]
  return onnxruntime.InferenceSession(str(model_path), sess_options=session_options, providers=provider)


def benchmark_accuracy(session: onnxruntime.InferenceSession, imagenet_data_path: Path, batch_size: int):
  val_dir = imagenet_data_path / "ILSVRC" / "Data" / "CLS-LOC" / "val"
  images = sorted([f for f in val_dir.iterdir() if f.suffix.upper() == ".JPEG"])

  # Parse labels
  labels_csv = imagenet_data_path / "LOC_val_solution.csv"
  imageid_to_label = {}
  with labels_csv.open("r", newline="") as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
      image_id = row["ImageId"]
      pred_str = row["PredictionString"].strip()
      assert pred_str is not None
      tokens = pred_str.split()
      synsets = [tokens[i] for i in range(0, len(tokens), 5)]
      assert len(set(synsets)) == 1  # if there are multiple ground-truth labels, they must be the same
      imageid_to_label[image_id] = synsets[0]
  assert len(imageid_to_label) == len(images)

  # Map synsets
  with open(IMAGENET_LABELS_FILEPATH, "r") as f:
    class_idx = json.load(f)
  gt_label_to_idx = {v[0]: int(k) for k, v in class_idx.items()}

  # Benchmark
  inputs = session.get_inputs()
  assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
  input_name = inputs[0].name

  correct_top1 = correct_top5 = 0
  for start in tqdm(range(0, len(images), batch_size), desc="Benchmarking Accuracy"):
    batch_paths = images[start : min(start + batch_size, len(images))]
    outputs = session.run(None, {input_name: load_and_preprocess(batch_paths, batch_size)})[0]
    for i in range(len(batch_paths)):
      pred = np.argsort(outputs[i])[-5:][::-1]
      gt_label_idx = gt_label_to_idx[imageid_to_label[batch_paths[i].stem]]

      if pred[0] == gt_label_idx:
        correct_top1 += 1
      if any(p == gt_label_idx for p in pred):
        correct_top5 += 1

  return correct_top1 / len(images) * 100, correct_top5 / len(images) * 100


def benchmark_speed(session: onnxruntime.InferenceSession, batch_size: int):
  input_data = np.zeros((batch_size, 3, 224, 224), np.float32)
  input_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_data, "cuda", 0)

  iobinding = session.io_binding()
  iobinding.bind_ortvalue_input(session.get_inputs()[0].name, input_ortvalue)
  for output in session.get_outputs():
    iobinding.bind_output(output.name, "cuda")

  # Warm up
  for _ in range(WARMUP_RUNS_COUNT):
    session.run_with_iobinding(iobinding)

  total_duration = 0.0
  for _ in range(BENCHMARK_RUNS_COUNT):
    start = time.perf_counter()
    session.run_with_iobinding(iobinding)
    batch_duration = (time.perf_counter() - start) * 1000
    total_duration += batch_duration
  return total_duration / BENCHMARK_RUNS_COUNT


def run_benchmark(
  model_path: Path, benchmark_data_path: Path, batch_size: int = 16, exec_provider: ExecProvider = ExecProvider.CUDA
) -> tuple[float, float, float]:
  session = setup_session(model_path, exec_provider)
  speed = benchmark_speed(session, batch_size)
  top1_acc, top5_acc = benchmark_accuracy(session, benchmark_data_path, batch_size)
  return top1_acc, top5_acc, speed
