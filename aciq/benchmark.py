import time
from pathlib import Path
from enum import Enum
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import PIL.Image as pil_image
from tinygrad.helpers import tqdm


WARMUP_RUNS_COUNT = 300
BENCHMARK_RUNS_COUNT = 100
IMAGENET_LABELS_FILEPATH = "aciq/imagenet_class_index.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_preprocess(image_paths: list[Path]) -> torch.Tensor:
  # timm style ImageNet pre-process
  images = [Image.open(p).convert("RGB") for p in image_paths]
  transform = transforms.Compose([
    transforms.Resize(256, interpolation=pil_image.Resampling.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  return torch.stack([transform(img) for img in images])


class ExecProvider(Enum):
  CPU = 0
  CUDA = 1

  def __repr__(self):
    return self.name


def benchmark_accuracy(model: nn.Module, device: str, imagenet_data_path: Path, batch_size: int):
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

  model.eval()
  correct_top1 = correct_top5 = 0
  with torch.no_grad():
    for start in tqdm(range(0, len(images), batch_size), desc="Benchmarking Accuracy"):
      batch_paths = images[start : start + batch_size]
      batch = load_and_preprocess(batch_paths).to(device)
      outputs = model(batch).cpu().numpy()
      for i in range(len(batch_paths)):
        pred = np.argsort(outputs[i])[-5:][::-1]
        gt_label_idx = gt_label_to_idx[imageid_to_label[batch_paths[i].stem]]

        if pred[0] == gt_label_idx:
          correct_top1 += 1
        if any(p == gt_label_idx for p in pred):
          correct_top5 += 1

  return correct_top1 / len(images) * 100, correct_top5 / len(images) * 100


def benchmark_speed(model: nn.Module, device: str, batch_size: int):
  model.eval()
  input_data = torch.zeros((batch_size, 3, 224, 224), device=device)

  if device == "cuda":
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
      for _ in range(WARMUP_RUNS_COUNT):
        model(input_data)
    torch.cuda.synchronize()

    total_duration = 0.0
    with torch.no_grad():
      for _ in range(BENCHMARK_RUNS_COUNT):
        starter.record()
        model(input_data)
        ender.record()
        torch.cuda.synchronize()
        total_duration += starter.elapsed_time(ender)
    return total_duration / BENCHMARK_RUNS_COUNT

  with torch.no_grad():
    for _ in range(WARMUP_RUNS_COUNT):
      model(input_data)
    total_duration = 0.0
    for _ in range(BENCHMARK_RUNS_COUNT):
      start = time.perf_counter()
      model(input_data)
      total_duration += (time.perf_counter() - start) * 1000
  return total_duration / BENCHMARK_RUNS_COUNT


def run_benchmark(
  model: nn.Module, benchmark_data_path: Path, batch_size: int = 16, exec_provider: ExecProvider = ExecProvider.CUDA
) -> tuple[float, float, float]:
  device = "cuda" if exec_provider == ExecProvider.CUDA else "cpu"
  model = model.to(device)
  speed = benchmark_speed(model, device, batch_size)
  top1_acc, top5_acc = benchmark_accuracy(model, device, benchmark_data_path, batch_size)
  return top1_acc, top5_acc, speed
