from pathlib import Path
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms._presets import ImageClassification
from tinygrad.helpers import tqdm


IMAGENET_LABELS_FILEPATH = "aciq/imagenet_class_index.json"

_PREPROCESS = ImageClassification(crop_size=224)


def load_and_preprocess(image_paths: list[Path]) -> torch.Tensor:
  return torch.stack([_PREPROCESS(Image.open(p).convert("RGB")) for p in image_paths])


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
