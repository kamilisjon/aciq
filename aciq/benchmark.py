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


def parse_imagenet_val_labels(dataset_path: Path) -> dict[str, str]:
  """Return {image_id: synset} from LOC_val_solution.csv."""
  labels_csv = dataset_path / "LOC_val_solution.csv"
  imageid_to_synset: dict[str, str] = {}
  with labels_csv.open("r", newline="") as f:
    for row in csv.DictReader(f, delimiter=","):
      tokens = row["PredictionString"].strip().split()
      synsets = [tokens[i] for i in range(0, len(tokens), 5)]
      assert len(set(synsets)) == 1  # if there are multiple ground-truth labels, they must be the same
      imageid_to_synset[row["ImageId"]] = synsets[0]
  return imageid_to_synset


def sample_imagenet_val(dataset_path: Path, n_per_class: int | None = None) -> list[Path]:
  """Sorted val paths, optionally limited to the first N files (by path) of each synset class."""
  val_dir = dataset_path / "ILSVRC" / "Data" / "CLS-LOC" / "val"
  images = sorted(p for p in val_dir.iterdir() if p.suffix.upper() == ".JPEG")
  if n_per_class is None:
    return images
  imageid_to_synset = parse_imagenet_val_labels(dataset_path)
  by_class: dict[str, list[Path]] = {}
  for p in images:
    by_class.setdefault(imageid_to_synset[p.stem], []).append(p)
  sampled: list[Path] = []
  for synset in sorted(by_class):
    sampled.extend(by_class[synset][:n_per_class])
  return sorted(sampled)


def benchmark_accuracy(model: nn.Module, device: str, imagenet_data_path: Path, batch_size: int):
  images = sample_imagenet_val(imagenet_data_path)
  imageid_to_label = parse_imagenet_val_labels(imagenet_data_path)

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
