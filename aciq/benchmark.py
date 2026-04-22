import csv
from pathlib import Path

import numpy as np
from tinygrad import GlobalCounters, Tensor, TinyJit
from tinygrad.helpers import tqdm

from aciq.imagenet import ImagenetClassIndex
from aciq.models.resnet import ResNet
from aciq.preprocess import load_and_preprocess


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


def benchmark_accuracy(model: ResNet, imagenet_data_path: Path, batch_size: int = 32) -> tuple[float, float]:
  images = sample_imagenet_val(imagenet_data_path)
  id2synset = parse_imagenet_val_labels(imagenet_data_path)
  synset2idx = ImagenetClassIndex.load().synset_to_idx

  jmodel = TinyJit(model)
  jmodel(Tensor.rand(batch_size, 3, 224, 224)).realize()
  GlobalCounters.reset()
  jmodel(Tensor.rand(batch_size, 3, 224, 224)).realize()

  correct_top1 = correct_top5 = 0
  for start in tqdm(range(0, len(images), batch_size), desc="Benchmarking Accuracy"):
    batch_paths = images[start : start + batch_size]
    logits = jmodel(load_and_preprocess(batch_paths, pad_to_batch_size=batch_size)).numpy()
    for i, p in enumerate(batch_paths):
      pred = np.argsort(logits[i])[-5:][::-1]
      gt = synset2idx[id2synset[p.stem]]
      correct_top1 += int(pred[0] == gt)
      correct_top5 += int(gt in pred)
  return correct_top1 / len(images) * 100, correct_top5 / len(images) * 100
