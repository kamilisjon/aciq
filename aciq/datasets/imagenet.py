from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from tinygrad import Tensor
from tinygrad.helpers import tqdm

from aciq.helpers import load_csv


# --- ImageNet dataset interface -----------------------------------------------------------

_DEFAULT_CLASS_INDEX_PATH = Path(__file__).parent / "imagenet_class_index.json"


@dataclass(frozen=True)
class ImagenetClass:
  idx: int
  synset: str
  name: str


@dataclass
class ImagenetValLabelRow:
  ImageId: str
  PredictionString: str


class ImagenetClassIndex:
  def __init__(self, classes: tuple[ImagenetClass, ...]):
    self.classes = classes

  @classmethod
  def load(cls, path: Path = _DEFAULT_CLASS_INDEX_PATH) -> "ImagenetClassIndex":
    with path.open("r") as f:
      raw = json.load(f)
    return cls(tuple(ImagenetClass(idx=int(k), synset=v[0], name=v[1]) for k, v in raw.items()))

  @property
  def synset_to_idx(self) -> dict[str, int]:
    return {c.synset: c.idx for c in self.classes}


def parse_imagenet_val_labels(dataset_path: Path) -> dict[str, str]:
  rows = load_csv(dataset_path / "LOC_val_solution.csv", ImagenetValLabelRow)
  imageid_to_synset: dict[str, str] = {}
  for row in rows:
    tokens = row.PredictionString.strip().split()
    synsets = [tokens[i] for i in range(0, len(tokens), 5)]
    assert len(set(synsets)) == 1
    imageid_to_synset[row.ImageId] = synsets[0]
  return imageid_to_synset


def sample_imagenet_val(dataset_path: Path, n_per_class: int | None = None) -> list[Path]:
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


# --- Preprocessing ---------------------------------------------------------

_RESIZE_SIZE = 256
_CROP_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_and_center_crop(img: Image.Image) -> Image.Image:
  w, h = img.size
  if w <= h:
    new_w, new_h = _RESIZE_SIZE, int(_RESIZE_SIZE * h / w)
  else:
    new_h, new_w = _RESIZE_SIZE, int(_RESIZE_SIZE * w / h)
  img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
  left = (new_w - _CROP_SIZE) // 2
  top = (new_h - _CROP_SIZE) // 2
  return img.crop((left, top, left + _CROP_SIZE, top + _CROP_SIZE))


def normalize_to_chw(img: Image.Image) -> np.ndarray:
  arr = np.asarray(img, dtype=np.float32) / 255.0
  arr = arr.transpose(2, 0, 1)
  return (arr - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]


def _preprocess_one(img: Image.Image) -> np.ndarray:
  return normalize_to_chw(resize_and_center_crop(img))


def load_and_preprocess(image_paths: list[Path], pad_to_batch_size: int | None = None) -> Tensor:
  processed = [_preprocess_one(Image.open(p).convert("RGB")) for p in image_paths]
  if pad_to_batch_size is not None:
    n = len(processed)
    if n > pad_to_batch_size:
      raise ValueError(f"Received {n} images, but batch size is {pad_to_batch_size}. Cannot exceed batch size.")
    if n < pad_to_batch_size:
      processed.extend([np.zeros_like(processed[0]) for _ in range(pad_to_batch_size - n)])
  return Tensor(np.stack(processed))


# --- Benchmark --------------------------------------


def benchmark_accuracy(
  infer_fn: Callable[[Tensor], Tensor],
  image_paths: list[Path],
  id2synset: dict[str, str],
  synset2idx: dict[str, int],
  batch_size: int = 32,
) -> tuple[float, float]:
  correct_top1 = correct_top5 = 0
  for start in tqdm(range(0, len(image_paths), batch_size), desc="Benchmarking Accuracy"):
    batch_paths = image_paths[start : start + batch_size]
    logits = infer_fn(load_and_preprocess(batch_paths, pad_to_batch_size=batch_size)).numpy()
    for i, p in enumerate(batch_paths):
      pred = np.argsort(logits[i])[-5:][::-1]
      gt = synset2idx[id2synset[p.stem]]
      correct_top1 += int(pred[0] == gt)
      correct_top5 += int(gt in pred)
  return correct_top1 / len(image_paths) * 100, correct_top5 / len(image_paths) * 100
