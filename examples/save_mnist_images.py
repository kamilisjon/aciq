import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn.datasets import mnist


def save_split(x: Tensor, y: Tensor, out_dir: Path) -> None:
  imgs = x.numpy().squeeze(1).astype(np.uint8)
  labels = y.numpy().astype(int)
  for class_id in range(10):
    (out_dir / str(class_id)).mkdir(parents=True, exist_ok=True)
  for idx, (img, label) in enumerate(tqdm(list(zip(imgs, labels)), desc=out_dir.name)):
    Image.fromarray(img, mode="L").save(out_dir / str(int(label)) / f"{idx}.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Save MNIST images to disk, organized by class")
  parser.add_argument("--out-dir", type=Path, default=Path("results/mnist_images"))
  args = parser.parse_args()
  x_train, y_train, x_test, y_test = mnist()
  save_split(x_train, y_train, args.out_dir / "train")
  save_split(x_test, y_test, args.out_dir / "test")
  print(f"Saved to {args.out_dir}")
