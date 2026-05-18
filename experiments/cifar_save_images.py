from pathlib import Path

import numpy as np
from PIL import Image
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn.datasets import cifar

from aciq.helpers import RESULTS_DIR, get_output_dir


def save_split(x: Tensor, y: Tensor, out_dir: Path) -> None:
  imgs = x.numpy().transpose(0, 2, 3, 1).astype(np.uint8)
  labels = y.numpy().astype(int)
  for class_id in range(10):
    (out_dir / str(class_id)).mkdir(parents=True, exist_ok=True)
  for idx, (img, label) in enumerate(tqdm(list(zip(imgs, labels)), desc=out_dir.name)):
    Image.fromarray(img, mode="RGB").save(out_dir / str(int(label)) / f"{idx}.png")


if __name__ == "__main__":
  out_dir = get_output_dir(RESULTS_DIR, "cifar10_images")
  x_train, y_train, x_test, y_test = cifar()
  save_split(x_train, y_train, out_dir / "train")
  save_split(x_test, y_test, out_dir / "test")
  print(f"Saved to {out_dir}")
