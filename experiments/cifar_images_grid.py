import numpy as np
from PIL import Image
from tinygrad.nn.datasets import cifar

from aciq.helpers import RESULTS_DIR

IMG_SIZE = 32
SEP = 4
ROWS, COLS = 4, 5
EXAMPLES_PER_CLASS = 2
OUT_PATH = RESULTS_DIR / "cifar10_grid.png"

x_train, y_train, _, _ = cifar()
images = x_train.numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # (N, 32, 32, 3)
labels = y_train.numpy()

class_images: dict[int, list[np.ndarray]] = {c: [] for c in range(10)}
for img, label in zip(images, labels):
  label = int(label)
  if len(class_images[label]) < EXAMPLES_PER_CLASS:
    class_images[label].append(img)
  if all(len(v) == EXAMPLES_PER_CLASS for v in class_images.values()):
    break

grid_h = ROWS * IMG_SIZE + (ROWS - 1) * SEP
grid_w = COLS * IMG_SIZE + (COLS - 1) * SEP
canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

for class_id in range(10):
  block_row, col = divmod(class_id, COLS)
  for example_idx in range(EXAMPLES_PER_CLASS):
    row = block_row * EXAMPLES_PER_CLASS + example_idx
    y = row * (IMG_SIZE + SEP)
    x = col * (IMG_SIZE + SEP)
    canvas[y : y + IMG_SIZE, x : x + IMG_SIZE] = class_images[class_id][example_idx]

Image.fromarray(canvas, mode="RGB").save(OUT_PATH)
print(f"Saved {OUT_PATH} ({grid_w}x{grid_h})")
