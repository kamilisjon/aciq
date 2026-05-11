import numpy as np
from PIL import Image
from tinygrad.nn.datasets import mnist

from aciq.helpers import RESULTS_DIR

IMG_SIZE = 28
SEP = 4
ROWS, COLS = 4, 5
EXAMPLES_PER_DIGIT = 2
OUT_PATH = RESULTS_DIR / "mnist_grid.png"

x_train, y_train, _, _ = mnist()
images = x_train.numpy().reshape(-1, IMG_SIZE, IMG_SIZE)
labels = y_train.numpy()

digit_images: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
for img, label in zip(images, labels):
  label = int(label)
  if len(digit_images[label]) < EXAMPLES_PER_DIGIT:
    digit_images[label].append(img)
  if all(len(v) == EXAMPLES_PER_DIGIT for v in digit_images.values()):
    break

grid_h = ROWS * IMG_SIZE + (ROWS - 1) * SEP
grid_w = COLS * IMG_SIZE + (COLS - 1) * SEP
canvas = np.full((grid_h, grid_w), 255, dtype=np.uint8)

for digit in range(10):
  block_row, col = divmod(digit, COLS)
  for example_idx in range(EXAMPLES_PER_DIGIT):
    row = block_row * EXAMPLES_PER_DIGIT + example_idx
    y = row * (IMG_SIZE + SEP)
    x = col * (IMG_SIZE + SEP)
    canvas[y : y + IMG_SIZE, x : x + IMG_SIZE] = digit_images[digit][example_idx]

Image.fromarray(canvas).save(OUT_PATH)
print(f"Saved {OUT_PATH} ({grid_w}x{grid_h})")
