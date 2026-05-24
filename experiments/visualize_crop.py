import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from aciq.datasets.imagenet import compute_orig_crop_box
from aciq.plotting_style import SERIES_COLORS, capped_savefig_dpi


def visualize_crop(image_path: Path, out_path: Path) -> None:
  img = Image.open(image_path).convert("RGB")
  left, top, right, bottom = compute_orig_crop_box(img.size)

  w, h = img.size
  fig_w_in = 8.0
  fig_h_in = fig_w_in * h / w
  fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
  ax.imshow(np.asarray(img))
  ax.add_patch(Rectangle(
    (left, top), right - left, bottom - top,
    fill=False, linewidth=2, edgecolor=SERIES_COLORS[0],
  ))
  ax.set_xticks([])
  ax.set_yticks([])
  ax.grid(False)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=capped_savefig_dpi(fig_w_in, fig_h_in))
  plt.close(fig)
  print(f"Saved {out_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Visualize the 224x224 center-crop region on an original image.")
  parser.add_argument("--image", type=Path, required=True, help="Path to source image.")
  parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: experiments/outputs/crop_<stem>.png).")
  args = parser.parse_args()

  out = args.out or Path(__file__).parent / "outputs" / f"crop_{args.image.stem}.png"
  visualize_crop(args.image, out)


if __name__ == "__main__":
  main()
