import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from aciq.cam import build_quantized_variant, cam_for_class, predict_batch_with_features
from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.imagenet import ImagenetClassIndex, load_and_preprocess, parse_imagenet_val_labels, sample_imagenet_val
from aciq.plotting_style import capped_savefig_dpi
from aciq.resnet import ResNet, _bias_correct_model, _weight_modules, compute_input_stats


COLUMNS: list[tuple[str, str | None]] = [
  ("Tikra klasė", None),
  ("FP32", "fp32"),
  ("MinMax", "per_channel_minmax"),
  ("MinMax + posl. korek.", "per_channel_minmax_bias"),
  ("ACIQ", "per_channel_aciq"),
  ("ACIQ + posl. korek.", "per_channel_aciq_bias"),
]


def _upsample_bilinear(cam: np.ndarray, size: int = 224) -> np.ndarray:
  """Resample a small CAM to a square `size`×`size` via PIL bilinear, preserving float values."""
  img = Image.fromarray(cam.astype(np.float32), mode="F")
  return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def _load_rgb_for_overlay(path: Path) -> np.ndarray:
  img = Image.open(path).convert("RGB")
  w, h = img.size
  if w <= h:
    new_w, new_h = 256, int(256 * h / w)
  else:
    new_h, new_w = 256, int(256 * w / h)
  img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
  left, top = (new_w - 224) // 2, (new_h - 224) // 2
  img = img.crop((left, top, left + 224, top + 224))
  return np.asarray(img)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ResNet18 class activation map grid across quantization variants.")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18"])
  parser.add_argument("--dataset-path", type=Path, required=True, help="ImageNet root.")
  parser.add_argument("--n-images", type=int, default=7)
  parser.add_argument("--bits", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  rng = random.Random(args.seed)

  print(f"Loading {args.model}")
  model = ResNet(int(args.model.removeprefix("resnet")))
  model.load_from_pretrained()
  model.fuse()

  print(f"Sampling {args.n_images} images (seed={args.seed})")
  all_paths = sample_imagenet_val(args.dataset_path, n_per_class=1)
  chosen_paths = rng.sample(all_paths, args.n_images)
  for p in chosen_paths:
    print(f"  {p.name}")

  save_dir = get_output_dir(RESULTS_DIR, f"{args.model}_cam")
  source_dir = save_dir / "source_images"
  source_dir.mkdir(parents=True, exist_ok=True)
  for i, p in enumerate(chosen_paths):
    shutil.copy(p, source_dir / f"{i:02d}_{p.name}")
  print(f"Copied source images to {source_dir}/")

  print("Preprocessing")
  x = load_and_preprocess(chosen_paths)
  rgb_images = [_load_rgb_for_overlay(p) for p in chosen_paths]

  print("Building quantized variants")
  variants_models: dict[str, ResNet] = {"fp32": model}
  for build_method in ("per_channel_minmax", "per_channel_aciq"):
    print(f"  {build_method}")
    variants_models[build_method] = build_quantized_variant(model, build_method, args.bits)
  print("  applying bias correction")
  fp_modules = dict(_weight_modules(model))
  input_stats = compute_input_stats(model)
  variants_models["per_channel_minmax_bias"] = _bias_correct_model(variants_models["per_channel_minmax"], fp_modules, input_stats)
  variants_models["per_channel_aciq_bias"] = _bias_correct_model(variants_models["per_channel_aciq"], fp_modules, input_stats)

  class_idx_db = ImagenetClassIndex.load()
  id2synset = parse_imagenet_val_labels(args.dataset_path)
  synset2idx = class_idx_db.synset_to_idx
  gt_idx = [synset2idx[id2synset[p.stem]] for p in chosen_paths]
  gt_names = [class_idx_db.classes[i].name.replace("_", " ") for i in gt_idx]

  print("Running inference per variant")
  results: dict[str, tuple[list[np.ndarray], np.ndarray, np.ndarray]] = {}
  for label, key in COLUMNS:
    if key is None:
      continue
    print(f"  {label}")
    m = variants_models[key]
    fc_weight = m.fc.weight.numpy()
    feat, class_indices, probs = predict_batch_with_features(m, x)
    cams = [cam_for_class(feat[i], fc_weight, int(class_indices[i])) for i in range(len(chosen_paths))]
    results[key] = (cams, class_indices, probs)

  print("Rendering grid")
  cols = len(COLUMNS)
  rows = len(chosen_paths)
  cell_size = 2.0
  fig_w, fig_h = cols * cell_size, rows * cell_size + 0.3
  fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
  for r in range(rows):
    for c, (label, key) in enumerate(COLUMNS):
      ax = axes[r, c]
      ax.imshow(rgb_images[r])
      if key is None:
        ax.set_title(gt_names[r], fontsize=9, pad=2)
      else:
        cams, class_indices, probs = results[key]
        cam = _upsample_bilinear(cams[r], size=224)
        cam_norm = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-9)
        ax.imshow(cam_norm, cmap="jet", alpha=0.45, extent=(0, 224, 224, 0), interpolation="nearest")
        pred_idx = int(class_indices[r])
        pred_name = class_idx_db.classes[pred_idx].name.replace("_", " ")
        title_color = "tab:red" if pred_idx != gt_idx[r] else "tab:green"
        ax.set_title(f"{pred_name}\n{float(probs[r]):.2f}", fontsize=9, pad=2, color=title_color)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.grid(False)
  for c, (label, _) in enumerate(COLUMNS):
    x_norm = (c + 0.5) / cols
    fig.text(x_norm, 0.99, label, ha="center", va="top", fontsize=10, fontweight="bold")
  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

  out_path = save_dir / "cam_grid.png"
  dpi = capped_savefig_dpi(fig_w, fig_h)
  fig.savefig(out_path, dpi=dpi)
  plt.close(fig)
  print(f"Saved {out_path}")
