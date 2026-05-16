import argparse
import copy
import random
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tinygrad import Tensor

from aciq.cam import compute_cam
from aciq.distributions import fit_distributions
from aciq.helpers import RESULTS_DIR, get_output_dir
from aciq.datasets.imagenet import ImagenetClassIndex, parse_imagenet_val_labels, resize_and_center_crop, sample_imagenet_val
from aciq.plotting_style import capped_savefig_dpi
from aciq.models.resnet import ResNet, _bias_correct_model, _weight_modules, compute_input_stats
from aciq.quantization.clipping import bound_symmetric_aciq_mae, bound_symmetric_minmax, quantize_symmetric


COLUMNS: list[tuple[str, str | None]] = [
  ("Tikra klasė", None),
  ("FP32", "fp32"),
  ("MinMax", "per_channel_minmax"),
  ("MinMax + posl. korek.", "per_channel_minmax_bias"),
  ("ACIQ", "per_channel_aciq"),
  ("ACIQ + posl. korek.", "per_channel_aciq_bias"),
]


_METHODS = {"per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"}


def _alpha_aciq(vec: np.ndarray, bits: int) -> float:
  best_dist = fit_distributions(vec)[0]
  alpha_max = bound_symmetric_minmax(vec)
  return float(bound_symmetric_aciq_mae(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_max))


def _quantize_per_tensor(weight: np.ndarray, bits: int, clip: str) -> np.ndarray:
  vec = weight.flatten().astype(np.float64)
  alpha = bound_symmetric_minmax(vec) if clip == "minmax" else _alpha_aciq(vec, bits)
  return quantize_symmetric(vec, alpha, bits).reshape(weight.shape)


def _quantize_per_channel(weight: np.ndarray, bits: int, clip: str) -> np.ndarray:
  out = np.empty_like(weight, dtype=np.float64)
  for c in range(weight.shape[0]):
    ch = weight[c].flatten().astype(np.float64)
    alpha = bound_symmetric_minmax(ch) if clip == "minmax" else _alpha_aciq(ch, bits)
    out[c] = quantize_symmetric(ch, alpha, bits).reshape(weight[c].shape)
  return out


def build_quantized_variant(base: ResNet, method: str, bits: int) -> ResNet:
  assert method in _METHODS, f"unknown method {method!r}; expected one of {sorted(_METHODS)}"
  q_model = copy.deepcopy(base)
  fp_mods = dict(_weight_modules(base))
  q_mods = dict(_weight_modules(q_model))
  for weight_name, q_module in q_mods.items():
    fp_weight = fp_mods[weight_name].weight.numpy().astype(np.float32)
    if method.startswith("per_tensor_"):
      q_weight = _quantize_per_tensor(fp_weight, bits, method.removeprefix("per_tensor_"))
    else:
      q_weight = _quantize_per_channel(fp_weight, bits, method.removeprefix("per_channel_"))
    q_module.weight = Tensor(q_weight.astype(np.float32))
  return q_model


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
  pils = [Image.open(p).convert("RGB") for p in chosen_paths]
  rgb_images = [np.asarray(resize_and_center_crop(pil)) for pil in pils]

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
  results: dict[str, list[tuple[np.ndarray, int, float]]] = {}
  for label, key in COLUMNS:
    if key is None:
      continue
    print(f"  {label}")
    m = variants_models[key]
    ResNet.clear_jit_caches()
    results[key] = [compute_cam(m, pil) for pil in pils]

  print("Rendering grid")
  cols = len(COLUMNS)
  rows = len(chosen_paths)
  cell_size = 2.0
  fig_w, fig_h = cols * cell_size, rows * cell_size + 0.3
  fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
  for r in range(rows):
    for c, (label, key) in enumerate(COLUMNS):
      ax = axes[r, c]
      if key is None:
        ax.imshow(rgb_images[r])
        ax.set_title(gt_names[r], fontsize=9, pad=2)
      else:
        overlaid, pred_idx, prob = results[key][r]
        ax.imshow(overlaid)
        pred_name = class_idx_db.classes[pred_idx].name.replace("_", " ")
        title_color = "tab:red" if pred_idx != gt_idx[r] else "tab:green"
        ax.set_title(f"{pred_name}\n{prob:.2f}", fontsize=9, pad=2, color=title_color)
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
