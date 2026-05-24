import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tinygrad import Tensor
from tinygrad.helpers import tqdm
from tinygrad.nn import Conv2d, Linear

from aciq.datasets.imagenet import ImagenetClassIndex, benchmark_accuracy, load_and_preprocess, parse_imagenet_val_labels, resize_and_center_crop, sample_imagenet_val
from aciq.quantization.bias_correction import ChannelMeansAccumulator, MeanShift
from aciq.helpers import RESULTS_DIR, get_output_dir, load_csv, mean_absolute_error, save_csv
from aciq.models.resnet import ResNet, compose_cam_overlay, compute_input_stats, _weight_modules, _bias_correct_model
from aciq.quantization.clipping import quantize_symmetric, bound_symmetric_minmax, bound_symmetric_aciq_mae
from aciq.distributions import fit_distributions, kurtosis, skewness
from aciq.plotting_style import DIST_COLORS, HIST_BINS, LINE_WIDTH, NEUTRAL_COLOR, SERIES_COLORS, STATS_TEXT_KW, capped_savefig_dpi


METHOD_NAMES = ["per_tensor_minmax", "per_tensor_aciq", "per_channel_minmax", "per_channel_aciq"]
PER_CHANNEL_METHODS = {"per_channel_minmax", "per_channel_aciq"}
BIT_WIDTHS: tuple[int, ...] = (4, 8)


@dataclass
class MaeComparisonRow:
  layer_idx: int
  op_type: str
  name: str
  n: int
  n_per_ch: int
  ch_count: int
  err: float
  err_aciq: float
  err_channel: float
  err_channel_aciq: float


@dataclass
class BenchmarkRow:
  method: str
  correction_mode: str
  top1: float
  top5: float


@dataclass
class CamCosineRow:
  image_id: str
  gt_class_idx: int
  gt_class_name: str
  bit_width: int
  method: str
  bias_corrected: bool
  cosine: float


@dataclass
class GlobalSummaryRow:
  bit_width: int
  method: str
  bias_corrected: bool
  mae: float
  top1: float
  n_images: int
  mean_cosine: float
  std_cosine: float


def analyze_layer(vec: np.ndarray, layer_name: str, layer_idx: int, bits: int, save_path: Path | None = None) -> tuple[float, float]:
  vec_sorted = np.sort(vec)

  fits = fit_distributions(vec_sorted)

  # MinMax quantization
  alpha_minmax = bound_symmetric_minmax(vec)
  mae_minmax = mean_absolute_error(vec, quantize_symmetric(vec, alpha_minmax, bits))

  # Optimal alpha*
  best_dist = fits[0]
  alpha_aciq = bound_symmetric_aciq_mae(cdf=lambda x: float(best_dist.cdf_at(np.asarray(x))), b=bits, alpha_max=alpha_minmax)

  if save_path is not None:
    save_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vec, bins=HIST_BINS, density=True, alpha=0.5, color=NEUTRAL_COLOR, label="Empirinis")

    for fitted in fits:
      ax.plot(
        vec_sorted,
        fitted.pdf(),
        color=DIST_COLORS[type(fitted)],
        linewidth=LINE_WIDTH,
        linestyle="--",
        label=f"{repr(fitted)}",
      )

    ax.axvline(-alpha_minmax, color=NEUTRAL_COLOR, linestyle=":", linewidth=LINE_WIDTH, label=f"MinMax α={alpha_minmax:.4f} MAE={mae_minmax:.2e}")
    ax.axvline(alpha_minmax, color=NEUTRAL_COLOR, linestyle=":", linewidth=LINE_WIDTH)

    if alpha_aciq != alpha_minmax:
      mae_aciq = mean_absolute_error(vec, quantize_symmetric(vec, alpha_aciq, bits))
      ax.axvline(
        -alpha_aciq,
        color=DIST_COLORS[type(best_dist)],
        linestyle="-",
        linewidth=LINE_WIDTH,
        label=f"ACIQ {best_dist.name} α={alpha_aciq:.4f} MAE={mae_aciq:.2e}",
      )
      ax.axvline(alpha_aciq, color=DIST_COLORS[type(best_dist)], linestyle="-", linewidth=LINE_WIDTH)

    eda_lines = [
      f"n {vec.size:,}",
      f"Min {float(np.min(vec)):.5f}",
      f"Max {float(np.max(vec)):.5f}",
      f"Mean {float(np.mean(vec)):.5f}",
      f"Variance {float(np.var(vec)):.6f}",
      f"Skewness {float(skewness(vec)):.4f}",
      f"Kurtosis {float(kurtosis(vec)):.4f}",
    ]
    ax.text(0.98, 0.96, "\n".join(eda_lines), transform=ax.transAxes, **STATS_TEXT_KW)
    safe = layer_name.replace("/", "_").replace(":", "_")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path / f"layer_{layer_idx:03d}_{safe[:60]}.png")
    plt.close(fig)

  return alpha_minmax, alpha_aciq


def plot_shift(
  rows: list[MeanShift],
  layer_names: list[str],
  save_dir: Path,
  filename: str = "mean_shift.png",
) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)

  by_method: dict[str, dict[str, float]] = {}
  for r in rows:
    by_method.setdefault(r.method, {})[r.layer] = r.mean_shift

  methods = list(by_method.keys())
  n_methods = len(methods)
  x_pos = np.arange(len(layer_names))
  bar_width = 0.7 / n_methods
  offsets = np.linspace(-0.35 + bar_width / 2, 0.35 - bar_width / 2, n_methods)

  fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.5), 5))
  for i, method in enumerate(methods):
    per_layer = [by_method[method][name] for name in layer_names]
    label = "Bias correction" if method.endswith("::bias") else "No correction"
    ax.bar(x_pos + offsets[i], per_layer, width=bar_width, color=SERIES_COLORS[i % len(SERIES_COLORS)], label=label)

  ax.set_xticks(x_pos)
  ax.set_xticklabels(layer_names, rotation=45, ha="right")
  ax.set_xlabel("Layer")
  ax.set_ylabel("Output mean shift")
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_dir / filename)
  plt.close(fig)


@dataclass
class PipelineConfig:
  model_depth: int
  bits: int
  dataset_path: Path
  plot_per_channel: bool
  output_dir: Path
  n_per_class: int | None = None

  @property
  def model_name(self) -> str:
    return f"resnet{self.model_depth}"

  @property
  def weight_results_dir(self) -> Path:
    return self.output_dir / "weight_analysis"

  @property
  def shift_results_dir(self) -> Path:
    return self.output_dir / "quantization_shift"

  @property
  def correction_results_dir(self) -> Path:
    return self.output_dir / "bias_variance_correction"


# ---------------------------------------------------------------------------
# Stage 2: Weight Distribution Analysis
# ---------------------------------------------------------------------------


def stage_weight_analysis(config: PipelineConfig, fused_model: ResNet, fq_models: dict[str, ResNet]) -> None:
  weight_modules = _weight_modules(fused_model)
  print(f"  Total Conv/Linear layers: {len(weight_modules)}")
  fq_lookups: dict[str, dict[str, Conv2d | Linear]] = {m: dict(_weight_modules(fq_models[m])) for m in METHOD_NAMES}

  config.weight_results_dir.mkdir(parents=True, exist_ok=True)
  csv_path = config.weight_results_dir / "mae_comparison.csv"
  rows: list[MaeComparisonRow] = []

  for layer_idx, (weight_name, module) in enumerate(weight_modules, 1):
    op_type = "Linear" if isinstance(module, Linear) else "Conv"
    weight_arr = module.weight.numpy().astype(np.float32)
    vec = weight_arr.flatten()
    safe_name = weight_name.replace("/", "_").replace(":", "_")[:60]

    # Per-tensor
    plot_dir = config.weight_results_dir / "per_tensor"
    alpha_minmax, alpha_aciq = analyze_layer(vec, weight_name, layer_idx, config.bits, plot_dir)
    q_mm = quantize_symmetric(vec, alpha_minmax, config.bits)
    q_ac = quantize_symmetric(vec, alpha_aciq, config.bits)

    fq_lookups["per_tensor_minmax"][weight_name].weight = Tensor(q_mm.reshape(weight_arr.shape).astype(np.float32))
    fq_lookups["per_tensor_aciq"][weight_name].weight = Tensor(q_ac.reshape(weight_arr.shape).astype(np.float32))

    # Per-channel (axis 0 = output channels/features)
    ch_plot_dir = config.weight_results_dir / "per_channel" / f"{layer_idx:03d}_{safe_name}" if config.plot_per_channel else None
    total_err_minmax = 0.0
    total_err_aciq = 0.0
    fq_ch_mm = np.empty_like(weight_arr)
    fq_ch_ac = np.empty_like(weight_arr)
    for ch in range(weight_arr.shape[0]):
      ch_vec = weight_arr[ch].flatten()
      ch_alpha_minmax, ch_alpha_aciq = analyze_layer(ch_vec, f"{weight_name}/ch{ch}", ch, config.bits, ch_plot_dir)
      q_mm_ch = quantize_symmetric(ch_vec, ch_alpha_minmax, config.bits)
      q_ac_ch = quantize_symmetric(ch_vec, ch_alpha_aciq, config.bits)
      total_err_minmax += float(np.sum(np.abs(ch_vec - q_mm_ch)))
      total_err_aciq += float(np.sum(np.abs(ch_vec - q_ac_ch)))
      fq_ch_mm[ch] = q_mm_ch.reshape(weight_arr[ch].shape)
      fq_ch_ac[ch] = q_ac_ch.reshape(weight_arr[ch].shape)

    fq_lookups["per_channel_minmax"][weight_name].weight = Tensor(fq_ch_mm.astype(np.float32))
    fq_lookups["per_channel_aciq"][weight_name].weight = Tensor(fq_ch_ac.astype(np.float32))

    print(f"  [{layer_idx:>3}] {op_type:6s} {weight_name:40} n={len(vec):,}")
    rows.append(
      MaeComparisonRow(
        layer_idx=layer_idx,
        op_type=op_type,
        name=weight_name,
        n=len(vec),
        n_per_ch=len(ch_vec),
        ch_count=int(len(vec) / len(ch_vec)),
        err=float(np.sum(np.abs(vec - q_mm))),
        err_aciq=float(np.sum(np.abs(vec - q_ac))),
        err_channel=total_err_minmax,
        err_channel_aciq=total_err_aciq,
      )
    )

  save_csv(rows, csv_path)
  print(f"  CSV written to {csv_path}")


def _collect_activations(model: ResNet, image_paths: list[Path], batch_size: int = 32) -> ChannelMeansAccumulator:
  acc = ChannelMeansAccumulator()
  for start in tqdm(range(0, len(image_paths), batch_size), desc="  activations"):
    batch_paths = image_paths[start : start + batch_size]
    real_n = len(batch_paths)
    activations = model.get_activations(load_and_preprocess(batch_paths, pad_to_batch_size=batch_size))
    for name, act in activations.items():
      # slice off the zero-padded tail so accumulated stats only reflect real images
      acc.update(name, (act[:real_n] if real_n < batch_size else act).numpy())
  return acc


def _collect_cams_and_predictions(
  model: ResNet,
  image_paths: list[Path],
  gt_indices: np.ndarray,
  batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """For each image, return the GT-class 7x7 CAM, the model's argmax prediction, and its probability.

  Returns three arrays aligned with ``image_paths`` (length N):
    - cams: (N, 7, 7) float32
    - pred_indices: (N,) int64
    - pred_probs: (N,) float64
  """
  n = len(image_paths)
  cams = np.empty((n, 7, 7), dtype=np.float32)
  pred_indices = np.empty(n, dtype=np.int64)
  pred_probs = np.empty(n, dtype=np.float64)

  fc_w = model.fc.weight.numpy()
  fc_b = model.fc.bias.numpy() if model.fc.bias is not None else np.zeros(fc_w.shape[0], dtype=np.float32)

  for start in tqdm(range(0, n, batch_size), desc="  cams"):
    batch_paths = image_paths[start : start + batch_size]
    real_n = len(batch_paths)
    x = load_and_preprocess(batch_paths, pad_to_batch_size=batch_size)
    activations = model.get_activations(x)
    feat = activations["layer4.1.activation_2"].numpy()[:real_n]  # (real_n, C, 7, 7)
    gap = feat.mean(axis=(2, 3))
    logits = gap @ fc_w.T + fc_b
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / exps.sum(axis=1, keepdims=True)
    batch_preds = np.argmax(probs, axis=1)
    batch_probs = np.take_along_axis(probs, batch_preds[:, None], axis=1).squeeze(1)
    batch_gt = gt_indices[start : start + real_n]
    gt_class_weights = fc_w[batch_gt]  # (real_n, C)
    batch_cams = np.einsum("bchw,bc->bhw", feat, gt_class_weights)
    cams[start : start + real_n] = batch_cams.astype(np.float32)
    pred_indices[start : start + real_n] = batch_preds.astype(np.int64)
    pred_probs[start : start + real_n] = batch_probs.astype(np.float64)

  return cams, pred_indices, pred_probs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  na = float(np.linalg.norm(a))
  nb = float(np.linalg.norm(b))
  if na == 0.0 or nb == 0.0:
    return float("nan")
  return float(np.dot(a.flatten(), b.flatten()) / (na * nb))


def _assemble_global_summary(
  cosine_rows: list[CamCosineRow],
  save_dir: Path,
) -> list[GlobalSummaryRow]:
  out: list[GlobalSummaryRow] = []
  variant_keys = [
    (False, "minmax"), (True, "minmax"), (False, "aciq"), (True, "aciq"),
  ]
  for bits in BIT_WIDTHS:
    bench = {(r.method, r.correction_mode): r for r in load_csv(save_dir / f"{bits}bits" / "bias_variance_correction" / "benchmark_results.csv", BenchmarkRow)}
    mae_rows = load_csv(save_dir / f"{bits}bits" / "weight_analysis" / "mae_comparison.csv", MaeComparisonRow)
    total_n = sum(r.n for r in mae_rows)
    mae_minmax = sum(r.err_channel for r in mae_rows) / total_n
    mae_aciq = sum(r.err_channel_aciq for r in mae_rows) / total_n
    mae_by_method = {"minmax": mae_minmax, "aciq": mae_aciq}
    for bias_corrected, method in variant_keys:
      mode = "bias" if bias_corrected else "none"
      full_method = f"per_channel_{method}"
      values = np.array([
        r.cosine for r in cosine_rows
        if r.bit_width == bits and r.method == method and r.bias_corrected == bias_corrected
      ])
      finite = values[np.isfinite(values)]
      mean_c = float(finite.mean()) if finite.size else float("nan")
      std_c = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
      out.append(GlobalSummaryRow(
        bit_width=bits,
        method=method,
        bias_corrected=bias_corrected,
        mae=mae_by_method[method],
        top1=float(bench[(full_method, mode)].top1),
        n_images=int(finite.size),
        mean_cosine=mean_c,
        std_cosine=std_c,
      ))
  return out


def plot_cam_cosine_bars(summary: list[GlobalSummaryRow], out_path: Path) -> None:
  group_labels = ["INT4 MinMax", "INT4 ACIQ", "INT8 MinMax", "INT8 ACIQ"]
  groups: list[tuple[int, str]] = [(4, "minmax"), (4, "aciq"), (8, "minmax"), (8, "aciq")]
  by_key: dict[tuple[int, str, bool], GlobalSummaryRow] = {(r.bit_width, r.method, r.bias_corrected): r for r in summary}
  x_pos = np.arange(len(groups))
  bar_width = 0.38
  no_corr_means = [by_key[(b, m, False)].mean_cosine for b, m in groups]
  no_corr_stds = [by_key[(b, m, False)].std_cosine for b, m in groups]
  bias_means = [by_key[(b, m, True)].mean_cosine for b, m in groups]
  bias_stds = [by_key[(b, m, True)].std_cosine for b, m in groups]

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.bar(x_pos - bar_width / 2, no_corr_means, bar_width, yerr=no_corr_stds, capsize=3, color=SERIES_COLORS[0], label="No correction")
  ax.bar(x_pos + bar_width / 2, bias_means, bar_width, yerr=bias_stds, capsize=3, color=SERIES_COLORS[1], label="Bias correction")
  ax.axvline(1.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8)
  ax.set_xticks(x_pos)
  ax.set_xticklabels(group_labels)
  ax.set_ylabel("Mean CAM cosine similarity vs FP32")
  ax.set_ylim(0.0, 1.0)
  ax.legend(loc="lower right")
  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path)
  plt.close(fig)


@dataclass
class Selection:
  letter: str            # "A" | "B" | "C" | "D"
  criterion: str         # human-readable label
  index: int | None      # index into image_paths; None when the set is empty


_SELECTION_CRITERIA = (
  ("A", "most_confident_correct"),
  ("B", "most_confident_incorrect"),
  ("C", "least_confident_correct"),
  ("D", "least_confident_incorrect"),
)


def select_fp32_confidence_images(fp32_preds: np.ndarray, fp32_probs: np.ndarray, gt_indices: np.ndarray) -> list[Selection]:
  correct = fp32_preds == gt_indices
  selections: list[Selection] = []
  for letter, criterion in _SELECTION_CRITERIA:
    if criterion.endswith("_correct"):
      mask = correct
    else:
      mask = ~correct
    candidates = np.where(mask)[0]
    if candidates.size == 0:
      selections.append(Selection(letter=letter, criterion=criterion, index=None))
      continue
    if criterion.startswith("most"):
      pick = int(candidates[np.argmax(fp32_probs[candidates])])
    else:
      pick = int(candidates[np.argmin(fp32_probs[candidates])])
    selections.append(Selection(letter=letter, criterion=criterion, index=pick))
  return selections


def _variant_key(bits: int, method: str, bias_corrected: bool) -> str:
  return f"INT{bits}_per_channel_{method}{'_bias' if bias_corrected else ''}"


def write_selected_cams_json(
  out_path: Path,
  selections: list[Selection],
  image_paths: list[Path],
  gt_indices: np.ndarray,
  gt_names: list[str],
  fp32_cams: np.ndarray,
  fp32_preds: np.ndarray,
  fp32_probs: np.ndarray,
  variant_cams: dict[tuple[int, str, bool], np.ndarray],
  variant_preds: dict[tuple[int, str, bool], np.ndarray],
  variant_probs: dict[tuple[int, str, bool], np.ndarray],
  class_idx_db: ImagenetClassIndex,
) -> None:
  selections_payload: dict[str, dict] = {}
  predictions_payload: dict[str, dict] = {}
  cams_payload: dict[str, dict] = {}
  for sel in selections:
    if sel.index is None:
      selections_payload[sel.letter] = {"criterion": sel.criterion, "image_path": None, "gt_idx": None, "gt_name": None}
      predictions_payload[sel.letter] = {}
      cams_payload[sel.letter] = {}
      continue
    i = sel.index
    selections_payload[sel.letter] = {
      "criterion": sel.criterion,
      "image_path": str(image_paths[i]),
      "gt_idx": int(gt_indices[i]),
      "gt_name": gt_names[i],
    }
    preds_section: dict[str, dict] = {}
    cams_section: dict[str, list] = {}
    preds_section["fp32"] = {
      "pred_idx": int(fp32_preds[i]),
      "pred_name": class_idx_db.classes[int(fp32_preds[i])].name.replace("_", " "),
      "prob": float(fp32_probs[i]),
    }
    cams_section["fp32"] = fp32_cams[i].tolist()
    for (bits, method, bias_corrected), cams in variant_cams.items():
      key = _variant_key(bits, method, bias_corrected)
      preds_section[key] = {
        "pred_idx": int(variant_preds[(bits, method, bias_corrected)][i]),
        "pred_name": class_idx_db.classes[int(variant_preds[(bits, method, bias_corrected)][i])].name.replace("_", " "),
        "prob": float(variant_probs[(bits, method, bias_corrected)][i]),
      }
      cams_section[key] = cams[i].tolist()
    predictions_payload[sel.letter] = preds_section
    cams_payload[sel.letter] = cams_section

  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    json.dump({"selections": selections_payload, "predictions": predictions_payload, "cams": cams_payload}, f, indent=2)


def load_selected_cams_json(parent_dir: Path) -> dict | None:
  path = parent_dir / "selected_cams.json"
  if not path.exists():
    return None
  with path.open() as f:
    return json.load(f)


def save_selected_image_sidecars(out_dir: Path, selections: list[Selection], image_paths: list[Path]) -> None:
  for sel in selections:
    if sel.index is None:
      continue
    src = image_paths[sel.index]
    cropped = resize_and_center_crop(Image.open(src).convert("RGB"))
    cropped.save(out_dir / f"selected_image_{sel.letter}.png")


_CAM_GRID_COLUMNS: list[tuple[str, str | None]] = [
  ("Ground Truth", None),
  ("FP32", "fp32"),
  ("MinMax", "per_channel_minmax"),
  ("MinMax + Bias Corr", "per_channel_minmax_bias"),
  ("ACIQ", "per_channel_aciq"),
  ("ACIQ + Bias Corr", "per_channel_aciq_bias"),
]

_SECTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


def _column_key(bits: int, kind: str | None) -> str | None:
  if kind is None or kind == "fp32":
    return kind
  return f"INT{bits}_{kind}"


def plot_qualitative_grid(
  sections: dict[str, dict],
  out_path: Path,
) -> None:
  """Render the 8-row x 6-col qualitative grid.

  ``sections`` maps section letter -> per-section payload:
    {
      "criterion": str,
      "gt_name": str | None,
      "image": np.ndarray (H, W, 3) uint8 or None when section is empty,
      "predictions": {column_key: {"pred_idx": int, "pred_name": str, "prob": float}, ...} or {},
      "cams": {column_key: np.ndarray (7, 7) float32, ...} or {},
    }
  """
  cols = len(_CAM_GRID_COLUMNS)
  rows = len(_SECTION_LETTERS) * len(BIT_WIDTHS)
  cell_size = 2.0
  fig_w, fig_h = cols * cell_size, rows * cell_size + 1.0
  fig = plt.figure(figsize=(fig_w, fig_h))
  gs = fig.add_gridspec(rows, cols, hspace=0.35)

  for section_idx, letter in enumerate(_SECTION_LETTERS):
    payload = sections.get(letter, {})
    img = payload.get("image")
    cams = payload.get("cams", {})
    preds = payload.get("predictions", {})
    gt_idx = payload.get("gt_idx")
    row_top = section_idx * len(BIT_WIDTHS)

    ax_gt = fig.add_subplot(gs[row_top : row_top + len(BIT_WIDTHS), 0])
    ax_fp32 = fig.add_subplot(gs[row_top : row_top + len(BIT_WIDTHS), 1])
    for ax in (ax_gt, ax_fp32):
      ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    if img is None:
      ax_gt.set_visible(False)
      ax_fp32.set_visible(False)
    else:
      ax_gt.imshow(img)
      ax_gt.set_title(payload.get("gt_name", ""), fontsize=9, pad=2)
      fp32_cam = cams.get("fp32")
      fp32_pred = preds.get("fp32")
      if fp32_cam is None or fp32_pred is None:
        ax_fp32.set_visible(False)
      else:
        composite = compose_cam_overlay(np.asarray(fp32_cam, dtype=np.float32), img)
        ax_fp32.imshow(composite)
        title_color = "tab:green" if fp32_pred["pred_idx"] == gt_idx else "tab:red"
        ax_fp32.set_title(f"{fp32_pred['pred_name']}\n{fp32_pred['prob']:.2f}", fontsize=9, pad=2, color=title_color)

    for bit_row_idx, bits in enumerate(BIT_WIDTHS):
      r = row_top + bit_row_idx
      for c_offset, (_col_label, kind) in enumerate(_CAM_GRID_COLUMNS[2:]):
        c = c_offset + 2
        ax = fig.add_subplot(gs[r, c])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if c_offset == 0:
          ax.set_ylabel(f"INT{bits}", rotation=0, ha="right", va="center", labelpad=20, fontsize=10)
        if img is None:
          ax.set_visible(False)
          continue
        key = _column_key(bits, kind)
        cam = cams.get(key)
        pred = preds.get(key)
        if cam is None or pred is None:
          ax.set_visible(False)
          continue
        composite = compose_cam_overlay(np.asarray(cam, dtype=np.float32), img)
        ax.imshow(composite)
        title_color = "tab:green" if pred["pred_idx"] == gt_idx else "tab:red"
        ax.set_title(f"{pred['pred_name']}\n{pred['prob']:.2f}", fontsize=9, pad=2, color=title_color)

  # Column titles at the top of the figure.
  for c, (col_label, _) in enumerate(_CAM_GRID_COLUMNS):
    x_norm = (c + 0.5) / cols
    fig.text(x_norm, 0.995, col_label, ha="center", va="top", fontsize=10, fontweight="bold")

  # Section headers: just the letter, on the left edge of each section.
  for section_idx, letter in enumerate(_SECTION_LETTERS):
    y_top = 1.0 - (section_idx * len(BIT_WIDTHS)) / rows * (1.0 - 0.02)
    fig.text(0.005, y_top - 0.01, letter, ha="left", va="top", fontsize=12, fontweight="bold")

  fig.tight_layout(rect=(0.05, 0.0, 1.0, 0.97))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  dpi = capped_savefig_dpi(fig_w, fig_h)
  fig.savefig(out_path, dpi=dpi)
  plt.close(fig)


def build_sections_from_disk(parent_dir: Path) -> dict[str, dict]:
  payload = load_selected_cams_json(parent_dir)
  if payload is None:
    return {}
  sections: dict[str, dict] = {}
  for letter, sel in payload["selections"].items():
    if sel.get("image_path") is None:
      sections[letter] = {"criterion": sel["criterion"], "image": None, "cams": {}, "predictions": {}, "gt_idx": None, "gt_name": None}
      continue
    img = np.asarray(Image.open(parent_dir / f"selected_image_{letter}.png").convert("RGB"))
    cams_section = {k: np.asarray(v, dtype=np.float32) for k, v in payload["cams"][letter].items()}
    sections[letter] = {
      "criterion": sel["criterion"],
      "image": img,
      "cams": cams_section,
      "predictions": payload["predictions"][letter],
      "gt_idx": sel["gt_idx"],
      "gt_name": sel["gt_name"],
    }
  return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def rerender_shift_plots(parent_dir: Path) -> None:
  for bits in BIT_WIDTHS:
    sub = parent_dir / f"{bits}bits"
    shifts_path = sub / "quantization_shift" / "shifts.csv"
    if not shifts_path.exists():
      print(f"skipping {bits}bits — {shifts_path} not found")
      continue
    loaded_rows: list[MeanShift] = load_csv(shifts_path, MeanShift)
    print(f"Loaded {len(loaded_rows)} shift rows from {shifts_path}")
    layer_names = list(dict.fromkeys(r.layer for r in loaded_rows))
    minmax_rows = [r for r in loaded_rows if r.method.startswith("per_channel_minmax")]
    aciq_rows = [r for r in loaded_rows if r.method.startswith("per_channel_aciq")]
    plot_shift(minmax_rows, layer_names, sub / "quantization_shift", filename="mean_shift_minmax.png")
    plot_shift(aciq_rows, layer_names, sub / "quantization_shift", filename="mean_shift_aciq.png")
    print(f"Plots saved to {sub / 'quantization_shift'}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Full ACIQ quantization analysis pipeline")
  parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
  parser.add_argument("--dataset-path", type=Path, default=None, help="Path to ImageNet dataset root (required unless --from-dir is set).")
  parser.add_argument("--plot-per-channel", action="store_true", help="Generate per-channel weight distribution plots (slow)")
  parser.add_argument(
    "--n-per-class",
    type=int,
    default=None,
    help="Sample N ImageNet val images per class for shift analysis and benchmarking. Default: use the full validation set.",
  )
  parser.add_argument(
    "--from-dir",
    type=Path,
    default=None,
    help="Load `{bits}bits/quantization_shift/shifts.csv` from each per-bit subdir of this parent directory and re-render the shift plots only (no model loading, no inference).",
  )
  args = parser.parse_args()

  save_dir = get_output_dir(RESULTS_DIR, args.model)
  print(f"Output directory: {save_dir}")

  if args.from_dir:
    rerender_shift_plots(args.from_dir)
    sections = build_sections_from_disk(args.from_dir)
    if sections:
      plot_qualitative_grid(sections, args.from_dir / "qualitative_grid.png")
      print(f"Re-rendered {args.from_dir / 'qualitative_grid.png'}")
    else:
      print(f"skipping qualitative grid — selected_cams.json missing in {args.from_dir}")
    return

  assert args.dataset_path is not None, "--dataset-path is required unless --from-dir is set"
  model_depth = int(args.model.removeprefix("resnet"))

  print(f"Loading {args.model}")
  model = ResNet(model_depth)
  model.load_from_pretrained()
  model.fuse()

  # Shared FP32 work (bit-width independent).
  image_paths = sample_imagenet_val(args.dataset_path, args.n_per_class)
  id2synset = parse_imagenet_val_labels(args.dataset_path)
  class_idx_db = ImagenetClassIndex.load()
  synset2idx = class_idx_db.synset_to_idx
  print(f"Using {len(image_paths)} images")
  print("Collecting FP32 activations...")
  ResNet.clear_jit_caches()
  fp32_acc = _collect_activations(model, image_paths)

  print("Benchmarking FP32")
  ResNet.clear_jit_caches()
  fp32_top1, fp32_top5 = benchmark_accuracy(model.infer, image_paths, id2synset, synset2idx)
  print(f"FP32: top1={fp32_top1:.2f}  top5={fp32_top5:.2f}")

  input_stats = compute_input_stats(model)
  fp_modules = dict(_weight_modules(model))

  variants_by_bit: dict[int, dict[tuple[str, str], ResNet]] = {}
  for bits in BIT_WIDTHS:
    print(f"\n========== Bit width: {bits} ==========")
    config = PipelineConfig(
      model_depth=model_depth,
      bits=bits,
      dataset_path=args.dataset_path,
      plot_per_channel=args.plot_per_channel,
      output_dir=save_dir / f"{bits}bits",
      n_per_class=args.n_per_class,
    )
    fq_models = {m: copy.deepcopy(model) for m in METHOD_NAMES}

    print(f"=== Stage 1: Weight Distribution Analysis (bits={bits}) ===")
    stage_weight_analysis(config, model, fq_models)

    print(f"=== Stage 2: Bias and Variance Correction (bits={bits}) ===")
    variants: dict[tuple[str, str], ResNet] = {}
    for method in METHOD_NAMES:
      variants[(method, "none")] = fq_models[method]
      if method in PER_CHANNEL_METHODS:
        variants[(method, "bias")] = _bias_correct_model(fq_models[method], fp_modules, input_stats)
        print(f"  applied 'bias' correction to {method}")
    variants_by_bit[bits] = variants

    print(f"=== Stage 3: Quantization Shift Analysis (bits={bits}) ===")
    shift_rows: list[MeanShift] = []
    for (method, mode), m in variants.items():
      label = f"{method}::{mode}"
      print(f"  Collecting {label} stats...")
      ResNet.clear_jit_caches()
      q_acc = _collect_activations(m, image_paths)
      shift_rows.extend(fp32_acc.layers_means_shifts(q_acc, label))
    shifts_csv = config.shift_results_dir / "shifts.csv"
    save_csv(shift_rows, shifts_csv)
    print(f"  CSV saved to {shifts_csv}")
    layer_names = list(fp32_acc.channels_sums.keys())
    minmax_rows = [r for r in shift_rows if r.method.startswith("per_channel_minmax")]
    aciq_rows = [r for r in shift_rows if r.method.startswith("per_channel_aciq")]
    plot_shift(minmax_rows, layer_names, config.shift_results_dir, filename="mean_shift_minmax.png")
    plot_shift(aciq_rows, layer_names, config.shift_results_dir, filename="mean_shift_aciq.png")
    print(f"  Plots saved to {config.shift_results_dir}/")

    print(f"=== Stage 4: Benchmarking (bits={bits}) ===")
    config.correction_results_dir.mkdir(parents=True, exist_ok=True)
    bench_rows: list[BenchmarkRow] = [BenchmarkRow(method="fp32", correction_mode="none", top1=float(fp32_top1), top5=float(fp32_top5))]
    for (method, mode), m in variants.items():
      print(f"  benchmarking {method}::{mode}")
      ResNet.clear_jit_caches()
      top1, top5 = benchmark_accuracy(m.infer, image_paths, id2synset, synset2idx)
      bench_rows.append(BenchmarkRow(method=method, correction_mode=mode, top1=float(top1), top5=float(top5)))
      print(f"  {method}::{mode}: top1={top1:.2f}  top5={top5:.2f}")
    bench_csv = config.correction_results_dir / "benchmark_results.csv"
    save_csv(bench_rows, bench_csv)
    print(f"  CSV saved to {bench_csv}")

  print(f"\n========== Stage 5: CAM cosine analysis ==========")

  gt_indices = np.array([synset2idx[id2synset[p.stem]] for p in image_paths], dtype=np.int64)
  gt_names = [class_idx_db.classes[i].name.replace("_", " ") for i in gt_indices]

  print("Computing FP32 GT-class CAMs")
  ResNet.clear_jit_caches()
  fp32_cams, fp32_preds, fp32_probs = _collect_cams_and_predictions(model, image_paths, gt_indices)

  variant_cams: dict[tuple[int, str, bool], np.ndarray] = {}
  variant_preds: dict[tuple[int, str, bool], np.ndarray] = {}
  variant_probs: dict[tuple[int, str, bool], np.ndarray] = {}

  cosine_rows: list[CamCosineRow] = []
  for bits in BIT_WIDTHS:
    per_bit_rows: list[CamCosineRow] = []
    for (full_method, mode), variant_model in variants_by_bit[bits].items():
      if not full_method.startswith("per_channel_"):
        continue
      method = full_method.removeprefix("per_channel_")
      bias_corrected = mode == "bias"
      print(f"  bits={bits} {full_method}::{mode}")
      ResNet.clear_jit_caches()
      cams, preds, probs = _collect_cams_and_predictions(variant_model, image_paths, gt_indices)
      variant_cams[(bits, method, bias_corrected)] = cams
      variant_preds[(bits, method, bias_corrected)] = preds
      variant_probs[(bits, method, bias_corrected)] = probs
      for i, path in enumerate(image_paths):
        cos = cosine_similarity(fp32_cams[i], cams[i])
        row = CamCosineRow(
          image_id=path.stem,
          gt_class_idx=int(gt_indices[i]),
          gt_class_name=gt_names[i],
          bit_width=bits,
          method=method,
          bias_corrected=bias_corrected,
          cosine=cos,
        )
        per_bit_rows.append(row)
        cosine_rows.append(row)
    save_csv(per_bit_rows, save_dir / f"{bits}bits" / "per_image_cosine.csv")
    print(f"  wrote {save_dir}/{bits}bits/per_image_cosine.csv")

  save_csv(cosine_rows, save_dir / "per_image_cosine.csv")
  print(f"wrote {save_dir}/per_image_cosine.csv")

  summary = _assemble_global_summary(cosine_rows, save_dir)
  save_csv(summary, save_dir / "global_summary.csv")
  print(f"wrote {save_dir}/global_summary.csv")

  plot_cam_cosine_bars(summary, save_dir / "cosine_bar_chart.png")
  print(f"wrote {save_dir}/cosine_bar_chart.png")

  selections = select_fp32_confidence_images(fp32_preds, fp32_probs, gt_indices)
  for sel in selections:
    if sel.index is None:
      print(f"  selection {sel.letter} ({sel.criterion}): empty pool — section will be blank")
    else:
      print(f"  selection {sel.letter} ({sel.criterion}): {image_paths[sel.index].stem} prob={fp32_probs[sel.index]:.3f}")

  write_selected_cams_json(
    save_dir / "selected_cams.json",
    selections,
    image_paths,
    gt_indices,
    gt_names,
    fp32_cams,
    fp32_preds,
    fp32_probs,
    variant_cams,
    variant_preds,
    variant_probs,
    class_idx_db,
  )
  save_selected_image_sidecars(save_dir, selections, image_paths)

  sections_payload: dict[str, dict] = {}
  for sel in selections:
    if sel.index is None:
      sections_payload[sel.letter] = {"criterion": sel.criterion, "image": None, "cams": {}, "predictions": {}, "gt_idx": None, "gt_name": None}
      continue
    i = sel.index
    img = np.asarray(resize_and_center_crop(Image.open(image_paths[i]).convert("RGB")))
    cams_section: dict[str, np.ndarray] = {"fp32": fp32_cams[i]}
    preds_section: dict[str, dict] = {"fp32": {"pred_idx": int(fp32_preds[i]), "pred_name": class_idx_db.classes[int(fp32_preds[i])].name.replace("_", " "), "prob": float(fp32_probs[i])}}
    for (bits, method, bias_corrected), cams in variant_cams.items():
      key = _variant_key(bits, method, bias_corrected)
      cams_section[key] = cams[i]
      preds_section[key] = {
        "pred_idx": int(variant_preds[(bits, method, bias_corrected)][i]),
        "pred_name": class_idx_db.classes[int(variant_preds[(bits, method, bias_corrected)][i])].name.replace("_", " "),
        "prob": float(variant_probs[(bits, method, bias_corrected)][i]),
      }
    sections_payload[sel.letter] = {"criterion": sel.criterion, "image": img, "cams": cams_section, "predictions": preds_section, "gt_idx": int(gt_indices[i]), "gt_name": gt_names[i]}

  plot_qualitative_grid(sections_payload, save_dir / "qualitative_grid.png")
  print(f"wrote {save_dir}/qualitative_grid.png")

  print(f"\nDone. All results in {save_dir}")


if __name__ == "__main__":
  main()
