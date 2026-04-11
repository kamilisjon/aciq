from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aciq.analysis import ShiftResult


DEFAULT_COLORS = ["steelblue", "indianred", "seagreen", "darkorange"]


def _plot_shift(
  shift_data: dict[str, ShiftResult],
  layer_names: list[str],
  shift_key: str,
  ylabel: str,
  title: str,
  save_dir: Path,
  filename: str,
  colors: list[str] | None = None,
) -> None:
  save_dir.mkdir(parents=True, exist_ok=True)
  colors = colors or DEFAULT_COLORS

  methods = list(shift_data.keys())
  n_methods = len(methods)
  x_pos = np.arange(len(layer_names))
  bar_width = 0.7 / n_methods
  offsets = np.linspace(-0.35 + bar_width / 2, 0.35 - bar_width / 2, n_methods)

  fig, ax = plt.subplots(figsize=(max(10, len(layer_names) * 1.2), 5))

  for i, method in enumerate(methods):
    shifts = getattr(shift_data[method], shift_key)
    per_layer = [shifts[name] for name in layer_names]

    color = colors[i % len(colors)]
    ax.bar(x_pos + offsets[i], per_layer, width=bar_width, color=color, alpha=0.5, label=method)

  ax.set_xticks(x_pos)
  ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
  ax.set_xlabel("Layer")
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.legend(fontsize=7, prop={"family": "monospace", "size": 7})
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(save_dir / filename, dpi=700)
  plt.close(fig)


def plot_shift(
  shift_data: dict[str, ShiftResult],
  layer_names: list[str],
  save_dir: Path,
  model_name: str = "",
  colors: list[str] | None = None,
) -> None:
  prefix = f"{model_name} — " if model_name else ""
  _plot_shift(
    shift_data,
    layer_names,
    shift_key="mean_shift",
    ylabel="Output mean shift |E[fp32] - E[quant]|",
    title=f"{prefix}Per-layer mean shift",
    save_dir=save_dir,
    filename="mean_shift.png",
    colors=colors,
  )
  _plot_shift(
    shift_data,
    layer_names,
    shift_key="var_shift",
    ylabel="Output variance shift |Var[fp32] - Var[quant]|",
    title=f"{prefix}Per-layer variance shift",
    save_dir=save_dir,
    filename="var_shift.png",
    colors=colors,
  )
