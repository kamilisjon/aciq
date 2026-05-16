from enum import StrEnum

import matplotlib.pyplot as plt

from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT


class TailwindColor(StrEnum):
  """Tailwind color pallete. shade: 600. https://tailwindcss.com/docs/colors"""

  SLATE = "#475569"
  GRAY = "#4B5563"
  ZINC = "#52525B"
  NEUTRAL = "#525252"
  STONE = "#57534E"
  RED = "#DC2626"
  ORANGE = "#EA580C"
  AMBER = "#D97706"
  YELLOW = "#CA8A04"
  LIME = "#65A30D"
  GREEN = "#16A34A"
  EMERALD = "#059669"
  TEAL = "#0D9488"
  CYAN = "#0891B2"
  SKY = "#0284C7"
  BLUE = "#2563EB"
  INDIGO = "#4F46E5"
  VIOLET = "#7C3AED"
  PURPLE = "#9333EA"
  FUCHSIA = "#C026D3"
  PINK = "#DB2777"
  ROSE = "#E11D48"


NEUTRAL_COLOR = TailwindColor.SLATE

DIST_COLORS: dict[type[Distribution], TailwindColor] = {
  Gaussian: TailwindColor.PURPLE,
  Laplace: TailwindColor.TEAL,
  StudentT: TailwindColor.AMBER,
  GeneralizedGaussian: TailwindColor.RED,
}

SERIES_COLORS: list[TailwindColor] = [
  TailwindColor.BLUE,
  TailwindColor.EMERALD,
  TailwindColor.AMBER,
  TailwindColor.VIOLET,
  TailwindColor.TEAL,
  TailwindColor.INDIGO,
  TailwindColor.SKY,
  TailwindColor.CYAN,
  TailwindColor.LIME,
  TailwindColor.ORANGE,
]

MAX_PIXEL_DIM = 5000  # cap rendered image's longest side, in pixels
SAVEFIG_DPI = 700  # preferred savefig DPI, capped per-figure via capped_savefig_dpi


def capped_savefig_dpi(fig_w_in: float, fig_h_in: float) -> float:
  """DPI that respects both SAVEFIG_DPI and MAX_PIXEL_DIM for the given figure size in inches."""
  return min(SAVEFIG_DPI, MAX_PIXEL_DIM / max(fig_w_in, fig_h_in))


plt.rcParams.update({
  "savefig.dpi": SAVEFIG_DPI,
  "savefig.bbox": "tight",
  "font.family": "sans-serif",
  "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
  "axes.titlesize": 12,
  "axes.labelsize": 11,
  "xtick.labelsize": 10,
  "ytick.labelsize": 10,
  "legend.fontsize": 10,
  "axes.grid": True,
  "axes.grid.axis": "y",
  "axes.axisbelow": True,
  "grid.color": "#E5E7EB",
  "grid.linewidth": 0.5,
  "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
})


STATS_TEXT_KW: dict[str, object] = {
  "fontsize": 8,
  "va": "top",
  "ha": "right",
  "multialignment": "left",
  "bbox": {"facecolor": "lightgrey"},
}

LINE_WIDTH = 1.1
HIST_BINS = 300
