from enum import StrEnum

import matplotlib.pyplot as plt

from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT


class TailwindColor(StrEnum):
  SLATE = "#64748B"
  GRAY = "#6B7280"
  ZINC = "#71717A"
  NEUTRAL = "#737373"
  STONE = "#78716C"
  RED = "#EF4444"
  ORANGE = "#F97316"
  AMBER = "#F59E0B"
  YELLOW = "#EAB308"
  LIME = "#84CC16"
  GREEN = "#22C55E"
  EMERALD = "#10B981"
  TEAL = "#14B8A6"
  CYAN = "#06B6D4"
  SKY = "#0EA5E9"
  BLUE = "#3B82F6"
  INDIGO = "#6366F1"
  VIOLET = "#8B5CF6"
  PURPLE = "#A855F7"
  FUCHSIA = "#D946EF"
  PINK = "#EC4899"
  ROSE = "#F43F5E"

NEUTRAL_COLOR = TailwindColor.SLATE

DIST_COLORS: dict[type[Distribution], TailwindColor] = {
  Gaussian: TailwindColor.BLUE,
  Laplace: TailwindColor.EMERALD,
  StudentT: TailwindColor.AMBER,
  GeneralizedGaussian: TailwindColor.PINK,
}

SERIES_COLORS: list[TailwindColor] = [
  TailwindColor.BLUE,
  TailwindColor.EMERALD,
  TailwindColor.AMBER,
  TailwindColor.PINK,
  TailwindColor.VIOLET,
  TailwindColor.TEAL,
  TailwindColor.ROSE,
  TailwindColor.CYAN,
  TailwindColor.LIME,
  TailwindColor.ORANGE,
]

plt.rcParams.update({
  "savefig.dpi": 700,
  "savefig.bbox": "tight",
  "axes.titlesize": 10,
  "axes.labelsize": 9,
  "xtick.labelsize": 8,
  "ytick.labelsize": 8,
  "legend.fontsize": 8,
  "axes.grid": True,
  "grid.alpha": 0.3,
  "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
})


MONOSPACE_LEGEND_KW: dict[str, object] = {"prop": {"family": "monospace"}}

STATS_TEXT_KW: dict[str, object] = {
  "fontsize": 8,
  "va": "top",
  "ha": "right",
  "multialignment": "left",
  "bbox": {"facecolor": "lightgrey"},
  "family": "monospace",
}
