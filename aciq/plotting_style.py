from enum import StrEnum

import matplotlib.pyplot as plt

from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT


class TailwindColor(StrEnum):
  """Tailwind color pallete. shade: 700. https://tailwindcss.com/docs/colors"""
  SLATE = "#334155"
  GRAY = "#374151"
  ZINC = "#3F3F46"
  NEUTRAL = "#404040"
  STONE = "#44403C"
  RED = "#B91C1C"
  ORANGE = "#C2410C"
  AMBER = "#B45309"
  YELLOW = "#A16207"
  LIME = "#4D7C0F"
  GREEN = "#15803D"
  EMERALD = "#047857"
  TEAL = "#0F766E"
  CYAN = "#0E7490"
  SKY = "#0369A1"
  BLUE = "#1D4ED8"
  INDIGO = "#4338CA"
  VIOLET = "#6D28D9"
  PURPLE = "#7E22CE"
  FUCHSIA = "#A21CAF"
  PINK = "#BE185D"
  ROSE = "#BE123C"

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
  TailwindColor.VIOLET,
  TailwindColor.TEAL,
  TailwindColor.INDIGO,
  TailwindColor.SKY,
  TailwindColor.CYAN,
  TailwindColor.LIME,
  TailwindColor.ORANGE,
]

plt.rcParams.update({
  "savefig.dpi": 700,
  "savefig.bbox": "tight",
  "font.family": "serif",
  "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
  "axes.titlesize": 10,
  "axes.labelsize": 9,
  "xtick.labelsize": 8,
  "ytick.labelsize": 8,
  "legend.fontsize": 8,
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
