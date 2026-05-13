import matplotlib.pyplot as plt

from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT


# Tailwind-500 palette
BLUE = "#3B82F6"
EMERALD = "#10B981"
AMBER = "#F59E0B"
PINK = "#EC4899"
VIOLET = "#8B5CF6"
TEAL = "#14B8A6"
ROSE = "#F43F5E"
CYAN = "#06B6D4"
LIME = "#84CC16"
ORANGE = "#F97316"
SLATE = "#64748B"

DistColor: dict[type[Distribution], str] = {
  Gaussian: BLUE,
  Laplace: EMERALD,
  StudentT: AMBER,
  GeneralizedGaussian: PINK,
}

SERIES_COLORS: list[str] = [BLUE, EMERALD, AMBER, PINK, VIOLET, TEAL, ROSE, CYAN, LIME, ORANGE]

NEUTRAL = SLATE


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
