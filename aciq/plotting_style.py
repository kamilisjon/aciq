from aciq.distributions import Distribution, Gaussian, GeneralizedGaussian, Laplace, StudentT


# Tailwind-500 palette — distinct hues, readable on white and dark backgrounds.
DIST_COLORS: dict[type[Distribution], str] = {
  Gaussian: "#3B82F6",
  Laplace: "#10B981",
  StudentT: "#F59E0B",
  GeneralizedGaussian: "#EC4899",
}
