import numpy as np


def minmax_alpha(data: np.ndarray) -> float:
  return float(np.max(np.abs(data)))


def quantize(data: np.ndarray, alpha: float, bits: int) -> np.ndarray:
  """Symmetric uniform quantization with range [-alpha, alpha]. Returns dequantized values."""
  qmax = 2 ** (bits - 1) - 1
  scale = alpha / qmax
  quantized = np.clip(np.round(data / scale), -qmax, qmax)
  return quantized * scale
