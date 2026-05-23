import math
import numpy as np

from experiments.resnet_pipeline import cosine_similarity
from experiments.resnet_pipeline import select_fp32_confidence_images


def test_cosine_identical_vectors():
  v = np.array([1.0, 2.0, 3.0])
  assert cosine_similarity(v, v) == 1.0


def test_cosine_orthogonal_vectors():
  a = np.array([1.0, 0.0])
  b = np.array([0.0, 1.0])
  assert cosine_similarity(a, b) == 0.0


def test_cosine_opposite_vectors():
  v = np.array([1.0, 2.0])
  assert math.isclose(cosine_similarity(v, -v), -1.0)


def test_cosine_handles_2d_input():
  a = np.array([[1.0, 0.0], [0.0, 1.0]])
  b = np.array([[1.0, 0.0], [0.0, 1.0]])
  assert math.isclose(cosine_similarity(a, b), 1.0)


def test_cosine_zero_vector_returns_nan():
  a = np.zeros(4)
  b = np.array([1.0, 2.0, 3.0, 4.0])
  assert math.isnan(cosine_similarity(a, b))
  assert math.isnan(cosine_similarity(b, a))


def test_selection_picks_correct_and_incorrect():
  # 4 images. Index 0/1 are correct, 2/3 are incorrect.
  preds = np.array([5, 5, 7, 7])
  probs = np.array([0.99, 0.40, 0.95, 0.30])
  gt = np.array([5, 5, 5, 5])
  sels = {s.letter: s for s in select_fp32_confidence_images(preds, probs, gt)}
  assert sels["A"].index == 0  # most confident correct: prob 0.99 at idx 0
  assert sels["B"].index == 2  # most confident incorrect: prob 0.95 at idx 2
  assert sels["C"].index == 1  # least confident correct: prob 0.40 at idx 1
  assert sels["D"].index == 3  # least confident incorrect: prob 0.30 at idx 3


def test_selection_handles_empty_set():
  # All correct -> B and D are empty.
  preds = np.array([5, 5])
  probs = np.array([0.9, 0.8])
  gt = np.array([5, 5])
  sels = {s.letter: s for s in select_fp32_confidence_images(preds, probs, gt)}
  assert sels["A"].index == 0
  assert sels["B"].index is None
  assert sels["C"].index == 1
  assert sels["D"].index is None
