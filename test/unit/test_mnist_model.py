import unittest

import torch

from aciq.batch_norm import collect_conv_bn_pairs
from aciq.mnist_model import MNISTModel, train_model


class TestMNISTModel(unittest.TestCase):
  def test_model_output_shape(self):
    model = MNISTModel()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    self.assertEqual(out.shape, (4, 10))

  def test_conv_bn_pairs(self):
    model = MNISTModel()
    pairs = collect_conv_bn_pairs(model)
    self.assertEqual(len(pairs), 5)

  @unittest.skip(reason="Takes too long.")
  def test_train_produces_reasonable_accuracy(self):
    model, accuracy = train_model(seed=42, epochs=2, batch_size=512, device="cpu")
    self.assertGreater(accuracy, 0.90)
