from pathlib import Path

import torch
import torchvision
from tinygrad.helpers import ContextVar


MODELS_DIR = Path("models")
OPSET_VERSION = 18
BATCH_SIZE = ContextVar("BATCH_SIZE", 16)
IMAGE_H_W = ContextVar("IMAGE_H_W", 224)


def main():
  MODELS_DIR.mkdir(exist_ok=True)

  model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
  model.eval()

  dummy_input = (torch.randn(BATCH_SIZE.value, 3, IMAGE_H_W.value, IMAGE_H_W.value),)

  for name, fold in [("not_fused", False), ("fused", True)]:
    save_path = MODELS_DIR / f"resnet50_Opset{OPSET_VERSION}_{name}.onnx"
    torch.onnx.export(model, dummy_input, str(save_path), opset_version=OPSET_VERSION, do_constant_folding=fold)
    print(f"Saved {save_path}")


if __name__ == "__main__":
  main()
