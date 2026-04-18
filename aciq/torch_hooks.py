from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tinygrad.helpers import tqdm

from aciq.analysis import LayerStats, StatsAccumulator
from aciq.benchmark import load_and_preprocess


def get_resnet_block_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
  """Return (name, module) pairs whose forward outputs are the block-final activations.

  Stem ReLU (post conv1/bn1/relu) plus each BasicBlock/Bottleneck — their forward output is the
  post-residual-add post-ReLU tensor, matching the ONNX `/act1/`, `/act2/`, `/act3/` semantics.
  """
  modules: list[tuple[str, nn.Module]] = [("stem", model.get_submodule("relu"))]
  for i in range(1, 5):
    layer = model.get_submodule(f"layer{i}")
    assert isinstance(layer, nn.Sequential)
    for j, block in enumerate(layer):
      modules.append((f"layer{i}.{j}", block))
  return modules


def collect_activations(
  model: nn.Module,
  named_modules: list[tuple[str, nn.Module]],
  image_paths: list[Path],
  batch_size: int = 1,
  device: str = "cpu",
) -> dict[str, LayerStats]:
  """Run inference over image_paths and accumulate per-channel stats for each named module's output."""
  model.eval()
  acc = StatsAccumulator()
  hooks: list[torch.utils.hooks.RemovableHandle] = []

  def make_hook(name: str):
    def hook_fn(_m: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
      acc.update(name, out.detach().cpu().numpy())

    return hook_fn

  for name, module in named_modules:
    hooks.append(module.register_forward_hook(make_hook(name)))

  try:
    with torch.no_grad():
      for start in tqdm(range(0, len(image_paths), batch_size), desc="  activations"):
        batch_paths = image_paths[start : start + batch_size]
        batch = load_and_preprocess(batch_paths).to(device)
        model(batch)
  finally:
    for h in hooks:
      h.remove()

  return acc.finalize()
