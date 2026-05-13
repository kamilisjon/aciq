# Vendored from tinygrad/extra/models/resnet.py @ commit f28ea84de235cdeaa7e028a2034b34f27b67d30f
import numpy as np
import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit, dtypes, function
from tinygrad.helpers import fetch, get_child
from tinygrad.nn.state import torch_load


from aciq.bias_correction import LayerInputStats
from aciq.distributions import ClippedGaussian
from aciq.nn import fuse_conv_bn_inplace


class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample: list = []
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = [
        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes),
      ]
    self.fused = False

  def __call__(self, x):
    out = self.conv1(x)
    if not self.fused:
      out = self.bn1(out)
    self.activation_1 = out.relu()
    out = self.conv2(self.activation_1)
    if not self.fused:
      out = self.bn2(out)
    self.activation_2 = (out + x.sequential(self.downsample)).relu()
    return self.activation_2

  def fuse(self):
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    fuse_conv_bn_inplace(self.conv2, self.bn2)
    if self.downsample:
      fuse_conv_bn_inplace(self.downsample[0], self.downsample[1])
      self.downsample = [self.downsample[0]]
    self.fused = True


class Bottleneck:
  # This is ResNet v1.5, not original. We load PyTorch weights, so there is no other option.
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)
    self.downsample: list = []
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = [
        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes),
      ]
    self.fused = False

  def __call__(self, x):
    out = self.conv1(x)
    if not self.fused:
      out = self.bn1(out)
    self.activation_1 = out.relu()
    out = self.conv2(self.activation_1)
    if not self.fused:
      out = self.bn2(out)
    self.activation_2 = out.relu()
    out = self.conv3(self.activation_2)
    if not self.fused:
      out = self.bn3(out)
    self.activation_3 = (out + x.sequential(self.downsample)).relu()
    return self.activation_3

  def fuse(self):
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    fuse_conv_bn_inplace(self.conv2, self.bn2)
    fuse_conv_bn_inplace(self.conv3, self.bn3)
    if self.downsample:
      fuse_conv_bn_inplace(self.downsample[0], self.downsample[1])
      self.downsample = [self.downsample[0]]
    self.fused = True


class ResNet:
  def __init__(self, num, num_classes=1000):
    assert num in [18, 34, 50, 101, 152]
    self.num = num
    block_map: dict[int, type[BasicBlock] | type[Bottleneck]] = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
    self.block = block_map[num]

    self.num_blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[num]

    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
    self.fc = nn.Linear(512 * self.block.expansion, num_classes)
    self.fused = False

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return layers

  @function
  def __call__(self, x: Tensor) -> Tensor:
    out = self.conv1(x)
    if not self.fused:
      out = self.bn1(out)
    self.stem_activation = out.relu()
    out = self.stem_activation.pad([1, 1, 1, 1]).max_pool2d((3, 3), 2)
    out = out.sequential(self.layer1)
    out = out.sequential(self.layer2)
    out = out.sequential(self.layer3)
    out = out.sequential(self.layer4)
    out = out.mean([2, 3])
    return self.fc(out.cast(dtypes.float32))

  @TinyJit
  def infer(self, X: Tensor) -> Tensor:
    return self(X).realize()

  @TinyJit
  def get_activations(self, X: Tensor) -> dict[str, Tensor]:
    self(X)
    return {name: act.realize() for name, act in self.activations.items()}

  @classmethod
  def clear_jit_caches(cls) -> None:
    for name in ("infer", "get_activations"):
      cls.__dict__[name].reset()

  def fuse(self):
    fuse_conv_bn_inplace(self.conv1, self.bn1)
    self.fused = True
    for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
      for block in layer:
        block.fuse()

  @property
  def activations(self) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {"stem": self.stem_activation}
    for i, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4), 1):
      for j, block in enumerate(layer):
        out[f"layer{i}.{j}.activation_1"] = block.activation_1
        out[f"layer{i}.{j}.activation_2"] = block.activation_2
        if isinstance(block, Bottleneck):
          out[f"layer{i}.{j}.activation_3"] = block.activation_3
    return out

  def load_from_pretrained(self):
    model_urls = {
      18: "https://download.pytorch.org/models/resnet18-5c106cde.pth",
      34: "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
      50: "https://download.pytorch.org/models/resnet50-19c8e357.pth",
      101: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
      152: "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    }

    self.url = model_urls[self.num]
    for k, dat_t in torch_load(fetch(self.url)).items():
      obj: Tensor = get_child(self, k)
      dat_shape = tuple(dat_t.shape)
      if "fc." in k and tuple(obj.shape) != dat_shape:
        print("skipping fully connected layer")
        continue

      if "bn" not in k and "downsample" not in k:
        assert tuple(obj.shape) == dat_shape, (k, obj.shape, dat_shape)
      obj.assign(Tensor(dat_t.detach().numpy()).to(obj.device).cast(obj.dtype).reshape(obj.shape))


def _bn_effective_params(bn: nn.BatchNorm) -> tuple[np.ndarray, np.ndarray]:
  assert bn.weight is not None and bn.bias is not None, "expected affine BatchNorm"
  gamma_eff = np.abs(bn.weight.numpy().astype(np.float64))
  beta_eff = bn.bias.numpy().astype(np.float64)
  return gamma_eff, beta_eff


def capture_bn_params(model: ResNet) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  out["stem"] = _bn_effective_params(model.bn1)
  for li, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
    for bi, block in enumerate(layer):
      prefix = f"layer{li}.{bi}"
      out[f"{prefix}.conv1"] = _bn_effective_params(block.bn1)
      out[f"{prefix}.conv2"] = _bn_effective_params(block.bn2)
      if isinstance(block, Bottleneck):
        out[f"{prefix}.conv3"] = _bn_effective_params(block.bn3)
      if block.downsample:
        out[f"{prefix}.downsample"] = _bn_effective_params(block.downsample[1])
  return out


def _post_relu_stats_from_bn(beta: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  return ClippedGaussian.mean(beta, gamma), ClippedGaussian.variance(beta, gamma)


def _post_residual_stats(beta_main: np.ndarray, gamma_main: np.ndarray, mu_skip: np.ndarray, var_skip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  mu = beta_main + mu_skip
  var = gamma_main**2 + var_skip
  sigma = np.sqrt(np.maximum(var, 0.0))
  return ClippedGaussian.mean(mu, sigma), ClippedGaussian.variance(mu, sigma)


def compute_input_stats(model: ResNet, bn_params: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, LayerInputStats]:
  out: dict[str, LayerInputStats] = {}

  # Stem input: per-channel zero mean and unit variance, matching the
  # per-channel input standardization that precedes the network (arXiv 1906.04721, p. 7).
  c_in_stem = int(model.conv1.weight.shape[1])
  out["stem"] = LayerInputStats(
    E_x=np.zeros(c_in_stem, dtype=np.float64),
    Var_x=np.ones(c_in_stem, dtype=np.float64),
  )

  # Post-stem ReLU(BN_stem) gives the input to the first block's conv1.
  gamma_stem, beta_stem = bn_params["stem"]
  current_mu, current_var = _post_relu_stats_from_bn(beta_stem, gamma_stem)
  # Spatial size after stem (stride 2 conv + stride 2 maxpool from 224 input) = 56x56.
  # layer1 keeps the same size (stride 1); layer2/3/4 each halve it via the first block's stride-2 conv.
  spatial_hw = 56 * 56

  for li, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
    if li >= 2:
      spatial_hw = max(1, spatial_hw // 4)
    for bi, block in enumerate(layer):
      prefix = f"layer{li}.{bi}"

      # conv1 input = previous block's post-residual activation
      out[f"{prefix}.conv1"] = LayerInputStats(E_x=current_mu.copy(), Var_x=current_var.copy())
      if block.downsample:
        out[f"{prefix}.downsample"] = LayerInputStats(E_x=current_mu.copy(), Var_x=current_var.copy())

      # conv2 input = ReLU(BN of conv1)
      gamma_bn1, beta_bn1 = bn_params[f"{prefix}.conv1"]
      mu_h1, var_h1 = _post_relu_stats_from_bn(beta_bn1, gamma_bn1)
      out[f"{prefix}.conv2"] = LayerInputStats(E_x=mu_h1, Var_x=var_h1)

      if isinstance(block, Bottleneck):
        # conv3 input = ReLU(BN of conv2)
        gamma_bn2, beta_bn2 = bn_params[f"{prefix}.conv2"]
        mu_h2, var_h2 = _post_relu_stats_from_bn(beta_bn2, gamma_bn2)
        out[f"{prefix}.conv3"] = LayerInputStats(E_x=mu_h2, Var_x=var_h2)
        last_bn_key = f"{prefix}.conv3"
      else:
        last_bn_key = f"{prefix}.conv2"

      # Update post-residual stats after this block
      gamma_last, beta_last = bn_params[last_bn_key]
      if block.downsample:
        gamma_ds, beta_ds = bn_params[f"{prefix}.downsample"]
        mu_skip = beta_ds
        var_skip = gamma_ds**2
      else:
        mu_skip = current_mu
        var_skip = current_var
      current_mu, current_var = _post_residual_stats(beta_last, gamma_last, mu_skip, var_skip)

  # FC input = global average pool of layer4 last activation. Mean is preserved;
  # variance shrinks by 1/(H*W) under spatial i.i.d. assumption.
  out["fc"] = LayerInputStats(E_x=current_mu, Var_x=current_var / float(spatial_hw))
  return out
