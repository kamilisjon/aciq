# Vendored from tinygrad/extra/models/resnet.py @ commit f28ea84de235cdeaa7e028a2034b34f27b67d30f
import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, get_child
from tinygrad.nn.state import torch_load


from aciq.fusion import fuse_conv_bn


def fuse_inplace(conv: nn.Conv2d, bn: nn.BatchNorm) -> None:
  conv.weight, conv.bias = fuse_conv_bn(conv, bn)


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
    fuse_inplace(self.conv1, self.bn1)
    fuse_inplace(self.conv2, self.bn2)
    if self.downsample:
      fuse_inplace(self.downsample[0], self.downsample[1])
      self.downsample = [self.downsample[0]]
    self.fused = True


class Bottleneck:
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
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
    fuse_inplace(self.conv1, self.bn1)
    fuse_inplace(self.conv2, self.bn2)
    fuse_inplace(self.conv3, self.bn3)
    if self.downsample:
      fuse_inplace(self.downsample[0], self.downsample[1])
      self.downsample = [self.downsample[0]]
    self.fused = True


class ResNet:
  def __init__(self, num, num_classes=1000):
    assert num in [18, 34, 50, 101, 152]
    self.num = num
    self.block = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}[num]

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

  def forward(self, x):
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
    out = self.fc(out.cast(dtypes.float32))
    return out

  def __call__(self, x: Tensor) -> Tensor:
    return self.forward(x)

  def fuse(self):
    fuse_inplace(self.conv1, self.bn1)
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
      18: "https://download.pytorch.org/models/resnet18-f37072fd.pth",
      34: "https://download.pytorch.org/models/resnet34-b627a593.pth",
      50: "https://download.pytorch.org/models/resnet50-0676ba61.pth",
      101: "https://download.pytorch.org/models/resnet101-63fe2227.pth",
      152: "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    }

    self.url = model_urls[self.num]
    for k, dat_t in torch_load(fetch(self.url)).items:
      obj: Tensor = get_child(self, k)
      dat_shape = tuple(dat_t.shape)
      if "fc." in k and tuple(obj.shape) != dat_shape:
        print("skipping fully connected layer")
        continue

      if "bn" not in k and "downsample" not in k:
        assert tuple(obj.shape) == dat_shape, (k, obj.shape, dat_shape)
      obj.assign(Tensor(dat_t.detach().numpy()).to(obj.device).cast(obj.dtype).reshape(obj.shape))
