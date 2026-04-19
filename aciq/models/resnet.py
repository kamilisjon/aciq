# Vendored from tinygrad/extra/models/resnet.py @ commit f28ea84de235cdeaa7e028a2034b34f27b67d30f
import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, get_child

# allow monkeypatching in layer implementations
BatchNorm = nn.BatchNorm2d
Conv2d = nn.Conv2d
Linear = nn.Linear


class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = BatchNorm(planes)
    self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = BatchNorm(planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = [Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), BatchNorm(self.expansion * planes)]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out


class Bottleneck:
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    self.bn1 = BatchNorm(planes)
    self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = BatchNorm(planes)
    self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
    self.bn3 = BatchNorm(self.expansion * planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = [Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), BatchNorm(self.expansion * planes)]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out)).relu()
    out = self.bn3(self.conv3(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out


class ResNet:
  def __init__(self, num, num_classes=None):
    assert num in [18, 34, 50, 101, 152]
    self.num = num
    self.block = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}[num]

    self.num_blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[num]

    self.in_planes = 64

    self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = BatchNorm(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
    self.fc = Linear(512 * self.block.expansion, num_classes) if num_classes is not None else None

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return layers

  def forward(self, x):
    is_feature_only = self.fc is None
    if is_feature_only:
      features = []
    out = self.bn1(self.conv1(x)).relu()
    out = out.pad([1, 1, 1, 1]).max_pool2d((3, 3), 2)
    out = out.sequential(self.layer1)
    if is_feature_only:
      features.append(out)
    out = out.sequential(self.layer2)
    if is_feature_only:
      features.append(out)
    out = out.sequential(self.layer3)
    if is_feature_only:
      features.append(out)
    out = out.sequential(self.layer4)
    if is_feature_only:
      features.append(out)
    if not is_feature_only:
      out = out.mean([2, 3])
      out = self.fc(out.cast(dtypes.float32))
      return out
    return features

  def __call__(self, x: Tensor) -> Tensor:
    return self.forward(x)

  def load_from_pretrained(self):
    import torch  # one-shot torch dep at weight-load time; see header note

    model_urls = {
      18: "https://download.pytorch.org/models/resnet18-f37072fd.pth",
      34: "https://download.pytorch.org/models/resnet34-b627a593.pth",
      50: "https://download.pytorch.org/models/resnet50-0676ba61.pth",
      101: "https://download.pytorch.org/models/resnet101-63fe2227.pth",
      152: "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    }

    self.url = model_urls[self.num]
    state = torch.load(str(fetch(self.url)), map_location="cpu", weights_only=True)
    for k, dat_t in state.items():
      try:
        obj: Tensor = get_child(self, k)
      except AttributeError as e:
        if "fc." in k and self.fc is None:
          continue

        raise e

      dat_shape = tuple(dat_t.shape)
      if "fc." in k and tuple(obj.shape) != dat_shape:
        print("skipping fully connected layer")
        continue  # Skip FC if transfer learning

      if "bn" not in k and "downsample" not in k:
        assert tuple(obj.shape) == dat_shape, (k, obj.shape, dat_shape)
      obj.assign(Tensor(dat_t.detach().numpy()).to(obj.device).cast(obj.dtype).reshape(obj.shape))
