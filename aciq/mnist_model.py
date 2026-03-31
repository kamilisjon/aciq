from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tinygrad.helpers import tqdm


class MNISTModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
    self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU())
    self.block3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    self.block4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU())
    self.block5 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(128, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
    x = self.gap(x).flatten(1)
    return self.classifier(x)


def get_mnist_loaders(batch_size: int = 512, data_dir: str = "data") -> tuple[DataLoader[datasets.MNIST], DataLoader[datasets.MNIST]]:
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
  test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
  train_loader: DataLoader[datasets.MNIST] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  test_loader: DataLoader[datasets.MNIST] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
  return train_loader, test_loader


def evaluate_model(model: MNISTModel, test_loader: DataLoader[datasets.MNIST], device: str = "cuda") -> float:
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      preds = model(images).argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)
  return correct / total


def train_model(seed: int, epochs: int = 10, lr: float = 1e-3, batch_size: int = 512, device: str = "cuda") -> tuple[MNISTModel, float]:
  torch.manual_seed(seed)
  if device == "cuda":
    torch.cuda.manual_seed_all(seed)

  model = MNISTModel().to(device)
  train_loader, test_loader = get_mnist_loaders(batch_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  model.train()
  for epoch in range(epochs):
    for images, labels in tqdm(train_loader):
      images, labels = images.to(device), labels.to(device)
      loss = F.cross_entropy(model(images), labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  accuracy = evaluate_model(model, test_loader, device)
  return model, accuracy
