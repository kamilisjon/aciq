from dataclasses import dataclass
from pathlib import Path
import json


_DEFAULT_CLASS_INDEX_PATH = Path(__file__).parent / "imagenet_class_index.json"


@dataclass(frozen=True)
class ImagenetClass:
  idx: int
  synset: str
  name: str


class ImagenetClassIndex:
  def __init__(self, classes: tuple[ImagenetClass, ...]):
    self.classes = classes

  @classmethod
  def load(cls, path: Path = _DEFAULT_CLASS_INDEX_PATH) -> "ImagenetClassIndex":
    with path.open("r") as f:
      raw = json.load(f)
    return cls(tuple(ImagenetClass(idx=int(k), synset=v[0], name=v[1]) for k, v in raw.items()))

  @property
  def synset_to_idx(self) -> dict[str, int]:
    return {c.synset: c.idx for c in self.classes}
