import contextlib
import os
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


def getenv(key: str, default: T) -> T:
  return type(default)(os.environ[key]) if key in os.environ else default


class ContextVar(Generic[T]):
  _cache: ClassVar[dict[str, "ContextVar"]] = {}
  value: T
  key: str
  def __init__(self, key: str, default_value: T):
    if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __eq__(self, x): return self.value == x
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x
  def tolist(self, obj=None):
    assert isinstance(self.value, str)
    return [getattr(obj, x) if obj else x for x in self.value.split(',') if x]


class Context(contextlib.ContextDecorator):
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    self.old_context: dict[str, Any] = {k: ContextVar._cache[k].value for k in self.kwargs}
    for k, v in self.kwargs.items(): ContextVar._cache[k].value = v
  def __exit__(self, *args):
    for k, v in self.old_context.items(): ContextVar._cache[k].value = v


DEBUG = ContextVar("DEBUG", 0)
