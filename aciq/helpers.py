import csv
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints


PATH_DATETIME_FORMAT = "%Y%m%d_%H%M%S"
RESULTS_DIR: Path = Path("results")
CSV_SEPARATOR = ","


def get_output_dir(root: Path, prefix: str = "") -> Path:
  timestamp = datetime.now().strftime(PATH_DATETIME_FORMAT)
  return root / (f"{prefix}_{timestamp}" if prefix else timestamp)


def save_csv(rows: list[Any], path: Path) -> None:
  assert rows, "save_csv requires a non-empty list (dataclass schema is read from rows[0])"
  assert is_dataclass(rows[0]), f"save_csv expects dataclass instances, got {type(rows[0]).__name__}"
  field_names = [f.name for f in fields(rows[0])]
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=field_names, delimiter=CSV_SEPARATOR)
    writer.writeheader()
    for row in rows:
      writer.writerow(asdict(row))


def load_csv(path: Path, row_type: type[Any]) -> list[Any]:
  assert is_dataclass(row_type), f"load_csv expects a dataclass type, got {row_type.__name__}"
  hints = get_type_hints(row_type)
  with open(path) as f:
    reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
    return [row_type(**{k: hints[k](v) for k, v in raw.items()}) for raw in reader]
