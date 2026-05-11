from datetime import datetime
from pathlib import Path


PATH_DATETIME_FORMAT = "%Y%m%d_%H%M%S"
RESULTS_DIR: Path = Path("results")


def get_output_dir(root: Path, prefix: str = "") -> Path:
  timestamp = datetime.now().strftime(PATH_DATETIME_FORMAT)
  return root / (f"{prefix}_{timestamp}" if prefix else timestamp)
