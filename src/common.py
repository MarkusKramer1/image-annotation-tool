"""Shared constants and helpers used across all Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
WEDETECT_DIR = PROJECT_ROOT / "WeDetect"
DATA_DIR = PROJECT_ROOT / "data"

# Path to the few-shot-object-detection package (YOLO-E backend).
# Override via the UI text input if yours lives elsewhere.
FSDET_DIR = Path.home() / "hector" / "src" / "few-shot-object-detection"

TEST_VIDEO_PATH = Path.home() / "Videos" / "athena_driving3.mp4"
TEST_DATASET_NAME = "athena_driving3_test"

DEFAULT_BAG_DIR = Path.home() / "bags"

CONFIG_MAP = {
    "tiny":  ("config/wedetect_tiny.py",  "checkpoints/wedetect_tiny.pth"),
    "base":  ("config/wedetect_base.py",  "checkpoints/wedetect_base.pth"),
    "large": ("config/wedetect_large.py", "checkpoints/wedetect_large.pth"),
}

IMAGE_ROOT = "images/default"


def discover_datasets(data_dir: Path | None = None) -> list[str]:
    """Return sorted dataset names that contain an ``images/default/`` folder."""
    root = data_dir or DATA_DIR
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / IMAGE_ROOT).is_dir()
    )


def load_extraction_meta(dataset_dir: Path) -> dict | None:
    """Read ``annotations/extraction.json`` and return it, or *None*."""
    meta_path = dataset_dir / "annotations" / "extraction.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as fh:
        return json.load(fh)


def dataset_status(dataset_dir: Path) -> dict[str, bool]:
    """Return which pipeline stages have been completed for a dataset."""
    ann = dataset_dir / "annotations"
    return {
        "extracted": (dataset_dir / IMAGE_ROOT).is_dir(),
        "base_detection": (ann / "base_detection.json").exists(),
        "exact_detection": (ann / "exact_detection.json").exists(),
        "segmentation": (ann / "segmentation.json").exists(),
    }
