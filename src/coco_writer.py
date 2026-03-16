"""COCO JSON dataset builder.

Phase 8: assembles COCO-format annotation file from frames + detections.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import List


def build_coco_dataset(
    frames: List[dict],
    detections: List[dict],
    class_names: List[str],
    output_dir: str,
) -> dict:
    """Build and write a COCO JSON file; return the COCO dict."""
    raise NotImplementedError("Phase 8: not yet implemented")
