"""WeDetect detection wrapper.

Phases 2+: wraps mmdetection's init_detector / inference_detector API.
"""
from __future__ import annotations

from typing import List

from PIL import Image


class WeDetectDetector:
    """Wrapper around WeDetect open-vocabulary detector."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        raise NotImplementedError("Phase 2: not yet implemented")

    def detect(
        self,
        image: Image.Image,
        class_names: List[str],
        threshold: float,
    ) -> List[dict]:
        """Return list of {bbox_xyxy, score, class_name, class_id} dicts."""
        raise NotImplementedError("Phase 2: not yet implemented")
