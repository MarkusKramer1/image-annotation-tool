"""Second-pass visual similarity mining.

Phase 7: uses C-RADIOv4 spatial features to find additional detections
similar to kept cluster prototypes.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


def mine_visual_similarity(
    query_embedding: np.ndarray,
    frames: List[dict],
    embedder,
    threshold: float = 0.8,
) -> List[dict]:
    """Find regions similar to query_embedding across frames."""
    raise NotImplementedError("Phase 7: not yet implemented")
