"""C-RADIOv4-SO400M crop embedder.

Phase 3: wraps nvidia/C-RADIOv4-SO400M for summary embeddings.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


class CRADIOEmbedder:
    """Embed image crops with C-RADIOv4-SO400M."""

    def __init__(self, device: str = "cuda"):
        raise NotImplementedError("Phase 3: not yet implemented")

    def embed_crops(self, image: Image.Image, bboxes: List[List[int]]) -> np.ndarray:
        """Embed bbox crops; returns (N, C) array."""
        raise NotImplementedError("Phase 3: not yet implemented")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed full image; returns (C,) array."""
        raise NotImplementedError("Phase 3: not yet implemented")
