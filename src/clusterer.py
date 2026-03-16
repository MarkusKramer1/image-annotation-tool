"""Embedding clustering and prototype selection.

Phase 4: HDBSCAN / KMeans over crop embeddings.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int = 10,
    min_cluster_size: int = 5,
) -> Tuple[np.ndarray, List[int]]:
    """Cluster embeddings; return (labels, unique_cluster_ids)."""
    raise NotImplementedError("Phase 4: not yet implemented")


def compute_cluster_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    detections: List[dict],
) -> Dict[int, dict]:
    """Return {cluster_id: {mean_embedding, prototype_detection}} for each cluster."""
    raise NotImplementedError("Phase 4: not yet implemented")
