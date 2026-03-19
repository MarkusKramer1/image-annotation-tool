"""similarity_search.py

Visual similarity search using real DINOv2 embeddings.

The module lazily loads Facebook's DINOv2 ViT-B/14 model via torch.hub on first
use and caches it for subsequent requests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Image preprocessing (DINOv2 ViT-B/14, ImageNet-style normalization) ─────
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model: torch.nn.Module | None = None
_device: torch.device | None = None


def _load_model() -> tuple[torch.nn.Module, torch.device]:
    """Load and cache DINOv2 ViT-B/14 from torch.hub."""
    global _model, _device
    if _model is not None and _device is not None:
        return _model, _device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Downloads once to torch hub cache, then reuses locally.
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        pretrained=True,
        trust_repo=True,
    )
    model.eval()
    model = model.to(device)

    _model = model
    _device = device
    return model, device


@torch.no_grad()
def embed_crops(
    crops: list[Image.Image],
    batch_size: int = 32,
) -> np.ndarray:
    """Compute L2-normalised embeddings for a list of PIL image crops.

    Args:
        crops: List of PIL Images (any size; will be resized).
        batch_size: Images to process per GPU batch.

    Returns:
        Float32 numpy array of shape (N, 768).
    """
    model, device = _load_model()

    tensors = [_TRANSFORM(c.convert("RGB")) for c in crops]
    embeddings: list[np.ndarray] = []

    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device)
        feats = model(batch)  # (B, 768)
        feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def find_similar(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_meta: list[dict[str, Any]],
    top_k: int = 10,
    min_similarity: float = 0.70,
) -> list[dict[str, Any]]:
    """Find the most visually similar candidates for a set of query embeddings.

    Similarity is computed as the maximum cosine similarity across all queries
    (i.e. any query matching a candidate is surfaced).

    Args:
        query_embeddings: (Q, D) float32 array — the wrong/seed crops.
        candidate_embeddings: (C, D) float32 array — all annotation crops.
        candidate_meta: List of dicts with metadata for each candidate;
            each must have at least ``bbox_id``.
        top_k: Maximum number of results to return.
        min_similarity: Minimum cosine similarity threshold.

    Returns:
        List of candidate_meta dicts sorted descending by similarity,
        each augmented with ``similarity`` (float).
    """
    if query_embeddings.shape[0] == 0 or candidate_embeddings.shape[0] == 0:
        return []

    # (Q, C) cosine similarity matrix — both arrays are already L2-normalised
    sim_matrix = query_embeddings @ candidate_embeddings.T  # (Q, C)
    max_sim = sim_matrix.max(axis=0)  # (C,) — best match across all queries

    results = []
    for idx, (sim, meta) in enumerate(zip(max_sim, candidate_meta)):
        if float(sim) >= min_similarity:
            results.append({**meta, "similarity": float(sim)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
