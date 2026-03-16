"""Frame sampler: video or image folder → list of frame dicts.

Each returned dict has keys:
    image      PIL.Image
    source     str  (file path or video filename)
    frame_id   str  (unique identifier)
    timestamp  float | None  (seconds, video only)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def sample_video(video_path: str, fps: float = 1.0) -> List[dict]:
    """Sample frames from a video file at the given fps rate."""
    raise NotImplementedError("Phase 1: not yet implemented")


def sample_folder(folder_path: str, every_nth: int = 1) -> List[dict]:
    """Load every Nth image from a folder, sorted by filename."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )[::every_nth]

    frames = []
    for idx, path in enumerate(paths):
        frames.append(
            {
                "image": Image.open(path).convert("RGB"),
                "source": str(path),
                "frame_id": f"folder_{idx:06d}",
                "timestamp": None,
            }
        )
    return frames
