"""Shared utilities: bbox drawing, image helpers."""
from __future__ import annotations

from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(
    image: Image.Image,
    bboxes: List[List[int]],
    labels: List[str] | None = None,
    color: str = "red",
    line_width: int = 2,
) -> Image.Image:
    """Draw bounding boxes (XYXY) onto a copy of image; return annotated copy."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if labels and i < len(labels):
            draw.text((x1 + 2, y1 + 2), labels[i], fill=color)
    return img
