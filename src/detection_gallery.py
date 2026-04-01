"""detection_gallery.py

Helpers for rendering detection results as a structured gallery.

Bounding-box IDs follow the deterministic scheme ``{img_idx}-{box_idx}``
where ``img_idx`` is the 0-based position of the image in the sorted gallery
and ``box_idx`` is the 0-based position of the annotation within that image.
These IDs are stable across UI re-renders as long as the annotation JSON is
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# ── Palette for drawing (cycles through classes) ─────────────────────────────
_PALETTE = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 130, 255),
    (255, 180, 0),
    (200, 80, 255),
    (0, 210, 210),
]


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Return a usable PIL font, falling back to default if none is available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def load_detection_data(ann_path: Path) -> dict[str, Any]:
    """Load a COCO-format base_detection.json and return the raw dict."""
    with open(ann_path) as fh:
        return json.load(fh)


def build_gallery_entries(
    coco_data: dict[str, Any],
    images_dir: Path,
) -> list[dict[str, Any]]:
    """Build an ordered list of gallery entries, one per image.

    Each entry contains:
        img_idx     -- 0-based gallery index (stable, used for bbox IDs)
        image_id    -- COCO image id
        file_name   -- bare filename
        image_path  -- absolute Path to the image file
        annotations -- list of annotation dicts, each augmented with ``bbox_id``
        width, height

    Images without detections are included with an empty ``annotations`` list.
    """
    # Map image_id -> list of annotations
    ann_by_img: dict[int, list[dict]] = {}
    for ann in coco_data.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Build lookup from COCO images by file_name (may be sparse)
    coco_images = {img["file_name"]: img for img in coco_data.get("images", [])}

    # Use ALL image files from the extracted-frames folder as gallery backbone.
    # This guarantees the gallery shows every frame, not only frames present in
    # the annotation JSON.
    frame_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.name,
    )

    cat_name: dict[int, str] = {
        c["id"]: c["name"] for c in coco_data.get("categories", [])
    }

    entries: list[dict[str, Any]] = []
    for img_idx, frame_path in enumerate(frame_files):
        img_info = coco_images.get(frame_path.name, {})
        image_id = img_info.get("id")
        anns = ann_by_img.get(image_id, []) if image_id is not None else []

        # Assign deterministic bbox IDs within this gallery slot
        annotated_anns = []
        for box_idx, ann in enumerate(anns):
            aug = dict(ann)
            aug["bbox_id"] = f"{img_idx}-{box_idx}"
            aug["category_name"] = cat_name.get(aug.get("category_id"), "?")
            annotated_anns.append(aug)

        entries.append(
            {
                "img_idx": img_idx,
                "image_id": image_id,
                "file_name": frame_path.name,
                "image_path": frame_path,
                "annotations": annotated_anns,
                "width": img_info.get("width"),
                "height": img_info.get("height"),
            }
        )

    return entries


def draw_detections(
    image_path: Path,
    annotations: list[dict[str, Any]],
    img_idx: int,
    highlighted_bbox_ids: set[str] | None = None,
    draw_index_badge: bool = True,
) -> Image.Image:
    """Render bounding boxes and IDs on a PIL image.

    Args:
        image_path: Path to the source image.
        annotations: List of annotation dicts (each must have ``bbox_id``).
        img_idx: Gallery index; painted as a badge in the top-left corner.
        highlighted_bbox_ids: If provided, these bbox_ids are drawn in a
            distinct highlight colour (red outline + thicker border).
        draw_index_badge: Whether to paint the image index badge.

    Returns:
        A PIL Image with annotations drawn.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_small = _get_font(13)
    font_badge = _get_font(16)

    if highlighted_bbox_ids is None:
        highlighted_bbox_ids = set()

    for ann in annotations:
        bbox_id: str = ann["bbox_id"]
        x, y, w, h = ann["bbox"]
        color_idx = ann.get("category_id", 1) % len(_PALETTE)
        color = _PALETTE[color_idx]

        is_highlighted = bbox_id in highlighted_bbox_ids
        outline_color = (255, 40, 40) if is_highlighted else color
        line_width = 3 if is_highlighted else 2

        draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=line_width)

        # Label: class  score  (id)
        cls_name = ann.get("category_name", "")
        label = cls_name if cls_name and cls_name != "?" else bbox_id
        if "score" in ann:
            label += f"  {ann['score']:.2f}"
        if cls_name and cls_name != "?":
            label += f"  ({bbox_id})"

        # Measure text for background rect
        bbox_text = draw.textbbox((0, 0), label, font=font_small)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        pad = 2
        lx, ly = int(x), max(0, int(y) - th - pad * 2)
        draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=outline_color)
        draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255), font=font_small)

    if draw_index_badge:
        badge_text = f"#{img_idx}"
        bbox_badge = draw.textbbox((0, 0), badge_text, font=font_badge)
        bw, bh = bbox_badge[2] - bbox_badge[0], bbox_badge[3] - bbox_badge[1]
        pad = 5
        draw.rectangle([4, 4, 4 + bw + pad * 2, 4 + bh + pad * 2], fill=(20, 20, 20))
        draw.text((4 + pad, 4 + pad), badge_text, fill=(255, 220, 0), font=font_badge)

    return img


_RETRIEVAL_COLOR = (255, 200, 0)  # golden-yellow for proposed / retrieved boxes


def draw_frame_with_proposals(
    image_path: Path,
    existing_annotations: list[dict[str, Any]],
    proposals: list[tuple[int, dict[str, Any]]],
    img_idx: int,
) -> Image.Image:
    """Render a frame showing existing detections and zero or more proposed bboxes.

    Existing annotations are drawn in their normal palette colours.
    Each proposed (retrieved) bbox is drawn in golden-yellow with a ``#N``
    badge so the user can cross-reference cards with the selection list.

    Args:
        image_path: Path to the source frame.
        existing_annotations: Confirmed annotations (must have ``bbox_id`` and
            ``category_id`` as produced by ``build_gallery_entries``).
        proposals: Ordered list of ``(match_index, match_dict)`` pairs.
            ``match_dict`` must contain ``"bbox"`` [x,y,w,h], ``"similarity"``,
            and ``"matched_query"`` with ``"category_name"``.
            Pass an empty list for frames that have no proposed detections.
        img_idx: Gallery index used by the existing-annotation draw call.

    Returns:
        A PIL Image with all annotations and proposals drawn.
    """
    img = draw_detections(image_path, existing_annotations, img_idx, draw_index_badge=False)
    draw = ImageDraw.Draw(img)
    font = _get_font(13)
    font_badge = _get_font(15)

    for match_index, m in proposals:
        x, y, w, h = m["bbox"]
        category_name = m.get("matched_query", {}).get("category_name", "?")
        similarity = m.get("similarity", 0.0)

        draw.rectangle([x, y, x + w, y + h], outline=_RETRIEVAL_COLOR, width=4)

        # Top-left label: "#N  ? class  0.87"
        label = f"#{match_index}  ? {category_name}  {similarity:.2f}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        pad = 2
        lx = int(x)
        ly = max(0, int(y) - th - pad * 2)
        draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=_RETRIEVAL_COLOR)
        draw.text((lx + pad, ly + pad), label, fill=(0, 0, 0), font=font)

        # Top-right corner badge with dark background for visibility
        badge = f"#{match_index}"
        bb = draw.textbbox((0, 0), badge, font=font_badge)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        bpad = 4
        bx = max(0, int(x + w) - bw - bpad * 2)
        by = int(y)
        draw.rectangle([bx, by, bx + bw + bpad * 2, by + bh + bpad * 2], fill=(20, 20, 20))
        draw.text((bx + bpad, by + bpad), badge, fill=_RETRIEVAL_COLOR, font=font_badge)

    return img


def draw_retrieval_match(
    image_path: Path,
    existing_annotations: list[dict[str, Any]],
    proposed_bbox: list[float],
    similarity: float,
    category_name: str,
    img_idx: int,
    match_index: int | None = None,
) -> Image.Image:
    """Convenience wrapper: render one proposed bbox on a frame.

    Delegates to ``draw_frame_with_proposals`` for a single proposal entry.
    """
    proposal_dict: dict[str, Any] = {
        "bbox": proposed_bbox,
        "similarity": similarity,
        "matched_query": {"category_name": category_name},
    }
    proposals = [(match_index if match_index is not None else 0, proposal_dict)]
    return draw_frame_with_proposals(image_path, existing_annotations, proposals, img_idx)


def draw_detections_with_masks(
    image_path: Path,
    annotations: list[dict[str, Any]],
    img_idx: int,
    mask_alpha: int = 90,
) -> Image.Image:
    """Render bounding boxes AND semi-transparent polygon segmentation masks.

    Annotations that have no ``segmentation`` field are drawn with their bbox
    only.  Annotations with a ``segmentation`` field (COCO polygon list) get a
    colour-matched semi-transparent fill underneath the bounding box outline.

    Args:
        image_path: Path to the source image.
        annotations: List of annotation dicts (must have ``bbox_id`` and
            optionally ``segmentation``).
        img_idx: Gallery index; painted as a badge in the top-left corner.
        mask_alpha: Opacity of the polygon fill (0 = invisible, 255 = opaque).

    Returns:
        A PIL Image with bounding boxes and mask overlays drawn.
    """
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for ann in annotations:
        color_idx = ann.get("category_id", 1) % len(_PALETTE)
        r, g, b = _PALETTE[color_idx]
        for poly in ann.get("segmentation", []):
            if len(poly) < 6:
                continue
            points = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            if len(points) >= 3:
                overlay_draw.polygon(points, fill=(r, g, b, mask_alpha))

    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Draw bounding boxes and labels on top of the mask layer
    draw = ImageDraw.Draw(img)
    font_small = _get_font(13)
    font_badge = _get_font(16)

    for ann in annotations:
        bbox_id: str = ann.get("bbox_id", "")
        x, y, w, h = ann["bbox"]
        color_idx = ann.get("category_id", 1) % len(_PALETTE)
        color = _PALETTE[color_idx]
        has_mask = bool(ann.get("segmentation"))

        line_width = 3 if has_mask else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)

        cls_name = ann.get("category_name", "")
        label = cls_name if cls_name and cls_name != "?" else bbox_id
        if "score" in ann:
            label += f"  {ann['score']:.2f}"
        if cls_name and cls_name != "?":
            label += f"  ({bbox_id})"
        if has_mask:
            label += " ✓"

        bbox_text = draw.textbbox((0, 0), label, font=font_small)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        pad = 2
        lx, ly = int(x), max(0, int(y) - th - pad * 2)
        draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=color)
        draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255), font=font_small)

    badge_text = f"#{img_idx}"
    bbox_badge = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw, bh = bbox_badge[2] - bbox_badge[0], bbox_badge[3] - bbox_badge[1]
    pad = 5
    draw.rectangle([4, 4, 4 + bw + pad * 2, 4 + bh + pad * 2], fill=(20, 20, 20))
    draw.text((4 + pad, 4 + pad), badge_text, fill=(255, 220, 0), font=font_badge)

    return img


def crop_bbox(image_path: Path, bbox: list[float], padding: int = 4) -> Image.Image:
    """Return a cropped PIL image for a single bounding box.

    Args:
        image_path: Path to the source image.
        bbox: COCO bbox [x, y, w, h].
        padding: Extra pixels around the crop.

    Returns:
        Cropped PIL Image.
    """
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = bbox
    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(img.width, int(x + w) + padding)
    y2 = min(img.height, int(y + h) + padding)
    return img.crop((x1, y1, x2, y2))
