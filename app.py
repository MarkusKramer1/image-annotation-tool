"""Image Annotation Tool — Landing page.

Multi-page Streamlit app for semi-automated dataset annotation:
  1. Data Extraction — video to frames
  2. Base Class Detection — WeDetect open-vocabulary detection
  3. Exact Class Detection — embedding clustering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.common import DATA_DIR, IMAGE_ROOT, discover_datasets, dataset_status, load_extraction_meta
from src.detection_gallery import (
    build_gallery_entries,
    draw_detections,
    draw_detections_with_masks,
    load_detection_data,
)

_PREVIEW_COLS = 6
_PREVIEW_MAX = 6  # max frames shown per dataset on the landing page


@st.cache_data(show_spinner=False)
def _load_overview_gallery(
    ann_path: str, images_dir: str, file_sig: float
) -> list[dict[str, Any]]:
    """Load COCO annotations and build gallery entries (cached by file mtime)."""
    _ = file_sig
    return build_gallery_entries(load_detection_data(Path(ann_path)), Path(images_dir))


def _render_dataset_preview(ds_dir: Path, has_annotations: bool) -> None:
    """Render a small image grid for a dataset.

    Shows annotated frames (with bounding-box / mask overlays) when
    ``base_detection.json`` exists, otherwise shows raw frames.
    """
    frames_dir = ds_dir / IMAGE_ROOT
    ann_path = ds_dir / "annotations" / "base_detection.json"

    if has_annotations and ann_path.exists():
        try:
            file_sig = ann_path.stat().st_mtime
            gallery = _load_overview_gallery(str(ann_path), str(frames_dir), file_sig)
        except Exception:
            gallery = []

        annotated = [e for e in gallery if e["annotations"]]
        entries_to_show = (annotated or gallery)[:_PREVIEW_MAX]

        if not entries_to_show:
            return

        cols = st.columns(_PREVIEW_COLS)
        for i, entry in enumerate(entries_to_show):
            col = cols[i % _PREVIEW_COLS]
            try:
                has_masks = any(ann.get("segmentation") for ann in entry["annotations"])
                if has_masks:
                    img = draw_detections_with_masks(
                        entry["image_path"], entry["annotations"], entry["img_idx"]
                    )
                else:
                    img = draw_detections(
                        entry["image_path"], entry["annotations"], entry["img_idx"]
                    )
                col.image(
                    img,
                    caption=f"#{entry['img_idx']} · {entry['file_name']} · {len(entry['annotations'])} det",
                    use_container_width=True,
                )
            except Exception as exc:
                col.warning(f"Could not render {entry['file_name']}: {exc}")

    else:
        frame_files = sorted(
            [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
            key=lambda p: p.name,
        )[: _PREVIEW_MAX]

        if not frame_files:
            return

        cols = st.columns(_PREVIEW_COLS)
        for i, fp in enumerate(frame_files):
            cols[i % _PREVIEW_COLS].image(str(fp), caption=fp.name, use_container_width=True)


st.set_page_config(
    page_title="Image Annotation Tool",
    page_icon="🏷️",
    layout="wide",
)

st.title("Image Annotation Tool")
st.caption(
    "Semi-automated annotation pipeline: extract frames from video, detect "
    "objects with WeDetect, and refine classes. "
    "Use the sidebar to navigate between pipeline stages."
)
st.divider()

# ─── Pipeline overview ────────────────────────────────────────────────────────

st.subheader("Pipeline")

stages = [
    ("1 — Data Extraction", "Extract frames from a video file."),
    ("2 — Base Class Detection", "Run WeDetect open-vocabulary detection on extracted frames."),
    ("3 — Exact Class Detection", "Refine base classes via embedding clustering."),
]

cols = st.columns(len(stages))
for col, (name, desc) in zip(cols, stages):
    col.markdown(f"**{name}**")
    col.caption(desc)

st.divider()

# ─── Dataset overview ─────────────────────────────────────────────────────────

st.subheader("Datasets")

datasets = discover_datasets()

if not datasets:
    st.info(
        "No datasets found. Go to **Data Extraction** in the sidebar to get started."
    )
else:
    for ds_name in datasets:
        ds_dir = DATA_DIR / ds_name
        status = dataset_status(ds_dir)
        meta = load_extraction_meta(ds_dir)

        badges = []
        if status["extracted"]:
            badges.append("Extracted")
        if status["base_detection"]:
            badges.append("Base Detection")
        if status["exact_detection"]:
            badges.append("Exact Detection")

        badge_str = " · ".join(f"**{b}**" for b in badges) if badges else "*no stages completed*"

        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"### {ds_name}")
            left.caption(badge_str)
            if meta:
                right.metric("Frames", meta.get("num_frames", "?"))
            _render_dataset_preview(ds_dir, has_annotations=status["base_detection"])
