"""Page 2 — Base Class Detection: run WeDetect on extracted frames."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.common import (
    CONFIG_MAP,
    DATA_DIR,
    FSDET_DIR,
    IMAGE_ROOT,
    PROJECT_ROOT,
    WEDETECT_DIR,
    discover_datasets,
    load_extraction_meta,
)
from src.detection_gallery import (
    build_gallery_entries,
    crop_bbox,
    draw_detections,
    draw_detections_with_masks,
    draw_frame_with_proposals,
    load_detection_data,
)
from src.similarity_search import embed_crops, find_similar

try:
    from streamlit_drawable_canvas import st_canvas
    _SDC_AVAILABLE = True
except ImportError:
    _SDC_AVAILABLE = False

# ── WeDetect-Uni checkpoint config ────────────────────────────────────────────
UNI_CHECKPOINT_NAMES = {
    "base":  "checkpoints/wedetect_base_uni.pth",
    "large": "checkpoints/wedetect_large_uni.pth",
}

st.set_page_config(page_title="Base Class Detection", page_icon="🔍", layout="wide")

if "gallery_ready_dataset" not in st.session_state:
    st.session_state["gallery_ready_dataset"] = None


# ─── Gallery helpers (defined early so _remove_annotations can reference them) ─

@st.cache_data(show_spinner=False)
def _load_frame_display(image_path: str, display_w: int) -> tuple[bytes, int, int]:
    """Load a frame, resize to display_w, and return (jpeg_bytes, orig_w, orig_h).

    Cached so repeated reruns (e.g. while drawing a bbox) skip the disk I/O and
    resize completely.  Returns JPEG bytes so the caller can draw overlays on a
    decoded copy without re-encoding the base image each time.
    """
    import io as _io
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    display_h = max(1, round(orig_h * display_w / orig_w))
    img_small = img.resize((display_w, display_h), Image.LANCZOS)
    buf = _io.BytesIO()
    img_small.save(buf, format="JPEG", quality=80)
    return buf.getvalue(), orig_w, orig_h


@st.cache_data(show_spinner=False)
def _load_gallery(ann_path: str, images_dir: str, file_sig: float) -> list[dict[str, Any]]:
    """Load and build gallery entries.

    file_sig is typically the annotation file mtime and is included only to bust
    Streamlit cache whenever base_detection.json changes.
    """
    _ = file_sig
    coco = load_detection_data(Path(ann_path))
    return build_gallery_entries(coco, Path(images_dir))


GALLERY_COLS = 3
CROP_COLS = 6
_THUMB_HEIGHT = 96


def _render_gallery(
    entries: list[dict[str, Any]],
    highlighted_bbox_ids: set[str] | None = None,
    container_height: int = 600,
) -> None:
    """Render a gallery of annotated images in a scrollable container."""
    with st.container(height=container_height):
        render_cols = st.columns(GALLERY_COLS)
        for i, entry in enumerate(entries):
            col = render_cols[i % GALLERY_COLS]
            try:
                rendered = draw_detections(
                    entry["image_path"],
                    entry["annotations"],
                    entry["img_idx"],
                    highlighted_bbox_ids=highlighted_bbox_ids,
                )
                ann_count = len(entry["annotations"])
                col.image(
                    rendered,
                    caption=f"#{entry['img_idx']} · {entry['file_name']} · {ann_count} det",
                    use_container_width=True,
                )
            except Exception as exc:
                col.warning(f"Could not render {entry['file_name']}: {exc}")


def _collect_crops_by_class(
    entries: list[dict[str, Any]],
    only_ids: set[str] | None = None,
    thumb_height: int = _THUMB_HEIGHT,
    id_captions: dict[str, str] | None = None,
) -> dict[str, list[tuple[Image.Image, str]]]:
    """Return a dict mapping class name -> list of (crop_image, caption).

    All crops are resized to a uniform height so the grid looks consistent.
    """
    by_class: dict[str, list[tuple[Image.Image, str]]] = {}
    for entry in entries:
        for ann in entry["annotations"]:
            bid = ann["bbox_id"]
            if only_ids is not None and bid not in only_ids:
                continue
            cls = ann.get("category_name", "?")
            if id_captions and bid in id_captions:
                caption = id_captions[bid]
            else:
                score = ann.get("score")
                caption = bid if score is None else f"{bid}  {score:.2f}"
            try:
                crop = crop_bbox(entry["image_path"], ann["bbox"])
                w, h = crop.size
                if h > 0:
                    new_w = max(1, round(w * thumb_height / h))
                    crop = crop.resize((new_w, thumb_height), Image.LANCZOS)
                by_class.setdefault(cls, []).append((crop, caption))
            except Exception:
                pass
    return by_class


def _render_crops_by_class(
    entries: list[dict[str, Any]],
    only_ids: set[str] | None = None,
    container_height: int = 500,
    cols: int = CROP_COLS,
    id_captions: dict[str, str] | None = None,
) -> None:
    """Render bbox crops grouped by class in a scrollable container."""
    by_class = _collect_crops_by_class(entries, only_ids=only_ids, id_captions=id_captions)
    if not by_class:
        st.info("No detections to display.")
        return
    with st.container(height=container_height):
        for cls_name, items in sorted(by_class.items()):
            st.markdown(f"**{cls_name}** — {len(items)} detection(s)")
            grid = st.columns(cols)
            for i, (crop_img, caption) in enumerate(items):
                grid[i % cols].image(crop_img, caption=caption)


def _remove_annotations(ann_path: Path, ann_ids: set[int]) -> None:
    """Remove annotations by ID from the COCO JSON file and refresh UI state."""
    if not ann_ids:
        st.warning("No annotation IDs to remove.")
        return

    try:
        with open(ann_path) as fh:
            coco_data = json.load(fh)

        original_count = len(coco_data.get("annotations", []))
        coco_data["annotations"] = [
            a for a in coco_data["annotations"] if a["id"] not in ann_ids
        ]
        removed = original_count - len(coco_data["annotations"])

        with open(ann_path, "w") as fh:
            json.dump(coco_data, fh, indent=2)

        _load_gallery.clear()
        st.session_state.pop("selected_wrong_bboxes", None)
        st.session_state.pop("similarity_matches", None)
        st.session_state.pop("similarity_wrong_ids", None)

        st.success(f"Removed {removed} annotation(s). Refreshing…")
        st.rerun()

    except Exception as exc:
        st.error(f"Failed to update annotation file: {exc}")


# ─── Page header ──────────────────────────────────────────────────────────────

st.title("Base Class Detection")
st.caption(
    "Select a dataset with extracted frames and run WeDetect open-vocabulary "
    "detection to produce COCO-format base-class annotations."
)
st.divider()

# ─── Dataset selector ─────────────────────────────────────────────────────────

st.subheader("Dataset")

datasets = discover_datasets()

if not datasets:
    st.warning(
        "No datasets with extracted frames found in `data/`. "
        "Go to **Data Extraction** first."
    )
    st.stop()

selected_dataset = st.selectbox("Dataset", options=datasets)

dataset_dir = DATA_DIR / selected_dataset
frames_dir = dataset_dir / IMAGE_ROOT
output_json_path = dataset_dir / "annotations" / "base_detection.json"

_rm_col, _ = st.columns([1, 3])
with _rm_col:
    if st.button(
        "Remove Annotations",
        disabled=not output_json_path.exists(),
        help=f"Permanently delete all annotations for `{selected_dataset}`.",
        use_container_width=True,
    ):
        try:
            output_json_path.unlink()
            _load_gallery.clear()
            st.session_state.pop("gallery_ready_dataset", None)
            st.rerun()
        except Exception as _exc:
            st.error(f"Failed to remove annotations: {_exc}")

meta = load_extraction_meta(dataset_dir)
if meta:
    _source_bags = meta.get("source_bags")
    if _source_bags:
        _source_str = ", ".join(f"`{b}`" for b in _source_bags)
        _filters = meta.get("filters", {})
        _fps = _filters.get("sample_fps", "all")
        _step_str = f"topic `{meta.get('extraction_topic', '?')}` · {_fps} fps"
    else:
        _source_str = f"`{meta.get('source_video', '?')}`"
        _step_str = f"step {meta.get('frame_step', '?')}"
    st.caption(
        f"Source: {_source_str} · "
        f"{meta.get('num_frames', '?')} frames · "
        f"{_step_str}"
    )

# ─── Annotation overview ──────────────────────────────────────────────────────

st.divider()
st.subheader("Current Annotations")

_overview_gallery: list[dict[str, Any]] | None = None
_existing_classes: list[str] = []

if output_json_path.exists():
    try:
        _ov_sig = output_json_path.stat().st_mtime
        _overview_gallery = _load_gallery(str(output_json_path), str(frames_dir), _ov_sig)
        with open(output_json_path) as _fh:
            _ov_coco = json.load(_fh)
        _existing_classes = [c["name"] for c in _ov_coco.get("categories", [])]
    except Exception:
        _overview_gallery = None

def _render_plain_frames(images_dir: Path, container_height: int = 480) -> None:
    """Show all raw frames from images_dir without any annotation overlay."""
    _frame_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.name,
    )
    if not _frame_files:
        st.info("No frames found in this dataset.")
        return
    st.caption(f"{len(_frame_files)} frames · no annotations yet")
    with st.container(height=container_height):
        _plain_cols = st.columns(GALLERY_COLS)
        for _fi, _fp in enumerate(_frame_files):
            _plain_cols[_fi % GALLERY_COLS].image(
                str(_fp), caption=_fp.name, use_container_width=True
            )


if _overview_gallery:
    _ov_annotated = [e for e in _overview_gallery if e["annotations"]]
    _ov_total_dets = sum(len(e["annotations"]) for e in _overview_gallery)
    _ov_with_masks = sum(
        1 for e in _overview_gallery
        if any(ann.get("segmentation") for ann in e["annotations"])
    )
    if _ov_annotated:
        st.caption(
            f"{len(_overview_gallery)} frames · {len(_ov_annotated)} with detections · "
            f"{_ov_total_dets} bounding boxes"
            + (f" · {_ov_with_masks} frame(s) with segmentation masks" if _ov_with_masks else "")
        )
        with st.container(height=480):
            _ov_cols = st.columns(GALLERY_COLS)
            for _i, _entry in enumerate(_ov_annotated):
                _col = _ov_cols[_i % GALLERY_COLS]
                try:
                    _has_masks = any(ann.get("segmentation") for ann in _entry["annotations"])
                    if _has_masks:
                        _img = draw_detections_with_masks(
                            _entry["image_path"], _entry["annotations"], _entry["img_idx"]
                        )
                    else:
                        _img = draw_detections(
                            _entry["image_path"], _entry["annotations"], _entry["img_idx"]
                        )
                    _col.image(
                        _img,
                        caption=(
                            f"#{_entry['img_idx']} · {_entry['file_name']} · "
                            f"{len(_entry['annotations'])} det"
                        ),
                        use_container_width=True,
                    )
                except Exception as _exc:
                    _col.warning(f"Could not render {_entry['file_name']}: {_exc}")
    else:
        _render_plain_frames(frames_dir)
else:
    _render_plain_frames(frames_dir)

st.divider()

# ─── 1. Object Detection ──────────────────────────────────────────────────────

with st.expander("1. Object Detection", expanded=True):

    # ── Mode selector ─────────────────────────────────────────────────────────
    detect_mode = st.radio(
        "Detection mode",
        options=["Text prompts", "Visual prompts"],
        horizontal=True,
        key="detect_mode",
        help=(
            "**Text prompts** — describe what to find with class names (e.g. *robot, screw*). "
            "Uses WeDetect open-vocabulary detection.  \n"
            "**Visual prompts** — draw example bounding boxes on frames to show what to find. "
            "Uses WeDetect-Uni visual similarity matching."
        ),
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # TEXT PROMPT MODE
    # ══════════════════════════════════════════════════════════════════════════
    if detect_mode == "Text prompts":

        col_left, col_right = st.columns(2)

        with col_left:
            model_size = st.radio(
                "Model size",
                options=["tiny", "base", "large"],
                index=1,
                horizontal=True,
            )
            st.caption(
                f"Config: `{CONFIG_MAP[model_size][0]}`  |  "
                f"Checkpoint: `{CONFIG_MAP[model_size][1]}`"
            )

        with col_right:
            adaptive_threshold = st.checkbox(
                "Adaptive per-class threshold",
                value=True,
                help=(
                    "After the global confidence threshold, keep only detections that "
                    "score within a ratio of the best detection for their class."
                ),
            )
            threshold = st.slider(
                "Confidence threshold",
                min_value=0.01,
                max_value=0.90,
                value=0.01 if adaptive_threshold else 0.30,
                step=0.01,
            )
            adaptive_threshold_ratio = st.slider(
                "Adaptive threshold ratio",
                min_value=0.50,
                max_value=1.00,
                value=0.90,
                step=0.10,
                disabled=not adaptive_threshold,
            )
            topk = st.number_input(
                "Max detections per frame",
                min_value=1,
                max_value=500,
                value=100,
                step=10,
            )

        if _existing_classes:
            _default_classes = ", ".join(_existing_classes)
        else:
            _default_classes = ""

        classes_input = st.text_input(
            "Object classes (comma-separated)",
            value=_default_classes,
            placeholder="screw,nut,flange",
        )
        if classes_input.strip():
            parsed = [c.strip() for c in classes_input.split(",") if c.strip()]
            st.caption(f"{len(parsed)} class(es): {', '.join(f'`{c}`' for c in parsed)}")

        st.divider()

        if not WEDETECT_DIR.exists():
            st.error(
                f"WeDetect directory not found at `{WEDETECT_DIR}`. "
                "Make sure the WeDetect/ folder is present in the project root."
            )
            st.stop()

        checkpoint_path = WEDETECT_DIR / CONFIG_MAP[model_size][1]
        if not checkpoint_path.exists():
            st.error(
                f"Checkpoint `{checkpoint_path.name}` not found. "
                "Download the model weights into `WeDetect/checkpoints/`."
            )
            st.stop()

        can_run = bool(classes_input.strip())

        run_clicked = st.button(
            "Run Detection",
            type="primary",
            disabled=not can_run,
        )

        if not can_run and not run_clicked:
            st.info("Please provide at least one class name to continue.")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUAL PROMPT MODE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.caption(
            "Draw bounding boxes on example frames to define *what* to detect. "
            "The selected backend will search for visually similar objects across all frames."
        )

        # ── Backend selector ──────────────────────────────────────────────────
        _vp_backend = st.radio(
            "Detection backend",
            options=["WeDetect-Uni", "YOLO-E"],
            horizontal=True,
            key="vp_backend",
            help=(
                "**WeDetect-Uni** — embedding-based proposal matching (cosine similarity).  \n"
                "**YOLO-E** — few-shot segmentation model; directly runs YOLO-E inference "
                "with visual prototype embeddings."
            ),
        )

        if not _SDC_AVAILABLE:
            st.error(
                "`streamlit-drawable-canvas` is not installed. "
                "Run: `pip install streamlit-drawable-canvas` in the project environment."
            )

        # ── Frame selector ────────────────────────────────────────────────────
        _vp_frame_files = sorted(
            [p for p in frames_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
            key=lambda p: p.name,
        )
        _vp_frame_names = [f.name for f in _vp_frame_files]

        if not _vp_frame_files:
            st.warning("No frames found in this dataset yet.")
        else:
            _vp_fc1, _vp_fc2 = st.columns([2, 1])
            with _vp_fc1:
                _sel_frame_name = st.selectbox(
                    "Select reference frame",
                    options=_vp_frame_names,
                    key="vp_frame_select",
                )
            with _vp_fc2:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                _canvas_mode = st.radio(
                    "Canvas mode",
                    options=["Draw box", "Move / resize box"],
                    horizontal=False,
                    key="vp_canvas_mode",
                    help=(
                        "**Draw box** — drag to create a new bounding box.  \n"
                        "**Move / resize box** — select and drag handles to adjust the existing box."
                    ),
                )

            _sel_frame_path = frames_dir / _sel_frame_name

            # Fixed display width — all coordinates are in this pixel space.
            _vp_display_w = 700

            # Cached load+resize: disk I/O + resize only on first view of each frame.
            _vp_jpeg_bytes, _vp_iw, _vp_ih = _load_frame_display(
                str(_sel_frame_path), _vp_display_w
            )

            # Increment the canvas key when the user switches frames so it resets.
            if st.session_state.get("vp_coords_frame") != _sel_frame_name:
                st.session_state.vp_img_key = st.session_state.get("vp_img_key", 0) + 1
                st.session_state["vp_coords_frame"] = _sel_frame_name

            if "vp_img_key" not in st.session_state:
                st.session_state.vp_img_key = 0

            # ── Drawable canvas widget ────────────────────────────────────────
            # update_streamlit=False prevents reruns while drawing.  The canvas
            # preserves its fabric.js state across Streamlit reruns (same key),
            # so drawn boxes stay visible and json_data is returned on the next
            # natural rerun (button click, text input, etc.).
            # Changing drawing_mode does NOT clear the canvas (only a key change does).
            import io as _io2
            _canvas_bg = Image.open(_io2.BytesIO(_vp_jpeg_bytes)).convert("RGB")
            _canvas_h = _canvas_bg.height

            _canvas_draw_mode = (
                "rect" if _canvas_mode == "Draw box" else "transform"
            )

            _clear_col, _ = st.columns([1, 4])
            with _clear_col:
                if st.button("Clear box", key="vp_clear_canvas", help="Remove the drawn box and start over."):
                    st.session_state.vp_img_key += 1
                    st.rerun()

            if _SDC_AVAILABLE:
                _canvas_result = st_canvas(
                    fill_color="rgba(255, 50, 50, 0.15)",
                    stroke_width=3,
                    stroke_color="#FF3232",
                    background_image=_canvas_bg,
                    update_streamlit=True,
                    height=_canvas_h,
                    width=_vp_display_w,
                    drawing_mode=_canvas_draw_mode,
                    key=f"vp_canvas_{st.session_state.vp_img_key}",
                )
            else:
                _canvas_result = None

            # Derive _canvas_objects from whatever the canvas reports on this rerun.
            _canvas_objects: list = []
            if _canvas_result is not None and _canvas_result.json_data is not None:
                _canvas_objects = _canvas_result.json_data.get("objects", [])

            _has_boxes = bool(_canvas_objects)
            # Use the first drawn box just to keep the existing disabled-check variable name.
            _last_drag = _canvas_objects[0] if _has_boxes else None

            if _has_boxes:
                _n_boxes = len(_canvas_objects)
                st.caption(
                    f"{_n_boxes} box{'es' if _n_boxes > 1 else ''} drawn · "
                    "switch to **Move / resize** mode to adjust · draw more or click **Add**."
                )
            else:
                st.caption("Draw bounding boxes on the image above, then enter a label and click **Add**.")

            # ── Label + Add button ────────────────────────────────────────────
            _add_col1, _add_col2 = st.columns([4, 1])
            with _add_col1:
                _vp_label = st.text_input(
                    "Label for this prompt",
                    key="vp_label_input",
                    placeholder="e.g. robot",
                )
            with _add_col2:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                _add_vp_clicked = st.button(
                    "Add",
                    key="vp_add_btn",
                    use_container_width=True,
                    disabled=not (_SDC_AVAILABLE and _last_drag is not None and bool(_vp_label.strip())),
                )

            if _add_vp_clicked:
                # Re-read canvas objects on this rerun (triggered by the button click).
                _add_objects = (
                    _canvas_result.json_data.get("objects", [])
                    if _canvas_result is not None and _canvas_result.json_data is not None
                    else []
                )
                if not _vp_label.strip():
                    st.warning("Please enter a label before adding.")
                elif not _add_objects:
                    st.warning("Please draw at least one bounding box on the canvas first.")
                else:
                    _ascale = _vp_iw / _vp_display_w
                    if "visual_prompts" not in st.session_state:
                        st.session_state.visual_prompts = []

                    for _obj in _add_objects:
                        _rx = _obj.get("left", 0)
                        _ry = _obj.get("top", 0)
                        _rw = abs(_obj.get("width", 0) * _obj.get("scaleX", 1))
                        _rh = abs(_obj.get("height", 0) * _obj.get("scaleY", 1))
                        st.session_state.visual_prompts.append({
                            "image_name": _sel_frame_name,
                            "image_path": str(_sel_frame_path),
                            "bbox":       [
                                round(_rx * _ascale),
                                round(_ry * _ascale),
                                max(1, round(_rw * _ascale)),
                                max(1, round(_rh * _ascale)),
                            ],
                            "label":      _vp_label.strip(),
                        })
                    # Reset canvas for the next round of drawing.
                    st.session_state.vp_img_key += 1
                    st.rerun()

        # ── Visual prompts list ───────────────────────────────────────────────
        if "visual_prompts" not in st.session_state:
            st.session_state.visual_prompts = []

        _vp_list: list[dict] = st.session_state.visual_prompts
        if _vp_list:
            st.markdown(f"**{len(_vp_list)} visual prompt(s)**")
            _del_idx: int | None = None
            for _vi, _vp in enumerate(_vp_list):
                _vp_c1, _vp_c2, _vp_c3 = st.columns([1, 5, 1])
                with _vp_c1:
                    try:
                        _vp_img = Image.open(_vp["image_path"]).convert("RGB")
                        _vx, _vy, _vw, _vh = _vp["bbox"]
                        _vp_crop = _vp_img.crop((
                            max(0, _vx - 4), max(0, _vy - 4),
                            min(_vp_img.width,  _vx + _vw + 4),
                            min(_vp_img.height, _vy + _vh + 4),
                        ))
                        _vp_crop.thumbnail((80, 80), Image.LANCZOS)
                        _vp_buf = io.BytesIO()
                        _vp_crop.save(_vp_buf, format="PNG")
                        _vp_b64 = base64.b64encode(_vp_buf.getvalue()).decode()
                        st.markdown(
                            f'<img src="data:image/png;base64,{_vp_b64}" width="80"/>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        st.markdown("⬛")
                with _vp_c2:
                    st.markdown(
                        f"**{_vp['label']}** &nbsp;·&nbsp; `{_vp['image_name']}` &nbsp;·&nbsp; "
                        f"bbox `{_vp['bbox']}`",
                        unsafe_allow_html=True,
                    )
                with _vp_c3:
                    if st.button("✕", key=f"del_vp_{_vi}", help="Remove this prompt"):
                        _del_idx = _vi

            if _del_idx is not None:
                st.session_state.visual_prompts.pop(_del_idx)
                st.rerun()
        else:
            st.info("No visual prompts yet — draw a bounding box above and click **Add**.")

        st.divider()

        # ── Backend-specific parameters ────────────────────────────────────────
        if _vp_backend == "WeDetect-Uni":
            st.markdown("**WeDetect-Uni parameters**")
            _vp_pc1, _vp_pc2 = st.columns([1, 2])
            with _vp_pc1:
                _vp_uni_size = st.radio(
                    "Model size",
                    options=["base", "large"],
                    index=0,
                    horizontal=True,
                    key="vp_uni_size",
                )
            _vp_default_ckpt = str(WEDETECT_DIR / UNI_CHECKPOINT_NAMES[_vp_uni_size])
            with _vp_pc2:
                _vp_ckpt_input = st.text_input(
                    "Checkpoint path",
                    value=_vp_default_ckpt,
                    key="vp_uni_checkpoint",
                    help="Path to wedetect_base_uni.pth or wedetect_large_uni.pth",
                )

            _vp_ckpt_path = Path(_vp_ckpt_input.strip())
            if not _vp_ckpt_path.exists():
                st.warning(
                    f"Checkpoint `{_vp_ckpt_path.name}` not found. "
                    "Download **WeDetect-Base-Uni** or **WeDetect-Large-Uni** from "
                    "[fushh7/WeDetect](https://huggingface.co/fushh7/WeDetect) "
                    "and place it in `WeDetect/checkpoints/`."
                )

            _vp_param_c1, _vp_param_c2, _vp_param_c3, _vp_param_c4 = st.columns(4)
            with _vp_param_c1:
                _vp_topk = st.slider(
                    "Top-K matches",
                    min_value=5, max_value=500, value=100, step=5,
                    key="vp_topk",
                    help="Maximum total detections to keep (sorted by similarity).",
                )
            with _vp_param_c2:
                _vp_min_sim = st.slider(
                    "Min similarity",
                    min_value=0.50, max_value=1.00, value=0.75, step=0.01,
                    key="vp_min_sim",
                    help="Cosine similarity threshold — higher = stricter.",
                )
            with _vp_param_c3:
                _vp_score_thr = st.slider(
                    "Proposal score threshold",
                    min_value=0.00, max_value=0.50, value=0.00, step=0.01,
                    key="vp_score_thr",
                    help="WeDetect-Uni objectness score threshold for candidate proposals.",
                )
            with _vp_param_c4:
                _vp_max_overlap = st.slider(
                    "Max overlap (dedup)",
                    min_value=0.00, max_value=0.90, value=0.30, step=0.05,
                    key="vp_max_overlap",
                    help=(
                        "Proposals with IoU ≥ this threshold against existing annotations "
                        "are excluded. Set to 0 to disable."
                    ),
                )

            _vp_nms_c1, _vp_nms_c2, _vp_nms_c3 = st.columns([1, 2, 3])
            with _vp_nms_c1:
                _vp_nms = st.checkbox(
                    "Apply NMS",
                    value=True,
                    key="vp_wedetect_nms",
                    help=(
                        "Per-frame per-class Non-Maximum Suppression: removes overlapping "
                        "detections after similarity matching, before the top-K cut."
                    ),
                )
            with _vp_nms_c2:
                _vp_nms_iou = st.slider(
                    "NMS IoU threshold",
                    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                    key="vp_wedetect_nms_iou",
                    disabled=not _vp_nms,
                    help="Boxes with IoU ≥ this are suppressed (lower = more aggressive).",
                )

            _vp_backend_ready = _vp_ckpt_path.exists()

        else:  # YOLO-E
            st.markdown("**YOLO-E parameters**")
            _yoloe_pc1, _yoloe_pc2 = st.columns([1, 2])
            with _yoloe_pc1:
                _yoloe_model_size = st.radio(
                    "Model size",
                    options=["small", "medium", "large"],
                    index=0,
                    horizontal=True,
                    key="vp_yoloe_size",
                    help="small=yoloe-11s, medium=yoloe-11m, large=yoloe-11l",
                )
            with _yoloe_pc2:
                _yoloe_fsdet_input = st.text_input(
                    "YOLO-E package dir",
                    value=str(FSDET_DIR),
                    key="vp_yoloe_fsdet_dir",
                    help="Path to the few-shot-object-detection package root.",
                )

            _yoloe_fsdet_path = Path(_yoloe_fsdet_input.strip())
            _yoloe_pkg_ok = (_yoloe_fsdet_path / "few_shot_object_detection").is_dir()
            if not _yoloe_pkg_ok:
                st.warning(
                    f"`few_shot_object_detection` package not found at "
                    f"`{_yoloe_fsdet_path}`. "
                    "Point to the root of the `few-shot-object-detection` repository."
                )

            _yoloe_p1, _yoloe_p2, _yoloe_p3, _yoloe_p4 = st.columns(4)
            with _yoloe_p1:
                _yoloe_conf = st.slider(
                    "Confidence threshold",
                    min_value=0.05, max_value=0.90, value=0.25, step=0.05,
                    key="vp_yoloe_conf",
                    help="YOLO-E detection confidence threshold.",
                )
            with _yoloe_p2:
                _yoloe_nms = st.checkbox(
                    "Apply NMS",
                    value=True,
                    key="vp_yoloe_nms",
                    help="Per-class non-maximum suppression.",
                )
            with _yoloe_p3:
                _yoloe_nms_iou = st.slider(
                    "NMS IoU threshold",
                    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                    key="vp_yoloe_nms_iou",
                    disabled=not _yoloe_nms,
                )
            with _yoloe_p4:
                _vp_max_overlap = st.slider(
                    "Max overlap (dedup)",
                    min_value=0.00, max_value=0.90, value=0.30, step=0.05,
                    key="vp_yoloe_max_overlap",
                    help=(
                        "Detections with IoU ≥ this against existing annotations "
                        "are excluded. Set to 0 to disable."
                    ),
                )

            _vp_backend_ready = _yoloe_pkg_ok

        can_run = False  # text mode variable — not used in visual mode
        run_clicked = False

        can_run_visual = (
            _SDC_AVAILABLE
            and bool(_vp_list)
            and _vp_backend_ready
        )

        run_visual_clicked = st.button(
            "Run Visual Detection",
            type="primary",
            disabled=not can_run_visual,
            key="run_visual_detection_btn",
        )

        if not can_run_visual and not run_visual_clicked:
            if not _SDC_AVAILABLE:
                st.info("Install `streamlit-drawable-canvas` to enable visual detection.")
            elif not _vp_list:
                st.info("Add at least one visual prompt above to enable detection.")
            elif not _vp_backend_ready:
                if _vp_backend == "WeDetect-Uni":
                    st.info("Provide a valid WeDetect-Uni checkpoint to enable detection.")
                else:
                    st.info("Point to a valid `few-shot-object-detection` package directory.")

    # ── Detection pipeline ────────────────────────────────────────────────────
    _pending_key = f"pending_detection_{selected_dataset}"
    should_run_text   = detect_mode == "Text prompts"   and run_clicked
    should_run_visual = detect_mode == "Visual prompts" and run_visual_clicked

    # ── TEXT detection pipeline ───────────────────────────────────────────────
    if should_run_text and can_run:
        ann_dir = dataset_dir / "annotations"
        vis_dir = dataset_dir / "_vis"
        pending_json = ann_dir / "base_detection_pending.json"

        ann_dir.mkdir(parents=True, exist_ok=True)

        class_names = [c.strip() for c in classes_input.split(",") if c.strip()]

        status_text = st.empty()
        status_text.info("Launching WeDetect…")

        runner_script = str(PROJECT_ROOT / "src" / "annotation_runner.py")

        cmd = [
            sys.executable,
            runner_script,
            "--wedetect-dir", str(WEDETECT_DIR),
            "--config", CONFIG_MAP[model_size][0],
            "--checkpoint", str(checkpoint_path),
            "--images-dir", str(frames_dir),
            "--classes", ",".join(class_names),
            "--output-json", str(pending_json),
            "--vis-dir", str(vis_dir),
            "--threshold", str(threshold),
            "--topk", str(int(topk)),
            "--adaptive-threshold-ratio", str(adaptive_threshold_ratio),
            "--image-root", IMAGE_ROOT,
        ]
        if adaptive_threshold:
            cmd.append("--adaptive-threshold")

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONSTARTUP", None)
        clean_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        clean_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=clean_env,
        )

        stderr_lines: list[str] = []

        def _read_stderr() -> None:
            for line in proc.stderr:
                stderr_lines.append(line.rstrip())

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        det_progress = st.progress(0.0, text="Waiting for model to load…")
        thumbnails_placeholder = st.empty()
        vis_entries: list[tuple[str, str, int]] = []
        done_data: dict | None = None

        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                status_text.text(line)
                continue

            msg_type = data.get("type")

            if msg_type == "log":
                status_text.info(data["msg"])

            elif msg_type == "frame":
                fraction = data["done"] / max(data["total"], 1)
                det_progress.progress(
                    fraction,
                    text=f"Frame {data['done']}/{data['total']} — {data['n_det']} detection(s)",
                )
                if data.get("vis_path"):
                    vis_entries.append(
                        (data["vis_path"], data["frame_name"], data["n_det"])
                    )

                with thumbnails_placeholder.container(height=450):
                    visible = vis_entries[-50:]
                    if visible:
                        thumb_cols = st.columns(4)
                        for i, (vp, fn, nd) in enumerate(visible):
                            thumb_cols[i % 4].image(
                                vp,
                                caption=f"{fn} · {nd} det",
                                use_container_width=True,
                            )

            elif msg_type == "done":
                done_data = data

            elif msg_type == "error":
                st.error(data["msg"])

        proc.wait()
        stderr_thread.join(timeout=3.0)

        status_text.empty()

        if done_data:
            det_progress.progress(1.0, text="Detection complete — review results and click Apply.")
            st.session_state[_pending_key] = {
                "pending_path": str(pending_json),
                "done_data": done_data,
            }
        else:
            det_progress.empty()
            thumbnails_placeholder.empty()
            st.warning("Processing ended without a completion message.")

        if stderr_lines:
            if st.checkbox("Show process output / errors", key="tp_stderr_expander"):
                st.code("\n".join(stderr_lines[-200:]), language=None)

    # ── VISUAL detection pipeline ─────────────────────────────────────────────
    elif should_run_visual:
        import tempfile

        ann_dir = dataset_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        pending_json = ann_dir / "base_detection_pending.json"

        # Write visual prompts to a temporary JSON file
        _vq_fd, _vq_path = tempfile.mkstemp(suffix=".json", prefix="visual_prompts_")
        try:
            with os.fdopen(_vq_fd, "w") as _vq_fh:
                json.dump(st.session_state.visual_prompts, _vq_fh)

            _backend_label = st.session_state.get("vp_backend", "WeDetect-Uni")
            vp_status = st.empty()
            vp_status.info(f"Launching {_backend_label} for visual prompt detection…")

            vp_vis_dir = dataset_dir / "_vis_visual"
            vp_runner = str(PROJECT_ROOT / "src" / "visual_detection_runner.py")

            # Shared base command
            vp_cmd = [
                sys.executable,
                vp_runner,
                "--query-json",  _vq_path,
                "--images-dir",  str(frames_dir),
                "--output-json", str(pending_json),
                "--image-root",  IMAGE_ROOT,
                "--vis-dir",     str(vp_vis_dir),
            ]

            if _backend_label == "WeDetect-Uni":
                vp_cmd += [
                    "--backend",         "wedetect",
                    "--wedetect-dir",    str(WEDETECT_DIR),
                    "--uni-checkpoint",  str(_vp_ckpt_path),
                    "--top-k",           str(_vp_topk),
                    "--min-similarity",  str(_vp_min_sim),
                    "--score-threshold", str(_vp_score_thr),
                    "--max-overlap",     str(_vp_max_overlap),
                    "--nms-iou",         str(_vp_nms_iou),
                ]
                if _vp_nms:
                    vp_cmd.append("--nms")
                if output_json_path.exists() and _vp_max_overlap > 0:
                    vp_cmd += ["--annotation-json", str(output_json_path)]
            else:  # YOLO-E
                vp_cmd += [
                    "--backend",          "yoloe",
                    "--fsdet-dir",        str(_yoloe_fsdet_path),
                    "--yoloe-model-size", _yoloe_model_size,
                    "--yoloe-confidence", str(_yoloe_conf),
                    "--yoloe-nms-iou",    str(_yoloe_nms_iou),
                    "--max-overlap",      str(_vp_max_overlap),
                ]
                if _yoloe_nms:
                    vp_cmd.append("--yoloe-nms")
                if output_json_path.exists() and _vp_max_overlap > 0:
                    vp_cmd += ["--annotation-json", str(output_json_path)]

            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)
            clean_env.pop("PYTHONSTARTUP", None)
            clean_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            clean_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            vp_proc = subprocess.Popen(
                vp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=clean_env,
            )

            vp_stderr_lines: list[str] = []

            def _read_vp_stderr() -> None:
                for line in vp_proc.stderr:
                    vp_stderr_lines.append(line.rstrip())

            vp_stderr_thread = threading.Thread(target=_read_vp_stderr, daemon=True)
            vp_stderr_thread.start()

            vp_progress = st.progress(0.0, text="Waiting for model to load…")
            vp_thumbnails_placeholder = st.empty()
            vp_done_data: dict | None = None

            for raw_line in vp_proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    vp_status.text(line)
                    continue

                msg_type = data.get("type")

                if msg_type == "log":
                    vp_status.info(data["msg"])

                elif msg_type == "frame":
                    frac = data["done"] / max(data["total"], 1)
                    vp_progress.progress(
                        frac,
                        text=(
                            f"Frame {data['done']}/{data['total']} · "
                            f"{data.get('n_props', 0)} proposals · "
                            f"{data.get('n_candidates', 0)} candidates"
                        ),
                    )

                elif msg_type == "done":
                    vp_done_data = data

                elif msg_type == "error":
                    st.error(data["msg"])

            vp_proc.wait()
            vp_stderr_thread.join(timeout=3.0)
            vp_status.empty()

            if vp_done_data:
                vp_progress.progress(
                    1.0,
                    text="Visual detection complete — review and click Apply.",
                )
                # Show detection thumbnails (same layout as text prompt mode)
                _vp_vis_entries = vp_done_data.get("vis_entries", [])
                if _vp_vis_entries:
                    with vp_thumbnails_placeholder.container(height=450):
                        _vp_thumb_cols = st.columns(4)
                        for _vi, _ve in enumerate(_vp_vis_entries):
                            _vp_thumb_cols[_vi % 4].image(
                                _ve["vis_path"],
                                caption=f"{_ve['frame_name']} · {_ve['n_det']} det",
                                use_container_width=True,
                            )

                st.session_state[_pending_key] = {
                    "pending_path": str(pending_json),
                    "done_data":    vp_done_data,
                }
            else:
                vp_progress.empty()
                vp_thumbnails_placeholder.empty()
                st.warning("Visual detection ended without a completion message.")

            if vp_stderr_lines:
                if st.checkbox("Show process output / errors", key="vp_stderr_expander"):
                    st.code("\n".join(vp_stderr_lines[-200:]), language=None)

        finally:
            try:
                os.unlink(_vq_path)
            except OSError:
                pass

    # ── Pending detection results (shared by both modes) ──────────────────────
    _pending = st.session_state.get(_pending_key)
    if _pending:
        _pending_path = Path(_pending["pending_path"])
        if not _pending_path.exists():
            st.session_state.pop(_pending_key, None)
        else:
            _done_data = _pending["done_data"]

            st.divider()
            st.success(
                f"Detection finished — **{_done_data['num_annotations']}** annotation(s) "
                "ready to apply. Review below and click **Apply** to save."
            )

            _counts = _done_data.get("counts_per_class", {})
            _df_pending = pd.DataFrame(
                [{"Class": name, "Detections": cnt} for name, cnt in _counts.items()]
            )
            st.dataframe(_df_pending, use_container_width=True, hide_index=True)
            st.caption(
                f"{_done_data['num_images']} images processed · "
                f"{_done_data['num_annotations']} total annotations"
            )

            _apply_col, _discard_col, _ = st.columns([1, 1, 2])
            with _apply_col:
                if st.button(
                    "Apply Detection Results",
                    type="primary",
                    key="apply_detection_btn",
                    use_container_width=True,
                ):
                    try:
                        ann_dir = dataset_dir / "annotations"
                        ann_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(_pending_path), str(output_json_path))
                        st.session_state.pop(_pending_key, None)
                        st.session_state["gallery_ready_dataset"] = selected_dataset
                        _load_gallery.clear()
                        st.rerun()
                    except Exception as _exc:
                        st.error(f"Failed to apply results: {_exc}")
            with _discard_col:
                if st.button(
                    "Discard",
                    key="discard_detection_btn",
                    use_container_width=True,
                ):
                    try:
                        if _pending_path.exists():
                            _pending_path.unlink()
                    except Exception:
                        pass
                    st.session_state.pop(_pending_key, None)
                    st.rerun()

# ─── Results gallery (shown after Run Detection or when results already exist) ─

if output_json_path.exists():
    st.session_state["gallery_ready_dataset"] = selected_dataset

if st.session_state.get("gallery_ready_dataset") != selected_dataset:
    st.info("Click **Run Detection** to generate annotations for this dataset.")
    st.stop()

if not output_json_path.exists():
    st.warning("No detection results found for this dataset yet.")
    st.stop()

st.divider()

# ── Load annotation data ──────────────────────────────────────────────────────

file_sig = output_json_path.stat().st_mtime
gallery = _load_gallery(str(output_json_path), str(frames_dir), file_sig)

if not gallery:
    st.info("No images found in the annotation file.")
    st.stop()

num_with_dets = sum(1 for e in gallery if e["annotations"])
num_total_dets = sum(len(e["annotations"]) for e in gallery)

# ─── 2. Remove Wrong Detections ──────────────────────────────────────────────

with st.expander("2. Remove Wrong Detections", expanded=False):
    st.caption(
        f"{len(gallery)} images · {num_with_dets} with detections · "
        f"{num_total_dets} bounding boxes"
    )
    _render_crops_by_class(gallery)

    st.divider()

    st.markdown(
        "Pick image + bbox combinations that were incorrectly detected. "
        "Use the **bbox IDs** shown on each crop above (format `imgIndex-boxIndex`). "
        "You can delete them directly or first search for visually similar detections to remove in bulk."
    )

    bbox_lookup: dict[str, tuple[dict, dict]] = {}
    all_bbox_ids: list[str] = []
    all_classes: list[str] = []
    for entry in gallery:
        for ann in entry["annotations"]:
            bid = ann["bbox_id"]
            bbox_lookup[bid] = (entry, ann)
            all_bbox_ids.append(bid)
            cls = ann.get("category_name", "?")
            if cls not in all_classes:
                all_classes.append(cls)

    selected_classes = st.multiselect(
        "Filter by class",
        options=sorted(all_classes),
        default=sorted(all_classes),
        key="wrong_class_filter",
    )

    filtered_bbox_ids = [
        bid for bid in all_bbox_ids
        if bbox_lookup[bid][1].get("category_name", "?") in selected_classes
    ]

    prev_wrong = [
        bid for bid in st.session_state.get("selected_wrong_bboxes", [])
        if bid in set(filtered_bbox_ids)
    ]

    selected_wrong = st.multiselect(
        "Wrong bbox IDs",
        options=filtered_bbox_ids,
        default=prev_wrong,
        format_func=lambda bid: (
            f"{bbox_lookup[bid][1].get('category_name', '?')}  ·  "
            f"{bid}  —  {bbox_lookup[bid][0]['file_name']}  "
            f"bbox {bbox_lookup[bid][1]['bbox']}"
        ),
        key="wrong_bbox_multiselect",
    )
    st.session_state["selected_wrong_bboxes"] = selected_wrong

    if selected_wrong:
        st.info(f"{len(selected_wrong)} bbox(es) selected as wrong.")
        st.markdown("**Preview of selected crops (grouped by class):**")
        _render_crops_by_class(
            gallery,
            only_ids=set(selected_wrong),
            container_height=300,
        )

    st.divider()
    st.markdown("#### Find Similar Objects (Visual Similarity Search)")
    st.caption(
        "Uses a pretrained ViT-B/16 vision transformer to embed bounding-box crops "
        "and retrieve visually similar detections across all images."
    )

    col_sim1, col_sim2 = st.columns([2, 1])
    with col_sim1:
        top_k = st.slider("Max similar matches to return", 1, 50, 10)
    with col_sim2:
        min_sim = st.slider("Minimum similarity threshold", 0.10, 1.00, 0.70, step=0.01)

    run_similarity = st.button(
        "Search for Similar Objects",
        disabled=not bool(selected_wrong),
        help="Select wrong bbox IDs above first.",
    )

    if run_similarity:
        if not selected_wrong:
            st.warning("No wrong bboxes selected.")
        else:
            with st.spinner("Computing embeddings…"):
                query_crops: list[Image.Image] = []
                for bid in selected_wrong:
                    entry, ann = bbox_lookup[bid]
                    query_crops.append(crop_bbox(entry["image_path"], ann["bbox"]))

                candidate_crops: list[Image.Image] = []
                candidate_meta: list[dict[str, Any]] = []
                for entry in gallery:
                    for ann in entry["annotations"]:
                        if ann.get("category_name", "?") not in selected_classes:
                            continue
                        candidate_crops.append(crop_bbox(entry["image_path"], ann["bbox"]))
                        candidate_meta.append(
                            {
                                "bbox_id": ann["bbox_id"],
                                "file_name": entry["file_name"],
                                "image_path": str(entry["image_path"]),
                                "img_idx": entry["img_idx"],
                                "bbox": ann["bbox"],
                                "ann_id": ann["id"],
                                "image_id": entry["image_id"],
                                "entry": entry,
                                "ann": ann,
                            }
                        )

                query_embs = embed_crops(query_crops)
                cand_embs = embed_crops(candidate_crops)

            matches = find_similar(
                query_embs,
                cand_embs,
                candidate_meta,
                top_k=top_k,
                min_similarity=min_sim,
            )
            st.session_state["similarity_matches"] = matches
            st.session_state["similarity_wrong_ids"] = set(selected_wrong)

    if st.session_state.get("similarity_matches"):
        matches: list[dict] = st.session_state["similarity_matches"]

        matches_filtered = [
            m for m in matches
            if m["ann"].get("category_name", "?") in selected_classes
        ]

        st.markdown(
            f"**{len(matches_filtered)} similar match(es) found**"
            + (
                f" ({len(matches) - len(matches_filtered)} hidden by class filter)"
                if len(matches_filtered) != len(matches) else ""
            )
        )

        matched_ids = {m["bbox_id"] for m in matches_filtered}
        sim_captions = {
            m["bbox_id"]: f"{m['bbox_id']}  sim {m['similarity']:.2f}"
            for m in matches_filtered
        }
        _render_crops_by_class(
            gallery,
            only_ids=matched_ids,
            id_captions=sim_captions,
            container_height=400,
        )

        st.markdown("**Match details:**")
        df_matches = pd.DataFrame(
            [
                {
                    "Similarity": f"{m['similarity']:.3f}",
                    "Class": m["ann"].get("category_name", "?"),
                    "bbox_id": m["bbox_id"],
                    "Image": m["file_name"],
                    "BBox [x,y,w,h]": m["bbox"],
                }
                for m in matches_filtered
            ]
        )
        st.dataframe(df_matches, use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("#### Remove Annotations from Dataset")

        remove_options = {m["bbox_id"]: m for m in matches_filtered}
        remove_options.update(
            {
                bid: {
                    "bbox_id": bid,
                    "ann_id": bbox_lookup[bid][1]["id"],
                    "image_id": bbox_lookup[bid][0]["image_id"],
                    "file_name": bbox_lookup[bid][0]["file_name"],
                }
                for bid in (selected_wrong or [])
                if bid in bbox_lookup
                and bbox_lookup[bid][1].get("category_name", "?") in selected_classes
            }
        )

        to_remove = st.multiselect(
            "Bbox IDs to remove from annotation JSON",
            options=list(remove_options.keys()),
            default=list(remove_options.keys()),
            format_func=lambda bid: (
                f"{remove_options[bid].get('ann', {}).get('category_name') or remove_options[bid].get('file_name', '?')}  ·  "
                f"{bid}  —  {remove_options[bid].get('file_name', '?')}"
            ),
            key="remove_bbox_multiselect",
        )

        if to_remove:
            remove_ann_ids = {
                remove_options[bid]["ann_id"]
                for bid in to_remove
                if "ann_id" in remove_options.get(bid, {})
            }
            st.warning(
                f"This will permanently remove {len(remove_ann_ids)} annotation(s) "
                f"from `{output_json_path.name}`."
            )
            if st.button(
                f"Remove {len(remove_ann_ids)} annotation(s)",
                type="primary",
                key="remove_similarity_result",
            ):
                _remove_annotations(output_json_path, remove_ann_ids)

    elif st.session_state.get("similarity_matches") == []:
        st.info("No matches found above the similarity threshold.")

    if selected_wrong:
        st.divider()
        wrong_ann_ids_direct = {
            bbox_lookup[bid][1]["id"]
            for bid in selected_wrong
            if bid in bbox_lookup
        }
        st.warning(
            f"This will permanently remove **{len(wrong_ann_ids_direct)}** "
            f"selected annotation(s) from `{output_json_path.name}`."
        )
        if st.button(
            f"Delete {len(wrong_ann_ids_direct)} selected annotation(s)",
            type="primary",
            key="delete_selected_direct",
        ):
            _remove_annotations(output_json_path, wrong_ann_ids_direct)


# ─── 3. Search for Missing Detections ────────────────────────────────────────

st.divider()

with st.expander("3. Search for Missing Detections", expanded=False):
    st.caption(
        "Use confirmed bounding boxes as query embeddings. "
        "WeDetect-Uni generates class-agnostic proposals across all frames and retrieves "
        "proposals visually similar to the query objects that are not yet annotated."
    )

    uni_col1, uni_col2 = st.columns([2, 1])
    with uni_col1:
        uni_model_size = st.radio(
            "WeDetect-Uni model size",
            options=["base", "large"],
            index=0,
            horizontal=True,
            key="uni_model_size",
        )

    default_uni_ckpt = str(WEDETECT_DIR / UNI_CHECKPOINT_NAMES[uni_model_size])
    with uni_col2:
        uni_checkpoint_input = st.text_input(
            "Checkpoint path",
            value=default_uni_ckpt,
            key="uni_checkpoint_input",
            help="Absolute or relative path to wedetect_base_uni.pth / wedetect_large_uni.pth",
        )

    uni_checkpoint_path = Path(uni_checkpoint_input.strip())

    if not uni_checkpoint_path.exists():
        st.warning(
            f"WeDetect-Uni checkpoint not found at `{uni_checkpoint_path}`. "
            "Download **WeDetect-Base-Uni** or **WeDetect-Large-Uni** from "
            "[fushh7/WeDetect](https://huggingface.co/fushh7/WeDetect) on HuggingFace "
            "and place it in `WeDetect/checkpoints/`."
        )

    all_classes_for_retrieval = sorted({
        ann.get("category_name", "?")
        for entry in gallery
        for ann in entry["annotations"]
    })

    ret_query_classes = st.multiselect(
        "Query classes — which object classes to search for",
        options=all_classes_for_retrieval,
        default=all_classes_for_retrieval,
        key="ret_query_classes",
        help=(
            "Only annotations of these classes are used as query embeddings. "
            "All existing annotations are still used for the candidate-exclusion IoU check "
            "so already-detected regions are always filtered out, regardless of class."
        ),
    )

    ret_col1, ret_col2, ret_col3 = st.columns(3)
    with ret_col1:
        ret_top_k = st.slider(
            "Max results (top-k)",
            min_value=5,
            max_value=200,
            value=50,
            step=5,
            key="ret_top_k",
        )
    with ret_col2:
        ret_min_sim = st.slider(
            "Minimum similarity",
            min_value=0.50,
            max_value=1.00,
            value=0.75,
            step=0.01,
            key="ret_min_sim",
            help="Cosine similarity threshold. Higher = more strict.",
        )
    with ret_col3:
        ret_max_overlap = st.slider(
            "Max overlap with existing detections",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            key="ret_max_overlap",
            help=(
                "Proposals with IoU ≥ this threshold against any existing annotation "
                "are excluded from the candidate pool (already detected)."
            ),
        )

    ret_nms_c1, ret_nms_c2, _ = st.columns([1, 2, 3])
    with ret_nms_c1:
        ret_nms = st.checkbox(
            "Apply NMS",
            value=True,
            key="ret_nms",
            help=(
                "Per-frame Non-Maximum Suppression on the retrieved proposals: "
                "overlapping matches are suppressed, keeping the highest-similarity box."
            ),
        )
    with ret_nms_c2:
        ret_nms_iou = st.slider(
            "NMS IoU threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key="ret_nms_iou",
            disabled=not ret_nms,
            help="Boxes with IoU ≥ this are suppressed (lower = more aggressive).",
        )

    can_run_retrieval = (
        uni_checkpoint_path.exists()
        and num_total_dets > 0
        and bool(ret_query_classes)
    )

    if num_total_dets == 0:
        st.info("No annotations in the current detection file — run detection first.")
    elif not ret_query_classes:
        st.info("Select at least one query class to enable retrieval.")

    run_retrieval = st.button(
        "Run Object Retrieval",
        type="primary",
        disabled=not can_run_retrieval,
        key="run_retrieval_btn",
    )

    retrieval_output_path = dataset_dir / "annotations" / "retrieval_results.json"
    RETRIEVAL_RUNNER = str(PROJECT_ROOT / "src" / "retrieval_runner.py")

    if run_retrieval and can_run_retrieval:
        st.session_state.pop("retrieval_results", None)

        ret_status = st.empty()
        ret_status.info("Launching WeDetect-Uni…")

        ret_cmd = [
            sys.executable,
            RETRIEVAL_RUNNER,
            "--wedetect-dir", str(WEDETECT_DIR),
            "--uni-checkpoint", str(uni_checkpoint_path),
            "--annotation-json", str(output_json_path),
            "--images-dir", str(frames_dir),
            "--output-json", str(retrieval_output_path),
            "--top-k", str(ret_top_k),
            "--min-similarity", str(ret_min_sim),
            "--max-overlap", str(ret_max_overlap),
            "--score-threshold", "0.0",
            "--query-classes", ",".join(ret_query_classes),
            "--nms-iou", str(ret_nms_iou),
        ]
        if ret_nms:
            ret_cmd.append("--nms")

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONSTARTUP", None)
        clean_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        clean_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        ret_proc = subprocess.Popen(
            ret_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=clean_env,
        )

        ret_stderr_lines: list[str] = []

        def _read_ret_stderr() -> None:
            for line in ret_proc.stderr:
                ret_stderr_lines.append(line.rstrip())

        ret_stderr_thread = threading.Thread(target=_read_ret_stderr, daemon=True)
        ret_stderr_thread.start()

        ret_progress = st.progress(0.0, text="Initialising…")
        ret_done_data: dict | None = None

        for raw_line in ret_proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                ret_status.text(line)
                continue

            msg_type = data.get("type")
            if msg_type == "log":
                ret_status.info(data["msg"])
            elif msg_type == "frame":
                frac = data["done"] / max(data["total"], 1)
                ret_progress.progress(
                    frac,
                    text=(
                        f"Frame {data['done']}/{data['total']} · "
                        f"{data['n_props']} proposals · "
                        f"{data['n_candidates']} candidates · "
                        f"{data['n_queries']} queries"
                    ),
                )
            elif msg_type == "done":
                ret_done_data = data
            elif msg_type == "error":
                st.error(data["msg"])

        ret_proc.wait()
        ret_stderr_thread.join(timeout=3.0)
        ret_status.empty()
        ret_progress.empty()

        if ret_done_data:
            st.success(
                f"Retrieval complete — **{ret_done_data['match_count']}** potential missed "
                f"detection(s) found from {ret_done_data['query_count']} queries against "
                f"{ret_done_data['candidate_count']} candidates."
            )
            if retrieval_output_path.exists():
                with open(retrieval_output_path) as fh:
                    st.session_state["retrieval_results"] = json.load(fh)
        else:
            st.warning("Retrieval ended without a completion message.")

        if ret_stderr_lines:
            if st.checkbox("Show process output / errors", key="ret_stderr_expander"):
                st.code("\n".join(ret_stderr_lines[-200:]), language=None)

    if not st.session_state.get("retrieval_results") and retrieval_output_path.exists():
        with open(retrieval_output_path) as fh:
            st.session_state["retrieval_results"] = json.load(fh)

    if st.session_state.get("retrieval_results"):
        ret_data: dict = st.session_state["retrieval_results"]
        ret_matches: list[dict] = ret_data.get("matches", [])

        if not ret_matches:
            st.info("No matches found above the similarity threshold.")
        else:
            st.markdown(
                f"**{len(ret_matches)} potential missed detection(s)** "
                f"— from {ret_data.get('query_count', '?')} queries, "
                f"{ret_data.get('candidate_count', '?')} candidates."
            )
            st.caption(
                "Each card shows the **full frame** with existing detections in their normal "
                "colours and the proposed new bounding box in **golden yellow** (? label). "
                "Frames without any existing detection show just the proposed bbox."
            )

            fname_to_entry: dict[str, dict] = {e["file_name"]: e for e in gallery}
            indexed_matches: list[tuple[int, dict]] = list(enumerate(ret_matches))

            proposals_by_fname: dict[str, list[tuple[int, dict]]] = {}
            for global_idx, m in indexed_matches:
                proposals_by_fname.setdefault(m["file_name"], []).append((global_idx, m))

            frames_with_proposals = sum(1 for e in gallery if e["file_name"] in proposals_by_fname)
            st.caption(
                f"{frames_with_proposals} / {len(gallery)} frames have proposed detections  ·  "
                f"Existing detections shown in colour — proposed in **golden yellow (#N)**"
            )

            with st.container(height=620):
                grid_cols = st.columns(GALLERY_COLS)
                for frame_i, entry in enumerate(gallery):
                    col = grid_cols[frame_i % GALLERY_COLS]
                    fname = entry["file_name"]
                    existing_anns = entry.get("annotations", [])
                    img_idx = entry.get("img_idx", frame_i)
                    proposals = proposals_by_fname.get(fname, [])

                    try:
                        rendered = draw_frame_with_proposals(
                            image_path=entry["image_path"],
                            existing_annotations=existing_anns,
                            proposals=proposals,
                            img_idx=img_idx,
                        )
                        n_proposed = len(proposals)
                        n_existing = len(existing_anns)
                        proposal_ids = " ".join(f"#{gi}" for gi, _ in proposals)
                        caption = (
                            f"{fname} · {n_existing} existing"
                            + (f" · proposed {proposal_ids}" if n_proposed else "")
                        )
                        col.image(rendered, caption=caption, use_container_width=True)
                    except Exception as exc:
                        col.warning(f"Could not render {fname}: {exc}")

            if st.checkbox("Show match details", key="ret_match_details"):
                df_ret = pd.DataFrame(
                    [
                        {
                            "#": global_idx,
                            "Similarity": f"{m['similarity']:.3f}",
                            "Query class": m.get("matched_query", {}).get("category_name", "?"),
                            "Image": m["file_name"],
                            "BBox [x,y,w,h]": [round(v, 1) for v in m["bbox"]],
                            "Score": f"{m.get('score', 0):.3f}",
                            "Existing dets": len(
                                fname_to_entry.get(m["file_name"], {}).get("annotations", [])
                            ),
                        }
                        for global_idx, m in indexed_matches
                    ]
                )
                st.dataframe(df_ret, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### Add Matches to Annotations")
            st.caption(
                "Select which matches to add back to `base_detection.json`. "
                "Each match inherits the category of its matched query. "
                "The **#N** numbers match the badges on the image cards above."
            )

            ret_sel_mode = st.radio(
                "Selection mode",
                options=[
                    "Exclude selected — pick boxes to exclude, all others will be added",
                    "Include selected — pick boxes to add, only those will be added",
                ],
                index=0,
                horizontal=False,
                key="ret_sel_mode",
            )
            exclude_mode = ret_sel_mode.startswith("Exclude")

            match_options: dict[str, int] = {
                f"#{global_idx}  {m['file_name']}  ·  {m.get('matched_query', {}).get('category_name', '?')}  ·  sim {m['similarity']:.2f}": global_idx
                for global_idx, m in indexed_matches
            }

            if st.session_state.get("_ret_sel_mode_prev") != ret_sel_mode:
                st.session_state["ret_matches_to_add"] = []
                st.session_state["_ret_sel_mode_prev"] = ret_sel_mode

            multiselect_label = (
                "Select matches to EXCLUDE — these will NOT be added, all others will"
                if exclude_mode
                else "Select matches to ADD — only these will be added"
            )

            selected_in_widget = st.multiselect(
                multiselect_label,
                options=list(match_options.keys()),
                default=[],
                key="ret_matches_to_add",
            )

            n_total = len(match_options)
            selected_widget_set = {match_options[k] for k in selected_in_widget}

            if exclude_mode:
                to_add_indices = [gi for gi, _ in indexed_matches if gi not in selected_widget_set]
                n_to_add = len(to_add_indices)
                n_excluded = len(selected_widget_set)
                if n_excluded == 0:
                    st.info(
                        f"All {n_total} match(es) will be added. "
                        "Select boxes above to exclude any."
                    )
                else:
                    st.info(
                        f"**{n_to_add}** of {n_total} match(es) will be added · "
                        f"**{n_excluded}** excluded."
                    )
            else:
                to_add_indices = [match_options[k] for k in selected_in_widget]
                n_to_add = len(to_add_indices)
                if n_to_add == 0:
                    st.info("No matches selected — pick the ones you want to add.")
                else:
                    st.info(f"**{n_to_add}** match(es) selected for addition.")

            selected_global_indices = to_add_indices

            if st.button(
                f"Add {n_to_add} annotation(s) to base_detection.json",
                type="primary",
                disabled=n_to_add == 0,
                key="add_retrieval_annotations",
            ):
                try:
                    with open(output_json_path) as fh:
                        coco_out = json.load(fh)

                    fname_to_img: dict[str, dict] = {
                        img["file_name"]: img for img in coco_out.get("images", [])
                    }
                    cat_name_to_id: dict[str, int] = {
                        c["name"]: c["id"] for c in coco_out.get("categories", [])
                    }

                    existing_ann_ids = {a["id"] for a in coco_out.get("annotations", [])}
                    next_ann_id = max(existing_ann_ids, default=0) + 1

                    added = 0
                    for idx in selected_global_indices:
                        m = ret_matches[idx]
                        fname = m["file_name"]
                        if fname not in fname_to_img:
                            continue
                        img_info = fname_to_img[fname]

                        query_cat_name = m.get("matched_query", {}).get("category_name", "")
                        cat_id = cat_name_to_id.get(query_cat_name)
                        if cat_id is None:
                            continue

                        bbox = [round(v, 2) for v in m["bbox"]]
                        new_ann: dict = {
                            "id": next_ann_id,
                            "image_id": img_info["id"],
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": round(bbox[2] * bbox[3], 2),
                            "iscrowd": 0,
                            "score": round(m.get("similarity", 0.0), 4),
                            "retrieval_source": "wedetect_uni",
                        }
                        coco_out.setdefault("annotations", []).append(new_ann)
                        next_ann_id += 1
                        added += 1

                    with open(output_json_path, "w") as fh:
                        json.dump(coco_out, fh, indent=2)

                    added_set = set(selected_global_indices)
                    if st.session_state.get("retrieval_results"):
                        remaining = [
                            m for gi, m in indexed_matches
                            if gi not in added_set
                        ]
                        st.session_state["retrieval_results"]["matches"] = remaining
                        st.session_state["retrieval_results"]["match_count"] = len(remaining)
                        with open(retrieval_output_path, "w") as fh:
                            json.dump(st.session_state["retrieval_results"], fh, indent=2)

                    st.session_state.pop("ret_matches_to_add", None)
                    st.session_state.pop("_ret_sel_mode_prev", None)

                    _load_gallery.clear()

                    st.success(f"Added {added} annotation(s). Refreshing…")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Failed to add annotations: {exc}")


# ─── Generate Segmentation Masks ─────────────────────────────────────────────

st.divider()

SEG_RUNNER = str(PROJECT_ROOT / "src" / "segmentation_runner.py")

SEG_MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "SAM 2": {
        "SAM 2.1 large  (sam2.1_l)":  "sam2.1_l.pt",
        "SAM 2.1 base   (sam2.1_b)":  "sam2.1_b.pt",
        "SAM 2.1 small  (sam2.1_s)":  "sam2.1_s.pt",
        "SAM 2.1 tiny   (sam2.1_t)":  "sam2.1_t.pt",
        "SAM 2 large    (sam2_l)":    "sam2_l.pt",
        "SAM 2 base     (sam2_b)":    "sam2_b.pt",
        "SAM 2 small    (sam2_s)":    "sam2_s.pt",
        "SAM 2 tiny     (sam2_t)":    "sam2_t.pt",
    },
    "FastSAM": {
        "FastSAM-x  (large)": "FastSAM-x.pt",
        "FastSAM-s  (small)": "FastSAM-s.pt",
    },
}

with st.expander("4. Generate Segmentation Masks", expanded=False):
    st.caption(
        "Generate pixel-level segmentation masks for every bounding box in the "
        "current detection file using **SAM 2** or **FastSAM** (both via "
        "[ultralytics](https://docs.ultralytics.com/models/sam-2/)). "
        "Masks are stored as COCO polygon segmentations directly in "
        "`base_detection.json`. Model weights are stored in `checkpoints/` and "
        "auto-downloaded there on first use. Requires `pip install ultralytics`."
    )

    seg_top_col1, seg_top_col2 = st.columns([1, 2])

    with seg_top_col1:
        seg_model_family = st.radio(
            "Model",
            options=list(SEG_MODEL_OPTIONS.keys()),
            index=0,
            horizontal=False,
            key="seg_model_family",
        )

    version_options = list(SEG_MODEL_OPTIONS[seg_model_family].keys())
    sam2_small_label = "SAM 2.1 small  (sam2.1_s)"
    default_version_idx = (
        version_options.index(sam2_small_label)
        if seg_model_family == "SAM 2" and sam2_small_label in version_options
        else 0
    )
    with seg_top_col2:
        seg_model_version = st.radio(
            "Version",
            options=version_options,
            index=default_version_idx,
            horizontal=False,
            key=f"seg_model_version_{seg_model_family}",
        )

    seg_ckpt_stem = SEG_MODEL_OPTIONS[seg_model_family][seg_model_version]
    seg_ckpt_path = PROJECT_ROOT / "checkpoints" / seg_ckpt_stem
    # Always pass the absolute path so ultralytics auto-downloads into checkpoints/
    ckpt_arg = str(seg_ckpt_path)

    seg_all_classes = sorted({
        ann.get("category_name", "?")
        for entry in gallery
        for ann in entry["annotations"]
    })

    seg_classes = st.multiselect(
        "Classes to segment",
        options=seg_all_classes,
        default=seg_all_classes,
        key="seg_classes",
        help="Only annotations of the selected classes receive a segmentation mask.",
    )

    seg_param_col1, seg_param_col2 = st.columns(2)
    with seg_param_col1:
        seg_conf = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            key="seg_conf",
            help="FastSAM confidence (not used by SAM 2).",
        )
    with seg_param_col2:
        seg_imgsz = st.select_slider(
            "Inference image size",
            options=[512, 640, 768, 1024, 1280],
            value=1024,
            key="seg_imgsz",
            help="Larger = more detail but slower.",
        )

    if num_total_dets == 0:
        st.info("No annotations in the current detection file — run detection first.")

    run_seg = st.button(
        f"Run {seg_model_family} Segmentation",
        type="primary",
        disabled=num_total_dets == 0,
        key="run_seg_btn",
    )

    if run_seg:
        seg_status = st.empty()
        seg_status.info(f"Launching {seg_model_family}…")
        seg_log_box = st.empty()
        seg_logs: list[str] = []

        model_type_arg = "sam2" if seg_model_family == "SAM 2" else "fastsam"
        _vis_seg_dir = dataset_dir / "_vis_seg"

        seg_cmd = [
            sys.executable,
            SEG_RUNNER,
            "--model-type",      model_type_arg,
            "--checkpoint",      ckpt_arg,
            "--annotation-json", str(output_json_path),
            "--images-dir",      str(frames_dir),
            "--output-json",     str(output_json_path),
            "--classes",         ",".join(seg_classes),
            "--conf",            str(seg_conf),
            "--imgsz",           str(seg_imgsz),
            "--vis-dir",         str(_vis_seg_dir),
        ]

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONSTARTUP", None)
        clean_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        seg_proc = subprocess.Popen(
            seg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env,
        )

        seg_progress = st.progress(0.0)
        seg_thumbnails_placeholder = st.empty()
        seg_vis_entries: list[tuple[str, str, int]] = []

        for raw_line in seg_proc.stdout:
            raw_line = raw_line.rstrip()
            if not raw_line:
                continue
            try:
                evt = json.loads(raw_line)
            except json.JSONDecodeError:
                seg_logs.append(raw_line)
                seg_log_box.code("\n".join(seg_logs[-30:]))
                continue

            t = evt.get("type", "")
            if t == "log":
                seg_logs.append(evt["msg"])
                seg_log_box.code("\n".join(seg_logs[-30:]))
            elif t == "progress":
                current = evt.get("current", 0)
                total   = evt.get("total", 1)
                seg_progress.progress(current / max(total, 1))
                seg_status.info(evt.get("msg", "Running…"))
                seg_logs.append(evt.get("msg", ""))
                seg_log_box.code("\n".join(seg_logs[-30:]))

                if evt.get("vis_path"):
                    seg_vis_entries.append((
                        evt["vis_path"],
                        evt.get("frame_name", ""),
                        evt.get("n_masks", 0),
                    ))
                    with seg_thumbnails_placeholder.container(height=450):
                        visible = seg_vis_entries[-50:]
                        thumb_cols = st.columns(4)
                        for _i, (_vp, _fn, _nm) in enumerate(visible):
                            thumb_cols[_i % 4].image(
                                _vp,
                                caption=f"{_fn} · {_nm} mask(s)",
                                use_container_width=True,
                            )

            elif t == "error":
                seg_status.error(evt["msg"])
                seg_logs.append("ERROR: " + evt["msg"])
                seg_log_box.code("\n".join(seg_logs[-30:]))
            elif t == "done":
                n_masks = evt.get("total_masks", 0)
                seg_progress.progress(1.0)
                seg_status.success(
                    f"Done — generated **{n_masks}** segmentation mask(s)."
                )
                st.session_state["seg_masks_ready"] = True
                _load_gallery.clear()

        seg_proc.wait()
        if seg_proc.returncode != 0 and not any(
            ln.startswith("ERROR") for ln in seg_logs
        ):
            seg_status.error(
                f"Runner exited with code {seg_proc.returncode}. "
                "See the log above for details."
            )
        else:
            st.rerun()

    masked_entries = [
        e for e in gallery
        if any(ann.get("segmentation") for ann in e["annotations"])
    ]

    if masked_entries:
        n_masked_anns = sum(
            sum(1 for ann in e["annotations"] if ann.get("segmentation"))
            for e in masked_entries
        )
        st.markdown(
            f"**{n_masked_anns}** annotation(s) across **{len(masked_entries)}** "
            "frame(s) have segmentation masks."
        )

        mask_cols_per_row = 3
        for row_start in range(0, len(masked_entries), mask_cols_per_row):
            row_entries = masked_entries[row_start : row_start + mask_cols_per_row]
            mask_row_cols = st.columns(mask_cols_per_row)
            for col, entry in zip(mask_row_cols, row_entries):
                with col:
                    mask_img = draw_detections_with_masks(
                        entry["image_path"],
                        entry["annotations"],
                        entry["img_idx"],
                    )
                    n_with_mask = sum(
                        1 for ann in entry["annotations"] if ann.get("segmentation")
                    )
                    _mask_buf = io.BytesIO()
                    mask_img.save(_mask_buf, format="PNG")
                    _mask_b64 = base64.b64encode(_mask_buf.getvalue()).decode()
                    caption = (
                        f"{entry['file_name']}  ·  "
                        f"{n_with_mask}/{len(entry['annotations'])} masked"
                    )
                    st.markdown(
                        f'<img src="data:image/png;base64,{_mask_b64}" '
                        f'style="width:100%" alt="{caption}"/>'
                        f'<p style="font-size:0.8em;color:grey">{caption}</p>',
                        unsafe_allow_html=True,
                    )
