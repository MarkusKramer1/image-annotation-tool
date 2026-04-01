"""Page 2 — Base Class Detection: run WeDetect on extracted frames."""

from __future__ import annotations

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
    IMAGE_ROOT,
    PROJECT_ROOT,
    TEST_DATASET_NAME,
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

col_ds, col_test = st.columns([3, 1])

with col_test:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    test_available = TEST_DATASET_NAME in datasets
    test_clicked = st.button(
        "Test Detection",
        disabled=not test_available,
        help=f"Run detection on `{TEST_DATASET_NAME}` with class 'robot'.",
        use_container_width=True,
    )

if "detection_test_mode" not in st.session_state:
    st.session_state.detection_test_mode = False
if test_clicked:
    st.session_state.detection_test_mode = True

is_test_mode = st.session_state.detection_test_mode

_default_idx = 0
if is_test_mode and TEST_DATASET_NAME in datasets:
    _default_idx = datasets.index(TEST_DATASET_NAME)

with col_ds:
    selected_dataset = st.selectbox(
        "Dataset",
        options=datasets,
        index=_default_idx,
    )

dataset_dir = DATA_DIR / selected_dataset
frames_dir = dataset_dir / IMAGE_ROOT
output_json_path = dataset_dir / "annotations" / "base_detection.json"

meta = load_extraction_meta(dataset_dir)
if meta:
    st.caption(
        f"Source: `{meta.get('source_video', '?')}` · "
        f"{meta.get('num_frames', '?')} frames · "
        f"step {meta.get('frame_step', '?')}"
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

    if is_test_mode:
        _default_classes = "robot"
    elif _existing_classes:
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

    # ── Detection pipeline ────────────────────────────────────────────────────
    _pending_key = f"pending_detection_{selected_dataset}"
    should_run = run_clicked or (is_test_mode and test_clicked)

    if should_run and can_run:
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
        st.session_state.detection_test_mode = False

        if done_data:
            det_progress.progress(1.0, text="Detection complete — review results and click Apply.")
            st.session_state[_pending_key] = {
                "pending_path": str(pending_json),
                "done_data": done_data,
            }
            # No st.rerun() — keep thumbnails visible and render Apply section below
        else:
            det_progress.empty()
            thumbnails_placeholder.empty()
            st.warning("Processing ended without a completion message.")

        if stderr_lines:
            with st.expander("Process output / errors"):
                st.code("\n".join(stderr_lines[-200:]), language=None)

    # ── Pending detection results ─────────────────────────────────────────────
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
        ]

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
            with st.expander("Process output / errors"):
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

            with st.expander("Match details", expanded=False):
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
                    st.image(
                        mask_img,
                        caption=(
                            f"{entry['file_name']}  ·  "
                            f"{n_with_mask}/{len(entry['annotations'])} masked"
                        ),
                        use_container_width=True,
                    )
