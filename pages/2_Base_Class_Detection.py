"""Page 2 — Base Class Detection: run WeDetect on extracted frames."""

from __future__ import annotations

import json
import os
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
    load_detection_data,
)
from src.similarity_search import embed_crops, find_similar

st.set_page_config(page_title="Base Class Detection", page_icon="🔍", layout="wide")

if "gallery_ready_dataset" not in st.session_state:
    st.session_state["gallery_ready_dataset"] = None


# ─── Helpers (defined at module top to avoid forward-reference issues) ────────

def _remove_annotations(ann_path: Path, ann_ids: set[int]) -> None:
    """Remove annotations by ID from the COCO JSON file and refresh UI state.

    Args:
        ann_path: Path to the base_detection.json file.
        ann_ids: Set of COCO annotation IDs to delete.
    """
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

        # Clear cached gallery so it reloads from updated file
        _load_gallery.clear()
        st.session_state.pop("selected_wrong_bboxes", None)
        st.session_state.pop("similarity_matches", None)
        st.session_state.pop("similarity_wrong_ids", None)

        st.success(f"Removed {removed} annotation(s). Refreshing…")
        st.rerun()

    except Exception as exc:
        st.error(f"Failed to update annotation file: {exc}")

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

meta = load_extraction_meta(dataset_dir)
if meta:
    st.caption(
        f"Source: `{meta.get('source_video', '?')}` · "
        f"{meta.get('num_frames', '?')} frames · "
        f"step {meta.get('frame_step', '?')}"
    )

st.divider()

# ─── Configuration ────────────────────────────────────────────────────────────

st.subheader("Configuration")
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
        value=0.95,
        step=0.05,
        disabled=not adaptive_threshold,
    )
    topk = st.number_input(
        "Max detections per frame",
        min_value=1,
        max_value=500,
        value=100,
        step=10,
    )

_default_classes = "robot" if is_test_mode else ""
classes_input = st.text_input(
    "Object classes (comma-separated)",
    value=_default_classes,
    placeholder="screw,nut,flange",
)
if classes_input.strip():
    parsed = [c.strip() for c in classes_input.split(",") if c.strip()]
    st.caption(f"{len(parsed)} class(es): {', '.join(f'`{c}`' for c in parsed)}")

st.divider()

# ─── Validation ───────────────────────────────────────────────────────────────

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

# ─── Detection pipeline ──────────────────────────────────────────────────────

should_run = run_clicked or (is_test_mode and test_clicked)

if should_run and can_run:
    ann_dir = dataset_dir / "annotations"
    vis_dir = dataset_dir / "_vis"
    output_json = ann_dir / "base_detection.json"

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
        "--output-json", str(output_json),
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

    # ── Stream progress ───────────────────────────────────────────────────────
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
                    cols = st.columns(4)
                    for i, (vp, fn, nd) in enumerate(visible):
                        cols[i % 4].image(
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
    det_progress.empty()

    st.session_state.detection_test_mode = False

    # ── Summary table ─────────────────────────────────────────────────────────
    if done_data:
        st.session_state["gallery_ready_dataset"] = selected_dataset
        st.success(
            f"Done — annotation file saved to `{done_data['annotation_path']}`"
        )
        counts = done_data.get("counts_per_class", {})
        df = pd.DataFrame(
            [{"Class": name, "Detections": cnt} for name, cnt in counts.items()]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.info(
            f"Dataset at `{dataset_dir}` — "
            f"{done_data['num_images']} images, "
            f"{done_data['num_annotations']} total annotations."
        )
    else:
        st.warning("Processing ended without a completion message.")

    if stderr_lines:
        with st.expander("Process output / errors"):
            st.code("\n".join(stderr_lines[-200:]), language=None)

# ─── Results gallery (shown only after pressing Run Detection) ───────────────

if st.session_state.get("gallery_ready_dataset") != selected_dataset:
    st.info("Klicke **Run Detection**, dann erscheint direkt darunter die scrollbare Bildliste.")
    st.stop()

output_json_path = dataset_dir / "annotations" / "base_detection.json"

if not output_json_path.exists():
    st.warning("Noch keine Detection-Ergebnisse für dieses Dataset gefunden.")
    st.stop()

st.divider()

# ── Load annotation data ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_gallery(ann_path: str, images_dir: str, file_sig: float) -> list[dict[str, Any]]:
    """Load and build gallery entries.

    file_sig is typically the annotation file mtime and is included only to bust
    Streamlit cache whenever base_detection.json changes.
    """
    _ = file_sig
    coco = load_detection_data(Path(ann_path))
    return build_gallery_entries(coco, Path(images_dir))


file_sig = output_json_path.stat().st_mtime
gallery = _load_gallery(str(output_json_path), str(frames_dir), file_sig)

if not gallery:
    st.info("No images found in the annotation file.")
    st.stop()

num_with_dets = sum(1 for e in gallery if e["annotations"])
num_total_dets = sum(len(e["annotations"]) for e in gallery)
st.caption(
    f"Scrollbare Liste: {len(gallery)} Bilder gesamt, {num_with_dets} mit Detections, "
    f"{num_total_dets} Bounding Boxen."
)

# ── Gallery rendering ─────────────────────────────────────────────────────────

GALLERY_COLS = 3


def _render_gallery(
    entries: list[dict[str, Any]],
    highlighted_bbox_ids: set[str] | None = None,
    container_height: int = 600,
) -> None:
    """Render a gallery of annotated images in a scrollable container."""
    with st.container(height=container_height):
        cols = st.columns(GALLERY_COLS)
        for i, entry in enumerate(entries):
            col = cols[i % GALLERY_COLS]
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


_render_gallery(gallery)

st.divider()

# ─── Misclassification selector ───────────────────────────────────────────────

st.subheader("Review & Correct Detections")

with st.expander("Select misclassified / wrong bounding boxes", expanded=False):
    st.markdown(
        "Pick image + bbox combinations that were incorrectly detected. "
        "Use the **bbox IDs** shown on each image card above (format `imgIndex-boxIndex`)."
    )

    # Build a flat lookup: bbox_id -> (entry, ann)
    bbox_lookup: dict[str, tuple[dict, dict]] = {}
    all_bbox_ids: list[str] = []
    for entry in gallery:
        for ann in entry["annotations"]:
            bid = ann["bbox_id"]
            bbox_lookup[bid] = (entry, ann)
            all_bbox_ids.append(bid)

    selected_wrong = st.multiselect(
        "Wrong bbox IDs",
        options=all_bbox_ids,
        default=st.session_state.get("selected_wrong_bboxes", []),
        format_func=lambda bid: (
            f"{bid}  —  {bbox_lookup[bid][0]['file_name']}  "
            f"bbox {bbox_lookup[bid][1]['bbox']}"
        ),
        key="wrong_bbox_multiselect",
    )
    st.session_state["selected_wrong_bboxes"] = selected_wrong

    if selected_wrong:
        st.info(f"{len(selected_wrong)} bbox(es) selected as wrong.")
        st.markdown("**Preview of selected (highlighted in red):**")
        # Show only the images that contain at least one selected bbox
        affected_img_idxs = {bid.split("-")[0] for bid in selected_wrong}
        highlighted_entries = [
            e for e in gallery if str(e["img_idx"]) in affected_img_idxs
        ]
        _render_gallery(
            highlighted_entries,
            highlighted_bbox_ids=set(selected_wrong),
            container_height=400,
        )

st.divider()

# ─── Similarity search ────────────────────────────────────────────────────────

st.subheader("Find Similar Objects (Visual Similarity Search)")
st.caption(
    "Uses a pretrained ViT-B/16 vision transformer to embed bounding-box crops "
    "and retrieve visually similar detections across all images."
)

col_sim1, col_sim2 = st.columns([2, 1])
with col_sim1:
    top_k = st.slider("Max similar matches to return", 1, 50, 10)
with col_sim2:
    min_sim = st.slider("Minimum similarity threshold", 0.50, 1.00, 0.70, step=0.01)

run_similarity = st.button(
    "Search for Similar Objects",
    disabled=not bool(st.session_state.get("selected_wrong_bboxes")),
    help="Select wrong bbox IDs in the expander above first.",
)

if run_similarity:
    wrong_ids: list[str] = st.session_state.get("selected_wrong_bboxes", [])
    if not wrong_ids:
        st.warning("No wrong bboxes selected.")
    else:
        with st.spinner("Computing embeddings…"):
            # ── Build query crops from selected wrong bboxes ──────────────────
            query_crops: list[Image.Image] = []
            for bid in wrong_ids:
                entry, ann = bbox_lookup[bid]
                query_crops.append(crop_bbox(entry["image_path"], ann["bbox"]))

            # ── Build candidate crops from ALL annotations ────────────────────
            candidate_crops: list[Image.Image] = []
            candidate_meta: list[dict[str, Any]] = []
            for entry in gallery:
                for ann in entry["annotations"]:
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
        st.session_state["similarity_wrong_ids"] = set(wrong_ids)

# Display similarity results if available
if st.session_state.get("similarity_matches"):
    matches: list[dict] = st.session_state["similarity_matches"]
    wrong_set: set[str] = st.session_state.get("similarity_wrong_ids", set())

    st.markdown(f"**{len(matches)} similar match(es) found:**")

    # Group matches by image for gallery display
    matched_entry_map: dict[int, dict] = {}
    matched_ann_map: dict[int, list[dict]] = {}
    for m in matches:
        img_idx = m["img_idx"]
        if img_idx not in matched_entry_map:
            matched_entry_map[img_idx] = m["entry"]
            matched_ann_map[img_idx] = []
        matched_ann_map[img_idx].append(m["ann"])

    # Synthesise gallery-like entries restricted to matching annotations
    match_gallery: list[dict[str, Any]] = [
        {**matched_entry_map[idx], "annotations": matched_ann_map[idx]}
        for idx in sorted(matched_entry_map)
    ]

    _render_gallery(
        match_gallery,
        highlighted_bbox_ids={m["bbox_id"] for m in matches},
        container_height=500,
    )

    # Tabular view
    with st.expander("Match details table"):
        df_matches = pd.DataFrame(
            [
                {
                    "Similarity": f"{m['similarity']:.3f}",
                    "bbox_id": m["bbox_id"],
                    "Image": m["file_name"],
                    "BBox [x,y,w,h]": m["bbox"],
                }
                for m in matches
            ]
        )
        st.dataframe(df_matches, use_container_width=True, hide_index=True)

    st.divider()

    # ── Remove annotations button ─────────────────────────────────────────────
    st.subheader("Remove Annotations from Dataset")

    remove_options = {m["bbox_id"]: m for m in matches}
    remove_options.update(
        {
            bid: {
                "bbox_id": bid,
                "ann_id": bbox_lookup[bid][1]["id"],
                "image_id": bbox_lookup[bid][0]["image_id"],
                "file_name": bbox_lookup[bid][0]["file_name"],
            }
            for bid in (st.session_state.get("selected_wrong_bboxes") or [])
            if bid in bbox_lookup
        }
    )

    to_remove = st.multiselect(
        "Bbox IDs to remove from annotation JSON",
        options=list(remove_options.keys()),
        default=list(remove_options.keys()),
        format_func=lambda bid: (
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
        f"Remove {len(to_remove)} annotation(s)",
        type="primary",
        disabled=not bool(to_remove),
    ):
        _remove_annotations(output_json_path, remove_ann_ids)

elif st.session_state.get("similarity_matches") == []:
    st.info("No matches found above the similarity threshold.")

# ─── Stand-alone remove (without running similarity search) ──────────────────

wrong_ids_current: list[str] = st.session_state.get("selected_wrong_bboxes", [])
if wrong_ids_current and not st.session_state.get("similarity_matches"):
    st.divider()
    st.subheader("Remove Selected Wrong Annotations")
    wrong_ann_ids = {
        bbox_lookup[bid][1]["id"]
        for bid in wrong_ids_current
        if bid in bbox_lookup
    }
    st.warning(
        f"Remove {len(wrong_ann_ids)} selected wrong annotation(s) from "
        f"`{output_json_path.name}` without running similarity search?"
    )
    if st.button(
        f"Remove {len(wrong_ann_ids)} annotation(s)",
        type="primary",
        key="remove_wrong_only",
    ):
        _remove_annotations(output_json_path, wrong_ann_ids)


