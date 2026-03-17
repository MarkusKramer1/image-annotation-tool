"""Page 2 — Base Class Detection: run WeDetect on extracted frames."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Base Class Detection", page_icon="🔍", layout="wide")

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

    # ── Results ───────────────────────────────────────────────────────────────
    if done_data:
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
