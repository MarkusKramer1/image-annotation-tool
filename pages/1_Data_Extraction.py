"""Page 1 — Data Extraction: extract frames from a video file."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import streamlit as st

from src.common import DATA_DIR, TEST_VIDEO_PATH, TEST_DATASET_NAME, IMAGE_ROOT

st.set_page_config(page_title="Data Extraction", page_icon="🎞️", layout="wide")

st.title("Data Extraction")
st.caption(
    "Upload a video (or use the test pipeline) and extract frames at a "
    "configurable interval. The extracted images become the shared image "
    "source for all subsequent pipeline stages."
)
st.divider()

# ─── Configuration ────────────────────────────────────────────────────────────

st.subheader("Configuration")

frame_step = st.number_input(
    "Extract every N-th frame",
    min_value=1,
    max_value=200,
    value=5,
    help="1 = every frame. For a 30 fps video, 5 gives ~6 fps.",
)

st.divider()

# ─── Video source ─────────────────────────────────────────────────────────────

st.subheader("Video Source")

col_upload, col_test = st.columns([3, 1])

with col_upload:
    video_file = st.file_uploader("Video file (.mp4, .mov)", type=["mp4", "mov"])

with col_test:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    test_available = TEST_VIDEO_PATH.exists()
    test_clicked = st.button(
        "Test Extraction",
        disabled=not test_available,
        help=(
            f"Extract frames from `{TEST_VIDEO_PATH.name}` into "
            f"`data/{TEST_DATASET_NAME}/`."
        )
        if test_available
        else "Test video not found.",
        use_container_width=True,
    )

if "extraction_test_mode" not in st.session_state:
    st.session_state.extraction_test_mode = False
if test_clicked:
    st.session_state.extraction_test_mode = True

is_test_mode = st.session_state.extraction_test_mode

# ─── Output directory ─────────────────────────────────────────────────────────

_default_name = ""
if is_test_mode:
    _default_name = TEST_DATASET_NAME
elif video_file is not None:
    _default_name = Path(video_file.name).stem

dataset_name = st.text_input(
    "Dataset name",
    value=_default_name,
    placeholder="my_video",
    help=f"Frames will be saved to `data/<name>/{IMAGE_ROOT}/`.",
)

if dataset_name.strip():
    st.caption(f"Output directory: `{DATA_DIR / dataset_name.strip() / IMAGE_ROOT}`")

st.divider()

# ─── Validation & run ─────────────────────────────────────────────────────────

has_video = is_test_mode or video_file is not None
can_run = has_video and bool(dataset_name.strip())

run_clicked = st.button(
    "Extract Frames",
    type="primary",
    disabled=not can_run,
)

if not can_run and not run_clicked and not is_test_mode:
    missing = []
    if not has_video:
        missing.append("a video file (or click Test Extraction)")
    if not dataset_name.strip():
        missing.append("a dataset name")
    if missing:
        st.info(f"Please provide {', '.join(missing)} to continue.")

# ─── Extraction ───────────────────────────────────────────────────────────────

should_run = run_clicked or (is_test_mode and test_clicked)

if should_run and can_run:
    out_dir = (DATA_DIR / dataset_name.strip()).resolve()
    frames_dir = out_dir / IMAGE_ROOT
    ann_dir = out_dir / "annotations"

    frames_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    status_text = st.empty()
    cleanup_video = False

    if is_test_mode:
        status_text.info(f"Using test video: `{TEST_VIDEO_PATH.name}`")
        tmp_video_path = str(TEST_VIDEO_PATH)
    else:
        status_text.info("Saving uploaded video to disk…")
        upload_suffix = Path(video_file.name).suffix.lower()
        if upload_suffix not in {".mp4", ".mov"}:
            st.error(f"Unsupported file type `{upload_suffix}`.")
            st.stop()
        with tempfile.NamedTemporaryFile(suffix=upload_suffix, delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_video_path = tmp.name
        cleanup_video = True

    status_text.info("Extracting frames…")
    cap = cv2.VideoCapture(tmp_video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_progress = st.progress(0.0, text="Extracting frames…")
    extract_count = 0
    video_frame_idx = 0
    resolution: tuple[int, int] | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if video_frame_idx % int(frame_step) == 0:
            out_path = frames_dir / f"frame_{extract_count:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            if resolution is None:
                h, w = frame.shape[:2]
                resolution = (w, h)
            extract_count += 1
        video_frame_idx += 1
        if total_video_frames > 0:
            frame_progress.progress(
                video_frame_idx / total_video_frames,
                text=f"Extracting frames ({video_frame_idx}/{total_video_frames})…",
            )

    cap.release()
    if cleanup_video:
        os.unlink(tmp_video_path)
    frame_progress.empty()

    if extract_count == 0:
        st.error("No frames could be extracted from the video.")
        st.stop()

    # Write extraction metadata
    source_name = TEST_VIDEO_PATH.name if is_test_mode else video_file.name
    meta = {
        "source_video": source_name,
        "frame_step": int(frame_step),
        "num_frames": extract_count,
        "resolution": list(resolution) if resolution else None,
        "fps": round(fps, 2) if fps else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(ann_dir / "extraction.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    st.session_state.extraction_test_mode = False
    status_text.empty()

    st.success(
        f"Extracted **{extract_count}** frames to `{frames_dir.relative_to(out_dir.parent.parent)}`"
    )

    # Show a thumbnail grid of extracted frames
    sample_frames = sorted(frames_dir.iterdir())[:20]
    if sample_frames:
        cols = st.columns(5)
        for i, fp in enumerate(sample_frames):
            cols[i % 5].image(str(fp), caption=fp.name, use_container_width=True)
        if extract_count > 20:
            st.caption(f"Showing first 20 of {extract_count} frames.")
