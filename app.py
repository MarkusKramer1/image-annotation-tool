"""Image Annotation Tool — Streamlit entry point.

Single-page app. No sidebar. All controls inline, top-to-bottom.
Steps are enabled progressively as prior steps complete.
"""
import os

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Annotation Tool",
    page_icon="🏷️",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "frames": None,
    "all_detections": None,
    "embeddings": None,
    "cluster_labels": None,
    "cluster_prototypes": None,
    "cluster_decisions": {},
    "active_detections": None,
    "mined_detections": None,
    "coco_data": None,
    "step": "input",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Image Annotation Tool")
st.caption(
    "Semi-automated annotation: frame sampling → WeDetect detection → "
    "embedding clustering → cluster review → COCO export."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Input Mode
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1 · Input")

input_mode = st.radio(
    "Input mode",
    ["Upload video", "Image folder path"],
    horizontal=True,
)

uploaded_video = None
folder_path = ""
fps_value = 1.0
every_nth = 1

if input_mode == "Upload video":
    uploaded_video = st.file_uploader(
        "Upload video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Video will be sampled at the fps rate below.",
    )
    fps_value = st.slider(
        "Sample at N fps",
        min_value=0.1,
        max_value=30.0,
        value=1.0,
        step=0.1,
        help="Frames per second to extract from the video.",
    )
else:
    folder_path = st.text_input(
        "Image folder path",
        placeholder="/path/to/images",
        help="Absolute or relative path to a folder of images.",
    )
    every_nth = st.number_input(
        "Use every Nth image",
        min_value=1,
        max_value=100,
        value=1,
        help="Subsample: 1 = all images, 5 = every 5th image.",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — WeDetect Model Paths
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("2 · WeDetect Models")

col_cfg, col_ckpt = st.columns(2)
with col_cfg:
    wedetect_config = st.text_input(
        "WeDetect config path",
        placeholder="/path/to/wedetect_config.py",
    )
with col_ckpt:
    wedetect_checkpoint = st.text_input(
        "WeDetect checkpoint path",
        placeholder="/path/to/wedetect.pth",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Detection Controls
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("3 · Detection")

class_query_raw = st.text_input(
    "Detection class names (comma-separated; Chinese OK)",
    placeholder="person, car, bicycle  —or—  人, 汽车, 自行车",
    help=(
        "WeDetect uses XLM-RoBERTa text encoder; Chinese names work natively. "
        "English names will be auto-translated before inference."
    ),
)

# Parse class names into a list (stripped, non-empty)
class_names: list[str] = [
    c.strip() for c in class_query_raw.split(",") if c.strip()
]

if class_names:
    st.caption(f"Parsed categories ({len(class_names)}): " + " · ".join(class_names))

# Dynamic per-class threshold sliders
st.markdown("**Detection thresholds**")

if class_names:
    thresholds: dict[str, float] = {}
    threshold_cols = st.columns(min(len(class_names), 4))
    for i, name in enumerate(class_names):
        col = threshold_cols[i % len(threshold_cols)]
        with col:
            thresholds[name] = st.slider(
                f"`{name}`",
                min_value=0.01,
                max_value=1.0,
                value=0.30,
                step=0.01,
                key=f"threshold_{name}",
            )
else:
    st.info("Enter class names above to configure per-class thresholds.")
    thresholds = {}

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Run
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("4 · Run")

_input_ready = (
    (input_mode == "Upload video" and uploaded_video is not None)
    or (input_mode == "Image folder path" and folder_path.strip() != "")
)
_models_ready = wedetect_config.strip() != "" and wedetect_checkpoint.strip() != ""
_classes_ready = len(class_names) > 0

_run_disabled = not (_input_ready and _models_ready and _classes_ready)

if _run_disabled:
    missing = []
    if not _input_ready:
        missing.append("input source")
    if not _models_ready:
        missing.append("WeDetect model paths")
    if not _classes_ready:
        missing.append("class names")
    st.warning("Please provide: " + ", ".join(missing) + ".")

if st.button(
    "▶ Run Detection",
    type="primary",
    disabled=_run_disabled,
    help="Samples frames then runs WeDetect with the configured settings.",
):
    st.info("Detection pipeline not yet implemented (Phase 1–2).")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Downstream placeholders (disabled until detection runs)
# ─────────────────────────────────────────────────────────────────────────────
_detected = st.session_state["step"] not in ("input",)

with st.container():
    st.subheader("5 · Embedding & Clustering")
    cluster_method = st.selectbox(
        "Clustering method",
        ["HDBSCAN", "KMeans"],
        disabled=not _detected,
    )
    if cluster_method == "KMeans":
        st.number_input("Number of clusters", 2, 100, 10, disabled=not _detected)
    else:
        st.number_input("Min cluster size", 2, 50, 5, disabled=not _detected)
    st.button("Cluster", disabled=not _detected)

st.divider()

with st.container():
    st.subheader("6 · Cluster Review")
    st.info("Cluster review cards will appear here after clustering completes.")

st.divider()

with st.container():
    st.subheader("7 · Second-Pass Mining")
    _clustered = st.session_state["step"] not in ("input", "detected")
    st.radio(
        "Mining method",
        ["Embedding similarity (fast)", "WeDetect-Ref (accurate)"],
        disabled=not _clustered,
    )
    st.slider(
        "Visual similarity threshold",
        0.5,
        1.0,
        0.8,
        disabled=not _clustered,
    )
    st.button("Run Mining", disabled=not _clustered)

st.divider()

with st.container():
    st.subheader("8 · Export")
    _mined = st.session_state["step"] not in ("input", "detected", "clustered", "reviewed")
    output_dir = st.text_input(
        "Output folder",
        value=os.path.join(os.getcwd(), "output"),
        disabled=not _mined,
    )
    st.button("Export COCO JSON", disabled=not _mined)

st.divider()

with st.container():
    st.subheader("9 · Gallery")
    st.info("Annotated image gallery will appear here after detection runs.")

# ── Ensure output dir exists at startup ──────────────────────────────────────
os.makedirs(os.path.join(os.getcwd(), "output", "images"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "output", "annotated"), exist_ok=True)
