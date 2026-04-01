"""Image Annotation Tool — Landing page.

Multi-page Streamlit app for semi-automated dataset annotation:
  1. Data Extraction — video to frames
  2. Base Class Detection — WeDetect open-vocabulary detection
  3. Exact Class Detection — embedding clustering
  4. Segmentation Masks — pixel-level masks (SAM2)
"""

import streamlit as st

from src.common import DATA_DIR, discover_datasets, dataset_status, load_extraction_meta

st.set_page_config(
    page_title="Image Annotation Tool",
    page_icon="🏷️",
    layout="wide",
)

st.title("Image Annotation Tool")
st.caption(
    "Semi-automated annotation pipeline: extract frames from video, detect "
    "objects with WeDetect, refine classes, and generate segmentation masks. "
    "Use the sidebar to navigate between pipeline stages."
)
st.divider()

# ─── Pipeline overview ────────────────────────────────────────────────────────

st.subheader("Pipeline")

stages = [
    ("1 — Data Extraction", "Extract frames from a video file."),
    ("2 — Base Class Detection", "Run WeDetect open-vocabulary detection on extracted frames."),
    ("3 — Exact Class Detection", "Refine base classes via embedding clustering."),
    ("4 — Segmentation Masks", "Generate pixel-level masks for detections."),
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
        if status["segmentation"]:
            badges.append("Segmentation")

        badge_str = " · ".join(f"**{b}**" for b in badges) if badges else "*no stages completed*"

        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"### {ds_name}")
            left.caption(badge_str)
            if meta:
                right.metric("Frames", meta.get("num_frames", "?"))
