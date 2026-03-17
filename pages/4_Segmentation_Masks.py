"""Page 4 — Segmentation Mask Generation (placeholder)."""

import streamlit as st

st.set_page_config(page_title="Segmentation Masks", page_icon="🎭", layout="wide")

st.title("Segmentation Mask Generation")
st.caption(
    "Generate pixel-level segmentation masks for detected objects using a "
    "foundation segmentation model."
)
st.divider()

st.info(
    "**Coming soon.** This page will:\n\n"
    "1. Load exact-class detections from `annotations/exact_detection.json`.\n"
    "2. Run a segmentation model (e.g. SAM) guided by detection bounding boxes.\n"
    "3. Store RLE-encoded masks in the COCO annotation format.\n"
    "4. Save the result to `annotations/segmentation.json`.",
    icon="🚧",
)
