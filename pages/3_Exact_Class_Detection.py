"""Page 3 — Exact Class Detection (placeholder)."""

import streamlit as st

st.set_page_config(page_title="Exact Class Detection", page_icon="🔬", layout="wide")

st.title("Exact Class Detection")
st.caption(
    "Refine the broad base-class detections from the previous stage into "
    "precise sub-classes using crop embeddings and clustering."
)
st.divider()

st.info(
    "**Coming soon.** This page will:\n\n"
    "1. Load base-class detections from `annotations/base_detection.json`.\n"
    "2. Extract crop embeddings (e.g. C-RADIOv4) for each detection.\n"
    "3. Cluster the embeddings (HDBSCAN / KMeans) to discover sub-classes.\n"
    "4. Present an interactive review UI to assign exact class labels.\n"
    "5. Save the result to `annotations/exact_detection.json`.",
    icon="🚧",
)
