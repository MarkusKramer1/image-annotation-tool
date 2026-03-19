"""Page 3 — Exact Class Detection.

Pipeline:
  1. Dataset selector (datasets with base_detection.json)
  2. Category filter (multiselect)
  3. Crop embedding (DINOv2 ViT-B/14)
  4. Clustering (HDBSCAN or KMeans)
  5. Cluster review UI (keep / discard each cluster)
  6. Export to exact_detection.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src.common import DATA_DIR, IMAGE_ROOT, dataset_status, discover_datasets
from src.similarity_search import embed_crops

st.set_page_config(page_title="Exact Class Detection", page_icon="🔬", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────

for _key in (
    "p3_embeddings",        # np.ndarray (N, D)
    "p3_ann_ids",           # list[int] — annotation IDs parallel to embeddings
    "p3_embed_dataset",     # str — dataset name when embeddings were built
    "p3_cluster_labels",    # np.ndarray (N,) — cluster assignment (-1 = noise)
    "p3_prototypes",        # dict[int, int] — cluster_id -> index in ann_ids
    "p3_cluster_dataset",   # str — dataset name when clustering was run
    "p3_decisions",         # dict[str, str] — str(cluster_id) -> 'keep'|'discard'
    "p3_confirmed",         # bool
    "p3_selected_cats",     # list[str]
):
    if _key not in st.session_state:
        st.session_state[_key] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_coco(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _crop_with_bbox_drawn(
    image_path: Path,
    bbox: list[float],
    target_width: int = 300,
    padding: int = 40,
) -> Image.Image:
    """Return a region crop with the bbox drawn in red.

    The crop is centred on the bbox and padded to ``padding`` pixels on all
    sides, then scaled to ``target_width`` wide (aspect-preserving).
    """
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = [float(v) for v in bbox]

    # Region to crop (bbox + padding, clamped to image bounds)
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) / 2 + padding
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(img.width, int(cx + half))
    y2 = min(img.height, int(cy + half))

    region = img.crop((x1, y1, x2, y2)).copy()
    draw = ImageDraw.Draw(region)

    # Draw the bbox relative to the crop origin
    bx1 = x - x1
    by1 = y - y1
    draw.rectangle([bx1, by1, bx1 + w, by1 + h], outline=(220, 40, 40), width=3)

    # Scale to target width
    ratio = target_width / region.width
    new_h = int(region.height * ratio)
    return region.resize((target_width, new_h), Image.LANCZOS)


def _build_id_maps(coco: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    """Return (image_id -> image_info, ann_id -> annotation) lookup dicts."""
    img_map = {img["id"]: img for img in coco.get("images", [])}
    ann_map = {ann["id"]: ann for ann in coco.get("annotations", [])}
    return img_map, ann_map


def _image_path(dataset_dir: Path, img_info: dict) -> Path:
    return dataset_dir / IMAGE_ROOT / img_info["file_name"]


def _run_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan  # type: ignore
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return clusterer.fit_predict(embeddings)


def _run_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans  # type: ignore
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    return km.fit_predict(embeddings)


def _compute_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, int]:
    """For each cluster, find the index whose embedding is closest to the mean.

    Returns dict mapping cluster_id -> index into embeddings array.
    """
    prototypes: dict[int, int] = {}
    unique_ids = [c for c in np.unique(labels) if c != -1]
    for cid in unique_ids:
        mask = labels == cid
        idxs = np.where(mask)[0]
        cluster_embs = embeddings[idxs]  # (K, D)
        mean_emb = cluster_embs.mean(axis=0, keepdims=True)  # (1, D)
        # Cosine similarity (embeddings are already L2-normalised)
        sims = (cluster_embs @ mean_emb.T).squeeze()  # (K,)
        best_local = int(np.argmax(sims))
        prototypes[int(cid)] = int(idxs[best_local])
    return prototypes


# ── Title ─────────────────────────────────────────────────────────────────────

st.title("Exact Class Detection")
st.caption(
    "Refine base-class detections into precise sub-classes using crop embeddings "
    "and interactive cluster review."
)
st.divider()

# ─── 1. Dataset selector ──────────────────────────────────────────────────────

st.subheader("1 · Dataset")

all_datasets = discover_datasets()
datasets_with_base = [
    d for d in all_datasets
    if (DATA_DIR / d / "annotations" / "base_detection.json").exists()
]

if not datasets_with_base:
    st.warning(
        "No datasets with `base_detection.json` found. "
        "Run **Base Class Detection** first."
    )
    st.stop()

selected_dataset = st.selectbox("Dataset", options=datasets_with_base)
dataset_dir = DATA_DIR / selected_dataset
ann_path = dataset_dir / "annotations" / "base_detection.json"

coco = _load_coco(ann_path)
img_map, ann_map = _build_id_maps(coco)

# Category name lookup
cat_map: dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}

st.caption(
    f"`{ann_path.relative_to(DATA_DIR)}` — "
    f"{len(img_map)} images · "
    f"{len(ann_map)} annotations · "
    f"{len(cat_map)} categories"
)

st.divider()

# ─── 2. Category filter ───────────────────────────────────────────────────────

st.subheader("2 · Category Filter")

all_cat_names = sorted(cat_map.values())

if st.session_state.p3_selected_cats is None:
    st.session_state.p3_selected_cats = all_cat_names

selected_cats = st.multiselect(
    "Categories to include",
    options=all_cat_names,
    default=st.session_state.p3_selected_cats,
    help="Annotations from unselected categories are excluded from embedding & clustering.",
)
st.session_state.p3_selected_cats = selected_cats

# IDs of selected categories
selected_cat_ids = {cid for cid, name in cat_map.items() if name in selected_cats}
filtered_anns = [
    ann for ann in coco.get("annotations", [])
    if ann.get("category_id") in selected_cat_ids
]

st.caption(f"{len(filtered_anns)} annotation(s) after category filter.")

st.divider()

# ─── 3. Crop Embedding ────────────────────────────────────────────────────────

st.subheader("3 · Crop Embedding")

embeddings_ready = (
    st.session_state.p3_embeddings is not None
    and st.session_state.p3_embed_dataset == selected_dataset
    and st.session_state.p3_ann_ids is not None
    and len(st.session_state.p3_ann_ids) == len(filtered_anns)
)

if embeddings_ready:
    st.success(
        f"Embeddings ready: {len(st.session_state.p3_ann_ids)} crops × "
        f"{st.session_state.p3_embeddings.shape[1]}d"
    )

run_embed = st.button(
    "Run Embedding",
    disabled=not bool(filtered_anns),
    help="Embed all filtered annotation crops with DINOv2 ViT-B/14.",
)

if run_embed:
    crops: list[Image.Image] = []
    ann_ids: list[int] = []

    progress = st.progress(0.0, text="Loading crops…")
    total = len(filtered_anns)

    for i, ann in enumerate(filtered_anns):
        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue
        img_path = _image_path(dataset_dir, img_info)
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
            if x2 > x1 and y2 > y1:
                crops.append(img.crop((x1, y1, x2, y2)))
                ann_ids.append(ann["id"])
        except Exception:
            pass
        if (i + 1) % 50 == 0 or i == total - 1:
            progress.progress((i + 1) / total, text=f"Crop {i+1}/{total}")

    progress.empty()

    if not crops:
        st.error("No valid crops found.")
    else:
        with st.spinner(f"Embedding {len(crops)} crops with DINOv2…"):
            embs = embed_crops(crops, batch_size=64)

        st.session_state.p3_embeddings = embs
        st.session_state.p3_ann_ids = ann_ids
        st.session_state.p3_embed_dataset = selected_dataset
        # Invalidate downstream state
        st.session_state.p3_cluster_labels = None
        st.session_state.p3_prototypes = None
        st.session_state.p3_cluster_dataset = None
        st.session_state.p3_decisions = None
        st.session_state.p3_confirmed = None

        st.success(f"Embedded {len(crops)} crops → shape {embs.shape}")
        st.rerun()

st.divider()

# ─── 4. Clustering ────────────────────────────────────────────────────────────

st.subheader("4 · Clustering")

cluster_algo = st.radio(
    "Algorithm",
    options=["HDBSCAN", "KMeans"],
    horizontal=True,
)

col_param, _ = st.columns([1, 2])
with col_param:
    if cluster_algo == "HDBSCAN":
        min_cluster_size = st.number_input(
            "Min cluster size", min_value=2, max_value=500, value=5, step=1
        )
        n_clusters = None
    else:
        n_clusters = st.number_input(
            "Number of clusters", min_value=2, max_value=200, value=10, step=1
        )
        min_cluster_size = None

clustering_ready = (
    st.session_state.p3_cluster_labels is not None
    and st.session_state.p3_cluster_dataset == selected_dataset
)

if clustering_ready:
    labels = st.session_state.p3_cluster_labels
    n_valid = int(np.sum(labels >= 0))
    n_noise = int(np.sum(labels == -1))
    n_clusters_found = int(len(np.unique(labels[labels >= 0])))
    st.success(
        f"Clustering ready: {n_clusters_found} clusters, "
        f"{n_valid} assigned, {n_noise} noise points."
    )

run_cluster = st.button(
    "Run Clustering",
    disabled=not embeddings_ready,
    help="Requires embeddings from step 3.",
)

if run_cluster:
    embs = st.session_state.p3_embeddings
    with st.spinner("Clustering…"):
        if cluster_algo == "HDBSCAN":
            labels = _run_hdbscan(embs, int(min_cluster_size))
        else:
            labels = _run_kmeans(embs, int(n_clusters))
        prototypes = _compute_prototypes(embs, labels)

    st.session_state.p3_cluster_labels = labels
    st.session_state.p3_prototypes = prototypes
    st.session_state.p3_cluster_dataset = selected_dataset
    st.session_state.p3_decisions = None
    st.session_state.p3_confirmed = None

    n_found = len(prototypes)
    n_noise = int(np.sum(labels == -1))
    st.success(f"Found {n_found} clusters ({n_noise} noise points).")
    st.rerun()

st.divider()

# ─── 5. Cluster review UI ─────────────────────────────────────────────────────

st.subheader("5 · Cluster Review")

if not clustering_ready:
    st.info("Run clustering (step 4) to see clusters here.")
else:
    labels_arr: np.ndarray = st.session_state.p3_cluster_labels
    prototypes_map: dict[int, int] = st.session_state.p3_prototypes
    ann_ids_list: list[int] = st.session_state.p3_ann_ids

    # Build ann_id -> annotation lookup for selected annotations
    filtered_ann_map = {ann["id"]: ann for ann in filtered_anns}

    # Initialise decisions dict if needed
    if st.session_state.p3_decisions is None:
        st.session_state.p3_decisions = {
            str(cid): "keep" for cid in prototypes_map.keys()
        }

    decisions: dict[str, str] = st.session_state.p3_decisions

    cluster_ids = sorted(prototypes_map.keys())
    n_noise = int(np.sum(labels_arr == -1))

    if n_noise > 0:
        st.caption(
            f"{n_noise} noise point(s) (HDBSCAN label -1) will be excluded from export."
        )

    st.markdown("**Review each cluster — Keep or Discard:**")

    # Render in 2-column grid
    grid_cols = st.columns(2)

    for grid_i, cid in enumerate(cluster_ids):
        col = grid_cols[grid_i % 2]
        proto_idx = prototypes_map[cid]
        proto_ann_id = ann_ids_list[proto_idx]
        proto_ann = filtered_ann_map.get(proto_ann_id)
        cluster_size = int(np.sum(labels_arr == cid))
        cat_name = cat_map.get(proto_ann.get("category_id", -1), "?") if proto_ann else "?"

        with col:
            with st.container(border=True):
                header_cols = st.columns([3, 1])
                with header_cols[0]:
                    st.markdown(f"**Cluster {cid}** · {cluster_size} items · `{cat_name}`")

                # Render prototype image
                if proto_ann is not None:
                    img_info = img_map.get(proto_ann["image_id"])
                    if img_info:
                        img_path = _image_path(dataset_dir, img_info)
                        try:
                            thumb = _crop_with_bbox_drawn(img_path, proto_ann["bbox"])
                            st.image(thumb, use_container_width=True)
                        except Exception as exc:
                            st.warning(f"Could not render crop: {exc}")
                    else:
                        st.caption("(image not found)")
                else:
                    st.caption("(annotation not found)")

                # Keep / Discard toggle
                current = decisions.get(str(cid), "keep")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    keep_type = "primary" if current == "keep" else "secondary"
                    if st.button(
                        "Keep",
                        key=f"keep_{cid}",
                        type=keep_type,
                        use_container_width=True,
                    ):
                        st.session_state.p3_decisions[str(cid)] = "keep"
                        st.session_state.p3_confirmed = None
                        st.rerun()
                with btn_col2:
                    discard_type = "primary" if current == "discard" else "secondary"
                    if st.button(
                        "Discard",
                        key=f"discard_{cid}",
                        type=discard_type,
                        use_container_width=True,
                    ):
                        st.session_state.p3_decisions[str(cid)] = "discard"
                        st.session_state.p3_confirmed = None
                        st.rerun()

    st.divider()

    # Decision summary
    n_keep = sum(1 for v in decisions.values() if v == "keep")
    n_discard = sum(1 for v in decisions.values() if v == "discard")
    kept_ids = {int(cid) for cid, v in decisions.items() if v == "keep"}
    kept_ann_count = int(np.sum(np.isin(labels_arr, list(kept_ids))))

    st.markdown(
        f"Keeping **{n_keep}** cluster(s), discarding **{n_discard}**. "
        f"→ **{kept_ann_count}** annotation(s) will be exported."
    )

    if st.button("Confirm Decisions", type="primary", disabled=n_keep == 0):
        st.session_state.p3_confirmed = True
        st.rerun()

    st.divider()

# ─── 6. Export ────────────────────────────────────────────────────────────────

st.subheader("6 · Export")

confirmed = st.session_state.p3_confirmed
if not confirmed:
    st.info("Confirm your cluster decisions (step 5) to enable export.")
else:
    labels_arr = st.session_state.p3_cluster_labels
    ann_ids_list = st.session_state.p3_ann_ids
    decisions = st.session_state.p3_decisions
    filtered_ann_map = {ann["id"]: ann for ann in filtered_anns}

    kept_cluster_ids = {int(cid) for cid, v in decisions.items() if v == "keep"}

    # Determine which annotation IDs to keep
    kept_ann_ids: set[int] = set()
    for i, aid in enumerate(ann_ids_list):
        if labels_arr[i] in kept_cluster_ids:
            kept_ann_ids.add(aid)

    # Build COCO output from the original base_detection.json,
    # restricting to filtered categories and kept annotation IDs.
    kept_anns = [
        ann for ann in coco.get("annotations", [])
        if ann["id"] in kept_ann_ids
    ]

    # Keep only images that have at least one retained annotation
    kept_image_ids = {ann["image_id"] for ann in kept_anns}
    kept_images = [img for img in coco.get("images", []) if img["id"] in kept_image_ids]

    # Keep only relevant categories
    kept_categories = [c for c in coco.get("categories", []) if c["id"] in selected_cat_ids]

    exact_coco: dict[str, Any] = {
        "info": {
            **(coco.get("info") or {}),
            "description": "Exact-class detection (post-clustering)",
        },
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": kept_anns,
        "categories": kept_categories,
    }

    out_path = dataset_dir / "annotations" / "exact_detection.json"

    col_save, col_dl = st.columns([1, 1])

    with col_save:
        if st.button("Save exact_detection.json", type="primary"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(exact_coco, fh, indent=2)
            st.success(f"Saved to `{out_path.relative_to(DATA_DIR)}`")

    json_bytes = json.dumps(exact_coco, indent=2).encode()
    with col_dl:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="exact_detection.json",
            mime="application/json",
            use_container_width=True,
        )

    # Stats
    before_count = len(ann_map)
    after_count = len(kept_anns)
    st.markdown(
        f"**Stats:** {before_count} annotations in base → "
        f"**{after_count}** kept after clustering "
        f"({before_count - after_count} removed, "
        f"{after_count / max(before_count, 1) * 100:.1f}% retained)."
    )
    st.markdown(
        f"Images: {len(img_map)} → **{len(kept_images)}** with annotations. "
        f"Categories: {len(kept_categories)}."
    )
