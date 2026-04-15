"""Page 3 — Label Refinement.

Two modes
---------
Divide into Subclasses
    Embed all crops of ONE selected category with DINOv2, cluster them with
    HDBSCAN or KMeans, assign each cluster a precise sub-label, and export.

Union to Superclass
    Select two or more *categories* (classes) to merge globally.  For every
    image that contains annotations from those categories, all matching
    annotations are fused into a single compound bounding box / segmentation
    mask under a new super-label.  A live preview gallery shows the result
    before the file is written.

Both modes open with an overview gallery (full images with bbox overlays,
identical to page 2) and write their output to exact_detection.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src.common import DATA_DIR, IMAGE_ROOT, discover_datasets
from src.detection_gallery import (
    build_gallery_entries,
    draw_detections,
    draw_detections_with_masks,
    load_detection_data,
)
from src.similarity_search import embed_crops

GALLERY_COLS = 3

st.set_page_config(page_title="Label Refinement", page_icon="🔬", layout="wide")

# ── Session-state defaults ────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "p3_mode": "divide",
    # divide mode
    "p3_embeddings": None,
    "p3_ann_ids": None,
    "p3_embed_dataset": None,
    "p3_embed_category": None,
    "p3_cluster_labels": None,
    "p3_prototypes": None,
    "p3_cluster_dataset": None,
    "p3_decisions": None,
    "p3_cluster_names": None,
    "p3_confirmed": None,
    "p3_selected_cat": None,
    # union mode
    "p3_union_preview": None,   # list[dict] computed preview entries
    "p3_union_dataset": None,
    "p3_union_cats": [],
    "p3_union_label": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Cached gallery loader (mirrors page 2) ────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_gallery(
    ann_path: str, images_dir: str, file_sig: float
) -> list[dict[str, Any]]:
    _ = file_sig
    coco = load_detection_data(Path(ann_path))
    return build_gallery_entries(coco, Path(images_dir))


# ── Divide-mode helpers ───────────────────────────────────────────────────────

def _run_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan  # type: ignore
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean"
    ).fit_predict(embeddings)


def _run_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans  # type: ignore
    return KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(
        embeddings
    )


def _compute_prototypes(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[int, int]:
    prototypes: dict[int, int] = {}
    for cid in [c for c in np.unique(labels) if c != -1]:
        idxs = np.where(labels == cid)[0]
        cluster_embs = embeddings[idxs]
        mean_emb = cluster_embs.mean(axis=0, keepdims=True)
        sims = (cluster_embs @ mean_emb.T).squeeze()
        prototypes[int(cid)] = int(idxs[int(np.argmax(sims))])
    return prototypes


def _crop_with_bbox_drawn(
    image_path: Path,
    bbox: list[float],
    segmentation: list | None = None,
    target_width: int = 280,
    padding: int = 40,
) -> Image.Image:
    """Padded crop centred on *bbox* with bbox outline and optional segmentation mask."""
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) / 2 + padding
    x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
    x2, y2 = min(img.width, int(cx + half)), min(img.height, int(cy + half))
    region = img.crop((x1, y1, x2, y2)).copy()

    if segmentation and isinstance(segmentation, list):
        overlay = Image.new("RGBA", region.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = [(poly[i] - x1, poly[i + 1] - y1) for i in range(0, len(poly) - 1, 2)]
            if len(pts) >= 3:
                od.polygon(pts, fill=(220, 40, 40, 90))
        region = Image.alpha_composite(region.convert("RGBA"), overlay).convert("RGB")

    ImageDraw.Draw(region).rectangle(
        [x - x1, y - y1, x - x1 + w, y - y1 + h], outline=(220, 40, 40), width=3
    )
    ratio = target_width / region.width
    return region.resize((target_width, int(region.height * ratio)), Image.LANCZOS)


def _load_cluster_crops(
    cluster_id: int,
    labels_arr: np.ndarray,
    ann_ids_list: list[int],
    filtered_ann_map: dict[int, dict],
    img_map: dict[int, dict],
    dataset_dir: Path,
    max_crops: int = 60,
    thumb_size: int = 120,
) -> list[tuple[Image.Image, str]]:
    results: list[tuple[Image.Image, str]] = []
    for i, aid in enumerate(ann_ids_list):
        if labels_arr[i] != cluster_id:
            continue
        ann = filtered_ann_map.get(aid)
        if ann is None:
            continue
        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue
        img_path = dataset_dir / IMAGE_ROOT / img_info["file_name"]
        if not img_path.exists():
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(pil.width, int(x + w)), min(pil.height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil.crop((x1, y1, x2, y2))

            segmentation = ann.get("segmentation", [])
            if segmentation and isinstance(segmentation, list):
                overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                for poly in segmentation:
                    if len(poly) < 6:
                        continue
                    pts = [(poly[i] - x1, poly[i + 1] - y1) for i in range(0, len(poly) - 1, 2)]
                    if len(pts) >= 3:
                        od.polygon(pts, fill=(220, 40, 40, 90))
                crop = Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")

            crop.thumbnail((thumb_size, thumb_size))
            results.append((crop, img_info["file_name"]))
        except Exception:
            continue
        if len(results) >= max_crops:
            break
    return results


# ── Union-mode helpers ────────────────────────────────────────────────────────

def _render_preview_image(
    image_path: Path,
    annotations: list[dict[str, Any]],
    img_idx: int,
    show_bbox: bool,
    show_seg: bool,
) -> Image.Image:
    """Render a preview image with optional bbox outlines and segmentation masks.

    Unlike the detection_gallery helpers this function respects the two
    visibility toggles independently so the user can, for example, view
    masks without bounding-box outlines or vice-versa.
    """
    from PIL import ImageFont

    _COLORS = [
        (255, 80, 80), (80, 200, 80), (80, 130, 255),
        (255, 180, 0), (200, 80, 255), (0, 210, 210),
    ]

    try:
        _font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
        _font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except (OSError, IOError):
        _font = ImageFont.load_default()
        _font_lg = _font

    # Draw segmentation mask fill layer
    if show_seg and any(ann.get("segmentation") for ann in annotations):
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for ann in annotations:
            r, g, b = _COLORS[ann.get("category_id", 1) % len(_COLORS)]
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    od.polygon(pts, fill=(r, g, b, 90))
        img = Image.alpha_composite(img, overlay).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(img)

    # Draw bounding boxes + labels
    if show_bbox:
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            color = _COLORS[ann.get("category_id", 1) % len(_COLORS)]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label = ann.get("category_name", "")
            if label:
                bb = draw.textbbox((0, 0), label, font=_font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                lx, ly = int(x), max(0, int(y) - th - 4)
                draw.rectangle([lx, ly, lx + tw + 4, ly + th + 4], fill=color)
                draw.text((lx + 2, ly + 2), label, fill=(255, 255, 255), font=_font)

    # Image-index badge (always visible for reference)
    badge = f"#{img_idx}"
    bb = draw.textbbox((0, 0), badge, font=_font_lg)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    draw.rectangle([4, 4, 4 + bw + 10, 4 + bh + 10], fill=(20, 20, 20))
    draw.text((9, 9), badge, fill=(255, 220, 0), font=_font_lg)

    return img


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[0] + b[2] for b in bboxes)
    y2 = max(b[1] + b[3] for b in bboxes)
    return [x1, y1, x2 - x1, y2 - y1]


def _union_segmentation(anns: list[dict]) -> list:
    """Merge polygon segmentation from multiple annotations.

    For each annotation:
    - If it carries polygon segmentation (list of coordinate lists), all
      polygons are included as-is.
    - If it has no polygon segmentation (RLE, empty, or missing), a
      rectangular polygon is generated from its bounding box so the merged
      mask always covers every source object's area.

    COCO allows multiple disjoint polygons in one annotation, so the result
    is a valid multi-polygon segmentation list.
    """
    combined: list = []
    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and seg:
            # Polygon format — include all polygons directly
            combined.extend(seg)
        else:
            # No usable polygon mask → fall back to the bounding-box rectangle
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w > 0 and h > 0:
                rect = [x, y, x + w, y, x + w, y + h, x, y + h]
                combined.append(rect)
    return combined


def _build_union_preview_entries(
    gallery: list[dict[str, Any]],
    union_cat_ids: set[int],
    new_label: str,
) -> list[dict[str, Any]]:
    """Return gallery entries where union-category annotations are fused.

    For each image:
    - Annotations whose category is NOT in *union_cat_ids* pass through unchanged.
    - All annotations whose category IS in *union_cat_ids* are replaced by a
      single merged annotation with *new_label*.

    The merged annotation uses a synthetic ``category_id`` of 99999 so it
    maps to a distinct palette colour (orange, index 3).
    """
    preview: list[dict[str, Any]] = []
    for entry in gallery:
        anns = entry["annotations"]
        union_anns = [a for a in anns if a.get("category_id") in union_cat_ids]
        other_anns = [a for a in anns if a.get("category_id") not in union_cat_ids]

        merged_entry = dict(entry)

        if union_anns:
            merged_bbox = _union_bbox([a["bbox"] for a in union_anns])
            merged_seg = _union_segmentation(union_anns)
            merged_ann: dict[str, Any] = {
                "id": -1,
                "bbox_id": "union-0",
                "category_id": 99999,  # palette index 3 → orange
                "category_name": new_label,
                "bbox": merged_bbox,
                "segmentation": merged_seg,
                "area": float(merged_bbox[2] * merged_bbox[3]),
                "iscrowd": 0,
            }
            merged_entry["annotations"] = other_anns + [merged_ann]
        else:
            merged_entry["annotations"] = other_anns

        preview.append(merged_entry)
    return preview


# ═════════════════════════════════════════════════════════════════════════════
# Page header
# ═════════════════════════════════════════════════════════════════════════════

st.title("Label Refinement")
st.caption(
    "**Divide into Subclasses** — cluster a category's crops and assign precise sub-labels.  \n"
    "**Union to Superclass** — merge selected categories into a compound object with a new label."
)
st.divider()

# ── 1. Dataset + source file ──────────────────────────────────────────────────

st.subheader("1 · Dataset")

all_datasets = discover_datasets()
datasets_with_anns = [
    d for d in all_datasets
    if (DATA_DIR / d / "annotations" / "base_detection.json").exists()
    or (DATA_DIR / d / "annotations" / "exact_detection.json").exists()
]

if not datasets_with_anns:
    st.warning(
        "No annotated datasets found. Run **Base Class Detection** first."
    )
    st.stop()

selected_dataset = st.selectbox("Dataset", options=datasets_with_anns)
dataset_dir = DATA_DIR / selected_dataset
frames_dir = dataset_dir / IMAGE_ROOT

_exact = dataset_dir / "annotations" / "exact_detection.json"
_base = dataset_dir / "annotations" / "base_detection.json"
_source_opts = [p.name for p in [_exact, _base] if p.exists()]

ann_source = st.selectbox(
    "Source annotation file",
    options=_source_opts,
    help="File to read from. Output is always written to exact_detection.json.",
)
ann_path = dataset_dir / "annotations" / ann_source

coco = json.loads(ann_path.read_text())
img_map: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
ann_map: dict[int, dict] = {ann["id"]: ann for ann in coco.get("annotations", [])}
cat_map: dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}

st.caption(
    f"`{ann_path.relative_to(DATA_DIR)}` — "
    f"{len(img_map)} images · "
    f"{len(ann_map)} annotations · "
    f"{len(cat_map)} categories"
)

# ── Overview gallery ──────────────────────────────────────────────────────────

st.divider()

_file_sig = ann_path.stat().st_mtime
gallery = _load_gallery(str(ann_path), str(frames_dir), _file_sig)

_annotated = [e for e in gallery if e["annotations"]]
_has_masks = any(
    ann.get("segmentation")
    for e in _annotated
    for ann in e["annotations"]
)

if _annotated:
    st.caption(
        f"{len(gallery)} images · {len(_annotated)} with annotations · "
        f"{sum(len(e['annotations']) for e in _annotated)} bounding boxes"
        + (" · segmentation masks present" if _has_masks else "")
    )
    with st.container(height=480):
        _gcols = st.columns(GALLERY_COLS)
        for _i, _entry in enumerate(_annotated):
            _col = _gcols[_i % GALLERY_COLS]
            try:
                _rend = (
                    draw_detections_with_masks(
                        _entry["image_path"], _entry["annotations"], _entry["img_idx"]
                    )
                    if _has_masks
                    else draw_detections(
                        _entry["image_path"], _entry["annotations"], _entry["img_idx"]
                    )
                )
                _col.image(
                    _rend,
                    caption=(
                        f"#{_entry['img_idx']} · {_entry['file_name']} · "
                        f"{len(_entry['annotations'])} ann"
                    ),
                    use_container_width=True,
                )
            except Exception as _exc:
                _col.warning(f"Could not render {_entry['file_name']}: {_exc}")
else:
    st.info("No annotations found in this file.")

st.divider()

# ── Mode selector ─────────────────────────────────────────────────────────────

mode_label = st.radio(
    "Mode",
    options=["Divide into Subclasses", "Union to Superclass"],
    horizontal=True,
    index=0 if st.session_state.p3_mode == "divide" else 1,
)
st.session_state.p3_mode = "divide" if mode_label == "Divide into Subclasses" else "union"

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# DIVIDE INTO SUBCLASSES
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.p3_mode == "divide":

    # ── 2. Category ───────────────────────────────────────────────────────────

    st.subheader("2 · Category to Divide")

    all_cat_names = sorted(cat_map.values())
    if not all_cat_names:
        st.warning("No categories in this annotation file.")
        st.stop()

    if st.session_state.p3_selected_cat not in all_cat_names:
        st.session_state.p3_selected_cat = all_cat_names[0]

    selected_cat = st.selectbox(
        "Category",
        options=all_cat_names,
        index=all_cat_names.index(st.session_state.p3_selected_cat),
        help="All annotations of this category will be embedded and clustered into sub-classes.",
    )
    st.session_state.p3_selected_cat = selected_cat

    selected_cat_ids = {cid for cid, name in cat_map.items() if name == selected_cat}
    filtered_anns = [
        ann for ann in coco.get("annotations", [])
        if ann.get("category_id") in selected_cat_ids
    ]
    st.caption(f"{len(filtered_anns)} annotation(s) in category **{selected_cat}**.")

    st.divider()

    # ── 3. Crop embedding ─────────────────────────────────────────────────────

    st.subheader("3 · Crop Embedding")

    embeddings_ready = (
        st.session_state.p3_embeddings is not None
        and st.session_state.p3_embed_dataset == selected_dataset
        and st.session_state.p3_embed_category == selected_cat
        and st.session_state.p3_ann_ids is not None
        and len(st.session_state.p3_ann_ids) == len(filtered_anns)
    )

    if embeddings_ready:
        st.success(
            f"Embeddings ready — "
            f"{len(st.session_state.p3_ann_ids)} crops × "
            f"{st.session_state.p3_embeddings.shape[1]}d"
        )

    if st.button(
        "Run Embedding",
        disabled=not bool(filtered_anns),
        help="Embed every crop of the selected category with DINOv2 ViT-B/14.",
    ):
        crops: list[Image.Image] = []
        ann_ids: list[int] = []
        prog = st.progress(0.0, text="Loading crops…")
        total = len(filtered_anns)

        for i, ann in enumerate(filtered_anns):
            img_info = img_map.get(ann["image_id"])
            if img_info is None:
                continue
            ipath = dataset_dir / IMAGE_ROOT / img_info["file_name"]
            if not ipath.exists():
                continue
            try:
                pil = Image.open(ipath).convert("RGB")
                x, y, w, h = ann["bbox"]
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(pil.width, int(x + w)), min(pil.height, int(y + h))
                if x2 > x1 and y2 > y1:
                    crops.append(pil.crop((x1, y1, x2, y2)))
                    ann_ids.append(ann["id"])
            except Exception:
                pass
            if (i + 1) % 50 == 0 or i == total - 1:
                prog.progress((i + 1) / total, text=f"Crop {i + 1}/{total}")

        prog.empty()

        if not crops:
            st.error("No valid crops found.")
        else:
            with st.spinner(f"Embedding {len(crops)} crops with DINOv2…"):
                embs = embed_crops(crops, batch_size=64)

            st.session_state.p3_embeddings = embs
            st.session_state.p3_ann_ids = ann_ids
            st.session_state.p3_embed_dataset = selected_dataset
            st.session_state.p3_embed_category = selected_cat
            st.session_state.p3_cluster_labels = None
            st.session_state.p3_prototypes = None
            st.session_state.p3_cluster_dataset = None
            st.session_state.p3_decisions = None
            st.session_state.p3_confirmed = None
            st.success(f"Embedded {len(crops)} crops → shape {embs.shape}")
            st.rerun()

    st.divider()

    # ── 4. Clustering ─────────────────────────────────────────────────────────

    st.subheader("4 · Clustering")

    cluster_algo = st.radio("Algorithm", ["HDBSCAN", "KMeans"], horizontal=True)
    col_param, _ = st.columns([1, 2])
    with col_param:
        if cluster_algo == "HDBSCAN":
            min_cluster_size = st.number_input(
                "Min cluster size", min_value=2, max_value=500, value=5, step=1
            )
            n_clusters_param = None
        else:
            n_clusters_param = st.number_input(
                "Number of clusters", min_value=2, max_value=200, value=10, step=1
            )
            min_cluster_size = None

    clustering_ready = (
        st.session_state.p3_cluster_labels is not None
        and st.session_state.p3_cluster_dataset == selected_dataset
    )

    if clustering_ready:
        _lbl = st.session_state.p3_cluster_labels
        st.success(
            f"Clustering ready — "
            f"{len(np.unique(_lbl[_lbl >= 0]))} clusters · "
            f"{int(np.sum(_lbl >= 0))} assigned · "
            f"{int(np.sum(_lbl == -1))} noise"
        )

    if st.button("Run Clustering", disabled=not embeddings_ready):
        embs = st.session_state.p3_embeddings
        with st.spinner("Clustering…"):
            if cluster_algo == "HDBSCAN":
                labels = _run_hdbscan(embs, int(min_cluster_size))
            else:
                labels = _run_kmeans(embs, int(n_clusters_param))
            prototypes = _compute_prototypes(embs, labels)

        st.session_state.p3_cluster_labels = labels
        st.session_state.p3_prototypes = prototypes
        st.session_state.p3_cluster_dataset = selected_dataset
        st.session_state.p3_decisions = None
        st.session_state.p3_confirmed = None
        st.success(
            f"Found {len(prototypes)} clusters "
            f"({int(np.sum(labels == -1))} noise points)."
        )
        st.rerun()

    st.divider()

    # ── 5. Cluster review ─────────────────────────────────────────────────────

    st.subheader("5 · Cluster Review")

    if not clustering_ready:
        st.info("Run clustering (step 4) to review clusters here.")
    else:
        labels_arr: np.ndarray = st.session_state.p3_cluster_labels
        prototypes_map: dict[int, int] = st.session_state.p3_prototypes
        ann_ids_list: list[int] = st.session_state.p3_ann_ids
        filtered_ann_map = {ann["id"]: ann for ann in filtered_anns}

        if st.session_state.p3_decisions is None:
            st.session_state.p3_decisions = {
                str(cid): "keep" for cid in prototypes_map
            }
        if st.session_state.p3_cluster_names is None:
            st.session_state.p3_cluster_names = {}

        decisions: dict[str, str] = st.session_state.p3_decisions
        cluster_names: dict[str, str] = st.session_state.p3_cluster_names

        n_noise = int(np.sum(labels_arr == -1))
        if n_noise:
            st.caption(f"{n_noise} noise point(s) will be excluded from export.")

        max_gallery_crops = st.number_input(
            "Max images per cluster", min_value=4, max_value=200, value=20, step=4
        )
        thumb_px = 120

        st.markdown(
            "**Assign a sub-class label to each cluster, then keep or discard:**"
        )

        for cid in sorted(prototypes_map.keys()):
            proto_idx = prototypes_map[cid]
            proto_ann = filtered_ann_map.get(ann_ids_list[proto_idx])
            cluster_size = int(np.sum(labels_arr == cid))
            current_decision = decisions.get(str(cid), "keep")
            current_name = cluster_names.get(str(cid), "")

            status_icon = "✅" if current_decision == "keep" else "🗑️"
            label_hint = f" → _{current_name}_" if current_name else ""

            with st.expander(
                f"{status_icon} **Cluster {cid}** · {cluster_size} item(s){label_hint}",
                expanded=False,
            ):
                new_name = st.text_input(
                    "Sub-class label",
                    value=current_name,
                    placeholder=f"e.g. {selected_cat} — variant A",
                    key=f"cname_{cid}",
                    help=(
                        "Name for this sub-class. "
                        "Leave blank to keep the original category name."
                    ),
                )
                if new_name != current_name:
                    st.session_state.p3_cluster_names[str(cid)] = new_name
                    st.session_state.p3_confirmed = None

                # Prototype image (zoomed crop with bbox drawn)
                if proto_ann is not None:
                    proto_img_info = img_map.get(proto_ann["image_id"])
                    if proto_img_info:
                        proto_path = (
                            dataset_dir / IMAGE_ROOT / proto_img_info["file_name"]
                        )
                        try:
                            proto_thumb = _crop_with_bbox_drawn(
                                proto_path,
                                proto_ann["bbox"],
                                proto_ann.get("segmentation"),
                            )
                            st.image(
                                proto_thumb,
                                caption="Prototype (cluster centroid)",
                                width=280,
                            )
                        except Exception as exc:
                            st.warning(f"Could not render prototype: {exc}")

                # Crop thumbnails for all cluster members
                st.markdown(
                    f"**All images in this cluster** (up to {max_gallery_crops}):"
                )
                cluster_crops = _load_cluster_crops(
                    cid, labels_arr, ann_ids_list, filtered_ann_map,
                    img_map, dataset_dir,
                    max_crops=max_gallery_crops, thumb_size=thumb_px,
                )
                if not cluster_crops:
                    st.caption("No crops available.")
                else:
                    cols_per_row = max(1, min(10, 800 // thumb_px))
                    for row_start in range(0, len(cluster_crops), cols_per_row):
                        row = cluster_crops[row_start: row_start + cols_per_row]
                        rcols = st.columns(len(row))
                        for rc, (thumb, fname) in zip(rcols, row):
                            rc.image(thumb, caption=fname, use_container_width=False)
                    if (
                        len(cluster_crops) == max_gallery_crops
                        and cluster_size > max_gallery_crops
                    ):
                        st.caption(
                            f"Showing {max_gallery_crops} of {cluster_size} — "
                            "increase 'Max images per cluster' to see more."
                        )

                st.markdown("---")
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button(
                        "✅ Keep",
                        key=f"keep_{cid}",
                        type="primary" if current_decision == "keep" else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.p3_decisions[str(cid)] = "keep"
                        st.session_state.p3_confirmed = None
                        st.rerun()
                with bc2:
                    if st.button(
                        "🗑️ Discard",
                        key=f"discard_{cid}",
                        type=(
                            "primary" if current_decision == "discard" else "secondary"
                        ),
                        use_container_width=True,
                    ):
                        st.session_state.p3_decisions[str(cid)] = "discard"
                        st.session_state.p3_confirmed = None
                        st.rerun()

        st.divider()

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

    # ── 6. Export ─────────────────────────────────────────────────────────────

    st.subheader("6 · Export")

    if not st.session_state.p3_confirmed:
        st.info("Confirm cluster decisions (step 5) to enable export.")
    else:
        labels_arr = st.session_state.p3_cluster_labels
        ann_ids_list = st.session_state.p3_ann_ids
        decisions = st.session_state.p3_decisions
        cluster_names = st.session_state.p3_cluster_names or {}
        filtered_ann_map = {ann["id"]: ann for ann in filtered_anns}

        kept_cluster_ids = {int(cid) for cid, v in decisions.items() if v == "keep"}

        ann_id_to_cluster: dict[int, int] = {
            aid: int(labels_arr[i])
            for i, aid in enumerate(ann_ids_list)
            if labels_arr[i] in kept_cluster_ids
        }

        # Build new sub-class categories
        orig_cat_by_id = {c["id"]: c for c in coco.get("categories", [])}
        next_cat_id = (
            max((c["id"] for c in coco.get("categories", [])), default=0) + 1
        )
        cluster_to_cat_id: dict[int, int] = {}
        new_divide_cats: list[dict] = []
        seen_cat_ids: set[int] = set()

        for cid in sorted(kept_cluster_ids):
            precise = cluster_names.get(str(cid), "").strip()
            if precise:
                new_divide_cats.append(
                    {"id": next_cat_id, "name": precise, "supercategory": selected_cat}
                )
                cluster_to_cat_id[cid] = next_cat_id
                next_cat_id += 1
            else:
                pi = st.session_state.p3_prototypes.get(cid)
                pa = filtered_ann_map.get(ann_ids_list[pi]) if pi is not None else None
                orig_cat_id = pa.get("category_id") if pa else None
                if orig_cat_id and orig_cat_id in orig_cat_by_id:
                    cluster_to_cat_id[cid] = orig_cat_id
                    if orig_cat_id not in seen_cat_ids:
                        new_divide_cats.append(orig_cat_by_id[orig_cat_id])
                        seen_cat_ids.add(orig_cat_id)

        # Pass through annotations from other categories
        other_cat_ids = {cid for cid, name in cat_map.items() if name != selected_cat}
        passthrough_anns = [
            ann for ann in coco.get("annotations", [])
            if ann.get("category_id") in other_cat_ids
        ]
        passthrough_cat_ids = {ann["category_id"] for ann in passthrough_anns}
        passthrough_cats = [
            c for c in coco.get("categories", []) if c["id"] in passthrough_cat_ids
        ]

        # Relabelled annotations from the divided category
        divided_anns = []
        for ann in coco.get("annotations", []):
            if ann["id"] not in ann_id_to_cluster:
                continue
            new_cat_id = cluster_to_cat_id.get(ann_id_to_cluster[ann["id"]])
            if new_cat_id is None:
                continue
            ann_copy = dict(ann)
            ann_copy["category_id"] = new_cat_id
            divided_anns.append(ann_copy)

        all_export_anns = passthrough_anns + divided_anns
        all_export_cats = passthrough_cats + new_divide_cats
        kept_img_ids = {ann["image_id"] for ann in all_export_anns}
        export_images = [
            img for img in coco.get("images", []) if img["id"] in kept_img_ids
        ]

        exact_coco: dict[str, Any] = {
            "info": {
                **(coco.get("info") or {}),
                "description": "Label refinement — divide into subclasses",
            },
            "licenses": coco.get("licenses", []),
            "images": export_images,
            "annotations": all_export_anns,
            "categories": all_export_cats,
        }

        out_path = dataset_dir / "annotations" / "exact_detection.json"
        col_save, col_dl = st.columns(2)
        with col_save:
            if st.button("Save exact_detection.json", type="primary"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(exact_coco, indent=2))
                _load_gallery.clear()
                st.success(f"Saved → `{out_path.relative_to(DATA_DIR)}`")
        with col_dl:
            st.download_button(
                "Download JSON",
                data=json.dumps(exact_coco, indent=2).encode(),
                file_name="exact_detection.json",
                mime="application/json",
                use_container_width=True,
            )

        n_precise = sum(
            1 for cid in kept_cluster_ids if cluster_names.get(str(cid), "").strip()
        )
        st.markdown(
            f"**Stats:** {len(ann_map)} annotations in → "
            f"**{len(all_export_anns)}** out "
            f"({len(passthrough_anns)} pass-through + {len(divided_anns)} divided). "
            f"{n_precise} cluster(s) assigned a new sub-class label."
        )

        if cluster_names:
            named = {k: v for k, v in cluster_names.items() if v.strip()}
            if named:
                with st.expander("Sub-class label assignments", expanded=False):
                    import pandas as pd

                    rows = []
                    for cid_str, name in sorted(named.items(), key=lambda x: int(x[0])):
                        cid_int = int(cid_str)
                        pi = st.session_state.p3_prototypes.get(cid_int)
                        pa = (
                            filtered_ann_map.get(ann_ids_list[pi])
                            if pi is not None
                            else None
                        )
                        orig = cat_map.get(pa.get("category_id", -1), "?") if pa else "?"
                        rows.append(
                            {
                                "Cluster": cid_int,
                                "Original class": orig,
                                "Sub-class label": name,
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# UNION TO SUPERCLASS
# ═════════════════════════════════════════════════════════════════════════════

else:
    # Clear preview when switching dataset
    if st.session_state.p3_union_dataset != selected_dataset:
        st.session_state.p3_union_preview = None
        st.session_state.p3_union_cats = []
        st.session_state.p3_union_label = ""
        st.session_state.p3_union_dataset = selected_dataset

    all_cat_names = sorted(cat_map.values())
    if not all_cat_names:
        st.warning("No categories found in this annotation file.")
        st.stop()

    # ── 2. Category + label selection ─────────────────────────────────────────

    st.subheader("2 · Select Categories to Union")

    st.markdown(
        "Select two or more **class names**. For every image, all annotations of "
        "those classes are fused into a single bounding box and segmentation mask "
        "under the new super-label."
    )

    selected_union_cats = st.multiselect(
        "Categories to merge",
        options=all_cat_names,
        default=[
            c for c in st.session_state.p3_union_cats if c in all_cat_names
        ],
        help="All instances of these classes across all images will be merged per image.",
    )
    st.session_state.p3_union_cats = selected_union_cats

    new_label = st.text_input(
        "New super-class label",
        value=st.session_state.p3_union_label,
        placeholder="e.g. Fastener Assembly",
        help="Label assigned to every merged annotation.",
    )
    st.session_state.p3_union_label = new_label

    union_cat_ids: set[int] = {
        cid for cid, name in cat_map.items() if name in selected_union_cats
    }

    # Count how many annotations / images will be affected
    affected_ann_ids: set[int] = {
        ann["id"]
        for ann in coco.get("annotations", [])
        if ann.get("category_id") in union_cat_ids
    }
    affected_img_ids: set[int] = {
        ann["image_id"]
        for ann in coco.get("annotations", [])
        if ann["id"] in affected_ann_ids
    }

    if selected_union_cats and new_label.strip():
        st.caption(
            f"**{len(affected_ann_ids)}** annotation(s) across "
            f"**{len(affected_img_ids)}** image(s) will be merged into "
            f"**{len(affected_img_ids)}** new '{new_label.strip()}' annotation(s)."
        )

    can_preview = len(selected_union_cats) >= 1 and bool(new_label.strip())

    if not can_preview:
        if not selected_union_cats:
            st.info("Select at least one category to continue.")
        else:
            st.info("Enter a super-class label to continue.")

    if st.button(
        "Preview Union Result",
        type="primary",
        disabled=not can_preview,
    ):
        st.session_state.p3_union_preview = _build_union_preview_entries(
            gallery, union_cat_ids, new_label.strip()
        )
        # Clear the class-filter widget state so it resets to "all selected"
        # for the newly computed preview (Streamlit ignores `default` once a
        # key already exists in session state, so we must clear it manually).
        st.session_state.pop("p3_preview_cat_filter", None)
        st.rerun()

    st.divider()

    # ── 3. Preview gallery ────────────────────────────────────────────────────

    st.subheader("3 · Preview")

    preview: list[dict[str, Any]] | None = st.session_state.p3_union_preview

    if preview is None:
        st.info(
            "Click **Preview Union Result** above to see what the merged "
            "annotations will look like."
        )
    else:
        # Only show images where a merge actually happened
        preview_affected = [
            e for e in preview
            if any(a.get("category_id") == 99999 for a in e["annotations"])
        ]

        if not preview_affected:
            st.warning(
                "No images contain annotations of the selected categories — "
                "nothing to merge."
            )
        else:
            # ── Filter controls ───────────────────────────────────────────────
            # Collect every category_name that appears in the preview
            _all_preview_cats: list[str] = sorted({
                a.get("category_name", "?")
                for e in preview_affected
                for a in e["annotations"]
                if a.get("category_name")
            })

            _fc1, _fc2, _fc3 = st.columns([3, 1, 1])
            with _fc1:
                _visible_cats = st.multiselect(
                    "Show classes",
                    options=_all_preview_cats,
                    default=_all_preview_cats,
                    help=(
                        "Filter which annotation classes are drawn on the preview. "
                        f"The new merged class appears as '{new_label.strip()}'."
                    ),
                    key="p3_preview_cat_filter",
                )
            with _fc2:
                _show_bbox = st.checkbox(
                    "Bounding boxes", value=True, key="p3_preview_show_bbox"
                )
            with _fc3:
                _show_seg = st.checkbox(
                    "Segmentation masks", value=True, key="p3_preview_show_seg"
                )

            _visible_cat_set = set(_visible_cats)

            st.caption(
                f"Showing **{len(preview_affected)}** image(s) with merged annotations "
                f"(orange = new '{new_label.strip()}')."
            )
            with st.container(height=520):
                _pcols = st.columns(GALLERY_COLS)
                for _pi, _pentry in enumerate(preview_affected):
                    _pcol = _pcols[_pi % GALLERY_COLS]
                    try:
                        # Apply class filter
                        _visible_anns = [
                            a for a in _pentry["annotations"]
                            if a.get("category_name", "?") in _visible_cat_set
                        ]
                        _pimg = _render_preview_image(
                            _pentry["image_path"],
                            _visible_anns,
                            _pentry["img_idx"],
                            show_bbox=_show_bbox,
                            show_seg=_show_seg,
                        )
                        _merged_count = sum(
                            1
                            for a in _pentry["annotations"]
                            if a.get("category_id") == 99999
                        )
                        _pcol.image(
                            _pimg,
                            caption=(
                                f"{_pentry['file_name']} · "
                                f"{len(_visible_anns)} visible "
                                f"({_merged_count} merged)"
                            ),
                            use_container_width=True,
                        )
                    except Exception as _exc:
                        _pcol.warning(
                            f"Could not render {_pentry['file_name']}: {_exc}"
                        )

        st.divider()

        # ── 4. Apply ──────────────────────────────────────────────────────────

        st.subheader("4 · Apply to exact_detection.json")

        # Build the export COCO structure
        next_cat_id = (
            max((c["id"] for c in coco.get("categories", [])), default=0) + 1
        )
        label_to_cat_id: dict[str, int] = {}
        new_union_cats: list[dict] = []

        # Add new category for the super-label
        _super_lbl = new_label.strip()
        label_to_cat_id[_super_lbl] = next_cat_id
        new_union_cats.append(
            {"id": next_cat_id, "name": _super_lbl, "supercategory": ""}
        )
        next_cat_id += 1

        # Pass through annotations not consumed by the union
        passthrough_anns_u = [
            ann for ann in coco.get("annotations", [])
            if ann["id"] not in affected_ann_ids
        ]
        passthrough_cat_ids_u = {ann["category_id"] for ann in passthrough_anns_u}
        passthrough_cats_u = [
            c for c in coco.get("categories", [])
            if c["id"] in passthrough_cat_ids_u
        ]

        # Build one merged annotation per affected image
        next_ann_id = (
            max((ann["id"] for ann in coco.get("annotations", [])), default=0) + 1
        )
        merged_anns_u: list[dict] = []
        for img_id in sorted(affected_img_ids):
            img_anns = [
                ann for ann in coco.get("annotations", [])
                if ann["image_id"] == img_id
                and ann["id"] in affected_ann_ids
            ]
            if not img_anns:
                continue
            merged_bbox = _union_bbox([a["bbox"] for a in img_anns])
            merged_seg = _union_segmentation(img_anns)
            merged_anns_u.append(
                {
                    "id": next_ann_id,
                    "image_id": img_id,
                    "category_id": label_to_cat_id[_super_lbl],
                    "bbox": merged_bbox,
                    "segmentation": merged_seg,
                    "area": float(merged_bbox[2] * merged_bbox[3]),
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

        all_export_anns_u = passthrough_anns_u + merged_anns_u
        all_export_cats_u = passthrough_cats_u + new_union_cats
        kept_img_ids_u = {ann["image_id"] for ann in all_export_anns_u}
        export_images_u = [
            img for img in coco.get("images", []) if img["id"] in kept_img_ids_u
        ]

        exact_coco_u: dict[str, Any] = {
            "info": {
                **(coco.get("info") or {}),
                "description": "Label refinement — union to superclass",
            },
            "licenses": coco.get("licenses", []),
            "images": export_images_u,
            "annotations": all_export_anns_u,
            "categories": all_export_cats_u,
        }

        st.markdown(
            f"**Stats:** {len(ann_map)} annotations in → "
            f"**{len(all_export_anns_u)}** out "
            f"({len(passthrough_anns_u)} pass-through + {len(merged_anns_u)} merged). "
            f"{len(affected_ann_ids)} source annotation(s) replaced."
        )

        out_path_u = dataset_dir / "annotations" / "exact_detection.json"
        col_save_u, col_dl_u = st.columns(2)
        with col_save_u:
            if st.button(
                "Apply — Save exact_detection.json",
                type="primary",
                key="union_save_btn",
            ):
                out_path_u.parent.mkdir(parents=True, exist_ok=True)
                out_path_u.write_text(json.dumps(exact_coco_u, indent=2))
                _load_gallery.clear()
                st.session_state.p3_union_preview = None
                st.success(f"Saved → `{out_path_u.relative_to(DATA_DIR)}`")
                st.rerun()
        with col_dl_u:
            st.download_button(
                "Download JSON",
                data=json.dumps(exact_coco_u, indent=2).encode(),
                file_name="exact_detection.json",
                mime="application/json",
                use_container_width=True,
                key="union_dl_btn",
            )
