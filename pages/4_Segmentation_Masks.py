"""Page 4 — Segmentation Mask Generation using SAM2.

Pipeline:
  1. Dataset selector (datasets with exact_detection.json)
  2. Model selector + checkpoint download
  3. Run segmentation (bbox-prompted SAM2)
  4. Preview gallery with mask overlays
  5. Export to segmentation.json (COCO RLE format)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image

from src.common import DATA_DIR, IMAGE_ROOT, discover_datasets

st.set_page_config(page_title="Segmentation Masks", page_icon="🎭", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = DATA_DIR.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

MODELS: dict[str, dict[str, str]] = {
    "Tiny": {
        "config": "sam2.1_hiera_t.yaml",
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "url": "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt",
    },
    "Small": {
        "config": "sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
        "url": "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt",
    },
    "Base+": {
        "config": "sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "url": "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt",
    },
    "Large": {
        "config": "sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
        "url": "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt",
    },
}

PALETTE = [
    (220, 60,  60),
    (60,  180, 75),
    (67,  99,  216),
    (255, 165, 0),
    (145, 30,  180),
    (70,  240, 240),
    (240, 50,  230),
    (210, 245, 60),
    (250, 190, 212),
    (0,   128, 128),
    (220, 190, 255),
    (170, 110, 40),
]

# ── Session state defaults ─────────────────────────────────────────────────────

for _key, _default in (
    ("p4_results", None),       # list[dict] — per-annotation results after segmentation
    ("p4_seg_dataset", None),   # str — dataset name when segmentation was run
    ("p4_seg_model", None),     # str — model name when segmentation was run
):
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_coco(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _build_id_maps(coco: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    img_map = {img["id"]: img for img in coco.get("images", [])}
    ann_map = {ann["id"]: ann for ann in coco.get("annotations", [])}
    return img_map, ann_map


def _image_path(dataset_dir: Path, img_info: dict) -> Path:
    return dataset_dir / IMAGE_ROOT / img_info["file_name"]


def _checkpoint_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / MODELS[model_name]["checkpoint"]


def _color_for_category(cat_id: int) -> tuple[int, int, int]:
    return PALETTE[cat_id % len(PALETTE)]


@st.cache_resource(show_spinner="Loading SAM2 model…")
def _load_sam2(config: str, checkpoint: str):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def _run_sam2_on_image(
    predictor,
    image: Image.Image,
    bboxes: list[list[float]],
) -> list[tuple[np.ndarray, float]]:
    """Run SAM2 for a list of bbox prompts on a single image.

    Returns list of (best_mask [H,W bool], score) parallel to bboxes.
    """
    import torch

    img_array = np.array(image.convert("RGB"))
    predictor.set_image(img_array)

    results: list[tuple[np.ndarray, float]] = []
    for bbox in bboxes:
        x, y, w, h = bbox
        box_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                box=box_xyxy,
                multimask_output=True,
            )
        best_idx = int(np.argmax(scores))
        results.append((masks[best_idx].astype(bool), float(scores[best_idx])))
    return results


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a boolean mask (H, W) as COCO RLE."""
    from pycocotools import mask as mask_util

    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(fortran_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _overlay_masks(image: Image.Image, mask_data: list[dict], alpha: float = 0.4) -> Image.Image:
    from PIL import ImageFilter
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for item in mask_data:
        mask: np.ndarray = item["mask"]
        r, g, b = _color_for_category(item["cat_id"])
        fill_arr = np.zeros((*mask.shape, 4), dtype=np.uint8)
        fill_arr[mask] = [r, g, b, int(255 * alpha)]
        overlay = Image.alpha_composite(overlay, Image.fromarray(fill_arr, "RGBA"))
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
        edge = np.array(mask_img.filter(ImageFilter.MaxFilter(3))) - np.array(mask_img)
        outline_arr = np.zeros((*mask.shape, 4), dtype=np.uint8)
        outline_arr[edge > 0] = [r, g, b, 220]
        overlay = Image.alpha_composite(overlay, Image.fromarray(outline_arr, "RGBA"))
    return Image.alpha_composite(base, overlay).convert("RGB")


# ── Title ──────────────────────────────────────────────────────────────────────

st.title("Segmentation Mask Generation")
st.caption(
    "Generate pixel-level segmentation masks for detected objects using SAM2, "
    "guided by bounding boxes from exact-class detection."
)
st.divider()

# ─── 1. Dataset selector ──────────────────────────────────────────────────────

st.subheader("1 · Dataset")

all_datasets = discover_datasets()
datasets_with_exact = [
    d for d in all_datasets
    if (DATA_DIR / d / "annotations" / "exact_detection.json").exists()
]

if not datasets_with_exact:
    st.warning(
        "No datasets with `exact_detection.json` found. "
        "Run **Exact Class Detection** first."
    )
    st.stop()

selected_dataset = st.selectbox("Dataset", options=datasets_with_exact)
dataset_dir = DATA_DIR / selected_dataset
ann_path = dataset_dir / "annotations" / "exact_detection.json"

coco = _load_coco(ann_path)
img_map, ann_map = _build_id_maps(coco)
cat_map: dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}

st.caption(
    f"`{ann_path.relative_to(DATA_DIR)}` — "
    f"{len(img_map)} images · "
    f"{len(ann_map)} annotations · "
    f"{len(cat_map)} categories"
)

st.divider()

# ─── 2. Model selector ────────────────────────────────────────────────────────

st.subheader("2 · SAM2 Model")

selected_model = st.radio(
    "Model size",
    options=list(MODELS.keys()),
    horizontal=True,
    help="Larger models are more accurate but require more memory and time.",
)

model_info = MODELS[selected_model]
ckpt_path = _checkpoint_path(selected_model)
ckpt_exists = ckpt_path.exists()

st.text_input(
    "Checkpoint path",
    value=str(ckpt_path),
    disabled=True,
    label_visibility="collapsed",
)

if ckpt_exists:
    st.success(f"Checkpoint present ({ckpt_path.stat().st_size / 1e6:.0f} MB)")
else:
    st.warning("Checkpoint not downloaded yet.")
    if st.button("Download Checkpoint", type="primary"):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        url = model_info["url"]
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            progress_bar = st.progress(0.0, text="Downloading…")
            with open(ckpt_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total
                            mb = downloaded / 1e6
                            progress_bar.progress(
                                pct,
                                text=f"Downloading… {mb:.0f} / {total / 1e6:.0f} MB",
                            )
            progress_bar.empty()
            st.success("Download complete!")
            st.rerun()
        except Exception as exc:
            st.error(f"Download failed: {exc}")
            if ckpt_path.exists():
                ckpt_path.unlink()

st.divider()

# ─── 3. Run segmentation ──────────────────────────────────────────────────────

st.subheader("3 · Run Segmentation")

seg_ready = (
    st.session_state.p4_results is not None
    and st.session_state.p4_seg_dataset == selected_dataset
    and st.session_state.p4_seg_model == selected_model
)

if seg_ready:
    st.success(
        f"Segmentation ready: {len(st.session_state.p4_results)} masks "
        f"for **{selected_dataset}** with **{selected_model}** model."
    )

run_seg = st.button(
    "Run Segmentation",
    type="primary",
    disabled=not ckpt_exists,
    help="Download the checkpoint first." if not ckpt_exists else "",
)

if run_seg:
    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    predictor = _load_sam2(model_info["config"], str(ckpt_path))

    results: list[dict] = []
    total_images = len(anns_by_image)
    progress_bar = st.progress(0.0, text="Processing images…")
    status_text = st.empty()

    for img_idx, (image_id, anns) in enumerate(anns_by_image.items()):
        img_info = img_map.get(image_id)
        if img_info is None:
            continue

        img_path = _image_path(dataset_dir, img_info)
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        bboxes = [ann["bbox"] for ann in anns]
        status_text.text(
            f"Image {img_idx + 1}/{total_images}: {img_info['file_name']} "
            f"({len(bboxes)} annotations)"
        )

        try:
            mask_results = _run_sam2_on_image(predictor, image, bboxes)
        except Exception as exc:
            st.warning(f"SAM2 failed on {img_info['file_name']}: {exc}")
            continue

        for ann, (mask, score) in zip(anns, mask_results):
            results.append({
                "ann_id": ann["id"],
                "image_id": image_id,
                "category_id": ann.get("category_id", 0),
                "bbox": ann["bbox"],
                "mask": mask,
                "score": score,
            })

        progress_bar.progress(
            (img_idx + 1) / total_images,
            text=f"Processed {img_idx + 1}/{total_images} images…",
        )

    progress_bar.empty()
    status_text.empty()

    st.session_state.p4_results = results
    st.session_state.p4_seg_dataset = selected_dataset
    st.session_state.p4_seg_model = selected_model
    st.success(f"Segmentation complete: {len(results)} masks generated.")
    st.rerun()

st.divider()

# ─── 4. Preview gallery ───────────────────────────────────────────────────────

st.subheader("4 · Preview Gallery")

if not seg_ready:
    st.info("Run segmentation (step 3) to preview masks here.")
else:
    results_list: list[dict] = st.session_state.p4_results

    # Group results by image_id
    results_by_image: dict[int, list[dict]] = {}
    for r in results_list:
        results_by_image.setdefault(r["image_id"], []).append(r)

    image_ids = list(results_by_image.keys())
    total_images = len(image_ids)

    col_n, col_page = st.columns([1, 2])
    with col_n:
        per_page = st.number_input(
            "Images per page", min_value=1, max_value=20, value=4, step=1
        )
    with col_page:
        n_pages = max(1, (total_images + per_page - 1) // per_page)
        page = st.number_input(
            f"Page (1–{n_pages})", min_value=1, max_value=n_pages, value=1, step=1
        )

    start = (page - 1) * per_page
    page_image_ids = image_ids[start : start + per_page]

    for image_id in page_image_ids:
        img_info = img_map.get(image_id)
        if img_info is None:
            continue
        img_path = _image_path(dataset_dir, img_info)
        if not img_path.exists():
            st.warning(f"Image not found: {img_info['file_name']}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            st.warning(f"Cannot open {img_info['file_name']}: {exc}")
            continue

        mask_data = [
            {"mask": r["mask"], "cat_id": r["category_id"]}
            for r in results_by_image[image_id]
        ]
        composited = _overlay_masks(image, mask_data)

        anns_here = results_by_image[image_id]
        cat_names = ", ".join(
            sorted({cat_map.get(r["category_id"], "?") for r in anns_here})
        )
        st.image(
            composited,
            caption=f"{img_info['file_name']} — {len(anns_here)} mask(s) · {cat_names}",
            use_container_width=True,
        )

st.divider()

# ─── 5. Export ────────────────────────────────────────────────────────────────

st.subheader("5 · Export")

if not seg_ready:
    st.info("Run segmentation (step 3) to enable export.")
else:
    results_list = st.session_state.p4_results

    def _build_segmentation_coco() -> dict[str, Any]:
        seg_annotations = []
        for r in results_list:
            rle = _mask_to_rle(r["mask"])
            seg_annotations.append({
                "id": r["ann_id"],
                "image_id": r["image_id"],
                "category_id": r["category_id"],
                "bbox": r["bbox"],
                "area": float(r["mask"].sum()),
                "iscrowd": 0,
                "segmentation": {"counts": rle["counts"], "size": list(rle["size"])},
                "score": r["score"],
            })
        kept_image_ids = {r["image_id"] for r in results_list}
        return {
            "info": {**(coco.get("info") or {}), "description": "Segmentation masks (SAM2 bbox-prompted)"},
            "licenses": coco.get("licenses", []),
            "images": [img for img in coco.get("images", []) if img["id"] in kept_image_ids],
            "annotations": seg_annotations,
            "categories": coco.get("categories", []),
        }

    with st.spinner("Building COCO JSON…"):
        seg_coco = _build_segmentation_coco()

    out_path = dataset_dir / "annotations" / "segmentation.json"
    json_str = json.dumps(seg_coco, indent=2)
    json_bytes = json_str.encode()

    col_save, col_dl = st.columns(2)
    with col_save:
        if st.button("Save segmentation.json", type="primary"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                fh.write(json_str)
            st.success(f"Saved to `{out_path.relative_to(DATA_DIR)}`")

    with col_dl:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="segmentation.json",
            mime="application/json",
            use_container_width=True,
        )

    # Stats
    n_anns = len(seg_coco["annotations"])
    n_imgs = len(seg_coco["images"])
    n_cats = len(seg_coco["categories"])
    avg_score = float(np.mean([r["score"] for r in results_list])) if results_list else 0.0
    cat_counts: dict[str, int] = {}
    for r in results_list:
        name = cat_map.get(r["category_id"], "?")
        cat_counts[name] = cat_counts.get(name, 0) + 1

    st.markdown(
        f"**Stats:** {n_anns} annotations · {n_imgs} images · "
        f"{n_cats} categories · avg SAM2 score: {avg_score:.3f}"
    )
    if cat_counts:
        rows = " · ".join(f"`{k}`: {v}" for k, v in sorted(cat_counts.items()))
        st.markdown(f"Per-category: {rows}")
