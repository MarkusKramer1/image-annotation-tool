"""visual_detection_runner.py

Subprocess helper: visual-prompt-based detection.

Supports two backends selected with --backend:

  wedetect (default)
    Given user-supplied exemplar bounding boxes, runs WeDetect-Uni to get
    class-agnostic proposals + embeddings, finds best-matching proposals for
    every drawn box, then does cosine-similarity matching across all frames.

  yoloe
    Uses the YOLO-E few-shot segmentation model (yoloe-11*-seg.pt) via the
    few_shot_object_detection package.  Builds a temporary CVAT-format dataset
    from the visual prompts, computes per-class visual embeddings with
    YOLOEVPSegPredictor, then runs YOLO-E inference on every target frame.

Query JSON format (--query-json):
  [
    {"image_path": "/abs/path/to/frame.jpg", "bbox": [x, y, w, h], "label": "robot"},
    ...
  ]

Progress is emitted as newline-delimited JSON on stdout:
  {"type": "log",   "msg": "…"}
  {"type": "frame", "done": N, "total": T, "n_props": P, "n_candidates": C}
  {"type": "done",  "num_images": N, "num_annotations": A, "counts_per_class": {…},
                    "output_json": "…", "vis_entries": […]}
  {"type": "error", "msg": "…"}
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


# ── Colour palette ─────────────────────────────────────────────────────────────
_COLORS = [
    (255, 80,  80),
    (80,  200, 80),
    (80,  130, 255),
    (255, 180, 0),
    (200, 80,  255),
    (0,   210, 210),
]


# ── Shared helpers ──────────────────────────────────────────────────────────────

def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _emit_log(msg: str) -> None:
    _emit({"type": "log", "msg": msg})


def _iou_one_vs_many(box_xyxy: list[float], boxes_xyxy: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box_xyxy[0], boxes_xyxy[:, 0])
    y1 = np.maximum(box_xyxy[1], boxes_xyxy[:, 1])
    x2 = np.minimum(box_xyxy[2], boxes_xyxy[:, 2])
    y2 = np.minimum(box_xyxy[3], boxes_xyxy[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    area_b = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _xyxy_to_xywh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def _normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-6)


def _nms(
    boxes_xywh: list[list[float]],
    scores: list[float],
    iou_threshold: float,
) -> list[int]:
    """Greedy IoU-based NMS. Returns indices of kept boxes (original ordering)."""
    if not boxes_xywh:
        return []
    boxes_xyxy = np.array([_xywh_to_xyxy(b) for b in boxes_xywh], dtype=np.float32)
    order = np.array(scores, dtype=np.float32).argsort()[::-1]
    kept: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        kept.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = _iou_one_vs_many(boxes_xyxy[i].tolist(), boxes_xyxy[rest])
        order = rest[ious < iou_threshold]
    return kept


def _apply_nms_to_matches(
    matches: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Apply per-frame per-class NMS to a flat list of match dicts."""
    from collections import defaultdict
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(matches):
        groups[(m["file_name"], m["label"])].append(i)

    kept_indices: set[int] = set()
    for group_indices in groups.values():
        boxes  = [matches[i]["bbox"]        for i in group_indices]
        scores = [matches[i]["similarity"]   for i in group_indices]
        for local_k in _nms(boxes, scores, iou_threshold):
            kept_indices.add(group_indices[local_k])

    return [m for i, m in enumerate(matches) if i in kept_indices]


def _load_existing_annotations(
    annotation_json: str,
    max_overlap: float,
) -> dict[str, list[list[float]]]:
    """Load existing COCO annotations for overlap-based duplicate exclusion."""
    existing_by_file: dict[str, list[list[float]]] = {}
    ann_path = Path(annotation_json)
    if ann_path.exists():
        with open(ann_path) as fh:
            existing_coco: dict[str, Any] = json.load(fh)
        id_to_fname = {img["id"]: img["file_name"] for img in existing_coco.get("images", [])}
        for ann in existing_coco.get("annotations", []):
            fname = id_to_fname.get(ann["image_id"])
            if fname:
                existing_by_file.setdefault(fname, []).append(
                    _xywh_to_xyxy(ann["bbox"])
                )
        _emit_log(
            f"Loaded existing annotations from {ann_path.name} for "
            f"duplicate-exclusion (IoU ≥ {max_overlap})."
        )
    return existing_by_file


def _generate_vis(
    coco_out: dict[str, Any],
    fname_to_path: dict[str, Path],
    vis_dir_str: str,
) -> list[dict[str, Any]]:
    """Draw detection boxes on frames and save to vis_dir. Returns vis_entries."""
    if not vis_dir_str:
        return []

    vis_entries: list[dict[str, Any]] = []
    try:
        from PIL import Image as _PilImg, ImageDraw as _ImageDraw, ImageFont as _ImageFont

        vis_dir = Path(vis_dir_str)
        vis_dir.mkdir(parents=True, exist_ok=True)

        try:
            _font = _ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
            )
        except (OSError, IOError):
            _font = _ImageFont.load_default()

        anns_by_file: dict[str, list[dict[str, Any]]] = {}
        for ann in coco_out["annotations"]:
            fname = next(
                (img["file_name"] for img in coco_out["images"] if img["id"] == ann["image_id"]),
                None,
            )
            if fname:
                anns_by_file.setdefault(fname, []).append(ann)

        cat_id_to_name = {c["id"]: c["name"] for c in coco_out["categories"]}

        for fname, anns in anns_by_file.items():
            frame_path = fname_to_path.get(fname)
            if frame_path is None:
                continue
            try:
                img = _PilImg.open(frame_path).convert("RGB")
                draw = _ImageDraw.Draw(img)
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    cat_id = ann["category_id"]
                    label = cat_id_to_name.get(cat_id, "?")
                    score = ann.get("score", 0.0)
                    color = _COLORS[cat_id % len(_COLORS)]
                    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                    text = f"{label}  {score:.2f}"
                    label_y = max(0, int(y) - 15)
                    try:
                        tb = draw.textbbox((int(x), label_y), text, font=_font)
                        draw.rectangle(tb, fill=color)
                        draw.text((int(x), label_y), text, fill="white", font=_font)
                    except Exception:
                        draw.text((int(x), int(y)), text, fill=color)
                vis_path = str(vis_dir / fname)
                img.save(vis_path)
                vis_entries.append({
                    "vis_path":   vis_path,
                    "frame_name": fname,
                    "n_det":      len(anns),
                })
            except Exception:
                pass
    except Exception as vis_exc:
        _emit_log(f"Warning: could not generate vis images: {vis_exc}")

    return vis_entries


# ── WeDetect-Uni backend ────────────────────────────────────────────────────────

def _load_wedetect_model(wedetect_dir: str, checkpoint_path: str):
    import torch

    sys.path.insert(0, wedetect_dir)
    from generate_proposal import SimpleYOLOWorldDetector  # noqa: PLC0415

    ckpt_name = Path(checkpoint_path).stem.lower()
    backbone_size = "large" if "large" in ckpt_name else "base"

    model = SimpleYOLOWorldDetector(
        backbone_size=backbone_size,
        prompt_dim=768,
        num_prompts=256,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    keys = list(checkpoint.keys())
    for key in keys:
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            checkpoint[new_key] = checkpoint.pop(key)

    keys = list(checkpoint.keys())
    for key in keys:
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            checkpoint[new_key] = checkpoint.pop(key)

    msg = model.load_state_dict(checkpoint, strict=False)
    missing = [k for k in msg.missing_keys if "embeddings" not in k]
    if missing:
        _emit_log(f"Warning: {len(missing)} unexpected missing keys when loading model.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


def _run_wedetect(args: argparse.Namespace) -> None:
    """WeDetect-Uni visual-prompt detection pipeline."""
    import torch
    from PIL import Image

    _emit_log("Loading WeDetect-Uni model…")
    model, device = _load_wedetect_model(args.wedetect_dir, args.uni_checkpoint)
    _emit_log(f"Model on {device}. Ready.")

    with open(args.query_json) as fh:
        visual_prompts: list[dict[str, Any]] = json.load(fh)

    if not visual_prompts:
        _emit({"type": "error", "msg": "No visual prompts found in query JSON."})
        return

    _emit_log(f"Loaded {len(visual_prompts)} visual prompt(s).")

    unique_labels: list[str] = []
    for vp in visual_prompts:
        if vp["label"] not in unique_labels:
            unique_labels.append(vp["label"])
    label_to_cat_id: dict[str, int] = {lb: i + 1 for i, lb in enumerate(unique_labels)}

    existing_by_file: dict[str, list[list[float]]] = {}
    if args.max_overlap > 0 and args.annotation_json:
        existing_by_file = _load_existing_annotations(args.annotation_json, args.max_overlap)

    images_dir = Path(args.images_dir)
    frame_files = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.name,
    )
    total = len(frame_files)
    _emit_log(f"Found {total} target frames in {images_dir}.")

    # Phase 1: query embeddings
    _emit_log("Extracting query embeddings from visual prompts…")

    query_embeds: list[np.ndarray] = []
    query_meta: list[dict[str, Any]] = []
    prompts_by_image: dict[str, list[dict[str, Any]]] = {}
    for vp in visual_prompts:
        prompts_by_image.setdefault(vp["image_path"], []).append(vp)

    with torch.no_grad():
        for img_path_str, prompts_in_img in prompts_by_image.items():
            img = Image.open(img_path_str).convert("RGB")
            outputs = model([img])
            result = outputs[0]

            n_props = int(result["bboxes"].shape[0]) if result["bboxes"].numel() > 0 else 0

            if n_props == 0:
                _emit_log(
                    f"Warning: no proposals found in {Path(img_path_str).name} — "
                    "prompts from this frame will be skipped."
                )
                continue

            prop_bboxes: np.ndarray = result["bboxes"].cpu().float().numpy()
            prop_embeds: np.ndarray = result["embeddings"].cpu().float().numpy()
            prop_embeds_norm = _normalise(prop_embeds)

            for vp in prompts_in_img:
                bbox_xyxy = _xywh_to_xyxy(vp["bbox"])
                ious = _iou_one_vs_many(bbox_xyxy, prop_bboxes)
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou < 0.1:
                    _emit_log(
                        f"Warning: low IoU ({best_iou:.2f}) for prompt "
                        f"'{vp['label']}' in {Path(img_path_str).name}. "
                        "Consider drawing the box more tightly around the object."
                    )

                query_embeds.append(prop_embeds_norm[best_idx].copy())
                query_meta.append({
                    "label":             vp["label"],
                    "cat_id":            label_to_cat_id[vp["label"]],
                    "source_image":      Path(img_path_str).name,
                    "bbox":              vp["bbox"],
                    "iou_at_extraction": best_iou,
                })

    if not query_embeds:
        _emit({
            "type": "error",
            "msg": (
                "No query embeddings could be extracted from visual prompts. "
                "The model found no proposals in any of the prompt images."
            ),
        })
        return

    _emit_log(
        f"Extracted {len(query_embeds)} query embedding(s) "
        f"across {len(prompts_by_image)} prompt image(s)."
    )

    # Phase 2: candidate collection
    cand_embeds: list[np.ndarray] = []
    cand_meta: list[dict[str, Any]] = []

    with torch.no_grad():
        for frame_idx, frame_path in enumerate(frame_files):
            img = Image.open(frame_path).convert("RGB")
            W, H = img.size

            outputs = model([img])
            result = outputs[0]

            n_props = (
                int(result["bboxes"].shape[0]) if result["bboxes"].numel() > 0 else 0
            )

            if n_props == 0:
                _emit({
                    "type": "frame",
                    "done": frame_idx + 1,
                    "total": total,
                    "n_props": 0,
                    "n_candidates": 0,
                })
                continue

            prop_bboxes = result["bboxes"].cpu().float().numpy()
            prop_embeds = result["embeddings"].cpu().float().numpy()
            prop_scores = result["scores"].cpu().float().numpy()
            prop_embeds_norm = _normalise(prop_embeds)

            existing_xyxy: np.ndarray | None = None
            ex_bboxes = existing_by_file.get(frame_path.name)
            if ex_bboxes and args.max_overlap > 0:
                existing_xyxy = np.array(ex_bboxes, dtype=np.float32)

            n_candidates = 0
            for j in range(n_props):
                if float(prop_scores[j]) < args.score_threshold:
                    continue
                if existing_xyxy is not None and len(existing_xyxy) > 0:
                    ious = _iou_one_vs_many(prop_bboxes[j].tolist(), existing_xyxy)
                    if float(ious.max()) >= args.max_overlap:
                        continue

                cand_embeds.append(prop_embeds_norm[j].copy())
                cand_meta.append({
                    "file_name":  frame_path.name,
                    "image_path": str(frame_path),
                    "bbox":       _xyxy_to_xywh(prop_bboxes[j].tolist()),
                    "score":      float(prop_scores[j]),
                    "width":      W,
                    "height":     H,
                })
                n_candidates += 1

            _emit({
                "type":         "frame",
                "done":         frame_idx + 1,
                "total":        total,
                "n_props":      n_props,
                "n_candidates": n_candidates,
            })

    if not cand_embeds:
        _emit({"type": "error", "msg": "No candidate proposals found across all frames."})
        return

    # Phase 3: build per-label prototype embeddings (mean of all examples for
    # that label, re-normalised).  This is more robust than matching against
    # individual queries with max-pooling: it captures the *typical* appearance
    # of each class and reduces the influence of outlier / imprecise boxes.
    label_to_q_indices: dict[str, list[int]] = {}
    for i, meta in enumerate(query_meta):
        label_to_q_indices.setdefault(meta["label"], []).append(i)

    prototype_embeds: list[np.ndarray] = []
    prototype_meta: list[dict[str, Any]] = []
    for label in unique_labels:
        indices = label_to_q_indices.get(label, [])
        if not indices:
            continue
        vecs = np.stack([query_embeds[i] for i in indices], axis=0)
        proto = vecs.mean(axis=0)
        norm = float(np.linalg.norm(proto))
        proto = proto / max(norm, 1e-6)
        prototype_embeds.append(proto)
        prototype_meta.append({
            "label":   label,
            "cat_id":  label_to_cat_id[label],
            "n_shots": len(indices),
        })

    _emit_log(
        f"Built {len(prototype_embeds)} class prototype(s) from {len(query_embeds)} "
        f"query embedding(s): "
        + ", ".join(
            f"'{m['label']}' ({m['n_shots']} shot{'s' if m['n_shots'] > 1 else ''})"
            for m in prototype_meta
        )
        + f".  Matching against {len(cand_embeds)} candidates…"
    )

    P = np.stack(prototype_embeds, axis=0)   # (num_classes, D)
    C = np.stack(cand_embeds,      axis=0)   # (num_candidates, D)
    sim_matrix = P @ C.T                      # (num_classes, num_candidates)
    max_sim       = sim_matrix.max(axis=0)    # (num_candidates,) — best class match
    best_proto_idx = sim_matrix.argmax(axis=0)

    matches: list[dict[str, Any]] = []
    for c_idx in range(len(cand_embeds)):
        sim = float(max_sim[c_idx])
        if sim < args.min_similarity:
            continue
        p_idx = int(best_proto_idx[c_idx])
        entry = dict(cand_meta[c_idx])
        entry["similarity"] = sim
        entry["label"]      = prototype_meta[p_idx]["label"]
        entry["cat_id"]     = prototype_meta[p_idx]["cat_id"]
        matches.append(entry)

    _emit_log(f"Found {len(matches)} match(es) above similarity threshold.")

    if args.nms and matches:
        before_nms = len(matches)
        matches = _apply_nms_to_matches(matches, args.nms_iou)
        _emit_log(
            f"After NMS (IoU ≥ {args.nms_iou}): {len(matches)} match(es) "
            f"({before_nms - len(matches)} suppressed)."
        )

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    matches = matches[: args.top_k]
    _emit_log(f"Kept {len(matches)} match(es) after top-{args.top_k} cut.")

    # Phase 4: COCO JSON
    fname_to_img_id: dict[str, int] = {f.name: i + 1 for i, f in enumerate(frame_files)}

    coco_out: dict[str, Any] = {
        "info":       {"image_root": args.image_root},
        "images":     [
            {"id": i + 1, "file_name": f.name, "width": 0, "height": 0}
            for i, f in enumerate(frame_files)
        ],
        "categories": [
            {"id": label_to_cat_id[lb], "name": lb} for lb in unique_labels
        ],
        "annotations": [],
    }

    ann_id = 1
    counts_per_class: dict[str, int] = {}
    for m in matches:
        img_id = fname_to_img_id.get(m["file_name"])
        if img_id is None:
            continue
        x, y, w, h = [round(v, 2) for v in m["bbox"]]
        coco_out["annotations"].append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": m["cat_id"],
            "bbox":        [x, y, w, h],
            "area":        round(w * h, 2),
            "iscrowd":     0,
            "score":       round(m["similarity"], 4),
            "source":      "visual_prompt",
        })
        counts_per_class[m["label"]] = counts_per_class.get(m["label"], 0) + 1
        ann_id += 1

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(coco_out, fh, indent=2)

    vis_entries = _generate_vis(
        coco_out, {f.name: f for f in frame_files}, args.vis_dir
    )

    _emit({
        "type":             "done",
        "num_images":       total,
        "num_annotations":  len(coco_out["annotations"]),
        "counts_per_class": counts_per_class,
        "output_json":      str(args.output_json),
        "vis_entries":      vis_entries,
    })


# ── YOLO-E backend ──────────────────────────────────────────────────────────────

def _build_yoloe_dataset(
    visual_prompts: list[dict[str, Any]],
    unique_labels: list[str],
    tmp_dir: Path,
) -> None:
    """Create a minimal CVAT-format dataset directory from visual prompts.

    Layout produced:
        tmp_dir/
            annotations/
                instances_default.json
            images/
                default/
                    <symlinks to source images>
    """
    ann_dir = tmp_dir / "annotations"
    img_dir = tmp_dir / "images" / "default"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    label_to_cat_id = {lb: i + 1 for i, lb in enumerate(unique_labels)}

    images_list: list[dict[str, Any]] = []
    annotations_list: list[dict[str, Any]] = []
    ann_id = 1
    img_id = 1
    seen_names: dict[str, str] = {}  # orig_path → fname used in dataset

    prompts_by_image: dict[str, list[dict[str, Any]]] = {}
    for vp in visual_prompts:
        prompts_by_image.setdefault(vp["image_path"], []).append(vp)

    for img_path_str, prompts in prompts_by_image.items():
        img_path = Path(img_path_str)
        fname = img_path.name

        # Deduplicate filenames from different directories
        if img_path_str not in seen_names:
            if (img_dir / fname).exists():
                fname = f"{img_path.stem}_{img_id}{img_path.suffix}"
            seen_names[img_path_str] = fname
            symlink = img_dir / fname
            if not symlink.exists():
                symlink.symlink_to(img_path.resolve())
        else:
            fname = seen_names[img_path_str]

        images_list.append({"id": img_id, "file_name": fname, "width": 0, "height": 0})

        for vp in prompts:
            x, y, w, h = vp["bbox"]
            annotations_list.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": label_to_cat_id[vp["label"]],
                "bbox":        [x, y, w, h],
                "area":        max(1, w * h),
                "iscrowd":     0,
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images":      images_list,
        "categories":  [{"id": label_to_cat_id[lb], "name": lb} for lb in unique_labels],
        "annotations": annotations_list,
    }
    with open(ann_dir / "instances_default.json", "w") as fh:
        json.dump(coco, fh, indent=2)


def _run_yoloe(args: argparse.Namespace) -> None:
    """YOLO-E few-shot detection pipeline."""
    import shutil
    import tempfile
    import cv2

    # Make the package importable
    fsdet_dir = str(args.fsdet_dir)
    if fsdet_dir not in sys.path:
        sys.path.insert(0, fsdet_dir)

    try:
        from few_shot_object_detection import FewShotDetector  # noqa: PLC0415
    except ImportError as exc:
        _emit({
            "type": "error",
            "msg": (
                f"Cannot import few_shot_object_detection from '{fsdet_dir}': {exc}. "
                "Check the 'YOLO-E package dir' path."
            ),
        })
        return

    with open(args.query_json) as fh:
        visual_prompts: list[dict[str, Any]] = json.load(fh)

    if not visual_prompts:
        _emit({"type": "error", "msg": "No visual prompts found in query JSON."})
        return

    _emit_log(f"Loaded {len(visual_prompts)} visual prompt(s).")

    unique_labels: list[str] = []
    for vp in visual_prompts:
        if vp["label"] not in unique_labels:
            unique_labels.append(vp["label"])
    label_to_cat_id: dict[str, int] = {lb: i + 1 for i, lb in enumerate(unique_labels)}

    existing_by_file: dict[str, list[list[float]]] = {}
    if args.max_overlap > 0 and args.annotation_json:
        existing_by_file = _load_existing_annotations(args.annotation_json, args.max_overlap)

    images_dir = Path(args.images_dir)
    frame_files = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.name,
    )
    total = len(frame_files)
    _emit_log(f"Found {total} target frames in {images_dir}.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="yoloe_prompts_"))
    try:
        # Build temp dataset from visual prompts
        _emit_log("Building temporary reference dataset from visual prompts…")
        _build_yoloe_dataset(visual_prompts, unique_labels, tmp_dir)

        # Load model
        _emit_log(f"Loading YOLO-E model (size: {args.yoloe_model_size})…")
        detector = FewShotDetector(device="auto", model_size=args.yoloe_model_size)
        if not detector.load_model():
            _emit({"type": "error", "msg": "Failed to load YOLO-E model."})
            return

        # Compute visual embeddings
        _emit_log(f"Computing visual embeddings from {len(visual_prompts)} prompt(s)…")
        if not detector.setup_from_cvat(
            str(tmp_dir),
            dataset_format="cvat",
            force_recompute=True,
            cache_file=str(tmp_dir / "embeddings_cache.pt"),
        ):
            _emit({"type": "error", "msg": "Failed to compute visual embeddings from prompts."})
            return

        _emit_log(f"YOLO-E ready — classes: {', '.join(unique_labels)}")

        # Build COCO skeleton
        fname_to_img_id = {f.name: i + 1 for i, f in enumerate(frame_files)}
        coco_out: dict[str, Any] = {
            "info":       {"image_root": args.image_root},
            "images":     [
                {"id": i + 1, "file_name": f.name, "width": 0, "height": 0}
                for i, f in enumerate(frame_files)
            ],
            "categories": [
                {"id": label_to_cat_id[lb], "name": lb} for lb in unique_labels
            ],
            "annotations": [],
        }

        ann_id = 1
        counts_per_class: dict[str, int] = {}

        for frame_idx, frame_path in enumerate(frame_files):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                _emit({
                    "type": "frame",
                    "done": frame_idx + 1,
                    "total": total,
                    "n_props": 0,
                    "n_candidates": 0,
                })
                continue

            dets = detector.detect(
                frame_bgr,
                confidence=args.yoloe_confidence,
                nms_enabled=args.yoloe_nms,
                nms_iou_threshold=args.yoloe_nms_iou,
            )

            # Optional dedup against existing annotations
            existing_xyxy: np.ndarray | None = None
            ex_bboxes = existing_by_file.get(frame_path.name)
            if ex_bboxes and args.max_overlap > 0:
                existing_xyxy = np.array(ex_bboxes, dtype=np.float32)

            n_det = 0
            for det in dets:
                label = det["class"]
                if label not in label_to_cat_id:
                    continue

                box = det["box"]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                bbox_xyxy = [x1, y1, x2, y2]

                if existing_xyxy is not None and len(existing_xyxy) > 0:
                    ious = _iou_one_vs_many(bbox_xyxy, existing_xyxy)
                    if float(ious.max()) >= args.max_overlap:
                        continue

                w, h = x2 - x1, y2 - y1
                coco_out["annotations"].append({
                    "id":          ann_id,
                    "image_id":    fname_to_img_id[frame_path.name],
                    "category_id": label_to_cat_id[label],
                    "bbox":        [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area":        round(w * h, 2),
                    "iscrowd":     0,
                    "score":       round(float(det["confidence"]), 4),
                    "source":      "visual_prompt_yoloe",
                })
                counts_per_class[label] = counts_per_class.get(label, 0) + 1
                ann_id += 1
                n_det += 1

            _emit({
                "type":         "frame",
                "done":         frame_idx + 1,
                "total":        total,
                "n_props":      len(dets),
                "n_candidates": n_det,
            })

        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(coco_out, fh, indent=2)

        _emit_log(
            f"Wrote {len(coco_out['annotations'])} annotation(s) to {args.output_json}."
        )

        vis_entries = _generate_vis(
            coco_out, {f.name: f for f in frame_files}, args.vis_dir
        )

        _emit({
            "type":             "done",
            "num_images":       total,
            "num_annotations":  len(coco_out["annotations"]),
            "counts_per_class": counts_per_class,
            "output_json":      str(args.output_json),
            "vis_entries":      vis_entries,
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual-prompt detection — WeDetect-Uni or YOLO-E backend"
    )

    # Shared
    parser.add_argument("--backend",          default="wedetect",
                        choices=["wedetect", "yoloe"],
                        help="Detection backend to use.")
    parser.add_argument("--query-json",       required=True,
                        help="JSON file: [{image_path, bbox [x,y,w,h], label}, …]")
    parser.add_argument("--images-dir",       required=True)
    parser.add_argument("--output-json",      required=True)
    parser.add_argument("--image-root",       default="images/default")
    parser.add_argument("--vis-dir",          default="",
                        help="Directory to save detection visualisation images.")
    parser.add_argument("--max-overlap",      type=float, default=0.0,
                        help="IoU threshold for excluding already-detected regions.")
    parser.add_argument("--annotation-json",  default="",
                        help="Existing COCO JSON for duplicate exclusion.")

    # WeDetect-Uni specific
    parser.add_argument("--wedetect-dir",     default="")
    parser.add_argument("--uni-checkpoint",   default="")
    parser.add_argument("--top-k",            type=int,   default=50)
    parser.add_argument("--min-similarity",   type=float, default=0.75)
    parser.add_argument("--score-threshold",  type=float, default=0.0)
    parser.add_argument("--nms",              action="store_true",
                        help="Apply per-frame per-class NMS after similarity matching.")
    parser.add_argument("--nms-iou",          type=float, default=0.5,
                        help="IoU threshold for NMS suppression.")

    # YOLO-E specific
    parser.add_argument("--fsdet-dir",        default="",
                        help="Path to the few-shot-object-detection package directory.")
    parser.add_argument("--yoloe-model-size", default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--yoloe-confidence", type=float, default=0.25)
    parser.add_argument("--yoloe-nms",        action="store_true",
                        help="Enable per-class NMS in YOLO-E.")
    parser.add_argument("--yoloe-nms-iou",    type=float, default=0.5)

    args = parser.parse_args()

    try:
        if args.backend == "wedetect":
            _run_wedetect(args)
        else:
            _run_yoloe(args)
    except Exception as exc:
        _emit({
            "type": "error",
            "msg":  f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        })


if __name__ == "__main__":
    main()
