"""retrieval_runner.py

Subprocess helper: runs WeDetect-Uni proposal generation on all frames of a
dataset, extracts embeddings from confirmed (true) bounding-box annotations as
queries, and retrieves visually similar proposals that were *not* already
detected — i.e. potential missed detections.

Called by pages/2_Base_Class_Detection.py via subprocess.Popen.
All progress is emitted as newline-delimited JSON on stdout.

Workflow
--------
Single pass over every frame:
  1. Run WeDetect-Uni → generic object proposals with embeddings (xyxy, dim=768).
  2. For each existing annotation in that frame, find the best-overlapping
     proposal and store its normalised embedding as a *query*.
  3. Collect all proposals whose IoU with every existing annotation is below
     ``max_overlap`` as *candidates* (not yet detected).
After the pass:
  4. Cosine-similarity matrix (Q × C); keep top-k above ``min_similarity``.
  5. Write results to ``output_json`` and emit a "done" message.

Usage
-----
python retrieval_runner.py \\
    --wedetect-dir /path/to/WeDetect \\
    --uni-checkpoint checkpoints/wedetect_base_uni.pth \\
    --annotation-json /path/to/annotations/base_detection.json \\
    --images-dir /path/to/images/default \\
    --output-json /path/to/annotations/retrieval_results.json \\
    --top-k 50 \\
    --min-similarity 0.75 \\
    --max-overlap 0.30 \\
    --score-threshold 0.0 \\
    --query-classes "robot,screw"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _emit_log(msg: str) -> None:
    _emit({"type": "log", "msg": msg})


def _iou_one_vs_many(box_xyxy: list[float], boxes_xyxy: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an (N, 4) array of boxes (all xyxy)."""
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
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-6)


def _nms_matches(matches: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """Per-frame greedy NMS on retrieved matches.

    Within each frame, iterate matches in descending similarity order and
    suppress any later match whose IoU with an already-kept match meets or
    exceeds ``iou_threshold``.  Matches are already sorted by similarity
    before this function is called.
    """
    from collections import defaultdict

    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in matches:
        by_frame[m["file_name"]].append(m)

    kept: list[dict[str, Any]] = []
    for frame_matches in by_frame.values():
        kept_boxes: list[list[float]] = []
        for m in frame_matches:  # already sorted by similarity desc
            bbox_xyxy = _xywh_to_xyxy(m["bbox"])
            suppressed = False
            if kept_boxes:
                ious = _iou_one_vs_many(bbox_xyxy, np.array(kept_boxes, dtype=np.float32))
                if float(ious.max()) >= iou_threshold:
                    suppressed = True
            if not suppressed:
                kept.append(m)
                kept_boxes.append(bbox_xyxy)

    kept.sort(key=lambda x: x["similarity"], reverse=True)
    return kept


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_model(wedetect_dir: str, checkpoint_path: str):
    """Load the WeDetect-Uni (SimpleYOLOWorldDetector) model."""
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

    # Key remapping: backbone
    keys = list(checkpoint.keys())
    for key in keys:
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            checkpoint[new_key] = checkpoint.pop(key)

    # Key remapping: detection head
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WeDetect-Uni missed-detection retrieval")
    parser.add_argument("--wedetect-dir", required=True)
    parser.add_argument("--uni-checkpoint", required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--max-overlap", type=float, default=0.30)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--query-classes",
        default="",
        help=(
            "Comma-separated class names whose annotations are used as query embeddings. "
            "Empty string = use all classes. All annotations are still used for the "
            "candidate-exclusion IoU check regardless of this filter."
        ),
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        default=False,
        help="Apply per-frame Non-Maximum Suppression to retrieved matches.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
        help="IoU threshold for NMS suppression (lower = more aggressive).",
    )
    args = parser.parse_args()

    # Parse class filter once, outside the hot loop
    query_class_filter: set[str] | None = None
    if args.query_classes.strip():
        query_class_filter = {c.strip() for c in args.query_classes.split(",") if c.strip()}

    try:
        import torch
        from PIL import Image

        # ── Load model ────────────────────────────────────────────────────────
        _emit_log("Loading WeDetect-Uni model…")
        model, device = _load_model(args.wedetect_dir, args.uni_checkpoint)
        _emit_log(f"Model on {device}. Ready.")

        # ── Load annotation file ──────────────────────────────────────────────
        with open(args.annotation_json) as fh:
            coco: dict[str, Any] = json.load(fh)

        img_id_to_name: dict[int, str] = {
            img["id"]: img["file_name"] for img in coco.get("images", [])
        }
        cat_map: dict[int, str] = {
            c["id"]: c["name"] for c in coco.get("categories", [])
        }

        # file_name → list of COCO annotation dicts
        ann_by_file: dict[str, list[dict]] = {}
        for ann in coco.get("annotations", []):
            fname = img_id_to_name.get(ann["image_id"])
            if fname:
                ann_by_file.setdefault(fname, []).append(ann)

        # ── Enumerate frames ──────────────────────────────────────────────────
        images_dir = Path(args.images_dir)
        frame_files = sorted(
            [p for p in images_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
            key=lambda p: p.name,
        )
        total = len(frame_files)
        _emit_log(f"Found {total} frames in {images_dir}.")

        # Accumulators
        query_embeds: list[np.ndarray] = []
        query_meta: list[dict[str, Any]] = []
        cand_embeds: list[np.ndarray] = []
        cand_meta: list[dict[str, Any]] = []

        # ── Single pass ───────────────────────────────────────────────────────
        with torch.no_grad():
            for frame_idx, frame_path in enumerate(frame_files):
                img = Image.open(frame_path).convert("RGB")
                W, H = img.size

                outputs = model([img])
                result = outputs[0]

                n_props = int(result["bboxes"].shape[0]) if result["bboxes"].numel() > 0 else 0

                if n_props == 0:
                    _emit({
                        "type": "frame",
                        "done": frame_idx + 1,
                        "total": total,
                        "file": frame_path.name,
                        "n_props": 0,
                        "n_candidates": 0,
                        "n_queries": 0,
                    })
                    continue

                prop_bboxes: np.ndarray = result["bboxes"].cpu().float().numpy()   # (N,4) xyxy
                prop_embeds: np.ndarray = result["embeddings"].cpu().float().numpy()  # (N,D)
                prop_scores: np.ndarray = result["scores"].cpu().float().numpy()    # (N,)

                prop_embeds_norm = _normalise(prop_embeds)

                fname = frame_path.name
                existing_anns = ann_by_file.get(fname, [])

                # Existing annotation bboxes in xyxy for overlap tests
                existing_xyxy: np.ndarray | None = None
                if existing_anns:
                    existing_xyxy = np.array(
                        [_xywh_to_xyxy(a["bbox"]) for a in existing_anns],
                        dtype=np.float32,
                    )

                # ── Extract query embeddings ───────────────────────────────
                n_queries = 0
                for ann in existing_anns:
                    cat_name = cat_map.get(ann.get("category_id", -1), "?")
                    # Skip if this class is not in the user-selected query filter
                    if query_class_filter is not None and cat_name not in query_class_filter:
                        continue

                    bbox_xyxy = _xywh_to_xyxy(ann["bbox"])
                    ious = _iou_one_vs_many(bbox_xyxy, prop_bboxes)
                    best_idx = int(np.argmax(ious))
                    best_iou = float(ious[best_idx])

                    query_embeds.append(prop_embeds_norm[best_idx].copy())
                    query_meta.append({
                        "ann_id": ann["id"],
                        "file_name": fname,
                        "bbox": ann["bbox"],
                        "category_id": ann.get("category_id"),
                        "category_name": cat_name,
                        "iou_at_extraction": best_iou,
                    })
                    n_queries += 1

                # ── Extract candidates ─────────────────────────────────────
                n_candidates = 0
                for j in range(n_props):
                    if float(prop_scores[j]) < args.score_threshold:
                        continue
                    # Skip proposals that heavily overlap with existing annotations
                    if existing_xyxy is not None and len(existing_xyxy) > 0:
                        ious = _iou_one_vs_many(prop_bboxes[j].tolist(), existing_xyxy)
                        if float(ious.max()) >= args.max_overlap:
                            continue

                    cand_embeds.append(prop_embeds_norm[j].copy())
                    cand_meta.append({
                        "file_name": fname,
                        "image_path": str(frame_path),
                        "bbox": _xyxy_to_xywh(prop_bboxes[j].tolist()),
                        "score": float(prop_scores[j]),
                        "width": W,
                        "height": H,
                    })
                    n_candidates += 1

                _emit({
                    "type": "frame",
                    "done": frame_idx + 1,
                    "total": total,
                    "file": frame_path.name,
                    "n_props": n_props,
                    "n_candidates": n_candidates,
                    "n_queries": n_queries,
                })

        # ── Matching ──────────────────────────────────────────────────────────
        if not query_embeds:
            filter_hint = (
                f" (query-class filter: {sorted(query_class_filter)})"
                if query_class_filter else ""
            )
            _emit({
                "type": "error",
                "msg": (
                    f"No query embeddings could be extracted{filter_hint}. "
                    "Make sure base_detection.json contains at least one annotation "
                    "of the selected class(es)."
                ),
            })
            return

        if not cand_embeds:
            _emit({
                "type": "error",
                "msg": (
                    "No candidate proposals found. "
                    "All proposals overlap with existing annotations, or no proposals were generated."
                ),
            })
            return

        _emit_log(
            f"Matching {len(query_embeds)} queries against "
            f"{len(cand_embeds)} candidates…"
        )

        Q = np.stack(query_embeds, axis=0)  # (Q, D)
        C = np.stack(cand_embeds, axis=0)   # (C, D)

        # Cosine similarity — arrays are already L2-normalised
        sim_matrix = Q @ C.T          # (Q, C)
        max_sim = sim_matrix.max(axis=0)         # (C,)
        best_query_idx = sim_matrix.argmax(axis=0)  # (C,) index into query_meta

        matches: list[dict[str, Any]] = []
        for c_idx in range(len(cand_embeds)):
            sim = float(max_sim[c_idx])
            if sim < args.min_similarity:
                continue
            q_idx = int(best_query_idx[c_idx])
            entry = dict(cand_meta[c_idx])
            entry["similarity"] = sim
            entry["matched_query"] = query_meta[q_idx]
            matches.append(entry)

        matches.sort(key=lambda x: x["similarity"], reverse=True)

        if args.nms and matches:
            before_nms = len(matches)
            matches = _nms_matches(matches, args.nms_iou)
            _emit_log(
                f"NMS (IoU ≥ {args.nms_iou:.2f}): {before_nms} → {len(matches)} matches."
            )

        matches = matches[: args.top_k]

        # ── Save results ──────────────────────────────────────────────────────
        output: dict[str, Any] = {
            "query_count": len(query_embeds),
            "candidate_count": len(cand_embeds),
            "match_count": len(matches),
            "matches": matches,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(output, fh, indent=2)

        _emit({
            "type": "done",
            "match_count": len(matches),
            "query_count": len(query_embeds),
            "candidate_count": len(cand_embeds),
            "output_json": str(args.output_json),
        })

    except Exception as exc:
        _emit({"type": "error", "msg": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
