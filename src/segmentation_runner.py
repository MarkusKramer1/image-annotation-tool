"""segmentation_runner.py

Subprocess helper: generates COCO polygon segmentation masks for existing
bounding-box annotations using either FastSAM or SAM2 (both via ultralytics).

Called by pages/2_Base_Class_Detection.py via subprocess.Popen.
All progress is emitted as newline-delimited JSON on stdout.

Usage
-----
python segmentation_runner.py \\
    --model-type   fastsam                        # or sam2
    --checkpoint   FastSAM-x.pt                   # or sam2.1_l.pt, etc.
    --annotation-json /abs/path/base_detection.json \\
    --images-dir      /abs/path/images/default \\
    --output-json     /abs/path/base_detection.json \\
    [--classes        "Schraube,Mutter"] \\
    [--conf           0.25] \\
    [--imgsz          1024]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _emit_log(msg: str) -> None:
    _emit({"type": "log", "msg": msg})


def _masks_to_polygons(masks_xy) -> list[list[float]]:
    """Convert ultralytics masks.xy (list of Nx2 arrays) to COCO polygon lists."""
    polys = []
    for xy in masks_xy:
        flat = xy.flatten().tolist()
        if len(flat) >= 6:
            polys.append(flat)
    return polys


def _run_fastsam(
    predictor,
    img_path: str,
    qualifying_anns: list[dict],
    ann_by_id: dict[int, dict],
) -> int:
    """Run FastSAM on one image; returns the number of masks generated."""
    from ultralytics.models.fastsam import FastSAMPredictor  # noqa: F401 (type hint)

    everything_results = predictor(img_path)
    total = 0
    for ann in qualifying_anns:
        x, y, w, h = ann["bbox"]
        prompted = predictor.prompt(everything_results, bboxes=[[x, y, x + w, y + h]])
        if not prompted or prompted[0].masks is None:
            _emit_log(f"  WARNING: no mask for ann {ann['id']}")
            continue
        polys = _masks_to_polygons(prompted[0].masks.xy)
        if polys:
            ann_by_id[ann["id"]]["segmentation"] = polys
            total += 1
        else:
            _emit_log(f"  WARNING: empty polygon for ann {ann['id']}")
    return total


def _run_sam2(
    model,
    img_path: str,
    qualifying_anns: list[dict],
    ann_by_id: dict[int, dict],
    imgsz: int,
) -> int:
    """Run SAM2 on one image; returns the number of masks generated."""
    total = 0
    for ann in qualifying_anns:
        x, y, w, h = ann["bbox"]
        results = model(
            img_path,
            bboxes=[[x, y, x + w, y + h]],
            imgsz=imgsz,
            verbose=False,
        )
        if not results or results[0].masks is None:
            _emit_log(f"  WARNING: no mask for ann {ann['id']}")
            continue
        polys = _masks_to_polygons(results[0].masks.xy)
        if polys:
            ann_by_id[ann["id"]]["segmentation"] = polys
            total += 1
        else:
            _emit_log(f"  WARNING: empty polygon for ann {ann['id']}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation runner (FastSAM / SAM2)")
    parser.add_argument("--model-type", choices=["fastsam", "sam2"], required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint file path or ultralytics model name, e.g. "
                             "FastSAM-x.pt / sam2.1_l.pt (auto-downloaded if not found)")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--classes", default="",
                        help="Comma-separated class names to segment (empty = all)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (FastSAM); ignored for SAM2")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Inference image size")
    args = parser.parse_args()

    # ── Verify ultralytics ────────────────────────────────────────────────────
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        _emit({
            "type": "error",
            "msg": (
                "ultralytics is not installed in the current Python environment.\n"
                "Install it with:\n"
                "  pip install ultralytics\n"
                "or point the 'Python executable' field in the UI to an environment "
                "that has ultralytics installed."
            ),
        })
        sys.exit(1)

    # ── Load annotation JSON ──────────────────────────────────────────────────
    ann_path = Path(args.annotation_json)
    images_dir = Path(args.images_dir)
    _emit_log(f"Loading annotations from {ann_path}")

    with open(ann_path) as fh:
        coco = json.load(fh)

    id_to_image: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
    cat_id_to_name: dict[int, str] = {
        c["id"]: c["name"] for c in coco.get("categories", [])
    }

    class_filter: set[str] | None = None
    if args.classes.strip():
        class_filter = {c.strip() for c in args.classes.split(",")}
        _emit_log(f"Segmenting classes: {', '.join(sorted(class_filter))}")
    else:
        _emit_log("Segmenting all classes")

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    target_image_ids = [
        img_id for img_id, anns in anns_by_image.items()
        if any(
            class_filter is None
            or cat_id_to_name.get(a.get("category_id", -1), "") in class_filter
            for a in anns
        )
    ]

    total_images = len(target_image_ids)
    _emit_log(f"Found {total_images} image(s) with qualifying annotations")
    if total_images == 0:
        _emit({"type": "done", "total_masks": 0})
        return

    ann_by_id: dict[int, dict] = {a["id"]: a for a in coco.get("annotations", [])}

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint = args.checkpoint
    ckpt_path = Path(checkpoint)

    # Ensure the parent directory exists so that ultralytics downloads the
    # weights file there when the checkpoint is not present yet.
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists():
        _emit_log(f"Loading {args.model_type.upper()} model from: {ckpt_path}")
    else:
        _emit_log(
            f"Checkpoint not found — ultralytics will auto-download "
            f"{ckpt_path.name} into {ckpt_path.parent}"
        )

    try:
        if args.model_type == "fastsam":
            from ultralytics.models.fastsam import FastSAMPredictor
            overrides = dict(
                conf=args.conf,
                task="segment",
                mode="predict",
                model=checkpoint,
                save=False,
                imgsz=args.imgsz,
                verbose=False,
            )
            predictor = FastSAMPredictor(overrides=overrides)
            _emit_log("FastSAM model loaded successfully")
        else:
            from ultralytics import SAM
            model = SAM(checkpoint)
            model.info()
            _emit_log("SAM2 model loaded successfully")
    except Exception as exc:
        _emit({"type": "error", "msg": f"Failed to load model: {exc}"})
        sys.exit(1)

    # ── Process images ────────────────────────────────────────────────────────
    total_masks = 0

    for img_idx, img_id in enumerate(target_image_ids):
        img_info = id_to_image[img_id]
        fname = img_info["file_name"]
        img_path = images_dir / fname

        _emit({
            "type": "progress",
            "current": img_idx + 1,
            "total": total_images,
            "msg": f"[{img_idx + 1}/{total_images}] {fname}",
        })

        if not img_path.exists():
            _emit_log(f"  WARNING: image not found: {img_path}, skipping")
            continue

        qualifying_anns = [
            a for a in anns_by_image.get(img_id, [])
            if class_filter is None
            or cat_id_to_name.get(a.get("category_id", -1), "") in class_filter
        ]

        try:
            if args.model_type == "fastsam":
                n = _run_fastsam(predictor, str(img_path), qualifying_anns, ann_by_id)
            else:
                n = _run_sam2(model, str(img_path), qualifying_anns, ann_by_id, args.imgsz)
            total_masks += n
        except Exception as exc:
            _emit_log(f"  ERROR on {fname}: {exc}\n{traceback.format_exc()}")

    _emit_log(f"Generated {total_masks} mask(s)")

    # ── Write updated COCO JSON ───────────────────────────────────────────────
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(coco, fh, indent=2)

    _emit_log(f"Saved updated annotations to {out_path}")
    _emit({"type": "done", "total_masks": total_masks})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _emit({"type": "error", "msg": f"Unhandled exception: {exc}\n{traceback.format_exc()}"})
        sys.exit(1)
