"""annotation_runner.py

Subprocess helper: runs WeDetect text-prompted detection on a directory of
extracted video frames and writes a COCO-format annotation file compatible
with the Embedding Analysis page.

Called by pages/1_Automatic_Annotation.py via subprocess.Popen.
All progress is emitted as newline-delimited JSON on stdout.

Usage
-----
python annotation_runner.py \\
    --wedetect-dir /path/to/WeDetect \\
    --config config/wedetect_base.py \\
    --checkpoint checkpoints/wedetect_base.pth \\
    --images-dir /abs/path/images/default \\
    --classes "Schraube,Mutter,Flansch" \\
    --output-json /abs/path/annotations/instances_default.json \\
    --vis-dir /abs/path/_vis \\
    --threshold 0.3 \\
    --topk 100
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _emit_log(msg: str) -> None:
    _emit({"type": "log", "msg": msg})


def main() -> None:
    parser = argparse.ArgumentParser(description="WeDetect annotation runner")
    parser.add_argument("--wedetect-dir", required=True)
    parser.add_argument("--config", required=True,
                        help="Config file path relative to wedetect-dir")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint file path relative to wedetect-dir (or absolute)")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--classes", required=True,
                        help="Comma-separated class names")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        default=False,
        help=(
            "Enable adaptive per-class thresholding: after applying the global "
            "threshold, keep only detections that score >= ratio * best_score "
            "for their class. Useful when WeDetect returns many low-confidence "
            "duplicates alongside a strong top detection."
        ),
    )
    parser.add_argument(
        "--adaptive-threshold-ratio",
        type=float,
        default=0.95,
        help="Ratio of per-class best score below which detections are suppressed (default: 0.95).",
    )
    parser.add_argument(
        "--image-root",
        default="images/default",
        help="Relative path from the dataset directory to the image folder (written into info.image_root).",
    )
    args = parser.parse_args()

    wedetect_dir = os.path.abspath(args.wedetect_dir)
    # Change into the WeDetect directory so that all relative paths inside
    # the mmengine configs (e.g. ./xlm-roberta-base/) resolve correctly.
    os.chdir(wedetect_dir)
    sys.path.insert(0, wedetect_dir)

    _emit_log(f"WeDetect directory: {wedetect_dir}")

    try:
        import torch
        from mmdet.utils import register_all_modules
        register_all_modules()
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmengine.runner.amp import autocast
        from mmdet.apis import init_detector
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        _emit({
            "type": "error",
            "msg": f"Import failed: {exc}\nMake sure mmdet, mmengine, mmcv and their dependencies are installed.",
        })
        sys.exit(1)

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    # WeDetect expects a list of [text] entries + a trailing [" "] sentinel
    texts = [[c] for c in class_names] + [[" "]]
    _emit_log(f"Categories ({len(class_names)}): {', '.join(class_names)}")

    # ── Load model ──────────────────────────────────────────────────────────
    _emit_log("Loading model…")

    def _load_model(device: str):
        config_path = os.path.join(wedetect_dir, args.config)
        checkpoint_path = (
            args.checkpoint
            if os.path.isabs(args.checkpoint)
            else os.path.join(wedetect_dir, args.checkpoint)
        )
        cfg = Config.fromfile(config_path)
        # Monkey-patch torch.load so the checkpoint loads without weights_only error
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        try:
            m = init_detector(cfg, checkpoint=checkpoint_path, device=device, palette=["red"])
        finally:
            torch.load = _orig
        pipeline = Compose(cfg.test_pipeline)
        m.reparameterize(texts)
        # Reshape text_feats: [N_classes, 1, embed] → [1, N_classes, embed]
        if hasattr(m, "text_feats") and m.text_feats is not None:
            m.text_feats = m.text_feats.squeeze(1).unsqueeze(0)
        return m, pipeline, device

    try:
        if torch.cuda.is_available():
            # Release any cached allocations left by other processes
            torch.cuda.empty_cache()
            try:
                model, test_pipeline, device = _load_model("cuda:0")
            except RuntimeError as cuda_err:
                if any(k in str(cuda_err) for k in ("CUDA", "cuda", "CUBLAS", "out of memory")):
                    _emit_log(
                        f"GPU load failed ({cuda_err.__class__.__name__}: {cuda_err}). "
                        "Falling back to CPU — inference will be slower."
                    )
                    torch.cuda.empty_cache()
                    model, test_pipeline, device = _load_model("cpu")
                else:
                    raise
        else:
            model, test_pipeline, device = _load_model("cpu")

        _emit_log(f"Model ready on {device}.")
    except Exception:
        _emit({"type": "error", "msg": f"Model load failed:\n{traceback.format_exc()}"})
        sys.exit(1)

    # ── Collect frames ───────────────────────────────────────────────────────
    images_dir = Path(args.images_dir)
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    total = len(image_files)
    _emit_log(f"Found {total} frames to process.")

    Path(args.vis_dir).mkdir(parents=True, exist_ok=True)

    COLORS = [
        "red", "lime", "dodgerblue", "yellow", "orange", "magenta",
        "cyan", "hotpink", "chartreuse", "coral", "violet", "gold",
        "deepskyblue", "tomato", "springgreen", "darkorange",
    ]

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    coco: dict = {
        "info": {
            "description": "Base class detections",
            "image_root": args.image_root,
        },
        "images": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": ""}
            for i, name in enumerate(class_names)
        ],
        "annotations": [],
    }
    ann_id = 1

    # ── Run inference ────────────────────────────────────────────────────────
    for idx, img_path in enumerate(image_files):
        img_id = idx + 1
        try:
            data_info = dict(img_id=img_id, img_path=str(img_path), texts=texts)
            data_info = test_pipeline(data_info)

            # Remove texts from data_samples so the model uses the cached
            # text_feats set by reparameterize (correct shape/encoding)
            # rather than re-encoding the pipeline-flattened texts.
            ds = data_info["data_samples"]
            if hasattr(ds, "texts"):
                del ds.texts

            data_batch = dict(
                inputs=data_info["inputs"].unsqueeze(0),
                data_samples=[ds],
            )

            with autocast(enabled=False), torch.no_grad():
                output = model.test_step(data_batch)[0]
                pred = output.pred_instances
                pred = pred[pred.scores.float() > args.threshold]

            # Adaptive per-class threshold: keep only detections within
            # `ratio` of the best score for their class.
            if args.adaptive_threshold and len(pred.scores) > 0:
                scores = pred.scores.float()
                labels = pred.labels
                keep = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
                for cls_id in labels.unique():
                    mask = labels == cls_id
                    best = scores[mask].max()
                    keep[mask & (scores < args.adaptive_threshold_ratio * best)] = False
                pred = pred[keep]

            if len(pred.scores) > args.topk:
                indices = pred.scores.float().topk(args.topk)[1]
                pred = pred[indices]

            pred_np = pred.cpu().numpy()
            bboxes   = pred_np["bboxes"]    # xyxy float32
            label_ids = pred_np["labels"]   # 0-indexed into texts list
            scores   = pred_np["scores"]

            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            coco["images"].append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

            for bbox, label_idx, score in zip(bboxes, label_ids, scores):
                cat_id = int(label_idx) + 1
                if cat_id > len(class_names):
                    continue  # skip the required trailing " " sentinel
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                bw, bh = x2 - x1, y2 - y1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                    "score": round(float(score), 4),
                })
                ann_id += 1

            # Draw detection boxes for the thumbnail
            draw = ImageDraw.Draw(pil_img)
            for bbox, label_idx, score in zip(bboxes, label_ids, scores):
                cat_id = int(label_idx) + 1
                if cat_id > len(class_names):
                    continue
                color = COLORS[int(label_idx) % len(COLORS)]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label_text = f"{class_names[int(label_idx)]}  {float(score):.2f}"
                label_y = max(0, y1 - 15)
                try:
                    tb = draw.textbbox((x1, label_y), label_text, font=font)
                    draw.rectangle(tb, fill=color)
                    draw.text((x1, label_y), label_text, fill="white", font=font)
                except Exception:
                    draw.text((x1, y1), label_text, fill=color)

            vis_path = str(Path(args.vis_dir) / img_path.name)
            pil_img.save(vis_path)

            _emit({
                "type": "frame",
                "frame_name": img_path.name,
                "vis_path": vis_path,
                "n_det": int(len(bboxes)),
                "done": idx + 1,
                "total": total,
            })

        except Exception:
            _emit_log(f"Error processing {img_path.name}:\n{traceback.format_exc()}")
            try:
                pil_img = Image.open(img_path).convert("RGB")
                iw, ih = pil_img.size
                if not any(im["id"] == img_id for im in coco["images"]):
                    coco["images"].append({"id": img_id, "file_name": img_path.name, "width": iw, "height": ih})
            except Exception:
                pass
            _emit({
                "type": "frame",
                "frame_name": img_path.name,
                "vis_path": None,
                "n_det": 0,
                "done": idx + 1,
                "total": total,
            })

    # ── Write COCO JSON ──────────────────────────────────────────────────────
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(coco, fh, indent=2)

    counts_per_class = {
        name: sum(1 for a in coco["annotations"] if a["category_id"] == i + 1)
        for i, name in enumerate(class_names)
    }
    _emit({
        "type": "done",
        "annotation_path": str(output_json),
        "num_images": len(coco["images"]),
        "num_annotations": len(coco["annotations"]),
        "counts_per_class": counts_per_class,
    })


if __name__ == "__main__":
    main()
