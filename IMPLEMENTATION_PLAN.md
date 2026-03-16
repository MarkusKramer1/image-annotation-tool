# Implementation Plan: Image Annotation Tool (Streamlit)

**Date:** 2026-03-16  
**Repo:** `/home/kramer/.openclaw/workspace/image-annotation-tool`  
**Status:** Planning — no code changes yet

---

## Overview

A Streamlit app for semi-automated image dataset annotation. Given video or image folder input, the tool:
1. Samples frames, runs WeDetect open-vocabulary detection
2. Embeds detected crops with C-RADIOv4-SO400M
3. Clusters embeddings to discover visual categories
4. Shows a prototype + bounding box per cluster and asks keep/discard
5. On keep: runs a second-pass visual similarity search (C-RADIO image-prompt) to mine additional detections
6. On discard: removes the cluster's annotations from the dataset
7. Exports a COCO-format annotation file
8. Continuously shows a scrollable annotated gallery and live statistics

---

## Architecture Summary

```
Input (video / image folder)
        │
        ▼
  Frame Sampler
  (fps or every-nth)
        │
        ▼
  WeDetect Detection
  (text prompt + threshold)
        │
        ▼
  C-RADIOv4 Crop Embedder
  (summary embedding per crop)
        │
        ▼
  Clustering (HDBSCAN / KMeans)
        │
        ▼
  Cluster Review UI
  ┌─────────────────────────────────┐
  │  Prototype image + bbox         │
  │  [Keep] [Discard]               │
  └─────────────────────────────────┘
        │                   │
      Keep               Discard
        │                   │
  Second-pass           Remove cluster
  Visual Mining         annotations
  (RADIO image-prompt
   cosine similarity)
        │
  Add mined detections
        │
        ▼
  COCO JSON export
        │
        ▼
  Before/After stats + Gallery
```

---

## 1. Conda Environment Setup

### Environment name: `image-annotation`

```bash
conda create -n image-annotation python=3.11 -y
conda activate image-annotation

# PyTorch with CUDA 12.4 (match WeDetect requirement)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# WeDetect dependencies (mmlab stack)
pip install mmengine==0.10.7
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.5/index.html
pip install mmdet==3.3.0

# Transformers stack
pip install transformers==4.57.1 accelerate==1.10.0 trl==0.17.0

# C-RADIOv4 deps
pip install einops

# Clustering and numerics
pip install scikit-learn hdbscan numpy scipy

# Streamlit and image handling
pip install streamlit==1.43.0 Pillow opencv-python-headless

# Video handling
pip install av  # PyAV for frame extraction

# Utilities
pip install tqdm pycocotools

# WeDetect — clone and install in-place
# git clone https://github.com/WeChatCV/WeDetect
# pip install -e WeDetect/
```

**requirements.txt** should be generated after env is stable with `pip freeze > requirements.txt`.

### Notes
- WeDetect needs **Chinese-language class names** for the base detector (XLM-RoBERTa text encoder). The UI must accept Chinese text OR translate automatically.
- If English-only class names are required, use WeDetect-Ref (2B) or override with a translated prompt.
- C-RADIOv4-SO400M (~431M params) requires ~2 GB VRAM; WeDetect-Base needs additional ~2–4 GB. Recommend GPU with ≥8 GB VRAM (e.g. RTX 3080+).

---

## 2. Project File Structure

```
image-annotation-tool/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
├── IMPLEMENTATION_PLAN.md    # This file
├── src/
│   ├── __init__.py
│   ├── frame_sampler.py      # Video/folder → frame list
│   ├── detector.py           # WeDetect wrapper
│   ├── embedder.py           # C-RADIOv4 crop embedder
│   ├── clusterer.py          # Embedding clustering + prototype selection
│   ├── miner.py              # Second-pass visual similarity mining
│   ├── coco_writer.py        # COCO JSON builder / updater
│   └── utils.py              # Image drawing, helpers
├── output/                   # Default output folder (created at runtime)
│   ├── annotations.json      # COCO output
│   └── annotated/            # Annotated image previews
└── assets/                   # Logo / placeholder images
```

---

## 3. Implementation Phases

### Phase 0: Scaffold (1–2 h)

- Create `src/` package with empty modules + docstrings
- Create `app.py` skeleton with `st.title`, placeholder sections
- Verify conda env imports all required packages cleanly
- Set up `output/` directory creation logic

---

### Phase 1: Input & Frame Sampling (2–3 h)

**`src/frame_sampler.py`**

Two input modes — no sidebar, all controls inline:

**Video upload:**
```
st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"])
fps_control = st.slider("Sample at N fps", 0.1, 30.0, 1.0)
```
- Write uploaded file to a temp path
- Use `av` (PyAV) to decode: seek to each frame at 1/fps intervals
- Return list of `PIL.Image` objects + metadata (source file, frame index, timestamp)

**Image folder path:**
```
folder_path = st.text_input("Image folder path")
every_nth = st.number_input("Use every Nth image", 1, 100, 1)
```
- `os.listdir` filtered by extension (jpg, jpeg, png, bmp, tiff, webp)
- Sorted by filename; slice `[::every_nth]`
- Return list of `PIL.Image` + metadata (path, filename)

Output: `List[dict]` with keys `image` (PIL), `source`, `frame_id`, optionally `timestamp`.

---

### Phase 2: Detection with WeDetect (3–4 h)

**`src/detector.py`**

```python
class WeDetectDetector:
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        from mmdet.apis import init_detector
        self.model = init_detector(config_path, checkpoint_path, device=device)

    def detect(self, image: PIL.Image, class_names: List[str], threshold: float) -> List[dict]:
        # Returns list of {bbox: [x,y,w,h], score: float, class_name: str, class_id: int}
        ...
```

**UI controls (inline, no sidebar):**
```
class_query = st.text_input("Detection class names (comma-separated, Chinese OK)")
threshold = st.slider("Detection threshold", 0.01, 1.0, 0.3, 0.01)
```

**Implementation notes:**
- WeDetect uses mmdetection's `inference_detector()` API
- Class names fed via config or pre-computed text embeddings (check WeDetect README)
- Per-class threshold override via `cfg.test_cfg.score_thr` before `init_detector()`
- Dynamic threshold: slider updates the threshold; re-detection triggered via `st.rerun()` if threshold changes
- Output bbox format: `[x_min, y_min, x_max, y_max]` (XYXY), converted to COCO `[x,y,w,h]` for export

**Model paths:** Accept via `st.text_input("WeDetect config path")` and `st.text_input("WeDetect checkpoint path")` at top of page.

---

### Phase 3: Crop Embedding with C-RADIOv4-SO400M (2–3 h)

**`src/embedder.py`**

```python
class CRADIOEmbedder:
    def __init__(self, device="cuda"):
        from transformers import AutoModel, CLIPImageProcessor
        self.processor = CLIPImageProcessor.from_pretrained("nvidia/C-RADIOv4-SO400M")
        self.model = AutoModel.from_pretrained(
            "nvidia/C-RADIOv4-SO400M", trust_remote_code=True
        ).eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed_crops(self, image: PIL.Image, bboxes: List[List[int]]) -> np.ndarray:
        # Crop each bbox from image, embed, return (N, C) array of summary embeddings
        embeddings = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = image.crop((x1, y1, x2, y2))
            pv = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
            summary, _ = self.model(pv)
            embeddings.append(summary.cpu().numpy().squeeze())
        return np.array(embeddings)

    @torch.no_grad()
    def embed_image(self, image: PIL.Image) -> np.ndarray:
        # Full-image summary embedding (used as visual query in second pass)
        pv = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        summary, _ = self.model(pv)
        return summary.cpu().numpy().squeeze()
```

**Batching:** Process crops in batches of 32 to avoid OOM. Cache embeddings in `st.session_state` keyed by frame_id+bbox so re-runs don't recompute.

---

### Phase 4: Clustering (2–3 h)

**`src/clusterer.py`**

```python
def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",   # or "kmeans"
    n_clusters: int = 10,       # only for kmeans
    min_cluster_size: int = 5,  # only for hdbscan
) -> Tuple[np.ndarray, List[int]]:
    # Returns (labels array, list of unique cluster ids excluding noise=-1)
    ...

def compute_cluster_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    detections: List[dict],
) -> Dict[int, dict]:
    # For each cluster: compute mean embedding, find detection closest to mean
    # Returns {cluster_id: {"mean_embedding": ..., "prototype_detection": {...}}}
    ...
```

**UI controls:**
```
cluster_method = st.selectbox("Clustering method", ["HDBSCAN", "KMeans"])
if cluster_method == "KMeans":
    n_clusters = st.number_input("Number of clusters", 2, 100, 10)
else:
    min_cluster_size = st.number_input("Min cluster size", 2, 50, 5)
```

**Prototype selection:** For each cluster, compute the mean embedding, then find the detection whose embedding has the highest cosine similarity to the mean — that's the prototype shown to the user.

---

### Phase 5: Cluster Review UI (3–4 h)

This is the core interactive section of the app.

**Layout per cluster:**
```
┌──────────────────────────────────────────────────────┐
│ Cluster N  (K detections from M images)              │
│                                                      │
│ [Prototype image with bbox drawn in color]           │
│  Confidence: 0.72   Predicted class: "person"        │
│                                                      │
│  [Keep — this is my target class]  [Discard]         │
└──────────────────────────────────────────────────────┘
```

**Implementation:**
- Use `st.columns` for multi-column cluster grid (2 or 3 wide)
- Draw bbox on prototype image using `PIL.ImageDraw` or `cv2` (red box, label)
- `st.image()` to display prototype
- Two `st.button()` calls with unique keys `f"keep_{cluster_id}"` / `f"discard_{cluster_id}"`
- Store decisions in `st.session_state["cluster_decisions"]` dict `{cluster_id: "keep" | "discard"}`
- Show a "Confirm decisions" button that triggers Phase 6+7

**State management:**
- All detections stored as `st.session_state["all_detections"]` (list of dicts)
- Cluster labels stored as `st.session_state["cluster_labels"]`
- Cluster prototypes in `st.session_state["cluster_prototypes"]`

---

### Phase 6: Discard — Remove Annotations (1 h)

**Logic in `app.py` or `src/coco_writer.py`:**

For clusters marked `discard`:
- Filter `all_detections` to remove any detection where `cluster_label == discarded_id`
- Update `st.session_state["active_detections"]`
- Log removal count for before/after statistics

---

### Phase 7: Keep — Second-Pass Visual Mining (3–4 h)

**`src/miner.py`**

For clusters marked `keep`:
1. Take the prototype crop image (or mean of kept cluster's crop images)
2. Embed it with `CRADIOEmbedder.embed_image()` → **visual query embedding**
3. Run WeDetect on ALL frames again (or a broader set) — OR: directly compare the visual query embedding against spatial features of all frames to find similar regions

**Preferred approach (pure embedding similarity, no second WeDetect pass):**
```
For each frame not yet annotated (or all frames):
    Extract spatial features with C-RADIOv4 (B, T, D) tensor
    Reshape to (H, W, D) spatial map
    Compute cosine similarity between visual query (1, D) and each spatial cell (H*W, D)
    Find local maxima in similarity map above a threshold
    Convert high-similarity regions to bounding boxes (multiply by patch stride = 16)
    Add as new detections
```

**Alternative approach (WeDetect image-prompt if supported):**
- Check if WeDetect-Ref supports image-prompt queries (Qwen3-VL back-end)
- If so, pass the prototype crop as a visual reference to WeDetect-Ref
- This gives richer semantic matching

**UI controls:**
```
mining_threshold = st.slider("Visual similarity threshold (second pass)", 0.5, 1.0, 0.8)
mining_method = st.radio("Mining method", ["Embedding similarity (fast)", "WeDetect-Ref (accurate)"])
```

**Output:** Additional detections added to `active_detections` with `source: "mined"` tag.

---

### Phase 8: COCO Export (2 h)

**`src/coco_writer.py`**

```python
def build_coco_dataset(
    frames: List[dict],
    detections: List[dict],
    class_names: List[str],
    output_dir: str,
) -> dict:
    # Build valid COCO JSON structure
    # Copy/symlink source images to output_dir/images/
    # Write output_dir/annotations.json
    ...
```

**COCO format structure:**
```json
{
  "info": {"description": "image-annotation-tool export", "date_created": "..."},
  "licenses": [],
  "images": [
    {"id": 1, "file_name": "frame_0001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "object"}
  ]
}
```

**Notes:**
- `bbox` in COCO is `[x_min, y_min, width, height]` (not XYXY)
- Auto-increment annotation IDs
- Images copied to `output_dir/images/` with original filenames
- JSON written atomically (write to temp, then rename)
- Download button: `st.download_button("Download annotations.json", ...)`

---

### Phase 9: Statistics & Gallery (2 h)

**Before/after class statistics:**

```
┌─────────────────────────────────────────────────────┐
│  Class Statistics                                    │
│  ─────────────────────────────────────────────────  │
│  Category     Before    After    Delta               │
│  person       1,203     1,847    +644 (mined)        │
│  car          342       0        -342 (discarded)    │
│  total        1,545     1,847                        │
└─────────────────────────────────────────────────────┘
```

Use `st.dataframe()` or `st.table()` for this. Shown immediately after decisions are confirmed.

**Scrollable annotated image gallery:**

- Always visible at bottom of page (or in an `st.expander("Image Gallery")`)
- Paginated: show 20 images at a time with `st.number_input("Page", ...)` or `st.selectbox`
- Each image: draw all active bboxes from `active_detections` using PIL
- Use `st.image()` with `use_container_width=True`
- Show filename + annotation count under each image

**Output folder display:**
```
st.text(f"Output folder: {output_dir}")
st.code(output_dir)
```
With a button to open it (Linux: `subprocess.run(["xdg-open", output_dir])`).

---

## 4. UI Flow (Single-Page, Top-to-Bottom)

```
[Title: Image Annotation Tool]

─── Input ──────────────────────────────────────────
  ○ Upload video   ○ Image folder path
  [upload widget / text input]
  [fps slider / every-nth input]

─── WeDetect Models ────────────────────────────────
  Config path: [text input]
  Checkpoint path: [text input]

─── Detection ──────────────────────────────────────
  Class names: [text input]
  Threshold: [slider 0.01–1.0]
  [Run Detection] button → progress bar → results count

─── Embedding & Clustering ─────────────────────────
  Method: [HDBSCAN / KMeans]
  Min cluster size / N clusters: [input]
  [Cluster] button → spinner

─── Cluster Review ─────────────────────────────────
  [Cluster cards grid — prototype + Keep/Discard]
  [Confirm Decisions] button

─── Mining (second pass) ───────────────────────────
  Mining method: [radio]
  Mining threshold: [slider]
  [Run Mining] button → progress bar

─── Statistics ─────────────────────────────────────
  [Before/After table]

─── Export ─────────────────────────────────────────
  Output folder: [text input, default ./output]
  [Export COCO] button
  [Download annotations.json]

─── Gallery ─────────────────────────────────────────
  [Scrollable annotated image grid, always visible]
  Page: [1 / N]
```

All state transitions use `st.session_state`. Steps after "Run Detection" are disabled (greyed via `st.button(..., disabled=True)`) until prior steps complete.

---

## 5. Session State Schema

```python
st.session_state = {
    "frames": List[dict],               # from frame_sampler
    "all_detections": List[dict],       # from detector
    "embeddings": np.ndarray,           # shape (N_detections, C)
    "cluster_labels": np.ndarray,       # shape (N_detections,)
    "cluster_prototypes": dict,         # {cluster_id: {...}}
    "cluster_decisions": dict,          # {cluster_id: "keep"|"discard"}
    "active_detections": List[dict],    # after discard filtering + mining
    "mined_detections": List[dict],     # from second pass
    "coco_data": dict,                  # final COCO JSON
    "step": str,                        # "input"|"detected"|"clustered"|"reviewed"|"mined"|"exported"
}
```

---

## 6. Detection Data Schema

Each detection dict throughout the pipeline:
```python
{
    "frame_id": str,       # unique frame identifier
    "image_path": str,     # absolute path to source image
    "bbox_xyxy": [x1,y1,x2,y2],  # original detection bbox
    "score": float,
    "class_name": str,
    "class_id": int,
    "cluster_label": int,  # assigned after clustering
    "embedding": np.ndarray,  # shape (C,), assigned after embedding
    "source": str,         # "wedetect" | "mined"
}
```

---

## 7. Testing Plan

### Unit Tests (`tests/`)

| Module | Test | Expected |
|---|---|---|
| `frame_sampler` | Video at 1fps, 10s clip | ~10 frames returned |
| `frame_sampler` | Folder with 100 images, every_nth=5 | 20 frames returned |
| `frame_sampler` | Invalid path | Raises `FileNotFoundError` |
| `detector` | Single image, known class | bbox + score returned |
| `detector` | Threshold 0.99 on test image | 0 detections |
| `embedder` | Single crop 224×224 | shape (C,) numpy array |
| `embedder` | Batch of 50 crops | shape (50, C), no OOM |
| `clusterer` | 200 random embeddings → HDBSCAN | Labels array, no exception |
| `clusterer` | All-same embedding → KMeans k=2 | Degenerate clusters handled |
| `coco_writer` | 3 images, 5 annotations | Valid COCO JSON structure |
| `coco_writer` | 0 annotations | Valid empty COCO JSON |

### Integration Tests

1. **Full pipeline test (smoke):** Provide a 5-second test video + class names → run full pipeline → verify COCO JSON file produced with ≥1 annotation.
2. **Discard logic:** Mark all clusters as discard → verify `active_detections` is empty after confirm.
3. **Mining test:** Keep one cluster → run mining → verify `mined_detections` is non-empty.
4. **Gallery rendering:** With 50 annotated frames, render page 1 → verify no Streamlit error.

### Manual UI Tests

- [ ] Upload `.mp4` video → frames appear in gallery
- [ ] Change fps slider → frame count updates correctly
- [ ] Run detection → correct number of detections shown
- [ ] Move threshold slider → detection count changes
- [ ] Cluster → prototype images look visually coherent per cluster
- [ ] Discard cluster → stats table shows negative delta
- [ ] Keep cluster → mining adds new annotations
- [ ] Export → valid JSON downloadable, images present in `output/images/`
- [ ] Open output folder button works

---

## 8. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **WeDetect requires Chinese class names** by default (XLM-RoBERTa) | High | Auto-translate English → Chinese using `transformers` MarianMT or `deep_translator`; OR use WeDetect-Ref which supports both; document in UI |
| **WeDetect-Ref (2B/4B) is too large for GPU with <16 GB VRAM** | Medium | Make WeDetect-Ref optional; fall back to base WeDetect with translated prompts |
| **C-RADIOv4 download (~1.7 GB) slow or fails** | Low | Cache in `~/.cache/huggingface`; show download progress bar; document model pre-download step |
| **OOM during embedding 1000s of crops** | Medium | Batch crops in groups of 16–32; use `torch.no_grad()` + `.half()` for FP16 inference |
| **WeDetect mmdet stack install conflicts** | High | Pin exact versions from WeDetect README; use isolated conda env; test install before any code |
| **HDBSCAN on large embedding matrices (>50k)** | Low | Subsample to 10k for clustering; use approximate ANN (faiss) if needed |
| **Streamlit re-runs wipe state** | High | All pipeline outputs stored in `st.session_state`; use `@st.cache_resource` for model loading |
| **Video with no detections above threshold** | Medium | Show warning; suggest lowering threshold |
| **Prototype bbox drawn incorrectly (off-by-one)** | Low | Unit test bbox drawing with known fixture image |
| **COCO JSON invalid (missing required fields)** | Low | Validate with `pycocotools.coco.COCO()` after writing |
| **WeDetect not yet pip-installable** | Medium | Clone repo, `pip install -e .`; pin commit hash in README |

---

## 9. Model Pre-download Checklist

Before first run, pre-download models to avoid slow cold starts:

```bash
conda activate image-annotation

# Pre-download C-RADIOv4-SO400M (~1.7 GB)
python -c "
from transformers import AutoModel, CLIPImageProcessor
AutoModel.from_pretrained('nvidia/C-RADIOv4-SO400M', trust_remote_code=True)
CLIPImageProcessor.from_pretrained('nvidia/C-RADIOv4-SO400M')
print('C-RADIO OK')
"

# Download WeDetect checkpoints
# (See WeDetect README for HuggingFace checkpoint links)
# Place in: ./checkpoints/wedetect/

# Verify GPU available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 10. Overnight Execution Checklist

Use this checklist before leaving the app to run overnight on a large dataset:

- [ ] Conda env `image-annotation` is active
- [ ] GPU memory is ≥ 8 GB free (`nvidia-smi` check)
- [ ] Input video / folder path is correct and readable
- [ ] WeDetect config + checkpoint paths are set
- [ ] Output folder has sufficient disk space (estimate: ~10 MB/1000 frames)
- [ ] Detection class names entered (translated to Chinese if using base WeDetect)
- [ ] FPS / every-nth sampling configured to expected frame count
- [ ] Detection threshold reviewed (test on ~5 frames first)
- [ ] Clustering method + parameters set
- [ ] Streamlit is running in a `tmux` or `screen` session (not terminal that might close)
- [ ] App is accessible over Tailscale for mobile review
- [ ] `output/annotations.json` does NOT exist from a previous run (or output dir renamed)
- [ ] Reviewed cluster prototypes look visually reasonable before triggering mining
- [ ] Mining threshold set conservatively (≥ 0.75) to avoid false positives
- [ ] After export: verify JSON with `python -c "from pycocotools.coco import COCO; c=COCO('output/annotations.json'); print(len(c.anns), 'annotations')"`

**Run command:**
```bash
tmux new -s annotation
conda activate image-annotation
cd /home/kramer/.openclaw/workspace/image-annotation-tool
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

---

## 11. Phase Timeline Estimate

| Phase | Task | Estimated Time |
|---|---|---|
| 0 | Scaffold + env setup | 1–2 h |
| 1 | Frame sampler (video + folder) | 2–3 h |
| 2 | WeDetect detector wrapper | 3–4 h |
| 3 | C-RADIOv4 embedder | 2–3 h |
| 4 | Clustering + prototype selection | 2–3 h |
| 5 | Cluster review UI | 3–4 h |
| 6 | Discard logic | 1 h |
| 7 | Second-pass visual mining | 3–4 h |
| 8 | COCO export | 2 h |
| 9 | Stats + gallery | 2 h |
| — | Testing + bug fixes | 3–5 h |
| **Total** | | **~24–35 h** |

---

## 12. Open Questions (Decide Before Implementation)

1. **Language for class names:** Should the UI auto-translate to Chinese, or require the user to enter Chinese names? Recommend: accept English, auto-translate with MarianMT, show the translated names as info text.

2. **Second-pass mining method:** Pure embedding similarity (spatial RADIO features, fast) vs. WeDetect-Ref image-prompt (slower, GPU-heavy, more semantically accurate). Recommend: implement both, let user choose.

3. **Clustering algorithm default:** HDBSCAN is more principled (auto number of clusters, handles noise) but slower. KMeans is faster but requires specifying K. Recommend HDBSCAN as default.

4. **One-class vs multi-class annotation:** Does the user annotate one target class per session or multiple? Current design supports multiple via separate keep/discard per cluster, each cluster representing a visual category. Clarify if needed.

5. **Segmentation masks:** COCO supports both bbox and segmentation. Should the tool output only bboxes (simpler) or also attempt mask generation via SAM? Recommend bboxes only for v1.

6. **WeDetect model tier:** Base WeDetect (fast) or WeDetect-Uni (class-agnostic proposals)? Recommend starting with WeDetect-Base; add Uni support in v2.
