# Image Annotation Tool

Semi-automated image annotation pipeline: upload a video, extract frames, run
[WeDetect](https://github.com/WeChatCV/WeDetect) open-vocabulary detection,
and export a COCO-format dataset.

## Setup

### 1. Clone and enter the repository

```bash
git clone <repo-url>
cd image-annotation-tool
```

### 2. Download WeDetect checkpoints

The WeDetect source code is included in `WeDetect/`, but the model weights
must be downloaded separately from
[HuggingFace](https://huggingface.co/fushh7/WeDetect).

Place the checkpoint in `WeDetect/checkpoints/`:

```bash
mkdir -p WeDetect/checkpoints
# Download wedetect_base.pth (~1.5 GB) from https://huggingface.co/fushh7/WeDetect
# and place it at WeDetect/checkpoints/wedetect_base.pth
```

If you have the `embedding_analysis` repo available locally, you can copy
the checkpoint directly:

```bash
cp /path/to/embedding_analysis/WeDetect/checkpoints/wedetect_base.pth WeDetect/checkpoints/
```

### 3. Create the conda environment

The `setup.sh` script creates a `image-annotation` conda environment with
all dependencies (PyTorch, mmdet, mmcv, mmengine, streamlit, etc.):

```bash
bash setup.sh
```

This takes 5-10 minutes on the first run. It handles:
- Conda env creation from `environment.yml`
- mmcv CUDA wheel auto-detection and installation
- mmdet version guard patching
- Pillow/libtiff ABI fix for Ubuntu 22+

### 4. Run the app

```bash
conda activate image-annotation
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Configure** detection parameters (model size, threshold, frame step)
2. **Upload** a video file (.mp4, .mov) or click **Test Pipeline** to use a
   bundled test video
3. **Enter** object class names (comma-separated, e.g. `robot,person,car`)
4. Click **Start Annotation** to run the pipeline
5. Watch live detection thumbnails and progress
6. View per-class detection counts when complete

Output is saved to `data/<video-name>/` with:
- `images/default/` — extracted frames
- `annotations/instances_default.json` — COCO-format annotations
- `_vis/` — frames with detection overlays

## Requirements

- NVIDIA GPU with >= 4 GB VRAM (8 GB recommended)
- CUDA 12.4 compatible driver
- Conda (Miniconda or Anaconda)
