# Image Annotation Tool

Semi-automated image annotation pipeline: extract frames from a **video file** or a
**ROS bag** (ROS1 `.bag` / ROS2 `.mcap`), run
[WeDetect](https://github.com/WeChatCV/WeDetect) open-vocabulary detection, refine
labels, and export a COCO-format dataset.

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
all dependencies (PyTorch, mmdet, mmcv, mmengine, streamlit, rosbags, etc.):

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

## Pipeline Overview

The tool is structured as a three-page Streamlit app.  Each page feeds into
the next; you must complete them in order for a given dataset.

| Page | Purpose |
|------|---------|
| **1 · Data Extraction** | Extract frames from a video file or ROS bag |
| **2 · Base Class Detection** | Run WeDetect open-vocabulary detection on the extracted frames |
| **3 · Exact Class Detection** | Refine labels: split into sub-classes or merge into super-classes |

Output for every dataset lives under `data/<dataset-name>/`:

```
data/<dataset-name>/
├── images/default/          # extracted frames
├── annotations/
│   ├── extraction.json      # extraction metadata
│   ├── instances_default.json   # COCO annotations (after page 2)
│   └── exact_detection.json     # refined annotations (after page 3)
├── _vis/                    # frames with detection overlays
└── discarded/               # rejected frames (optional, ROS bag only)
```

## Usage

### Page 1 — Data Extraction

Select the source type with the radio button at the top of the page.

#### Video file

1. Upload a `.mp4` or `.mov` file (or click **Test Extraction** to use the
   bundled test video)
2. Set **Extract every N-th frame** (e.g. `5` → ~6 fps for a 30 fps video)
3. Enter a dataset name
4. Click **Extract Frames**

#### ROS bag (ROS1 `.bag` / ROS2 `.mcap`)

The ROS bag workflow requires the `rosbags`, `ImageHash`, and `pyyaml`
packages, which are included in `environment.yml`.

**Providing bags**

- Enter the path to a parent directory — all `.bag` files and `.mcap`
  folders inside it (searched recursively) are discovered automatically.
- Or upload one or more bag files directly via the file uploader.
- Both sources can be combined.

**Image topic**

The app scans each bag and lists all image topics
(`sensor_msgs/Image`, `sensor_msgs/CompressedImage`) for selection.  If
auto-detection fails, enter the topic name manually (e.g.
`/camera/image_raw`).

Supported message encodings: `rgb8`, `bgr8`, `mono8`, `mono16`,
`bayer_*`, and any `CompressedImage` format decodable by OpenCV.

**Frame preview**

Expand the **Frame preview** panel to build an interactive slider over the
first N frames of any bag/topic combination before committing to extraction.

**Filtering options**

Use the **Quick presets** (Outdoor / Indoor) or configure filters manually:

| Filter | Description |
|--------|-------------|
| Sampling rate (fps) | Keep at most this many frames per second; `0` = keep all |
| Temporal bounds | Skip frames before / after a time offset from bag start |
| Max frames per bag | Hard cap on extracted frames per bag |
| Motion threshold | Discard frames with too little motion (static camera) |
| Blur gate | Discard frames below a Laplacian-variance sharpness threshold |
| Deduplication | Drop near-duplicate frames by perceptual hash (dHash) |
| Brightness | Reject over- or under-exposed frames |
| Contrast | Reject low-contrast frames |

Enable **Save discarded frames** to write rejected frames to
`data/<name>/discarded/` with a suffix indicating the reason
(`_motion`, `_blur`, `_duplicate`, `_lowcontrast`, `_dark`, `_bright`).

**360° fisheye undistortion (Kalibr omni-radtan)**

Enable **360° camera undistortion** in the *Additional options* expander
when working with wide-angle or omnidirectional cameras calibrated with
[Kalibr](https://github.com/ethz-asl/kalibr).  Upload a calibration YAML
with `left_camera` and `right_camera` sections (each containing
`camera_matrix_K`, `distortion_coeffs_D`, and `xi`).  The app splits each
frame into left/right halves, undistorts and rectifies each half
independently, and saves them as `_left.png` / `_right.png` pairs.

**Running extraction**

Select the bags and topic to extract from, give the dataset a name, apply
filter settings with **Apply filter settings**, then click **Extract
Frames**.  A per-bag progress bar and summary table are shown on completion.

### Page 2 — Base Class Detection

1. Select the dataset produced in page 1
2. Enter object class names (comma-separated, e.g. `robot,person,car`)
3. Choose model size and detection threshold
4. Click **Start Detection** to run WeDetect on all extracted frames
5. Browse live detection thumbnails and per-class counts

### Page 3 — Exact Class Detection (Label Refinement)

Two refinement modes are available:

- **Divide into Subclasses** — embed all crops of one category with DINOv2,
  cluster them (HDBSCAN or KMeans), assign precise sub-labels, and export
- **Union to Superclass** — select two or more categories to merge into a
  single compound bounding box / segmentation under a new label

Both modes display a gallery of full images with bounding-box overlays and
write their output to `exact_detection.json`.

## Requirements

- NVIDIA GPU with >= 4 GB VRAM (8 GB recommended)
- CUDA 12.4 compatible driver
- Conda (Miniconda or Anaconda)
