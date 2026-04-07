# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is an **Image Annotation Tool** — a multi-page Streamlit app for semi-automated COCO-format dataset creation from video. The pipeline has 4 stages: Data Extraction, Base Class Detection (WeDetect), Exact Class Detection (DINOv2 + clustering), and Segmentation Masks (SAM 2.1).

### Environment

- **Runtime**: Conda environment `image-annotation` (Python 3.11)
- **Conda init**: `source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate image-annotation`
- The VM has **no GPU**; PyTorch is installed as CPU-only (`torch==2.5.1+cpu`). The Streamlit UI, Data Extraction, and Base Class Detection (WeDetect) all work on CPU. Detection inference takes ~20-30s for 13 frames with the tiny model. Use the `tiny` model size for faster CPU inference. Stages 3–4 (DINOv2 clustering, SAM2 segmentation) also support CPU fallback but are slower.
- mmcv is installed from the OpenMMLab CPU wheel index (`torch2.4.0` index works for torch 2.5.1 CPU). After installing mmdet, the version guard in `mmdet/__init__.py` must be patched (change `mmcv_maximum_version` from `'2.2.0'` to `'2.3.0'`); see `setup.sh` for the exact patch.

### Running the app

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate image-annotation
cd /workspace
streamlit run app.py --server.headless true --server.port 8501
```

The app runs on http://localhost:8501.

### Lint / Tests

- No lint config (pyproject.toml, ruff, flake8, etc.) or test framework is configured in this repo.
- Basic validation: `python -m py_compile <file>` for all `.py` files under `app.py`, `pages/`, and `src/`.

### Key caveats

- `setuptools<72` is needed in the conda env so `pkg_resources` is available (required by mmcv/mmdet build tooling).
- WeDetect model checkpoints are **not** included in the repo. They must be downloaded from HuggingFace and placed in `WeDetect/checkpoints/`. The `wedetect_tiny.pth` checkpoint is the smallest (~1.2 GB) and works best on CPU. Download: `huggingface_hub.hf_hub_download(repo_id='fushh7/WeDetect', filename='wedetect_tiny.pth', local_dir='WeDetect/checkpoints')`.
- The `setup.sh` script expects conda and CUDA GPU drivers. On CPU-only VMs, follow the manual install steps documented here instead.
- The Base Class Detection page runs `annotation_runner.py` as a subprocess. If detection seems slow via the UI, test from CLI first: `python src/annotation_runner.py --wedetect-dir WeDetect --config config/wedetect_tiny.py --checkpoint checkpoints/wedetect_tiny.pth --images-dir data/<dataset>/images/default --classes "car,person" --output-json data/<dataset>/annotations/base_detection.json --vis-dir data/<dataset>/_vis --threshold 0.05 --topk 50 --adaptive-threshold --image-root images/default`.
