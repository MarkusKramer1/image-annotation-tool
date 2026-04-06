# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is an **Image Annotation Tool** — a multi-page Streamlit app for semi-automated COCO-format dataset creation from video. The pipeline has 4 stages: Data Extraction, Base Class Detection (WeDetect), Exact Class Detection (DINOv2 + clustering), and Segmentation Masks (SAM 2.1).

### Environment

- **Runtime**: Conda environment `image-annotation` (Python 3.11)
- **Conda init**: `source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate image-annotation`
- The VM has **no GPU**; PyTorch is installed as CPU-only (`torch==2.5.1+cpu`). The Streamlit UI and Data Extraction stage work fine on CPU. Stages 2–4 (detection/segmentation) require GPU and will not run inference in this environment.
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
- WeDetect model checkpoints are **not** included in the repo. They must be downloaded from HuggingFace and placed in `WeDetect/checkpoints/`. See `README.md` for details.
- The `setup.sh` script expects conda and CUDA GPU drivers. On CPU-only VMs, follow the manual install steps documented here instead.
