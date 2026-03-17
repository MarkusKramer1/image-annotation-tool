#!/bin/bash
# setup.sh — one-shot environment setup for image-annotation-tool
# Usage: bash setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="image-annotation"

# ── Find conda ────────────────────────────────────────────────────────────────
CONDA_BASE=""
for d in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$d/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$d"
        break
    fi
done
if [ -z "$CONDA_BASE" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first." && exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Create conda environment ──────────────────────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '$ENV_NAME' already exists — skipping creation."
    echo "(To recreate from scratch: conda env remove -n $ENV_NAME)"
else
    echo "=== Creating conda environment '$ENV_NAME' ==="
    conda env create -f "$SCRIPT_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

# ── Fix Pillow / libtiff ABI mismatch ─────────────────────────────────────────
CONDA_LIB="$CONDA_BASE/envs/$ENV_NAME/lib"
if [ ! -f "$CONDA_LIB/libtiff.so.5" ] && [ -f "/lib/x86_64-linux-gnu/libtiff.so.6" ]; then
    echo "  Symlinking libtiff.so.5 → system libtiff.so.6"
    ln -sf /lib/x86_64-linux-gnu/libtiff.so.6 "$CONDA_LIB/libtiff.so.5"
fi

# ── Detect PyTorch / CUDA versions ───────────────────────────────────────────
echo "=== Detecting PyTorch and CUDA version ==="
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
TORCH_MAJ_MIN=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')")
CUDA_FULL=$(python -c "import torch; print(torch.version.cuda or '')")
if [ -z "$CUDA_FULL" ]; then
    echo "ERROR: PyTorch has no CUDA support. Re-create the environment." && exit 1
fi
CUDA_TAG="cu$(echo "$CUDA_FULL" | tr -d '.')"
echo "  PyTorch $TORCH_FULL | CUDA tag $CUDA_TAG"

# ── openmim ───────────────────────────────────────────────────────────────────
echo "=== Installing openmim ==="
pip install openmim -q

# ── mmengine ──────────────────────────────────────────────────────────────────
echo "=== Installing mmengine 0.10.7 ==="
mim install mmengine==0.10.7 -q

# ── mmcv — find the closest pre-built wheel ───────────────────────────────────
echo "=== Installing mmcv ==="
MMCV_DONE=false
for TRY_CUDA in "$CUDA_TAG" "cu121" "cu118"; do
    for TRY_TORCH in "torch${TORCH_MAJ_MIN}.0" "torch2.5.0" "torch2.4.0" "torch2.3.0"; do
        IDX="https://download.openmmlab.com/mmcv/dist/${TRY_CUDA}/${TRY_TORCH}/index.html"
        CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$IDX" 2>/dev/null || echo "000")
        if [ "$CODE" = "200" ]; then
            echo "  wheel index: $IDX"
            pip install mmcv==2.2.0 -f "$IDX" -q && MMCV_DONE=true && break 2
        fi
    done
done
if [ "$MMCV_DONE" = false ]; then
    echo "  No pre-built wheel found — installing setuptools and building from source"
    pip install setuptools -q
    mim install mmcv==2.1.0 -q
fi

# ── mmdet ─────────────────────────────────────────────────────────────────────
echo "=== Installing mmdet 3.3.0 ==="
mim install mmdet==3.3.0 -q

# ── Patch mmdet version guard ─────────────────────────────────────────────────
echo "=== Patching mmdet version guard ==="
python -c "
import site, pathlib
for sp in site.getsitepackages():
    f = pathlib.Path(sp) / 'mmdet/__init__.py'
    if f.exists():
        txt = f.read_text()
        if \"mmcv_maximum_version = '2.2.0'\" in txt:
            f.write_text(txt.replace(\"mmcv_maximum_version = '2.2.0'\",
                                     \"mmcv_maximum_version = '2.3.0'\"))
            print(f'  patched: {f}')
        else:
            print(f'  already OK: {f}')
        break
"

# ── remaining Python deps (WeDetect needs transformers for XLM-RoBERTa) ──────
echo "=== Installing remaining dependencies ==="
pip install transformers accelerate einops timm webdataset supervision -q

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
cd "$SCRIPT_DIR"
PYTHONPATH="" python -c "
import sys, os
os.chdir('WeDetect')
sys.path.insert(0, '.')
from mmdet.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmcv.ops import nms
cfg = Config.fromfile('config/wedetect_base.py')
print('  mmcv CUDA ops ......... OK')
print('  WeDetect config ....... OK  (' + cfg.model.type + ')')
import streamlit, torch
print('  streamlit ............. ' + streamlit.__version__)
print('  torch ................. ' + torch.__version__)
"
echo ""
echo "=== Setup complete! ==="
echo "Start the app with:"
echo "  conda activate $ENV_NAME && streamlit run app.py"
