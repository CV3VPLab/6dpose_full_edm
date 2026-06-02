#!/usr/bin/env bash
set -e

# Full setup for the professor baseline + ONNXRuntime TensorRT path.
#
# Usage:
#   bash setup_full_edm.sh
#
# Optional overrides:
#   ENV_NAME=6dpose_test REPO_DIR=$PWD bash setup_full_edm.sh

ENV_NAME="${ENV_NAME:-6dpose_test}"
REPO_DIR="${REPO_DIR:-$PWD}"
SAM2_DIR="${SAM2_DIR:-$REPO_DIR/sam2_repo}"

CONDA_BASE="$(conda info --base)"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  ENV      : $ENV_NAME  (Python 3.10)"
echo "  torch    : 2.5.1+cu118"
echo "  REPO     : $REPO_DIR"
echo "============================================================"

# ── 1. conda env ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/8] Creating conda env '$ENV_NAME' if needed ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    conda env '$ENV_NAME' already exists, skipping create"
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/8] PyTorch 2.5.1+cu118 ..."
"$PIP" install \
    torch==2.5.1+cu118 \
    torchvision==0.20.1+cu118 \
    torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 3. Core packages ──────────────────────────────────────────────────────────
echo ""
echo ">>> [3/8] Core packages ..."
"$PIP" install \
    numpy==1.26.4 \
    scipy \
    h5py \
    opencv-python \
    pillow \
    matplotlib \
    tqdm \
    plyfile \
    joblib \
    lpips \
    einops \
    imageio \
    ninja

# ── 4. 6dpose packages ────────────────────────────────────────────────────────
echo ""
echo ">>> [4/8] 6dpose packages ..."
"$PIP" install \
    ultralytics==8.4.37 \
    kornia==0.8.2 \
    timm==1.0.26 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    iopath \
    portalocker \
    transformers \
    huggingface_hub \
    safetensors \
    tokenizers \
    psutil \
    polars \
    typer \
    rich \
    yacs \
    loguru

# ── 5. SAM2 ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> [5/8] SAM2 ..."
if [ ! -d "$SAM2_DIR" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
else
    echo "    sam2_repo already exists, skipping clone"
fi
"$PIP" install -e "$SAM2_DIR"

# ── 6. gsplat + GS CUDA submodules ───────────────────────────────────────────
echo ""
echo ">>> [6/8] gsplat + GS CUDA submodules ..."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
"$PIP" install --upgrade setuptools wheel
"$PIP" install gsplat==1.5.3

mkdir -p "$REPO_DIR/submodules"

if [ ! -d "$REPO_DIR/submodules/diff-gaussian-rasterization" ]; then
    git clone --recursive --branch dr_aa \
        https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
        "$REPO_DIR/submodules/diff-gaussian-rasterization"
fi
cd "$REPO_DIR/submodules/diff-gaussian-rasterization"
git submodule update --init --recursive
"$PIP" install --no-build-isolation -e .

if [ ! -d "$REPO_DIR/submodules/simple-knn" ]; then
    git clone https://gitlab.inria.fr/bkerbl/simple-knn.git \
        "$REPO_DIR/submodules/simple-knn"
fi
cd "$REPO_DIR/submodules/simple-knn"
"$PIP" install --no-build-isolation -e .

if [ ! -d "$REPO_DIR/submodules/fused-ssim" ]; then
    git clone https://github.com/rahul-goel/fused-ssim.git \
        "$REPO_DIR/submodules/fused-ssim"
fi
cd "$REPO_DIR/submodules/fused-ssim"
"$PIP" install --no-build-isolation -e .

cd "$REPO_DIR"

# ── 7. ONNXRuntime TensorRT runtime deps ─────────────────────────────────────
echo ""
echo ">>> [7/8] ONNXRuntime + TensorRT runtime dependencies ..."
"$PIP" install \
    onnxruntime-gpu==1.18.0 \
    tensorrt-cu11 \
    tensorrt-lean-cu11 \
    tensorrt-dispatch-cu11

# ── 8. Runtime library path helper + verify ──────────────────────────────────
echo ""
echo ">>> [8/8] Runtime library path + verify ..."
source "$REPO_DIR/setup_edm_env.sh"
"$PYTHON" - <<'PYEOF'
import sys
print(f"Python  : {sys.version.split()[0]}")

import torch
print(f"PyTorch : {torch.__version__} CUDA={torch.version.cuda} GPU={torch.cuda.is_available()}")

import cv2; print(f"OpenCV  : {cv2.__version__}")
import ultralytics; print(f"YOLO    : {ultralytics.__version__}")
import kornia; print(f"Kornia  : {kornia.__version__}")
import timm; print(f"timm    : {timm.__version__}")
import sam2; print("SAM2    : OK")
import gsplat; print(f"gsplat  : {gsplat.__version__}")
import onnxruntime as ort
print(f"ORT     : {ort.__version__} providers={ort.get_available_providers()}")

from diff_gaussian_rasterization import GaussianRasterizer
print("diff-gaussian-rasterization : OK")
import simple_knn; print("simple-knn : OK")
import fused_ssim; print("fused-ssim : OK")

print("\n============================================================")
print("  All checks passed")
print("  Before running later shells: conda activate 6dpose_test && source setup_edm_env.sh")
print("============================================================")
PYEOF

echo ""
echo "NOTE:"
echo "  This script installs code dependencies only."
echo "  Place model/data files separately:"
echo "    weights/best_3cls_v2.pt"
echo "    weights/sam2.1_hiera_tiny.pt"
echo "    weights/dino_vits14_224.onnx"
echo "    weights/edm_outdoor_w672_h672_topk2469.onnx"
echo "  EDM repo is not required for normal EDM_TRT inference."
echo "  Clone EDM separately only if you need to regenerate the ONNX."
