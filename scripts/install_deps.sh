#!/bin/bash
set -e

# Use the created conda environment
CONDA_ENV=openpi_arx

# Get the openpi-arx root directory relative to this script (scripts/install_deps.sh)
OPENPI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "Starting installation in $CONDA_ENV..."
echo "OpenPi root: $OPENPI_ROOT"

# 1. Install PyTorch
echo "Installing PyTorch (>=2.5.1)..."
conda run -n $CONDA_ENV pip install "torch>=2.5.1" --index-url https://download.pytorch.org/whl/cu121

# 2. Install JAX with CUDA 12
echo "Installing JAX..."
conda run -n $CONDA_ENV pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Install other core dependencies
echo "Installing other dependencies..."
conda run -n $CONDA_ENV pip install \
    "augmax" \
    "dm-tree" \
    "einops" \
    "equinox" \
    "flatbuffers" \
    "flax" \
    "fsspec[gcs]" \
    "gym-aloha" \
    "imageio" \
    "jaxtyping" \
    "ml_collections" \
    "numpydantic" \
    "opencv-python" \
    "orbax-checkpoint" \
    "pillow" \
    "sentencepiece" \
    "tqdm-loggable" \
    "typing-extensions" \
    "tyro" \
    "wandb" \
    "filelock" \
    "beartype" \
    "treescope" \
    "transformers" \
    "rich" \
    "polars" \
    "pyrealsense2" \
    "zerorpc" \
    "gevent"

# 4. Install LeRobot from GitHub
echo "Installing LeRobot..."
conda run -n $CONDA_ENV pip install git+https://github.com/huggingface/lerobot.git

# 5. Install the openpi package itself
echo "Installing openpi..."
cd "$OPENPI_ROOT"
conda run -n $CONDA_ENV pip install -e . --no-deps
conda run -n $CONDA_ENV pip install -e packages/openpi-client --no-deps

echo "Installation complete!"
