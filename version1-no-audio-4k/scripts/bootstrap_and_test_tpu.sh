#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "Google Cloud TPU Trillium (v6e) Remote Bootstrap & Benchmark Harness"
echo "======================================================================"

export DEBIAN_FRONTEND=noninteractive
export PJRT_DEVICE=TPU

echo -e "\n[Step 1/3] Checking Python and PyTorch/XLA Environment..."
if ! python3 -c "import torch, torch_xla" 2>/dev/null; then
    echo "PyTorch/XLA not detected in base system. Installing torch & torch_xla[tpu] matching wheels..."
    sudo apt-get update -qq && sudo apt-get install -y -qq libopenblas-dev libomp5 >/dev/null 2>&1 || true
    pip3 install --quiet --no-cache-dir torch~=2.6.0 torchvision torch_xla[tpu]~=2.6.0 \
        -f https://storage.googleapis.com/libtpu-releases/index.html \
        -f https://storage.googleapis.com/libtpu-wheels/index.html
fi

echo -e "\n[Step 2/3] Verifying Installed Libraries..."
python3 -c "
import torch
import torch_xla
print(f'PyTorch Version: {torch.__version__}')
print(f'Torch_XLA Version: {torch_xla.__version__}')
"

echo -e "\n[Step 3/3] Running Trillium v6e Hardware Diagnostics & MXU Benchmark..."
python3 ~/test_pika_tpu_remote.py

echo -e "\n======================================================================"
echo "End-to-End TPU Trillium Verification Run Complete!"
echo "======================================================================"
