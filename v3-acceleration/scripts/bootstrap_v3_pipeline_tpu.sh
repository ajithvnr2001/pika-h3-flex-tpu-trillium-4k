#!/usr/bin/env bash
set -e

echo "=============================================================================="
echo "Google Cloud TPU Trillium (v6e-4) — Pika H3 Flex v3 Lossless Accelerated Suite"
echo "=============================================================================="

export DEBIAN_FRONTEND=noninteractive
export PJRT_DEVICE=TPU
export HF_XET_HIGH_PERFORMANCE=1

echo -e "\n[Step 1/4] Preparing High-Speed 150GB RAM Disk Storage (/mnt/pika_data -> /dev/shm/pika_data)..."
sudo mount -o remount,size=150G /dev/shm || true
sudo mkdir -p /dev/shm/pika_data /dev/shm/pika_data/xla_cache /dev/shm/pika_data/outputs
sudo chmod -R 777 /dev/shm/pika_data
sudo rm -rf /mnt/pika_data
sudo ln -sfn /dev/shm/pika_data /mnt/pika_data
df -h /mnt/pika_data

echo -e "\n[Step 2/4] Installing System Media Packages (ffmpeg, ffprobe, libsndfile1)..."
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libopenblas-dev libomp5 libsndfile1 >/dev/null 2>&1 || true
ffmpeg -version | head -n 1

echo -e "\n[Step 3/4] Upgrading pip & Installing PyTorch/XLA and Fast Safetensors Tooling..."
pip3 install --quiet --no-cache-dir --upgrade pip

pip3 install --quiet --no-cache-dir \
    torch~=2.6.0 \
    torch_xla[tpu]~=2.6.0 \
    "huggingface_hub>=0.25.0" \
    "safetensors>=0.4.5" \
    "pillow>=10.4.0" \
    "numpy" \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html

echo -e "\n[Step 4/4] Executing Pika H3 Flex v3 Lossless Accelerated Pipeline..."
python3 -u ~/generate_pika_v3_accelerated_tpu.py

echo -e "\n=============================================================================="
echo "Pika H3 Flex v3 Lossless Accelerated Pipeline on Cloud TPU Trillium Completed!"
echo "=============================================================================="
