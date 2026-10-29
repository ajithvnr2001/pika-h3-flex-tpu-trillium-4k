#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "Google Cloud TPU Trillium (v6e) Remote Video Bootstrap Harness"
echo "======================================================================"

export DEBIAN_FRONTEND=noninteractive
export PJRT_DEVICE=TPU

echo -e "\n[Step 1/4] Preparing High-Speed RAM Disk Storage (/mnt/pika_data -> /dev/shm/pika_data)..."
sudo mount -o remount,size=150G /dev/shm || true
sudo mkdir -p /dev/shm/pika_data
sudo chmod 777 /dev/shm/pika_data
sudo rm -rf /mnt/pika_data
sudo ln -sfn /dev/shm/pika_data /mnt/pika_data
df -h /mnt/pika_data

echo -e "\n[Step 2/4] Installing System Media Packages (ffmpeg, ffprobe)..."
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libopenblas-dev libomp5 >/dev/null 2>&1 || true
ffmpeg -version | head -n 1

echo -e "\n[Step 3/4] Installing PyTorch/XLA and Video Tooling..."
pip3 install --quiet --no-cache-dir torch~=2.6.0 torchvision torch_xla[tpu]~=2.6.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html

pip3 install --quiet --no-cache-dir \
    "huggingface_hub[hf_transfer]>=0.25.0" \
    "diffusers>=0.30.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.5" \
    "av>=12.0.0" \
    "pillow>=10.4.0" \
    "numpy"

echo -e "\n[Step 4/4] Executing Video Generation and Timing Pipeline..."
python3 -u ~/generate_pika_video_tpu.py

echo -e "\n======================================================================"
echo "Video Generation on Cloud TPU Trillium Completed Successfully!"
echo "======================================================================"
