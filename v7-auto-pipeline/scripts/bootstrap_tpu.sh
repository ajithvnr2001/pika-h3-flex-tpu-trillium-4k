#!/usr/bin/env bash
# Prepares a fresh Cloud TPU v6e-4 VM (runtime v2-alpha-tpuv6e) to run Pika H3 Flex for real.
# Safe to re-run: every step skips itself when it is already done, so a second video on the
# same VM does not reinstall packages or re-download weights.
#
# Required env: HF_TOKEN (Hugging Face token with access to Pika-Labs/Pika-H3-Flex)
set -euo pipefail

MODEL_REVISION="${MODEL_REVISION:-3ff166447d5edb1406107ab8370c42000ce173e4}"
DIFFUSERS_COMMIT="d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f"   # pinned by Pika's official examples
WORK=/dev/shm/pika                                              # RAM disk: the boot disk is too small for ~150 GB
VENV="$HOME/pika-venv"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "1/4 RAM disk"
sudo mount -o remount,size=420G /dev/shm
mkdir -p "$WORK/model" "$WORK/outputs" "$WORK/xla_cache"
df -h /dev/shm | tail -1

log "2/4 System packages (ffmpeg)"
if ! command -v ffmpeg >/dev/null; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
fi
ffmpeg -version | head -1

log "3/4 Python 3.12 venv with torch/torch_xla 2.9 and the pinned Pika stack"
if [ ! -f "$VENV/.pika_ready" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 "$VENV"
  # CPU build of torch: the TPU is driven by torch_xla, so the multi-GB CUDA wheels are not needed.
  uv pip install --python "$VENV/bin/python" torch==2.9.0 torchvision==0.24.0 \
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python "$VENV/bin/python" "torch_xla[tpu]==2.9.0" \
      -f https://storage.googleapis.com/libtpu-releases/index.html
  uv pip install --python "$VENV/bin/python" \
      "diffusers @ git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_COMMIT}" \
      transformers==5.15.0 peft==0.20.0 accelerate==1.14.0 huggingface-hub==1.32.0 \
      safetensors==0.8.0 av==18.1.0 pillow==12.3.0 numpy==2.5.2 hf_xet
  # jax is only used to build the Pallas flash-attention kernel (torch_xla 2.9's "pallas" extra pins 0.7.1).
  uv pip install --python "$VENV/bin/python" jax==0.7.1 jaxlib==0.7.1
  touch "$VENV/.pika_ready"
fi
# torch_xla's _XLAC links against libpython3.12.so, which lives next to uv's interpreter.
PYLIB="$(dirname "$(dirname "$(readlink -f "$VENV/bin/python")")")/lib"
grep -q "LD_LIBRARY_PATH" "$VENV/bin/activate" || echo "export LD_LIBRARY_PATH=\"$PYLIB:\${LD_LIBRARY_PATH:-}\"" >> "$VENV/bin/activate"
export LD_LIBRARY_PATH="$PYLIB:${LD_LIBRARY_PATH:-}"
"$VENV/bin/python" -c "import torch, torch_xla, diffusers, transformers; print('torch', torch.__version__, '| torch_xla', torch_xla.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)"

log "4/4 Model weights (only the t2va/fl2va parts, ~147 GB) into $WORK/model"
if [ ! -f "$WORK/model/.complete-${MODEL_REVISION}" ]; then
  : "${HF_TOKEN:?HF_TOKEN must be set}"
  t0=$(date +%s)
  # One --include per pattern: passing several patterns after a single --include dropped the first one.
  inc=()
  for p in "transformer/*" "text_encoder/*" "vae/*" "audio_vae/*" "fl2va/*" "scheduler/*" \
           "audio_scheduler/*" "tokenizer/*" "processor/*" "modular_model_index.json" "LICENSE" "NOTICE"; do
    inc+=(--include "$p")
  done
  HF_XET_HIGH_PERFORMANCE=1 "$VENV/bin/hf" download Pika-Labs/Pika-H3-Flex \
      --revision "$MODEL_REVISION" --local-dir "$WORK/model" --max-workers 32 "${inc[@]}"
  for d in transformer text_encoder vae audio_vae fl2va; do
    ls "$WORK/model/$d"/*.safetensors >/dev/null 2>&1 || { echo "MISSING $d weights"; exit 1; }
  done
  t1=$(date +%s)
  size=$(du -sb "$WORK/model" | cut -f1)
  echo "Downloaded $(( size / 1000000000 )) GB in $(( t1 - t0 )) s"
  touch "$WORK/model/.complete-${MODEL_REVISION}"
fi
du -sh "$WORK/model"

log "5/5 Latent upscaler (LBH-123-AI/Minimax_h3_latent_Upscaler v1, bf16, 691 MB)"
if [ ! -s "$WORK/upscaler/model.safetensors" ]; then
  mkdir -p "$WORK/upscaler"
  "$VENV/bin/hf" download LBH-123-AI/Minimax_h3_latent_Upscaler \
      minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
      --revision 3f941d5d182014dd5c0a5e16330420ee2d4aa0c6 --local-dir "$WORK/upscaler/repo"
  ln -sf "$WORK/upscaler/repo/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors" \
      "$WORK/upscaler/model.safetensors"
fi
ls -lL "$WORK/upscaler/model.safetensors"
log "Bootstrap done"
