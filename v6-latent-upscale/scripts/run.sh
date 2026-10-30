#!/usr/bin/env bash
# Runs on the TPU VM. Usage: ~/run.sh <prompt_file> <output.mp4> [first_frame.png] [duration]
set -eo pipefail
. ~/pika-venv/bin/activate
export PJRT_DEVICE=TPU XLA_CACHE_DIR=/dev/shm/pika/xla_cache
img=(); [ -n "$3" ] && img=(--image "$3")
python ~/generate_pika_tpu.py --prompt-file "$1" --output "$2" --duration "${4:-5}" "${img[@]}"
