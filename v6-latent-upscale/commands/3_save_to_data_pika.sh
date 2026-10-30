#!/usr/bin/env bash
# STEP 3: copy every generated MP4 + its timing JSON from the TPU into $LOCAL_OUT (default: this folder's outputs/),
# and save the last frame of each clip (input for an extension run).
set -euo pipefail
. "$(dirname "$0")/common.sh"
mkdir -p "$LOCAL_OUT"
tpu_scp "$TPU:/dev/shm/pika/outputs/*" "$LOCAL_OUT/"
for f in "$LOCAL_OUT"/*.mp4; do
  case "$f" in *_4k.mp4|*_10s.mp4) continue;; esac
  ffmpeg -v error -y -sseof -0.1 -i "$f" -frames:v 1 -update 1 "${f%.mp4}_last.png"
done
ls -lh "$LOCAL_OUT"
echo "STEP 3 done: files are in $LOCAL_OUT"
