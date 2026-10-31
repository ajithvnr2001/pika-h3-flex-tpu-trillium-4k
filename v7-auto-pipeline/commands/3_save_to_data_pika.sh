#!/usr/bin/env bash
# STEP 3: copy the finished files (MP4s, JSON records, last-frame PNGs, joined videos) from the TPU to $LOCAL_OUT
# (default: this folder's outputs/). Nothing is processed locally.
set -euo pipefail
. "$(dirname "$0")/common.sh"
mkdir -p "$LOCAL_OUT"
tpu_scp --recurse "$TPU:/dev/shm/pika/outputs/*" "$LOCAL_OUT/"
ls -lh "$LOCAL_OUT"
echo "STEP 3 done: files are in $LOCAL_OUT"
