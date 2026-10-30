#!/usr/bin/env bash
# STEP 2 (batch): generate many videos in ONE process on the TPU, so the model is loaded and compiled once.
# The first clip pays the compile (~1.5-2 min); every later clip with the same duration skips it.
#
#   commands/2_generate_batch.sh <jobs.txt>
#
# jobs.txt, one video per line (lines starting with # are ignored):
#   <prompt.txt> <name> [duration 5-15] [first_frame.png]
# Example:
#   prompts/prompt_5s.txt            kerala_5s     5
#   prompts/prompt_rajasthan_5s.txt  rajasthan_5s  5
#   prompts/prompt_extend.txt        kerala_ext    5   outputs/kerala_5s_last.png
#
# Note: a line with a first frame must use a frame that already exists locally before the batch starts.
set -euo pipefail
. "$(dirname "$0")/common.sh"
JOBS="${1:?jobs file}"

remote_jobs=$(mktemp)
while read -r prompt name dur img _; do
  [ -z "${prompt:-}" ] && continue
  case "$prompt" in \#*) continue;; esac
  tpu_scp "$prompt" "$TPU":~/prompt_"$name".txt
  rimg="-"
  if [ -n "${img:-}" ]; then tpu_scp "$img" "$TPU":~/frame_"$name".png; rimg="/root/frame_$name.png"; fi
  echo "/root/prompt_$name.txt /dev/shm/pika/outputs/$name.mp4 $rimg ${dur:-5}" >> "$remote_jobs"
done < "$JOBS"
tpu_scp "$remote_jobs" "$TPU":~/jobs.txt; rm -f "$remote_jobs"

tpu_ssh "set -o pipefail; . ~/pika-venv/bin/activate; export PJRT_DEVICE=TPU XLA_CACHE_DIR=/dev/shm/pika/xla_cache; \
  python ~/generate_pika_tpu.py --jobs ~/jobs.txt 2>&1 | tee ~/run_batch.log | grep --line-buffered -E '^\[|Error|Traceback'"

missing=0
while read -r prompt name _; do
  [ -z "${prompt:-}" ] && continue
  case "$prompt" in \#*) continue;; esac
  tpu_ssh "test -s /dev/shm/pika/outputs/$name.mp4" || { echo "MISSING: $name.mp4"; missing=1; }
done < "$JOBS"
[ "$missing" = 0 ] || { echo "STEP 2 FAILED for some jobs (preempted? check: gcloud compute tpus tpu-vm describe $TPU ${G[*]})"; exit 1; }
echo "STEP 2 (batch) done"
