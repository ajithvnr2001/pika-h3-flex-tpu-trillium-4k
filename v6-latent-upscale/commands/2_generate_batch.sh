#!/usr/bin/env bash
# STEP 2: generate all clips of a jobs file in ONE process on the TPU, upscaled to 1664x960 by default.
#
#   commands/2_generate_batch.sh <jobs.txt>
#
# jobs.txt, one clip per line (lines starting with # are ignored):
#   <prompt.txt> <name> [duration 5-15] [first_frame] [key=value ...]
#   first_frame:  -            text only
#                 @<name>      continue from the last frame of clip <name> earlier in this file (made on the TPU)
#                 path.png     a local image (copied to the TPU)
#   options:      refine_from=250 (default: 3 refine steps, closest to the 480p draft) | 500 (4 steps), seed=N
# Example:
#   prompts/shot01.txt  shot01  5
#   prompts/shot02.txt  shot02  5  @shot01
#
# Extra generator flags: GEN_ARGS (default "--upscale 2"). GEN_ARGS="" gives plain 480p.
set -euo pipefail
. "$(dirname "$0")/common.sh"
JOBS="${1:?jobs file}"
GEN_ARGS="${GEN_ARGS---upscale 2}"

remote_jobs=$(mktemp)
while read -r prompt name rest; do
  [ -z "${prompt:-}" ] && continue
  case "$prompt" in \#*) continue;; esac
  dur=5; img="-"; opts=""
  for tok in ${rest:-}; do
    case "$tok" in
      *=*) opts+=" $tok" ;;
      @*|-) img="$tok" ;;
      *.png|*.jpg|*.jpeg) tpu_scp "$tok" "$TPU":~/frame_"$name".png; img="/root/frame_$name.png" ;;
      *) dur="$tok" ;;
    esac
  done
  tpu_scp "$prompt" "$TPU":~/prompt_"$name".txt
  echo "/root/prompt_$name.txt /dev/shm/pika/outputs/$name.mp4 $img $dur$opts" >> "$remote_jobs"
done < "$JOBS"
tpu_scp "$remote_jobs" "$TPU":~/jobs.txt; rm -f "$remote_jobs"

tpu_ssh "set -o pipefail; . ~/pika-venv/bin/activate; export PJRT_DEVICE=TPU XLA_CACHE_DIR=/dev/shm/pika/xla_cache; \
  python ~/generate_pika_tpu.py --jobs ~/jobs.txt $GEN_ARGS 2>&1 | tee ~/run_batch.log | grep --line-buffered -E '^\[|Error|Traceback'"

missing=0
while read -r prompt name _; do
  [ -z "${prompt:-}" ] && continue
  case "$prompt" in \#*) continue;; esac
  tpu_ssh "test -s /dev/shm/pika/outputs/$name.mp4" || { echo "MISSING: $name.mp4"; missing=1; }
done < "$JOBS"
[ "$missing" = 0 ] || { echo "STEP 2 FAILED for some clips (preempted? check: gcloud compute tpus tpu-vm describe $TPU ${G[*]})"; exit 1; }
echo "STEP 2 done"
