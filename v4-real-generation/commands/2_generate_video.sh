#!/usr/bin/env bash
# STEP 2: generate a real video on the TPU from your prompt.
#
#   commands/2_generate_video.sh "<prompt text or path to a .txt file>" <name> [duration 5-15] [first_frame.png]
#
# Examples:
#   commands/2_generate_video.sh prompts/prompt_5s.txt kerala_5s
#   commands/2_generate_video.sh "integrated_multimodal_description: A red kite flying over a beach ..." kite 5
#   commands/2_generate_video.sh prompts/prompt_extend.txt kerala_ext 5 outputs/kerala_5s_last.png   # continue a clip
#
# Takes ~6 min for a 5 s clip at 480p (measured 360 s; ~3.5 min of that is one-time XLA compilation).
set -euo pipefail
. "$(dirname "$0")/common.sh"
PROMPT="${1:?prompt text or prompt file}"; NAME="${2:?output name}"; DUR="${3:-5}"; IMG="${4:-}"

tmp=$(mktemp)
if [ -f "$PROMPT" ]; then cp "$PROMPT" "$tmp"; else printf '%s\n' "$PROMPT" > "$tmp"; fi
tpu_scp "$tmp" "$TPU":~/prompt_"$NAME".txt; rm -f "$tmp"
remote_img="''"   # empty placeholder keeps the duration in the 4th position
if [ -n "$IMG" ]; then tpu_scp "$IMG" "$TPU":~/frame_"$NAME".png; remote_img="~/frame_$NAME.png"; fi

tpu_ssh "set -o pipefail; ~/run.sh ~/prompt_$NAME.txt /dev/shm/pika/outputs/$NAME.mp4 $remote_img $DUR 2>&1 | tee ~/run_$NAME.log | grep --line-buffered -E '^\[|denoise step|Error|Traceback'"
# Do not trust the ssh exit code alone (a preempted VM can drop the session): check the file exists.
tpu_ssh "test -s /dev/shm/pika/outputs/$NAME.mp4" || {
  echo "STEP 2 FAILED: no output. Check 'gcloud compute tpus tpu-vm describe $TPU ${G[*]}' (PREEMPTED? rerun step 1)."
  exit 1
}
echo "STEP 2 done: /dev/shm/pika/outputs/$NAME.mp4 on $TPU"
