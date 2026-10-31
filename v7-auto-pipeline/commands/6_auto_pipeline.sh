#!/usr/bin/env bash
# AUTOMATED PIPELINE (survives Spot preemption): generate every clip of a jobs file, join them into one video ON THE
# TPU, make QA frames on the TPU, and copy everything to $LOCAL_OUT.
#
#   . /root/.secrets/pika.env        # HF_TOKEN is needed if the TPU has to be recreated
#   TPU=pika-europe-tpu ZONE=europe-west4-a \
#     nohup commands/6_auto_pipeline.sh jobs_pixar_1min.txt pixar_1min > logs/pixar_1min.log 2>&1 &
#   tail -f logs/pixar_1min.log
#
# What happens if Google takes the Spot TPU away in the middle:
#   * The generator runs DETACHED on the TPU (a dropped SSH connection does not stop or duplicate it). Every
#     $SYNC_EVERY s this script copies every FINISHED clip (<name>.mp4, <name>_last.png,
#     <name>.json) to $LOCAL_OUT. The .json is written last by the generator, so it marks a complete clip.
#   * When the generation run fails, the script checks the TPU. If it is gone / PREEMPTED / not READY, it deletes it and
#     recreates it with step 1 (create + install + model download, ~8 min). If no capacity, it waits and retries.
#     After $ONDEMAND_AFTER failed Spot attempts it switches to an on-demand TPU (set ONDEMAND_AFTER=0 to never do that).
#   * It then restarts ONLY the clips that are not in $LOCAL_OUT yet. A clip that continues a finished clip
#     (@<name>) gets that clip's saved <name>_last.png from $LOCAL_OUT, so the story continues exactly where it stopped.
#   * Lost work per preemption: the clip in progress (<= ~4.5 min) + TPU re-setup (~8 min) + first-clip compile (~3 min).
#   * Before joining, clips that exist only locally (made on an earlier TPU) are uploaded back to the TPU.
# Running the same command again later also resumes (finished clips in $LOCAL_OUT are skipped).
#
# Settings (env): MAX_ATTEMPTS (default 10 generation runs), ONDEMAND_AFTER (default 3), SYNC_EVERY (default 60 s),
# AUTO_TEARDOWN=1 (delete the TPU when the video is done, so billing stops; default 0), LOUDNORM (see step 5),
# plus TPU, ZONE, SPOT, GEN_ARGS, LOCAL_OUT from common.sh.
set -uo pipefail
. "$(dirname "$0")/common.sh"
JOBS="${1:?jobs file}"; FINAL="${2:?final video name}"
D="$(cd "$(dirname "$0")" && pwd)"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"; ONDEMAND_AFTER="${ONDEMAND_AFTER:-3}"; SYNC_EVERY="${SYNC_EVERY:-60}"
mkdir -p "$LOCAL_OUT"
ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] PIPELINE $*"; }
names() { grep -vE '^\s*(#|$)' "$JOBS" | awk '{print $2}'; }
done_local() { [ -s "$LOCAL_OUT/$1.json" ] && [ -s "$LOCAL_OUT/$1.mp4" ] && [ -s "$LOCAL_OUT/$1_last.png" ]; }
tpu_state() { gcloud compute tpus tpu-vm describe "$TPU" "${G[@]}" --format='value(state)' 2>/dev/null || echo GONE; }

# Copy finished clips (json present = complete) from the TPU to $LOCAL_OUT. Quiet; never fails the pipeline.
sync_once() {
  local files n
  files=$(tpu_ssh "cd /dev/shm/pika/outputs 2>/dev/null && ls" 2>/dev/null) || return 0
  for n in $(names); do
    done_local "$n" && continue
    grep -qx "$n.json" <<<"$files" || continue
    grep -qx "${n}_base.mp4" <<<"$files" && tpu_scp "$TPU:/dev/shm/pika/outputs/${n}_base.mp4" "$LOCAL_OUT/" >/dev/null 2>&1
    tpu_scp "$TPU:/dev/shm/pika/outputs/$n.mp4" "$TPU:/dev/shm/pika/outputs/${n}_last.png" "$LOCAL_OUT/" >/dev/null 2>&1 &&
      tpu_scp "$TPU:/dev/shm/pika/outputs/$n.json" "$LOCAL_OUT/" >/dev/null 2>&1 && say "saved locally: $n"
  done
  return 0
}

# Jobs that are not finished locally. "@prev" becomes the saved local frame when prev was made on an earlier run.
remaining_jobs() {
  local out="$1"; : > "$out"
  local prompt name rest tok line
  while read -r prompt name rest; do
    [ -z "${prompt:-}" ] && continue
    case "$prompt" in \#*) continue;; esac
    done_local "$name" && continue
    line="$prompt $name"
    for tok in ${rest:-}; do
      if [[ "$tok" == @* ]] && done_local "${tok#@}"; then tok="$LOCAL_OUT/${tok#@}_last.png"; fi
      line+=" $tok"
    done
    echo "$line" >> "$out"
  done < "$JOBS"
}

# Make sure a READY, set-up TPU exists. Recreates after preemption; waits and retries when there is no capacity.
ensure_tpu() {
  local st tries=0
  while true; do
    st=$(tpu_state)
    if [ "$st" = READY ] && tpu_ssh "test -f ~/pika-venv/bin/activate && test -s /dev/shm/pika/upscaler/model.safetensors" >/dev/null 2>&1; then
      return 0
    fi
    : "${HF_TOKEN:?TPU $TPU needs (re)creating but HF_TOKEN is not set; run: . /root/.secrets/pika.env}"
    say "TPU $TPU is $st (preempted or not set up) -> recreating (SPOT=$SPOT)"
    [ "$st" != GONE ] && [ "$st" != READY ] && gcloud compute tpus tpu-vm delete "$TPU" "${G[@]}" --quiet >/dev/null 2>&1
    if "$D/1_spin_up_spot_tpu.sh"; then say "TPU $TPU ready again"; return 0; fi
    tries=$((tries + 1))
    if [ "$SPOT" = 1 ] && [ "$ONDEMAND_AFTER" -gt 0 ] && [ "$tries" -ge "$ONDEMAND_AFTER" ]; then
      say "Spot failed $tries times -> switching to on-demand"; export SPOT=0
    fi
    say "could not get TPU (attempt $tries); retrying in 5 min"
    gcloud compute tpus tpu-vm delete "$TPU" "${G[@]}" --quiet >/dev/null 2>&1
    sleep 300
  done
}

say "start: $JOBS -> $FINAL.mp4 on $TPU ($ZONE), $(names | wc -l) clips, outputs -> $LOCAL_OUT"
attempt=0; code_fail=0
while true; do
  rem=$(mktemp); remaining_jobs "$rem"
  left=$(grep -c . "$rem"); [ "$left" = 0 ] && { rm -f "$rem"; break; }
  attempt=$((attempt + 1))
  [ "$attempt" -gt "$MAX_ATTEMPTS" ] && { say "FAILED: $left clips still missing after $MAX_ATTEMPTS runs"; exit 1; }
  say "run $attempt: $left clip(s) to make"
  ensure_tpu
  tpu_ssh "pkill -f '[g]enerate_pika_tpu.py'" >/dev/null 2>&1 || true
  rc=""
  if DETACH=1 "$D/2_generate_batch.sh" "$rem" 2>/dev/null | grep -q "started detached"; then
    # Poll: copy finished clips, show progress, stop when the TPU run ends (~/batch.rc) or the TPU disappears.
    last=""
    while sleep "$SYNC_EVERY"; do
      sync_once
      st=$(tpu_state); [ "$st" = READY ] || break
      prog=$(tpu_ssh "grep -E '^\[.*(Job [0-9]|DONE|Error|Traceback)' ~/run_batch.log | tail -1" 2>/dev/null)
      [ -n "$prog" ] && [ "$prog" != "$last" ] && { echo "$prog"; last="$prog"; }
      # batch.rc = finished with that exit code; lock free and no rc = the run died (killed / crashed).
      rc=$(tpu_ssh "cat ~/batch.rc 2>/dev/null || { flock -n /tmp/pika_batch.lock true && echo died; }" 2>/dev/null | tr -dc '0-9a-z')
      [ -n "$rc" ] && break
    done
    sync_once
  fi
  rm -f "$rem"
  [ "$rc" = 0 ] && continue
  st=$(tpu_state)
  if [ "$st" = READY ]; then
    code_fail=$((code_fail + 1))
    say "generation failed but TPU is READY (rc=${rc:-none}); retry $code_fail/2, see ~/run_batch.log on the TPU"
    [ "$code_fail" -ge 2 ] && { say "FAILED twice on a healthy TPU: fix the error, then rerun this command to resume"; exit 1; }
  else
    say "TPU is $st: Spot preemption, will recreate and resume"
  fi
done
say "all clips are saved locally"

ensure_tpu
for n in $(names); do
  tpu_ssh "test -s /dev/shm/pika/outputs/$n.mp4" >/dev/null 2>&1 && continue
  say "uploading $n.mp4 back to the TPU for the join"
  tpu_ssh "mkdir -p /dev/shm/pika/outputs" >/dev/null 2>&1
  tpu_scp "$LOCAL_OUT/$n.mp4" "$TPU:/dev/shm/pika/outputs/" >/dev/null 2>&1 || { say "upload of $n failed"; exit 1; }
done
say "joining on TPU"
"$D/5_join_on_tpu.sh" "$JOBS" "$FINAL" || { say "join failed"; exit 1; }
say "QA frames on TPU"
tpu_ssh "~/qa_frames.sh $(names | tr '\n' ' ') $FINAL 2>&1 | grep SSIM" || echo "QA step failed (video is still fine)"
say "copying to $LOCAL_OUT"
"$D/3_save_to_data_pika.sh" > /dev/null
ls -lh "$LOCAL_OUT/$FINAL.mp4"
say "DONE: $LOCAL_OUT/$FINAL.mp4"
if [ "${AUTO_TEARDOWN:-0}" = 1 ]; then say "AUTO_TEARDOWN=1: deleting TPU $TPU"; "$D/4_tear_down_tpu.sh"; fi
