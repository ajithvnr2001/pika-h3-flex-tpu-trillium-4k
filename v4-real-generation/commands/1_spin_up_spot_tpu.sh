#!/usr/bin/env bash
# STEP 1: create a Spot TPU v6e-4, copy the scripts, install everything and download the model.
# Needs: gcloud logged in, HF_TOKEN exported locally.   Takes ~6-8 min (create ~1-5 min, install ~3 min, download ~1 min).
set -euo pipefail
. "$(dirname "$0")/common.sh"
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"

if gcloud compute tpus tpu-vm describe "$TPU" "${G[@]}" --format='value(state)' >/dev/null 2>&1; then
  state=$(gcloud compute tpus tpu-vm describe "$TPU" "${G[@]}" --format='value(state)')
  if [ "$state" != "READY" ]; then
    echo "TPU $TPU is $state (Spot preemption?). Deleting and recreating."
    gcloud compute tpus tpu-vm delete "$TPU" "${G[@]}" --quiet
  fi
fi
if ! gcloud compute tpus tpu-vm describe "$TPU" "${G[@]}" >/dev/null 2>&1; then
  spot_flag=(); kind="on-demand"
  [ "$SPOT" = "1" ] && { spot_flag=(--spot); kind="Spot"; }
  echo "Creating $kind TPU $TPU ($ACCEL) in $ZONE ..."
  gcloud compute tpus tpu-vm create "$TPU" "${G[@]}" \
      --accelerator-type="$ACCEL" --version="$RUNTIME" "${spot_flag[@]}"
fi

tpu_scp "$V4_DIR/scripts/bootstrap_tpu.sh" "$V4_DIR/scripts/generate_pika_tpu.py" "$V4_DIR/scripts/run.sh" "$TPU":~/
# The token goes into a 600-mode file on the VM; it is never put on a command line that gets logged.
printf 'export HF_TOKEN=%q\n' "$HF_TOKEN" | tpu_ssh 'umask 077; cat > ~/.hf.env'
tpu_ssh 'chmod +x ~/*.sh; . ~/.hf.env; ~/bootstrap_tpu.sh 2>&1 | tee ~/bootstrap.log | grep -E "^\[|Downloaded|torch "'
echo "STEP 1 done: $TPU is ready."
