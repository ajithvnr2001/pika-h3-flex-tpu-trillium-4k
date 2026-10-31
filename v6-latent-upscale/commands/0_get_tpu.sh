#!/usr/bin/env bash
# STEP 0: get a v6e-4 TPU when capacity is short. Tries every zone with quota; if none has capacity, queues a
# request in $ZONE and waits until Google provides it. Prints "ZONE=<zone>" at the end: use that ZONE for steps 1-5.
#   SPOT=0 commands/0_get_tpu.sh      (on-demand)      SPOT=1 commands/0_get_tpu.sh   (Spot, cheaper, preemptible)
set -uo pipefail
. "$(dirname "$0")/common.sh"
spot=(); [ "$SPOT" = "1" ] && spot=(--spot)
ZONES="${ZONES:-europe-west4-a us-east5-b us-east5-a us-south1-a us-west1-c asia-northeast1-b asia-east1-c asia-south1-b asia-southeast1-b southamerica-east1-c us-central1-a}"
for z in $ZONES; do
  o=$(timeout 420 gcloud compute tpus tpu-vm create "$TPU" --zone="$z" --project="$PROJECT" --accelerator-type="$ACCEL" \
        --version="$RUNTIME" "${spot[@]}" 2>&1 | tr '\r' '\n' | grep -oE 'Created tpu.*|ALREADY_EXISTS|no more capacity|Insufficient capacity|does not have permission|[A-Z_]{8,}' | head -1)
  echo "$(date -u +%T) $z: ${o:-?}"
  case "$o" in "Created tpu"*|ALREADY_EXISTS) echo "ZONE=$z"; exit 0;; esac
  timeout 40 gcloud compute tpus tpu-vm describe "$TPU" --zone="$z" --project="$PROJECT" >/dev/null 2>&1 && \
    timeout 600 gcloud compute tpus tpu-vm delete "$TPU" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1
done
QR="${TPU}-qr"
echo "$(date -u +%T) no zone had capacity; queueing $QR in $ZONE"
gcloud compute tpus queued-resources create "$QR" --node-id="$TPU" "${G[@]}" --accelerator-type="$ACCEL" \
  --runtime-version="$RUNTIME" --valid-until-duration=8h "${spot[@]}" 2>&1 | tail -1
while true; do
  st=$(timeout 60 gcloud compute tpus queued-resources describe "$QR" "${G[@]}" --format='value(state.state)' 2>/dev/null)
  echo "$(date -u +%T) queued: $st"
  [ "$st" = ACTIVE ] && { echo "ZONE=$ZONE"; exit 0; }
  case "$st" in FAILED|SUSPENDED) exit 1;; esac
  sleep 60
done
