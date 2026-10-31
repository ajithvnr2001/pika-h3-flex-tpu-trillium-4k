#!/usr/bin/env bash
# STEP 4: delete the TPU (and its queued request, if step 0 queued one) so billing stops; list what is left.
set -euo pipefail
. "$(dirname "$0")/common.sh"
if gcloud compute tpus queued-resources describe "${TPU}-qr" "${G[@]}" >/dev/null 2>&1; then
  gcloud compute tpus queued-resources delete "${TPU}-qr" "${G[@]}" --quiet --force
fi
gcloud compute tpus tpu-vm delete "$TPU" "${G[@]}" --quiet 2>/dev/null || true
echo "Remaining TPUs in $ZONE:"; gcloud compute tpus tpu-vm list "${G[@]}" --format='table(name,state,acceleratorType)'
echo "STEP 4 done."
