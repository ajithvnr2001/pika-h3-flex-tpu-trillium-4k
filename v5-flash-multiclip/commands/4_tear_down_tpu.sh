#!/usr/bin/env bash
# STEP 4: delete the TPU so it stops billing, then confirm nothing is left in the zone.
# Everything on the TPU (weights, venv, outputs not yet saved by step 3) is lost.
set -euo pipefail
. "$(dirname "$0")/common.sh"
gcloud compute tpus tpu-vm delete "$TPU" "${G[@]}" --quiet || echo "$TPU was already gone"
echo "Remaining TPUs in $ZONE:"
gcloud compute tpus tpu-vm list "${G[@]}" --format='table(name,state,acceleratorType)'
echo "STEP 4 done."
