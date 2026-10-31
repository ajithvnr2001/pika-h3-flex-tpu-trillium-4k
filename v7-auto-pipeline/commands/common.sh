# Shared settings for all v7 commands. Override any of these with environment variables (e.g. TPU=... ZONE=...).
export PROJECT="${PROJECT:-genai25th01}"
export ZONE="${ZONE:-europe-west4-a}"
export TPU="${TPU:-pika-v7-tpu}"
export ACCEL="${ACCEL:-v6e-4}"
export RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
# SPOT=1 (default): cheaper, but Google can reclaim it any time (about 5 times during this project; 6_auto_pipeline.sh recovers automatically).
# SPOT=0: on-demand ("general") TPU, more expensive per hour, not preempted.
export SPOT="${SPOT:-1}"
export LOCAL_OUT="${LOCAL_OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
PIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPE_DIR
G=(--zone="$ZONE" --project="$PROJECT")
tpu_ssh() { gcloud compute tpus tpu-vm ssh "$TPU" "${G[@]}" --command "$1"; }
tpu_scp() { gcloud compute tpus tpu-vm scp "$@" "${G[@]}"; }
