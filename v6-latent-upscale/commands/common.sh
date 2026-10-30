# Shared settings for the 4 step scripts. Override any of these with environment variables.
export PROJECT="${PROJECT:-genai25th01}"
export ZONE="${ZONE:-europe-west4-a}"
export TPU="${TPU:-pika-v6-tpu}"
export ACCEL="${ACCEL:-v6e-4}"
export RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
# SPOT=1 (default): cheaper, but Google can reclaim it any time (happened 3 times in one day here).
# SPOT=0: on-demand ("general") TPU, more expensive per hour, not preempted.
export SPOT="${SPOT:-1}"
export LOCAL_OUT="${LOCAL_OUT:-/data/pika/v6-latent-upscale/outputs}"
V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export V4_DIR
G=(--zone="$ZONE" --project="$PROJECT")
tpu_ssh() { gcloud compute tpus tpu-vm ssh "$TPU" "${G[@]}" --command "$1"; }
tpu_scp() { gcloud compute tpus tpu-vm scp "$@" "${G[@]}"; }
