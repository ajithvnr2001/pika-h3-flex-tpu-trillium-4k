#!/usr/bin/env bash
# Every INTERVAL seconds: record TPU state and the last lines of the remote run log.
# Usage: nohup scripts/monitor.sh > logs/monitor.log 2>&1 &
TPU="${TPU:-pika-v4-tpu}"; ZONE="${ZONE:-europe-west4-a}"; PROJECT="${PROJECT:-genai25th01}"
INTERVAL="${INTERVAL:-300}"
while true; do
  echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  state=$(timeout 60 gcloud compute tpus tpu-vm describe "$TPU" --zone="$ZONE" --project="$PROJECT" \
          --format='value(state)' 2>&1 | tail -1)
  echo "TPU $TPU: ${state:-unknown}"
  if [ "$state" = "READY" ]; then
    timeout 90 gcloud compute tpus tpu-vm ssh "$TPU" --zone="$ZONE" --project="$PROJECT" --command \
      'for f in ~/bootstrap.log ~/run.log; do [ -f $f ] && { echo "--- $f"; tr "\r" "\n" < $f | grep -v "^\s*$" | tail -4; }; done; free -g | sed -n 2p; df -h /dev/shm | tail -1' \
      2>/dev/null
  fi
  sleep "$INTERVAL"
done
