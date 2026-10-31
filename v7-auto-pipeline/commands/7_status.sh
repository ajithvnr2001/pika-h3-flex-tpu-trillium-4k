#!/usr/bin/env bash
# STATUS: how far is a video, and what is it costing?   commands/7_status.sh <jobs.txt>
# Shows: TPU state, clips finished (saved locally), the clip being made now, estimated time left, and the TPU cost so
# far (from the TPU's creation time). Safe to run any time, also while 6_auto_pipeline.sh is running.
set -uo pipefail
. "$(dirname "$0")/common.sh"
JOBS="${1:?jobs file}"
names=$(grep -vE '^\s*(#|$)' "$JOBS" | awk '{print $2}')
total=$(echo "$names" | grep -c .)
done_n=0; for n in $names; do [ -s "$LOCAL_OUT/$n.json" ] && done_n=$((done_n + 1)); done

read -r st created < <(gcloud compute tpus tpu-vm describe "$TPU" "${G[@]}" --format='value(state,createTime)' 2>/dev/null || echo "GONE -")
echo "TPU $TPU ($ZONE): ${st:-GONE}"
if [ "${st:-}" = READY ]; then
  now=$(tpu_ssh "grep -E '^\[.*Job [0-9]+/' ~/run_batch.log 2>/dev/null | tail -1; test -f ~/batch.rc && echo \"run finished, exit code \$(cat ~/batch.rc)\"" 2>/dev/null)
  [ -n "$now" ] && echo "On the TPU: $now"
fi
echo "Clips saved locally: $done_n / $total  (in $LOCAL_OUT)"
left=$((total - done_n))
[ "$left" -gt 0 ] && echo "Time left: about $(( left * 44 / 10 )) min (~4.4 min per 5 s clip; +8 min for a first clip)"
if [ -n "${created:-}" ] && [ "$created" != "-" ]; then
  hours=$(python3 -c "import datetime as d,sys;import re;c=d.datetime.fromisoformat(re.sub(r'(\.\d{6})\d*','\\1',sys.argv[1]).replace('Z','+00:00'));print(f'{(d.datetime.now(d.timezone.utc)-c).total_seconds()/3600:.2f}')" "$created")
  rate=$([ "$SPOT" = 1 ] && echo 7.13 || echo 11.88)
  echo "TPU running for ${hours} h -> about \$$(python3 -c "print(f'{$hours*$rate:.2f}')") so far (\$$rate/h, SPOT=$SPOT)"
fi
