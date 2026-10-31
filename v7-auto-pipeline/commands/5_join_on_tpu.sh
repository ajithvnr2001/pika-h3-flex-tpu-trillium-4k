#!/usr/bin/env bash
# STEP 2b (optional): join the clips of a jobs file, in file order, into one long video ON THE TPU VM.
#
#   commands/5_join_on_tpu.sh <jobs.txt> <final_name>
#
# A clip whose first frame is @<previous clip> starts with a copy of that clip's last frame; that duplicate frame
# (and 1/24 s of audio) is dropped so the cut is seamless. Run before step 3; step 3 copies <final_name>.mp4 back.
# LOUDNORM=1 (default) levels the joined soundtrack to -16 LUFS (Pika's raw audio is quiet, about -37 dB mean);
# LOUDNORM=0 keeps the original levels.
set -euo pipefail
. "$(dirname "$0")/common.sh"
JOBS="${1:?jobs file}"; FINAL="${2:?final name}"
AF="[a]anull"; [ "${LOUDNORM:-1}" = 1 ] && AF="[a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=32000,aformat=sample_fmts=fltp:channel_layouts=stereo"

list=""; prev=""
while read -r prompt name rest; do
  [ -z "${prompt:-}" ] && continue
  case "$prompt" in \#*) continue;; esac
  trim=0
  for tok in ${rest:-}; do [ "$tok" = "@$prev" ] && [ -n "$prev" ] && trim=1; done
  list+=" $name:$trim"; prev="$name"
done < "$JOBS"

tpu_ssh "cd /dev/shm/pika/outputs && set -e; inputs=(); filter=''; n=0; \
  for item in $list; do c=\${item%%:*}; t=\${item##*:}; inputs+=(-i \$c.mp4); \
    if [ \$t = 1 ]; then filter+=\"[\$n:v]trim=start_frame=1,setpts=PTS-STARTPTS[v\$n];[\$n:a]atrim=start=0.041667,asetpts=PTS-STARTPTS[a\$n];\"; \
    else filter+=\"[\$n:v]setpts=PTS-STARTPTS[v\$n];[\$n:a]asetpts=PTS-STARTPTS[a\$n];\"; fi; n=\$((n+1)); done; \
  for i in \$(seq 0 \$((n-1))); do filter+=\"[v\$i][a\$i]\"; done; filter+=\"concat=n=\$n:v=1:a=1[v][a];$AF[ao]\"; \
  ffmpeg -v error -y \"\${inputs[@]}\" -filter_complex \"\$filter\" -map '[v]' -map '[ao]' \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -threads 0 -c:a aac -b:a 192k $FINAL.mp4; \
  ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames -show_entries format=duration -of compact $FINAL.mp4"
echo "Joined on TPU: /dev/shm/pika/outputs/$FINAL.mp4 (run step 3 to copy it)"
