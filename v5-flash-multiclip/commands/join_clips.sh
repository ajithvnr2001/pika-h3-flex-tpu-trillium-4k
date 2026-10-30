#!/usr/bin/env bash
# Join any number of clips (with their audio) into one long video. Runs locally, no TPU needed.
#
#   commands/join_clips.sh <output.mp4> <clip1.mp4> <clip2.mp4> [clip3.mp4 ...]
#
# Set CHAINED=1 when every clip after the first was generated from the LAST frame of the clip before it
# (an extension chain). Then the first frame of clips 2..N duplicates the previous clip's last frame, so it is
# dropped (with the matching 1/24 s of audio) to make the cut seamless. Leave CHAINED=0 for separate shots.
set -euo pipefail
OUT="${1:?output.mp4}"; shift
[ "$#" -ge 2 ] || { echo "need at least 2 clips"; exit 1; }
CHAINED="${CHAINED:-0}"

inputs=(); filter=""; n=0
for clip in "$@"; do
  inputs+=(-i "$clip")
  if [ "$CHAINED" = "1" ] && [ "$n" -gt 0 ]; then
    filter+="[$n:v]trim=start_frame=1,setpts=PTS-STARTPTS[v$n];[$n:a]atrim=start=0.041667,asetpts=PTS-STARTPTS[a$n];"
  else
    filter+="[$n:v]setpts=PTS-STARTPTS[v$n];[$n:a]asetpts=PTS-STARTPTS[a$n];"
  fi
  n=$((n + 1))
done
for i in $(seq 0 $((n - 1))); do filter+="[v$i][a$i]"; done
filter+="concat=n=$n:v=1:a=1[v][a]"

ffmpeg -v error -y "${inputs[@]}" -filter_complex "$filter" -map "[v]" -map "[a]" \
  -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -c:a aac -b:a 192k "$OUT"
ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames,sample_rate,channels \
  -show_entries format=duration -of compact "$OUT"
