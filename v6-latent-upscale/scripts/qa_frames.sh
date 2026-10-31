#!/usr/bin/env bash
# Quality check, run ON THE TPU VM:  ~/qa_frames.sh <name> [<name> ...]
# For each clip in /dev/shm/pika/outputs: a 5-frame sheet over time, a 1:1 centre crop at t=2.5 s, and the mean
# frame-to-frame SSIM (temporal stability; flicker lowers it). Writes /dev/shm/pika/outputs/qa_<name>_*.png.
set -euo pipefail
cd /dev/shm/pika/outputs
for n in "$@"; do
  ffmpeg -v error -y -i "$n.mp4" -vf "select='not(mod(n,25))',scale=416:-1,tile=5x1" -frames:v 1 -update 1 "qa_${n}_sheet.png"
  ffmpeg -v error -y -ss 2.5 -i "$n.mp4" -frames:v 1 -vf "crop=iw*0.4:ih*0.45:iw*0.3:ih*0.3" "qa_${n}_crop.png"
  s=$(ffmpeg -i "$n.mp4" -i "$n.mp4" -filter_complex \
      "[0:v]trim=start_frame=1,setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]ssim" -f null - 2>&1 \
      | grep -o "All:[0-9.]*" | cut -d: -f2)
  echo "QA $n frame-to-frame SSIM $s"
done
