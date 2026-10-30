#!/usr/bin/env bash
# OPTIONAL (runs locally, no TPU needed): join a clip with its extension into one ~10 s video, and make a 4K copy.
#
#   commands/extra_join_and_4k.sh <first.mp4> <extension.mp4> <out_name>
#
# Join: the extension was generated from the LAST frame of the first clip, so its frame 0 duplicates that
# frame. It is dropped (video and the matching 1/24 s of audio) so the cut is seamless.
#
# 4K: Pika's open weights generate natively at 480p or 768p only. Pika's own 1080p/2K/4K "draft-to-video"
# upsampler is a paid hosted API (https://dev.pika.art/models/pika/pika-h3-flex/draft-to-video), not part of
# the open release. This script makes a 3840x2160 copy with ffmpeg Lanczos resizing + light sharpening. That is
# a resize: it adds pixels, not new detail.
set -euo pipefail
A="${1:?first clip}"; B="${2:?extension clip}"; NAME="${3:?output name}"
DIR="$(dirname "$A")"
OUT="$DIR/$NAME.mp4"; OUT4K="$DIR/${NAME}_4k.mp4"

ffmpeg -v error -y -i "$A" -i "$B" -filter_complex \
  "[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[v1];[1:a]atrim=start=0.041667,asetpts=PTS-STARTPTS[a1];\
   [0:v][0:a][v1][a1]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -c:a aac -b:a 192k "$OUT"

ffmpeg -v error -y -i "$OUT" -vf "scale=3840:2160:flags=lanczos,unsharp=5:5:0.6" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a copy "$OUT4K"

for f in "$OUT" "$OUT4K"; do
  ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames,sample_rate,channels \
    -show_entries format=duration -of compact "$f"
done
