#!/usr/bin/env bash
# STEP 2 (one clip): commands/2_generate_video.sh "<prompt text or prompt.txt>" <name> [duration] [first_frame|@name]
# Upscaled to 1664x960 by default (GEN_ARGS="" for plain 480p). For several clips use 2_generate_batch.sh:
# it loads and compiles the model once.
set -euo pipefail
here="$(dirname "$0")"
PROMPT="${1:?prompt text or prompt file}"; NAME="${2:?name}"; DUR="${3:-5}"; IMG="${4:--}"
tmpd=$(mktemp -d)
if [ -f "$PROMPT" ]; then cp "$PROMPT" "$tmpd/prompt.txt"; else printf '%s\n' "$PROMPT" > "$tmpd/prompt.txt"; fi
echo "$tmpd/prompt.txt $NAME $DUR $IMG" > "$tmpd/jobs.txt"
bash "$here/2_generate_batch.sh" "$tmpd/jobs.txt"
rm -rf "$tmpd"
