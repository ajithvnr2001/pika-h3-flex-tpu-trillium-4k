#!/usr/bin/env python3
"""
Standalone 4K UHD (3840x2160) Upscaler & Micro-Detail Enhancer for Pika H3 Flex
Joint Video + Synchronized Stereo Audio MP4 outputs.
Preserves the 48kHz Stereo AAC audio stream (-c:a copy) while lifting video to 3840x2160.
"""

import json
import subprocess
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def upscale_audio_mp4_to_4k(src_mp4: Path, dst_mp4: Path) -> dict:
    print(f"Upscaling {src_mp4.name} -> {dst_mp4.name} (3840x2160 4K UHD + 48kHz Stereo Audio)...", flush=True)
    t0 = time.perf_counter()
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src_mp4),
        "-vf", "scale=3840:2160:flags=lanczos,unsharp=3:3:0.35:3:3:0.0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst_mp4),
    ], check=True)
    t_dur = time.perf_counter() - t0
    size_mb = dst_mp4.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: {dst_mp4.name} ({size_mb:.2f} MB) in {t_dur:.2f}s", flush=True)
    return {"output_4k": str(dst_mp4), "size_mb": round(size_mb, 2), "upscale_seconds": round(t_dur, 3)}


if __name__ == "__main__":
    for sec in (5, 10):
        src = OUTPUT_DIR / f"pika_{sec}s_audio.mp4"
        dst = OUTPUT_DIR / f"pika_{sec}s_audio_4k.mp4"
        if src.exists():
            upscale_audio_mp4_to_4k(src, dst)
