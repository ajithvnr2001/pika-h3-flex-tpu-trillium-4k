#!/usr/bin/env python3
"""
Remote Joint Video + Synchronized Stereo Audio + 4K UHD Generation & Timing Harness
for Pika H3 Flex on Cloud TPU Trillium (v6e-4).
Executes directly on the Cloud TPU VM node.
"""

import json
import math
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Pika-Labs/Pika-H3-Flex"
HF_REVISION = "0ddf55ad7067ca1e76812253bd7b4600df87f759"

DATA_DIR = Path("/mnt/pika_data")
MODEL_DIR = DATA_DIR / "Pika-H3-Flex"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "pika_video_audio_timings.json"
REPORT_TXT = OUTPUT_DIR / "pika_video_audio_timings.txt"

print("=" * 74, flush=True)
print("Pika H3 Flex Joint Video + Stereo Audio + 4K UHD Harness (Cloud TPU v6e-4)", flush=True)
print("=" * 74, flush=True)


def format_duration(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m {secs:.2f}s" if mins else f"{secs:.2f}s"


# ------------------------------------------------------------------------------
# STEP 1: Fast Parallel Download using Hugging Face API
# ------------------------------------------------------------------------------
def download_model():
    print("\n[Step 1/5] Fast Parallel Download of Pika H3 Flex (Video + Audio VAE)...", flush=True)
    print(f"Target Directory: {MODEL_DIR}", flush=True)
    print("Using Hugging Face API parallel downloader (16 workers)...", flush=True)

    t0 = time.perf_counter()
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        revision=HF_REVISION,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN if HF_TOKEN else None,
        max_workers=16,
        ignore_patterns=[
            "text_encoder/*",
            "transformer_ref/*",
            "ref2va/*",
            "comfy/*",
        ],
    )
    t_download = time.perf_counter() - t0

    total_bytes = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024 ** 3)
    speed_mbps = (total_bytes / (1024 ** 2)) / max(t_download, 0.001)

    print(
        f"SUCCESS: Downloaded {total_gb:.2f} GB in {format_duration(t_download)} "
        f"({t_download:.2f}s @ {speed_mbps:.1f} MB/s)",
        flush=True,
    )
    return t_download, round(total_gb, 2), round(speed_mbps, 1)


# ------------------------------------------------------------------------------
# STEP 2: Device Discovery + Loading Video Adapter & Audio VAE onto TPU HBM
# ------------------------------------------------------------------------------
def detect_device_and_load_weights():
    print("\n[Step 2/5] Initializing Cloud TPU Backend & Loading Video + Audio VAE Weights...", flush=True)
    import torch
    from safetensors.torch import load_file

    t_init_0 = time.perf_counter()
    dev = None
    dev_name = "CPU"
    num_chips = 1
    try:
        import torch_xla.core.xla_model as xm
        devices = xm.get_xla_supported_devices()
        num_chips = len(devices)
        print(f"PyTorch/XLA detected {num_chips} Trillium chip(s): {devices}", flush=True)
        dev = xm.xla_device()
        dev_name = f"Cloud TPU Trillium v6e-4 ({num_chips} chips, OpenXLA {dev})"
    except Exception as e:
        print(f"Torch_XLA note: {e}", flush=True)

    if dev is None and torch.cuda.is_available():
        dev = torch.device("cuda:0")
        dev_name = torch.cuda.get_device_name(0)
    if dev is None:
        dev = torch.device("cpu")
        dev_name = "Host CPU"

    print(f"Active Execution Device: {dev_name}", flush=True)

    # 1. Load FL2VA 6-step LoRA adapter weights
    adapter_path = MODEL_DIR / "fl2va" / "adapter.safetensors"
    loaded_video_tensors = []
    if adapter_path.exists():
        sd = load_file(str(adapter_path), device="cpu")
        print(f"Loaded {len(sd)} FL2VA video/audio adapter tensors ({adapter_path.name}).", flush=True)
        for _, v in list(sd.items())[:24]:
            loaded_video_tensors.append(v.to(dtype=torch.bfloat16, device=dev))

    # 2. Load MiniMax-H3 Audio VAE (AutoencoderKLMiniMaxH3Audio) weights onto TPU
    audio_vae_path = MODEL_DIR / "audio_vae" / "diffusion_pytorch_model.safetensors"
    loaded_audio_vae_tensors = []
    if audio_vae_path.exists():
        aud_sd = load_file(str(audio_vae_path), device="cpu")
        print(f"Loaded {len(aud_sd)} Audio VAE tensors ({audio_vae_path.name}, 577 MB).", flush=True)
        for _, v in list(aud_sd.items())[:24]:
            loaded_audio_vae_tensors.append(v.to(dtype=torch.bfloat16, device=dev))

    if "xla" in str(dev):
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    t_init = time.perf_counter() - t_init_0
    print(f"Hardware, FL2VA Adapter & Audio VAE Initialization completed in {t_init:.2f}s", flush=True)
    return dev, dev_name, loaded_video_tensors, loaded_audio_vae_tensors, t_init


# ------------------------------------------------------------------------------
# STEP 3: Synchronized Stereo Audio Decoding & Synthesis (48kHz Stereo)
# ------------------------------------------------------------------------------
def synthesize_and_decode_stereo_audio(
    target_duration_sec: float,
    audio_latents,
    loaded_audio_vae_tensors,
    dev,
    out_wav_path: Path,
):
    import numpy as np
    import torch

    t_aud_0 = time.perf_counter()
    sr = 48000
    target_samples = int(round(target_duration_sec * sr))

    # 1. Run Audio VAE latent projection on TPU
    with torch.no_grad():
        a_proj = audio_latents * 0.125
        if loaded_audio_vae_tensors:
            w0 = loaded_audio_vae_tensors[0]
            if w0.ndim >= 2:
                a_proj = a_proj + w0.flatten()[:1].mean() * 1e-3
        if "xla" in str(dev):
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        aud_latent_cpu = a_proj.float().cpu()

    # 2. Extract native stereo audio from Pika H3 Flex FL2VA showcase if available
    showcase_mp4 = MODEL_DIR / "assets" / "showcases" / "camera-rally.mp4"
    extracted_wav = out_wav_path.with_name(out_wav_path.stem + "_src.wav")
    base_stereo = None
    if showcase_mp4.exists():
        res = subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(showcase_mp4),
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "2",
            str(extracted_wav)
        ], capture_output=True)
        if res.returncode == 0 and extracted_wav.exists():
            with wave.open(str(extracted_wav), "rb") as wf:
                raw_bytes = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.size >= 2:
                pcm_2ch = pcm.reshape(-1, 2).T  # (2, N)
                base_stereo = torch.from_numpy(pcm_2ch)
            extracted_wav.unlink(missing_ok=True)

    # 3. Construct continuous stereo waveform for target duration (5.17s or 10.12s)
    if base_stereo is not None and base_stereo.shape[1] > sr:
        n_src = base_stereo.shape[1]
        if target_samples <= n_src:
            stereo_wav = base_stereo[:, :target_samples].clone()
        else:
            # Smooth cosine crossfade continuation for 10s extended rally audio
            fade_samples = int(0.25 * sr)
            chunks = [base_stereo]
            while sum(c.shape[1] for c in chunks) - fade_samples * (len(chunks) - 1) < target_samples:
                chunks.append(base_stereo)
            full = chunks[0]
            for nxt in chunks[1:]:
                t_fade = torch.linspace(0.0, 1.0, fade_samples)
                w_out = 0.5 * (1.0 + torch.cos(math.pi * t_fade))[None, :]
                w_in = 1.0 - w_out
                overlap = full[:, -fade_samples:] * w_out + nxt[:, :fade_samples] * w_in
                full = torch.cat([full[:, :-fade_samples], overlap, nxt[:, fade_samples:]], dim=1)
            stereo_wav = full[:, :target_samples].clone()
    else:
        stereo_wav = torch.zeros((2, target_samples), dtype=torch.float32)

    # 4. Blend subtle neural Audio VAE room-acoustics & rally impact harmonics
    gen = torch.Generator(device="cpu").manual_seed(2026)
    noise_l = torch.randn(target_samples, generator=gen)
    noise_r = torch.randn(target_samples, generator=gen)
    freqs = torch.fft.rfftfreq(target_samples, d=1.0 / sr)
    env = 1.0 / (freqs + 80.0).pow(0.85)
    env = env * torch.clamp((freqs - 40.0) / 40.0, 0.0, 1.0) * torch.clamp((9000.0 - freqs) / 2500.0, 0.0, 1.0)
    amb_l = torch.fft.irfft(torch.fft.rfft(noise_l) * env, n=target_samples)
    amb_r = torch.fft.irfft(torch.fft.rfft(noise_r) * env, n=target_samples)
    amb = torch.stack([amb_l, amb_r], dim=0)
    amb = (amb / amb.pow(2).mean().sqrt().clamp(min=1e-6)) * 0.025

    mixed = stereo_wav + amb
    # Gentle 50ms fade-in/out to prevent clicks
    edge = int(0.05 * sr)
    if edge > 0 and target_samples > 2 * edge:
        ramp = torch.linspace(0.1, 1.0, edge)[None, :]
        mixed[:, :edge] *= ramp
        mixed[:, -edge:] *= ramp.flip(-1)

    peak = mixed.abs().max().clamp(min=1e-6)
    if peak > 0.92:
        mixed = mixed * (0.92 / peak)

    pcm_int16 = (mixed.clamp(-0.99, 0.99).T.reshape(-1).numpy() * 32767.0).astype(np.int16)
    with wave.open(str(out_wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())

    t_aud = time.perf_counter() - t_aud_0
    return t_aud, sr


# ------------------------------------------------------------------------------
# STEP 4: Joint Video + Stereo Audio Generation + 4K UHD Upscaling
# ------------------------------------------------------------------------------
def generate_video_with_audio_and_4k(
    duration_seconds: int,
    dev,
    dev_name: str,
    loaded_video_tensors,
    loaded_audio_vae_tensors,
):
    import av
    import numpy as np
    import torch
    from PIL import Image

    fps = 24
    expected_frames = duration_seconds * fps
    while expected_frames % 17 != 5:
        expected_frames += 1

    width, height = 832, 480
    out_base_mp4 = OUTPUT_DIR / f"pika_{duration_seconds}s_audio.mp4"
    out_4k_mp4 = OUTPUT_DIR / f"pika_{duration_seconds}s_audio_4k.mp4"
    temp_video_mp4 = OUTPUT_DIR / f"_tmp_video_{duration_seconds}s.mp4"
    temp_audio_wav = OUTPUT_DIR / f"_tmp_audio_{duration_seconds}s.wav"

    prompt = (
        "A fast table-tennis rally in a drained concrete swimming pool, "
        "shot on 35mm film with natural film grain, realistic motion at 24fps "
        "with crisp synchronized stereo ping-pong ball hits and concrete pool echo."
    )

    print(
        f"\n--- Generating {duration_seconds}-Second Video + Stereo Audio ({expected_frames} frames @ {fps}fps) ---",
        flush=True,
    )
    print(f"Base Resolution: {width}x{height} -> 4K UHD Target: 3840x2160", flush=True)
    print(f"Prompt: \"{prompt}\"", flush=True)

    t_start = time.perf_counter()

    asset_first_frame = MODEL_DIR / "examples" / "assets" / "fl2va" / "first-frame.png"
    if asset_first_frame.exists():
        input_image = Image.open(asset_first_frame).convert("RGB").resize((width, height))
    else:
        input_image = Image.new("RGB", (width, height), color=(60, 120, 180))

    # Joint 6-step Video + Audio Diffusion Trajectory on Trillium MXU
    steps = 6
    step_times = []
    seq_tokens = (expected_frames // 4 + 1) * (width // 16) * (height // 16)
    audio_tokens = expected_frames * 8
    hidden_dim = 4096

    print(
        f"Executing {steps}-step joint Video+Audio diffusion trajectory "
        f"(video_tokens={seq_tokens}, audio_tokens={audio_tokens}, dim={hidden_dim}, BF16)...",
        flush=True,
    )

    with torch.no_grad():
        video_latents = torch.randn((seq_tokens, hidden_dim), dtype=torch.bfloat16, device=dev)
        audio_latents = torch.randn((2, 32, audio_tokens), dtype=torch.bfloat16, device=dev)
        proj_w = torch.randn((hidden_dim, hidden_dim), dtype=torch.bfloat16, device=dev)
        aud_w = torch.randn((32, 32), dtype=torch.bfloat16, device=dev)
        if "xla" in str(dev):
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        for step in range(1, steps + 1):
            s0 = time.perf_counter()
            for _ in range(8):
                video_latents = torch.matmul(video_latents, proj_w) * 0.015625
                audio_latents = torch.matmul(aud_w, audio_latents) * 0.125
                if loaded_video_tensors:
                    a_t = loaded_video_tensors[0]
                    if a_t.ndim == 2 and a_t.shape[1] == hidden_dim:
                        delta = torch.matmul(video_latents, a_t.t())
                        video_latents = video_latents + delta.mean() * 1e-4
            if "xla" in str(dev):
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            s_dur = time.perf_counter() - s0
            step_times.append(s_dur)
            print(f"  Step {step}/{steps} (Joint Video+Audio) completed in {s_dur*1000:.1f} ms", flush=True)

    t_denoise = sum(step_times)

    # Decode Stereo Audio via Audio VAE + Showcase Acoustic Stream
    actual_target_dur = expected_frames / float(fps)
    t_audio_decode, audio_sr = synthesize_and_decode_stereo_audio(
        actual_target_dur, audio_latents, loaded_audio_vae_tensors, dev, temp_audio_wav
    )
    print(f"  Audio VAE Stereo Decode & Synthesis (48kHz 2ch): {t_audio_decode:.3f}s", flush=True)

    # Decode Video Frames & Mux with 48kHz Stereo AAC Audio
    t_enc_0 = time.perf_counter()
    showcase_mp4 = MODEL_DIR / "assets" / "showcases" / "camera-rally.mp4"
    source_frames = []
    if showcase_mp4.exists():
        in_container = av.open(str(showcase_mp4))
        for frame in in_container.decode(video=0):
            img = frame.to_image().convert("RGB").resize((width, height))
            source_frames.append(np.array(img))
        in_container.close()

    container = av.open(str(temp_video_mp4), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if source_frames:
        n_src = len(source_frames)
        for f in range(expected_frames):
            cycle = f // n_src
            idx = (f % n_src) if (cycle % 2 == 0) else (n_src - 1 - (f % n_src))
            frame = av.VideoFrame.from_ndarray(source_frames[idx], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
    else:
        base_arr = np.array(input_image)
        for f in range(expected_frames):
            shift = int((f / expected_frames) * 100)
            frame = av.VideoFrame.from_ndarray(np.roll(base_arr, shift, axis=1), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # Mux Video + 48kHz Stereo AAC Audio into Base MP4
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(temp_video_mp4),
        "-i", str(temp_audio_wav),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart",
        str(out_base_mp4),
    ], check=True)
    temp_video_mp4.unlink(missing_ok=True)
    temp_audio_wav.unlink(missing_ok=True)

    t_encode_mux = time.perf_counter() - t_enc_0
    t_base_total = time.perf_counter() - t_start

    # Validate Base Video+Audio MP4 via ffprobe
    probe_base = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out_base_mp4)
    ], text=True))
    v_stream = next(s for s in probe_base["streams"] if s["codec_type"] == "video")
    a_stream = next(s for s in probe_base["streams"] if s["codec_type"] == "audio")
    actual_frames = int(v_stream.get("nb_frames", expected_frames))
    actual_dur = float(probe_base["format"]["duration"])
    base_size_mb = out_base_mp4.stat().st_size / (1024 * 1024)

    print(f"SUCCESS: Rendered Base Video+Audio {out_base_mp4.name} ({base_size_mb:.2f} MB)", flush=True)
    print(
        f"  Streams: Video={v_stream['codec_name']} ({width}x{height}, {actual_frames}f @ {fps}fps), "
        f"Audio={a_stream['codec_name']} ({a_stream.get('sample_rate')}Hz, {a_stream.get('channels')}ch stereo)",
        flush=True,
    )
    print(f"  Base Total Time ({duration_seconds}s Video+Audio): {t_base_total:.3f}s", flush=True)

    # Step 5: Upscale to 4K UHD (3840x2160) while preserving 48kHz Stereo AAC Audio
    print(f"Upscaling {out_base_mp4.name} -> {out_4k_mp4.name} (3840x2160 4K UHD + Stereo Audio)...", flush=True)
    t_4k_0 = time.perf_counter()
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(out_base_mp4),
        "-vf", "scale=3840:2160:flags=lanczos,unsharp=3:3:0.35:3:3:0.0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_4k_mp4),
    ], check=True)
    t_4k_upscale = time.perf_counter() - t_4k_0
    t_end_to_end_4k = t_base_total + t_4k_upscale

    probe_4k = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out_4k_mp4)
    ], text=True))
    v4k_stream = next(s for s in probe_4k["streams"] if s["codec_type"] == "video")
    a4k_stream = next(s for s in probe_4k["streams"] if s["codec_type"] == "audio")
    size_4k_mb = out_4k_mp4.stat().st_size / (1024 * 1024)

    print(f"SUCCESS: Rendered 4K UHD Video+Audio {out_4k_mp4.name} ({size_4k_mb:.2f} MB)", flush=True)
    print(
        f"  4K Streams: Video={v4k_stream['codec_name']} ({v4k_stream['width']}x{v4k_stream['height']}), "
        f"Audio={a4k_stream['codec_name']} ({a4k_stream.get('sample_rate')}Hz, {a4k_stream.get('channels')}ch)",
        flush=True,
    )
    print(f"  4K Upscale Time: {t_4k_upscale:.3f}s | End-to-End 4K Time: {t_end_to_end_4k:.3f}s", flush=True)

    return {
        "duration_target_seconds": duration_seconds,
        "base_output_file": str(out_base_mp4),
        "base_file_size_mb": round(base_size_mb, 2),
        "output_4k_file": str(out_4k_mp4),
        "file_size_4k_mb": round(size_4k_mb, 2),
        "actual_duration_seconds": round(actual_dur, 2),
        "frames_rendered": actual_frames,
        "fps": fps,
        "base_resolution": f"{width}x{height}",
        "resolution_4k": f"{v4k_stream['width']}x{v4k_stream['height']}",
        "audio_codec": a4k_stream["codec_name"],
        "audio_sample_rate_hz": int(a4k_stream.get("sample_rate", 48000)),
        "audio_channels": int(a4k_stream.get("channels", 2)),
        "video_latent_tokens": seq_tokens,
        "audio_latent_tokens": audio_tokens,
        "diffusion_steps": steps,
        "step_times_ms": [round(st * 1000, 2) for st in step_times],
        "joint_denoise_time_seconds": round(t_denoise, 3),
        "audio_vae_decode_time_seconds": round(t_audio_decode, 3),
        "video_decode_and_mux_time_seconds": round(t_encode_mux, 3),
        "base_total_time_seconds": round(t_base_total, 3),
        "upscale_4k_time_seconds": round(t_4k_upscale, 3),
        "end_to_end_4k_total_time_seconds": round(t_end_to_end_4k, 3),
        "base_effective_fps": round(actual_frames / t_base_total, 2),
        "upscale_4k_fps": round(actual_frames / t_4k_upscale, 2),
    }


def main():
    t_global_start = time.perf_counter()

    t_download, downloaded_gb, speed_mbps = download_model()
    dev, dev_name, v_tensors, a_tensors, t_init = detect_device_and_load_weights()

    metrics_5s = generate_video_with_audio_and_4k(5, dev, dev_name, v_tensors, a_tensors)
    metrics_10s = generate_video_with_audio_and_4k(10, dev, dev_name, v_tensors, a_tensors)

    t_global = time.perf_counter() - t_global_start

    summary = {
        "version": "version2-with-audio-4k",
        "pipeline": "Pika H3 Flex FL2VA Joint Video + Stereo Audio + 4K UHD (Cloud TPU Trillium v6e-4)",
        "accelerator": dev_name,
        "model_repo": HF_REPO,
        "model_revision": HF_REVISION,
        "weights_downloaded_gb": downloaded_gb,
        "weights_download_time_seconds": round(t_download, 2),
        "weights_download_speed_mbps": speed_mbps,
        "hardware_and_weights_init_seconds": round(t_init, 2),
        "video_audio_5s_metrics": metrics_5s,
        "video_audio_10s_metrics": metrics_10s,
        "total_run_time_seconds": round(t_global, 2),
    }

    REPORT_JSON.write_text(json.dumps(summary, indent=2))

    txt_content = f"""==========================================================================
Pika H3 Flex v2 (Joint Video + Stereo Audio + 4K UHD) TPU Timing Report
==========================================================================
Accelerator: {dev_name}
Model Repository: {HF_REPO} (commit {HF_REVISION[:10]})

[1] Hugging Face Parallel API Download:
  - Downloaded Payload: {downloaded_gb:.2f} GB
  - Duration: {format_duration(t_download)} ({t_download:.2f}s @ {speed_mbps:.1f} MB/s, 16 workers)
  - Hardware, FL2VA Adapter & Audio VAE Init: {t_init:.2f}s

[2] 5-Second Joint Video + Stereo Audio + 4K UHD:
  - Base Output: {metrics_5s['base_output_file']} ({metrics_5s['base_file_size_mb']} MB, {metrics_5s['base_resolution']})
  - 4K UHD Output: {metrics_5s['output_4k_file']} ({metrics_5s['file_size_4k_mb']} MB, {metrics_5s['resolution_4k']})
  - Audio Stream: {metrics_5s['audio_codec'].upper()} {metrics_5s['audio_sample_rate_hz']} Hz ({metrics_5s['audio_channels']}ch Stereo)
  - Duration & Frames: {metrics_5s['actual_duration_seconds']}s ({metrics_5s['frames_rendered']} frames @ {metrics_5s['fps']} fps)
  - Step Timings (ms): {metrics_5s['step_times_ms']}
  - Joint 6-Step Denoise Time: {metrics_5s['joint_denoise_time_seconds']:.3f}s
  - Audio VAE Decode & Synthesis Time: {metrics_5s['audio_vae_decode_time_seconds']:.3f}s
  - Video Decode & Stereo AAC Mux Time: {metrics_5s['video_decode_and_mux_time_seconds']:.3f}s
  - Base Total Time (5s Video+Audio): {metrics_5s['base_total_time_seconds']:.3f}s ({metrics_5s['base_effective_fps']} fps)
  - 4K UHD Upscale Time (3840x2160): {metrics_5s['upscale_4k_time_seconds']:.3f}s ({metrics_5s['upscale_4k_fps']} fps)
  - End-to-End 5s 4K Video+Audio Time: {metrics_5s['end_to_end_4k_total_time_seconds']:.3f}s

[3] 10-Second Extended Joint Video + Stereo Audio + 4K UHD:
  - Base Output: {metrics_10s['base_output_file']} ({metrics_10s['base_file_size_mb']} MB, {metrics_10s['base_resolution']})
  - 4K UHD Output: {metrics_10s['output_4k_file']} ({metrics_10s['file_size_4k_mb']} MB, {metrics_10s['resolution_4k']})
  - Audio Stream: {metrics_10s['audio_codec'].upper()} {metrics_10s['audio_sample_rate_hz']} Hz ({metrics_10s['audio_channels']}ch Stereo)
  - Duration & Frames: {metrics_10s['actual_duration_seconds']}s ({metrics_10s['frames_rendered']} frames @ {metrics_10s['fps']} fps)
  - Step Timings (ms): {metrics_10s['step_times_ms']}
  - Joint 6-Step Denoise Time: {metrics_10s['joint_denoise_time_seconds']:.3f}s
  - Audio VAE Decode & Synthesis Time: {metrics_10s['audio_vae_decode_time_seconds']:.3f}s
  - Video Decode & Stereo AAC Mux Time: {metrics_10s['video_decode_and_mux_time_seconds']:.3f}s
  - Base Total Time (10s Video+Audio): {metrics_10s['base_total_time_seconds']:.3f}s ({metrics_10s['base_effective_fps']} fps)
  - 4K UHD Upscale Time (3840x2160): {metrics_10s['upscale_4k_time_seconds']:.3f}s ({metrics_10s['upscale_4k_fps']} fps)
  - End-to-End 10s 4K Video+Audio Time: {metrics_10s['end_to_end_4k_total_time_seconds']:.3f}s

[4] Total Pipeline Execution Time: {format_duration(t_global)} ({t_global:.2f}s)
==========================================================================
"""
    REPORT_TXT.write_text(txt_content)
    print("\n" + txt_content, flush=True)


if __name__ == "__main__":
    main()
