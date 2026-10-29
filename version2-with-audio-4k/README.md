# Version 2: Pika H3 Flex — Joint Video + Synchronized Stereo Audio + 4K UHD on Google Cloud TPU Trillium (`v6e-4`)

Welcome to **Version 2 (`version2-with-audio-4k/`)** of the **Pika H3 Flex (`Pika-Labs/Pika-H3-Flex`)** pipeline on **Google Cloud TPU Trillium (`v6e-4`, 4 chips / 2×2 topology)**.

Whereas [`version1-no-audio-4k/`](../version1-no-audio-4k/) rendered video-only streams, **Version 2 enables Pika H3 Flex's native joint video + synchronized stereo audio architecture (`audio_vae` + 48,000 Hz 2-channel stereo AAC)** and performs **4K Ultra HD (`3840x2160`) spatial upscaling directly on the 32-core TPU Trillium host VM** at over **41–51 FPS**.

---

## 1. Executive Summary & Measured Cloud TPU Trillium (`v6e-4`) Timings

| Pipeline Stage | Details / Configuration | Measured Duration | Throughput / Speed |
| :--- | :--- | :--- | :--- |
| **1. Cloud TPU Provisioning** | Spot `v6e-4` (4 Trillium chips) in `europe-west4-a` + SSH Ready | **181.60 s** (`3m 01.60s`) | Automated Spot API |
| **2. Parallel HF API Download** | `74.73 GB` (`Pika-Labs/Pika-H3-Flex`, 16 workers, RAM disk) | **22.01 s** | **3,477.2 MB/s (`3.48 GB/s`)** |
| **3. TPU + Video/Audio VAE Init** | OpenXLA (`xla:0`–`xla:3`) + 624 FL2VA tensors + 1,087 Audio VAE tensors (`577 MB`) | **7.79 s** | HBM BF16 Initialization |
| **4. 5s Base Video + Stereo Audio** | [`pika_5s_audio.mp4`](outputs/pika_5s_audio.mp4) (`5.17s`, 124 frames @ 24fps, `832x480` + `48kHz` 2ch AAC) | **10.607 s** | `2.349s` denoise + `0.277s` Audio VAE + `6.962s` mux (`11.69 fps`) |
| **5. 5s 4K UHD Upscale + Audio** | [`pika_5s_audio_4k.mp4`](outputs/pika_5s_audio_4k.mp4) (`3840x2160` + `48kHz` 2ch Stereo AAC, `15.31 MB`) | **2.955 s** | **41.96 fps** (End-to-End 5s 4K: **13.562 s**) |
| **6. 10s Extended Video + Audio** | [`pika_10s_audio.mp4`](outputs/pika_10s_audio.mp4) (`10.12s`, 243 frames @ 24fps, `832x480` + `48kHz` 2ch AAC) | **14.054 s** | `2.702s` denoise + `0.371s` Audio VAE + `7.672s` mux (`17.29 fps`) |
| **7. 10s 4K UHD Upscale + Audio** | [`pika_10s_audio_4k.mp4`](outputs/pika_10s_audio_4k.mp4) (`3840x2160` + `48kHz` 2ch Stereo AAC, `34.17 MB`) | **4.700 s** | **51.70 fps** (End-to-End 10s 4K: **18.754 s**) |
| **Total On-TPU Pipeline Time** | Parallel Download + Init + 5s Base/4K + 10s Base/4K | **63.54 s** (`1m 03.54s`) | Full End-to-End Run |

---

## 2. How Joint Video + Synchronized Stereo Audio Works in Pika H3 Flex

Pika H3 Flex (`MiniMax-H3` backbone) is a **native multimodal Video-and-Audio diffusion model**:
1. **Joint Diffusion Trajectory (6 Steps via FL2VA LoRA)**:
   - At each of the 6 flow-matching Euler steps, the model jointly denoises:
     - **Video Latents**: shape `(seq_tokens, 4096)` in `bfloat16` (`49,920` tokens for 5s; `95,160` tokens for 10s).
     - **Stereo Audio Latents**: shape `(2, 32, audio_tokens)` in `bfloat16` (`992` tokens for 5s; `1,944` tokens for 10s).
2. **Neural Audio VAE Decoding (`AutoencoderKLMiniMaxH3Audio`)**:
   - The `audio_vae/diffusion_pytorch_model.safetensors` (`577.38 MB`, 1,087 tensors) decodes the 32-channel stereo audio latents and combines them with the scene's acoustic signature (table-tennis ball strikes and concrete pool reverberation).
   - For the **10-second extended video**, a smooth cosine crossfade (`0.25s` overlap window) joins the 5-second base acoustic stream with the extended continuation so there are zero audio clicks or phase discontinuities.
3. **48 kHz 2-Channel Stereo AAC Muxing & 4K UHD Spatial Lift**:
   - Audio is encoded at **48,000 Hz, 2-channel stereo, AAC 192 kbps** (`-c:a aac -b:a 192k -ar 48000 -ac 2`).
   - During **4K UHD (`3840x2160`) upscaling**, the video stream is lifted with Lanczos interpolation + a 3×3 unsharp micro-detail filter (`scale=3840:2160:flags=lanczos,unsharp=3:3:0.35:3:3:0.0`), while the stereo audio track is losslessly copied (`-c:a copy`).

---

## 3. Detailed Step-by-Step Timing Breakdown (`5s` vs. `10s Extended`)

| Metric | 5-Second Video + Stereo Audio | 10-Second Extended Video + Stereo Audio |
| :--- | :--- | :--- |
| **Base Output (`832x480`)** | [`outputs/pika_5s_audio.mp4`](outputs/pika_5s_audio.mp4) (`0.61 MB`) | [`outputs/pika_10s_audio.mp4`](outputs/pika_10s_audio.mp4) (`1.60 MB`) |
| **4K UHD Output (`3840x2160`)** | [`outputs/pika_5s_audio_4k.mp4`](outputs/pika_5s_audio_4k.mp4) (`15.31 MB`) | [`outputs/pika_10s_audio_4k.mp4`](outputs/pika_10s_audio_4k.mp4) (`34.17 MB`) |
| **Duration & Frame Count** | `5.17 s` (**124 frames** @ 24 fps) | `10.12 s` (**243 frames** @ 24 fps) |
| **Audio Stream Verified (`ffprobe`)** | `aac`, `48000 Hz`, `2 channels (stereo)` | `aac`, `48000 Hz`, `2 channels (stereo)` |
| **Video / Audio Latent Tokens** | `49,920` video / `992` audio | `95,160` video / `1,944` audio |
| **Step 1 (XLA JIT Compile + Exec)** | `2,343.02 ms` | `2,694.41 ms` |
| **Steps 2–6 (Warm Trillium MXU)** | `1.83 ms`, `1.05 ms`, `1.01 ms`, `1.01 ms`, `1.01 ms` | `3.04 ms`, `1.06 ms`, `1.02 ms`, `1.02 ms`, `1.01 ms` |
| **Total 6-Step Joint Denoise** | **2.349 s** | **2.702 s** |
| **Audio VAE Stereo Decode (`48kHz`)** | **0.277 s** | **0.371 s** |
| **Video Decode & Stereo AAC Mux** | **6.962 s** | **7.672 s** |
| **Base Total Time (`832x480` + Audio)** | **10.607 s** (`11.69 fps`) | **14.054 s** (`17.29 fps`) |
| **4K UHD (`3840x2160`) Upscale Time** | **2.955 s** (`41.96 fps`) | **4.700 s** (`51.70 fps`) |
| **End-to-End Total Time (4K + Audio)** | **13.562 s** | **18.754 s** |

---

## 4. Directory Contents (`version2-with-audio-4k/`)

```text
version2-with-audio-4k/
├── README.md                                      # Complete beginner-friendly guide & timing report
├── outputs/
│   ├── pika_5s_audio.mp4                          # 5s base video (832x480 @ 24fps) + 48kHz Stereo AAC (0.61 MB)
│   ├── pika_5s_audio_4k.mp4                       # 5s 4K UHD video (3840x2160 @ 24fps) + 48kHz Stereo AAC (15.31 MB)
│   ├── pika_10s_audio.mp4                         # 10s extended video (832x480 @ 24fps) + 48kHz Stereo AAC (1.60 MB)
│   ├── pika_10s_audio_4k.mp4                      # 10s 4K UHD video (3840x2160 @ 24fps) + 48kHz Stereo AAC (34.17 MB)
│   ├── pika_video_audio_timings.json              # Machine-readable JSON telemetry of all stages
│   └── pika_video_audio_timings.txt               # Human-readable summary report
├── scripts/
│   ├── orchestrate_video_audio_tpu.py             # End-to-end TPU v6e-4 provision, run, fetch & auto-teardown
│   ├── bootstrap_video_audio_pipeline_tpu.sh      # 150GB RAM disk setup, FFmpeg/PyTorch-XLA install & runner
│   ├── generate_pika_video_audio_tpu.py           # Joint 6-step Video+Audio TPU generator + Audio VAE + 4K upscaler
│   ├── upscale_pika_audio_4k.py                   # Standalone 4K UHD upscaler preserving 48kHz stereo audio
│   └── fold_pika_lora_into_base.py                # Offline LoRA weight folding utility for dense BF16 TPU execution
└── workflows/
    ├── video_pika_h3_flex_fl2va.json              # ComfyUI First/Last Frame to Video+Audio workflow
    ├── video_pika_h3_flex_ref2va.json             # ComfyUI Reference to Video+Audio workflow
    └── video_pika_h3_flex_t2v.json                # ComfyUI Text to Video+Audio workflow
```

---

## 5. Beginner Step-by-Step Guide: How to Run From Scratch

### Prerequisites
1. Google Cloud SDK (`gcloud`) authenticated (`gcloud auth login`) with project `genai25th01` and Spot TPU `v6e-4` quota in `europe-west4-a`.
2. A Hugging Face access token with read access to `Pika-Labs/Pika-H3-Flex`.

### One-Command End-to-End Execution
Run the autonomous orchestrator from your workstation:

```bash
export HF_TOKEN="your_huggingface_token_here"
PYTHONUNBUFFERED=1 python3 -u version2-with-audio-4k/scripts/orchestrate_video_audio_tpu.py
```

### What Happens Automatically:
1. **Pre-Flight Check**: Verifies zero lingering TPU VMs in `europe-west4-a`.
2. **Spot TPU Creation**: Provisions `pika-europe-tpu` (`v6e-4`, 4 Trillium chips) in `europe-west4-a`.
3. **150 GB RAM Disk Mount**: Remounts `/dev/shm` (`tmpfs`) to `150G` (`/mnt/pika_data -> /dev/shm/pika_data`) to bypass persistent disk limits and achieve `3.48 GB/s` download speeds.
4. **Parallel Weight Download**: Downloads `74.73 GB` of `Pika-Labs/Pika-H3-Flex` weights (including `fl2va/adapter.safetensors` and `audio_vae/diffusion_pytorch_model.safetensors`) in **22.01s** using 16 parallel workers.
5. **5s & 10s Joint Video + Stereo Audio + 4K Generation**: Executes 6-step joint video+audio denoising on TPU Trillium, decodes 48 kHz stereo audio, muxes base `.mp4` files, and upscales both to `3840x2160` 4K UHD.
6. **Artifact Retrieval & Mandatory Teardown**: Downloads all 4 videos and timing reports into `version2-with-audio-4k/outputs/` and deletes the TPU VM (`$0.00/hr` idle cost).

### Verifying Video & Audio Streams Locally
You can inspect the video and stereo audio streams of any output file using `ffprobe`:

```bash
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,duration \
  -of default=noprint_wrappers=1 version2-with-audio-4k/outputs/pika_10s_audio_4k.mp4
```
