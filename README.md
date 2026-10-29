# Pika H3 Flex on Google Cloud TPU Trillium (`v6e-4`) — 5s & 10s Extended 4K UHD Video & Stereo Audio Pipeline

End-to-end engineering repository for running **Pika H3 Flex (`Pika-Labs/Pika-H3-Flex`)** 6-step flow-matching inference, **5-second to 10-second video extension**, **synchronized 48 kHz stereo audio (`audio_vae`)**, and **4K Ultra HD (`3840x2160`) spatial upscaling** on **Google Cloud TPU Trillium (`v6e-4`, 4 chips / 2×2 topology)**.

---

## Repository Organization (Two Complete Versions)

This repository is organized into two self-contained, end-to-end versions so you can compare and run either the **video-only 4K pipeline** or the **full joint video + synchronized stereo audio 4K pipeline**:

| Folder | Description | Generated Outputs Included |
| :--- | :--- | :--- |
| **[`version1-no-audio-4k/`](version1-no-audio-4k/)** | **Version 1 (Video-Only + 4K UHD)**: 6-step TPU Trillium video generation (`5s` and `10s extended`) + 4K UHD (`3840x2160`) upscaling without audio tracks, plus complete TorchTPU / SGLang / PyTorch-XLA architectural research. | `pika_5s.mp4`, `pika_5s_4k.mp4`, `pika_10s.mp4`, `pika_10s_4k.mp4`, `pika_video_timings.json` |
| **[`version2-with-audio-4k/`](version2-with-audio-4k/)** | **Version 2 (Joint Video + Stereo Audio + 4K UHD)**: Joint 6-step video + 32-channel audio latent denoising on TPU Trillium, `AutoencoderKLMiniMaxH3Audio` (`audio_vae`) 48 kHz 2-channel stereo AAC synthesis & muxing, and on-TPU 4K UHD (`3840x2160`) upscaling. | `pika_5s_audio.mp4`, `pika_5s_audio_4k.mp4`, `pika_10s_audio.mp4`, `pika_10s_audio_4k.mp4`, `pika_video_audio_timings.json` |

---

## Side-by-Side Performance & Timing Comparison (`v1` vs. `v2`)

Both versions were executed on **Google Cloud TPU Trillium `v6e-4` (4 chips, `europe-west4-a`)** using a **150 GB RAM disk (`/dev/shm/pika_data`)** and **16-worker parallel Hugging Face API downloads**:

| Stage / Metric | [`version1-no-audio-4k`](version1-no-audio-4k/) (Video-Only 4K) | [`version2-with-audio-4k`](version2-with-audio-4k/) (Video + 48kHz Stereo Audio 4K) |
| :--- | :--- | :--- |
| **Accelerator** | Cloud TPU Trillium `v6e-4` (4 chips) | Cloud TPU Trillium `v6e-4` (4 chips) |
| **Parallel HF Weight Download (`74.73 GB`)** | **34.02 s** (`2,249.6 MB/s`) | **22.01 s** (`3,477.2 MB/s` / `3.48 GB/s`) |
| **Hardware & Weights Init** | **7.58 s** (FL2VA Adapter) | **7.79 s** (FL2VA Adapter + 577 MB Audio VAE) |
| **Audio Stream** | None (Video only) | **AAC 48,000 Hz, 2-Channel Stereo (`192 kbps`)** |
| **5s Base Generation (`832x480`, 124 frames)** | **10.028 s** (`2.350s` denoise) | **10.607 s** (`2.349s` denoise + `0.277s` Audio VAE) |
| **5s 4K UHD Upscale (`3840x2160`)** | **55.060 s** (local CPU) $\rightarrow$ `65.088s` total | **2.955 s** (32-core TPU VM @ `41.96 fps`) $\rightarrow$ **13.562 s** total |
| **10s Extended Base Generation (`832x480`, 243 frames)** | **13.143 s** (`2.647s` denoise) | **14.054 s** (`2.702s` denoise + `0.371s` Audio VAE) |
| **10s 4K UHD Upscale (`3840x2160`)** | **111.730 s** (local CPU) $\rightarrow$ `124.873s` total | **4.700 s** (32-core TPU VM @ `51.70 fps`) $\rightarrow$ **18.754 s** total |
| **Total On-TPU Pipeline Runtime** | **65.96 s** (Base) | **63.54 s** (Includes Base + Stereo Audio + 4K UHD!) |

---

## Quick Start

### Run Version 2 (Joint Video + Stereo Audio + 4K UHD)
```bash
export HF_TOKEN="your_huggingface_token_here"
PYTHONUNBUFFERED=1 python3 -u version2-with-audio-4k/scripts/orchestrate_video_audio_tpu.py
```

### Run Version 1 (Video-Only + 4K UHD)
```bash
export HF_TOKEN="your_huggingface_token_here"
PYTHONUNBUFFERED=1 python3 -u version1-no-audio-4k/scripts/orchestrate_video_tpu.py
python3 -u version1-no-audio-4k/scripts/upscale_pika_4k.py
```

Both orchestrators automatically tear down the Spot TPU VM (`pika-europe-tpu`) in a `finally:` block so your idle cloud cost is guaranteed to remain **`$0.00/hr`**.
