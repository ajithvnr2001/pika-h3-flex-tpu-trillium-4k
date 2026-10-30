# v4: real Pika H3 Flex video generation on a Spot Cloud TPU v6e-4

This folder runs the actual **Pika-Labs/Pika-H3-Flex** model (14B video+audio diffusion transformer, Qwen3-VL text
encoder, ViT video VAE, audio VAE, FL2VA LoRA adapter) through the **official diffusers pipeline** on a
**Spot TPU v6e-4** in project `genai25th01`. Your prompt changes the video. Nothing is simulated.

> Earlier folders (`version1-no-audio-4k`, `version2-with-audio-4k`, `v3-acceleration`) did **not** run the model.
> See [../SIMULATED_RESULTS_NOTICE.md](../SIMULATED_RESULTS_NOTICE.md).

## Results

| File | What it is |
| --- | --- |
| `outputs/kerala_5s.mp4` | Text-to-video, 832x480, 124 frames (5.17 s) @ 24 fps, stereo 32 kHz audio. Kerala houseboat prompt. |
| `outputs/kerala_ext.mp4` | Extension: generated from the last frame of `kerala_5s.mp4` with `prompts/prompt_extend.txt`. |
| `outputs/kerala_10s.mp4` | The two clips joined: 247 frames, 10.33 s, seamless cut (duplicate frame removed). |
| `outputs/kerala_10s_4k.mp4` | 3840x2160 copy of the 10 s clip (ffmpeg Lanczos resize, see "About 4K"). |
| `outputs/rajasthan_5s.mp4` | Second, different prompt (camel caravan in the Thar desert), made with the final pipeline (VAE on TPU). |
| `outputs/*.json` | Per-clip record: prompt, seed, every stage timing, ffprobe output, SHA-256 of the MP4. |
| `frames/*.png` | Contact sheets used to check that the video matches the prompt. |

### Measured timings (TPU v6e-4, 480p, 5 s clip, 6 denoising steps)

Each column is a real run on a fresh VM (no compile cache). Run 3 is the final pipeline in this folder.

| Stage | Run 1 (kerala_5s): VAE on CPU | Run 2 (same Kerala prompt, VAE on TPU; MP4 lost to preemption) | **Run 3 (rajasthan_5s): final** | Where |
| --- | ---: | ---: | ---: | --- |
| Load components from RAM disk | 3.3 s | 2.7 s | 3.0 s | host |
| Move transformer (+ VAE decoder) to TPU | 15.0 s | 15.1 s | 15.5 s | 4 TPU chips |
| Text encoder (Qwen3-VL) | 4.4 s | 4.5 s | 4.3 s | host CPU |
| Denoise step 1 (includes XLA compilation) | 131.5 s | 203.2 s | 202.2 s | TPU |
| Denoise step 2 | 132.4 s (compiles again) | 204.7 s (compiles again) | **14.1 s** | TPU |
| Denoise steps 3-6 (compiled) | 12.2-12.4 s each | 13.9-14.1 s each | 14.0-14.1 s each | TPU |
| Video VAE decode (37 latent frames to 124 frames) | 256.9 s | 65.0 s | **62.0 s** | CPU / TPU / TPU |
| Audio VAE decode | 2.0 s | 1.7 s | 1.7 s | host CPU |
| MP4 encode | 1.0 s | 1.1 s | 0.9 s | host CPU |
| **End to end** | **596 s** | **555 s** | **360 s** | |

The extension clip (first-frame mode, VAE on CPU) took 1,326 s, of which 956 s was the CPU VAE decode. The CPU
decode time varied 257-956 s between runs, which is why v4 decodes on the TPU.

**Where the time goes now (run 3).** About 200 s is the one-time XLA compilation in step 1. The 6 denoising steps
then take about 84 s and the TPU VAE decode about 62 s (which also includes its own compile). Run 2's compile was
longer than run 1's because float32 matmul precision was raised to "highest", so that the TPU VAE matches the CPU
decoder exactly. That setting only affects float32 ops; the bf16 transformer math is the same.

### Why this is slower than an H100

Pika validated the model on one H100 80 GB. A v6e chip has 32 GB, so the 62 GB transformer must be split over 4
chips by blocks (pipeline style: chip 0 runs blocks 0-12, then chip 1 runs 13-25, and so on). Only one chip
works at a time, and attention uses plain SDPA a few heads at a time, not a fused flash-attention kernel.

## The 4 commands (run from `/data/pika/v4-real-generation` on this machine)

```bash
cd /data/pika/v4-real-generation
export HF_TOKEN=hf_xxx                       # your Hugging Face token (rotate the old one)

# 1. Spin up the Spot TPU, install everything, download the model (~6-8 min)
commands/1_spin_up_spot_tpu.sh

# 2. Generate a video from a prompt (~6 min for 5 s at 480p on a fresh VM, measured 360 s)
commands/2_generate_video.sh prompts/prompt_5s.txt kerala_5s
#    or pass the prompt text directly:
commands/2_generate_video.sh "integrated_multimodal_description: A red kite flying over a quiet beach at dawn. overall_soundscape: Waves and wind. No speech. non_diegetic_music: None." kite_5s

# 3. Save the results into /data/pika/v4-real-generation/outputs
commands/3_save_to_data_pika.sh

# 4. Tear down the TPU so it stops billing
commands/4_tear_down_tpu.sh
```

To make a second video, repeat steps 2 and 3 before step 4. Step 1's installs and downloads are skipped when they
are already done, and nothing is reinstalled on a second run of step 2.

**Optional: 10 s video and a 4K copy (local, no TPU):**

```bash
# a. generate the extension from the last frame of the first clip (needs the TPU, so before step 4)
commands/2_generate_video.sh prompts/prompt_extend.txt kerala_ext 5 outputs/kerala_5s_last.png
commands/3_save_to_data_pika.sh
# b. join them and make the 4K copy
commands/extra_join_and_4k.sh outputs/kerala_5s.mp4 outputs/kerala_ext.mp4 kerala_10s
```

**Spot preemption.** Spot TPUs can be reclaimed at any time (it happened 3 times during this work). If a step
fails with `PREEMPTED`, run step 1 again. It deletes the dead TPU, creates a new one and reinstalls. Then rerun step 2.

## Writing prompts

Pika H3 Flex was trained on structured prompts. Use these three labelled parts (see `prompts/`):

```text
integrated_multimodal_description: <what is seen: subject, action, camera, light, style>
overall_soundscape: <what is heard>. No speech.
non_diegetic_music: None.
```

For an extension (first-frame mode) start with
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
and describe how the scene continues from that frame (`prompts/prompt_extend.txt`).

Supported: durations 5-15 s (`2_generate_video.sh ... <name> 10`), 480p (default) or 768p (`--resolution 768p`
on `generate_pika_tpu.py`). Longer clips and 768p increase sequence length, memory use and time.

## How it works

```
this machine (gcloud)                      Spot TPU VM v6e-4 (180 vCPU, 708 GB RAM, 4 chips x 32 GB HBM)
---------------------                      ----------------------------------------------------------
1_spin_up_spot_tpu.sh  -- create --------> bootstrap_tpu.sh
                       -- scp scripts -->    /dev/shm RAM disk (420 GB), ffmpeg
                                             Python 3.12 venv: torch 2.9 CPU + torch_xla 2.9 + diffusers@d0b9b1b
                                             hf download (146 GB, ~20-30 s inside Google's network)
2_generate_video.sh    -- prompt -------->  run.sh -> generate_pika_tpu.py
                                             host CPU : Qwen3-VL text encoder, audio VAE, MP4 encode
                                             xla:0-3  : transformer blocks 0-12 | 13-25 | 26-38 | 39-49
                                             xla:3    : video VAE decoder (float32)
3_save_to_data_pika.sh <-- scp MP4+JSON --  /dev/shm/pika/outputs
4_tear_down_tpu.sh     -- delete -------->  (gone)
```

### Files

| File | Purpose |
| --- | --- |
| `commands/common.sh` | Project, zone, TPU name, accelerator type. Override with env vars (`TPU=... ZONE=...`). |
| `commands/1_spin_up_spot_tpu.sh` | Create the Spot TPU (recreate if preempted), copy scripts, store `HF_TOKEN` in a 600-mode file, run bootstrap. |
| `commands/2_generate_video.sh` | Send a prompt (text or file) and optional first frame, run generation, stream progress. |
| `commands/3_save_to_data_pika.sh` | Copy outputs back and save each clip's last frame for extensions. |
| `commands/4_tear_down_tpu.sh` | Delete the TPU and list what is left in the zone. |
| `commands/extra_join_and_4k.sh` | Join a clip and its extension into ~10 s, make a 3840x2160 copy. |
| `scripts/bootstrap_tpu.sh` | Idempotent VM setup (runs on the TPU). |
| `scripts/generate_pika_tpu.py` | The generator (runs on the TPU). |
| `scripts/run.sh` | Activates the venv and calls the generator. |
| `scripts/test_vae_tpu.py` | Check that the VAE decoder gives the same pixels on TPU as on CPU. |
| `scripts/monitor.sh` | Logs TPU state and run progress every 5 minutes. |

### What `generate_pika_tpu.py` changes and what it does not

It loads the pipeline exactly as Pika's examples do (`ModularPipeline.from_pretrained`, `load_components` with
float32 VAEs and bf16 everything else, `load_lora_weights("fl2va/adapter.safetensors")`) and calls `pipe(...)`.
The sampler, the 6-step schedule, the packed sequence and the weights are untouched. Only placement changes:

1. **Transformer split across 4 chips.** Each `MiniMaxH3TransformerBlock` is moved to a chip and a forward
   pre-hook moves the activations to that chip. `torch_xla.sync()` after each block keeps each XLA graph to one
   block's work.
2. **Exact attention in head chunks.** Full attention over ~14,950 tokens for 56 heads at once needs a
   `f32[56, 14950, 14950]` buffer (50 GB), which does not fit a 32 GB chip. `HeadChunkedAttnProcessor` runs the same
   `scaled_dot_product_attention` on 4 heads at a time and concatenates the results. Heads are independent, so
   the output is identical.
3. **Video VAE decoder on TPU.** The 9.7 GB float32 ViT decoder runs on chip 3 at matmul precision "highest".
   `scripts/test_vae_tpu.py` measured a max absolute pixel difference of 9.4e-6 against the CPU decoder (pixel
   range about -4 to 5), and a warm 5 s decode of 16.6 s against 257 s on CPU.
4. **One compile instead of two.** Step 1 has one distinct noise level and steps 2-6 have two, so `timestep` has
   shape 1 then 2 and XLA would compile twice. The wrapper repeats the single value. All `timestep_indices` are 0
   in step 1, so the rows that are read are the same and the result does not change.
5. **Persistent XLA cache** at `/dev/shm/pika/xla_cache`: a later run with the same sequence length reuses the
   compiled programs. A new prompt with a different token count changes the sequence length and compiles again.

### Problems hit and how they were fixed

| Problem | Fix |
| --- | --- |
| TPU image ships Python 3.10; diffusers@d0b9b1b and numpy 2.5 need >= 3.11/3.12 | Separate Python 3.12 venv made with `uv`, no patching of diffusers |
| `ImportError: libpython3.12.so.1.0` from torch_xla | Add uv's Python `lib/` to `LD_LIBRARY_PATH` in the venv's `activate` |
| `hf download --include a b c` silently skipped `transformer/` | One `--include` per pattern, plus a check that every weight folder exists |
| Boot disk too small for 146 GB | Weights live in `/dev/shm` (RAM disk, 420 GB of the 708 GB RAM) |
| 50 GB attention buffer, `RESOURCE_EXHAUSTED` | Head-chunked exact attention |
| CPU video VAE decode 257-956 s | Decoder on TPU, verified identical, 65 s with compile / 17 s warm |
| Spot preemption mid-run (3 times) | Step 1 detects `PREEMPTED`, deletes and recreates |

## About 4K

Pika's open weights generate natively at **480p or 768p** only (model card: "Native 480p / 768p · Up to 4K via
Pika API"). Pika's 1080p/2K/4K upsampler is the hosted, paid
[draft-to-video API](https://dev.pika.art/models/pika/pika-h3-flex/draft-to-video), which is not in the open
release. The earlier folders made "4K" with an ffmpeg Lanczos resize, and so does `extra_join_and_4k.sh`. That gives
a 3840x2160 file but no new detail. Real 4K needs Pika's API or an AI upscaler such as Real-ESRGAN.

## About torch_tpu and sglang-torchtpu

Both repos were read before choosing the stack (cloned at `torch_tpu@6d33a5f`, `sglang-torchtpu@c477647`):

- **sglang-torchtpu** is an LLM serving engine. Its Pallas attention kernels are tuned for TPU7x (Ironwood) only;
  `sglang_tpu/multimodal_gen/tpu_splash_attn.py` refuses other device kinds. Its only diffusion model is
  Wan2.2-TI2V-5B (nightly test). There is no MiniMax-H3 / Pika integration, so it would mean writing a new model
  port, and its kernels would not run on v6e.
- **torch_tpu 0.2.0** (Python 3.11+) supports v6e. It offers eager `device="tpu"`, `torch.compile(backend="tpu")`
  and a Pallas `splash_sdpa`. Its FLUX case study on one v6e chip got 1.46x on a transformer block from splash
  attention + compile. It is the most promising next step for the 12-14 s per step, but it replaces torch_xla as the
  backend, so it needs its own test run.

v4 uses torch_xla 2.9 because it already runs the unmodified diffusers pipeline on v6e-4 with verified output.
No port was needed.

## Cost and cleanup

Run `commands/4_tear_down_tpu.sh` when done, then confirm with
`gcloud compute tpus tpu-vm list --zone=europe-west4-a --project=genai25th01` (should be empty).
Never commit tokens: `HF_TOKEN` is read from the environment and stored on the VM in `~/.hf.env` (mode 600) only.

## On-demand instead of Spot

`SPOT=0 commands/1_spin_up_spot_tpu.sh` creates an on-demand TPU (not preemptible, higher hourly price). The project
has on-demand v6e quota (16 chips per zone) in many zones. On 2026-10-30 every zone tried had no on-demand v6e-4
capacity: europe-west4-a, us-east5-a/b, us-south1-a, us-west1-c and asia-northeast1-b. The project lacks
permission for v6e-4 in us-central1-b and us-east1-d. Spot in europe-west4-a was the only option that worked. To
try another zone: `ZONE=us-east5-b SPOT=0 commands/1_spin_up_spot_tpu.sh` (then pass the same `ZONE=` to steps 2-4).
