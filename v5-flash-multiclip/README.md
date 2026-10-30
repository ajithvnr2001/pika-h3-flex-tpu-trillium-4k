# v5: Pika H3 Flex on Cloud TPU v6e-4 — fast, real text-to-video for long videos

This folder generates **real** videos with the open **Pika-Labs/Pika-H3-Flex** model (14B video+audio diffusion
transformer) on a Google Cloud **TPU v6e-4**, and joins clips into long videos.

v5 = [v4](../v4-real-generation/) + two speedups, both measured:

| | v4 | **v5** |
| --- | ---: | ---: |
| First 5 s clip on a fresh TPU | 377 s | **142 s** |
| Every later 5 s clip (batch mode) | ~377 s | **51-77 s** |
| 10-minute video (117 clips, one TPU) | ~12 h | **~1.7-2.5 h** |

1. **Pallas flash attention** (`--attention flash`, default): attention is ~5x faster, each denoising step 14 s -> 4.8 s.
2. **Batch mode** (`commands/2_generate_batch.sh`): many clips in one process; the model is loaded and compiled once.

Output: 832x480 (480p), 24 fps, stereo 32 kHz audio, 5-15 s per clip. v4 stays unchanged in its own folder.

---

## Contents

1. [Prerequisites (one time)](#1-prerequisites-one-time)
2. [Quick start: one video](#2-quick-start-one-video)
3. [Long videos: batch + join](#3-long-videos-batch--join)
4. [Writing prompts](#4-writing-prompts)
5. [Getting a TPU when capacity is short](#5-getting-a-tpu-when-capacity-is-short)
6. [Measured results](#6-measured-results)
7. [How it works](#7-how-it-works)
8. [Files](#8-files)
9. [Troubleshooting](#9-troubleshooting)
10. [Cost](#10-cost)
11. [Limits and not done](#11-limits-and-not-done)

---

## 1. Prerequisites (one time)

You run everything from a normal Linux machine (no GPU needed). The scripts create the TPU, run the model on it and
copy the results back.

| Need | How to check / get it |
| --- | --- |
| `gcloud` CLI, logged in | `gcloud auth login` then `gcloud config set project genai25th01` |
| Cloud TPU API enabled | `gcloud services enable tpu.googleapis.com` |
| TPU v6e quota (16 chips per zone here) | Console > IAM & Admin > Quotas, filter `tpu-v6e` (on-demand) / `tpu-v6e-preemptible` (Spot) |
| Hugging Face token | https://huggingface.co/settings/tokens (read access). Open https://huggingface.co/Pika-Labs/Pika-H3-Flex once and accept the license if asked |
| `ffmpeg` locally | `sudo apt-get install -y ffmpeg` (used by step 3 and by `join_clips.sh`) |

Everything else (Python 3.12, torch 2.9, torch_xla 2.9, jax 0.7.1, diffusers @ `d0b9b1b`, the 146 GB of model
weights) is installed **on the TPU** by step 1. Nothing heavy is installed on your machine.

```bash
git clone https://github.com/ajithvnr2001/pika-h3-flex-tpu-trillium-4k.git /data/pika
cd /data/pika/v5-flash-multiclip
export HF_TOKEN=hf_your_token_here      # never commit this; scripts read it from the environment
```

Settings live in `commands/common.sh` and can be overridden per command:
`PROJECT` (genai25th01), `ZONE` (europe-west4-a), `TPU` (pika-v5-tpu), `SPOT` (1 = Spot, 0 = on-demand),
`LOCAL_OUT` (outputs folder). Use the **same** `ZONE`/`TPU` values for every step of one session.

---

## 2. Quick start: one video

```bash
cd /data/pika/v5-flash-multiclip
export HF_TOKEN=hf_...

# STEP 1 - create the TPU, install everything, download the model (~7 min)
SPOT=0 commands/1_spin_up_spot_tpu.sh          # on-demand; use SPOT=1 (or omit) for cheaper Spot

# STEP 2 - generate (prompt file or prompt text in quotes) (~142 s for the first clip)
commands/2_generate_video.sh prompts/prompt_rajasthan_5s.txt rajasthan_5s
commands/2_generate_video.sh "integrated_multimodal_description: A red kite flying over a quiet beach at dawn. overall_soundscape: Waves and wind. No speech. non_diegetic_music: None." kite_5s

# STEP 3 - copy the MP4s + timing JSONs to ./outputs (and save each clip's last frame)
commands/3_save_to_data_pika.sh

# STEP 4 - delete the TPU so it stops billing
commands/4_tear_down_tpu.sh
```

What each step does:

| Step | On your machine | On the TPU |
| --- | --- | --- |
| 1 | Creates the TPU (or deletes and recreates a preempted one), copies scripts, writes `HF_TOKEN` into `~/.hf.env` (mode 600) on the VM | `bootstrap_tpu.sh`: 420 GB RAM disk at `/dev/shm`, ffmpeg, Python 3.12 venv, model download (~146 GB in ~20-35 s). Every part is skipped if already done |
| 2 | Copies the prompt (and optional first frame), streams progress | `generate_pika_tpu.py`: generate, check frame count/audio, write `<name>.mp4` + `<name>.json` |
| 3 | Copies `/dev/shm/pika/outputs/*` back, extracts `<name>_last.png` from each clip | - |
| 4 | Deletes the TPU and lists what is left in the zone | - |

`2_generate_video.sh` starts a new process each time, so **every call pays the ~90 s compile**. For more than one
clip use batch mode below.

---

## 3. Long videos: batch + join

A long video is many 5-15 s clips joined together. Two ways to make the clips:

### a) Separate shots (recommended: fastest, no drift)

Write one prompt file per shot, list them in a jobs file, run them all in **one** process:

```text
# my_jobs.txt  —  <prompt.txt> <name> [duration 5-15] [first_frame.png]
prompts/shot01.txt  shot01  5
prompts/shot02.txt  shot02  5
prompts/shot03.txt  shot03  5
```

```bash
SPOT=0 commands/1_spin_up_spot_tpu.sh
commands/2_generate_batch.sh my_jobs.txt          # clip 1 ~142 s, every later clip 51-77 s
commands/3_save_to_data_pika.sh
commands/4_tear_down_tpu.sh
commands/join_clips.sh outputs/my_long_video.mp4 outputs/shot01.mp4 outputs/shot02.mp4 outputs/shot03.mp4
```

`prompts/test_jobs.txt` is a working 4-clip example.

### b) One continuous shot (extension chain)

Each clip starts from the last frame of the previous one (first-frame mode, see `prompts/prompt_extend.txt`).
Each link needs the previous clip's frame, so run them one after another:

```bash
commands/2_generate_video.sh prompts/prompt_5s.txt part1 5
commands/3_save_to_data_pika.sh                                      # writes outputs/part1_last.png
commands/2_generate_video.sh prompts/prompt_extend.txt part2 5 outputs/part1_last.png
commands/3_save_to_data_pika.sh
CHAINED=1 commands/join_clips.sh outputs/continuous.mp4 outputs/part1.mp4 outputs/part2.mp4
```

`CHAINED=1` drops the duplicated first frame (and 1/24 s of audio) of every clip after the first so the cuts are
seamless. Quality drifts after many links; for minutes of video, use separate shots with occasional extensions.

### Time for long videos (one v6e-4, 480p)

| Length | Clips (5 s) | Batch time |
| --- | ---: | ---: |
| 1 min | 12 | ~12-17 min |
| 10 min | 117 | ~1.7-2.5 h |

The low end applies when prompts have the same token count (no recompile); each new prompt length adds ~25 s once.

---

## 4. Writing prompts

Pika H3 Flex was trained on structured prompts with three labelled parts:

```text
integrated_multimodal_description: <what is seen: subject, action, camera move, light, style>
overall_soundscape: <what is heard>. No speech.
non_diegetic_music: None.
```

Examples: `prompts/prompt_5s.txt` (Kerala houseboat), `prompts/prompt_rajasthan_5s.txt` (camel caravan),
`prompts/prompt_himalaya_5s.txt` (mountain monastery).

For an extension (first frame given), start with
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
and describe how the scene continues from that frame (`prompts/prompt_extend.txt`).

Durations 5-15 s are supported (third column / third argument). Longer clips cost more than proportionally
(attention grows with the square of the clip length), so 5 s clips are the fastest per second of video.

---

## 5. Getting a TPU when capacity is short

v6e-4 capacity is often exhausted (on 2026-10-30 every zone with quota ran out at least once; Spot was preempted 4
times). Options:

- **Another zone**: `ZONE=us-east5-b SPOT=0 commands/1_spin_up_spot_tpu.sh`, then pass the same `ZONE=` to steps 2-4.
  Zones with v6e quota for this project: europe-west4-a, us-east5-a/b, us-south1-a, us-west1-c, asia-northeast1-b,
  asia-east1-c, asia-south1-b, asia-southeast1-b, southamerica-east1-c, us-central1-a.
- **Queued request** (Google creates the TPU when capacity frees up; the on-demand TPU used for the v5 test arrived
  after ~30 min):

```bash
gcloud compute tpus queued-resources create pika-v5-qr --node-id=pika-v5-tpu --zone=europe-west4-a \
  --project=genai25th01 --accelerator-type=v6e-4 --runtime-version=v2-alpha-tpuv6e --valid-until-duration=8h  # add --spot for Spot
gcloud compute tpus queued-resources describe pika-v5-qr --zone=europe-west4-a --project=genai25th01 --format='value(state.state)'
# WAITING_FOR_RESOURCES -> PROVISIONING -> ACTIVE.  When ACTIVE:
SPOT=0 commands/1_spin_up_spot_tpu.sh        # finds the existing node and only installs
# ... steps 2 and 3 ...
gcloud compute tpus queued-resources delete pika-v5-qr --zone=europe-west4-a --project=genai25th01 --quiet --force   # instead of step 4
```

- **Spot preempted mid-run**: step 2 reports `STEP 2 FAILED` / missing clips. Run step 1 again (it deletes the dead
  node and recreates it), then rerun step 2 for the missing clips.

---

## 6. Measured results

On-demand TPU v6e-4, europe-west4-a, 2026-10-30, 480p, 5 s clips (124 frames), seed 42. The 4 flash clips ran in one
process (`prompts/test_jobs.txt`); the SDPA run is v4's method on the same VM. Per-clip records:
`outputs/*.json`; batch log: `logs/run_batch_tpu.log`.

| Clip | Method | Clip time | Denoise steps (s) | Video VAE |
| --- | --- | ---: | --- | ---: |
| `rajasthan_5s_v5a` (1st in batch, fresh VM) | flash | **142 s** (17.5 s setup + 122.6 s) | 30.1, 4.9, 5.1, 5.0, 4.9, 4.8 | 60.0 s |
| `kerala_5s_v5` (new prompt length) | flash | **76.6 s** | 31.2, 4.9, 4.8, 4.8, 4.8, 4.7 | 16.6 s |
| `rajasthan_5s_v5b` (same prompt length as clip 1) | flash | **50.8 s** | 4.8, 4.7, 4.9, 4.7, 4.8, 4.9 | 17.1 s |
| `himalaya_5s_v5` (new prompt length) | flash | **76.0 s** | 30.5, 4.9, 4.7, 4.7, 4.8, 4.7 | 16.9 s |
| `rajasthan_5s_sdpa_samevm` (v4 method) | SDPA | **376.6 s** | 200.9, 13.8, 14.0, 13.9, 13.9, 14.0 | 61.6 s |

Where a later clip's 51 s goes: text encoder ~2 s (host CPU), 6 denoising steps ~29 s (TPU), video VAE ~17 s
(TPU), audio VAE ~1.3 s and MP4 encode ~1 s (host CPU).

### Quality

`frames/sdpa_vs_flash_samevm.png` (top row SDPA, bottom row flash; same prompt, seed and VM) and
`frames/kerala_himalaya_v5.png`.

| Comparison | Result |
| --- | --- |
| SDPA on two different VMs | **byte-identical MP4** |
| Flash on two different VMs | **byte-identical MP4** |
| Flash, same prompt twice in one process | SSIM 0.985, PSNR 44.3 dB (visually identical) |
| Flash vs SDPA | SSIM 0.860, PSNR 27.5 dB |

Flash attention is the same formula computed in tiles; per call it differs from SDPA by bf16 rounding (max 5e-4).
Pika's sampler adds fresh noise at every step, so these tiny differences change small details of the result. The flash
video shows the same scene (desert, sun, three decorated camels, man in red turban, same walk) with slightly different
framing and leg positions: a different, equally sharp sample, not a degraded one. For output identical to v4, run
`generate_pika_tpu.py --attention sdpa` (v4 speed).

---

## 7. How it works

```
your machine (gcloud)                      TPU VM v6e-4: 4 chips x 32 GB HBM, 180 vCPU, 708 GB RAM
---------------------                      ----------------------------------------------------
1_spin_up_spot_tpu.sh  -- create/scp ----> bootstrap_tpu.sh -> /dev/shm RAM disk, venv, 146 GB weights
2_generate_*.sh        -- prompts ------->  generate_pika_tpu.py (one process for the whole batch)
                                              host CPU : Qwen3-VL text encoder, audio VAE, MP4 encode
                                              xla:0    : transformer blocks 0-12
                                              xla:1    : transformer blocks 13-25
                                              xla:2    : transformer blocks 26-38
                                              xla:3    : transformer blocks 39-49 + video VAE decoder
3_save_to_data_pika.sh <-- MP4 + JSON ----  /dev/shm/pika/outputs
4_tear_down_tpu.sh     -- delete -------->  (gone)
```

The official diffusers pipeline (`PikaH3FlexModularPipeline`, diffusers commit `d0b9b1b`, FL2VA adapter) runs
unchanged: same weights, same 6-step schedule `[1000, 750, 500, 250, 100, 40, 0]`, same sampler. Only placement and
the attention kernel change:

1. **Transformer split by blocks over 4 chips.** The 62 GB transformer does not fit one 32 GB chip. Note: the chips
   run one after another (only one computes at a time).
2. **Pallas flash attention** (`FlashAttnProcessor`). The profile (`scripts/profile_step.py`, `logs/profile_step.json`)
   showed attention was ~92% of each block: SDPA wrote a `heads x 15004 x 15004` float32 score matrix to memory 14
   times per block. The Pallas kernel (`torch_xla.experimental.custom_kernel.flash_attention`) computes it in tiles
   on-chip. The sequence is padded to a multiple of 512 and the padding is masked with segment ids, so real tokens
   never attend to padding. Block sizes `block_q=2048, block_k_major=1024, block_k=512` were the fastest of four
   tested (`scripts/test_flash_tpu.py`): 12.9 ms vs 63-76 ms per block.
   **Required fix:** torch_xla builds Pallas kernels through jax without the TPU attached, so jax writes Mosaic format
   v8, which libtpu 0.0.21 cannot read (`Unsupported version: expected <= 7 but got 8`). `enable_pallas()` forces v7.
3. **Batch mode** (`--jobs`). torch_xla's on-disk compile cache was written but not reused by a new process
   ([pytorch/xla#9094](https://github.com/pytorch/xla/issues/9094)), so v5 keeps one process for all clips; compiled
   programs stay in memory. A clip only recompiles step 1 (~25 s) when its prompt token count is new.
4. **Static timestep shape** (from v4): step 1 has one noise level, steps 2-6 two; the single value is repeated so
   the network compiles once (indices still point at entry 0, so the result is unchanged).
5. **Video VAE decoder on TPU** (from v4, float32 at matmul precision "highest"; max pixel difference vs CPU 9.4e-6).

---

## 8. Files

| File | Runs on | Purpose |
| --- | --- | --- |
| `commands/common.sh` | local | Project, zone, TPU name, Spot/on-demand, output folder |
| `commands/1_spin_up_spot_tpu.sh` | local | Create/recreate the TPU, copy scripts, store `HF_TOKEN`, run bootstrap |
| `commands/2_generate_video.sh` | local | One clip (prompt file or text, optional first frame) |
| `commands/2_generate_batch.sh` | local | Many clips from a jobs file in one TPU process |
| `commands/3_save_to_data_pika.sh` | local | Copy outputs back, save `<name>_last.png` |
| `commands/4_tear_down_tpu.sh` | local | Delete the TPU, list what is left |
| `commands/join_clips.sh` | local | Join clips with audio (`CHAINED=1` for extension chains) |
| `scripts/bootstrap_tpu.sh` | TPU | Idempotent setup |
| `scripts/generate_pika_tpu.py` | TPU | Generator (`--prompt-file/--output` or `--jobs`, `--attention flash|sdpa`, `--duration`, `--image`, `--seed`) |
| `scripts/run.sh` | TPU | Venv + one clip (used by step 2) |
| `scripts/profile_step.py` | TPU | Per-block / per-chip profile of one denoising step |
| `scripts/test_flash_tpu.py` | TPU | Flash attention vs SDPA speed and accuracy |
| `scripts/test_vae_tpu.py` | TPU | VAE decoder TPU vs CPU check |
| `scripts/monitor.sh` | local | Optional: log TPU state and run progress every 5 min |
| `prompts/` | - | Example prompts and `test_jobs.txt` |
| `outputs/` | - | Test videos and their JSON records |
| `frames/`, `logs/` | - | Quality contact sheets, profile, batch and provisioning logs |

---

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `There is no more capacity in the zone` / `Insufficient capacity` | Try another `ZONE=`, or queue a request (section 5) |
| `This TPU has terminal state "PREEMPTED"` | Spot was reclaimed. Run step 1 again, then rerun step 2 |
| `STEP 2 FAILED` / `MISSING: <name>.mp4` | Check `gcloud compute tpus tpu-vm describe $TPU --zone=$ZONE`; if READY, read `~/run_batch.log` on the VM via `gcloud compute tpus tpu-vm ssh` |
| `HF_TOKEN must be set` | `export HF_TOKEN=hf_...` before step 1 |
| `Unsupported version: expected <= 7 but got 8` | You are calling flash attention without `enable_pallas()`; use `generate_pika_tpu.py` as shipped |
| `RESOURCE_EXHAUSTED` / out of HBM | Longer or 768p clips use more memory; use 480p 5-10 s, or `--attention sdpa --head-chunk 4` |
| First clip slow (~2 min) | Normal: compilation. Later clips in the same batch are fast |

---

## 10. Cost

Billing runs while the TPU is READY, from step 1 until step 4. Prices for v6e in europe-west4 (Google billing
catalog, 2026-10-30): **on-demand $2.97/chip-h = $11.88/h**, **Spot $1.782/chip-h = $7.13/h** for a v6e-4.

| Job | Time incl. ~7 min setup | On-demand | Spot |
| --- | ---: | ---: | ---: |
| 1 clip | ~10 min | ~$2 | ~$1.2 |
| 1-min video (12 clips) | ~20-25 min | ~$4-5 | ~$2.5-3 |
| 10-min video (117 clips) | ~1.8-2.6 h | ~$21-31 | ~$13-19 (+ preemption retries) |

Always finish with step 4 and confirm with
`gcloud compute tpus tpu-vm list --zone=$ZONE --project=genai25th01` (empty).

---

## 11. Limits and not done

- **480p only here.** Pika's open weights generate 480p/768p natively; 1080p/2K/4K is Pika's paid
  draft-to-video API. ffmpeg resizing to 4K adds pixels but no detail. A latent upscaler is being evaluated for v6.
- **One chip computes at a time** (block split). Using all 4 chips at once (tensor parallelism) would speed each step
  ~2.5-3.5x but was not needed for long videos.
- **New prompt lengths recompile step 1 (~25 s).** Padding prompts to a fixed token count would remove that; not
  implemented.
- **Persistent compile cache across processes/VMs does not work** in torch_xla 2.9 (above).
- **Not bit-identical to v4** (flash attention; see Quality). `--attention sdpa` reproduces v4.
- PyTorch/XLA is deprecated (repository to be archived 2026-11-15) in favour of TorchTPU. The pinned versions keep
  working; future speed work should target TorchTPU.
