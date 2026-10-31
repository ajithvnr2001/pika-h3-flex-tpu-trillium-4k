# v6: Pika H3 Flex upscaled videos (1664x960) on Cloud TPU v6e-4

Real Pika H3 Flex text-to-video, **upscaled 2x to 1664x960** with the MiniMax H3 latent upscaler plus a short Pika
refine pass, for long videos made of many clips. All video work runs on the TPU VM; your machine only sends
commands and receives the finished files.

Tested 2026-10-31 on a Spot TPU v6e-4 (europe-west4-a): 6 clips, an extension chain and a join, all successful.

## Commands

```bash
cd /data/pika/v6-latent-upscale
export HF_TOKEN=hf_...

# 0. (only if no TPU exists) get one; waits in a queue if every zone is full. Prints ZONE=...
SPOT=0 commands/0_get_tpu.sh          # SPOT=1 for Spot

# 1. install + download model and upscaler on the TPU (~7 min). Use the TPU name/zone you have:
TPU=pika-v6-tpu ZONE=europe-west4-a commands/1_spin_up_spot_tpu.sh

# 2. generate all clips (upscaled), one process
TPU=... ZONE=... commands/2_generate_batch.sh my_shots.txt

# 2b. join them into one long video on the TPU
TPU=... ZONE=... commands/5_join_on_tpu.sh my_shots.txt my_long_video

# 3. copy MP4s to ./outputs
TPU=... ZONE=... commands/3_save_to_data_pika.sh

# 4. delete the TPU
TPU=... ZONE=... commands/4_tear_down_tpu.sh
```

Or set them once: `export TPU=pika-europe-tpu ZONE=europe-west4-a` and run the commands without the prefix.

### Shot list (`my_shots.txt`)

```text
# <prompt.txt>        <name>   [seconds] [first frame]  [options]
prompts/shot01.txt    shot01   5
prompts/shot02.txt    shot02   5         @shot01        # continues from the last frame of shot01
prompts/shot03.txt    shot03   5
```

- `@name` = start from the last frame of an earlier clip in the same file (continuous shot). The join drops the
  duplicated frame, so the cut is seamless.
- Options: `refine_from=250` (default) or `refine_from=500`, `seed=N`.
- `GEN_ARGS=""` before step 2 gives plain 480p (v5 speed). `GEN_ARGS="--upscale 2 --keep-base"` also keeps the 480p
  version of each clip.
- Prompt format: see `prompts/` and the v5 README (three labelled parts: description, soundscape, music).

## Results (`outputs/`, quality images in `frames/production/`)

| Clip | Refine | Time | Notes |
| --- | --- | ---: | --- |
| `raj_r500` (1st clip, includes all compiles) | from 500 (4 steps) | 485 s | |
| `raj_r250` | from 250 (3 steps) | **206 s** | steady-state reference |
| `him_r500` | from 500 | 271 s | |
| `him_r250` | from 250 | 275 s | text encoder was slow this time (48 s vs ~4 s) |
| `kerala_a` | from 500 | 318 s | |
| `kerala_b` (extension of kerala_a) | from 500 | 600 s | rerun in a fresh process after the memory fix (includes compiles) |
| `kerala_10s` | - | - | kerala_a + kerala_b joined on the TPU: 1664x960, 247 frames, 10.33 s |

**Quality.** Checked at 1:1 pixels (`frames/production/qa_*_crop.png`) and over time (`qa_*_sheet.png`):

- Rajasthan: saddle embroidery, tassels, bells, ropes and camel fur sharp at both settings; no artifacts.
- Himalaya: snow ridges, glacier, cloud and gold roof detail sharp at both; no artifacts.
- `refine_from=250` keeps the 480p clip's composition closest and is ~25% faster, with detail as good as 500, so it
  is the default.
- Extension seam (`qa_kerala_10s_seam.png`, frames 121-126): no visible jump.
- Temporal stability (frame-to-frame SSIM): Rajasthan 0.94-0.95, Himalaya 0.90, Kerala 0.85 (water and birds move a
  lot), close to the 480p originals.

**Typical time per 5 s clip: ~3.5-4.5 min** after the first clip of a batch (first clip ~8 min including compiles).
10-minute video (117 clips): roughly **7-9 h** on one TPU v6e-4.

## How it works (per clip, all on the TPU VM)

1. 480p generation, 6 Pika steps on 4 TPU chips (flash attention). The 480p decode is skipped.
2. Latent upscale 2x on TPU chip 1 (MiniMax H3 latent upscaler, 345M params).
3. Noise to Pika's sigma at timestep 250, then Pika's last 3 steps at 1664x960 (same model and sampler).
4. Video VAE decode on chip 3, audio VAE on chip 2 (480p soundtrack kept), MP4 encode, last frame saved.

Placement: transformer blocks 14/14/13/9 over chips 0-3 (chip 3 also holds the 9.7 GB VAE decoder; an even split ran
out of memory on the extension clip). Video VAE encoder (for `@name` first frames) and audio VAE on chip 2. Only the
text encoder (67 GB, does not fit next to the transformer) and MP4 writing use the TPU VM's own CPU.

Other details:
- **Fixed sequence bucket** (`--seq-bucket 1024`): the packed sequence is padded with masked dummy rows so prompts of
  different lengths reuse one compile (padding is excluded from attention by flash segment ids; results unchanged).
- **Upscaler input convention:** it must receive Pika's normalized latents normalized once more (measured with real
  frames: correlation 0.978 with the true 2x latent, vs -0.11 for the obvious form). See
  `scripts/check_upscaler_convention.py`.
- Upscaler: [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
  rev `3f941d5`, Apache-2.0, ported in `scripts/latent_upscaler.py` (322/322 tensors verified).

## Files

| File | Purpose |
| --- | --- |
| `commands/0_get_tpu.sh` | Find capacity in any zone or queue a request |
| `commands/1_spin_up_spot_tpu.sh` | Install + download on the TPU |
| `commands/2_generate_batch.sh`, `2_generate_video.sh` | Generate clips (upscaled by default) |
| `commands/5_join_on_tpu.sh` | Join clips on the TPU |
| `commands/3_save_to_data_pika.sh`, `4_tear_down_tpu.sh` | Copy results, delete the TPU |
| `scripts/generate_pika_tpu.py` | Generator (`--upscale`, `--refine-from`, `--seq-bucket`, `--keep-base`, jobs with `@name`) |
| `scripts/latent_upscaler.py` | Upscaler model |
| `scripts/qa_frames.sh` | Quality sheets / crops / SSIM, runs on the TPU |
| `outputs/test_script_run/` | First experiment (includes the no-refine variant, which has artifacts) |

## Limits

- Refine re-draws small details relative to the 480p draft (more with `refine_from=500`).
- TPU capacity: v6e-4 is often full; Spot can be preempted (rerun steps 1-2 for missing clips).
- Clips are generated one after another; using idle chip time for the next clip (overlap) is not implemented.
