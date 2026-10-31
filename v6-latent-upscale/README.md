# v6: Pika H3 Flex upscaled videos (1664x960) on Cloud TPU v6e-4

> **Newer: [`../v7-auto-pipeline/`](../v7-auto-pipeline/)** packages this pipeline as one self-contained folder with
> full instructions, preemption recovery, a status/cost command, loudness levelling and auto teardown.

Real Pika H3 Flex text-to-video, **upscaled 2x to 1664x960** with the MiniMax H3 latent upscaler plus a short Pika
refine pass, for long videos made of many clips. All video work runs on the TPU VM; your machine only sends
commands and receives the finished files.

Tested 2026-10-31 on a Spot TPU v6e-4 (europe-west4-a): 6 clips, an extension chain and a join, all successful.
Also tested: a **1-minute Pixar-style video made by the automated pipeline** (12 clips, 62 min) and **resuming after a
cutoff** (resumed clips are byte-identical to an uninterrupted run).

## Easiest way: one automated command (recommended for long videos)

```bash
cd /data/pika/v6-latent-upscale
. /root/.secrets/pika.env                      # or: export HF_TOKEN=hf_...  (needed if the TPU must be recreated)
export TPU=pika-europe-tpu ZONE=europe-west4-a  # your TPU name and zone

# once per TPU: install + download model and upscaler (~7 min)
commands/1_spin_up_spot_tpu.sh

# make the whole video: all clips -> join -> QA -> copy to ./outputs. Runs in the background.
nohup commands/6_auto_pipeline.sh jobs_pixar_1min.txt pixar_1min > logs/pixar_1min.log 2>&1 &
tail -f logs/pixar_1min.log        # Ctrl+C stops watching, not the pipeline

# when finished (or whenever you want to stop paying):
commands/4_tear_down_tpu.sh
```

The log shows one line per clip (`Job 3/12 ...`, `PIPELINE saved locally: pixar03`) and ends with
`PIPELINE DONE: .../outputs/pixar_1min.mp4`. Each finished clip is already in `outputs/` while the rest are made.

### What if the Spot TPU is taken away (preempted) in the middle?

Spot TPUs are cheaper ($7.13/h vs $11.88/h for a v6e-4) but Google can reclaim them at any time; we were preempted
about 5 times during this project. `6_auto_pipeline.sh` handles it without you doing anything:

1. **Finished clips are saved off the TPU right away.** The generator runs detached on the TPU; every 60 s the script
   copies each finished clip (`<name>.mp4`, `<name>_last.png`, `<name>.json`) to `outputs/` on your machine. The TPU's
   own disk is lost when it is preempted, your local copy is not.
2. **It notices the cutoff.** When the run stops, it checks the TPU state. Gone / `PREEMPTED` / not `READY` means
   preemption.
3. **It gets a new TPU.** It deletes the dead one and runs step 1 again (create + install + download, ~8 min). If
   there is no capacity it waits 5 min and tries again; after 3 failed Spot tries it switches to an on-demand TPU
   (`ONDEMAND_AFTER=0` to stay on Spot only, `ONDEMAND_AFTER=5` to wait longer).
4. **It continues where it stopped.** Only clips missing from `outputs/` are made. If the next clip continues an
   earlier one (`@pixar05`), it is given the saved `outputs/pixar05_last.png`, so the story continues exactly.
5. **Join on the new TPU.** Clips made on the old TPU are uploaded back to the TPU, then joined there as usual.

Cost of one preemption: the clip that was in progress (up to ~4.5 min) + new TPU setup (~8 min) + the first-clip
compile again (~3 min), so about 15 min. Nothing that was finished is redone.

Other cases:
- **Your own machine reboots / you close the terminal:** run the same `6_auto_pipeline.sh` command again. Finished
  clips in `outputs/` are skipped. (If the TPU is still running it kills the old run and continues.)
- **A real error on a healthy TPU** (bad prompt file, out of memory): it retries once, then stops with
  `FAILED twice on a healthy TPU`. Read the error on the TPU with
  `gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --command "tail -30 ~/run_batch.log"`, fix it, rerun.
- **Gives up** after `MAX_ATTEMPTS` (default 10) generation runs. Rerunning resumes.
- **Want a fresh start of a clip?** delete its files from `outputs/` (`rm outputs/pixar07*`) and rerun.

How it was tested: a 3-clip chain, the generator killed in the middle of clip 2 twice and the TPU's outputs and frame
files deleted (what a new TPU looks like). The pipeline retried, made clips 2-3 from the saved local frame, uploaded
clip 1 back, joined and copied the result (`logs/resume_test2.log`). The three clips' SHA-256 equal those of the
uninterrupted Pixar run. The "TPU deleted by Google" branch reuses step 1, which is the same command used to recreate
TPUs throughout this project, but an actual preemption did not happen during this test.

### Pixar-style 1-minute test (`jobs_pixar_1min.txt`, `prompts/pixar/`)

12 linked 5 s shots (a girl, a small robot and a red kite), each `@` the previous one, upscaled to 1664x960.
Result: `outputs/pixar_1min.mp4`, 61.75 s, 1482 frames, with audio. Overview: `frames/production/pixar_1min_story.png`.

| Stage | Time |
| --- | ---: |
| Upload + model load | 2 min 01 s |
| Clip 1 (includes compiles) | 7 min 48 s |
| Clip 2 (first image-start clip, compiles once) | 6 min 19 s |
| Clips 3-12 | 3 min 53 s - 4 min 31 s each (avg ~4.4 min) |
| Join + QA + copy | 1 min 43 s |
| **Total** | **61 min 45 s** (~$7.30 on Spot) |

Rule of thumb: **~1 min of TPU time per 1 s of upscaled video** (5 min video ~4.5 h, 10 min ~9 h).

What it showed: strong Pixar look and the same two characters for the whole minute. Because every shot starts from
the previous last frame, things carry over: the robot's "idea" light bulb from shot 2 stayed on its head for the rest
of the video, and the hilltop scenes drifted back to the village lane. Tips:
- Start a shot fresh (no `@`) when the place changes (new location, sunset), and repeat the character descriptions.
- Don't add props you want gone later, or say in the next prompt that they are gone.
- The soundtrack is quiet (mean -37 dB); raise it when editing if needed.

### Prompt tips for realistic people

Tested face crops 480p vs upscaled (`frames/production/face_*`):
- Words like "beautiful", "cinematic", "photorealistic", "35mm film look", "realistic skin texture",
  "looks directly at the camera" give an airbrushed, glamour/advert face. The upscaler then makes it sharper and
  glossier but does not cause it.
- "visible skin pores, small blemishes, uneven skin tone" over-shoots: bumpy patches on the cheeks.
- Best (`prompts/natural_woman_v2.txt`): "handheld documentary footage, overcast natural daylight, ordinary ...,
  no makeup, natural plain complexion, talks to someone just off-camera, glancing away and back, muted natural
  colors, documentary realism". The upscale then adds real detail (hair, cloth, roof tiles) to a natural face.

## Commands (step by step)

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
| `commands/6_auto_pipeline.sh` | Whole video in one command; saves clips as they finish; survives Spot preemption |
| `commands/0_get_tpu.sh` | Find capacity in any zone or queue a request |
| `commands/1_spin_up_spot_tpu.sh` | Install + download on the TPU |
| `commands/2_generate_batch.sh`, `2_generate_video.sh` | Generate clips (upscaled by default; `DETACH=1` = run detached on the TPU) |
| `commands/5_join_on_tpu.sh` | Join clips on the TPU |
| `commands/3_save_to_data_pika.sh`, `4_tear_down_tpu.sh` | Copy results (incl. subfolders), delete the TPU |
| `jobs_pixar_1min.txt`, `prompts/pixar/` | The 1-minute Pixar test |
| `prompts/natural_woman_v2.txt` | Realistic-person prompt template |
| `scripts/generate_pika_tpu.py` | Generator (`--upscale`, `--refine-from`, `--seq-bucket`, `--keep-base`, jobs with `@name`) |
| `scripts/latent_upscaler.py` | Upscaler model |
| `scripts/qa_frames.sh` | Quality sheets / crops / SSIM, runs on the TPU |
| `outputs/test_script_run/` | First experiment (includes the no-refine variant, which has artifacts) |

## Limits

- Refine re-draws small details relative to the 480p draft (more with `refine_from=500`).
- TPU capacity: v6e-4 is often full; Spot can be preempted (`6_auto_pipeline.sh` resumes automatically; with the
  step-by-step commands, rerun steps 1-2 for missing clips).
- Clips are generated one after another; using idle chip time for the next clip (overlap) is not implemented.
