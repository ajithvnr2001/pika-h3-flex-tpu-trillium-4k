# v7: Pika H3 Flex long upscaled videos on Cloud TPU v6e-4 — one automated command, survives Spot preemption

This folder makes **real** videos with the open **[Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex)**
model (video + audio diffusion transformer) on a Google Cloud **TPU v6e-4**, upscales every clip to **1664x960**, and
joins the clips into one long video. You write a shot list, run one command, and get the finished MP4 on your machine.

All video work (generation, upscaling, decoding, joining, QA) runs **on the TPU VM**. Your machine only sends commands
and receives files, so it can be a small VM with 2 CPUs and no GPU.

v7 = the tested [v6](../v6-latent-upscale/) upscale pipeline packaged as a self-contained folder, with:

| Feature | What it does |
| --- | --- |
| **One-command pipeline** (`6_auto_pipeline.sh`) | Generate all clips → join on the TPU → QA frames → copy to `outputs/`, unattended |
| **Spot preemption recovery** | Saves each clip locally as it finishes; if Google takes the TPU, it gets a new one and resumes from the next clip |
| **On-demand fallback** | After 3 failed Spot attempts it switches to an on-demand TPU (configurable) |
| **Resume any time** | Rerun the same command: finished clips are skipped |
| **Detached TPU run** | A dropped SSH connection does not stop or duplicate the generation |
| **Loudness levelling** (new) | The joined soundtrack is levelled to -16 LUFS (Pika's raw audio is quiet, about -37 dB) |
| **Status command** (new, `7_status.sh`) | TPU state, clips done, current clip, time left, money spent so far |
| **Auto teardown** (new, `AUTO_TEARDOWN=1`) | Deletes the TPU when the video is done so billing stops |
| **Latent upscale + refine** | MiniMax H3 latent upscaler 2x + Pika's last 3 steps at 1664x960 (real detail, not an ffmpeg resize) |
| **Continuous shots** | `@name` starts a clip from the last frame of the previous one; the join removes the duplicate frame |

Measured on a Spot v6e-4 in europe-west4-a (2026-10-31): a **1-minute video (12 clips) took 61 min 45 s, about
$7.30**. During the v7 test a **real Spot preemption** happened; the pipeline got a new TPU by itself and finished.
Clips made after a cutoff are byte-identical to an uninterrupted run.

---

## Contents

1. [Prerequisites (one time)](#1-prerequisites-one-time)
2. [Quick start: a whole video in one command](#2-quick-start-a-whole-video-in-one-command)
3. [Writing the shot list and prompts](#3-writing-the-shot-list-and-prompts)
4. [Spot preemption: what happens and what you do](#4-spot-preemption-what-happens-and-what-you-do)
5. [Watching progress and cost](#5-watching-progress-and-cost)
6. [Step-by-step commands (manual mode)](#6-step-by-step-commands-manual-mode)
7. [Getting a TPU when capacity is short](#7-getting-a-tpu-when-capacity-is-short)
8. [Measured results](#8-measured-results)
9. [How it works](#9-how-it-works)
10. [All settings](#10-all-settings)
11. [Files](#11-files)
12. [Troubleshooting](#12-troubleshooting)
13. [Cost](#13-cost)
14. [Limits](#14-limits)

---

## 1. Prerequisites (one time)

| Need | How to check / get it |
| --- | --- |
| `gcloud` CLI, logged in | `gcloud auth login` then `gcloud config set project genai25th01` (your project) |
| Cloud TPU API enabled | `gcloud services enable tpu.googleapis.com` |
| TPU v6e quota | Console > IAM & Admin > Quotas, filter `tpu-v6e` (on-demand) and `tpu-v6e-preemptible` (Spot). A v6e-4 needs 4 chips |
| Hugging Face token | https://huggingface.co/settings/tokens (read). Open https://huggingface.co/Pika-Labs/Pika-H3-Flex once and accept the license if asked |
| `ffmpeg`, `python3` locally | `sudo apt-get install -y ffmpeg python3` (only for looking at results; all processing is on the TPU) |

Everything heavy (Python 3.12, torch 2.9, torch_xla 2.9, jax 0.7.1, diffusers @ `d0b9b1b`, ~147 GB of Pika weights,
the 691 MB upscaler) is installed **on the TPU** by step 1.

```bash
git clone https://github.com/ajithvnr2001/pika-h3-flex-tpu-trillium-4k.git /data/pika
cd /data/pika/v7-auto-pipeline
```

Keep your token in a private file instead of typing it each time (never commit it):

```bash
mkdir -p ~/.secrets && chmod 700 ~/.secrets
echo 'export HF_TOKEN=hf_your_token_here' > ~/.secrets/pika.env && chmod 600 ~/.secrets/pika.env
. ~/.secrets/pika.env
```

Settings are in `commands/common.sh` and can be overridden per command. The ones you normally set:

```bash
export TPU=pika-v7-tpu          # name of your TPU (any name)
export ZONE=europe-west4-a      # zone with v6e quota/capacity
export SPOT=1                   # 1 = Spot (cheaper, can be preempted), 0 = on-demand
```

---

## 2. Quick start: a whole video in one command

```bash
cd /data/pika/v7-auto-pipeline
. ~/.secrets/pika.env
export TPU=pika-v7-tpu ZONE=europe-west4-a SPOT=1

# 1. Create the TPU, install everything, download the model + upscaler (~7-8 min, once per TPU)
commands/1_spin_up_spot_tpu.sh

# 2. Make the whole video (runs in the background; safe to close the terminal)
nohup commands/6_auto_pipeline.sh jobs_pixar_1min.txt pixar_1min > logs/pixar_1min.log 2>&1 &

# 3. Watch it (Ctrl+C stops watching, not the pipeline)
tail -f logs/pixar_1min.log
commands/7_status.sh jobs_pixar_1min.txt         # any time: progress, time left, cost so far

# 4. When it says "PIPELINE DONE", your video is in outputs/pixar_1min.mp4. Delete the TPU:
commands/4_tear_down_tpu.sh
```

To delete the TPU automatically at the end, start step 2 with `AUTO_TEARDOWN=1`:

```bash
AUTO_TEARDOWN=1 nohup commands/6_auto_pipeline.sh my_video.txt my_video > logs/my_video.log 2>&1 &
```

What the log looks like:

```text
[09:22:12] PIPELINE start: jobs_pixar_1min.txt -> pixar_1min.mp4 on pika-europe-tpu (europe-west4-a), 12 clips
[09:22:12] PIPELINE run 1: 12 clip(s) to make
[09:24:13] Job 1/12: pixar01.mp4 t2va 5s seed=42 upscale->1664x960 refine_from=250
[09:32:20] PIPELINE saved locally: pixar01
[09:32:01] Job 2/12: pixar02.mp4 fl2va 5s ... first_frame=pixar01_last.png
...
[10:22:31] PIPELINE all clips are saved locally
[10:22:31] PIPELINE joining on TPU
[10:23:57] PIPELINE DONE: /data/pika/v7-auto-pipeline/outputs/pixar_1min.mp4
```

(TPU timestamps and local timestamps can differ by a few seconds; lines can appear slightly out of order.)

What you get in `outputs/`:

| File | What |
| --- | --- |
| `<final>.mp4` | The joined video, 1664x960, 24 fps, stereo AAC, loudness levelled |
| `<clip>.mp4` | Each 5 s clip (1664x960 with audio) |
| `<clip>_last.png` | Lossless last frame of each clip (used to continue the next clip) |
| `<clip>.json` | Prompt, seed, settings, stage timings, SHA-256 of the clip |
| `qa_<clip>_sheet.png`, `qa_<clip>_crop.png` | 5 frames over time and a 1:1 crop, for checking quality |

---

## 3. Writing the shot list and prompts

### Shot list (jobs file)

One clip per line. Lines starting with `#` are ignored. Copy `jobs_template.txt` to start.

```text
# <prompt.txt>             <name>    [seconds] [first frame]  [options]
prompts/pixar/shot01.txt   pixar01   5
prompts/pixar/shot02.txt   pixar02   5         @pixar01       # continue from the last frame of pixar01
prompts/pixar/shot03.txt   pixar03   5         @pixar02       seed=7
prompts/pixar/shot06.txt   pixar06   5                        # fresh start: new location
```

| Field | Values |
| --- | --- |
| `name` | Unique, letters/digits/`_` only. Output is `outputs/<name>.mp4` |
| `seconds` | 5 (tested); 5-15 accepted by the generator. Longer clips at 1664x960 were not tested for memory |
| first frame | `-` or empty = text only; `@<name>` = continue from that earlier clip; `path/to/image.png` = start from your image |
| options | `refine_from=250` (default, 3 refine steps, closest to the draft) or `500` (4 steps); `seed=N` (default 42) |

Clips are joined **in file order**. A 1-minute video is 12 lines of 5 s; 10 minutes is about 120 lines.

### Prompt files

Pika H3 Flex was trained with three labelled parts. Copy `prompts/TEMPLATE.txt`:

```text
integrated_multimodal_description: <style>. <characters, described the same way in every shot>. <what happens in this shot, camera>.
overall_soundscape: <ambient sounds, speech (language), effects>.
non_diegetic_music: <music, or None.>
```

### Tips we measured

**Continuity (from the 1-minute Pixar test):**
- Every `@` shot starts from the previous last frame, so anything visible carries over. The robot's "idea" light bulb
  from shot 2 stayed on its head for the rest of the video, and hilltop scenes drifted back to the village.
- When the place changes (new location, time of day), start that shot fresh (no `@`) and repeat the full character
  description, so the characters still match.
- Don't add props you'll want gone later, or say in the next prompt that they are gone.

**Realistic people (face test, `prompts/natural_woman_v2.txt`):**
- Avoid "beautiful", "cinematic", "photorealistic", "35mm film look", "realistic skin texture", "looks directly at the
  camera": they give an airbrushed, advert-style face.
- Don't ask for "pores, blemishes, uneven skin": it over-shoots into bumpy patches.
- Use: "handheld documentary footage, overcast natural daylight, ordinary ..., no makeup, natural plain complexion,
  talks to someone just off-camera, glancing away and back, muted natural colors, documentary realism".
- Exact spoken lines with lip sync (for example Tamil dialogue to camera) are at the edge of what this open model does.

---

## 4. Spot preemption: what happens and what you do

Spot TPUs cost $7.13/h instead of $11.88/h for a v6e-4, but Google can take them back at any time (we were preempted
about 5 times during this project). **You don't need to do anything**; `6_auto_pipeline.sh` handles it:

1. **Finished clips are saved off the TPU right away.** The generator runs detached on the TPU. Every 60 s the
   pipeline copies each finished clip (`.mp4`, `_last.png`, `.json`) to `outputs/`. The `.json` is written last, so
   only complete clips are copied. The TPU's disk is lost when it is preempted; your copy is not.
2. **It notices the cutoff.** If the TPU is gone, `PREEMPTED` or not `READY`, that is a preemption.
3. **It gets a new TPU.** It deletes the dead one and runs step 1 again (~8 min). With no capacity it waits 5 min and
   retries. After 3 failed Spot attempts it switches to on-demand (`ONDEMAND_AFTER=0` = never switch).
4. **It continues where it stopped.** Only clips missing from `outputs/` are made. A clip that continues a finished
   clip (`@pixar05`) is given `outputs/pixar05_last.png`, so the story continues exactly.
5. **It joins on the new TPU.** Clips made on the old TPU are uploaded back first.

Cost of one preemption: about 15 min (the clip in progress, up to ~4.5 min + new TPU ~8 min + first-clip compile
~3 min). Nothing finished is redone.

| Situation | What happens / what you do |
| --- | --- |
| Spot TPU preempted | Automatic: new TPU, resume |
| No capacity anywhere | Keeps retrying every 5 min; on-demand after 3 tries. Or stop it and use [step 0](#7-getting-a-tpu-when-capacity-is-short) |
| Your machine reboots / you close the terminal (without `nohup`) | Run the same `6_auto_pipeline.sh` command again; finished clips are skipped |
| SSH connection drops | Nothing: the TPU run is detached and the next poll reconnects |
| Real error on a healthy TPU (bad prompt path, out of memory) | Retries once, then stops with `FAILED twice on a healthy TPU`. Read `~/run_batch.log` on the TPU (see [Troubleshooting](#12-troubleshooting)), fix, rerun |
| You want to redo one clip | `rm outputs/<name>*` and rerun the command |
| Gives up after `MAX_ATTEMPTS` (10) runs | Rerun; it resumes |

How it was tested:
- **A real Spot preemption** happened during the v7 test ([`V7_TEST.md`](V7_TEST.md)): the TPU was taken 3.5 min into
  clip 1. The pipeline recreated it (8 min 39 s) and finished the video with no user action.
- **Killed mid-run** (v6 test): the generator was killed twice in the middle of clip 2 and the TPU's outputs were
  deleted. The pipeline retried, made clips 2-3 from the saved local frame, uploaded clip 1 back and joined.
- In both tests the clips had the same SHA-256 as an uninterrupted run, so a cutoff does not change the video.

---

## 5. Watching progress and cost

```bash
commands/7_status.sh jobs_pixar_1min.txt
```

```text
TPU pika-europe-tpu (europe-west4-a): READY
On the TPU: [09:55:46] Job 7/12: pixar07.mp4 fl2va 5s seed=42 first_frame=pixar06_last.png upscale->1664x960
Clips saved locally: 6 / 12  (in /data/pika/v7-auto-pipeline/outputs)
Time left: about 26 min (~4.4 min per 5 s clip; +8 min for a first clip)
TPU running for 2.62 h -> about $18.68 so far ($7.13/h, SPOT=1)
```

The cost is counted from when the TPU was created at list price, so it includes idle time.

---

## 6. Step-by-step commands (manual mode)

The automated pipeline calls these; you can also run them yourself.

```bash
commands/0_get_tpu.sh                                  # only if no TPU: try all zones / queue; prints ZONE=...
commands/1_spin_up_spot_tpu.sh                         # create (if needed) + install + download (~7-8 min)
commands/2_generate_batch.sh my_video.txt              # all clips in one process (streams progress)
commands/2_generate_video.sh "prompt text or file" name 5 [@prev|img.png]   # a single clip
commands/5_join_on_tpu.sh my_video.txt my_video        # join on the TPU (LOUDNORM=0 to keep raw levels)
commands/3_save_to_data_pika.sh                        # copy everything from the TPU to outputs/
commands/4_tear_down_tpu.sh                            # delete the TPU (and a queued request)
```

Manual mode does **not** save clips while they are made. If a Spot TPU is preempted, clips not yet copied by step 3
are lost. Use `6_auto_pipeline.sh` for anything long.

`GEN_ARGS` changes the generator flags for steps 2 and 6: default `--upscale 2`; `GEN_ARGS=""` = plain 480p (about
1 min per clip); `GEN_ARGS="--upscale 2 --keep-base"` also keeps the 480p version as `<name>_base.mp4`.

---

## 7. Getting a TPU when capacity is short

v6e-4 is often sold out. `0_get_tpu.sh` tries every zone in `ZONES` (11 by default) and, if all are full, queues a
request in `$ZONE` that waits until Google provides one (up to 8 h):

```bash
SPOT=1 commands/0_get_tpu.sh        # prints e.g. ZONE=us-east5-b at the end
export ZONE=us-east5-b              # use that zone for every later command
commands/1_spin_up_spot_tpu.sh
```

Each zone needs its own quota. In this project `us-central1-b` and `us-east1-d` returned permission denied.

---

## 8. Measured results

### 1-minute Pixar-style video (`jobs_pixar_1min.txt`, `prompts/pixar/`)

12 linked 5 s shots (a girl, a small robot and a red kite), 1664x960, made with `6_auto_pipeline.sh` on a Spot v6e-4.
Result (made with the v6 copy of this pipeline, before loudness levelling):
[`../v6-latent-upscale/outputs/pixar_1min.mp4`](../v6-latent-upscale/outputs/pixar_1min.mp4), 61.75 s, 1482 frames.
Frames: `frames/pixar_1min_story.png`.

| Stage | Time |
| --- | ---: |
| Upload + model load | 2 min 01 s |
| Clip 1 (includes compiles) | 7 min 48 s |
| Clip 2 (first image-start clip, compiles once) | 6 min 19 s |
| Clips 3-12 | 3 min 53 s - 4 min 31 s each (avg ~4.4 min) |
| Join + QA + copy | 1 min 43 s |
| **Total** | **61 min 45 s** (~$7.30 on Spot) |

Rule of thumb: **about 1 min of TPU time per 1 s of upscaled video.**

| Video length | Clips | TPU time | Spot cost | On-demand cost |
| --- | ---: | ---: | ---: | ---: |
| 1 min | 12 | ~1 h | ~$7 | ~$12 |
| 5 min | 60 | ~4.5 h | ~$32 | ~$54 |
| 10 min | 120 | ~9 h | ~$64 | ~$107 |

(Plus ~8 min setup per TPU and ~15 min per Spot preemption.)

### v7 end-to-end test (`jobs_v7_test.txt`)

2 linked clips run from this folder with the v7 scripts; a real Spot preemption happened and was recovered
automatically. Result `outputs/v7_test.mp4` (1664x960, 10.33 s, loudness levelled). Details: [`V7_TEST.md`](V7_TEST.md).

### Quality

- Upscaled clips at 1:1 pixels: hair strands, fabric weave, roof tiles and fur are sharp; no grid artifacts with the
  refine pass (frames in `../v6-latent-upscale/frames/production/`).
- Frame-to-frame SSIM (temporal stability) per clip: 0.67-0.98. Low values come from fast motion and camera moves
  (running, wind), not flicker.
- Faces: see the prompt tips in section 3; the upscaler keeps the face identical to the 480p draft and adds detail.

---

## 9. How it works

Per clip, all on the TPU VM (`scripts/generate_pika_tpu.py`):

1. **480p draft:** 6 Pika denoising steps at 832x480 on 4 TPU chips with Pallas flash attention (~4.8 s per step).
   The text encoder (67 GB) runs on the TPU VM's CPU because it does not fit next to the transformer in TPU memory.
2. **Latent upscale 2x** on chip 1 with the MiniMax H3 latent upscaler (345M params).
3. **Refine:** noise to Pika's sigma at timestep 250, then Pika's last 3 steps at 1664x960 (~32 s per step).
4. **Decode:** video VAE on chip 3, audio VAE on chip 2, MP4 encode, lossless last frame saved, `.json` written last.

Placement: transformer blocks 14/14/13/9 on chips 0-3 (chip 3 also holds the VAE decoder). All clips of a run are made
in one process, so the model is loaded and compiled once (`--seq-bucket 1024` pads prompts so new prompt lengths
don't recompile).

The automated pipeline (`commands/6_auto_pipeline.sh`):

```text
loop:
  remaining = clips without outputs/<name>.json   (@prev -> outputs/prev_last.png if prev was made earlier)
  if none: break
  ensure TPU READY + set up      (else delete + step 1; Spot -> on-demand after ONDEMAND_AFTER failures)
  start generator DETACHED on the TPU (setsid + flock, exit code -> ~/batch.rc)
  every SYNC_EVERY s: copy finished clips to outputs/; stop when batch.rc exists, the run died, or the TPU is gone
upload clips missing on the TPU -> join on TPU (loudnorm) -> QA on TPU -> copy to outputs/ -> optional teardown
```

---

## 10. All settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `TPU` | `pika-v7-tpu` | TPU name |
| `ZONE` | `europe-west4-a` | Zone |
| `PROJECT` | `genai25th01` | Google Cloud project |
| `SPOT` | `1` | 1 = Spot, 0 = on-demand |
| `ACCEL` / `RUNTIME` | `v6e-4` / `v2-alpha-tpuv6e` | TPU type and image |
| `LOCAL_OUT` | `./outputs` | Where results are saved |
| `GEN_ARGS` | `--upscale 2` | Generator flags (`""` = 480p, add `--keep-base` to keep 480p) |
| `MAX_ATTEMPTS` | `10` | Generation runs before the pipeline gives up |
| `ONDEMAND_AFTER` | `3` | Failed Spot creations before switching to on-demand (`0` = never) |
| `SYNC_EVERY` | `60` | Seconds between saves/polls |
| `AUTO_TEARDOWN` | `0` | `1` = delete the TPU when the video is done |
| `LOUDNORM` | `1` | Level the joined soundtrack to -16 LUFS (`0` = raw levels) |
| `HF_TOKEN` | - | Needed by step 1 (and by the pipeline if it must recreate the TPU) |

---

## 11. Files

| File | Purpose |
| --- | --- |
| `commands/6_auto_pipeline.sh` | **Whole video in one command**, saves clips as they finish, survives preemption |
| `commands/7_status.sh` | Progress, time left, cost so far |
| `commands/0_get_tpu.sh` | Find capacity in any zone or queue a request |
| `commands/1_spin_up_spot_tpu.sh` | Create/recreate the TPU, install, download model + upscaler |
| `commands/2_generate_batch.sh`, `2_generate_video.sh` | Generate clips (`DETACH=1` = start detached on the TPU) |
| `commands/5_join_on_tpu.sh` | Join clips on the TPU (seamless `@` cuts, loudness levelling) |
| `commands/3_save_to_data_pika.sh`, `4_tear_down_tpu.sh` | Copy results, delete the TPU |
| `commands/common.sh` | Settings |
| `scripts/bootstrap_tpu.sh` | Runs on the TPU: RAM disk, ffmpeg, venv, model + upscaler download (skips what's done) |
| `scripts/generate_pika_tpu.py` | Generator (flash attention, batch, upscale + refine, `@name`) |
| `scripts/latent_upscaler.py` | MiniMax H3 latent upscaler port |
| `scripts/qa_frames.sh` | Runs on the TPU: frame sheets, crops, SSIM |
| `scripts/check_upscaler_convention.py` | How the upscaler's input scaling was verified |
| `jobs_template.txt`, `prompts/TEMPLATE.txt` | Start your own video here |
| `jobs_pixar_1min.txt`, `prompts/pixar/` | The 1-minute Pixar example |
| `jobs_v7_test.txt` | 2-clip test of v7 |
| `prompts/natural_woman_v2.txt` | Realistic-person example |
| `prompts/prompt_*_5s.txt` | Landscape examples (Rajasthan, Himalaya, Kerala) |

---

## 12. Troubleshooting

| Problem | Fix |
| --- | --- |
| `HF_TOKEN is not set` | `. ~/.secrets/pika.env` (or `export HF_TOKEN=...`) before the command |
| `/root/pika-venv/bin/activate: No such file or directory` | The TPU is new (recreated). Run `commands/1_spin_up_spot_tpu.sh` (the auto pipeline does this itself) |
| `no more capacity` / `Insufficient capacity` | Use `commands/0_get_tpu.sh`, another `ZONE`, or `SPOT=0` |
| `does not have permission` for a zone | No quota there; pick another zone |
| `FAILED twice on a healthy TPU` | `gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --command "tail -40 ~/run_batch.log"` |
| Out of memory (`RESOURCE_EXHAUSTED`) on a long clip | Use 5 s clips; the 1664x960 refine was tested at 5 s |
| A clip looks wrong | `rm outputs/<name>*`, edit its prompt, rerun the pipeline (only that clip is redone, but the join is redone) |
| Pipeline stopped and you are not sure why | `commands/7_status.sh <jobs>` and the end of `logs/<name>.log` |
| Kill a running pipeline | `pkill -f "[6]_auto_pipeline"` locally, then on the TPU `pkill -f "[g]enerate_pika_tpu.py"` |

Use the `[x]` bracket form in `pkill -f` patterns so the command does not match (and kill) its own shell.

---

## 13. Cost

v6e prices (2026-10): Spot $1.782 per chip-hour = **$7.13/h** for a v6e-4; on-demand $2.97 = **$11.88/h**. You pay
while the TPU exists, also when idle. Always finish with `commands/4_tear_down_tpu.sh`, or use `AUTO_TEARDOWN=1`, and
check nothing is left:

```bash
gcloud compute tpus tpu-vm list --zone=$ZONE
gcloud compute tpus queued-resources list --zone=$ZONE
```

---

## 14. Limits

- Resolution: 1664x960. The open weights are made for 480p/768p; Pika's 4K is its paid API. An ffmpeg resize to 4K
  adds no detail.
- Clips are made one after another on one TPU (the four chips split the model). A 10-minute video takes ~9 h.
- Long multi-shot stories drift (props carry over, places change); see the continuity tips.
- Realistic talking faces with exact lip sync in a given language are at the model's limit.
- The refine pass redraws small details compared with the 480p draft.
