# v7 end-to-end test (2026-10-31, Spot TPU v6e-4, europe-west4-a)

Command, run from this folder with the v7 scripts:

```bash
commands/1_spin_up_spot_tpu.sh
nohup commands/6_auto_pipeline.sh jobs_v7_test.txt v7_test > logs/v7_test.log 2>&1 &
```

`jobs_v7_test.txt` = Pixar shots 1 and 2 (`v7test02` continues from `@v7test01`), upscaled to 1664x960.

## A real Spot preemption happened during the test

From `logs/v7_test.log`:

```text
[12:15:40] PIPELINE run 1: 2 clip(s) to make
[12:16:47] Job 1/2: v7test01.mp4 t2va 5s seed=42 upscale->1664x960 refine_from=250
[12:20:11] PIPELINE TPU is STOPPING: Spot preemption, will recreate and resume
[12:20:13] PIPELINE TPU pika-europe-tpu is STOPPING (preempted or not set up) -> recreating (SPOT=1)
[12:28:52] PIPELINE TPU pika-europe-tpu ready again
[12:29:52] Job 1/2: v7test01.mp4 t2va 5s ...
[12:39:01] PIPELINE saved locally: v7test01
[12:44:40] PIPELINE saved locally: v7test02
[12:45:01] PIPELINE all clips are saved locally
```

Google preempted the Spot TPU 3.5 min into clip 1. With no user action, the pipeline detected it, deleted the TPU,
created a new Spot TPU and installed everything (8 min 39 s), and made both clips. Time lost to the preemption:
about 13 min.

## A bug found and fixed

The first join failed: the new loudness filter changed the channel layout and ffmpeg refused to connect it
(`Cannot select channel layout`). Fixed in `commands/5_join_on_tpu.sh` (`aformat=...:channel_layouts=stereo`). Then
**the same pipeline command was run again** (`logs/v7_test_rerun.log`): it skipped both finished clips, joined,
ran QA and copied in 50 s:

```text
[12:49:53] PIPELINE all clips are saved locally
[12:50:13] PIPELINE joining on TPU
QA v7test01 frame-to-frame SSIM 0.974969
QA v7test02 frame-to-frame SSIM 0.980503
QA v7_test frame-to-frame SSIM 0.977481
[12:50:43] PIPELINE DONE: /data/pika/v7-auto-pipeline/outputs/v7_test.mp4
```

## Result checks

| Check | Result |
| --- | --- |
| `outputs/v7_test.mp4` | 1664x960, 247 frames (124 + 123: duplicate seam frame removed), 10.33 s, AAC 32 kHz stereo |
| Clips identical to the earlier uninterrupted v6 run (same prompt, seed) | `v7test01` SHA-256 `3d1b4210...` = `pixar01`; `v7test02` `362df3a7...` = `pixar02` |
| Loudness levelling | mean volume -46.4 dB (raw clip) -> -18.7 dB (joined), peak -1.6 dB |
| Temporal stability (SSIM) | 0.97-0.98 |
