# v6 (experimental): 2x latent upscale + Pika refine -> 1664x960

**Status: tested once on 2 clips; works and looks clearly sharper; not yet packaged into the 4-step commands.**
Use [v5](../v5-flash-multiclip/) for production 480p.

This tests the community **MiniMax H3 latent upscaler**
([LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler), v1, 345M
params, Apache-2.0, revision `3f941d5`) on real Pika H3 Flex output. Pika H3 Flex is built on MiniMax H3 and uses
the same 24-channel video VAE (identical latent mean/std), so the upscaler applies to Pika latents.

## Pipeline tested (`scripts/upscale_test.py`)

1. **Stage 1:** normal v5 generation at 832x480 (flash attention). The final latents `(1, 24, 37, 30, 52)` and audio
   latents are captured at the VAE decoders.
2. **Latent upscale 2x** on TPU chip 1: `(1, 24, 37, 30, 52) -> (1, 24, 37, 60, 104)`.
3. **Variant A, no refine:** decode the upscaled latents directly at 1664x960.
4. **Variant B, refine:** noise the upscaled latents to Pika's sigma at timestep 500 (video 0.923, audio 0.750) and
   run Pika's own last 4 steps `[500, 250, 100, 40]` at 1664x960 (same model, adapter and sampler), then decode.
   Audio is the stage-1 soundtrack.

## Important: latent convention (measured, `scripts/check_upscaler_convention.py`)

Real frames of a clip were encoded at 416x240 and 832x480 with Pika's VAE encoder; the upscaled low-res latent was
compared with the real high-res latent (normalized space):

| Input fed to the upscaler | Correlation with real 2x latent | RMSE |
| --- | ---: | ---: |
| Plain trilinear 2x (baseline) | 0.831 | 0.655 |
| Pika normalized latents `n` | -0.112 | 6.137 |
| Raw VAE latents `z` | -0.074 | 10.251 |
| **`(n - mean) / std`, output `* std + mean`** | **0.978** | **0.236** |

The upscaler expects Pika's already-normalized latents to be normalized **once more** (the ComfyUI node normalizes the
ComfyUI latent, which for MiniMax H3 is already in normalized form). Fed that way it is far better than trilinear;
fed any other way the output is garbage. `upscale_test.py --convention comfy` (default) does this.

## Results (on-demand TPU v6e-4, europe-west4-a, 2026-10-30, seed 42)

Files in `outputs/`: `<name>_480p.mp4`, `<name>_up2x_norefine.mp4`, `<name>_up2x_refine.mp4`, `<name>_upscale.json`.
All 124 frames, 24 fps, with audio.

### Quality (`frames/`)

`*_full_*.png` and `*_crop_*.png`: left = 480p Lanczos-resized to 1664x960, middle = variant A, right = variant B,
same frame (t = 2.5 s); crops are 1:1 pixels.

| | Variant A (no refine) | **Variant B (refine)** |
| --- | --- | --- |
| Sharpness | about the same as Lanczos | **clearly sharper**: saddle weave, ropes, camel fur, monastery windows, prayer flags, snow ridges |
| Artifacts | **yes**: grid/ghost pattern on prayer flags and mountain texture | none seen in the checked frames |
| Same content as the 480p clip? | yes | **mostly**: same scene, layout, motion and camera, but details are re-drawn (e.g. saddle colours and patterns, flag arrangement) |
| Temporal | stable | stable across the clip (`frames/*_refine_over_time.png`) |

Verdict: **variant A is not useful** (worse than a plain resize). **Variant B gives real added detail** at 1664x960.
Its trade-off is that starting from 92% noise lets the model change small details relative to the 480p draft.

### Time per 5 s clip

| Stage | rajasthan (1st, fresh process) | himalaya (2nd) |
| --- | ---: | ---: |
| Stage 1, 480p (6 steps) | 124.0 s | 76.5 s |
| Latent upscale | 59.4 s (compile) / 9.0 s | 9.7 s |
| Variant A decode | 62.4 s | 62.4 s |
| Variant B: 4 refine steps | 72.4 + 32.2 + 32.3 + 32.2 = 169 s | 72.6 + 32.4 + 32.5 + 32.5 = 170 s |
| Variant B: decode at 1664x960 | 62.2 s | 63.7 s |
| **Variant B total** (stage 1 + upscale + refine + decode) | ~415 s | ~325 s |

The first refine step includes ~40 s of compilation each time the prompt length changes; with an unchanged prompt
length a refine clip would be ~76 + 9 + 129 + 62 = **~4.6 min**. At 1664x960 the sequence is ~58k tokens (3.9x
the 480p one), so each step costs 32 s vs 4.8 s.

**10-minute video (117 clips) with variant B, one TPU, sequential: ~9-11 h** (~$105-130 on-demand, ~$65-80 Spot),
vs ~1.7-2.5 h at 480p (v5).

## Possible next steps

- **Refine from a later step** (`--refine-from 250`: sigma 0.80, 3 steps instead of 4): should keep the 480p
  draft's details closer and cost ~25% less. Not tested yet.
- **Overlap clips** (generate clip N+1's 480p stage in the idle chip time of clip N's refine): estimated ~2-3x more
  clips per hour. Not built.
- Package as `--upscale 2` in the v5 batch generator.

## Files

| File | Purpose |
| --- | --- |
| `scripts/latent_upscaler.py` | The upscaler model ported from the ComfyUI node (no ComfyUI dependency); verified 322/322 tensors, 345,280,216 params |
| `scripts/upscale_test.py` | The test above (`--jobs`, `--scale`, `--refine-from`, `--skip-refine`, `--convention`) |
| `scripts/check_upscaler_convention.py` | Real-frame check of the latent convention (CPU) |
| `scripts/bootstrap_tpu.sh` | v5 bootstrap + upscaler download (step 5/5) |
| `commands/` | v5 commands with `TPU=pika-v6-tpu` |
| `logs/upscale_tpu.log` | Full TPU log of the run; `logs/*run1*` is the first attempt (failed on a video-encode bug and the wrong convention) |

## TPU cost of this test

On-demand v6e-4 from ~22:09 to 22:46 UTC on 2026-10-30 (obtained via a queued request after ~40 min of waiting):
~37 min, ~$7.3, plus ~6 min on the first attempt.
