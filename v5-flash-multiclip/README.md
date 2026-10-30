# v5: faster Pika H3 Flex on TPU v6e-4 (flash attention + many videos per process)

v5 is [v4](../v4-real-generation/) with two speed changes. The model, weights, 6-step sampler, prompts and outputs
format are the same.

1. **Pallas flash attention** for the transformer's self-attention (`--attention flash`, default).
2. **Batch mode**: `commands/2_generate_batch.sh jobs.txt` runs many videos in one process, so the model is loaded
   and compiled once and later clips skip compilation.

v4 is unchanged and still works (`v4-real-generation/`).

## Results

Measured on an **on-demand** TPU v6e-4, europe-west4-a, 2026-10-30, 480p, 5 s clips (124 frames), seed 42.
All 4 batch clips ran in one process (`prompts/test_jobs.txt`); the SDPA run is v4's method on the same VM.

| Clip | Method | Clip time | Denoise steps (s) | Video VAE |
| --- | --- | ---: | --- | ---: |
| `rajasthan_5s_v5a` (1st in batch, fresh VM) | flash | **142 s** (17.5 s setup + 122.6 s) | 30.1, 4.9, 5.1, 5.0, 4.9, 4.8 | 60.0 s |
| `kerala_5s_v5` (new prompt length) | flash | **76.6 s** | 31.2, 4.9, 4.8, 4.8, 4.8, 4.7 | 16.6 s |
| `rajasthan_5s_v5b` (same prompt length as clip 1) | flash | **50.8 s** | 4.8, 4.7, 4.9, 4.7, 4.8, 4.9 | 17.1 s |
| `himalaya_5s_v5` (new prompt length) | flash | **76.0 s** | 30.5, 4.9, 4.7, 4.7, 4.8, 4.7 | 16.9 s |
| `rajasthan_5s_sdpa_samevm` (v4 method) | SDPA | **376.6 s** | 200.9, 13.8, 14.0, 13.9, 13.9, 14.0 | 61.6 s |

- **First clip on a fresh VM: 377 s -> 142 s (2.7x).**
- **Later clips in the same batch: 51-77 s (4.9-7.4x faster than v4).** A later clip recompiles step 1 (~30 s) only
  when its prompt has a different token count than every earlier clip, because the sequence length changes.
- Video VAE: compiles once per process (60 s), then 17 s.
- **10-minute video estimate** (117 clips of 5 s, one TPU, one batch): ~117 x 51-77 s = **1.7-2.5 h**, vs ~4.7 h
  with v4 at its best and ~12 h with the first v4 run. On-demand v6e-4 in europe-west4 costs $11.88/h
  ($1.782/chip-h on Spot).

## Commands

```bash
cd /data/pika/v5-flash-multiclip
export HF_TOKEN=hf_xxx

# 1. Create the TPU, install, download the model (~7 min). SPOT=0 for on-demand, SPOT=1 (default) for Spot.
SPOT=0 commands/1_spin_up_spot_tpu.sh

# 2a. One video
commands/2_generate_video.sh prompts/prompt_rajasthan_5s.txt rajasthan_5s
# 2b. Many videos in one process (recommended for long videos): one line per clip
#     <prompt.txt> <name> [duration] [first_frame.png]
commands/2_generate_batch.sh prompts/test_jobs.txt

# 3. Copy results to /data/pika/v5-flash-multiclip/outputs
commands/3_save_to_data_pika.sh

# 4. Delete the TPU
commands/4_tear_down_tpu.sh
```

To use another zone, prefix every step with the same `ZONE=...`, e.g. `ZONE=us-east5-b SPOT=0 commands/1_spin_up_spot_tpu.sh`.

**When no zone has capacity** (common for v6e-4), queue a request; Google creates the TPU when capacity frees up
(today's on-demand TPU arrived after ~30 min). Then run step 1 as usual, it reuses the existing node:

```bash
gcloud compute tpus queued-resources create pika-v5-qr --node-id=pika-v5-tpu --zone=europe-west4-a \
  --project=genai25th01 --accelerator-type=v6e-4 --runtime-version=v2-alpha-tpuv6e --valid-until-duration=8h  # add --spot for Spot
gcloud compute tpus queued-resources describe pika-v5-qr --zone=europe-west4-a --project=genai25th01 --format='value(state.state)'
# when ACTIVE:  SPOT=0 commands/1_spin_up_spot_tpu.sh   ...   and to finish:
gcloud compute tpus queued-resources delete pika-v5-qr --zone=europe-west4-a --project=genai25th01 --quiet --force
```

**Extending a clip in batch mode:** a line with a first frame needs that frame locally before the batch starts. For a
chain of extensions (clip 2 starts from the last frame of clip 1), run one batch per link, or use
`2_generate_video.sh` + `3_save_to_data_pika.sh` per clip (step 3 saves `<name>_last.png`).

## How the speedups work

### Profile (v4 pipeline, one denoising step = 14.5 s) — `scripts/profile_step.py`, `logs/profile_step.json`

| Measurement | Value |
| --- | --- |
| One transformer block on its chip | 0.257 s device time |
| of which attention | 0.238 s (**~92%**) |
| of which feed-forward | 0.014 s |
| Python tracing per block | 0.006 s (negligible) |
| Exact SDPA, larger head chunks | 8 heads: no faster; 14 heads: out of HBM |

Attention dominated because SDPA writes a `heads x 15004 x 15004` float32 score matrix to HBM, 14 times per block.

### 1. Pallas flash attention — `FlashAttnProcessor` in `scripts/generate_pika_tpu.py`

The Pallas kernel (`torch_xla.experimental.custom_kernel.flash_attention`) computes the same softmax attention in
tiles inside the chip and never writes the score matrix. The 15,004-token sequence is padded to a multiple of 512
(15,360) and the padding is masked with segment ids (real tokens = 1, padding = 2), so real tokens never attend to
padding; padded rows are dropped. Block sizes `block_q=2048, block_k_major=1024, block_k=512` were the fastest of the
four tested (`scripts/test_flash_tpu.py`):

| Attention on (1, 56, 15004, 128) bf16 | Time per block | vs current | Max / mean abs difference |
| --- | ---: | ---: | --- |
| SDPA, 4 heads at a time (v4) | 63-76 ms | 1x | - |
| Flash 512/512 | 16.1 ms | 3.9x | 7.3e-4 / 3.6e-5 |
| Flash 2048/1024 (used) | 12.9 ms | 4.9-5.9x | 4.9e-4 / 1.7e-5 |

**One required fix.** torch_xla builds Pallas kernels through jax without the TPU attached, so jax cannot see the
libtpu version and writes the kernel in Mosaic format v8. libtpu 0.0.21 (bundled with torch_xla 2.9) reads up to v7
and fails with `Unsupported version: expected <= 7 but got 8`. `enable_pallas()` forces format 7.

### 2. Batch mode — `--jobs` in `scripts/generate_pika_tpu.py`

The torch_xla persistent compile cache (`XLA_PERSISTENT_CACHE_PATH`) wrote its files but a new process on the same VM
still recompiled (step 1: 31 s, VAE: 62 s), matching torch_xla issue
[#9094](https://github.com/pytorch/xla/issues/9094). Keeping one process alive for all clips avoids that: compiled
programs stay in memory, so only the first clip of each shape compiles.

### Quality

Flash attention is the same formula computed in a different order, so results differ only by bf16 rounding (max
5e-4 per attention call). Pika's sampler adds fresh noise every step, so rounding differences can still change small
details of the final video (Pika notes outputs are not bit-exact across devices or processes either).

Measured (`frames/sdpa_vs_flash_samevm.png`: top row SDPA, bottom row flash; same prompt, seed, VM):

| Comparison | Result |
| --- | --- |
| SDPA on two different VMs (v4 run vs this run) | **byte-identical MP4** |
| Flash on two different VMs (18:40 run vs `rajasthan_5s_v5a`) | **byte-identical MP4** |
| Flash, same prompt twice in one process (`v5a` vs `v5b`) | SSIM 0.985, PSNR 44.3 dB (visually identical) |
| Flash vs SDPA | SSIM 0.860, PSNR 27.5 dB |

Both methods are deterministic, so the flash-vs-SDPA difference comes from flash attention itself. The flash video is
the same scene (desert, sun, three decorated camels, man in red turban, same walk) with slightly different
framing and leg positions: a different, equally sharp sample, not a degraded one. If a bit-exact match with v4
outputs matters, run with `--attention sdpa` (v4 speed).

## Possible next step

Prompts of different lengths recompile step 1 (~30 s). Padding every prompt to a fixed token count and masking the
padding with the same segment-id mechanism would make every clip after the first ~51 s. Not implemented.

## Not done (and why)

- **Using all 4 chips at the same time** (tensor parallelism / pipelining clips): not needed for this project's goal
  of long videos, per the owner's decision. Estimated another ~2.5-3.5x per step if needed later.
- **Fewer steps / int8 / lower resolution**: would change quality.
- **Persistent compile cache across VMs**: does not work in torch_xla 2.9 (above).
- PyTorch/XLA is deprecated (repository to be archived 2026-11-15) in favour of TorchTPU. The pinned versions here keep
  working; future speed work should target TorchTPU.
