# Provisioning a Spot TPU v6e-4 and running the FreeVideo H3 render

Everything below is the exact sequence used on `trillium-opt-vm`.
Project `genai25th01`, zone `europe-west4-a`, accelerator `v6e-4` (2x2 topology, 4 chips).

---

## 1. Provision the Spot TPU

```bash
gcloud compute tpus tpu-vm create trillium-opt-vm \
  --zone=europe-west4-a \
  --project=genai25th01 \
  --accelerator-type=v6e-4 \
  --version=tpu-ubuntu2204-base \
  --spot \
  --network=default \
  --subnetwork=default \
  --tags=tpu \
  --quiet
```

Flags that matter:

- `--spot` makes this a Spot / Queued Resource. It is **not** a normal Compute instance, so
  `gcloud compute instances list` shows **nothing**. Use `gcloud compute tpus tpu-vm list`.
- `--accelerator-type=v6e-4` = 4 chips, 2x2 topology.
- `--version=tpu-ubuntu2204-base` matches the setup below.
- Drop `--network/--subnetwork` to let GCP pick defaults.

Failure modes are different problems:

- `no more capacity` -> the zone is full. Wait, or try another zone.
- `Quota limit 'TPUV6EPerProjectPerZoneForTPUAPI' ... Limit: 0` -> quota, not capacity. Needs a quota request.

## 2. Get the IP and SSH in

```bash
gcloud compute tpus tpu-vm describe trillium-opt-vm \
  --zone=europe-west4-a --project=genai25th01 \
  --format='value(networkEndpoints[0].ipAddress)'
```

```bash
gcloud compute tpus tpu-vm ssh trillium-opt-vm \
  --zone=europe-west4-a --project=genai25th01
```

Once the key is installed, direct SSH is much faster than the `gcloud` wrapper:

```bash
ssh root@<IP>
```

## 3. Verify the chips came up

```bash
ls -l /dev/vfio/
```

Expect `/dev/vfio/0` through `/dev/vfio/3` for a 2x2. If any are missing, TPU init will fail later
with `open(/dev/vfio/N): Device or resource busy`.

## 4. Rebuild the environment

```bash
sudo apt-get update -qq
sudo apt-get install -y python3-venv git
python3 -m venv /dev/shm/venv
source /dev/shm/venv/bin/activate

pip install --upgrade pip
pip install torch==2.6.0 torch_xla[tpu]==2.6.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```

Then the JAX pins that the Pallas flash-attention kernel needs:

```bash
pip install jax==0.4.38 jaxlib==0.4.38
pip uninstall -y libtpu-nightly
```

**Do not skip either line.**

- Without `jax`/`jaxlib`, `torch_xla.experimental.custom_kernel.flash_attention` is unavailable and
  silently falls back to eager SDPA, which OOMs.
- `libtpu-nightly` conflicts with the stable `libtpu` and crashes TPU init. Keep only the stable one.

`/dev/shm` is deliberate: it is RAM-backed, which is what the original setup used.

## 5. Clone the work

```bash
git clone https://github.com/<your-account>/trillium-video-studio.git /dev/shm/trillium-video-studio
cd /dev/shm/trillium-video-studio
git log --oneline -3
```

Expect `d9d7920` (fix(tpu): make the window/dense attention schedule fit in v6e-4 HBM) among the
top commits. That is the commit containing the flash-attention and memory work.

## 6. Download the weights (fresh VM only)

Weights are **not** in the repo. `--root-dir` expects them already present.

```bash
python3 -m freevideo_engine.tpu_run_10s_20s \
  --root-dir /dev/shm/runtime_data \
  --out-dir /dev/shm/outputs/tpu_5s \
  --library-dir /dev/shm/trillium-video-studio/FreeVideo/vendor/h3-text-encoder \
  --budget-gib 120 --resident-gib 110 --spmd --download-only
```

This pulls from Hugging Face and needs `HF_TOKEN` exported.

## 7. Run the render

Common environment:

```bash
cd /dev/shm/trillium-video-studio/FreeVideo
source /dev/shm/venv/bin/activate

export PJRT_DEVICE=TPU
export FREEVIDEO_TPU_SPMD=1
export FREEVIDEO_VDN_ROOT=/dev/shm/trillium-video-studio/FreeVideo/vendor/vdn
export FREEVIDEO_COMFY_ROOT=/dev/shm/trillium-video-studio/FreeVideo/vendor/h3-text-encoder
export PYTHONPATH=/dev/shm/trillium-video-studio/FreeVideo:/dev/shm/trillium-video-studio/FreeVideo/py_deps:/dev/shm/trillium-video-studio/FreeVideo/vendor/vdn:/dev/shm/trillium-video-studio/FreeVideo/vendor/vdn/diffusers/src:/dev/shm/trillium-video-studio/FreeVideo/vendor/h3-text-encoder
export FREEVIDEO_XLA_CACHE_DIR=/dev/shm/xla_compile_cache
```

### 7a. Small clip (768x512, 39 frames) -- the one that got furthest

```bash
python3 -m freevideo_engine.tpu_run_10s_20s \
  --root-dir /dev/shm/runtime_data \
  --out-dir /dev/shm/outputs/tpu_small \
  --library-dir /dev/shm/trillium-video-studio/FreeVideo/vendor/h3-text-encoder \
  --width 768 --height 512 \
  --clip1-frames 39 \
  --budget-gib 120 --resident-gib 110 \
  --skip-continuation --spmd
```

Reached **block 44 of 50** on step 1 of 11 before being stopped. It ran at roughly 686 s/step.

### 7b. Full 5s clip (1344x768, 123 frames)

```bash
python3 -m freevideo_engine.tpu_run_10s_20s \
  --root-dir /dev/shm/runtime_data \
  --out-dir /dev/shm/outputs/tpu_5s \
  --library-dir /dev/shm/trillium-video-studio/FreeVideo/vendor/h3-text-encoder \
  --clip1-frames 123 \
  --budget-gib 120 --resident-gib 110 \
  --skip-continuation --spmd
```

### 7c. 10s: 5s + a second 5s continuation

Drop `--skip-continuation` and pass the joined length:

```bash
python3 -m freevideo_engine.tpu_run_10s_20s \
  --root-dir /dev/shm/runtime_data \
  --out-dir /dev/shm/outputs/tpu_10s \
  --library-dir /dev/shm/trillium-video-studio/FreeVideo/vendor/h3-text-encoder \
  --clip1-frames 123 \
  --clip2-frames 123 \
  --joined-frames 241 \
  --budget-gib 120 --resident-gib 110 \
  --spmd
```

## 8. Tear it down

```bash
gcloud compute tpus tpu-vm delete trillium-opt-vm \
  --zone=europe-west4-a --project=genai25th01 --quiet
```

Verify nothing is left billing:

```bash
gcloud compute tpus tpu-vm list --zone=europe-west4-a --project=genai25th01
gcloud compute instances list --project=genai25th01
```

Both must list 0 items. Also check for stray queued resources:

```bash
gcloud compute tpus queued-resources list --zone=europe-west4-a --project=genai25th01
```

---

## Known-broken: the full 5s geometry

`--clip1-frames 123` (37,296 tokens) still aborts in the GSPMD partitioner:

```
F spmd_partitioner_util.h:112] Check failed:
  ShapeUtil::IsScalarWithElementType(value.shape(), primitive_type)
```

This fires after block 1 of the first refinement step. It is a **compile-time** abort, not an OOM,
and it does not reproduce in isolated tests of `WindowAttention` at the same geometry (which passes
in 24.5 s). The trigger is something in the real `MiniMaxH3TransformerBlock` that the isolated tests
do not exercise -- most likely the linear-branch state bank write
(`torch.baddbmm(..., out=prefix[frame])` in `vendor/vdn/src/models/linear_attention/scan.py`) combined
with the sharded head layout.

**Start with 7a.** It is the configuration that actually progressed.

## Fixed and verified (in `d9d7920`)

- `batched()` normalises both 3-D `(S,H,D)` and 4-D `(B,S,H,D)` operands to `(B,H,S,D)`. The model
  forwards a 3-D document; the old unconditional transpose turned it into 5-D and asked the kernel
  for an S x S block.
- Anchor rows are query-chunked (`attention_chunk` 65536 -> 1024). At 2016 rows in one call the
  kernel demanded a 15.72 G reservation against ~10.5 G free.
- The dense (anchor) leg executes before the window batches. Compiled as one program their peak
  allocations added up to 60.52 G against 31.25 G of HBM.
- Pallas flash attention is used at the 5s geometry instead of eager SDPA: measured 24.5 s for the
  full `WindowAttention` call, no OOM, no fallback. Previously this allocation was impossible.

## Secrets

The GitHub PAT and HF token used during this work are **unrotated** and should be rotated before
being placed in any script or on a fresh VM.
