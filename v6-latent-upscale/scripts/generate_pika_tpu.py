#!/usr/bin/env python3
"""Real Pika H3 Flex text-to-video+audio (and first-frame extension) on a Cloud TPU v6e-4.

This runs the *official* diffusers Modular pipeline (`PikaH3FlexModularPipeline`, diffusers commit
d0b9b1b5) with the official weights and FL2VA adapter. Nothing in the math is replaced or simulated.
Only device placement changes, because the 66 GB transformer does not fit one 32 GB TPU chip:

  * text encoder (Qwen3-VL, 67 GB) ...... host CPU, bf16 (runs once per video, ~seconds)
  * transformer (14B DiT, 50 blocks) ..... split across the 4 TPU chips by block (~13 blocks per chip)
  * video VAE decoder (9.7 GB) ........... last TPU chip, float32 at "highest" matmul precision
  * audio VAE + text encoder ............. host CPU (small / run once)

Attention is computed with exact scaled-dot-product attention, a few heads at a time, so the
(heads x seq x seq) score matrix fits in TPU memory. Splitting over heads does not change the result.

Usage (on the TPU VM):
  python generate_pika_tpu.py --prompt-file prompt.txt --duration 5 --output /dev/shm/pika/outputs/clip.mp4
  python generate_pika_tpu.py --prompt-file ext.txt --image last_frame.png --duration 5 --output ext.mp4
"""
import argparse
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = {"480p": (832, 480), "768p": (1344, 768)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="/dev/shm/pika/model")
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--jobs", type=Path, help="File with one job per line: <prompt_file> <output.mp4> [image|-] [duration]")
    p.add_argument("--image", type=Path, help="First frame (fl2va). Omit for pure text-to-video (t2va).")
    p.add_argument("--resolution", choices=BUCKETS, default="480p")
    p.add_argument("--duration", type=int, choices=range(5, 16), default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention", choices=["flash", "sdpa"], default="flash",
                   help="flash: Pallas flash-attention kernel (default, ~5x faster attention); sdpa: exact SDPA in head chunks")
    p.add_argument("--head-chunk", type=int, default=4, help="Attention heads computed at once with --attention sdpa")
    p.add_argument("--num-chips", type=int, default=4)
    p.add_argument("--upscale", type=float, default=0, help="2 = latent-upscale 2x and refine (832x480 -> 1664x960)")
    p.add_argument("--refine-from", type=int, default=250, help="refine starts at this timestep of Pika's grid")
    p.add_argument("--keep-base", action="store_true", help="also save the 480p clip as <name>_base.mp4")
    p.add_argument("--upscaler", default="/dev/shm/pika/upscaler/model.safetensors")
    p.add_argument("--seq-bucket", type=int, default=1024,
                   help="pad the packed sequence to a multiple of this (one compile for all prompt lengths); 0 = off")
    p.add_argument("--output", type=Path)
    return p.parse_args()


# --------------------------------------------------------------------------------------------
# Exact attention, sliced over heads (same math as MiniMaxH3AttnProcessor + SDPA)
# --------------------------------------------------------------------------------------------
class HeadChunkedAttnProcessor:
    def __init__(self, head_chunk):
        self.head_chunk = head_chunk

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        # (B, S, H, D) -> (B, H, S, D)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        outs = [
            F.scaled_dot_product_attention(
                q[:, s : s + self.head_chunk], k[:, s : s + self.head_chunk], v[:, s : s + self.head_chunk],
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
            for s in range(0, q.shape[1], self.head_chunk)
        ]
        out = torch.cat(outs, dim=1).transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


class FlashAttnProcessor(HeadChunkedAttnProcessor):
    """Same attention math, computed with the Pallas flash-attention kernel (torch_xla custom_kernel).

    The kernel never writes the (heads x seq x seq) score matrix to memory. The sequence is padded to a multiple of
    512 and the padding is masked with segment ids (real tokens = 1, padding = 2), so real tokens never attend to
    padding and the padded rows are dropped afterwards.
    """

    def __init__(self):
        super().__init__(head_chunk=0)
        self._seg = {}
        self.real_len = None  # set per transformer call when the sequence is padded to a fixed bucket

    def _segments(self, seq, pad, real, device):
        key = (seq, pad, real, str(device))
        if key not in self._seg:
            seg = torch.cat([torch.ones(real, dtype=torch.int32),
                             torch.full((seq + pad - real,), 2, dtype=torch.int32)])[None]
            self._seg[key] = seg.to(device)
        return self._seg[key]

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        from torch_xla.experimental.custom_kernel import flash_attention

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))  # (B, H, S, D)
        seq, dim = q.shape[2], q.shape[3]
        pad = (-seq) % 512
        real = self.real_len if self.real_len is not None else seq
        seg = self._segments(seq, pad, real, q.device)
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        out = flash_attention(q, k, v, causal=False, q_segment_ids=seg, kv_segment_ids=seg,
                              sm_scale=1.0 / math.sqrt(dim))[:, :, :seq]
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


def enable_pallas():
    """torch_xla lowers Pallas kernels through jax without the TPU attached, so jax cannot see the libtpu version and
    serializes Mosaic IR at its newest format (v8). libtpu 0.0.21 (bundled with torch_xla 2.9) reads <= v7."""
    import jax._src.tpu_custom_call as tcc
    tcc.get_ir_version = lambda ctx: 7
    from torch_xla.experimental.custom_kernel import FlashAttention
    # Fastest of the block sizes measured on v6e for seq 15004, 56 heads, head_dim 128 (12.9 ms vs 63-76 ms SDPA).
    FlashAttention.DEFAULT_BLOCK_SIZES.update(block_q=2048, block_k_major=1024, block_k=512)


# --------------------------------------------------------------------------------------------
# Place the transformer on the TPU chips
# --------------------------------------------------------------------------------------------
def _move(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj


class TransformerOnTPU:
    """Splits the 50 blocks across chips, moves activations between chips, records step timings."""

    HEAD_MODULES = ["proj_in", "audio_proj_in", "context_embedder", "token_refiner", "time_proj",
                    "time_embedder", "rope", "norm_out", "proj_out", "audio_proj_out"]

    def __init__(self, transformer, num_chips, head_chunk, attention="flash", seq_bucket=0):
        import torch_xla
        import torch_xla.core.xla_model as xm

        self.torch_xla = torch_xla
        self.xm = xm
        devices = [torch.device(d) for d in xm.get_xla_supported_devices()][:num_chips]
        log(f"TPU devices: {[str(d) for d in devices]}")
        self.devices = devices
        self.head = devices[0]
        self.step_times = []

        t0 = time.monotonic()
        for name in self.HEAD_MODULES:
            getattr(transformer, name).to(self.head)
        blocks = transformer.transformer_blocks
        # Blocks per chip. The last chip also holds the 9.7 GB video VAE decoder, so it gets fewer blocks
        # (an extension clip refined at 1664x960 ran out of memory there with an even 13/13/13/11 split).
        counts = [int(c) for c in os.environ.get("PIKA_BLOCKS_PER_CHIP", "14,14,13,9").split(",")]
        if len(counts) != len(devices) or sum(counts) != len(blocks):
            per = math.ceil(len(blocks) / len(devices))
            counts = [min(per, len(blocks) - per * i) for i in range(len(devices))]
        owner = [d for d, c in zip(devices, counts) for _ in range(c)]
        flash_proc = None
        if attention == "flash":
            enable_pallas()
            flash_proc = FlashAttnProcessor()
        self.flash_proc = flash_proc
        self.seq_bucket = seq_bucket if attention == "flash" else 0
        self.layout = {}
        for i, block in enumerate(blocks):
            dev = owner[i]
            block.attn.set_processor(flash_proc if attention == "flash" else HeadChunkedAttnProcessor(head_chunk))
            block.to(dev)
            self.layout.setdefault(str(dev), []).append(i)
            block.register_forward_pre_hook(self._pre_hook(dev), with_kwargs=True)
            block.register_forward_hook(lambda m, a, o: (torch_xla.sync(), o)[1])
        transformer.norm_out.register_forward_pre_hook(self._pre_hook(self.head), with_kwargs=True)
        torch_xla.sync()
        log(f"Transformer weights moved to TPU in {time.monotonic() - t0:.1f}s; blocks per chip: "
            + ", ".join(f"{d}={v[0]}-{v[-1]}" for d, v in self.layout.items()))

        original = transformer.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            t = time.monotonic()
            # Step 1 has one distinct timestep, steps 2-6 have two, which would make XLA compile the whole
            # network twice. Pad `timestep` to 2 entries: every row's timestep_indices still points at entry 0,
            # so the AdaLN rows that are actually read are identical and the output does not change.
            ts = kwargs.get("timestep")
            if ts is not None and ts.ndim == 1 and ts.shape[0] == 1:
                kwargs["timestep"] = ts.repeat(2)
            # Fixed sequence bucket: pad the packed sequence with dummy text rows so every prompt length has the
            # same shapes (one compile). Real rows never attend to the dummy rows (flash segment ids), every other
            # op is per row, and the outputs are read only from the real video/audio rows: results are unchanged.
            if self.seq_bucket:
                pos = kwargs["position_ids"]
                real = pos.shape[0]
                extra = (-real) % self.seq_bucket
                self.flash_proc.real_len = real
                if extra:
                    kwargs["position_ids"] = torch.cat([pos, pos.new_zeros((extra, pos.shape[1]))])
                    kwargs["token_tags"] = torch.cat([kwargs["token_tags"],
                                                      kwargs["token_tags"].new_ones((extra,))])
                    kwargs["timestep_indices"] = torch.cat([kwargs["timestep_indices"],
                                                            kwargs["timestep_indices"].new_zeros((extra,))])
            out = original(*_move(args, self.head), **_move(kwargs, self.head))
            out = _move(out, torch.device("cpu"))  # .to(cpu) forces execution
            dt = time.monotonic() - t
            self.step_times.append(round(dt, 3))
            log(f"  denoise step {len(self.step_times)}: transformer forward {dt:.2f}s")
            return out

        transformer.forward = forward

    @staticmethod
    def _pre_hook(device):
        def hook(module, args, kwargs):
            return _move(args, device), _move(kwargs, device)
        return hook


def timed(obj, attr, bucket, label):
    original = getattr(obj, attr)

    @functools.wraps(original)
    def wrapper(*a, **k):
        t = time.monotonic()
        r = original(*a, **k)
        bucket.setdefault(label, 0.0)
        bucket[label] += time.monotonic() - t
        return r

    setattr(obj, attr, wrapper)


def load_jobs(args):
    """--jobs file, one clip per line:
         <prompt_file> <output.mp4> [first_frame|@name|-] [duration] [key=value ...]
       `@name` = the last frame of clip <name> generated earlier in the same batch (extension chain, all on TPU).
       Options per line: refine_from=500|250|100, seed=N."""
    if args.jobs:
        jobs = []
        for line in args.jobs.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            opts = dict(p.split("=", 1) for p in parts if "=" in p)
            pos = [p for p in parts if "=" not in p]
            img = pos[2] if len(pos) > 2 and pos[2] != "-" else None
            dur = int(pos[3]) if len(pos) > 3 else args.duration
            jobs.append((Path(pos[0]), Path(pos[1]), img, dur, opts))
        return jobs
    if not (args.prompt_file and args.output):
        raise SystemExit("give --prompt-file and --output, or --jobs")
    return [(args.prompt_file, args.output, str(args.image) if args.image else None, args.duration, {})]


def _to_cpu(obj):
    from diffusers.utils import BaseOutput
    if isinstance(obj, BaseOutput):
        for k in list(obj.keys()):
            obj[k] = _to_cpu(obj[k])
        return obj
    return _move(obj, torch.device("cpu"))


def run_on_chip(owner, attr, module, device, label):
    """Run owner.<attr>(...) with `module` on a TPU chip; inputs/outputs stay CPU tensors for the pipeline.
    Falls back to CPU (and says so) only if the chip run fails."""
    orig = getattr(owner, attr)
    module.to(device)
    state = {"tpu": True}

    def call(*a, **k):
        if state["tpu"]:
            try:
                return _to_cpu(orig(*_move(a, device), **_move(k, device)))
            except Exception as e:  # pragma: no cover
                log(f"WARNING: {label} failed on {device} ({str(e)[:160]}); using host CPU for it")
                state["tpu"] = False
                module.to("cpu")
        return orig(*a, **k)

    setattr(owner, attr, call)
    log(f"{label} -> {device}")


def main():
    args = parse_args()
    torch.set_num_threads(os.cpu_count())
    import torch_xla.backends
    import torch_xla.runtime as xr
    xr.initialize_cache(os.environ.get("XLA_CACHE_DIR", "/dev/shm/pika/xla_cache"))
    torch_xla.backends.set_mat_mul_precision("highest")  # affects float32 ops only; bf16 matmuls are unchanged
    from PIL import Image
    import diffusers
    from diffusers import ModularPipeline
    from diffusers.utils.export_utils import encode_video

    jobs = load_jobs(args)
    base_w, base_h = BUCKETS[args.resolution]
    up = args.upscale if args.upscale and args.upscale > 1 else 0
    W2 = int(round(base_w * up / 32) * 32) if up else base_w
    H2 = int(round(base_h * up / 32) * 32) if up else base_h
    log(f"{len(jobs)} job(s); base {base_w}x{base_h}" + (f", upscale {up}x -> {W2}x{H2}" if up else ", no upscale"))

    t_load = time.monotonic()
    pipe = ModularPipeline.from_pretrained(args.snapshot)
    pipe.load_components(
        pretrained_model_name_or_path=args.snapshot, workflow="fl2va",
        dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
    )
    pipe.load_lora_weights(args.snapshot, weight_name="fl2va/adapter.safetensors")
    log(f"Components + FL2VA adapter loaded in {time.monotonic() - t_load:.1f}s")

    # Pipeline bookkeeping and the text encoder (67 GB, ~3 s per clip) run on the TPU VM host CPU; every model
    # below runs on TPU chips.
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))
    tpu = TransformerOnTPU(pipe.transformer, args.num_chips, args.head_chunk, args.attention, args.seq_bucket)
    devs = tpu.devices
    vae_dev = devs[-1]
    pipe.vae.decoder.to(vae_dev)
    dec_fwd = pipe.vae.decoder.forward
    pipe.vae.decoder.forward = lambda x: dec_fwd(x.to(vae_dev)).cpu()
    log(f"video VAE decoder -> {vae_dev}")
    enc_dev = devs[2 % len(devs)]
    pipe.vae.encoder.to(enc_dev)
    enc_fwd = pipe.vae.encoder.forward
    pipe.vae.encoder.forward = lambda x: enc_fwd(x.to(enc_dev)).cpu()
    log(f"video VAE encoder (first-frame conditioning) -> {enc_dev}")
    run_on_chip(pipe.audio_vae, "decode", pipe.audio_vae, enc_dev, "audio VAE")
    upscaler = None
    if up:
        from latent_upscaler import load_upscaler
        upscaler = load_upscaler(args.upscaler).to(devs[1 % len(devs)])
        log(f"latent upscaler -> {devs[1 % len(devs)]}")
    tpu.torch_xla.sync()
    setup_s = time.monotonic() - t_load

    lat_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    a_mean = torch.tensor(pipe.audio_vae.config.latents_mean).view(1, -1, 1)
    a_std = torch.tensor(pipe.audio_vae.config.latents_std).view(1, -1, 1)

    stage = {}
    timed(pipe.text_encoder.model, "forward", stage, "text_encoder_s")
    # Capture final latents; in upscale mode skip the (unused) 480p decode.
    cap, mode = {}, {"skip_decode": False, "frames": 0}
    real_decode = pipe.vae.decode
    def vae_decode(z, *a, **k):
        cap["latents"] = (z.float().cpu() - lat_mean) / lat_std
        if mode["skip_decode"]:
            return (torch.zeros(1, 3, mode["frames"], z.shape[3] * 16, z.shape[4] * 16),)
        t = time.monotonic()
        r = real_decode(z, *a, **k)
        stage["video_vae_decode_s"] = stage.get("video_vae_decode_s", 0.0) + time.monotonic() - t
        return r
    pipe.vae.decode = vae_decode
    real_audio_decode = pipe.audio_vae.decode
    def audio_decode(z, *a, **k):
        cap["audio"] = (z.float().cpu() - a_mean) / a_std
        t = time.monotonic()
        r = real_audio_decode(z, *a, **k)
        stage["audio_vae_decode_s"] = stage.get("audio_vae_decode_s", 0.0) + time.monotonic() - t
        return r
    pipe.audio_vae.decode = audio_decode

    # Refine schedule: Pika's own sigma grid from `refine_from` on (same model, adapter and sampler).
    try:
        steps_grid = list(pipe.config.denoising_step_list)
    except Exception:
        steps_grid = [1000, 750, 500, 250, 100, 40, 0]
    refine = {"start": None}
    for sch in (pipe.scheduler, pipe.audio_scheduler):
        orig = sch.set_timesteps
        def wrapped(*a, _orig=orig, **k):
            if refine["start"] is not None:
                k["sigmas"] = k["sigmas"][refine["start"]:]
            return _orig(*a, **k)
        sch.set_timesteps = wrapped
    shift_v, shift_a = pipe.scheduler.config.shift, pipe.audio_scheduler.config.shift
    shifted = lambda sg, sh: sh * sg / (1 + (sh - 1) * sg)

    outdir = None
    for n, (prompt_file, output, image, duration, opts) in enumerate(jobs, 1):
        outdir = output.parent
        outdir.mkdir(parents=True, exist_ok=True)
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        seed = int(opts.get("seed", args.seed))
        refine_from = int(opts.get("refine_from", args.refine_from))
        expected_frames = duration * 24
        while expected_frames % 17 != 5:
            expected_frames += 1
        img_path = None
        if image:
            img_path = outdir / f"{image[1:]}_last.png" if image.startswith("@") else Path(image)
        workflow = "fl2va" if img_path else "t2va"
        log(f"Job {n}/{len(jobs)}: {output.name} {workflow} {duration}s seed={seed}"
            + (f" first_frame={img_path.name}" if img_path else "")
            + (f" upscale->{W2}x{H2} refine_from={refine_from}" if up else ""))
        stage.clear()
        rec = {"job_index": n, "workflow": workflow, "prompt": prompt, "image": str(img_path) if img_path else None,
               "seed": seed, "duration_s": duration, "frames": expected_frames, "base": f"{base_w}x{base_h}",
               "output": f"{W2}x{H2}", "upscale": up, "refine_from": refine_from if up else None,
               "process_setup_s": round(setup_s, 2) if n == 1 else 0.0}
        t_job = time.monotonic()

        def call(height, width, extra, gen_seed):
            kw = dict(prompt=prompt, height=height, width=width, num_frames=duration * 24,
                      generator=torch.Generator("cpu").manual_seed(gen_seed),
                      output=["videos", "audio", "sampling_rate"], **extra)
            if img_path:
                kw["image"] = Image.open(img_path).convert("RGB")
            return pipe(**kw)

        # ---- stage 1: base resolution ----
        mode.update(skip_decode=bool(up) and not args.keep_base, frames=expected_frames)
        tpu.step_times = []
        t = time.monotonic()
        res = call(base_h, base_w, {}, seed)
        rec["stage1_s"] = round(time.monotonic() - t, 2)
        rec["stage1_steps_s"] = tpu.step_times
        audio = res["audio"][0].detach().float().cpu()
        sr = res["sampling_rate"]
        frames = res["videos"][0]
        if up and args.keep_base:
            encode_video(frames, fps=24, output_path=str(output.with_name(output.stem + "_base.mp4")),
                         audio=audio, audio_sample_rate=sr)

        if up:
            lat, aud = cap["latents"], cap["audio"]
            T = lat.shape[2]
            # ---- latent upscale (TPU). The upscaler expects Pika's normalized latents normalized once more ----
            t = time.monotonic()
            x_in = ((lat - lat_mean) / lat_std).to(upscaler.conv_in.weight.device, torch.bfloat16)
            up_lat = upscaler(x_in, float(up), (T, H2 // 16, W2 // 16)).float().cpu() * lat_std + lat_mean
            rec["upscale_s"] = round(time.monotonic() - t, 2)
            # ---- refine at the upscaled size with Pika's last steps (TPU) ----
            start = steps_grid.index(refine_from)
            sg = steps_grid[start] / 1000.0
            g = torch.Generator("cpu").manual_seed(seed + 1)
            sv, sa = shifted(sg, shift_v), shifted(sg, shift_a)
            x_v = (1 - sv) * up_lat + sv * torch.randn(up_lat.shape, generator=g)
            x_a = (1 - sa) * aud + sa * torch.randn(aud.shape, generator=g)
            mode["skip_decode"] = False
            refine["start"] = start
            tpu.step_times = []
            t = time.monotonic()
            try:
                res = call(H2, W2, {"latents": x_v, "audio_latents": x_a}, seed + 2)
            finally:
                refine["start"] = None
            rec["refine_s"] = round(time.monotonic() - t, 2)
            rec["refine_steps_s"] = tpu.step_times
            frames = res["videos"][0]   # keep the stage-1 soundtrack

        assert len(frames) == expected_frames, (len(frames), expected_frames)
        assert frames[0].size == (W2, H2), frames[0].size
        assert torch.isfinite(audio).all(), "audio has NaN/Inf"
        t = time.monotonic()
        encode_video(frames, fps=24, output_path=str(output), audio=audio, audio_sample_rate=sr)
        frames[-1].save(output.with_name(output.stem + "_last.png"))
        rec["encode_mp4_s"] = round(time.monotonic() - t, 2)
        rec.update({k: round(v, 2) for k, v in stage.items()})
        rec["job_s"] = round(time.monotonic() - t_job, 2)
        rec["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix(".json").write_text(json.dumps(rec, indent=2) + "\n")
        log("TIMINGS " + json.dumps({k: v for k, v in rec.items() if k != "prompt"}))
        log(f"DONE {output}")


if __name__ == "__main__":
    sys.exit(main())
