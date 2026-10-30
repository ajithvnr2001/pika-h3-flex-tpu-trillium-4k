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

    def _segments(self, seq, pad, device):
        key = (seq, pad, str(device))
        if key not in self._seg:
            seg = torch.cat([torch.ones(seq, dtype=torch.int32), torch.full((pad,), 2, dtype=torch.int32)])[None]
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
        seg = self._segments(seq, pad, q.device)
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

    def __init__(self, transformer, num_chips, head_chunk, attention="flash"):
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
        per_chip = math.ceil(len(blocks) / len(devices))
        if attention == "flash":
            enable_pallas()
            flash_proc = FlashAttnProcessor()
        self.layout = {}
        for i, block in enumerate(blocks):
            dev = devices[min(i // per_chip, len(devices) - 1)]
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
    """One job per line of --jobs: `<prompt_file> <output.mp4> [first_frame.png|-] [duration]`, or a single job
    from --prompt-file/--output. All jobs run in one process, so compiled TPU programs are reused."""
    if args.jobs:
        jobs = []
        for line in args.jobs.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            img = Path(parts[2]) if len(parts) > 2 and parts[2] != "-" else None
            dur = int(parts[3]) if len(parts) > 3 else args.duration
            jobs.append((Path(parts[0]), Path(parts[1]), img, dur))
        return jobs
    if not (args.prompt_file and args.output):
        raise SystemExit("give --prompt-file and --output, or --jobs")
    return [(args.prompt_file, args.output, args.image, args.duration)]


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
    width, height = BUCKETS[args.resolution]
    t_load = time.monotonic()
    pipe = ModularPipeline.from_pretrained(args.snapshot)
    pipe.load_components(
        pretrained_model_name_or_path=args.snapshot, workflow="fl2va",
        dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
    )
    pipe.load_lora_weights(args.snapshot, weight_name="fl2va/adapter.safetensors")
    load_components_s = time.monotonic() - t_load
    log(f"Components + FL2VA adapter loaded on host in {load_components_s:.1f}s")

    # Pipeline bookkeeping, the text encoder (runs once, ~5 s) and the small audio VAE stay on the host CPU.
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))
    t = time.monotonic()
    tpu = TransformerOnTPU(pipe.transformer, args.num_chips, args.head_chunk, args.attention)
    # The 9.7 GB float32 ViT video-VAE decoder goes on the last chip, which holds the fewest transformer blocks.
    # Matmul precision is 'highest' (true fp32) so TPU output matches the CPU decoder (max abs diff ~1e-5).
    vae_dev = tpu.devices[-1]
    pipe.vae.decoder.to(vae_dev)
    original_decoder = pipe.vae.decoder.forward
    pipe.vae.decoder.forward = lambda x: original_decoder(x.to(vae_dev)).cpu()
    tpu.torch_xla.sync()
    to_tpu_s = time.monotonic() - t

    stage = {}
    timed(pipe.text_encoder.model, "forward", stage, "text_encoder_s")
    timed(pipe.vae, "decode", stage, "video_vae_decode_s")
    timed(pipe.audio_vae, "decode", stage, "audio_vae_decode_s")
    setup_s = time.monotonic() - t_load

    for n, (prompt_file, output, image, duration) in enumerate(jobs, 1):
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        expected_frames = duration * 24
        while expected_frames % 17 != 5:
            expected_frames += 1
        workflow = "fl2va" if image else "t2va"
        log(f"Job {n}/{len(jobs)}: {output.name} workflow={workflow} {width}x{height} {duration}s "
            f"frames={expected_frames} seed={args.seed}")
        stage.clear()
        tpu.step_times = []
        timings = {"job_index": n, "process_setup_s": round(setup_s, 2) if n == 1 else 0.0,
                   "load_components_s": round(load_components_s, 2) if n == 1 else 0.0,
                   "transformer_to_tpu_s": round(to_tpu_s, 2) if n == 1 else 0.0}
        t_job = time.monotonic()
        call = dict(prompt=prompt, height=height, width=width, num_frames=duration * 24,
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                    output=["videos", "audio", "sampling_rate"])
        if image:
            call["image"] = Image.open(image).convert("RGB")
        log("Generating ...")
        t = time.monotonic()
        result = pipe(**call)
        timings["generation_s"] = time.monotonic() - t
        timings.update({k: round(v, 2) for k, v in stage.items()})
        timings["denoise_step_s"] = tpu.step_times
        timings["denoise_total_s"] = round(sum(tpu.step_times), 2)

        frames = result["videos"][0]
        audio = result["audio"][0].detach().float().cpu()
        assert len(frames) == expected_frames, (len(frames), expected_frames)
        assert frames[0].size == (width, height), frames[0].size
        assert torch.isfinite(audio).all(), "audio has NaN/Inf"

        output.parent.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        encode_video(frames, fps=24, output_path=str(output), audio=audio,
                     audio_sample_rate=result["sampling_rate"])
        timings["encode_mp4_s"] = round(time.monotonic() - t, 2)
        timings["job_s"] = round(time.monotonic() - t_job, 2)
        timings["end_to_end_s"] = round(timings["job_s"] + (setup_s if n == 1 else 0.0), 2)

        probe = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)], text=True))
        record = {
            "workflow": workflow, "prompt": prompt, "image": str(image) if image else None,
            "seed": args.seed, "resolution": args.resolution, "width": width, "height": height,
            "requested_duration_s": duration, "frames": expected_frames, "fps": 24,
            "audio_sample_rate": result["sampling_rate"],
            "hardware": f"Cloud TPU v6e-4 ({len(tpu.devices)} chips) + host CPU ({os.cpu_count()} vCPU)",
            "blocks_per_chip": tpu.layout, "attention": args.attention, "head_chunk": args.head_chunk,
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "timings_s": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()},
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "media": probe,
        }
        output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        log("TIMINGS " + json.dumps(record["timings_s"]))
        log(f"DONE {output}")


if __name__ == "__main__":
    sys.exit(main())
