#!/usr/bin/env python3
"""Profile one denoising step of Pika H3 Flex on TPU v6e-4 and test attention alternatives.

1. Runs the real pipeline (prompt -> text encoder -> layout) until the first transformer call, and captures the
   real transformer inputs and the real inputs of block 0. Generation is then stopped.
2. Times the full 50-block transformer forward as it runs in generate_pika_tpu.py (cold, then warm).
3. Splits one block on xla:0 into Python tracing time vs device time, attention vs feed-forward.
4. Benchmarks attention variants on the real q/k/v of block 0 and checks they match the current one:
   SDPA in head chunks of 4 (current), 8, 14, and the Pallas flash-attention kernel (torch_xla custom_kernel).
Writes a JSON report next to --output.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
import torch_xla.backends

xr.initialize_cache(os.environ.get("XLA_CACHE_DIR", "/dev/shm/pika/xla_cache"))
torch_xla.backends.set_mat_mul_precision("highest")
from generate_pika_tpu import TransformerOnTPU, HeadChunkedAttnProcessor, BUCKETS, log  # noqa: E402

REPORT = {}


class Captured(Exception):
    pass


def wait():
    torch_xla.sync()
    xm.wait_device_ops()


def bench(fn, warm=2, reps=3):
    """Returns (first_call_s, [trace_s, device_s] averaged over warm reps)."""
    t = time.monotonic(); out = fn(); wait(); first = time.monotonic() - t
    for _ in range(warm - 1):
        fn(); wait()
    tr, dv = [], []
    for _ in range(reps):
        t0 = time.monotonic(); out = fn(); t1 = time.monotonic(); wait(); t2 = time.monotonic()
        tr.append(t1 - t0); dv.append(t2 - t1)
    return out, round(first, 3), round(sum(tr) / reps, 4), round(sum(dv) / reps, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="/dev/shm/pika/model")
    p.add_argument("--prompt-file", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count())
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(args.snapshot)
    pipe.load_components(pretrained_model_name_or_path=args.snapshot, workflow="fl2va",
                         dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
    pipe.load_lora_weights(args.snapshot, weight_name="fl2va/adapter.safetensors")
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))
    tr = pipe.transformer
    tpu = TransformerOnTPU(tr, 4, 4)
    wrapped_forward = tr.forward  # TransformerOnTPU's wrapper (moves inputs, times, returns CPU tensors)

    cap = {}
    def tr_hook(module, a, kw):
        cap["transformer_kwargs"] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kw.items()}
    def blk_hook(module, a, kw):
        cap["block_args"] = [x if not torch.is_tensor(x) else x for x in a]
        raise Captured()
    h1 = tr.register_forward_pre_hook(tr_hook, with_kwargs=True)
    h2 = tr.transformer_blocks[0].register_forward_pre_hook(blk_hook, with_kwargs=True)
    prompt = args.prompt_file.read_text().strip()
    w, h = BUCKETS["480p"]
    t = time.monotonic()
    try:
        pipe(prompt=prompt, height=h, width=w, num_frames=120, generator=torch.Generator("cpu").manual_seed(42),
             output=["videos"])
    except Captured:
        pass
    h1.remove(); h2.remove()
    REPORT["pre_transformer_s"] = round(time.monotonic() - t, 2)
    kw = cap["transformer_kwargs"]
    seq = kw["position_ids"].shape[0]
    REPORT["seq_len"] = seq
    REPORT["timestep_shape"] = list(kw["timestep"].shape)
    log(f"Captured inputs: seq_len={seq}, video rows={kw['video_indices'].numel()}, "
        f"audio rows={kw['audio_indices'].numel()}, text rows={kw['text_indices'].numel()}")

    # ---- 2. full transformer forward, as in production ----
    ts = []
    for i in range(3):
        t = time.monotonic(); wrapped_forward(**kw); ts.append(round(time.monotonic() - t, 2))
        log(f"full transformer forward run {i}: {ts[-1]}s")
    REPORT["full_forward_s"] = ts
    REPORT["compile_metrics"] = {k: met.metric_data(k)[:2] for k in ("CompileTime", "ExecuteTime") if met.metric_data(k)}

    # Per-block device time inside a warm full forward (sync+wait after every block).
    per_block = []
    def tick(m, a, o):
        xm.wait_device_ops(); per_block.append(time.monotonic()); return o
    hs = [b.register_forward_hook(tick) for b in tr.transformer_blocks]
    t0 = time.monotonic(); wrapped_forward(**kw); total = time.monotonic() - t0
    for hh in hs: hh.remove()
    gaps = [round(b - a, 4) for a, b in zip([t0] + per_block[:-1], per_block)]
    REPORT["per_block_wall_s"] = gaps
    REPORT["instrumented_forward_s"] = round(total, 2)
    log(f"per-block wall (waited): first={gaps[0]} median={sorted(gaps)[len(gaps)//2]} "
        f"chip-boundary blocks 13/26/39={gaps[13]},{gaps[26]},{gaps[39]} sum={sum(gaps):.2f}s")

    # ---- 3. one block on xla:0: tracing vs device; attention vs FF ----
    blk = tr.transformer_blocks[0]
    dev = torch.device("xla:0")
    hsx, temb, idx, rope = cap["block_args"]
    hsx, temb, idx = hsx.to(dev), temb.to(dev), idx.to(dev)
    rope = tuple(r.to(dev) for r in rope)
    wait()
    with torch.no_grad():
        _, f, trc, dv = bench(lambda: blk(hsx, temb, idx, rope))
        REPORT["block"] = {"first_s": f, "trace_s": trc, "device_s": dv}
        log(f"block0: first {f}s, python trace {trc}s, device {dv}s")
        nh = blk.norm1(hsx); wait()
        _, f, trc, dv = bench(lambda: blk.attn(nh, rope))
        REPORT["block_attention"] = {"first_s": f, "trace_s": trc, "device_s": dv}
        log(f"attention (chunk 4): trace {trc}s, device {dv}s")
        _, f, trc, dv = bench(lambda: blk.ff(nh))
        REPORT["block_ff"] = {"first_s": f, "trace_s": trc, "device_s": dv}
        log(f"feed-forward: trace {trc}s, device {dv}s")

        # ---- 4. attention core variants on real q/k/v ----
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        a = blk.attn
        q = a.norm_q(a.to_q(nh).unflatten(-1, (a.heads, -1)))
        k = a.norm_k(a.to_k(nh).unflatten(-1, (a.heads, -1)))
        v = a.to_v(nh).unflatten(-1, (a.heads, -1))
        q = _apply_rotary_emb(q, *rope).transpose(1, 2).contiguous()
        k = _apply_rotary_emb(k, *rope).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        wait()
        S, D = q.shape[2], q.shape[3]
        variants = {}

        def sdpa_chunked(c):
            return torch.cat([F.scaled_dot_product_attention(q[:, s:s + c], k[:, s:s + c], v[:, s:s + c])
                              for s in range(0, q.shape[1], c)], dim=1)
        ref, f, trc, dv = bench(lambda: sdpa_chunked(4))
        ref_cpu = ref.float().cpu()
        variants["sdpa_chunk4 (current)"] = {"first_s": f, "trace_s": trc, "device_s": dv, "max_abs_diff": 0.0}
        log(f"SDPA chunk 4: device {dv}s")
        for c in (8, 14):
            try:
                o, f, trc, dv = bench(lambda: sdpa_chunked(c))
                d = (o.float().cpu() - ref_cpu).abs().max().item()
                variants[f"sdpa_chunk{c}"] = {"first_s": f, "trace_s": trc, "device_s": dv, "max_abs_diff": d}
                log(f"SDPA chunk {c}: device {dv}s, max diff {d:.3e}")
            except Exception as e:  # OOM
                variants[f"sdpa_chunk{c}"] = {"error": str(e)[:200]}
                log(f"SDPA chunk {c}: FAILED {str(e)[:120]}")
        try:
            from torch_xla.experimental.custom_kernel import flash_attention
            pad = (-S) % 512
            seg = torch.cat([torch.ones(S, dtype=torch.int32), torch.full((pad,), 2, dtype=torch.int32)])[None].to(dev)
            def flash():
                qp, kp, vp = (F.pad(t_, (0, 0, 0, pad)) for t_ in (q, k, v))
                o = flash_attention(qp, kp, vp, causal=False, q_segment_ids=seg, kv_segment_ids=seg,
                                    sm_scale=1.0 / math.sqrt(D))
                return o[:, :, :S]
            o, f, trc, dv = bench(flash)
            d = (o.float().cpu() - ref_cpu).abs().max().item()
            m = (o.float().cpu() - ref_cpu).abs().mean().item()
            variants["pallas_flash_attention"] = {"first_s": f, "trace_s": trc, "device_s": dv,
                                                  "max_abs_diff": d, "mean_abs_diff": m, "pad": pad}
            log(f"Pallas flash attention: device {dv}s, max diff {d:.3e}, mean diff {m:.3e} (pad {pad})")
        except Exception as e:
            variants["pallas_flash_attention"] = {"error": repr(e)[:300]}
            log(f"Pallas flash attention FAILED: {repr(e)[:200]}")
        REPORT["attention_variants"] = variants
        REPORT["ref_output_absmax"] = ref_cpu.abs().max().item()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(REPORT, indent=2) + "\n")
    log("REPORT " + json.dumps(REPORT))


if __name__ == "__main__":
    with torch.no_grad():  # the pipeline runs under no_grad; direct transformer calls must too, or autograd OOMs
        main()
