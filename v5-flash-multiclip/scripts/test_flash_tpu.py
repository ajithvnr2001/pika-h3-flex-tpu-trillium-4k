#!/usr/bin/env python3
"""Pallas flash attention vs the current head-chunked SDPA, on Pika's real attention shape.

q/k/v: (1, 56 heads, 15004 tokens, 128) bf16 on xla:0. The sequence is padded to a multiple of 512 and the padding
is masked out with segment ids, so real tokens never attend to padding.
"""
import math, os, time
os.environ.setdefault("PJRT_DEVICE", "TPU")
import torch, torch.nn.functional as F
import torch_xla, torch_xla.core.xla_model as xm

# torch_xla lowers Pallas kernels through jax without the TPU attached, so jax cannot detect the libtpu
# version and serializes Mosaic IR at its newest format (v8). libtpu 0.0.21 (torch_xla 2.9) reads <= v7.
if os.environ.get("PREC"):
    import torch_xla.backends; torch_xla.backends.set_mat_mul_precision(os.environ["PREC"]); print("matmul precision:", os.environ["PREC"])
import jax._src.tpu_custom_call as tcc
tcc.get_ir_version = lambda ctx: 7
from torch_xla.experimental.custom_kernel import flash_attention, FlashAttention


def wait():
    torch_xla.sync(); xm.wait_device_ops()


def timeit(fn, reps=3):
    out = fn(); wait()
    out = fn(); wait()
    t = time.monotonic()
    for _ in range(reps):
        out = fn()
    wait()
    return out, (time.monotonic() - t) / reps


dev = torch.device("xla:0")
B, H, S, D = 1, 56, int(os.environ.get("SEQ", 15004)), 128
g = torch.Generator().manual_seed(0)
q, k, v = (torch.randn(B, H, S, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev) for _ in range(3))
wait()

with torch.no_grad():
    ref, t_ref = timeit(lambda: torch.cat([F.scaled_dot_product_attention(q[:, s:s + 4], k[:, s:s + 4], v[:, s:s + 4])
                                           for s in range(0, H, 4)], dim=1))
    print(f"SDPA chunk 4 (current): {t_ref*1000:.1f} ms per block", flush=True)
    ref_cpu = ref.float().cpu()
    pad = (-S) % 512
    seg = torch.cat([torch.ones(S, dtype=torch.int32), torch.full((pad,), 2, dtype=torch.int32)])[None].to(dev)
    for bq, bk in [(512, 512), (1024, 512), (1024, 1024), (2048, 1024)]:
        FlashAttention.DEFAULT_BLOCK_SIZES.update(block_q=bq, block_k_major=bk, block_k=min(bk, 512))
        try:
            def fl():
                qp, kp, vp = (F.pad(t_, (0, 0, 0, pad)) for t_ in (q, k, v))
                return flash_attention(qp, kp, vp, causal=False, q_segment_ids=seg, kv_segment_ids=seg,
                                       sm_scale=1.0 / math.sqrt(D))[:, :, :S]
            o, t = timeit(fl)
            d = (o.float().cpu() - ref_cpu).abs()
            print(f"Pallas flash (block_q={bq}, block_k_major={bk}): {t*1000:.1f} ms per block, "
                  f"{t_ref/t:.2f}x, max diff {d.max():.3e}, mean diff {d.mean():.3e}", flush=True)
        except Exception as e:
            print(f"Pallas flash (block_q={bq}, block_k_major={bk}) FAILED: {repr(e)[:300]}", flush=True)
