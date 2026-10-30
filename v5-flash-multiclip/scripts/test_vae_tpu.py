#!/usr/bin/env python3
"""Check that the Pika video VAE decoder gives the same pixels on TPU as on CPU, and time both.

Decodes one 256x256-pixel tile (the VAE's own tiling unit) from random latents on CPU and on xla:3 in float32
with matmul precision 'highest', then decodes a full 5 s 832x480 latent on TPU.
"""
import os, time
os.environ.setdefault("PJRT_DEVICE", "TPU")
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.backends

xr.initialize_cache("/dev/shm/pika/xla_cache")
torch_xla.backends.set_mat_mul_precision("highest")
from diffusers import AutoencoderKLMiniMaxH3

torch.set_num_threads(os.cpu_count())
vae = AutoencoderKLMiniMaxH3.from_pretrained("/dev/shm/pika/model/vae", torch_dtype=torch.float32).eval()
dev = torch.device("xla:3")
g = torch.Generator().manual_seed(0)
z_tile = torch.randn(1, 24, 3 + 3, 16, 16, generator=g)  # one clip worth of latent frames, one tile

with torch.no_grad():
    t = time.monotonic(); ref = vae.decoder(vae.post_quant_conv(z_tile)); t_cpu = time.monotonic() - t
    vae.decoder.to(dev)
    for i in range(3):
        t = time.monotonic()
        out = vae.decoder(vae.post_quant_conv(z_tile).to(dev)).cpu()
        print(f"TPU tile run {i}: {time.monotonic() - t:.2f}s", flush=True)
    err = (out - ref).abs()
    print(f"tile shape {tuple(ref.shape)} | CPU {t_cpu:.2f}s | max abs diff {err.max():.2e} | mean {err.mean():.2e} "
          f"| pixel range {ref.min():.2f}..{ref.max():.2f}", flush=True)

    # Full 5 s clip: 37 latent frames, 30x52 latent canvas -> (1, 3, 124, 480, 832)
    orig = vae.decoder.forward
    vae.decoder.forward = lambda x: orig(x.to(dev)).cpu()
    z = torch.randn(1, 24, 37, 30, 52, generator=g)
    for i in range(2):
        t = time.monotonic(); video = vae.decode(z).sample
        print(f"Full 5s decode on TPU run {i}: {time.monotonic() - t:.1f}s -> {tuple(video.shape)}", flush=True)
