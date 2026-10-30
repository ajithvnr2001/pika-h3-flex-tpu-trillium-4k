#!/usr/bin/env python3
"""Find which latent form the upscaler expects, using real data (CPU only, no TPU).

Takes 34 real frames of a generated 832x480 clip as the "high-res" target and a 416x240 downscale as the "low-res"
input, encodes both with Pika's own VAE encoder, upscales the low-res latent 2x and measures how close it is to the
high-res latent, for two input conventions:
  A: normalized latents ((z - mean) / std)  -- Pika's pipeline form
  B: raw VAE latents (z)                    -- what a ComfyUI LATENT may hold
Baseline: plain trilinear 2x of the latent.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_upscaler import load_upscaler
from diffusers import AutoencoderKLMiniMaxH3
from safetensors import safe_open
import av

torch.set_num_threads(os.cpu_count())
VAE_DIR = os.environ.get("VAE_DIR", "/dev/shm/pika/model/vae")
UPSCALER = os.environ.get("UPSCALER", "/dev/shm/pika/upscaler/model.safetensors")
cfg = json.load(open(f"{VAE_DIR}/config.json"))
kw = {k: v for k, v in cfg.items() if not k.startswith("_")}
kw["decoder_num_layers"] = 0  # encoder-only check: keep the ViT decoder empty
vae = AutoencoderKLMiniMaxH3(**kw).eval()
sd = {}
with safe_open(f"{VAE_DIR}/diffusion_pytorch_model-00001-of-00002.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith(("encoder.", "quant_conv.")):
            sd[k] = f.get_tensor(k).float()
missing, unexpected = vae.load_state_dict(sd, strict=False)
assert not unexpected and not [m for m in missing if m.startswith(("encoder.", "quant_conv."))], missing
mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1, 1)
std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1, 1)
pm = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
ps = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

clip = sys.argv[1]
frames = [fr.to_ndarray(format="rgb24") for i, fr in zip(range(34), av.open(clip).decode(video=0))]
hi = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)[None].float() / 255  # (1,3,34,480,832)
lo = F.interpolate(hi[0].permute(1, 0, 2, 3), size=(240, 416), mode="area").permute(1, 0, 2, 3)[None]


def enc(x):
    with torch.no_grad():
        return vae.encode((x - pm) / ps).latent_dist.mode()


z_hi, z_lo = enc(hi), enc(lo)
print("raw latents", tuple(z_lo.shape), "->", tuple(z_hi.shape), flush=True)
up = load_upscaler(UPSCALER, dtype=torch.float32)
size = tuple(z_hi.shape[-3:])
n_hi, n_lo = (z_hi - mean) / std, (z_lo - mean) / std


def report(name, pred_norm):
    err = (pred_norm - n_hi)
    corr = torch.corrcoef(torch.stack([pred_norm.flatten(), n_hi.flatten()]))[0, 1].item()
    print(f"{name:34s} corr {corr:.3f}  rmse {err.pow(2).mean().sqrt():.3f}  out std {pred_norm.std():.3f} "
          f"(target std {n_hi.std():.3f})", flush=True)


with torch.no_grad():
    report("baseline trilinear (normalized)", F.interpolate(n_lo, size=size, mode="trilinear"))
    report("A: upscaler(normalized)", up(n_lo, 2.0, size))
    report("B: upscaler(raw) -> normalized", (up(z_lo, 2.0, size) - mean) / std)
    report("C: upscaler((norm-mean)/std)*std+mean", up((n_lo - mean) / std, 2.0, size) * std + mean)
    report("D: upscaler(raw), output already normalized", up(z_lo, 2.0, size))
