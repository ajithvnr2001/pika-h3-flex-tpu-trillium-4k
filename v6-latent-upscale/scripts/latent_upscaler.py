# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 latent upscaler (LatentResizer3D), ported without ComfyUI dependencies.

Model and architecture: https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0).
Code adapted from https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
(nodes/minimax_h3_latent_upscaler_3d.py, Apache-2.0): same modules and forward, ComfyUI/einops/attention parts removed
(the released v1 checkpoint has no attention blocks).

Input and output are **normalized** latents ((z - mean) / std per channel), which is the form Pika H3 Flex's pipeline
uses internally, shape (B, 24, T, H, W). Only H and W are upscaled.
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
REVISION = "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"
WEIGHTS = "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"


def normalization(channels):
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        self.out_norm = normalization(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout), nn.Conv3d(channels, channels, 3, padding=1))
        self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)[..., None, None, None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                                padding=(kernel_size // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        embed_dim = 64
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        def stack(n):
            blocks = nn.ModuleList()
            for b in range(n):
                blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
                if temporal_every > 0 and b % temporal_every == 0:
                    blocks.append(TemporalConv(channels, temporal_kernel))
            return blocks

        self.in_blocks = stack(in_blocks)
        self.out_blocks = stack(out_blocks)
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale, size):
        """x: normalized latents (B, 24, T, H, W); size: (T, H_out, W_out). Temporal convs are local in T, so the
        whole clip (T=37 for 5 s) is processed in one segment (the ComfyUI node chunks only above T=32 to save VRAM)."""
        emb = self.embed(torch.tensor([[scale - 1.0]], dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def load_upscaler(path, dtype=torch.bfloat16):
    from safetensors.torch import load_file
    sd = load_file(path, device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    in_blocks = len({int(m.group(1)) for k in sd if (m := re.match(r"in_blocks\.(\d+)\.in_layers\.", k))})
    out_blocks = len({int(m.group(1)) for k in sd if (m := re.match(r"out_blocks\.(\d+)\.in_layers\.", k))})
    channels = sd["conv_in.weight"].shape[0]
    tk = next(v.shape[2] for k, v in sd.items() if k.endswith("dwconv.weight"))
    model = LatentResizer3D(in_channels=sd["conv_in.weight"].shape[1], in_blocks=in_blocks, out_blocks=out_blocks,
                            channels=channels, temporal_kernel=tk)
    model.load_state_dict(sd, strict=True)
    return model.eval().requires_grad_(False).to(dtype)
