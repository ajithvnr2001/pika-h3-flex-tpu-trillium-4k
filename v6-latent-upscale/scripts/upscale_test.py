#!/usr/bin/env python3
"""Test the MiniMax H3 latent upscaler on real Pika H3 Flex clips (TPU v6e-4).

For each prompt, in one process:
  stage 1  : normal 480p generation (832x480), v5 pipeline (flash attention). Saves <name>_480p.mp4.
             The final video latents and audio latents are captured at the VAE decoders.
  upscale  : latent upscaler 2x on TPU, (1,24,37,30,52) -> (1,24,37,60,104).
  variant A: decode the upscaled latents directly          -> <name>_up2x_norefine.mp4 (1664x960)
  variant B: refine at 1664x960 with Pika's own last 4 steps (timesteps 500, 250, 100, 40), starting from the
             upscaled latents noised to the matching sigma    -> <name>_up2x_refine.mp4 (1664x960)
Audio for A and B is the stage-1 audio (same soundtrack). Timings go to <name>_upscale.json.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.backends

xr.initialize_cache("/dev/shm/pika/xla_cache")
torch_xla.backends.set_mat_mul_precision("highest")
from generate_pika_tpu import TransformerOnTPU, BUCKETS, log  # noqa: E402
from latent_upscaler import load_upscaler  # noqa: E402


def shifted(sigma, shift):
    return shift * sigma / (1 + (shift - 1) * sigma)


def frames_from_video(video):
    """(1, 3, F, H, W) in [0, 1] -> uint8 tensor (F, H, W, 3), the form diffusers' encode_video takes."""
    return (video[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="/dev/shm/pika/model")
    p.add_argument("--upscaler", default="/dev/shm/pika/upscaler/model.safetensors")
    p.add_argument("--jobs", type=Path, required=True, help="lines: <prompt_file> <name>")
    p.add_argument("--outdir", type=Path, default=Path("/dev/shm/pika/outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--refine-from", type=int, default=500, help="first timestep of the refine pass (Pika grid)")
    p.add_argument("--skip-refine", action="store_true")
    p.add_argument("--convention", choices=["comfy", "norm", "raw"], default="comfy",
                   help="latent form fed to the upscaler. comfy (correct, measured by check_upscaler_convention.py): "
                        "Pika's normalized latents normalized once more, as ComfyUI's node does")
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count())
    from diffusers import ModularPipeline
    from diffusers.utils.export_utils import encode_video

    pipe = ModularPipeline.from_pretrained(args.snapshot)
    pipe.load_components(pretrained_model_name_or_path=args.snapshot, workflow="fl2va",
                         dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
    pipe.load_lora_weights(args.snapshot, weight_name="fl2va/adapter.safetensors")
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))
    tpu = TransformerOnTPU(pipe.transformer, 4, 4, "flash")
    vae_dev = tpu.devices[-1]
    pipe.vae.decoder.to(vae_dev)
    dec_fwd = pipe.vae.decoder.forward
    pipe.vae.decoder.forward = lambda x: dec_fwd(x.to(vae_dev)).cpu()

    up_dev = tpu.devices[1]  # chips 0-2 hold ~13 blocks each; chip 3 also holds the 9.7 GB VAE decoder
    t = time.monotonic()
    upscaler = load_upscaler(args.upscaler).to(up_dev)
    torch_xla.sync()
    log(f"Latent upscaler loaded on {up_dev} in {time.monotonic() - t:.1f}s")

    lat_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    a_mean = torch.tensor(pipe.audio_vae.config.latents_mean).view(1, -1, 1)
    a_std = torch.tensor(pipe.audio_vae.config.latents_std).view(1, -1, 1)
    pix_mean = torch.tensor(pipe.pixel_mean).view(1, -1, 1, 1, 1)
    pix_std = torch.tensor(pipe.pixel_std).view(1, -1, 1, 1, 1)

    cap = {}
    vae_decode, audio_decode = pipe.vae.decode, pipe.audio_vae.decode
    def vae_decode_cap(z, *a, **k):
        cap["video_norm"] = ((z.float().cpu() - lat_mean) / lat_std)
        t = time.monotonic(); r = vae_decode(z, *a, **k); cap["vae_s"] = time.monotonic() - t
        return r
    def audio_decode_cap(z, *a, **k):
        cap["audio_norm"] = ((z.float().cpu() - a_mean) / a_std)
        return audio_decode(z, *a, **k)
    pipe.vae.decode, pipe.audio_vae.decode = vae_decode_cap, audio_decode_cap

    # Refine schedule: drop the first entries of Pika's sigma grid, for both schedulers.
    try:
        steps = list(pipe.config.denoising_step_list)
    except Exception:
        steps = [1000, 750, 500, 250, 100, 40, 0]  # Pika's released schedule
    grid = [s / 1000.0 for s in steps]
    start = grid.index(args.refine_from / 1000.0)
    refine = {"on": False}
    for sch in (pipe.scheduler, pipe.audio_scheduler):
        orig = sch.set_timesteps
        def wrapped(*a, _orig=orig, **k):
            if refine["on"]:
                k["sigmas"] = k["sigmas"][start:]
            return _orig(*a, **k)
        sch.set_timesteps = wrapped
    sig_v = shifted(grid[start], pipe.scheduler.config.shift)
    sig_a = shifted(grid[start], pipe.audio_scheduler.config.shift)
    log(f"Refine starts at timestep {args.refine_from}: video sigma {sig_v:.4f}, audio sigma {sig_a:.4f}, "
        f"{len(grid) - start - 1} model evaluations")

    w, h = BUCKETS["480p"]
    W2, H2 = int(round(w * args.scale / 32) * 32), int(round(h * args.scale / 32) * 32)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for line in args.jobs.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        prompt_file, name = line.split()[:2]
        prompt = Path(prompt_file).read_text().strip()
        rec = {"name": name, "convention": args.convention, "seed": args.seed, "scale": args.scale, "refine_from": args.refine_from,
               "base": f"{w}x{h}", "upscaled": f"{W2}x{H2}"}

        # ---- stage 1: 480p ----
        log(f"[{name}] stage 1: 480p generation")
        tpu.step_times = []
        t = time.monotonic()
        res = pipe(prompt=prompt, height=h, width=w, num_frames=120,
                   generator=torch.Generator("cpu").manual_seed(args.seed),
                   output=["videos", "audio", "sampling_rate"])
        rec["stage1_s"] = round(time.monotonic() - t, 2)
        rec["stage1_steps_s"] = tpu.step_times
        audio = res["audio"][0].detach().float().cpu()
        sr = res["sampling_rate"]
        encode_video(res["videos"][0], fps=24, output_path=str(args.outdir / f"{name}_480p.mp4"),
                     audio=audio, audio_sample_rate=sr)
        lat, aud = cap["video_norm"], cap["audio_norm"]
        rec["latent_shape"] = list(lat.shape)
        T, H, W = lat.shape[2], lat.shape[3], lat.shape[4]

        # ---- latent upscale ----
        size = (T, H2 // 16, W2 // 16)
        times = []
        for i in range(2):  # 1st call compiles; the 2nd is the warm time
            t = time.monotonic()
            with torch.no_grad():
                if args.convention == "comfy":
                    x_in = (lat - lat_mean) / lat_std
                elif args.convention == "raw":
                    x_in = lat * lat_std + lat_mean
                else:
                    x_in = lat
                up = upscaler(x_in.to(up_dev, torch.bfloat16), args.scale, size).float().cpu()
                if args.convention == "comfy":
                    up = up * lat_std + lat_mean
                elif args.convention == "raw":
                    up = (up - lat_mean) / lat_std
            times.append(round(time.monotonic() - t, 2))
        rec["upscaler_s"] = times
        rec["upscaled_latent_shape"] = list(up.shape)
        log(f"[{name}] latent upscaled {tuple(lat.shape)} -> {tuple(up.shape)} in {times}s "
            f"(input std {lat.std():.3f}, output std {up.std():.3f})")

        # ---- variant A: decode directly ----
        t = time.monotonic()
        video = vae_decode(up * lat_std + lat_mean, return_dict=False)[0]
        video = (video.float() * pix_std + pix_mean).clamp(0, 1)
        rec["decode_up2x_s"] = round(time.monotonic() - t, 2)
        encode_video(frames_from_video(video), fps=24, output_path=str(args.outdir / f"{name}_up2x_norefine.mp4"),
                     audio=audio, audio_sample_rate=sr)
        log(f"[{name}] variant A (no refine) decoded {tuple(video.shape)} in {rec['decode_up2x_s']}s")

        # ---- variant B: refine at 2x with Pika's last steps ----
        if not args.skip_refine:
            g = torch.Generator("cpu").manual_seed(args.seed + 1)
            x_v = (1 - sig_v) * up + sig_v * torch.randn(up.shape, generator=g)
            x_a = (1 - sig_a) * aud + sig_a * torch.randn(aud.shape, generator=g)
            refine["on"] = True
            tpu.step_times = []
            t = time.monotonic()
            try:
                res2 = pipe(prompt=prompt, height=H2, width=W2, num_frames=120, latents=x_v, audio_latents=x_a,
                            generator=torch.Generator("cpu").manual_seed(args.seed + 2),
                            output=["videos", "audio", "sampling_rate"])
            finally:
                refine["on"] = False
            rec["refine_total_s"] = round(time.monotonic() - t, 2)
            rec["refine_steps_s"] = tpu.step_times
            rec["refine_vae_s"] = round(cap.get("vae_s", 0.0), 2)
            encode_video(res2["videos"][0], fps=24, output_path=str(args.outdir / f"{name}_up2x_refine.mp4"),
                         audio=audio, audio_sample_rate=sr)
            log(f"[{name}] variant B (refine) {W2}x{H2} in {rec['refine_total_s']}s, steps {tpu.step_times}")

        (args.outdir / f"{name}_upscale.json").write_text(json.dumps(rec, indent=2) + "\n")
        log(f"[{name}] RESULT {json.dumps(rec)}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
