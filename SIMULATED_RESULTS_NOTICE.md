# Notice: version1, version2 and v3 did not run the Pika model

`version1-no-audio-4k/`, `version2-with-audio-4k/` and `v3-acceleration/` are kept unchanged for history, but
**their videos and timings are not real Pika H3 Flex generations.** Use [`v4-real-generation/`](v4-real-generation/).

What those scripts actually do:

| What the README says | What the code does |
| --- | --- |
| "6-step diffusion on TPU, 18 ms/step", "21x faster" | Multiplies random tensors (`torch.randn(...)` latents and a random `proj_w` matrix) on the TPU and times that. No transformer weights are used. |
| "Generated 5 s / 10 s video" | Re-encodes the sample clip `assets/showcases/camera-rally.mp4` that ships with the model repo. The prompt has no effect. |
| "48 kHz stereo audio from the Audio VAE" | Synthesised noise plus a few Audio VAE tensors, not a decoded model output. |
| "4K UHD" | ffmpeg Lanczos resize of the sample clip. |

Search the scripts for `camera-rally.mp4` and `torch.randn` to see this, for example
`version2-with-audio-4k/scripts/generate_pika_video_audio_tpu.py` lines 168, 305-307 and 341.

The real numbers on the same TPU v6e-4 are in [`v4-real-generation/README.md`](v4-real-generation/README.md):
about 6 minutes (360 s) for a 5 s 480p clip on a fresh VM, not 3-13 seconds.
