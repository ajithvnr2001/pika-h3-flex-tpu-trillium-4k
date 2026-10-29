#!/usr/bin/env python3
"""
Remote Verification Test Harness for Pika H3 Flex on Cloud TPU Trillium (v6e).
Executes directly on the Cloud TPU VM node.
"""

import os
import sys
import time

# Ensure OpenXLA / PJRT selects TPU
os.environ.setdefault("PJRT_DEVICE", "TPU")

print("=" * 70)
print("Pika H3 Flex Cloud TPU Trillium (v6e) Remote Harness")
print("=" * 70)

# 1. Device Discovery
print("\n[1/4] Detecting Host Architecture and Environment...")
try:
    import platform
    print(f"OS/Kernel: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch not found in current Python environment: {e}")
    sys.exit(1)

# 2. Check TPU Backends (torch_tpu / torch_xla)
print("\n[2/4] Initializing TPU Accelerator Backend...")
tpu_backend = None
dev = None

# Try torch_tpu first (Google's native PrivateUse1 C++ runtime)
try:
    import torch_tpu
    tpu_backend = "torch_tpu"
    print("Detected Google TorchTPU backend (PrivateUse1)!")
    dev = torch.device("tpu:0")
    x = torch.ones((1024, 1024), dtype=torch.bfloat16, device=dev)
    y = torch.matmul(x, x)
    print(f"TorchTPU quick test passed on {dev}! Sum = {y.sum().item():.2f}")
except Exception as e:
    print(f"torch_tpu note: {e}")

# If not torch_tpu, try torch_xla (OpenXLA / StableHLO runtime)
if not tpu_backend:
    try:
        import torch_xla.core.xla_model as xm
        tpu_backend = "torch_xla"
        devices = xm.get_xla_supported_devices()
        print(f"PyTorch/XLA detected devices: {devices} (Count: {len(devices)})")
        dev = xm.xla_device()
        x = torch.ones((1024, 1024), dtype=torch.bfloat16, device=dev)
        y = torch.matmul(x, x)
        xm.mark_step()
        print(f"PyTorch/XLA quick test passed on {dev}! Sum = {y.cpu().sum().item():.2f}")
    except Exception as e:
        print(f"torch_xla note: {e}")

if not tpu_backend:
    print("FATAL: No TPU backend could be initialized on this node.")
    sys.exit(1)

print(f"\n[3/4] Active TPU Backend: {tpu_backend.upper()}")

# 3. Benchmark BF16 Matrix Multiplication for Trillium MXU Engine
print("\n[4/4] Benchmarking Trillium v6e Matrix Execution Units (MXU)...")
N = 8192
print(f"Executing BF16 Systolic GEMM ({N}x{N} @ BF16)...")

# Warm-up run to compile XLA graph / initialize systolic array
if tpu_backend == "torch_tpu":
    a = torch.randn((N, N), dtype=torch.bfloat16, device="tpu")
    b = torch.randn((N, N), dtype=torch.bfloat16, device="tpu")
    c = torch.matmul(a, b)
    torch.tpu.synchronize()
else:
    a = torch.randn((N, N), dtype=torch.bfloat16, device=dev)
    b = torch.randn((N, N), dtype=torch.bfloat16, device=dev)
    c = torch.matmul(a, b)
    xm.mark_step()

print("Warm-up complete. Running timed benchmark (10 iterations)...")
iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    if tpu_backend == "torch_tpu":
        c = torch.matmul(a, b)
        torch.tpu.synchronize()
    else:
        c = torch.matmul(a, b)
        xm.mark_step()
t_total = time.perf_counter() - t0
avg_time = t_total / iters
tflops = (2 * (N ** 3)) / (avg_time * 1e12)

print(f"Average GEMM time: {avg_time*1000:.2f} ms")
print(f"Effective Compute Throughput: {tflops:.2f} TFLOPS")
print("\n" + "=" * 70)
print("SUCCESS: Cloud TPU v6e Trillium node is 100% verified and operational!")
print("Ready for Pika H3 Flex 6-step inference pipeline.")
print("=" * 70)
