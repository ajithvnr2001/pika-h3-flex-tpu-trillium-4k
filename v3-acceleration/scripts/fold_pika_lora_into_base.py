#!/usr/bin/env python3
"""
Pika H3 Flex LoRA Weight Folding Tool for Cloud TPU Trillium (v6e).

Performs host CPU offline weight fusion:
    W_fused = W_base + (alpha / rank) * (B @ A)
Casts to BF16 and produces a 100% native dense MiniMax-H3 checkpoint.
Zero dynamic LoRA branch overhead, zero custom TPU kernels, zero XLA recompilations.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def fold_lora_into_base(
    base_weights_path: str,
    lora_path: str,
    output_path: str,
    alpha: float = 8.0,
    rank: int = 128,
):
    print(f"=== Pika H3 Flex LoRA Weight Folding ===")
    print(f"Base Weights: {base_weights_path}")
    print(f"LoRA Adapter: {lora_path}")
    print(f"Output File:  {output_path}")
    print(f"Alpha: {alpha}, Rank: {rank} (Scale: {alpha/rank})")

    t0 = time.perf_counter()
    print("\n[1/4] Loading LoRA weights from disk...")
    lora_sd = load_file(lora_path, device="cpu")
    print(f"Loaded {len(lora_sd)} LoRA tensors in {time.perf_counter() - t0:.2f}s.")

    # Pair lora_A and lora_B
    lora_pairs = {}
    for key, weight in lora_sd.items():
        if key.endswith(".lora_A.weight"):
            prefix = key[:-len(".lora_A.weight")]
            b_key = f"{prefix}.lora_B.weight"
            if b_key in lora_sd:
                lora_pairs[prefix] = (weight.float(), lora_sd[b_key].float())

    print(f"[2/4] Identified {len(lora_pairs)} linear projection targets to fold.")

    scale = float(alpha) / float(rank)

    # Load base weights
    t1 = time.perf_counter()
    print(f"\n[3/4] Loading base weights and folding deltas...")
    base_sd = load_file(base_weights_path, device="cpu")
    print(f"Loaded {len(base_sd)} base model tensors in {time.perf_counter() - t1:.2f}s.")

    folded_count = 0
    with torch.no_grad():
        for prefix, (a_mat, b_mat) in lora_pairs.items():
            # Check possible weight key aliases
            target_key = f"{prefix}.weight"
            if target_key not in base_sd:
                # Strip diffusion_model. prefix or add transformer.
                cand = target_key.replace("diffusion_model.", "")
                if cand in base_sd:
                    target_key = cand
                elif f"transformer.{cand}" in base_sd:
                    target_key = f"transformer.{cand}"

            if target_key in base_sd:
                base_w = base_sd[target_key].float()
                delta = torch.matmul(b_mat, a_mat) * scale
                base_sd[target_key] = (base_w + delta).to(torch.bfloat16)
                folded_count += 1
            else:
                print(f"  [Warning] Target {target_key} not in base checkpoint, skipping.")

    print(f"Successfully folded {folded_count}/{len(lora_pairs)} targets.")

    t2 = time.perf_counter()
    print(f"\n[4/4] Saving fused weights to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(base_sd, output_path)
    print(f"Done in {time.perf_counter() - t2:.2f}s! Total time: {time.perf_counter() - t0:.2f}s.")
    print("Fused checkpoint is ready for 100% native dense TPU compilation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fold Pika H3 Flex LoRA into MiniMax-H3")
    parser.add_argument("--base", required=True, help="Path to base MiniMax-H3 safetensors")
    parser.add_argument("--lora", default="/data/pika/ComfyUI/models/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors", help="Path to LoRA")
    parser.add_argument("--out", required=True, help="Path to output fused safetensors")
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()

    fold_lora_into_base(args.base, args.lora, args.out, args.alpha, args.rank)
