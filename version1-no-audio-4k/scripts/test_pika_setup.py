#!/usr/bin/env python3
"""
Test and Verification Suite for Pika H3 Flex & MiniMax H3 ComfyUI Environment
Location: /data/pika/scripts/test_pika_setup.py
"""

import hashlib
import json
import os
import sys
from pathlib import Path

PIKA_ROOT = Path("/data/pika")
COMFY_ROOT = PIKA_ROOT / "ComfyUI"
LORAS_DIR = COMFY_ROOT / "models" / "loras"
WORKFLOWS_DIR = PIKA_ROOT / "workflows"

OFFICIAL_CHECKSUMS = {
    "pika_h3_flex_fl2va_comfy_bf16.safetensors": "68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89",
    "pika_h3_flex_ref2va_comfy_bf16.safetensors": "e4d058fbd027377b925a0ee607b514ddd9037873fee3afa05d73b302e6f7f8e1",
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_directory_structure():
    print("=" * 70)
    print("1. CHECKING DIRECTORY STRUCTURE")
    print("=" * 70)
    checks = [
        ("Pika Root", PIKA_ROOT),
        ("ComfyUI Core", COMFY_ROOT),
        ("LoRA Models Dir", LORAS_DIR),
        ("Workflows Dir", WORKFLOWS_DIR),
        ("Minimax H3 Core Nodes", COMFY_ROOT / "comfy_extras" / "nodes_minimax_h3.py"),
    ]
    all_ok = True
    for name, path in checks:
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status:9} {name}: {path}")
        if not exists:
            all_ok = False
    return all_ok


def test_lora_adapters():
    print("\n" + "=" * 70)
    print("2. VERIFYING PIKA H3 FLEX LORA ADAPTER CHECKSUMS")
    print("=" * 70)
    all_ok = True
    for filename, expected_hash in OFFICIAL_CHECKSUMS.items():
        filepath = LORAS_DIR / filename
        if not filepath.exists():
            print(f"  [MISSING] {filename} does not exist yet")
            all_ok = False
            continue

        size_gb = filepath.stat().st_size / (1024**3)
        actual_hash = compute_sha256(filepath)
        if actual_hash == expected_hash:
            print(f"  [VERIFIED] {filename} ({size_gb:.2f} GB)")
            print(f"             SHA256: {actual_hash}")
        else:
            print(f"  [MISMATCH] {filename}")
            print(f"             Expected: {expected_hash}")
            print(f"             Actual:   {actual_hash}")
            all_ok = False
    return all_ok


def test_comfy_minimax_nodes():
    print("\n" + "=" * 70)
    print("3. VERIFYING COMFYUI MINIMAX-H3 NATIVE EXTENSION")
    print("=" * 70)
    node_file = COMFY_ROOT / "comfy_extras" / "nodes_minimax_h3.py"
    if not node_file.exists():
        print(f"  [FAIL] {node_file} not found")
        return False

    with open(node_file, "r") as f:
        content = f.read()

    classes_found = []
    for line in content.splitlines():
        if line.startswith("class ") and "io.ComfyNode" in line or "ComfyExtension" in line:
            cls_name = line.split("(")[0].replace("class ", "").strip()
            classes_found.append(cls_name)

    expected_classes = [
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3AddGuide",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
        "MiniMaxH3FunControlNetApply",
        "MiniMaxH3LatentUpscaleWithModel",
        "MiniMaxH3Extension",
    ]

    print(f"  Found {len(classes_found)} MiniMax-H3 native classes in ComfyUI extras:")
    for cls in expected_classes:
        if cls in classes_found:
            print(f"    [OK] {cls}")
        else:
            print(f"    [WARN] {cls} missing")

    return len(classes_found) >= 5


def test_workflows():
    print("\n" + "=" * 70)
    print("4. VERIFYING PIKA H3 FLEX WORKFLOW TEMPLATES")
    print("=" * 70)
    all_ok = True
    templates = [
        ("Text-to-Video (T2V)", WORKFLOWS_DIR / "video_pika_h3_flex_t2v.json"),
        ("First/Last Frame & Extension (FL2VA)", WORKFLOWS_DIR / "video_pika_h3_flex_fl2va.json"),
        ("Multimodal Reference (Ref2VA)", WORKFLOWS_DIR / "video_pika_h3_flex_ref2va.json"),
    ]
    for label, path in templates:
        if not path.exists():
            print(f"  [MISSING] {label} at {path}")
            all_ok = False
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            node_count = len(data.get("nodes", []))
            print(f"  [VALID JSON] {label} ({node_count} top-level nodes / subgraphs)")
        except Exception as e:
            print(f"  [CORRUPT] {label}: {e}")
            all_ok = False
    return all_ok


def main():
    print("\nStarting Pika H3 Flex & ComfyUI Validation Suite...")
    s1 = test_directory_structure()
    s2 = test_lora_adapters()
    s3 = test_comfy_minimax_nodes()
    s4 = test_workflows()

    print("\n" + "=" * 70)
    print("OVERALL STATUS:")
    if s1 and s2 and s3 and s4:
        print("  ALL VERIFICATION CHECKS PASSED (Ready for Deployment & Inference)")
    else:
        print("  SOME CHECKS PENDING OR FAILED (See above for details)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
