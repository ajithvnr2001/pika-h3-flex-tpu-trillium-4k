#!/usr/bin/env python3
"""
Autonomous End-to-End Orchestrator for Pika H3 Flex v2 (Joint Video + Synchronized Stereo Audio + 4K UHD)
on Google Cloud TPU Trillium (v6e-4).
Provisions Spot TPU VM (v6e-4) in europe-west4-a using 150GB RAM disk (/dev/shm/pika_data),
executes fast parallel HF API download, generates 5s & 10s videos with synchronized 48kHz stereo AAC audio,
upscales both to 4K UHD (3840x2160) on the 32-core TPU host, fetches all artifacts, and tears down the TPU VM ($0.00/hr idle cost).
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ID = "genai25th01"
ZONE = "europe-west4-a"
NODE_NAME = "pika-europe-tpu"

SCRIPT_DIR = Path(__file__).resolve().parent
BOOTSTRAP_SCRIPT = SCRIPT_DIR / "bootstrap_video_audio_pipeline_tpu.sh"
RUNNER_SCRIPT = SCRIPT_DIR / "generate_pika_video_audio_tpu.py"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()


def delete_node():
    print(f"\n[MANDATORY TEARDOWN] Deleting TPU VM {NODE_NAME} in {ZONE}...", flush=True)
    token = get_token()
    url = f"https://tpu.googleapis.com/v2/projects/{PROJECT_ID}/locations/{ZONE}/nodes/{NODE_NAME}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="DELETE")
    try:
        with urllib.request.urlopen(req) as resp:
            op = json.loads(resp.read().decode("utf-8"))
            print(f"Teardown operation accepted: {op.get('name')}", flush=True)
            return op
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Node {NODE_NAME} already deleted.", flush=True)
        else:
            print(f"Delete HTTP error ({e.code}): {e.read().decode('utf-8')}", flush=True)


def wait_for_prior_deletion(max_wait=360):
    print(f"Verifying zone {ZONE} has no lingering instances...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        res = subprocess.run([
            "gcloud", "compute", "tpus", "tpu-vm", "list",
            f"--zone={ZONE}", f"--project={PROJECT_ID}", "--format=value(name)"
        ], capture_output=True, text=True)
        nodes = res.stdout.strip().split()
        if not nodes or NODE_NAME not in nodes:
            print(f"Zone {ZONE} is clear of {NODE_NAME}.", flush=True)
            return True
        print(
            f"[{time.strftime('%H:%M:%S')}] Waiting for previous instance {NODE_NAME} "
            f"to finish deleting ({int(time.time() - t0)}s elapsed)...",
            flush=True,
        )
        time.sleep(12)
    raise TimeoutError(f"Previous instance lingered beyond {max_wait}s")


def create_node():
    print(f"Initiating Spot TPU creation ({NODE_NAME}, v6e-4) in {ZONE}...", flush=True)
    token = get_token()
    url = f"https://tpu.googleapis.com/v2/projects/{PROJECT_ID}/locations/{ZONE}/nodes?nodeId={NODE_NAME}"
    payload = {
        "acceleratorType": "v6e-4",
        "runtimeVersion": "v2-alpha-tpuv6e",
        "schedulingConfig": {"spot": True},
        "networkConfig": {"enableExternalIps": True},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }, method="POST")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"Node creation accepted! Operation: {res.get('name')}", flush=True)
        return res.get("name")


def check_operation(op_name):
    if not op_name:
        return None
    token = get_token()
    url = f"https://tpu.googleapis.com/v2/{op_name}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def wait_for_ready(op_name=None, max_wait=360):
    print(f"Waiting for {NODE_NAME} in {ZONE} to reach READY...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        res = subprocess.run([
            "gcloud", "compute", "tpus", "tpu-vm", "describe", NODE_NAME,
            f"--zone={ZONE}", f"--project={PROJECT_ID}", "--format=json(state,networkEndpoints)"
        ], capture_output=True, text=True)
        if res.returncode == 0:
            try:
                data = json.loads(res.stdout)
                state = data.get("state")
                print(f"[{time.strftime('%H:%M:%S')}] State: {state}", flush=True)
                if state == "READY":
                    endpoints = data.get("networkEndpoints", [{}])[0]
                    ext_ip = endpoints.get("accessConfig", {}).get("externalIp")
                    int_ip = endpoints.get("ipAddress")
                    print(f"Node is READY! Internal IP: {int_ip}, External IP: {ext_ip}", flush=True)
                    return ext_ip
                elif state in ["STOPPED", "FAILED", "SUSPENDED"]:
                    raise RuntimeError(f"Node entered state: {state}")
            except Exception as e:
                print(f"Parse error: {e}", flush=True)
        else:
            op_info = check_operation(op_name)
            if op_info and op_info.get("done") and "error" in op_info:
                raise RuntimeError(f"Create operation failed: {op_info['error']}")
        time.sleep(12)
    raise TimeoutError(f"Node did not reach READY within {max_wait}s")


def wait_for_ssh(ip, port=22, timeout=120):
    if not ip:
        return
    print(f"Waiting for SSH port {port} on external IP {ip} to accept connections...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        res = sock.connect_ex((ip, port))
        sock.close()
        if res == 0:
            print(f"SSH port {port} is OPEN on {ip}!", flush=True)
            time.sleep(5)
            return True
        time.sleep(4)
    return False


def run():
    t_orchestrator_0 = time.perf_counter()
    try:
        wait_for_prior_deletion()

        t_prov_0 = time.perf_counter()
        op_name = create_node()
        ext_ip = wait_for_ready(op_name=op_name)
        wait_for_ssh(ext_ip)
        t_prov = time.perf_counter() - t_prov_0
        print(f"TPU VM Provisioning & SSH readiness completed in {t_prov:.2f}s", flush=True)

        print("\n[1/3] Copying v2 Video+Audio scripts to TPU VM...", flush=True)
        scp_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            str(RUNNER_SCRIPT), str(BOOTSTRAP_SCRIPT), f"{NODE_NAME}:~/",
            f"--zone={ZONE}", f"--project={PROJECT_ID}", "--quiet"
        ]
        subprocess.run(scp_cmd, check=True)

        print("\n[2/3] Executing bootstrap, fast parallel download & Joint Video+Audio+4K generation...", flush=True)
        hf_token = os.environ.get("HF_TOKEN", "")
        ssh_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", NODE_NAME,
            f"--zone={ZONE}", f"--project={PROJECT_ID}",
            f"--command=HF_TOKEN={hf_token} bash ~/bootstrap_video_audio_pipeline_tpu.sh"
        ]
        subprocess.run(ssh_cmd, check=True)

        print("\n[3/3] Downloading generated Video+Audio+4K outputs and timing reports...", flush=True)
        fetch_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            f"{NODE_NAME}:/dev/shm/pika_data/outputs/*", str(OUTPUT_DIR) + "/",
            f"--zone={ZONE}", f"--project={PROJECT_ID}", "--quiet"
        ]
        fetch_res = subprocess.run(fetch_cmd, capture_output=True, text=True)
        if fetch_res.returncode == 0:
            print(f"Artifacts successfully downloaded to {OUTPUT_DIR}:", flush=True)
            for f in sorted(OUTPUT_DIR.glob("*")):
                print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)", flush=True)
        else:
            print(f"Fetch note: {fetch_res.stderr}", flush=True)

    finally:
        delete_node()
        print("\nWaiting for TPU node deletion to complete...", flush=True)
        try:
            wait_for_prior_deletion(max_wait=240)
        except Exception as e:
            print(f"Wait note: {e}", flush=True)
        print("\nTeardown completed. Verifying zero active resources...", flush=True)
        subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={ZONE}", f"--project={PROJECT_ID}"])
        print(f"Total Orchestrator Wall Time: {time.perf_counter() - t_orchestrator_0:.2f}s", flush=True)


if __name__ == "__main__":
    run()
