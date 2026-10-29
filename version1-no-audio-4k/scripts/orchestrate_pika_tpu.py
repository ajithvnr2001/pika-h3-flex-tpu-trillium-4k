#!/usr/bin/env python3
"""
End-to-End Orchestrator for Pika H3 Flex on Cloud TPU Trillium (v6e).
Connects to pika-europe-tpu in europe-west4-a, deploys bootstrap & test harness,
executes BF16 Trillium GEMM, captures results, and auto-deletes the node to maintain $0.00 idle cost.
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
BOOTSTRAP_SCRIPT = Path("/data/pika/scripts/bootstrap_and_test_tpu.sh")
LOCAL_TEST_HARNESS = Path("/data/pika/scripts/test_pika_tpu_remote.py")
OUTPUT_DIR = Path("/data/pika/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()


def delete_node():
    print(f"\n[MANDATORY TEARDOWN] Deleting TPU VM {NODE_NAME} in {ZONE}...")
    token = get_token()
    url = f"https://tpu.googleapis.com/v2/projects/{PROJECT_ID}/locations/{ZONE}/nodes/{NODE_NAME}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="DELETE")
    try:
        with urllib.request.urlopen(req) as resp:
            op = json.loads(resp.read().decode("utf-8"))
            print(f"Teardown operation accepted: {op.get('name')}")
            return op
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Node {NODE_NAME} already deleted.")
        else:
            print(f"Delete HTTP error ({e.code}): {e.read().decode('utf-8')}")


def wait_for_prior_deletion(max_wait=360):
    print(f"Verifying zone {ZONE} has no lingering instances...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        res = subprocess.run([
            "gcloud", "compute", "tpus", "tpu-vm", "list",
            f"--zone={ZONE}", f"--project={PROJECT_ID}", "--format=value(name)"
        ], capture_output=True, text=True)
        nodes = res.stdout.strip().split()
        if not nodes or NODE_NAME not in nodes:
            print(f"Zone {ZONE} is clear of {NODE_NAME}.")
            return True
        print(f"[{time.strftime('%H:%M:%S')}] Waiting for previous instance {NODE_NAME} to finish deleting ({int(time.time() - t0)}s elapsed)...")
        time.sleep(12)
    raise TimeoutError(f"Previous instance lingered beyond {max_wait}s")


def create_node():
    print(f"Initiating Spot TPU creation ({NODE_NAME}) in {ZONE}...")
    token = get_token()
    url = f"https://tpu.googleapis.com/v2/projects/{PROJECT_ID}/locations/{ZONE}/nodes?nodeId={NODE_NAME}"
    payload = {
        "acceleratorType": "v6e-4",
        "runtimeVersion": "v2-alpha-tpuv6e",
        "schedulingConfig": {"spot": True},
        "networkConfig": {"enableExternalIps": True}
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }, method="POST")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"Node creation accepted! Operation: {res.get('name')}")
        return res


def wait_for_ready(max_wait=360):
    print(f"Waiting for {NODE_NAME} in {ZONE} to reach READY...")
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
                print(f"[{time.strftime('%H:%M:%S')}] State: {state}")
                if state == "READY":
                    endpoints = data.get("networkEndpoints", [{}])[0]
                    ext_ip = endpoints.get("accessConfig", {}).get("externalIp")
                    int_ip = endpoints.get("ipAddress")
                    print(f"Node is READY! Internal IP: {int_ip}, External IP: {ext_ip}")
                    return ext_ip
                elif state in ["STOPPED", "FAILED", "SUSPENDED"]:
                    raise RuntimeError(f"Node entered state: {state}")
            except Exception as e:
                print(f"Parse error: {e}")
        time.sleep(12)
    raise TimeoutError(f"Node did not reach READY within {max_wait}s")


def wait_for_ssh(ip, port=22, timeout=120):
    if not ip:
        print("No external IP found; will attempt gcloud tpu-vm ssh directly...")
        return
    print(f"Waiting for SSH port {port} on external IP {ip} to accept connections...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        res = sock.connect_ex((ip, port))
        sock.close()
        if res == 0:
            print(f"SSH port {port} is OPEN on {ip}!")
            time.sleep(5)  # allow sshd daemon to settle
            return True
        time.sleep(4)
    print("SSH port check timed out; attempting gcloud compute tpus tpu-vm ssh...")
    return False


def run():
    try:
        # Step 0: Ensure zone is clean
        wait_for_prior_deletion()

        # Step 1: Create node
        create_node()
        ext_ip = wait_for_ready()
        wait_for_ssh(ext_ip)

        # Step 2: Copy test harness and bootstrap script
        print("\n[1/3] Copying scripts to TPU VM...")
        scp_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            str(LOCAL_TEST_HARNESS), str(BOOTSTRAP_SCRIPT), f"{NODE_NAME}:~/",
            f"--zone={ZONE}", f"--project={PROJECT_ID}", "--quiet"
        ]
        subprocess.run(scp_cmd, check=True)

        # Step 3: Execute benchmark
        print("\n[2/3] Executing bootstrap & Trillium v6e BF16 systolic GEMM benchmark on TPU node...")
        ssh_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", NODE_NAME,
            f"--zone={ZONE}", f"--project={PROJECT_ID}",
            "--command=bash ~/bootstrap_and_test_tpu.sh"
        ]
        res = subprocess.run(ssh_cmd, capture_output=True, text=True)
        print("\n=== TPU REMOTE BENCHMARK OUTPUT ===")
        print(res.stdout)
        if res.stderr:
            print("=== STDERR ===")
            print(res.stderr)

        # Step 4: Save report
        report_file = OUTPUT_DIR / f"tpu_benchmark_report_{int(time.time())}.txt"
        report_file.write_text(res.stdout + "\n" + res.stderr)
        print(f"\n[3/3] Report successfully written to {report_file}")

    finally:
        delete_node()
        print("\nTeardown initiated. Verifying zero active resources...")
        time.sleep(5)
        subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={ZONE}", f"--project={PROJECT_ID}"])


if __name__ == "__main__":
    run()
