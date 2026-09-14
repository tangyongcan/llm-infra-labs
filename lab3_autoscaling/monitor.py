"""Samples the autoscaling loop once a second into a CSV.

Records, on one timeline:
  * queue_depth / in_flight        the signal KEDA scales on
  * desired / current / ready      what the HPA did about it
  * worker CPU millicores          the signal we deliberately did NOT scale on

The CPU columns are the counterfactual: during the same spike that drives
queue depth to 100+, worker CPU stays near idle, so a CPU-based HPA at a
typical 80%-of-request target would not have fired.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import urllib.request
from pathlib import Path

NS = "llmserving"


def sh(cmd: list[str], timeout: int = 10) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return out.stdout.strip()
    except Exception:
        return ""


def broker_metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}


def deploy_state() -> dict:
    raw = sh(
        [
            "kubectl", "-n", NS, "get", "deploy", "llm-worker",
            "-o", "jsonpath={.spec.replicas} {.status.replicas} "
                  "{.status.readyReplicas} {.status.unavailableReplicas}",
        ]
    )
    parts = (raw or "").split()

    def num(i: int) -> int:
        try:
            return int(parts[i])
        except Exception:
            return 0

    return {
        "spec_replicas": num(0),
        "status_replicas": num(1),
        "ready_replicas": num(2),
        "unavailable_replicas": num(3),
    }


def hpa_state() -> dict:
    raw = sh(
        [
            "kubectl", "-n", NS, "get", "hpa", "keda-hpa-llm-worker",
            "-o", "jsonpath={.status.desiredReplicas} "
                  "{.status.currentReplicas}",
        ]
    )
    parts = (raw or "").split()

    def num(i: int) -> int:
        try:
            return int(parts[i])
        except Exception:
            return 0

    return {"hpa_desired": num(0), "hpa_current": num(1)}


def worker_cpu() -> dict:
    """Total + mean CPU millicores across worker pods, via metrics-server."""
    raw = sh(
        ["kubectl", "-n", NS, "top", "pods", "--no-headers", "-l",
         "app=llm-worker"],
        timeout=15,
    )
    total_m = 0
    pods = 0
    for line in raw.splitlines():
        cols = line.split()
        if len(cols) < 2:
            continue
        cpu = cols[1]
        if cpu.endswith("m"):
            try:
                total_m += int(cpu[:-1])
                pods += 1
            except ValueError:
                pass
    return {
        "worker_cpu_total_m": total_m,
        "worker_cpu_pods": pods,
        "worker_cpu_mean_m": round(total_m / pods, 1) if pods else 0.0,
    }


FIELDS = [
    "t", "queue_depth", "in_flight", "backlog", "completed",
    "active_workers", "oldest_queued_wait_s", "avg_queue_wait_s",
    "spec_replicas", "status_replicas", "ready_replicas",
    "unavailable_replicas", "hpa_desired", "hpa_current",
    "worker_cpu_total_m", "worker_cpu_pods", "worker_cpu_mean_m",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default="timeline.csv")
    ap.add_argument(
        "--cpu-every",
        type=int,
        default=3,
        help="sample kubectl top every N ticks (it is slow)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    t0 = time.time()
    tick = 0
    last_cpu = {"worker_cpu_total_m": 0, "worker_cpu_pods": 0,
                "worker_cpu_mean_m": 0.0}

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        while time.time() - t0 < args.duration:
            row = {"t": round(time.time() - t0, 2)}
            row.update(broker_metrics(args.base))
            row.update(deploy_state())
            row.update(hpa_state())
            if tick % args.cpu_every == 0:
                last_cpu = worker_cpu()
            row.update(last_cpu)

            writer.writerow({k: row.get(k, "") for k in FIELDS})
            fh.flush()

            print(
                f"t={row['t']:6.1f}s q={row.get('queue_depth', '-'):>4} "
                f"inflight={row.get('in_flight', '-'):>3} "
                f"desired={row.get('hpa_desired', '-'):>2} "
                f"replicas={row.get('status_replicas', '-'):>2} "
                f"ready={row.get('ready_replicas', '-'):>2} "
                f"cpu={row.get('worker_cpu_mean_m', '-'):>6}m "
                f"done={row.get('completed', '-'):>4}",
                flush=True,
            )
            tick += 1
            time.sleep(args.interval)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
