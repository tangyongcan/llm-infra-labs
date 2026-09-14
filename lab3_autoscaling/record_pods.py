"""Polls worker pod lifecycle timestamps into a JSONL while the demo runs.

Needed because scale-down deletes the pods: by the time the run finishes,
`kubectl get pods` can no longer tell you how long the scaled-up replicas
took to become ready. Recording as we go keeps that evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def poll(namespace: str, selector: str) -> list[dict]:
    out = subprocess.run(
        ["kubectl", "-n", namespace, "get", "pods", "-l", selector,
         "-o", "json"],
        capture_output=True, text=True, timeout=30,
    )
    if not out.stdout.strip():
        return []
    items = json.loads(out.stdout).get("items", [])
    rows = []
    for pod in items:
        conds = {
            c["type"]: c
            for c in pod.get("status", {}).get("conditions", [])
        }
        ready = conds.get("Ready", {})
        scheduled = conds.get("PodScheduled", {})
        rows.append(
            {
                "pod": pod["metadata"]["name"],
                "created": pod["metadata"].get("creationTimestamp"),
                "phase": pod.get("status", {}).get("phase"),
                "ready": ready.get("status") == "True",
                "ready_at": (
                    ready.get("lastTransitionTime")
                    if ready.get("status") == "True"
                    else None
                ),
                "scheduled_at": (
                    scheduled.get("lastTransitionTime")
                    if scheduled.get("status") == "True"
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="llmserving")
    ap.add_argument("--selector", default="app=llm-worker")
    ap.add_argument("--duration", type=float, default=400)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default="logs/pods-lifecycle.jsonl")
    args = ap.parse_args()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with path.open("w") as fh:
        while time.time() - t0 < args.duration:
            for row in poll(args.namespace, args.selector):
                row["t"] = round(time.time() - t0, 2)
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
