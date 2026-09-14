"""Measures how long each scaled-up replica took to become useful.

Reads the JSONL written by record_pods.py during the run, because scale-down
deletes the pods and a post-hoc `kubectl get pods` only sees the survivor.

Cold start is reported as creation -> Ready, which covers container start and
the simulated weight load. The autoscaler can create a pod in about a
second, but the replica cannot serve until this window has elapsed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def load_pods(path: Path) -> dict[str, dict]:
    """Last observation wins, but keep the first time we saw it ready."""
    pods: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row["pod"]
        cur = pods.setdefault(name, {"pod": name, "first_seen_t": row["t"]})
        cur["created"] = row.get("created") or cur.get("created")
        cur["scheduled_at"] = row.get("scheduled_at") or cur.get("scheduled_at")
        if row.get("ready_at") and not cur.get("ready_at"):
            cur["ready_at"] = row["ready_at"]
            cur["ready_seen_t"] = row["t"]
    return pods


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="logs/pods-lifecycle.jsonl")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"no lifecycle recording at {path}")
        return

    pods = load_pods(path)
    rows = []
    for p in pods.values():
        created = parse_ts(p.get("created"))
        ready_at = parse_ts(p.get("ready_at"))
        if not created or not ready_at:
            continue
        scheduled = parse_ts(p.get("scheduled_at"))
        rows.append(
            {
                "pod": p["pod"],
                "created": created,
                "ready": ready_at,
                "created_to_ready_s": (ready_at - created).total_seconds(),
                "sched_to_ready_s": (
                    (ready_at - scheduled).total_seconds() if scheduled else None
                ),
            }
        )

    if not rows:
        print("no pod reached Ready in the recording")
        return

    rows.sort(key=lambda r: (r["created"], r["pod"]))
    t0 = rows[0]["created"]

    print(
        f"{'pod':32s} {'created(+s)':>11s} {'ready(+s)':>10s} "
        f"{'created->ready':>15s}"
    )
    for r in rows:
        print(
            f"{r['pod']:32s} "
            f"{(r['created'] - t0).total_seconds():11.0f} "
            f"{(r['ready'] - t0).total_seconds():10.0f} "
            f"{r['created_to_ready_s']:15.0f}"
        )

    # The replica that existed before the spike is not a cold start.
    scaled = [r for r in rows if (r["created"] - t0).total_seconds() > 1]
    values = [r["created_to_ready_s"] for r in rows]

    print()
    print(f"pods observed:                {len(rows)}")
    print(f"pods added by the autoscaler:  {len(scaled)}")
    print(
        f"cold start (all) min/mean/max: "
        f"{min(values):.0f}s / {sum(values) / len(values):.1f}s / "
        f"{max(values):.0f}s"
    )
    if scaled:
        sv = [r["created_to_ready_s"] for r in scaled]
        print(
            f"cold start (scaled-up only):   "
            f"{min(sv):.0f}s / {sum(sv) / len(sv):.1f}s / {max(sv):.0f}s"
        )
    print(
        "note: STARTUP_DELAY_S=20 simulates weight loading, and the image was "
        "already cached on the node. A real 7B+ replica on a cold node also "
        "pays image pull and can take minutes."
    )


if __name__ == "__main__":
    main()
