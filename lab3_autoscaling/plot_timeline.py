"""Plots the autoscaling timeline captured by monitor.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    val = 0.0
                cols.setdefault(k, []).append(val)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="timeline.csv")
    ap.add_argument("--out", default="hpa_scaling_timeline.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    d = read(Path(args.csv))
    t = d["t"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, d["queue_depth"], color="#dc2626", linewidth=2,
            label="queue_depth (KEDA scaling signal)")
    ax.plot(t, d["in_flight"], color="#f59e0b", linewidth=1.5,
            label="in_flight (being served)")
    ax.set_ylabel("requests")
    ax.set_title("Queue depth: reacts on the first unserved request")
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.step(t, d["hpa_desired"], where="post", color="#2563eb", linewidth=2,
            label="HPA desiredReplicas")
    ax.step(t, d["status_replicas"], where="post", color="#7c3aed",
            linewidth=1.6, linestyle="--", label="pods created")
    ax.step(t, d["ready_replicas"], where="post", color="#16a34a",
            linewidth=2, label="pods READY (after warmup)")
    ax.axhline(8, color="#94a3b8", linestyle=":", linewidth=1)
    ax.text(t[0], 8.1, "maxReplicas=8", fontsize=8, color="#64748b")
    ax.set_ylabel("replicas")
    ax.set_title(
        "The gap between 'pods created' and 'pods READY' is the cold start"
    )
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, d["worker_cpu_mean_m"], color="#0891b2", linewidth=2,
            label="worker CPU, mean per pod (millicores)")
    # HPA's default CPU target is 80% of the 100m request = 80m.
    ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.text(
        t[0], 82,
        "a CPU-based HPA at 80% of a 100m request would trigger here",
        fontsize=8, color="#dc2626",
    )
    ax.set_ylabel("millicores")
    ax.set_xlabel("seconds since monitor start")
    ax.set_title(
        "CPU utilisation over the same spike: flat, and never crosses the line"
    )
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        args.title
        or "Lab 3: KEDA scaling on queue depth (minReplicas=1, maxReplicas=8)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
