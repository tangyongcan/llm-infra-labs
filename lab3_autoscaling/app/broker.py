"""Queue broker: stands in for the scheduler/waiting-queue of an LLM server.

Runs as a single replica and owns the only piece of state that matters for
autoscaling decisions: how many requests are waiting to be served.

This is deliberately the shape of a real serving stack (vLLM has a waiting
queue in front of a fixed set of GPU workers), because the whole point of
the lab is that *queue depth*, not CPU%, is the signal you scale on.

Endpoints:
    POST /enqueue?n=&work_ms=   admit n synthetic requests
    POST /lease                 a worker claims one request
    POST /complete              a worker reports a request finished
    GET  /metrics               JSON for the KEDA metrics-api scaler
    GET  /metrics/prometheus    same numbers in Prometheus text format
    GET  /stats                 human-readable snapshot
    POST /reset                 clear all state between demo runs
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time
from collections import deque

from fastapi import FastAPI, Response

app = FastAPI(title="llm-queue-broker")

_ids = itertools.count(1)
_lock = asyncio.Lock()

# Requests admitted but not yet claimed by a worker.
pending: deque[dict] = deque()
# Claimed but not yet completed, keyed by request id.
in_flight: dict[int, dict] = {}

completed_count = 0
total_wait_s = 0.0
total_service_s = 0.0
worker_last_seen: dict[str, float] = {}
started_at = time.time()

WORKER_STALE_S = float(os.getenv("WORKER_STALE_S", "20"))


def active_workers() -> int:
    now = time.time()
    return sum(1 for ts in worker_last_seen.values() if now - ts <= WORKER_STALE_S)


@app.post("/enqueue")
async def enqueue(n: int = 1, work_ms: int = 1000):
    now = time.time()
    async with _lock:
        for _ in range(n):
            pending.append(
                {"id": next(_ids), "work_ms": work_ms, "admitted_at": now}
            )
        depth = len(pending)
    return {"admitted": n, "queue_depth": depth}


@app.post("/lease")
async def lease(worker: str = "unknown"):
    now = time.time()
    async with _lock:
        worker_last_seen[worker] = now
        if not pending:
            return Response(status_code=204)
        item = pending.popleft()
        item["leased_at"] = now
        item["worker"] = worker
        in_flight[item["id"]] = item
    return {
        "id": item["id"],
        "work_ms": item["work_ms"],
        "waited_s": now - item["admitted_at"],
    }


@app.post("/complete")
async def complete(id: int, worker: str = "unknown"):
    global completed_count, total_wait_s, total_service_s
    now = time.time()
    async with _lock:
        worker_last_seen[worker] = now
        item = in_flight.pop(id, None)
        if item is None:
            return {"ok": False, "reason": "unknown id"}
        completed_count += 1
        total_wait_s += item["leased_at"] - item["admitted_at"]
        total_service_s += now - item["admitted_at"]
    return {"ok": True, "completed": completed_count}


def snapshot() -> dict:
    depth = len(pending)
    oldest_wait = (
        time.time() - pending[0]["admitted_at"] if depth else 0.0
    )
    return {
        # The scaling signal. KEDA reads exactly this field.
        "queue_depth": depth,
        "in_flight": len(in_flight),
        "backlog": depth + len(in_flight),
        "completed": completed_count,
        "active_workers": active_workers(),
        "oldest_queued_wait_s": round(oldest_wait, 3),
        "avg_queue_wait_s": round(
            total_wait_s / completed_count if completed_count else 0.0, 3
        ),
        "avg_e2e_s": round(
            total_service_s / completed_count if completed_count else 0.0, 3
        ),
        "uptime_s": round(time.time() - started_at, 1),
    }


@app.get("/metrics")
async def metrics():
    return snapshot()


@app.get("/metrics/prometheus")
async def metrics_prometheus():
    s = snapshot()
    lines = [
        "# HELP llm_queue_depth Requests admitted but not yet claimed.",
        "# TYPE llm_queue_depth gauge",
        f"llm_queue_depth {s['queue_depth']}",
        "# HELP llm_in_flight Requests currently being served.",
        "# TYPE llm_in_flight gauge",
        f"llm_in_flight {s['in_flight']}",
        "# HELP llm_backlog queue_depth + in_flight.",
        "# TYPE llm_backlog gauge",
        f"llm_backlog {s['backlog']}",
        "# HELP llm_active_workers Workers that leased or completed recently.",
        "# TYPE llm_active_workers gauge",
        f"llm_active_workers {s['active_workers']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


@app.get("/stats")
async def stats():
    return snapshot()


@app.post("/reset")
async def reset():
    global completed_count, total_wait_s, total_service_s, started_at
    async with _lock:
        pending.clear()
        in_flight.clear()
        worker_last_seen.clear()
        completed_count = 0
        total_wait_s = 0.0
        total_service_s = 0.0
        started_at = time.time()
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
