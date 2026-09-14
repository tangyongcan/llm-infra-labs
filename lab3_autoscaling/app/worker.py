"""Worker: stands in for one GPU serving replica. This is what KEDA scales.

Two properties are modelled on purpose, because they are what make LLM
autoscaling harder than stateless CPU autoscaling:

1. STARTUP_DELAY_S - a replica is not useful the moment the pod schedules.
   A real replica has to pull the image and load tens of GB of weights onto
   the GPU. Readiness is gated on that, so scale-up help arrives late.
2. WORKER_CONCURRENCY - a replica serves a bounded number of concurrent
   requests (its batch capacity), not "as many as CPU allows". Past that
   bound extra load turns straight into queue depth, not into CPU%.

The worker also burns a little CPU while "decoding" so that CPU utilisation
is observable and can be compared against queue depth.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time

import httpx
from fastapi import FastAPI, Response

BROKER_URL = os.getenv("BROKER_URL", "http://broker:8000")
WORKER_ID = os.getenv("WORKER_ID") or socket.gethostname()
STARTUP_DELAY_S = float(os.getenv("STARTUP_DELAY_S", "20"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "0.25"))
# Fraction of the simulated decode time spent spinning the CPU. Decode on a
# real GPU is memory-bandwidth-bound, so the host CPU stays mostly idle;
# keeping this low is what reproduces the "CPU% barely moves" effect.
CPU_BURN_RATIO = float(os.getenv("CPU_BURN_RATIO", "0.05"))

app = FastAPI(title="llm-worker")

state = {
    "ready": False,
    "started_at": time.time(),
    "ready_at": None,
    "served": 0,
    "busy": 0,
}


def burn_cpu(seconds: float) -> None:
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < end:
        x += 1.000000001
    return


async def serve_one(client: httpx.AsyncClient, item: dict) -> None:
    work_s = item["work_ms"] / 1000.0
    state["busy"] += 1
    try:
        burn_s = work_s * CPU_BURN_RATIO
        if burn_s > 0:
            await asyncio.to_thread(burn_cpu, burn_s)
        # The rest is "waiting on memory bandwidth": no CPU consumed.
        await asyncio.sleep(max(0.0, work_s - burn_s))
        await client.post(
            f"{BROKER_URL}/complete",
            params={"id": item["id"], "worker": WORKER_ID},
        )
        state["served"] += 1
    finally:
        state["busy"] -= 1


async def lease_loop() -> None:
    """One coroutine per concurrency slot: claim work, serve it, repeat."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            if not state["ready"]:
                await asyncio.sleep(0.5)
                continue
            try:
                r = await client.post(
                    f"{BROKER_URL}/lease", params={"worker": WORKER_ID}
                )
                if r.status_code == 204:
                    await asyncio.sleep(POLL_INTERVAL_S)
                    continue
                r.raise_for_status()
                await serve_one(client, r.json())
            except Exception:
                await asyncio.sleep(1.0)


async def warmup() -> None:
    """Simulate pulling weights onto the device before serving traffic."""
    print(
        f"[{WORKER_ID}] loading model, {STARTUP_DELAY_S:.0f}s simulated warmup",
        flush=True,
    )
    await asyncio.sleep(STARTUP_DELAY_S)
    state["ready"] = True
    state["ready_at"] = time.time()
    print(
        f"[{WORKER_ID}] ready after "
        f"{state['ready_at'] - state['started_at']:.1f}s",
        flush=True,
    )


@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(warmup())
    for _ in range(WORKER_CONCURRENCY):
        asyncio.create_task(lease_loop())


@app.get("/healthz")
async def healthz():
    """Liveness: the process is up even while weights are still loading."""
    return {"ok": True, "ready": state["ready"]}


@app.get("/readyz")
async def readyz():
    """Readiness: gated on warmup, exactly like a real model-loading replica."""
    if not state["ready"]:
        return Response(status_code=503, content="loading model")
    return {"ok": True}


@app.get("/stats")
async def stats():
    return {
        "worker": WORKER_ID,
        "ready": state["ready"],
        "startup_delay_s": STARTUP_DELAY_S,
        "time_to_ready_s": (
            round(state["ready_at"] - state["started_at"], 2)
            if state["ready_at"]
            else None
        ),
        "concurrency": WORKER_CONCURRENCY,
        "busy_slots": state["busy"],
        "served": state["served"],
        "uptime_s": round(time.time() - state["started_at"], 1),
    }
