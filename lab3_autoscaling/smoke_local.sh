#!/usr/bin/env bash
# Validates the broker/worker queue mechanics without Kubernetes.
# Useful on its own, and it de-risks the cluster demo: if the queue does not
# drain here, it will not drain in a pod either.
set -uo pipefail
cd "$(dirname "$0")/app"

PORT=${PORT:-18100}
WORKERS=${WORKERS:-2}
BASE="http://127.0.0.1:$PORT"
PIDS=()

cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup EXIT

echo "== starting broker on $PORT =="
uvicorn broker:app --host 127.0.0.1 --port "$PORT" --log-level warning &
PIDS+=($!)

for i in $(seq 1 20); do
  curl -sf "$BASE/healthz" >/dev/null && break
  sleep 0.5
done

echo "== starting $WORKERS workers (5s simulated warmup) =="
for i in $(seq 1 "$WORKERS"); do
  BROKER_URL="$BASE" WORKER_ID="local-w$i" STARTUP_DELAY_S=5 \
    WORKER_CONCURRENCY=2 CPU_BURN_RATIO=0.05 \
    uvicorn worker:app --host 127.0.0.1 --port $((PORT + i)) \
      --log-level warning &
  PIDS+=($!)
done

echo "== admitting 20 requests x 500ms =="
curl -s -X POST "$BASE/enqueue?n=20&work_ms=500" && echo

for i in $(seq 1 24); do
  echo "t=+$((i))s $(curl -s $BASE/metrics)"
  sleep 1
done

echo
echo "== worker stats =="
for i in $(seq 1 "$WORKERS"); do
  curl -s "http://127.0.0.1:$((PORT + i))/stats" && echo
done

DONE=$(curl -s "$BASE/metrics" | python -c "import sys,json;print(json.load(sys.stdin)['completed'])")
echo
if [ "$DONE" -eq 20 ]; then
  echo "SMOKE PASS: all 20 requests drained"
else
  echo "SMOKE FAIL: completed=$DONE / 20"
  exit 1
fi
