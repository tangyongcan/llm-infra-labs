#!/usr/bin/env bash
# Fires the load spike and captures the autoscaling evidence.
# Assumes the cluster, KEDA, the image and the manifests are already in place
# (see setup_cluster.sh).
set -uo pipefail
cd "$(dirname "$0")"

NS=llmserving
LOGS=logs
PF_PORT=${PF_PORT:-18000}
TOTAL=${TOTAL:-200}
WORK_MS=${WORK_MS:-3000}
MONITOR_S=${MONITOR_S:-400}
WARMUP_OBSERVE_S=${WARMUP_OBSERVE_S:-15}

mkdir -p "$LOGS"
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
}
trap cleanup EXIT

echo "==== scale back to the pre-spike baseline ===="
# Start every run from 1 replica so the 1 -> 8 story is honest.
kubectl -n "$NS" scale deploy/llm-worker --replicas=1 >/dev/null 2>&1 || true
kubectl -n "$NS" rollout status deploy/llm-worker --timeout=180s

echo "==== port-forward broker ===="
kubectl -n "$NS" port-forward svc/broker "$PF_PORT":8000 \
  >"$LOGS/port-forward.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PF_PORT/healthz" >/dev/null && break
  sleep 1
done
curl -s -X POST "http://127.0.0.1:$PF_PORT/reset" >/dev/null
echo "broker: $(curl -s http://127.0.0.1:$PF_PORT/metrics)"

echo "==== start watchers ===="
: >"$LOGS/hpa-watch.log"
kubectl -n "$NS" get hpa -w >"$LOGS/hpa-watch.log" 2>&1 &
PIDS+=($!)
kubectl -n "$NS" get pods -w >"$LOGS/pods-watch.log" 2>&1 &
PIDS+=($!)

python monitor.py --base "http://127.0.0.1:$PF_PORT" \
  --duration "$MONITOR_S" --out "$LOGS/timeline.csv" \
  >"$LOGS/monitor.log" 2>&1 &
MON_PID=$!
PIDS+=($MON_PID)

# Scale-down deletes pods, so per-pod readiness has to be recorded live or
# it is gone by the time the run ends.
python record_pods.py --duration "$MONITOR_S" \
  --out "$LOGS/pods-lifecycle.jsonl" >"$LOGS/record-pods.log" 2>&1 &
PIDS+=($!)

echo "==== observing idle baseline for ${WARMUP_OBSERVE_S}s ===="
sleep "$WARMUP_OBSERVE_S"

echo "==== firing burst: $TOTAL requests x ${WORK_MS}ms ===="
python loadgen.py --base "http://127.0.0.1:$PF_PORT" \
  --shape burst --total "$TOTAL" --work-ms "$WORK_MS" \
  | tee "$LOGS/loadgen.log"

echo "==== monitoring for the rest of ${MONITOR_S}s (scale up, drain, scale down) ===="
wait $MON_PID

echo "==== final state ===="
kubectl -n "$NS" get hpa,scaledobject,pods | tee "$LOGS/final-state.log"
kubectl -n "$NS" logs -l app=llm-worker --tail=3 --prefix \
  >"$LOGS/worker-logs.log" 2>&1 || true

echo "==== cold start analysis ===="
python analyze_cold_start.py --jsonl "$LOGS/pods-lifecycle.jsonl" \
  | tee "$LOGS/cold-start.log"

echo "==== plot ===="
python plot_timeline.py --csv "$LOGS/timeline.csv" \
  --out hpa_scaling_timeline.png

echo
echo "DONE. artifacts: $LOGS/ + hpa_scaling_timeline.png"
