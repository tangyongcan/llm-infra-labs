#!/usr/bin/env bash
# Brings up everything the load demo needs: cluster, metrics-server, image,
# KEDA, the deployments and the ScaledObject. Safe to re-run.
#
# Prereqs on PATH: minikube, kubectl, helm, docker (daemon reachable).
set -euo pipefail
cd "$(dirname "$0")"

NS=llmserving
IMAGE=llmlab/queue-app:v1

step() { echo; echo "==== $* ===="; }

step "1/6 cluster"
if minikube status >/dev/null 2>&1; then
  echo "minikube already running"
else
  # First run downloads ~650MB (kicbase + preloaded images) and can take
  # over 10 minutes on a slow link.
  minikube start --driver=docker --cpus=6 --memory=6144
fi
kubectl config use-context minikube

step "2/6 metrics-server"
# Only used to *observe* CPU for the comparison; nothing scales on it.
minikube addons enable metrics-server

step "3/6 build image inside minikube's docker"
eval "$(minikube docker-env)"
docker build -t "$IMAGE" ./app
eval "$(minikube docker-env -u)"

step "4/6 KEDA"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace --wait --timeout 10m
kubectl -n keda get pods

step "5/6 broker + worker"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deployment.yaml
kubectl -n "$NS" rollout status deploy/broker --timeout=180s
kubectl -n "$NS" rollout status deploy/llm-worker --timeout=180s

step "6/6 ScaledObject"
kubectl apply -f scaledobject.yaml
sleep 15
kubectl -n "$NS" get scaledobject,hpa

echo
echo "SETUP COMPLETE. Now run: ./run_load_demo.sh"
