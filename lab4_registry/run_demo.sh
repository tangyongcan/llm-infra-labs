#!/usr/bin/env bash
# Train v1 + v2, promote v2 to Production. Rollback is a separate command
# so it can be timed and screenshotted on its own.
set -euo pipefail
cd "$(dirname "$0")"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$PWD/tracking.db}"

echo "==> v1  lr=2e-4  lora_r=8"
python train_and_register.py train --run-name v1 --lr 2e-4 --lora-r 8

echo "==> v2  lr=1e-4  lora_r=16  (promote to Production)"
python train_and_register.py train --run-name v2 --lr 1e-4 --lora-r 16 --promote

echo "==> registry after v2=Production"
python train_and_register.py status
