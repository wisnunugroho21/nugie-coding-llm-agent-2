#!/usr/bin/env bash
# Stage-1: general pretraining. Resume with:  scripts/04_pretrain.sh --resume checkpoints/pretrain
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python -m training.train --config configs/pretrain.yaml "$@"
