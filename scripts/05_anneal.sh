#!/usr/bin/env bash
# Stage-2: annealing, warm-started from the final pretrain checkpoint.
# Pass the pretrain model checkpoint via --init-from (or set it in configs/anneal.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
INIT="${1:-checkpoints/pretrain/model_20000.msgpack}"
python -m training.train --config configs/anneal.yaml --init-from "$INIT"
