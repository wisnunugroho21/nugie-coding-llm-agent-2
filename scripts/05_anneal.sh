#!/usr/bin/env bash
# Stage-2: annealing, warm-started from the final pretrain checkpoint.
# Pass the pretrain model checkpoint via --init-from (or set it in configs/anneal.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
# Default matches configs/pretrain.yaml max_steps (H200 preset). Pass a different
# checkpoint path as $1 if you changed max_steps or use the T4 preset.
INIT="${1:-checkpoints/pretrain/model_38000.msgpack}"
python -m training.train --config configs/anneal.yaml --init-from "$INIT"
