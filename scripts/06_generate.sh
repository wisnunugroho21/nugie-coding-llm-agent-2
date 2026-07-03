#!/usr/bin/env bash
# Sample from a trained checkpoint. Override MODEL / prompt via env or args.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
# Default matches configs/anneal.yaml max_steps (H200 preset). Override via
# `MODEL=... scripts/06_generate.sh` if you changed max_steps or use the T4 preset.
MODEL="${MODEL:-checkpoints/anneal/model_4000.msgpack}"
python -m training.generate \
  --config configs/anneal.yaml \
  --model "$MODEL" \
  --prompt "def quicksort(arr):" \
  --max-new-tokens 128 --temperature 0.8 --top-p 0.95
