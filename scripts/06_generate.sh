#!/usr/bin/env bash
# Sample from a trained checkpoint. Override MODEL / prompt via env or args.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
MODEL="${MODEL:-checkpoints/anneal/model_3000.msgpack}"
python -m training.generate \
  --config configs/anneal.yaml \
  --model "$MODEL" \
  --prompt "def quicksort(arr):" \
  --max-new-tokens 128 --temperature 0.8 --top-p 0.95
