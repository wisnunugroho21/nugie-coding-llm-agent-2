#!/usr/bin/env bash
# Download, filter, dedup, tokenize and pack the Stage-2 annealing corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python -m training.data.prepare --config configs/data_anneal.yaml
