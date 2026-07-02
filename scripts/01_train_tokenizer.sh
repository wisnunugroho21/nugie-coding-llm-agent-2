#!/usr/bin/env bash
# Train the byte-level BPE tokenizer from the pretraining data source.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python -m training.data.train_tokenizer \
  --data-config configs/data_pretrain.yaml \
  --out tokenizers/code_bpe.json \
  --vocab-size 32000 \
  --max-docs 40000
