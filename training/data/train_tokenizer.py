"""Entry point: train the byte-level BPE tokenizer from a data source.

Usage:
    python -m training.data.train_tokenizer --data-config configs/data_pretrain.yaml \
        --out tokenizers/code_bpe.json --vocab-size 32000 --max-docs 40000

The corpus for tokenizer training is the same HF source used for pretraining (or a
capped prefix of it), so the vocabulary matches the data distribution.
"""

from __future__ import annotations

import argparse
import itertools

from training.config import PrepareConfig
from training.data.hf_source import iter_documents
from training.data.tokenizer import train_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a byte-level BPE code tokenizer.")
    ap.add_argument("--data-config", required=True, help="prepare/data YAML (uses its `hf:` block)")
    ap.add_argument("--out", default="tokenizers/code_bpe.json")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--max-docs", type=int, default=40000,
                    help="cap documents used for tokenizer training (speed)")
    args = ap.parse_args()

    cfg = PrepareConfig.load(args.data_config)
    docs = iter_documents(cfg.hf)
    if args.max_docs:
        docs = itertools.islice(docs, args.max_docs)

    print(f"[tokenizer] training vocab_size={args.vocab_size} from {cfg.hf.path} ...")
    tok = train_tokenizer(
        docs, args.out, vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    print(f"[tokenizer] saved {args.out} (vocab_size={tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
