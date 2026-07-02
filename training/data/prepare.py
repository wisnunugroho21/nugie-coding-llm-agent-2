"""Build a packed token corpus from a Hugging Face dataset.

Pipeline (OpenCoder Sec. 2.1 order): stream documents -> heuristic filter -> dedup
(exact + MinHash) -> tokenize -> optional FIM -> append <|endoftext|> -> pack into a
flat token stream written as a memory-mappable `.bin`. A small held-out `val.bin`
is split off at the document level. `meta.json` records everything the loader and
trainer need (dtype, token counts, special-token ids, vocab size).

Usage:
    python -m training.data.prepare --config configs/data_pretrain.yaml

Output (in cfg.out_dir):
    train.bin   flat uint16/uint32 token ids
    val.bin     held-out split
    meta.json   metadata
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from training.config import PrepareConfig
from training.data.dedup import Deduper
from training.data.fim import maybe_fim
from training.data.filters import CodeFilter
from training.data.hf_source import iter_documents
from training.data.tokenizer import CodeTokenizer


class BinWriter:
    """Appends token ids to a binary file in a fixed dtype, buffering for throughput."""

    def __init__(self, path: Path, dtype: np.dtype, flush_every: int = 1_000_000):
        self.path = path
        self.dtype = dtype
        self._buf: list[int] = []
        self._flush_every = flush_every
        self.n_tokens = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "wb")

    def add(self, ids: list[int]) -> None:
        self._buf.extend(ids)
        self.n_tokens += len(ids)
        if len(self._buf) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            np.asarray(self._buf, dtype=self.dtype).tofile(self._f)
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        self._f.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter, dedup, tokenize and pack a code corpus.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = PrepareConfig.load(args.config)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = CodeTokenizer(cfg.tokenizer)
    dtype = np.uint16 if tok.vocab_size < 2**16 else np.uint32
    rng = np.random.default_rng(cfg.seed)

    code_filter = CodeFilter() if cfg.filter.enabled else None
    deduper = (
        Deduper(
            exact=cfg.dedup.exact,
            minhash=cfg.dedup.minhash,
            threshold=cfg.dedup.minhash_threshold,
            num_perm=cfg.dedup.minhash_num_perm,
            ngram=cfg.dedup.minhash_ngram,
            seed=cfg.seed,
        )
        if (cfg.dedup.exact or cfg.dedup.minhash)
        else None
    )

    train_w = BinWriter(out_dir / "train.bin", dtype)
    val_w = BinWriter(out_dir / "val.bin", dtype)

    stats = {"seen": 0, "kept": 0, "rej_filter": 0, "rej_dedup": 0, "docs_val": 0}

    for text in tqdm(iter_documents(cfg.hf), desc="prepare", unit="doc"):
        stats["seen"] += 1
        if code_filter is not None and not code_filter.keep(text):
            stats["rej_filter"] += 1
            continue
        if deduper is not None and not deduper.add_if_new(text):
            stats["rej_dedup"] += 1
            continue

        ids = tok.encode(text)
        if cfg.fim.enabled:
            ids = maybe_fim(
                ids,
                tok.fim_prefix_id,
                tok.fim_middle_id,
                tok.fim_suffix_id,
                rng,
                rate=cfg.fim.rate,
                spm_rate=cfg.fim.spm_rate,
            )
        if cfg.append_eot:
            ids = ids + [tok.eot_id]

        if rng.random() < cfg.val_fraction:
            val_w.add(ids)
            stats["docs_val"] += 1
        else:
            train_w.add(ids)
        stats["kept"] += 1

    train_w.close()
    val_w.close()

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": tok.vocab_size,
        "train_tokens": train_w.n_tokens,
        "val_tokens": val_w.n_tokens,
        "eot_id": tok.eot_id,
        "pad_id": tok.pad_id,
        "fim_ids": {
            "prefix": tok.fim_prefix_id,
            "middle": tok.fim_middle_id,
            "suffix": tok.fim_suffix_id,
        },
        "tokenizer": str(cfg.tokenizer),
        "source": {
            "path": cfg.hf.path,
            "name": cfg.hf.name,
            "data_dir": cfg.hf.data_dir,
            "split": cfg.hf.split,
        },
        "stats": stats,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[prepare] {json.dumps(stats)}")
    print(
        f"[prepare] train={train_w.n_tokens:,} tokens  val={val_w.n_tokens:,} tokens "
        f"dtype={meta['dtype']}  -> {out_dir}"
    )


if __name__ == "__main__":
    main()
