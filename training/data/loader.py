"""Grain input pipeline over the packed token corpus.

Grain gives us deterministic, checkpointable, sharded data loading. The packed
`.bin` is a flat token stream; we expose it as a random-access data source of
fixed-length windows and let Grain handle shuffling, batching, and multiprocess
prefetch.

Each record is one length-(seq_len+1) window sliced at a window-aligned offset;
we return `input_ids = window[:-1]` and `labels = window[1:]` (next-token targets).
Because documents were packed contiguously with `<|endoftext|>` separators, windows
may span document boundaries — the standard, efficient packing scheme.
"""

from __future__ import annotations

import json
from pathlib import Path

import grain.python as grain
import numpy as np


def load_meta(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json", "r") as f:
        return json.load(f)


class PackedTokenSource(grain.RandomAccessDataSource):
    """Random-access view of `bin_path` as non-overlapping (seq_len+1)-token windows.

    The memmap is opened LAZILY, once per process. Grain pickles the data source
    into each worker process, and pickling an open np.memmap silently materializes
    the whole file as an in-memory array — for a multi-GB corpus that would copy
    the entire dataset into every worker. So we only ship the path + dtype and let
    each worker open its own memmap on first access.
    """

    def __init__(self, bin_path: str | Path, dtype: str, seq_len: int):
        self._path = str(bin_path)
        self._dtype = np.dtype(dtype)
        self._seq_len = seq_len
        self._data: np.memmap | None = None  # opened per-process in __getitem__
        # Window count from the file size alone (no need to open the memmap here);
        # each window needs seq_len+1 tokens (input + shifted label).
        total = Path(bin_path).stat().st_size // self._dtype.itemsize
        self._n = max(0, (total - 1) // seq_len)
        if self._n == 0:
            raise ValueError(
                f"{bin_path}: only {total} tokens — need > seq_len ({seq_len}). "
                "Prepare more data or lower seq_len."
            )

    def __getstate__(self) -> dict:
        # Drop the open memmap before pickling (see class docstring).
        return {**self.__dict__, "_data": None}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if self._data is None:
            self._data = np.memmap(self._path, dtype=self._dtype, mode="r")
        start = idx * self._seq_len
        window = np.asarray(
            self._data[start : start + self._seq_len + 1], dtype=np.int32
        )
        return {"input_ids": window[:-1], "labels": window[1:]}


def make_loader(
    data_dir: str | Path,
    split: str,
    seq_len: int,
    batch_size: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    num_epochs: int | None = None,
    worker_count: int = 2,
    drop_remainder: bool = True,
) -> grain.DataLoader:
    """Build a Grain DataLoader yielding dicts of stacked int32 numpy arrays.

    split: "train" or "val" (reads {split}.bin). num_epochs=None loops forever
    (used for training); pass an int (e.g. 1) for a finite eval pass.
    """
    meta = load_meta(data_dir)
    source = PackedTokenSource(
        Path(data_dir) / f"{split}.bin", meta["dtype"], seq_len
    )
    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )
    ops = [grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder)]
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=worker_count,
    )
