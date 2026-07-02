"""Turn a Hugging Face dataset into an iterator of plain-text code documents.

Kept tiny and shared by both the tokenizer trainer and the packer. A row becomes a
document either by reading a single `text_key` column, or by filling a `template`
string with the row's columns (e.g. combining an instruction dataset's `problem`
and `solution` fields). `.format` only substitutes the template's own placeholders,
so code containing literal braces in the substituted values is safe.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from training.config import DataSourceConfig


def _row_to_text(row: dict, cfg: DataSourceConfig) -> str | None:
    if cfg.template is not None:
        try:
            return cfg.template.format(**row)
        except (KeyError, IndexError):
            return None
    if cfg.text_key is not None:
        val = row.get(cfg.text_key)
        return val if isinstance(val, str) else None
    raise ValueError("DataSourceConfig needs either `text_key` or `template`.")


def iter_documents(cfg: DataSourceConfig) -> Iterator[str]:
    """Yield up to `max_documents` non-empty text documents from the HF source."""
    ds = load_dataset(
        cfg.path,
        name=cfg.name,
        data_dir=cfg.data_dir,
        split=cfg.split,
        streaming=cfg.streaming,
    )
    n = 0
    for row in ds:
        text = _row_to_text(row, cfg)
        if not text:
            continue
        yield text
        n += 1
        if cfg.max_documents is not None and n >= cfg.max_documents:
            break
