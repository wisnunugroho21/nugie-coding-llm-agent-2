"""Byte-level BPE tokenizer for code, trained from scratch with 🤗 tokenizers.

We train our own BPE (not reuse an existing vocab) because the whole project is
"from scratch". A byte-level BPE never emits <unk> — every byte is representable —
which matters for code (arbitrary identifiers, unicode strings, binary-ish blobs).

Special tokens (ids 0..) follow the code-LLM convention used by StarCoder / OpenCoder:
document boundary + Fill-in-the-Middle sentinels + repo/file separators. Their ids
are fixed by listing them first in the trainer's special_tokens, so downstream code
can rely on `CodeTokenizer.eot_id`, `.fim_prefix_id`, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|endoftext|>",   # 0: document / sequence boundary
    "<|pad|>",         # 1: padding (unused when packing, kept for batched decode)
    "<|fim_prefix|>",  # 2: FIM prefix sentinel
    "<|fim_middle|>",  # 3: FIM middle sentinel
    "<|fim_suffix|>",  # 4: FIM suffix sentinel
    "<|repo_name|>",   # 5: repo-context sentinel (repo-level packing)
    "<|file_sep|>",    # 6: file separator (repo-level packing)
]


def train_tokenizer(
    text_iter: Iterable[str],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer and save it to `out_path` (a .json file)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


class CodeTokenizer:
    """Thin wrapper exposing encode/decode and the fixed special-token ids."""

    def __init__(self, path: str | Path):
        self.tok = Tokenizer.from_file(str(path))
        ids = {t: self.tok.token_to_id(t) for t in SPECIAL_TOKENS}
        missing = [t for t, i in ids.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer {path} is missing special tokens: {missing}")
        self.eot_id = ids["<|endoftext|>"]
        self.pad_id = ids["<|pad|>"]
        self.fim_prefix_id = ids["<|fim_prefix|>"]
        self.fim_middle_id = ids["<|fim_middle|>"]
        self.fim_suffix_id = ids["<|fim_suffix|>"]
        self.repo_name_id = ids["<|repo_name|>"]
        self.file_sep_id = ids["<|file_sep|>"]

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)
