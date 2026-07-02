"""Config objects for the training pipeline.

All configs are plain dataclasses loaded from YAML so an experiment is fully
described by a few small files under `configs/`. Three concerns are kept separate:

  * model   -> a `configs/model_*.yaml` mapped 1:1 onto `KimiLinearConfig`.
  * data    -> a `configs/data_*.yaml` consumed by `training.data.prepare`.
  * train   -> a `configs/*.yaml` (pretrain/anneal) with the optimizer schedule.

The train YAML references the model YAML by path (`model_config:`), so the two
training phases can share one architecture file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kimi_linear_gdn2 import KimiLinearConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def model_config_from_dict(d: dict[str, Any]) -> KimiLinearConfig:
    """Build a KimiLinearConfig, keeping only keys it actually declares (so a YAML
    can carry extra annotations without breaking construction)."""
    fields = {f.name for f in dataclasses.fields(KimiLinearConfig)}
    kept = {k: v for k, v in d.items() if k in fields}
    unknown = set(d) - fields
    if unknown:
        # Non-fatal: surface typos in the model YAML instead of silently ignoring them.
        print(f"[config] ignoring unknown model keys: {sorted(unknown)}")
    return KimiLinearConfig(**kept)


def load_model_config(path: str | Path) -> KimiLinearConfig:
    return model_config_from_dict(_read_yaml(path))


# --------------------------------------------------------------------------- #
#  Training-phase config (optimizer schedule, logging, checkpointing).
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    phase: str = "pretrain"  # "pretrain" or "anneal" — affects only defaults/logging
    out_dir: str = "checkpoints/run"
    seed: int = 0

    # data
    data_dir: str = "data/pretrain"  # dir holding train.bin / val.bin / meta.json
    seq_len: int = 512  # MUST be a multiple of the GDN-2 chunk size

    # batch / accumulation
    batch_size: int = 8  # micro-batch fed to one forward
    grad_accum: int = 4  # optimizer steps every `grad_accum` micro-batches
    max_steps: int = 20000  # optimizer steps (not micro-steps)

    # optimizer (AdamW + cosine schedule with warmup)
    lr: float = 3.0e-4
    min_lr: float = 3.0e-5  # end LR of the cosine decay (0.0 for a clean anneal)
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # MoE aux-loss-free load balancing (bias nudged outside the gradient each step)
    router_bias_lr: float = 1.0e-3

    # logging / eval / checkpoint cadence (in optimizer steps)
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 50
    ckpt_every: int = 1000

    # warm-start: path to a model checkpoint to initialize from (anneal loads pretrain)
    init_from: str | None = None
    # resume: path to a full checkpoint dir to continue (params + optimizer + step)
    resume: str | None = None

    # numerics
    dtype: str = "float32"  # T4 has no bf16; float32 is the safe default (see README)


@dataclass
class DataSourceConfig:
    """The Hugging Face source + how to turn a row into a text document."""
    path: str = "bigcode/the-stack-smol"  # HF dataset id
    name: str | None = None  # dataset config name (`load_dataset(path, name)`)
    data_dir: str | None = None  # for datasets that shard by directory
    split: str = "train"
    streaming: bool = False
    text_key: str | None = "content"  # column holding the document text
    template: str | None = None  # OR a str.format template over row columns
    max_documents: int | None = None  # cap for quick/T4-sized runs


@dataclass
class FilterConfig:
    enabled: bool = True


@dataclass
class DedupConfig:
    exact: bool = True
    minhash: bool = True
    minhash_threshold: float = 0.8
    minhash_num_perm: int = 64
    minhash_ngram: int = 5


@dataclass
class FimConfig:
    enabled: bool = True
    rate: float = 0.5  # fraction of documents transformed
    spm_rate: float = 0.5  # of those, fraction using the SPM (vs PSM) ordering


@dataclass
class PrepareConfig:
    out_dir: str = "data/pretrain"
    tokenizer: str = "tokenizers/code_bpe.json"
    hf: DataSourceConfig = field(default_factory=DataSourceConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    fim: FimConfig = field(default_factory=FimConfig)
    val_fraction: float = 0.005
    append_eot: bool = True
    seed: int = 0

    @staticmethod
    def load(path: str | Path) -> "PrepareConfig":
        d = _read_yaml(path)
        return PrepareConfig(
            out_dir=d.get("out_dir", "data/pretrain"),
            tokenizer=d.get("tokenizer", "tokenizers/code_bpe.json"),
            hf=DataSourceConfig(**(d.get("hf") or {})),
            filter=FilterConfig(**(d.get("filter") or {})),
            dedup=DedupConfig(**(d.get("dedup") or {})),
            fim=FimConfig(**(d.get("fim") or {})),
            val_fraction=d.get("val_fraction", 0.005),
            append_eot=d.get("append_eot", True),
            seed=d.get("seed", 0),
        )


@dataclass
class Config:
    """Top-level config for a training phase (model + train)."""
    model: KimiLinearConfig
    train: TrainConfig

    @staticmethod
    def load(path: str | Path) -> "Config":
        d = _read_yaml(path)
        model_path = d.get("model_config")
        if model_path is None:
            raise ValueError(f"{path}: training config must set `model_config: <path>`")
        model = load_model_config(model_path)

        # `data:` block folds into TrainConfig for convenience.
        data = d.get("data") or {}
        train_d = dict(d.get("train") or {})
        train_d.setdefault("data_dir", data.get("bin_dir", "data/pretrain"))
        train_d.setdefault("seq_len", data.get("seq_len", 512))

        tfields = {f.name for f in dataclasses.fields(TrainConfig)}
        train = TrainConfig(**{k: v for k, v in train_d.items() if k in tfields})

        # Enforce the GDN-2 chunkwise constraint up front (clearer than a kernel crash).
        if train.seq_len % model.gdn_chunk_size != 0:
            raise ValueError(
                f"seq_len ({train.seq_len}) must be a multiple of gdn_chunk_size "
                f"({model.gdn_chunk_size})."
            )
        if train.seq_len > model.max_seq_len:
            raise ValueError(
                f"seq_len ({train.seq_len}) exceeds model.max_seq_len "
                f"({model.max_seq_len}); the MLA causal mask is built at max_seq_len."
            )
        return Config(model=model, train=train)
