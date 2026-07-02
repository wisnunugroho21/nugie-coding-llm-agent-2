"""Optimizer, loss, jitted train/eval steps, and checkpointing (Flax NNX + optax).

Objective per step:
    loss = cross_entropy(logits, next_tokens)  +  sum_layer MoE_aux_loss
The MoE also does aux-loss-free load balancing: after each optimizer update we nudge
every layer's router selection bias toward uniform load, OUTSIDE the gradient (the
DeepSeek-V3 / Kimi trick, `update_router_bias`).

Grad accumulation is delegated to `optax.MultiSteps`, so `max_steps` counts true
optimizer steps and the LR schedule is expressed in optimizer steps too.
"""

from __future__ import annotations

import json
from pathlib import Path

import flax.nnx as nnx
import flax.serialization as fser
import jax
import jax.numpy as jnp
import optax

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig
from multi_latent_attention.moe import update_router_bias
from training.config import TrainConfig


def build_model(cfg: KimiLinearConfig, seed: int = 0) -> KimiLinear:
    return KimiLinear(cfg, rngs=nnx.Rngs(seed))


def make_schedule(cfg: TrainConfig) -> optax.Schedule:
    """Warmup + cosine decay, expressed in optimizer steps."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.max_steps,
        end_value=cfg.min_lr,
    )


def make_optimizer(
    model: KimiLinear, cfg: TrainConfig
) -> tuple[nnx.Optimizer, optax.Schedule]:
    schedule = make_schedule(cfg)
    inner = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.beta1,
            b2=cfg.beta2,
            weight_decay=cfg.weight_decay,
        ),
    )
    tx = (
        optax.MultiSteps(inner, every_k_schedule=cfg.grad_accum)
        if cfg.grad_accum > 1
        else inner
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, schedule


# --------------------------------------------------------------------------- #
#  Loss + steps
# --------------------------------------------------------------------------- #
def _loss_fn(model: KimiLinear, batch: dict, pad_id: int):
    logits, aux = model(batch["input_ids"])  # [B,L,V]
    labels = batch["labels"]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)  # [B,L]
    mask = (labels != pad_id).astype(ce.dtype)
    ce = (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    aux_loss = aux["aux_loss"].astype(ce.dtype)
    loss = ce + aux_loss
    return loss, {"ce": ce, "aux_loss": aux_loss, "group_sizes": aux["group_sizes"]}


def make_train_step(pad_id: int, bias_lr: float):
    """Return a jitted train step closed over the (static) pad id and bias LR."""

    @nnx.jit
    def train_step(model: KimiLinear, optimizer: nnx.Optimizer, batch: dict):
        (loss, m), grads = nnx.value_and_grad(_loss_fn, has_aux=True)(
            model, batch, pad_id
        )
        optimizer.update(model, grads)
        # Aux-loss-free load balancing: nudge each layer's router bias (no gradient).
        group_sizes = m["group_sizes"]  # [n_layers, E]
        for i, layer in enumerate(model.layers):
            mm = layer.channel_mixer
            mm.router_bias.value = update_router_bias(
                mm.router_bias.value, group_sizes[i], bias_lr
            )
        return loss, m["ce"], m["aux_loss"]

    return train_step


def make_eval_step(pad_id: int):
    @nnx.jit
    def eval_step(model: KimiLinear, batch: dict):
        logits, _ = model(batch["input_ids"])
        labels = batch["labels"]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask = (labels != pad_id).astype(ce.dtype)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    return eval_step


# --------------------------------------------------------------------------- #
#  Checkpointing (Flax msgpack serialization of NNX state).
# --------------------------------------------------------------------------- #
def _save_state(path: Path, obj) -> None:
    """Serialize an NNX module/optimizer's state as a pure (array-leaf) dict."""
    with open(path, "wb") as f:
        f.write(fser.to_bytes(nnx.state(obj).to_pure_dict()))


def _load_state(path: Path, obj) -> None:
    """In-place restore of an NNX module/optimizer from `_save_state` output."""
    state = nnx.state(obj)
    with open(path, "rb") as f:
        restored = fser.from_bytes(state.to_pure_dict(), f.read())
    state.replace_by_pure_dict(restored)
    nnx.update(obj, state)


def save_model(path: str | Path, model: KimiLinear) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_state(path, model)


def load_model(path: str | Path, model: KimiLinear) -> None:
    """In-place restore of `model`'s parameters from a saved model state."""
    _load_state(Path(path), model)


def save_checkpoint(
    out_dir: str | Path, step: int, model: KimiLinear, optimizer: nnx.Optimizer
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(out / f"model_{step}.msgpack", model)
    _save_state(out / f"opt_{step}.msgpack", optimizer)
    with open(out / "latest.json", "w") as f:
        json.dump({"step": step}, f)


def restore_checkpoint(
    ckpt_dir: str | Path, model: KimiLinear, optimizer: nnx.Optimizer
) -> int:
    """Restore model + optimizer from the latest checkpoint in `ckpt_dir`. Returns step."""
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "latest.json", "r") as f:
        step = json.load(f)["step"]
    load_model(ckpt_dir / f"model_{step}.msgpack", model)
    _load_state(ckpt_dir / f"opt_{step}.msgpack", optimizer)
    return step


def param_count(model: KimiLinear) -> int:
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
