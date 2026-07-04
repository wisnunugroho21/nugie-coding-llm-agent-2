"""Run one training phase (pretrain or anneal) for the Kimi-Linear GDN-2 code model.

Usage:
    python -m training.train --config configs/pretrain.yaml
    python -m training.train --config configs/anneal.yaml            # init_from set in YAML
    python -m training.train --config configs/pretrain.yaml --resume checkpoints/pretrain

Two-phase recipe (OpenCoder Sec. 3): a long general pretraining phase on the large
filtered corpus, followed by a short *annealing* phase on a smaller, higher-quality
corpus with the LR decayed toward zero. Anneal is just this same loop with a lower
peak LR, `min_lr: 0`, fewer steps, a different `data_dir`, and `init_from` pointing
at the final pretrain model checkpoint.
"""

from __future__ import annotations

import sys

# Abseil-based deps (jax/grain/orbax) parse the whole process command line via
# absl.flags on import and abort on our argparse flags ("Unknown command line
# flag 'config'"). Hide our flags from them, then restore for our own argparse.
_saved_argv = sys.argv[:]
sys.argv = sys.argv[:1]

import argparse
import dataclasses
import time

import jax.numpy as jnp

from training.config import Config
from training.data.loader import load_meta, make_loader
from training.evaluate import run_eval
from training.trainer import (
    build_model,
    load_model,
    make_optimizer,
    make_schedule,
    make_train_step,
    param_count,
    restore_checkpoint,
    save_checkpoint,
)

sys.argv = _saved_argv

# grain defines its own absl flags and reads them when it starts data-loader
# workers. Since we launch via argparse (not absl.app.run), those flags are
# never parsed; mark them parsed so their defaults are usable. Runs at import
# time so grain's spawned worker processes inherit the parsed state too.
from absl import flags as _absl_flags

if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS.mark_as_parsed()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Kimi-Linear GDN-2 code LM.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument(
        "--init-from", default=None, help="model_*.msgpack to warm-start from"
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tcfg = cfg.train
    if args.resume:
        tcfg = dataclasses.replace(tcfg, resume=args.resume)
    if args.init_from:
        tcfg = dataclasses.replace(tcfg, init_from=args.init_from)

    # Model vocab MUST match the tokenizer used to pack the data.
    meta = load_meta(tcfg.data_dir)
    mcfg = dataclasses.replace(cfg.model, vocab_size=meta["vocab_size"])
    pad_id = meta["pad_id"]
    print(
        f"[train] phase={tcfg.phase} data={tcfg.data_dir} "
        f"vocab={mcfg.vocab_size} seq_len={tcfg.seq_len}"
    )

    model = build_model(mcfg, tcfg.seed)
    optimizer, _ = make_optimizer(model, tcfg)
    schedule = make_schedule(tcfg)
    print(f"[train] parameters: {param_count(model):,}")

    start_step = 0
    if tcfg.init_from:
        load_model(tcfg.init_from, model)
        print(f"[train] warm-started from {tcfg.init_from}")
    if tcfg.resume:
        # Restores params + optimizer state (incl. the LR-schedule step count).
        # NOTE: the data loader below is NOT fast-forwarded — a resumed run re-visits
        # the shuffled stream from the start. Acceptable at this scale; for exact
        # data resumption you would checkpoint the Grain iterator state as well.
        start_step = restore_checkpoint(tcfg.resume, model, optimizer)
        print(f"[train] resumed from {tcfg.resume} at step {start_step}")

    train_loader = make_loader(
        tcfg.data_dir,
        "train",
        tcfg.seq_len,
        tcfg.batch_size,
        seed=tcfg.seed,
        shuffle=True,
        num_epochs=None,
        worker_count=2,
    )
    train_step = make_train_step(pad_id, tcfg.router_bias_lr)

    step = start_step
    micro = 0
    tokens_per_opt = tcfg.batch_size * tcfg.seq_len * tcfg.grad_accum
    run_loss = run_ce = run_aux = 0.0
    n_since_log = 0
    t0 = time.time()

    print(
        f"[train] starting loop: max_steps={tcfg.max_steps} "
        f"tokens/opt-step={tokens_per_opt:,}"
    )
    for batch in train_loader:
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        loss, ce, aux = train_step(model, optimizer, batch)
        run_loss += float(loss)
        run_ce += float(ce)
        run_aux += float(aux)
        n_since_log += 1
        micro += 1

        if micro % tcfg.grad_accum != 0:
            continue
        step += 1  # one optimizer step completed

        if step % tcfg.log_every == 0:
            dt = time.time() - t0
            tok_s = tokens_per_opt * tcfg.log_every / max(dt, 1e-6)
            lr = float(schedule(step))
            print(
                f"step {step:>7} | loss {run_loss / n_since_log:.4f} "
                f"| ce {run_ce / n_since_log:.4f} | aux {run_aux / n_since_log:.5f} "
                f"| lr {lr:.2e} | {tok_s / 1e3:.1f}k tok/s"
            )
            run_loss = run_ce = run_aux = 0.0
            n_since_log = 0
            t0 = time.time()

        if step % tcfg.eval_every == 0:
            res = run_eval(
                model,
                tcfg.data_dir,
                tcfg.seq_len,
                tcfg.batch_size,
                pad_id,
                tcfg.eval_batches,
            )
            print(f"[eval] step {step} | {res}")
            t0 = time.time()  # don't count eval time in throughput

        if step % tcfg.ckpt_every == 0:
            save_checkpoint(tcfg.out_dir, step, model, optimizer)
            print(f"[ckpt] saved {tcfg.out_dir}/model_{step}.msgpack")

        if step >= tcfg.max_steps:
            break

    save_checkpoint(tcfg.out_dir, step, model, optimizer)
    print(f"[train] done at step {step}; final checkpoint in {tcfg.out_dir}")


if __name__ == "__main__":
    main()
