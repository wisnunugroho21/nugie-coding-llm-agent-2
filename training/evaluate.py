"""Validation loss / perplexity over the held-out split."""

from __future__ import annotations

import sys

# Abseil-based deps (jax/grain/orbax) parse the whole process command line via
# absl.flags on import and abort on our argparse flags ("Unknown command line
# flag 'config'"). Hide our flags from them, then restore for our own argparse.
_saved_argv = sys.argv[:]
sys.argv = sys.argv[:1]

import argparse
import dataclasses
import math

import jax.numpy as jnp

from training.config import Config
from training.data.loader import load_meta, make_loader
from training.trainer import build_model, load_model, make_eval_step

sys.argv = _saved_argv

# grain defines its own absl flags and reads them when it starts data-loader
# workers. Since we launch via argparse (not absl.app.run), those flags are
# never parsed; mark them parsed so their defaults are usable.
from absl import flags as _absl_flags

if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS.mark_as_parsed()


def run_eval(model, data_dir, seq_len, batch_size, pad_id, max_batches=50):
    eval_step = make_eval_step(pad_id)
    loader = make_loader(
        data_dir, "val", seq_len, batch_size,
        shuffle=False, num_epochs=1, worker_count=1,
    )
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        total += float(eval_step(model, batch))
        n += 1
        if n >= max_batches:
            break
    mean = total / max(n, 1)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 20.0)), "batches": n}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the val split.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="path to a model_*.msgpack")
    ap.add_argument("--batches", type=int, default=50)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    meta = load_meta(cfg.train.data_dir)
    mcfg = dataclasses.replace(cfg.model, vocab_size=meta["vocab_size"])
    model = build_model(mcfg, cfg.train.seed)
    load_model(args.model, model)

    res = run_eval(
        model, cfg.train.data_dir, cfg.train.seq_len,
        cfg.train.batch_size, meta["pad_id"], args.batches,
    )
    print(res)


if __name__ == "__main__":
    main()
