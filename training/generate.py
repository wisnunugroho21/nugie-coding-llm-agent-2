"""Sampling + FIM-infill decode from a trained checkpoint.

Uses the model's streaming interface (`init_cache` / `step`): the prompt is
prefilled in one pass, then tokens are decoded one at a time, reusing each layer's
state (fixed-size for GDN-2 layers, a growing latent cache for the MLA layers).

Modes:
    completion:  --prompt "def fib(n):"
    infill (FIM): --prefix "def add(a, b):\n    return " --suffix "\n\nprint(add(1,2))"

Sampling controls: --temperature, --top-k, --top-p (nucleus). --temperature 0 = greedy.
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

import jax.numpy as jnp
import numpy as np

from training.config import Config
from training.data.loader import load_meta
from training.data.tokenizer import CodeTokenizer
from training.trainer import build_model, load_model

sys.argv = _saved_argv

# grain defines its own absl flags and reads them when it starts data-loader
# workers. Since we launch via argparse (not absl.app.run), those flags are
# never parsed; mark them parsed so their defaults are usable.
from absl import flags as _absl_flags

if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS.mark_as_parsed()


def _sample_logits(logits: np.ndarray, temperature: float, top_k: int, top_p: float,
                   rng: np.random.Generator) -> int:
    """Sample one token id from a 1-D logits vector (host-side)."""
    if temperature <= 0.0:
        return int(logits.argmax())
    logits = logits.astype(np.float64) / temperature
    if top_k and top_k > 0:
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    # softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        cutoff = np.searchsorted(csum, top_p) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(
    model,
    tok: CodeTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int = 0,
    stop_at_eot: bool = True,
) -> list[int]:
    rng = np.random.default_rng(seed)
    B = 1
    max_len = len(prompt_ids) + max_new_tokens + 1
    caches = model.init_cache(B, max_len)

    ids = jnp.asarray([prompt_ids], jnp.int32)
    logits, caches = model.step(ids, caches)  # prefill
    out: list[int] = []
    for _ in range(max_new_tokens):
        nxt = _sample_logits(
            np.asarray(logits[0, -1]), temperature, top_k, top_p, rng
        )
        if stop_at_eot and nxt == tok.eot_id:
            break
        out.append(nxt)
        logits, caches = model.step(jnp.asarray([[nxt]], jnp.int32), caches)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate code from a trained checkpoint.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="path to a model_*.msgpack")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prefix", default=None, help="FIM infill: text before the hole")
    ap.add_argument("--suffix", default=None, help="FIM infill: text after the hole")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    meta = load_meta(cfg.train.data_dir)
    tok = CodeTokenizer(meta["tokenizer"])
    mcfg = dataclasses.replace(cfg.model, vocab_size=meta["vocab_size"])
    model = build_model(mcfg, cfg.train.seed)
    load_model(args.model, model)

    if args.prefix is not None or args.suffix is not None:
        prefix, suffix = args.prefix or "", args.suffix or ""
        prompt_ids = (
            [tok.fim_prefix_id] + tok.encode(prefix)
            + [tok.fim_suffix_id] + tok.encode(suffix)
            + [tok.fim_middle_id]
        )
        header = f"[FIM infill]\nprefix={prefix!r} suffix={suffix!r}\n--- middle ---"
    else:
        prompt = args.prompt if args.prompt is not None else "def "
        prompt_ids = tok.encode(prompt)
        header = f"[completion]\n{prompt}"

    out = generate(
        model, tok, prompt_ids, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed,
    )
    print(header + tok.decode(out))


if __name__ == "__main__":
    main()
