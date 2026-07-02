"""End-to-end smoke test WITHOUT any network/dataset download.

Fabricates a tiny packed corpus (random tokens + meta.json), then exercises the
real Grain loader, the jitted train step (loss + MoE aux + router-bias update),
checkpoint save/load, and streaming generation. Proves the whole pipeline is wired
correctly on CPU before spending T4/GPU time.

Run: JAX_PLATFORMS=cpu python -m pytest tests/test_pipeline_smoke.py -q
     JAX_PLATFORMS=cpu python tests/test_pipeline_smoke.py
"""

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from kimi_linear_gdn2 import KimiLinearConfig
from training.config import TrainConfig
from training.data.loader import make_loader
from training.trainer import (
    build_model,
    load_model,
    make_optimizer,
    make_train_step,
    param_count,
    save_model,
)


def _make_fake_corpus(d: Path, vocab_size=64, train_tokens=20000, val_tokens=4000):
    rng = np.random.default_rng(0)
    (d / "train.bin").write_bytes(
        rng.integers(0, vocab_size, train_tokens, dtype=np.uint16).tobytes()
    )
    (d / "val.bin").write_bytes(
        rng.integers(0, vocab_size, val_tokens, dtype=np.uint16).tobytes()
    )
    meta = {
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "pad_id": 1,
        "eot_id": 0,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        meta = _make_fake_corpus(d)

        seq_len = 64
        mcfg = KimiLinearConfig(
            vocab_size=meta["vocab_size"],
            d_model=64,
            n_layers=4,
            gdn_num_heads=2,
            gdn_head_k_dim=32,
            gdn_head_v_dim=32,
            gdn_chunk_size=16,
            mla_num_q_heads=4,
            mla_num_kv_heads=2,
            mla_head_dim=32,
            moe_d_ff=64,
            moe_n_routed=4,
            moe_top_k=2,
            max_seq_len=seq_len,
        )
        tcfg = TrainConfig(
            data_dir=str(d),
            seq_len=seq_len,
            batch_size=2,
            grad_accum=1,
            max_steps=3,
            warmup_steps=1,
            eval_batches=2,
        )

        model = build_model(mcfg, seed=0)
        optimizer, _ = make_optimizer(model, tcfg)
        train_step = make_train_step(meta["pad_id"], tcfg.router_bias_lr)
        assert param_count(model) > 0

        loader = make_loader(
            str(d),
            "train",
            seq_len,
            tcfg.batch_size,
            num_epochs=1,
            shuffle=True,
            worker_count=0,
        )
        losses = []
        for i, batch in enumerate(loader):
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            assert batch["input_ids"].shape == (tcfg.batch_size, seq_len)
            loss, ce, aux = train_step(model, optimizer, batch)
            losses.append(float(loss))
            if i >= 3:
                break
        assert all(np.isfinite(losses)), f"non-finite loss: {losses}"

        # checkpoint round-trip: reload into a fresh model and compare logits.
        ids = jnp.zeros((1, seq_len), jnp.int32)
        logits_before, _ = model(ids)
        save_model(d / "m.msgpack", model)
        model2 = build_model(mcfg, seed=1)  # different init on purpose
        load_model(d / "m.msgpack", model2)
        logits_after, _ = model2(ids)
        assert np.allclose(logits_before, logits_after, atol=1e-5), (
            "ckpt round-trip mismatch"
        )

        # streaming generation works and stays in-vocab.
        gen = model.generate(ids[:, :4], max_new_tokens=8)
        assert gen.shape == (1, 8)
        assert int(gen.max()) < meta["vocab_size"]
        print(f"ok: losses={['%.3f' % x for x in losses]} params={param_count(model)}")


if __name__ == "__main__":
    test_pipeline_smoke()
