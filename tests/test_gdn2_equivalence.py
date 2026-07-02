"""The chunkwise GDN-2 core (training path) must match the recurrent core (reference).

If these agree, the parallel algorithm used during training computes exactly the
same function as the O(L) token-by-token recurrence used at inference — the property
the whole linear-attention speedup rests on.

Run: JAX_PLATFORMS=cpu python -m pytest tests/test_gdn2_equivalence.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)


def _random_inputs(B=2, H=3, L=64, dk=16, dv=16, seed=0):
    k = jax.random.PRNGKey(seed)
    ks = jax.random.split(k, 7)
    q = jax.random.normal(ks[0], (B, H, L, dk))
    kk = jax.random.normal(ks[1], (B, H, L, dk))
    v = jax.random.normal(ks[2], (B, H, L, dv))
    # log-decay g <= 0 (alpha in (0,1)); mild decay for a stable fp32 comparison.
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, H, L, dk))) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, dk)))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, H, L, dv)))
    # L2-normalize q,k as the layer does.
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    kk = kk / (jnp.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    S0 = jnp.zeros((B, H, dk, dv))
    return q, kk, v, g, b, w, S0


def test_chunk_matches_recurrent():
    q, k, v, g, b, w, S0 = _random_inputs(L=64)
    for C in (16, 32, 64):
        o_chunk, s_chunk = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=C)
        o_rec, s_rec = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
        assert np.allclose(o_chunk, o_rec, atol=1e-3, rtol=1e-3), f"output mismatch C={C}"
        assert np.allclose(s_chunk, s_rec, atol=1e-3, rtol=1e-3), f"state mismatch C={C}"


if __name__ == "__main__":
    test_chunk_matches_recurrent()
    print("ok: chunkwise == recurrent")
