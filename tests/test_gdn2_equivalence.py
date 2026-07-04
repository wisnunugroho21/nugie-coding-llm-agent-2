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


def _random_inputs(B=2, H=3, L=64, dk=16, dv=16, seed=0, nonzero_S0=False,
                   expanded_erase=False):
    k = jax.random.PRNGKey(seed)
    ks = jax.random.split(k, 7)
    q = jax.random.normal(ks[0], (B, H, L, dk))
    kk = jax.random.normal(ks[1], (B, H, L, dk))
    v = jax.random.normal(ks[2], (B, H, L, dv))
    # log-decay g <= 0 (alpha in (0,1)); mild decay for a stable fp32 comparison.
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, H, L, dk))) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, dk)))
    if expanded_erase:
        b = 2.0 * b  # negative-eigenvalue variant: erase gate in [0, 2]
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, H, L, dv)))
    # L2-normalize q,k as the layer does.
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    kk = kk / (jnp.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    # Nonzero S0 exercises the warm-start / cross-call streaming path.
    S0 = (
        jax.random.normal(ks[6], (B, H, dk, dv)) * 0.5
        if nonzero_S0
        else jnp.zeros((B, H, dk, dv))
    )
    return q, kk, v, g, b, w, S0


def _assert_equivalent(q, k, v, g, b, w, S0, label):
    for C in (16, 32, 64):
        o_chunk, s_chunk = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=C)
        o_rec, s_rec = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
        assert np.allclose(o_chunk, o_rec, atol=1e-3, rtol=1e-3), (
            f"output mismatch C={C} ({label})"
        )
        assert np.allclose(s_chunk, s_rec, atol=1e-3, rtol=1e-3), (
            f"state mismatch C={C} ({label})"
        )


def test_chunk_matches_recurrent():
    _assert_equivalent(*_random_inputs(L=64), label="zero S0")


def test_chunk_matches_recurrent_nonzero_state():
    # The chunkwise state-threading (S_[n+1] = Diag(gamma_C) S0 + Ktail^T R) must
    # also hold when entering with a non-trivial state, as during streaming decode.
    _assert_equivalent(*_random_inputs(L=64, seed=1, nonzero_S0=True),
                       label="nonzero S0")


def test_chunk_matches_recurrent_expanded_erase():
    # gdn_expanded_erase=True scales the erase gate to [0, 2]; the transition matrix
    # (I - b k k^T) can then have negative eigenvalues — the equivalence must survive.
    _assert_equivalent(*_random_inputs(L=64, seed=2, nonzero_S0=True,
                                       expanded_erase=True),
                       label="expanded erase")


if __name__ == "__main__":
    test_chunk_matches_recurrent()
    test_chunk_matches_recurrent_nonzero_state()
    test_chunk_matches_recurrent_expanded_erase()
    print("ok: chunkwise == recurrent (zero S0, nonzero S0, expanded erase)")
