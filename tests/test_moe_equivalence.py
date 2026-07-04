"""The dispatched grouped-GEMM MoE path must match the dense reference path.

`GroupedGemmMoE.__call__` routes tokens through sort -> ragged_dot -> scatter-add
(the fast production pattern); `dense_forward` computes every expert densely with
the SAME weights. If the two agree, the dispatch/combine machinery (argsort,
group_sizes, gather/scatter) is wired correctly.

Run: JAX_PLATFORMS=cpu python -m pytest tests/test_moe_equivalence.py -q
"""

import flax.nnx as nnx
import jax
import numpy as np

from multi_latent_attention.moe import GroupedGemmMoE, update_router_bias


def _make_moe(**kw):
    defaults = dict(d_model=32, d_ff=48, n_routed=6, n_shared=1, top_k=2)
    defaults.update(kw)
    return GroupedGemmMoE(**defaults, rngs=nnx.Rngs(0))


def test_dispatched_matches_dense():
    moe = _make_moe()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 32))
    y_fast, aux = moe(x)
    y_dense = moe.dense_forward(x)
    assert np.allclose(np.asarray(y_fast), np.asarray(y_dense), atol=1e-4), (
        "dispatched grouped-GEMM output != dense reference"
    )
    # Every (token, expert) assignment is counted exactly once.
    assert int(aux["group_sizes"].sum()) == 2 * 16 * moe.top_k


def test_dispatched_matches_dense_with_bias():
    # A skewed selection bias changes WHICH experts win top-k; both paths must
    # follow the same (biased) selection and still agree.
    moe = _make_moe()
    moe.router_bias.value = moe.router_bias.value.at[0].add(0.5)
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 32, 32))
    y_fast, _ = moe(x)
    y_dense = moe.dense_forward(x)
    assert np.allclose(np.asarray(y_fast), np.asarray(y_dense), atol=1e-4)


def test_update_router_bias_direction():
    # Over-loaded experts must be pushed DOWN, under-loaded ones UP.
    bias = np.zeros(4, dtype=np.float32)
    group_sizes = np.array([10, 0, 5, 5])  # expert 0 over-, expert 1 under-loaded
    new = np.asarray(update_router_bias(bias, group_sizes, lr=1e-2))
    assert new[0] < 0 and new[1] > 0


if __name__ == "__main__":
    test_dispatched_matches_dense()
    test_dispatched_matches_dense_with_bias()
    test_update_router_bias_direction()
    print("ok: dispatched MoE == dense reference")
