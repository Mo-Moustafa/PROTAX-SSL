"""
Hybrid design matrix (same as model_bert.get_hybrid_X: 6 features per node)
with a per-rank MLP after X and before sibling log-softmax / softmax.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import model_bert as bert

# Matches hybrid linear beta layout: one parameter set per taxonomic rank row.
NUM_RANKS = 8
HYBRID_IN_DIM = 6


class HybridMLPParams(NamedTuple):
    """Per-rank MLP: h = relu(W_r @ x + b_r), z = v_r^T h + c_r."""

    W: jax.Array  # (NUM_RANKS, hidden, HYBRID_IN_DIM)
    b: jax.Array  # (NUM_RANKS, hidden)
    v: jax.Array  # (NUM_RANKS, hidden)
    c: jax.Array  # (NUM_RANKS,)


def init_hybrid_mlp_params(
    key: jax.Array,
    num_ranks: int = NUM_RANKS,
    in_dim: int = HYBRID_IN_DIM,
    hidden: int = 16,
    scale: float = 0.02,
) -> HybridMLPParams:
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (num_ranks, hidden, in_dim), dtype=jnp.float32) * scale
    b = jnp.zeros((num_ranks, hidden), dtype=jnp.float32)
    v = jax.random.normal(k2, (num_ranks, hidden), dtype=jnp.float32) * scale
    c = jnp.zeros((num_ranks,), dtype=jnp.float32)
    return HybridMLPParams(W=W, b=b, v=v, c=c)


def load_mlp_params_from_npz(par_path: str) -> HybridMLPParams:
    par = np.load(par_path)
    return HybridMLPParams(
        W=jnp.asarray(par["mlp_W"], dtype=jnp.float32),
        b=jnp.asarray(par["mlp_b"], dtype=jnp.float32),
        v=jnp.asarray(par["mlp_v"], dtype=jnp.float32),
        c=jnp.asarray(par["mlp_c"], dtype=jnp.float32),
    )


def per_rank_mlp_logits(
    X: jax.Array,
    mlp: HybridMLPParams,
    ranks: jax.Array,
) -> jax.Array:
    """
    X: (N, 6), ranks: (N,) rank id per node (same indexing as jnp.take(beta, lvl, axis=0)).
    """
    r = jnp.asarray(ranks, dtype=jnp.int32)
    r = jnp.clip(r, 0, mlp.W.shape[0] - 1)
    W_n = mlp.W[r]
    b_n = mlp.b[r]
    v_n = mlp.v[r]
    c_n = mlp.c[r]
    lin = jnp.einsum("nhk,nk->nh", W_n, X) + b_n
    h = jax.nn.relu(lin)
    return jnp.sum(h * v_n, axis=1) + c_n


def fill_log_bprob(
    X: jax.Array,
    mlp: HybridMLPParams,
    tree,
    segnum: int,
    ranks: jax.Array,
) -> jax.Array:
    z = per_rank_mlp_logits(X, mlp, ranks)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum)
    max_z = jnp.take(max_z, tree.segments)
    z = z - max_z
    branch_probs = bert.get_log_bprobs(z, tree.segments, segnum)
    filled_paths = jnp.take(branch_probs, tree.paths, fill_value=0)
    return filled_paths


def fill_bprob(
    X: jax.Array,
    mlp: HybridMLPParams,
    tree,
    segnum: int,
    ranks: jax.Array,
) -> jax.Array:
    z = per_rank_mlp_logits(X, mlp, ranks)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum)
    max_z = jnp.take(max_z, tree.segments)
    exp_z = jnp.exp(z - max_z)
    return bert.get_bprobs(exp_z, tree.segments, segnum)


def get_probs_hybrid_mlp(
    dists_bert: jax.Array,
    dists_mamba: jax.Array,
    tree,
    params,
    segnum: int,
    N: int,
    mlp: HybridMLPParams,
    dist_scaling=None,
    pt_min=None,
    pt_gap=None,
) -> jax.Array:
    X = bert.get_hybrid_X(
        dists_bert,
        dists_mamba,
        tree,
        N,
        params.sc_mean,
        params.sc_var,
        dist_scaling,
        pt_min,
        pt_gap,
    )
    return fill_bprob(X, mlp, tree, segnum, tree.ranks)
