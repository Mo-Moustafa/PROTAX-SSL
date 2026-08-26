"""
Hybrid design matrix (same as model_lin.get_hybrid_X: 6 features per node)
with a per-rank MLP after X and before sibling log-softmax / softmax.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import model as model_lin

# Matches hybrid linear beta layout: one parameter set per taxonomic rank row.
NUM_RANKS = 8           # from root to species (8 ranks)
HYBRID_IN_DIM = 6       # two node states / min and gap from bert / min and gap from mamba


class HybridMLPParams(NamedTuple):
    """Per-rank MLP with configurable depth."""

    hidden_W: tuple[jax.Array, ...]   # each (NUM_RANKS, hidden_dim, in_dim_prev)
    hidden_b: tuple[jax.Array, ...]   # each (NUM_RANKS, hidden_dim)
    out_W: jax.Array                  # (NUM_RANKS, hidden_last)
    out_b: jax.Array                  # (NUM_RANKS,)

def init_hybrid_mlp_params(
    seed_key: jax.Array,
    num_ranks: int = NUM_RANKS,
    in_dim: int = HYBRID_IN_DIM,
    hidden: int = 16,
    num_layers: int = 1,
    scale: float = 0.02,
) -> HybridMLPParams:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    keys = jax.random.split(seed_key, num_layers + 1)
    hidden_W = []
    hidden_b = []
    prev_dim = in_dim
    for li in range(num_layers):
        W = jax.random.normal(keys[li], (num_ranks, hidden, prev_dim), dtype=jnp.float32) * scale
        b = jnp.zeros((num_ranks, hidden), dtype=jnp.float32)
        hidden_W.append(W)
        hidden_b.append(b)
        prev_dim = hidden

    out_W = jax.random.normal(keys[-1], (num_ranks, prev_dim), dtype=jnp.float32) * scale
    out_b = jnp.zeros((num_ranks,), dtype=jnp.float32)
    return HybridMLPParams(hidden_W=tuple(hidden_W), hidden_b=tuple(hidden_b), out_W=out_W, out_b=out_b)

def load_mlp_params_from_npz(par_path: str) -> HybridMLPParams:
    par = np.load(par_path)
    # New checkpoint format: mlp_num_layers + per-layer hidden tensors.
    if "mlp_num_layers" in par:
        num_layers = int(np.asarray(par["mlp_num_layers"]).item())
        hidden_W = []
        hidden_b = []
        for li in range(num_layers):
            hidden_W.append(jnp.asarray(par[f"mlp_hidden_W_{li}"], dtype=jnp.float32))
            hidden_b.append(jnp.asarray(par[f"mlp_hidden_b_{li}"], dtype=jnp.float32))
        out_W = jnp.asarray(par["mlp_out_W"], dtype=jnp.float32)
        out_b = jnp.asarray(par["mlp_out_b"], dtype=jnp.float32)
        return HybridMLPParams(hidden_W=tuple(hidden_W), hidden_b=tuple(hidden_b), out_W=out_W, out_b=out_b)

    # Backward-compatible legacy format (single hidden layer).
    return HybridMLPParams(
        hidden_W=(jnp.asarray(par["mlp_W"], dtype=jnp.float32),),
        hidden_b=(jnp.asarray(par["mlp_b"], dtype=jnp.float32),),
        out_W=jnp.asarray(par["mlp_v"], dtype=jnp.float32),
        out_b=jnp.asarray(par["mlp_c"], dtype=jnp.float32),
    )

def per_rank_mlp_logits(X: jax.Array, mlp: HybridMLPParams, ranks: jax.Array) -> jax.Array:
    """
    X: (N, 6), ranks: (N,) rank id per node (same indexing as jnp.take(beta, lvl, axis=0)).
    """
    r = jnp.asarray(ranks, dtype=jnp.int32)
    h = X
    for W, b in zip(mlp.hidden_W, mlp.hidden_b):
        W_n = W[r]      # per-node rank-specific hidden layer weights
        b_n = b[r]
        lin = jnp.einsum("nhk,nk->nh", W_n, h) + b_n
        h = jax.nn.relu(lin)

    out_W_n = mlp.out_W[r]
    out_b_n = mlp.out_b[r]
    return jnp.sum(h * out_W_n, axis=1) + out_b_n

def fill_log_bprob(X: jax.Array, mlp: HybridMLPParams, tree, segnum: int, ranks: jax.Array) -> jax.Array:
    z = per_rank_mlp_logits(X, mlp, ranks)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum)
    max_z = jnp.take(max_z, tree.segments)
    z = z - max_z
    branch_probs = model_lin.get_log_bprobs(z, tree.segments, segnum)

    # puts it in a necessary structure for training
    filled_paths = jnp.take(branch_probs, tree.paths, fill_value=0)
    return filled_paths

def fill_bprob(X: jax.Array, mlp: HybridMLPParams, tree, segnum: int, ranks: jax.Array) -> jax.Array:
    z = per_rank_mlp_logits(X, mlp, ranks)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum)
    max_z = jnp.take(max_z, tree.segments)
    exp_z = jnp.exp(z - max_z)
    branch_probs = model_lin.get_bprobs(exp_z, tree.segments, segnum)
    return branch_probs

def get_probs_hybrid_mlp(dists_bert, dists_mamba, tree, params, segnum, N, mlp, dist_scaling=None, pt_min=None, pt_gap=None):
    X = model_lin.get_hybrid_X(dists_bert, dists_mamba, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)
    return fill_bprob(X, mlp, tree, segnum, tree.ranks)
