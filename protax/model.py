import jax
import jax.numpy as jnp
from jax.experimental import sparse
from functools import partial
import numpy as np
from functools import partial
from .ops import knn, knn_v2


def cosine_dist(q, base_embeddings):
    """
    Computes sequence distance between the query and 
    an array of reference sequences
    """
    return jnp.maximum(0.5 * (1.0 - (q @ base_embeddings.T)), 1e-10)


def p_dist(query, ok_query, refs, ok_refs):
    """
    Computes sequence distance (1 - fractional match) between the query and
    each reference sequence.

    Returns:
        jax.Array of shape (R,) and dtype float, where R is the number of
        reference sequences (rows of seqs). Entry i is the distance to ref i.
    """

    ok = jnp.bitwise_and(ok_query, ok_refs)
    ok = jnp.sum(jax.lax.population_count(ok), axis=1)
    match = jnp.bitwise_and(query, refs)

    match_tots = jnp.sum(jax.lax.population_count(match), axis=1)
    return jnp.maximum(1 - (match_tots / ok), 1e-10)


def get_X(dists, tree, N, sc_mean, sc_var, dist_scaling=None, pt_min=None, pt_gap=None):
    """
    KNN-based method for computing design matrix
    """

    node2seq = tree.node2seq    # Maps sequences to nodes
    new_dat = jnp.take(dists, node2seq.indices)  # fills data of the sparse matrix with the distances for sequences from the query rather than just 1s.
    new_dat = node2seq.data * new_dat            # mask out sequences for leave one out.

    X = knn(node2seq.indptr, node2seq.indices, new_dat, N)  # Compute KNN over sequences under each node

    if dist_scaling == "z-score":
        X = (X - sc_mean) / jnp.sqrt(sc_var)

    elif dist_scaling == "log":
        X = jnp.log(X + 1e-8)

    elif dist_scaling == "power":
        X_min = np.asarray(jax.device_get(X[:, 0])).reshape(-1, 1)
        X_gap = np.asarray(jax.device_get(X[:, 1])).reshape(-1, 1)
        X_min = jnp.asarray(pt_min.transform(X_min).ravel(), dtype=X.dtype)
        X_gap = jnp.asarray(pt_gap.transform(X_gap).ravel(), dtype=X.dtype)
        X = jnp.stack([X_min, X_gap], axis=1)

    elif dist_scaling == "hybrid":
        X_min = np.asarray(jax.device_get(X[:, 0])).reshape(-1, 1)
        X_min = jnp.asarray(pt_min.transform(X_min).ravel(), dtype=X.dtype)
        X_gap = jnp.log(X[:, 1] + 1e-8) 
        X = jnp.stack([X_min, X_gap], axis=1)     

    elif dist_scaling == "log-z":
        X = jnp.log(X + 1e-8)
        X = (X - sc_mean) / jnp.sqrt(sc_var)

    elif dist_scaling == "none":
        pass
    
    else:
        raise ValueError(f"Invalid dist_scaling: {dist_scaling}")

    # X = (X - sc_mean) / jnp.sqrt(sc_var)
    X = (X.T * (tree.node_state[:, 1])).T  # Multiply by has_refs mask
    X = jnp.concatenate((tree.node_state, X), axis=1)   # Concatenate node state (known, has_refs) and features
    return X


def get_z(X, params):
    """
    Compute weighted sum for each node
    """
    z = jnp.sum(jnp.multiply(X, params.beta), axis=1)
    return z


def get_bprobs(z, segments, segnum):
    """
    Compute branch probabilities of each node
    """
    norm_factors = jax.ops.segment_sum(z, segments, num_segments=segnum)
    norm_factors = jnp.take(norm_factors, segments)
    branch_probs =  jnp.nan_to_num(z / norm_factors)
    branch_probs = branch_probs.at[0].set(1)

    return branch_probs


def get_log_bprobs(z, segments, segnum):
    """
    Compute log branch probabilities of each node
    For training PROTAX
    """

    exp_z = jnp.exp(z)
    norm_factors = jnp.log(jax.ops.segment_sum(exp_z, segments, num_segments=segnum))
    norm_factors = jnp.take(norm_factors, segments)
    branch_probs =  jnp.nan_to_num(z - norm_factors)
    branch_probs = branch_probs.at[0].set(0)

    return branch_probs


def fill_bprob(X, beta, tree, segnum):
    """
    Compute branch probability of entire taxonomy, filled in relevant paths
    X: design matrix of shape [N, M]
    beta: param matrix of shape [N, M]
    segnum: array with shape [N]

    N = # nodes
    M = # features
    """
    z = jnp.sum(jnp.multiply(X, beta), axis=1)    
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum)
    max_z = jnp.take(max_z, tree.segments)
    exp_z = jnp.exp(z - max_z)
    branch_probs = get_bprobs(exp_z, tree.segments, segnum) # assign prob to each node relative to its siblings (sum of siblings under the parent is 1).
    return branch_probs


def fill_log_bprob(X, beta, tree, segnum):
    """
    Compute log probabilities over entire taxonomy
    Used for training PROTAX
    """
    z = jnp.sum(jnp.multiply(X, beta), axis=1)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum)
    max_z = jnp.take(max_z, tree.segments)
    z = z - max_z
    branch_probs = get_log_bprobs(z, tree.segments, segnum)

    # puts it in a necessary structure for training
    filled_paths = jnp.take(branch_probs, tree.paths, fill_value=0)
    return filled_paths


# -------------------------------------- Hybrid Model --------------------------------------
def get_X_single(dists, tree, N, sc_mean, sc_var, dist_scaling=None, pt_min=None, pt_gap=None):
    """
    KNN-based method for computing design matrix
    """

    node2seq = tree.node2seq    # Maps sequences to nodes
    new_dat = jnp.take(dists, node2seq.indices)  # fills data of the sparse matrix with the distances for sequences from the query rather than just 1s.
    new_dat = node2seq.data * new_dat            # mask out sequences for leave one out.

    X = knn(node2seq.indptr, node2seq.indices, new_dat, N)  # Compute KNN over sequences under each node

    if dist_scaling == "z-score":
        X = (X - sc_mean) / jnp.sqrt(sc_var)

    elif dist_scaling == "log":
        X = jnp.log(X + 1e-8)

    elif dist_scaling == "power":
        X_min = np.asarray(jax.device_get(X[:, 0])).reshape(-1, 1)
        X_gap = np.asarray(jax.device_get(X[:, 1])).reshape(-1, 1)
        X_min = jnp.asarray(pt_min.transform(X_min).ravel(), dtype=X.dtype)
        X_gap = jnp.asarray(pt_gap.transform(X_gap).ravel(), dtype=X.dtype)
        X = jnp.stack([X_min, X_gap], axis=1)

    elif dist_scaling == "hybrid":
        X_min = np.asarray(jax.device_get(X[:, 0])).reshape(-1, 1)
        X_min = jnp.asarray(pt_min.transform(X_min).ravel(), dtype=X.dtype)
        X_gap = jnp.log(X[:, 1] + 1e-8) 
        X = jnp.stack([X_min, X_gap], axis=1)     

    elif dist_scaling == "none":
        pass

    elif dist_scaling == "log-z":
        X = jnp.log(X + 1e-8)
        X = (X - sc_mean) / jnp.sqrt(sc_var)
    
    else:
        raise ValueError(f"Invalid dist_scaling: {dist_scaling}")

    # X = (X - sc_mean) / jnp.sqrt(sc_var)
    X = (X.T * (tree.node_state[:, 1])).T  # Multiply by has_refs mask
    return X

def get_hybrid_X(dists_bert, dists_mamba, tree, N, sc_mean, sc_var, dist_scaling=None, pt_min=None, pt_gap=None):
    if sc_mean.shape[1] > 2:
        sc_mean[:, [0, 1]]
        X_bert = get_X_single(dists_bert, tree, N, sc_mean[:, [0, 1]], sc_var[:, [0, 1]], dist_scaling, pt_min, pt_gap)
        X_mamba = get_X_single(dists_mamba, tree, N, sc_mean[:, [2, 3]], sc_var[:, [2, 3]], dist_scaling, pt_min, pt_gap)

    else:
        X_bert = get_X_single(dists_bert, tree, N, sc_mean, sc_var, dist_scaling, pt_min, pt_gap)
        X_mamba = get_X_single(dists_mamba, tree, N, sc_mean, sc_var, dist_scaling, pt_min, pt_gap)

    X = jnp.concatenate((tree.node_state, X_bert, X_mamba), axis=1)   # Concatenate node state (known, has_refs) and features
    return X

def get_probs_hybrid(dists_bert, dists_mamba, tree, params, segnum, N, dist_scaling=None, pt_min=None, pt_gap=None):
    X = get_hybrid_X(dists_bert, dists_mamba, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)
    bprobs = fill_bprob(X, params.beta, tree, segnum)   # holds [N, num_levels] arrays
    return bprobs
# -------------------------------------- Inference --------------------------------------

# @partial(jax.jit, static_argnums=(4, 5))
def get_probs(dists, tree, params, segnum, N, dist_scaling=None, pt_min=None, pt_gap=None):
    X = get_X(dists, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)
    bprobs = fill_bprob(X, params.beta, tree, segnum)   # holds [N, num_levels] arrays
    # return jnp.prod(bprobs, axis=1) # product of the probabilities of the paths.
    return bprobs