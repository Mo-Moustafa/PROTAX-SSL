"""Classify with a hybrid + per-rank MLP checkpoint (mlp_* arrays in npz)."""
import sys
import os

from protax import protax_utils
import protax.model as model
import protax.model_mlp as mlp_model

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import math

import time
from tqdm import tqdm

import joblib


def hierarchy_classification(probs, tree):
    max_rank = tree.ranks.max() + 1
    classified_layers = np.full(max_rank, -1)
    classified_probs = np.full(max_rank, np.nan)

    current_parent = 0
    classified_layers[0] = 0
    classified_probs[0] = 1.0

    for i in range(1, max_rank):
        children = np.where(tree.parents == current_parent)[0]

        levels = tree.ranks[children]
        if not np.all(levels == i):
            print("children do not belong to the same level!")

        if children.size > 0:
            child_probs = probs[children]

            # if not math.isclose(child_probs.sum(), 1.0, rel_tol=1e-5):
            #     print("branch probabilities do not sum to 1.0")
            #     print("prob sum: ", child_probs.sum(), "\n")

            local_idx = np.argmax(child_probs)
            current_prob = child_probs[local_idx]
            current_parent = children[local_idx]

            classified_layers[i] = current_parent
            classified_probs[i] = current_prob
        else:
            break

    return classified_layers, classified_probs


def create_design_matrices(base_dir, train_dir, tree, dist_scaling, N, params, train_eval, loo_map):
    n2s = tree.node2seq
    node_state = tree.node_state

    if dist_scaling == "power":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = joblib.load("models/scalings/gap_power_transformer.joblib")
    elif dist_scaling == "hybrid":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = None
    else:
        pt_min = None
        pt_gap = None

    print("Reading Embeddings...")
    base_embeddings = np.load(base_dir)
    train_embeddings = np.load(train_dir)
            
    bert_base = jnp.array(base_embeddings["bert"])
    mamba_base = jnp.array(base_embeddings["mamba"])
    assert bert_base.shape == mamba_base.shape
    base_length = int(bert_base.shape[0])

    bert_train = jnp.array(train_embeddings["bert"])
    mamba_train = jnp.array(train_embeddings["mamba"])
    assert bert_train.shape == mamba_train.shape
    Q = int(bert_train.shape[0])

    X_all = None    
    for i in tqdm(range(Q), desc="Computing design matrices", file=sys.stderr, dynamic_ncols=True, mininterval=1000):
        # mask the tree for the current sample
        if (train_eval and i < base_length):
            new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
            tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)
        
        elif (loo_map is not None and i in loo_map):
            new_n2s, new_ns = protax_utils.mask_n2s(n2s, node_state, loo_map[i])
            tree = tree._replace(node2seq=new_n2s, node_state=new_ns)
        else:
            tree = tree._replace(node2seq=n2s, node_state=node_state)
        
        dists_bert = model.cosine_dist(bert_train[i], bert_base)
        dists_mamba = model.cosine_dist(mamba_train[i], mamba_base)
        X = model.get_hybrid_X(dists_bert, dists_mamba, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        x_host = np.asarray(jax.device_get(X), dtype=np.float32)
        if X_all is None:
            n_nodes, F_dim = x_host.shape
            X_all = np.zeros((Q, n_nodes, F_dim), dtype=np.float32)
        X_all[i] = x_host

    return X_all, Q



def classify_file(q_dir, base_dir, par_dir, tax_dir, dist_scaling, run_id, train_eval, X_all=None, loo_map=None, loo_id=None):
    """
    Hybrid + MLP: precompute hybrid design matrices X on host (bert + mamba), then
    vmap(mlp_model.fill_bprob) in chunks. Optional X_all skips the design-matrix loop.
    """
    start_time = time.time()

    tree, N, segnum = protax_utils.read_tree(tax_dir)
    tree_for_bprob = tree
    ranks = tree.ranks
    cls_chunk = 256

    params = protax_utils.read_model(par_dir, tree.ranks)
    mlp_params = mlp_model.load_mlp_params_from_npz(par_dir)

    if X_all is None:
        X_all, Q = create_design_matrices(base_dir, q_dir, tree, dist_scaling, N, params, train_eval, loo_map)
    else:
        Q = X_all.shape[0]

    vmapped_bprob = jax.jit(jax.vmap(lambda x: mlp_model.fill_bprob(x, mlp_params, tree_for_bprob, segnum, ranks)))
    res = []
    final_probs = []

    for start in tqdm(range(0, Q, cls_chunk), desc="Classifying queries", file=sys.stderr, dynamic_ncols=True, mininterval=5):
        end = min(start + cls_chunk, Q)
        X_batch = jnp.asarray(X_all[start:end], dtype=jnp.float32)
        probs_batch = vmapped_bprob(X_batch)
        probs_batch = jax.device_get(jax.block_until_ready(probs_batch))
        for j in range(end - start):
            classified_layers, classified_probs = hierarchy_classification(probs_batch[j], tree_for_bprob)
            res.append(classified_layers)
            final_probs.append(classified_probs)

    cumulative_probs = np.array(final_probs)
    cumulative_probs = np.cumprod(cumulative_probs, axis=1)

    # For parallel classification (unknown)    
    # for i in tqdm(range(Q),desc="Classifying queries",file=sys.stderr,dynamic_ncols=True,mininterval=1000):
        
    #     bprobs = mlp_model.fill_bprob(X_all[i], mlp_params, tree_for_bprob, segnum, ranks)
    #     filled_paths = jnp.take(bprobs, tree.paths, fill_value=1)
    #     probs = jnp.prod(filled_paths, axis=1)
    #     probs = jnp.take(probs, tree.paths, fill_value=-1)
        
    #     res.append(jnp.argmax(probs, axis=0))
    #     final_probs.append(jnp.max(probs, axis=0))
    # cumulative_probs = np.array(final_probs)

    rank_names = ["root", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
    node_cols = [f"{r}_id" for r in rank_names]
    prob_cols = [f"{r}_prob" for r in rank_names]
    df_nodes = pd.DataFrame(res, columns=node_cols)
    df_probs = pd.DataFrame(cumulative_probs, columns=prob_cols)
    final_df = pd.concat([df_nodes, df_probs], axis=1)

    if train_eval:
        final_df.to_csv(f"results_{run_id}.csv", index=False)
    elif loo_id is not None:
        final_df.to_csv(f"results_{run_id}_test_{loo_id}.csv", index=False)
    else:
        final_df.to_csv(f"results_{run_id}_test.csv", index=False)

    end_time = time.time()
    tot_time = end_time - start_time
    print(f"finished in {(tot_time) / 60} minutes")
