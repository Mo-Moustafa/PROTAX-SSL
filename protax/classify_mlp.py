"""Classify with a hybrid + per-rank MLP checkpoint (mlp_* arrays in npz)."""
import sys

from .protax_utils import read_model_jax, mask_n2s
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

            if not math.isclose(child_probs.sum(), 1.0, rel_tol=1e-5):
                print("branch probabilities do not sum to 1.0")
                print("prob sum: ", child_probs.sum(), "\n")

            local_idx = np.argmax(child_probs)
            current_prob = child_probs[local_idx]
            current_parent = children[local_idx]

            classified_layers[i] = current_parent
            classified_probs[i] = current_prob
        else:
            break

    return classified_layers, classified_probs


def classify_file(qdir, base_dir, par_dir, tax_dir, dist_scaling, run_id, train_eval, X_all=None, loo_map=None):
    """
    Hybrid + MLP: precompute hybrid design matrices X on host (bert + mamba), then
    vmap(mlp_model.fill_bprob) in chunks. Optional X_all skips the design-matrix loop.
    """

    tree, params, N, segnum = read_model_jax(par_dir, tax_dir)
    mlp_params = mlp_model.load_mlp_params_from_npz(par_dir)

    if dist_scaling == "power":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = joblib.load("models/scalings/gap_power_transformer.joblib")
    elif dist_scaling == "hybrid":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = None
    else:
        pt_min = None
        pt_gap = None

    print("Reading base embeddings...")
    base_embeddings = np.load(base_dir)
    bert_base = jnp.array(base_embeddings["bert"])
    mamba_base = jnp.array(base_embeddings["mamba"])
    assert bert_base.shape == mamba_base.shape
    print("base embeddings.shape: ", bert_base.shape, mamba_base.shape)

    print("Reading query embeddings...")
    query_embeddings = np.load(qdir)
    bert_query = jnp.array(query_embeddings["bert"])
    mamba_query = jnp.array(query_embeddings["mamba"])
    assert bert_query.shape == mamba_query.shape
    print("query embeddings.shape: ", bert_query.shape, mamba_query.shape)

    tree_orig = tree
    n2s = node_state = None
    if train_eval or loo_map is not None:
        n2s = tree_orig.node2seq
        node_state = tree_orig.node_state
        print("Read nodes successfully")

    base_n = int(bert_base.shape[0])
    Q = int(bert_query.shape[0])
    start_time = time.time()

    if X_all is None:
        for i in tqdm(
            range(Q),
            desc="Computing design matrices",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=5,
        ):
            q_dist_bert = model.cosine_dist(bert_query[i], bert_base)
            q_dist_mamba = model.cosine_dist(mamba_query[i], mamba_base)

            if train_eval and i < base_n:
                new_n2s, new_ns = mask_n2s(n2s, node_state, i)
                tree_i = tree_orig._replace(node2seq=new_n2s, node_state=new_ns)
            elif loo_map is not None and i in loo_map:
                new_n2s, new_ns = mask_n2s(n2s, node_state, loo_map[i])
                tree_i = tree_orig._replace(node2seq=new_n2s, node_state=new_ns)
            else:
                tree_i = tree_orig

            X = model.get_hybrid_X(
                q_dist_bert,
                q_dist_mamba,
                tree_i,
                N,
                params.sc_mean,
                params.sc_var,
                dist_scaling,
                pt_min,
                pt_gap,
            )
            x_host = np.asarray(jax.device_get(X), dtype=np.float32)
            if X_all is None:
                n_nodes, f_dim = x_host.shape
                X_all = np.zeros((Q, n_nodes, f_dim), dtype=np.float32)
            X_all[i] = x_host

    tree_for_bprob = tree_orig
    ranks = tree_orig.ranks
    cls_chunk = 256
    vmapped_bprob = jax.jit(
        jax.vmap(lambda x: mlp_model.fill_bprob(x, mlp_params, tree_for_bprob, segnum, ranks))
    )

    res = []
    final_probs = []

    for start in tqdm(
        range(0, Q, cls_chunk),
        desc="Classifying queries",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=5,
    ):
        end = min(start + cls_chunk, Q)
        X_batch = jnp.asarray(X_all[start:end], dtype=jnp.float32)
        probs_batch = vmapped_bprob(X_batch)
        probs_batch = jax.device_get(jax.block_until_ready(probs_batch))
        for j in range(end - start):
            classified_layers, classified_probs = hierarchy_classification(probs_batch[j], tree_orig)
            res.append(classified_layers)
            final_probs.append(classified_probs)

    cumulative_probs = np.array(final_probs)
    cumulative_probs = np.cumprod(cumulative_probs, axis=1)

    rank_names = ["root", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
    node_cols = [f"{r}_id" for r in rank_names]
    prob_cols = [f"{r}_prob" for r in rank_names]
    df_nodes = pd.DataFrame(res, columns=node_cols)
    df_probs = pd.DataFrame(cumulative_probs, columns=prob_cols)
    final_df = pd.concat([df_nodes, df_probs], axis=1)

    if train_eval:
        final_df.to_csv(f"results_{run_id}.csv", index=False)
    else:
        final_df.to_csv(f"results_{run_id}_test.csv", index=False)

    end_time = time.time()
    tot_time = end_time - start_time
    print(f"finished in {(tot_time) / 60} minutes")
