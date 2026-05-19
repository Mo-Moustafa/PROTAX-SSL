"""Classify with a hybrid + per-rank MLP checkpoint (mlp_* arrays in npz)."""
import sys

from .protax_utils import read_model_jax, mask_n2s
import protax.model_bert as model
import protax.model_hybrid_mlp as mlp_model

import numpy as np
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


def classify_file(qdir, base_dir, par_dir, tax_dir, dist_scaling, run_id, train_eval):
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

    if train_eval:
        n2s = tree.node2seq
        node_state = tree.node_state
        print("Read nodes successfully")

    res = []
    final_probs = []

    start_time = time.time()

    for i in tqdm(
        range(bert_query.shape[0]),
        desc="Classifying queries (hybrid_mlp)",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=5,
    ):
        query_bert = bert_query[i]
        query_mamba = mamba_query[i]
        q_dist_bert = model.seq_dist(query_bert, bert_base)
        q_dist_mamba = model.seq_dist(query_mamba, mamba_base)

        if train_eval and i < bert_base.shape[0]:
            new_n2s, new_node_state = mask_n2s(n2s, node_state, i)
            tree = tree._replace(node2seq=new_n2s, node_state=new_node_state)
        elif train_eval and i >= bert_base.shape[0]:
            tree = tree._replace(node2seq=n2s, node_state=node_state)
        else:
            pass

        probs = mlp_model.get_probs_hybrid_mlp(
            q_dist_bert,
            q_dist_mamba,
            tree,
            params,
            segnum,
            N,
            mlp_params,
            dist_scaling,
            pt_min,
            pt_gap,
        ).block_until_ready()

        classified_layers, classified_probs = hierarchy_classification(probs, tree)

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
    print(f"finished in {(tot_time) / 3600} hours")
