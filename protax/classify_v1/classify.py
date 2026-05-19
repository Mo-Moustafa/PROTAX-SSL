import os
import sys

from .protax_utils import read_model_jax, read_refs, mask_n2s
from .model import get_probs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

import time
import joblib


def hierarchy_classification(probs, tree):
    max_rank = tree.ranks.max() + 1             # +1 to count for the 0 based indexing
    classified_layers = np.full(max_rank, -1)
    classified_probs = np.full(max_rank, np.nan)
    
    # Root initialization
    current_parent = 0
    classified_layers[0] = 0
    classified_probs[0] = 1.0

    for i in range(1, max_rank):
        children = np.where(tree.parents == current_parent)[0]      # Get indices of children for the current parent

        levels = tree.ranks[children]
        # Debugging: Check if the children all belong to the same level
        if not np.all(levels == i):
            print("children do not belong to the same level!")
        
        if children.size > 0:
            child_probs = probs[children]               # Extract only the probabilities of the children
            
            # Debugging: Check if the branch probabilities sum to 1.0
            if not math.isclose(child_probs.sum(), 1.0, rel_tol=1e-5):
                print("branch probabilities do not sum to 1.0")
                print("prob sum: ", child_probs.sum(), "\n")

            local_idx = np.argmax(child_probs)          # Find the index of the max probability WITHIN that subset
            current_prob = child_probs[local_idx]
            current_parent = children[local_idx]        # Map that local index back to the global tree index
            
            classified_layers[i] = current_parent
            classified_probs[i] = current_prob
        else:
            break               # No children found; hierarchy ends early (node was either unknown or missing)

    return classified_layers, classified_probs


def classify_file(qdir, par_dir, tax_dir, dist_scaling, run_id, train_eval):
    """
    Process a batch of queries given a model and taxonomy directory
    """

    tree, params, N, segnum = read_model_jax(par_dir, tax_dir)
    seq_list, ok_list = read_refs(qdir, padding_len=tree.max_seq_length)

    if dist_scaling == "power":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = joblib.load("models/scalings/gap_power_transformer.joblib")
    elif dist_scaling == "hybrid":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = None
    else:
        pt_min = None
        pt_gap = None

    if train_eval:
        n2s = tree.node2seq
        node_state = tree.node_state
        print("Read nodes successfully")


    res = []
    final_probs = []  # Store final probabilities

    start_time = time.time()

    for i in tqdm(range(len(seq_list)), desc="Classifying queries", file=sys.stderr, dynamic_ncols=True, mininterval=5):
        q = seq_list[i]
        ok = ok_list[i]

        if train_eval and i < len(seq_list):
            new_n2s, new_node_state = mask_n2s(n2s, node_state, i)
            tree = tree._replace(node2seq=new_n2s, node_state=new_node_state)
        elif train_eval and i >= len(seq_list):
            tree = tree._replace(node2seq=n2s, node_state=node_state)
        else:
            pass

        probs = get_probs(q, ok, tree, params, segnum, N, dist_scaling, pt_min, pt_gap).block_until_ready()
        # Hierarchical Function
        classified_layers, classified_probs = hierarchy_classification(probs, tree)

        res.append(classified_layers)
        final_probs.append(classified_probs)
    
    # Compute cumulative product across the levels (axis 1)
    cumulative_probs = np.array(final_probs)
    cumulative_probs = np.cumprod(cumulative_probs, axis=1)

    # saving results
    rank_names = ['root', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
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
