import sys

from .protax_utils import read_model_jax, mask_n2s
import protax.model_bert as model

import numpy as np
import jax.numpy as jnp
import pandas as pd
import math

import time
from tqdm import tqdm

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



def classify_file(qdir, base_dir, par_dir, tax_dir, dist_scaling, run_id, train_eval):

    """
    Process a batch of queries given a model and taxonomy directory
    """

    tree, params, N, segnum = read_model_jax(par_dir, tax_dir)
    
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
    final_probs = []  # Store final probabilities

    start_time = time.time()

    for i in tqdm(range(bert_query.shape[0]), desc="Classifying queries", file=sys.stderr, dynamic_ncols=True, mininterval=5):
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

        # conditional probs of each node relative to its siblings
        probs = model.get_probs_hybrid(q_dist_bert, q_dist_mamba, tree, params, segnum, N, dist_scaling, pt_min, pt_gap).block_until_ready()

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








# if __name__ == "__main__":
#     qdir = ""
#     train_dir = "datasets/mycoai/full_tax/train_embeddings.npz"
#     par_dir = "models/model_2813633.npz"
#     tax_dir = "datasets/mycoai/full_tax/taxonomy.npz"
#     dist_scaling = "None"
#     run_id = "0"

#     classify_file(qdir, train_dir, par_dir, tax_dir, dist_scaling, run_id, train_eval=True)

# from protax.model_bert import *
# from .ops import knn, knn_v2
# if __name__ == "__main__":
#     np.set_printoptions(threshold=np.inf)
#     par_dir = "models/model_2385148.npz"
#     tax_dir = "datasets/canadian_invertebrates/debug/taxonomy.npz"
#     tree, params, N, segnum = read_model_jax(par_dir, tax_dir)
#     # N is the number of unique nodes
#     # segnum is the max nodeID + 1

#     base_embeddings = jnp.array(np.load("datasets/canadian_invertebrates/debug/train_embeddings.npz")["embeddings"])
#     test_embeddings = jnp.array(np.load("datasets/canadian_invertebrates/debug/test_embeddings.npz")["embeddings"])
#     distances = model.seq_dist(test_embeddings, base_embeddings)
#     q_dist = distances[100]
#     node2seq = tree.node2seq
#     new_dat = jnp.take(q_dist, node2seq.indices)  # Get distances for sequences under each node
#     X = knn(node2seq.indptr, node2seq.indices, new_dat, N)  # Compute KNN over sequences under each node
#     X = (((X - 0) / jnp.sqrt(1)).T*(tree.node_state[:, 1])).T  # Standardize features then multiply by has_refs mask
#     X = jnp.concatenate((tree.node_state, X), axis=1)   # Concatenate node state (known, has_refs) and features

#     bprobs = fill_bprob(X, params.beta, tree, segnum)
#     probs = jnp.prod(bprobs, axis=1)

    # old classification
    # probs = jnp.take(probs, tree.paths, fill_value=-1)
    # classified_layer = jnp.argmax(probs, axis=0)
    # classified_probs = jnp.max(probs, axis=0)[-1]

    # classified_layer, classified_probs = hierarchy_classification(probs, tree)
    # adj_list = build_adjacency_list(tree.parents, N)
    # classified_layer, classified_probs = hierarchy_classification_fast(probs, adj_list, tree.ranks.max() + 1)

    # print(f"classified_layer:\n", classified_layer, "\n\n")
    # print(f"classified_prob:\n", classified_probs, "\n\n")