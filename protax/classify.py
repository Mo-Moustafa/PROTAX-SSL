import sys

from protax import protax_utils
import protax.model as model

import numpy as np
import jax
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
            # if not math.isclose(child_probs.sum(), 1.0, rel_tol=1e-5):
            #     print("branch probabilities do not sum to 1.0")
            #     print("prob sum: ", child_probs.sum(), "\n")

            local_idx = np.argmax(child_probs)          # Find the index of the max probability WITHIN that subset
            current_prob = child_probs[local_idx]
            current_parent = children[local_idx]        # Map that local index back to the global tree index
            
            classified_layers[i] = current_parent
            classified_probs[i] = current_prob
        else:
            break               # No children found; hierarchy ends early (node was either unknown or missing)

    return classified_layers, classified_probs


def create_design_matrices(variant, base_dir, q_dir, tree, dist_scaling, N, params, train_eval, loo_map):
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

    if variant in ("bert", "mamba", "hybrid_lin", "mlp"):
        print("Reading Embeddings...")
        base_embeddings = np.load(base_dir)
        q_embeddings = np.load(q_dir)

        if variant in ("bert", "mamba"):
            base_embeddings = jnp.array(base_embeddings[variant])
            print("base embeddings.shape: ", base_embeddings.shape)
            base_length = int(base_embeddings.shape[0])

            q_embeddings = jnp.array(q_embeddings[variant])
            print("query embeddings.shape: ", q_embeddings.shape)
            Q = int(q_embeddings.shape[0])
            
        else:
            bert_base = jnp.array(base_embeddings["bert"])
            mamba_base = jnp.array(base_embeddings["mamba"])
            assert bert_base.shape == mamba_base.shape
            base_length = int(bert_base.shape[0])

            bert_q = jnp.array(q_embeddings["bert"])
            mamba_q = jnp.array(q_embeddings["mamba"])
            assert bert_q.shape == mamba_q.shape
            Q = int(bert_q.shape[0])

    elif variant == "og":
        seq_list, ok_list = protax_utils.read_refs(q_dir, padding_len=tree.max_seq_length)
        print("Read reference sequences successfully")
        Q = int(seq_list.shape[0])
        base_length = int(tree.refs.shape[0])
    
    else:
        raise ValueError(f"Model {variant} not supported")

    X_all = None    
    for i in tqdm(range(Q), desc="Computing design matrices", file=sys.stderr, dynamic_ncols=True, mininterval=5):
        # mask the tree for the current sample
        if (train_eval and i < base_length):
            new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
            tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)
        
        elif (loo_map is not None and i in loo_map):
            new_n2s, new_ns = protax_utils.mask_n2s(n2s, node_state, loo_map[i])
            tree = tree._replace(node2seq=new_n2s, node_state=new_ns)
        else:
            tree = tree._replace(node2seq=n2s, node_state=node_state)
        
        # compute the distance matrix based on the model
        if variant in ("bert", "mamba"):
            dists = model.cosine_dist(q_embeddings[i], base_embeddings)
            X = model.get_X(dists, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        elif variant in ("hybrid_lin", "mlp"):
            dists_bert = model.cosine_dist(bert_q[i], bert_base)
            dists_mamba = model.cosine_dist(mamba_q[i], mamba_base)
            X = model.get_hybrid_X(dists_bert, dists_mamba, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        elif variant == "og":
            q_seq = seq_list[i]
            ok = ok_list[i]
            dists = model.p_dist(q_seq, ok, tree.refs, tree.ok_pos)
            X = model.get_X(dists, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        x_host = np.asarray(jax.device_get(X), dtype=np.float32)
        if X_all is None:
            n_nodes, F_dim = x_host.shape
            X_all = np.zeros((Q, n_nodes, F_dim), dtype=np.float32)
        X_all[i] = x_host

    return X_all, Q


def classify_file(variant, q_dir, base_dir, par_dir, tax_dir, dist_scaling, run_id, train_eval, X_all=None, loo_map=None):

    """
    Process a batch of queries given a model and taxonomy directory
    """

    tree, params, N, segnum = protax_utils.read_model_jax(par_dir, tax_dir)

    if X_all is None:
        X_all, Q = create_design_matrices(variant, base_dir, q_dir, tree, dist_scaling, N, params, train_eval, loo_map)
    else:
        Q = X_all.shape[0]

    start_time = time.time()

    tree_for_bprob = tree
    cls_chunk = 256
    vmapped_bprob = jax.jit(jax.vmap(lambda x: model.fill_bprob(x, params.beta, tree_for_bprob, segnum)))

    res = []
    final_probs = []

    for start in tqdm(range(0, Q, cls_chunk),desc="Classifying queries",file=sys.stderr,dynamic_ncols=True,mininterval=5):
        end = min(start + cls_chunk, Q)
        X_batch = jnp.asarray(X_all[start:end], dtype=jnp.float32)
        probs_batch = vmapped_bprob(X_batch)
        probs_batch = jax.device_get(jax.block_until_ready(probs_batch))
        for j in range(end - start):
            classified_layers, classified_probs = hierarchy_classification(probs_batch[j], tree_for_bprob)
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
    print(f"finished in {(tot_time) / 60} minutes")