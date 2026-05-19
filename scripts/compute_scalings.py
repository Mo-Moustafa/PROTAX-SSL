import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp
from tqdm import tqdm
import sys
import os

from protax import protax_utils
import protax.model as model
from BarcodeBERT.barcodebert import BarcodeBERT


if __name__ == "__main__":
    
    data_dir = "datasets/canadian_invertebrates/mamba"
    method = "BERT"
    # method = "OG"
    prefix = f"canadian_invertebrates_mamba_{method}" 
    dist_scaling = "none"  # must match training/inference dist scaling

    tree, params, N, segnum = protax_utils.read_model_jax("models/scalings/plain.npz", f"{data_dir}/taxonomy.npz")
    node_layer = tree.ranks
    has_refs = np.array(tree.node_state[:, 1], dtype=bool)

    if method == "BERT":
        train_dir = f"{data_dir}/train_embeddings.npz"
        bert = BarcodeBERT()
        print("Reading base embeddings...")
        base_embeddings = bert.load_embeddings(train_dir)

    elif method == "OG":
        train_dir = f"{data_dir}/train.aln"
        seq_list, ok_list = protax_utils.read_refs(train_dir, padding_len=tree.max_seq_length)
        distances = []
        for i,seq in enumerate(seq_list):
            distances.append(model.p_dist(seq, ok_list[i], tree.refs, tree.ok_pos))

        distances = np.array(distances)

    n2s = sp.csr_matrix(
        (tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr),
        shape=tree.node2seq.shape,
    )
    node_state = np.expand_dims(np.array(tree.node_state)[:, 0], 1)

    def raw_features_iter():
        if method == "BERT":
            n_queries = int(base_embeddings.shape[0])
        else:
            n_queries = int(distances.shape[0])

        for i in tqdm(range(n_queries), desc="Computing scalings", file=sys.stderr, dynamic_ncols=True, mininterval=5):
            if method == "BERT":
                # Compute distances one query at a time (like classify_bert.py) to avoid materializing NxN.
                dists = np.asarray(model.cosine_dist(base_embeddings[i], base_embeddings))
            else:
                dists = distances[i]

            new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
            masked_tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)

            raw = np.asarray(protax_utils.get_X_raw(dists, masked_tree, N))
            if dist_scaling == "log":
                raw = np.log(raw + 1e-8)
            yield raw

    scalings = protax_utils.compute_scalings_from_raw_features_streaming_vectorized(
        raw_features_iter(), node_layer, has_refs, eps=1e-8
    )
    
    if scalings.shape[0] == 7:
        padding_row = np.array([[0, 1, 0, 1]])
        scalings = np.vstack([padding_row, scalings])

    model = np.load('models/scalings/plain.npz')
    model_dict = dict(model)
    model_dict['scalings'] = scalings

    i = 1
    while os.path.exists(f"models/scalings/{prefix}_{i}.npz"):
        i += 1

    np.savez(f"models/scalings/{prefix}_{i}.npz", **model_dict)
    print(f"Saved scalings {prefix}_{i}: \n", scalings)
