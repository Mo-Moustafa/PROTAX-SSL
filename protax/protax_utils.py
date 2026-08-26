"""
Functions for reading from files used by PROTAX
"""
import jax
import numpy as np
import jax.numpy as jnp
from .taxonomy import TaxTree, ProtaxModel
from scipy.sparse import csr_matrix
# import scipy.sparse as sp
from jax.experimental import sparse
from .ops import knn, knn_v2

import pandas as pd
from pathlib import Path


def read_params(pdir):
    """
    Read parameters from text file
    """
    print("reading parameters")
    
    with open(pdir, 'r') as f:
        res = []
        for l in f.readlines():
            res.append(jnp.fromstring(l, sep=" "))
        return res


def read_scalings(pdir):
    """
    Read scalings from text file
    """
    print("reading scalings")
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(l.split(" "))

    res = np.array(res)[:, 1::2].astype("float32")
    return res


def read_taxonomy(tdir):
    """
    Read taxonomy tree from taxonomy.priors file
    """
    print("reading taxonomy file")
    f = open(tdir)
    node_dat = f.readlines()

    # making result arrays
    N = len(node_dat)
    parents = []  # parent of node at nid index belongs to (segments)
    child_col = []  # child nid for parent
    unks = np.zeros(N, dtype=bool)
    priors = np.zeros(N)
    layers = np.zeros(N, dtype=int)
    
    for l in node_dat:
        l = l.strip("\n")

        # collecting taxon data
        nid, pid, lvl, name, prior, _ = l.split("\t")
        nid, pid, lvl, prior = (int(nid), int(pid), int(lvl), float(prior))
        name = name.split(",")[-1]

        # assign children
        parents.append(pid)
        child_col.append(nid)
        
        # information about node
        unks[nid] = name=="unk"     # mask for unknown nodes
        layers[nid] = lvl            # level of each node
        priors[nid] = prior      # prior probability of each node
    
    # making segments
    child_col = np.array(child_col)
    parents = np.array(parents)
    parents[0] = -1                 # root is at index 0, and has nid = 0
    unq = np.unique(parents)
    segments = np.searchsorted(unq, parents)    # I'm not sure if this matches what's in conversion.py. (it doesn't!)

    descendants = get_descendants(parents, N, layers)

    # convert rest to cupy
    return segments, unks, layers, priors, descendants, parents


def get_descendants(nodes, N, layers):
    """
    Get the path to the root for each node.
    """

    # TODO fix this
    res = (np.ones((nodes.shape[0], 8))*N).astype(int)      # Initialize the result array with N to be treated as out of bounds when classified.
    for n in range(1, nodes.shape[0]):
        
        curr_nid = nodes[n]  # parent of node n
        res[n][layers[curr_nid]] = curr_nid # add parent to it's layer in the path.

        # while curr isn't root, walk up the tree and add parents to the path.
        while curr_nid != 0:    # assumes root is 0
            curr_nid = nodes[curr_nid]  # parent of parent
            res[n][layers[curr_nid]] = curr_nid
            
        res[n][layers[n]] = n # add self to the path.
    
    res[0][0] = 0 # root's path is just itself.
    return res


def read_refs(ref_dir, padding_len=None):
    """
    Read reference sequences from refs.aln file
    """

    print("reading sequences")
    f = open(ref_dir)
    ref_list = []
    ok_pos = []

    while True:
        name = f.readline().strip('\n').split('\t')[0]
        seq = f.readline().strip('\n')

        if not seq:
            break  # EOF
    
        if padding_len is not None:
            seq = seq.ljust(padding_len, 'N')

        seq_bits = get_seq_bits(seq)
        ref_list.append(np.packbits(seq_bits[:4], axis=None))
        ok_pos.append(np.packbits(seq_bits[4], axis=None))

    return jnp.array(ref_list), jnp.array(ok_pos)


def assign_refs(seq2tax_dir):
    """
    Assign reference sequences to nodes from file
    """

    print("\nassigning reference sequences to taxa")
    f = open(seq2tax_dir)

    seqs = np.array([], dtype=int)
    nids = np.array([], dtype=int)

    # assigning ref seq indices
    for n, l in enumerate(f.readlines()):
        nid, num_refs, ref_idx = l.split('\t')
        nid = int(nid)

        seq_ids = np.fromstring(ref_idx, sep=" ").astype(int)
        seqs = np.concatenate([seqs, seq_ids])
        nids = np.concatenate([nids, np.full(seq_ids.shape, nid)])

    return nids, seqs


def get_seq_bits(seq_str):
    """
    Convert seqence string to bit representation
    """
    seq_chars = np.frombuffer(seq_str.encode('ascii'), np.int8)
    a = seq_chars == 65
    t = seq_chars == 84
    g = seq_chars == 71
    c = seq_chars == 67
    ok = np.logical_or.reduce([a, t, g, c])

    seq_bits = np.array([a, t, g, c, ok])
    return seq_bits


def assign_params(beta, sc, lvl):
    """
    Assign parameters to each node given the levels each node is in
    """
    return np.take(beta, lvl, axis=0), np.take(sc, lvl, axis=0)


def read_model(model_dir, tree_ranks):
    """
    Read parameters from text file
    """
    print("reading parameters")
    model_dir = Path(model_dir)
    parameters = np.load(model_dir.resolve())

    beta_conc = parameters['beta']
    scalings_conc = parameters['scalings']

    # Applying beta mismatch fix to old trained models (should be removed eventually)
    if beta_conc.shape[0] == 7:
        print("Padding beta to (8, 4)...")
        zero_row = np.zeros((1, 4)) # Use zeros, not ones
        beta_conc = np.vstack([zero_row, beta_conc])   
    if scalings_conc.shape[0] == 7:
        print("Padding scalings to (8, 4)...")
        padding_row = np.array([[0, 1, 0, 1]])
        scalings_conc = np.vstack([padding_row, scalings_conc])

    beta, scalings = assign_params(beta_conc, scalings_conc, tree_ranks)

    # Allows to load scalings automatically from single or hybrid models
    if scalings.shape[1] > 4:
        sc_mean = jnp.array(scalings[:, [0, 2, 4, 6]])
        sc_var = jnp.array(scalings[:, [1, 3, 5, 7]])
    else:
        sc_mean = jnp.array(scalings[:, [0, 2]])
        sc_var = jnp.array(scalings[:, [1, 3]])
    
    parameters = ProtaxModel(
        beta=jnp.array(beta),
        beta_conc=jnp.array(beta_conc),
        sc_conc=jnp.array(scalings_conc),
        sc_mean=sc_mean,
        sc_var=sc_var
    )

    return parameters


def read_tree(tax_dir):
    """
    Read tree from taxonomy.npz
    """
    tax_dir = Path(tax_dir)
    tax = np.load(tax_dir.resolve(), allow_pickle=True)
    
    refs = jnp.array(tax['refs'])
    ok_pos = jnp.array(tax['ok_pos'])

    prior = jnp.array(tax['prior'])
    paths = jnp.array(tax['paths'])
    node_state = jnp.array(tax['node_state'])
    seg = jnp.array(tax['segments'])
    segnum = int(jnp.max(seg) + 1)
    
    N = seg.shape[0]
    indices = tax['n2s_indices']
    indptr = tax['n2s_indptr']
    data = np.ones(len(indices))
    node2seq = sparse.CSR((data, indices, indptr), shape=(N, refs.shape[0]))

    tree = TaxTree(
        refs=refs,
        ok_pos=ok_pos,
        segments=seg,
        node2seq=node2seq,
        paths=paths,
        node_state=node_state,
        prior=prior,
        ranks=tax['ranks'],
        parents=tax['parents'],
        max_seq_length=tax['max_seq_length']
    )

    return tree, N, segnum

    
def get_targets(target_dir):

    targ = pd.read_csv(target_dir).to_numpy()
    res = np.zeros((targ.shape[0],), dtype=np.int32)

    for i in range(targ.shape[0]):
        for j in range(targ.shape[1]):
            node_id = targ[i][j]
            if node_id != -1:
                res[i] = node_id
            else:
                break

    return jnp.array(res)


def read_query(q, padding_len=None):
    """
    Encodes a single query sequence into bit representation
    """
    if padding_len is not None:
        q = q.ljust(padding_len, 'N')

    s = get_seq_bits(q)
    return jnp.array(np.packbits(s[:4], axis=None)), jnp.array(np.packbits(s[4], axis=None))


def str2batch_query(q):
    """
    Encodes a batch of query sequences into bit representation
    """
    queries = []
    ok_pos = []
    for i in q:
        curr = get_seq_bits(i)
        queries.append(np.packbits(curr[:4], axis=None))
        ok_pos.append(np.packbits(curr[4], axis=None))
    
    return jnp.array(queries), jnp.array(ok_pos)


def mask_n2s(n2s, node_state, i):

    is_target = (n2s.indices == i)
    negative_mask = 1 - 2 * is_target.astype(n2s.data.dtype)
    zero_mask = 1 - is_target.astype(n2s.data.dtype)

    new_data = n2s.data * negative_mask
    new_n2s = sparse.CSR((new_data, n2s.indices, n2s.indptr), shape=n2s.shape)

    new_data = n2s.data * zero_mask
    n2s = sparse.CSR((new_data, n2s.indices, n2s.indptr), shape=n2s.shape)

    ones = jnp.ones((n2s.shape[1], 1), dtype=n2s.data.dtype)
    row_sums = n2s @ ones 
    has_refs = row_sums > 0
    empty = jnp.logical_not(has_refs)
    new_node_state = jnp.logical_or(node_state[:, :1], empty)
    new_node_state = jnp.concatenate((new_node_state, has_refs), axis=1)
    
    return new_n2s, new_node_state


# -------------------------------------- Scalings --------------------------------------

def get_X_raw(dists, tree, N):
    """
    Compute the raw (unscaled) KNN design features (N, 2) for standardization.
    Used when recomputing sc_mean/sc_var after changing the similarity measure.
    """
    node2seq = tree.node2seq
    new_dat = jnp.take(dists, node2seq.indices)
    return knn(node2seq.indptr, node2seq.indices, new_dat, N)
    # return knn_v2(node2seq.indptr, node2seq.indices, new_dat, N)

def compute_scalings_from_raw_features_streaming(raw_features_iter, node_layer, has_refs, eps=1e-8):
    """
    Same as compute_scalings_from_raw_features but consumes an iterator of (N, 2)
    arrays so that not all need to be in memory. Use this for large numbers of
    samples to avoid OOM.

    Args:
        raw_features_iter: Iterator yielding (N, 2) arrays (e.g. from get_X_raw).
        node_layer, has_refs, eps: As in compute_scalings_from_raw_features.

    Returns:
        scalings: (num_levels, 4) array, same format as compute_scalings_from_raw_features.
    """
    node_layer = np.asarray(node_layer).ravel()
    has_refs = np.asarray(has_refs).ravel().astype(bool)
    if node_layer.size == 0:
        raise ValueError("node_layer is empty")
    # 0-based ranks (e.g. 0..7): one bucket per rank, row r <-> taxonomic rank r.
    num_levels = int(np.max(node_layer)) + 1

    # Per level: count, mean0, mean1, M2_0, M2_1 (Welford's online variance)
    count = np.zeros(num_levels, dtype=np.int64)    # How many scalar pairs (x0, x1) have been seen for that level.
    mean0 = np.zeros(num_levels, dtype=np.float64)    # Mean of x0 for that level.
    mean1 = np.zeros(num_levels, dtype=np.float64)    # Mean of x1 for that level.
    M2_0 = np.zeros(num_levels, dtype=np.float64)    # Sum of squared differences from the mean for x0 for that level.
    M2_1 = np.zeros(num_levels, dtype=np.float64)    # Sum of squared differences from the mean for x1 for that level.

    for raw in raw_features_iter:
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("Each item must be (N, 2)")
        for lev in range(num_levels):
            mask = (node_layer == lev) & has_refs
            if not np.any(mask):
                continue
            vals = raw[mask, :]  # (N_lev, 2)
            n = vals.shape[0]
            for k in range(n):
                x0, x1 = vals[k, 0], vals[k, 1]
                c = count[lev]
                count[lev] += 1
                d0 = x0 - mean0[lev]
                d1 = x1 - mean1[lev]
                mean0[lev] += d0 / count[lev]
                mean1[lev] += d1 / count[lev]
                d0_new = x0 - mean0[lev]
                d1_new = x1 - mean1[lev]
                M2_0[lev] += d0 * d0_new
                M2_1[lev] += d1 * d1_new

    scalings = np.zeros((num_levels, 4), dtype=np.float32)
    for lev in range(num_levels):
        c = count[lev]
        if c == 0:
            scalings[lev] = [0.0, 1.0, 0.0, 1.0]
        else:
            v0 = (M2_0[lev] / c) + eps
            v1 = (M2_1[lev] / c) + eps
            scalings[lev] = [float(mean0[lev]), v0, float(mean1[lev]), v1]
    return scalings


# Faster vectorized streaming version (no per-element Python loops)
def compute_scalings_from_raw_features_streaming_vectorized(
    raw_features_iter, node_layer, has_refs, eps=1e-8
):
    """
    Vectorized streaming version computing per-level mean/variance for the
    two raw KNN features (feature 0 and feature 1).

    It keeps running sums and sum-of-squares per taxonomic level, which is
    equivalent to computing population variance over all (query, node) pairs
    that satisfy `has_refs == True` and `node_layer == level`.

    `node_layer` must use the same 0-based rank convention as `taxonomy.npz`
    ``ranks`` (e.g. 0..7). Output row ``r`` is the scaling for rank ``r``.

    `has_refs` can be either:
    - 1D bool array of shape `(N,)` (static mask; original behavior)
    - 2D bool array of shape `(num_queries, N)` (per-query mask)
    """
    node_layer = np.asarray(node_layer).ravel()
    if node_layer.size == 0:
        raise ValueError("node_layer is empty")
    # 0-based ranks: shape (max_rank + 1, 4), row r matches assign_params / read_model_jax.
    num_levels = int(np.max(node_layer)) + 1

    count = np.zeros(num_levels, dtype=np.int64)
    sum0 = np.zeros(num_levels, dtype=np.float64)
    sum1 = np.zeros(num_levels, dtype=np.float64)
    sumsq0 = np.zeros(num_levels, dtype=np.float64)
    sumsq1 = np.zeros(num_levels, dtype=np.float64)

    has_refs_arr = np.asarray(has_refs)
    if has_refs_arr.ndim == 1:
        has_refs_static = has_refs_arr.astype(bool).ravel()
        if node_layer.shape[0] != has_refs_static.shape[0]:
            raise ValueError("node_layer and has_refs must have the same length")

        # Include rank 0 (root) when it has refs; ranks are 0-based.
        contributing = has_refs_static & (node_layer >= 0) & (node_layer < num_levels)

        lev_idx = node_layer[contributing].astype(np.int64)
        if lev_idx.size == 0:
            return np.array([[0.0, 1.0, 0.0, 1.0]] * num_levels, dtype=np.float32)

        # This does not change across samples.
        count_per_query = np.bincount(lev_idx, minlength=num_levels).astype(np.int64)

        for raw in raw_features_iter:
            raw = np.asarray(raw)
            if raw.ndim != 2 or raw.shape[1] != 2:
                raise ValueError("Each item must be (N, 2)")

            x0 = raw[contributing, 0].astype(np.float64, copy=False)
            x1 = raw[contributing, 1].astype(np.float64, copy=False)

            count += count_per_query
            sum0 += np.bincount(lev_idx, weights=x0, minlength=num_levels)
            sum1 += np.bincount(lev_idx, weights=x1, minlength=num_levels)
            sumsq0 += np.bincount(lev_idx, weights=x0 * x0, minlength=num_levels)
            sumsq1 += np.bincount(lev_idx, weights=x1 * x1, minlength=num_levels)

    elif has_refs_arr.ndim == 2:
        # Per-query mask: has_refs_arr[qi, node] tells whether that node contributes
        # for query qi (after masking out that query's reference(s)).
        if has_refs_arr.shape[1] != node_layer.shape[0]:
            raise ValueError("For 2D has_refs, has_refs.shape[1] must equal len(node_layer)")

        has_refs_arr = has_refs_arr.astype(bool)
        num_queries_expected = has_refs_arr.shape[0]

        valid_level = (node_layer >= 0) & (node_layer < num_levels)
        num_queries_seen = 0

        for raw in raw_features_iter:
            if num_queries_seen >= num_queries_expected:
                raise ValueError(
                    "raw_features_iter contains more samples than has_refs (2D) provides"
                )

            raw = np.asarray(raw)
            if raw.ndim != 2 or raw.shape[1] != 2:
                raise ValueError("Each item must be (N, 2)")

            mask = has_refs_arr[num_queries_seen] & valid_level
            lev_idx = node_layer[mask].astype(np.int64)
            if lev_idx.size == 0:
                num_queries_seen += 1
                continue

            x0 = raw[mask, 0].astype(np.float64, copy=False)
            x1 = raw[mask, 1].astype(np.float64, copy=False)

            count += np.bincount(lev_idx, minlength=num_levels).astype(np.int64)
            sum0 += np.bincount(lev_idx, weights=x0, minlength=num_levels)
            sum1 += np.bincount(lev_idx, weights=x1, minlength=num_levels)
            sumsq0 += np.bincount(lev_idx, weights=x0 * x0, minlength=num_levels)
            sumsq1 += np.bincount(lev_idx, weights=x1 * x1, minlength=num_levels)

            num_queries_seen += 1

        if num_queries_seen != num_queries_expected:
            raise ValueError(
                f"raw_features_iter produced {num_queries_seen} samples, but has_refs (2D) has {num_queries_expected}"
            )

    else:
        raise ValueError("has_refs must be a 1D or 2D array")

    scalings = np.zeros((num_levels, 4), dtype=np.float32)
    nonzero = count > 0

    if np.any(nonzero):
        c = count[nonzero].astype(np.float64)
        mean0 = sum0[nonzero] / c
        mean1 = sum1[nonzero] / c
        var0 = (sumsq0[nonzero] / c) - mean0 * mean0
        var1 = (sumsq1[nonzero] / c) - mean1 * mean1

        # Numerical safety: variance should not be negative; add eps for stability.
        var0 = np.maximum(var0, 0.0) + eps
        var1 = np.maximum(var1, 0.0) + eps

        scalings[nonzero, 0] = mean0.astype(np.float32)
        scalings[nonzero, 2] = mean1.astype(np.float32)
        scalings[nonzero, 1] = var0.astype(np.float32)
        scalings[nonzero, 3] = var1.astype(np.float32)

    empty = ~nonzero
    if np.any(empty):
        scalings[empty] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    return scalings


# if __name__ == "__main__":
    # convert_model(r"/home/roy/Documents/PROTAX-dsets/30k_small")
    # convert_taxonomy(r"/home/roy/Documents/PROTAX-dsets/30k_small")