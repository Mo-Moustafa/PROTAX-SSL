import jax
from typing import NamedTuple
from jax.experimental import sparse

# class CSRWrapper(NamedTuple):
#     """
#     Jax wrapper for CSR matrix. Represents the mapping between nodes and reference sequences.
#     """
#     data: jax.Array # [R] 1D array of ones.
#     indices: jax.Array # [R] column indices for each value in data (reference IDs).
#     indptr: jax.Array # [N+1] Pointer to the start of each row in the CSR matrix.
#                       # Start: The values for row i begin at data[indptr[i]]
#                       # End: The values for row i end at data[indptr[i+1]]
                
#     shape: tuple    # (N, R)

# TODO: Make this class mutable
class TaxTree(NamedTuple):
    """
    State of the taxonomic tree
    N = total number of nodes
    R = total number of reference sequences
    """
    refs: jax.Array                  # [R, n_channels]  All reference sequences encoded. Likely each reference sequence is represented by 5 channels (e.g. A/C/G/T/N ).
    ok_pos: jax.Array                # [R, n_channels]  Bit-packed boolean mask of "valid" nucleotide positions for each reference sequence. (masks out "-")
    segments: jax.Array              # [N] Index of the group of each node. ex: [0, 0, 0, 1, 2, 2, 2, ...]
    node2seq: sparse.CSR             #  A sparse matrix mapping between nodes and reference sequences.
    paths: jax.Array                 # [N, L_max] Ancestor table, where each row is the node’s full path through the taxonomy (root → … → node)
    node_state: jax.Array            # [N, 2]   Per-node state features [empty but known, has_refs].
    prior: jax.Array                 # [N] Prior probability of each node in the taxonomy.
    ranks: jax.Array                 # [N] Rank of each node in the taxonomy.
    parents: jax.Array                # [N] Parent of each node in the taxonomy.
    max_seq_length: int               # [1] Maximum sequence length of the reference sequences.

class ProtaxModel(NamedTuple):
    """
    Holds the parameters for PROTAX model
    """
    
    beta: jax.Array                  # [N, 4]   Beta parameters for the PROTAX model.
    sc_mean: jax.Array              # [N, 2]   Mean scaling factors for the PROTAX model.
    sc_var: jax.Array               # [N, 2]   Variance scaling factors for the PROTAX model.