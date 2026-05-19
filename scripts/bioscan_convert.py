import pandas as pd
import numpy as np
import scipy.sparse as sparse
from bioscan_dataset import CanadianInvertebrates
from protax.protax_utils import get_descendants, get_seq_bits
from Bio import SeqIO


# ------------------------------------------------ For MyCoAI Datasets ------------------------------------------------

def remove_question_marks(df):
    initial_count = len(df)
    rows_with_question_mark = df.isin(['?']).any(axis=1)
    df = df[~rows_with_question_mark]
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows containing '?'")

    return df


def mycoai_fasta_to_df(file_path):
    # Define the expected levels in the taxonomy
    levels = {'p__': 'phylum', 'c__': 'class', 
              'o__': 'order', 'f__': 'family', 'g__': 'genus', 's__': 'species'}
    
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        # prefill with NA
        row = {name: pd.NA for name in levels.values()}
        
        # Split header: Accession|Taxonomy|ID
        parts = record.description.split('|')
        # row['accession'] = parts[0]
        row['dna_barcode'] = str(record.seq)
        
        # Parse taxonomy string if it exists
        if len(parts) > 1:
            taxa_parts = parts[1].split(';')
            for item in taxa_parts:
                prefix = item[:3] # Get 'k__', 'p__', etc.
                if prefix in levels:
                    row[levels[prefix]] = item[3:].replace('_', ' ')
        
        data.append(row)

    hierarchy = ["phylum", "class", "order", "family", "genus", "species"]

    df = pd.DataFrame(data)
    # df = remove_question_marks(df)
    
    return df, hierarchy


# ------------------------------------------------ For BioScan Datasets ------------------------------------------------
def load_dataset(split):
    dataset = CanadianInvertebrates(root="~/Datasets/bioscan/", target_type=["phylum", "class", "order", "family", "genus", "species"], split=split, download=False)
    df = dataset.metadata
    hierarchy = ["phylum", "class", "order", "family", "genus", "species"]
    df = df[hierarchy + ["dna_barcode"]]
    return df, hierarchy

### ------------------------------------------------ Creating Taxonomy ------------------------------------------------
# def assign_priors(df, hierarchy):

#     total_rows = len(df)
#     taxon_to_prior_map = {}

#     for rank in hierarchy:
#         counts = df[rank].value_counts(dropna=False)
#         priors = counts / total_rows
        
#         taxon_to_prior_map[rank] = priors.to_dict()

#     unknown_prior_per_rank = {}
#     for rank in hierarchy:
#         unknown_prior = taxon_to_prior_map[rank].get(np.nan, 0)
#         unknown_prior_per_rank[rank] = unknown_prior

#     return taxon_to_prior_map, unknown_prior_per_rank


def unk_node_count_per_child_rank(prior_df, hierarchy):
    """
    Count of 'unk' rows whose prior is unknown_prior_per_rank[rank], matching
    add_unknown_nodes with parents[2:] (no unk under kingdom).

    Child rank hierarchy[i] gets one unk per node at parent rank hierarchy[i-1];
    phylum has no unk child under kingdom, so unk_counts['phylum'] == 0.
    """
    unk_n = {}
    for i, rank in enumerate(hierarchy):
        if i == 0:
            unk_n[rank] = 0
        else:
            cols = hierarchy[:i]
            unk_n[rank] = int(prior_df[cols].dropna(how="any").drop_duplicates().shape[0])
    return unk_n


def assign_priors(df, hierarchy):
    unk_base = 2.6
    unk_decay = 10

    prior_df = df.copy()
    prior_df = prior_df.drop(columns=['dna_barcode'])
    prior_df = prior_df.drop_duplicates().reset_index(drop=True)

    unk_counts = unk_node_count_per_child_rank(prior_df, hierarchy)

    taxon_to_prior_map = {}
    unknown_prior_per_rank = {}

    for idx, rank in enumerate(hierarchy):
        counts = prior_df[rank].value_counts(dropna=True)
        priors = counts / counts.sum()

        unknown_prior = unk_base / (unk_decay ** idx)
        unknown_prior_per_rank[rank] = unknown_prior

        n_unk = unk_counts[rank]
        total_unknown_priors_mass = unknown_prior * n_unk
        priors = (1 - total_unknown_priors_mass) * priors

        taxon_to_prior_map[rank] = priors.to_dict()
    
    return taxon_to_prior_map, unknown_prior_per_rank


def assign_global_ids(df, hierarchy):

    current_id = 2                          # 0 is reserved for root, 1 is reserved for kingdom
    taxon_to_id_map = {}
    df = df.sort_values(by=hierarchy)

    for rank in hierarchy:
        unique_names = df[rank].dropna().unique()
        taxon_to_id_map[rank] = {name: i + current_id for i, name in enumerate(unique_names)}    
        current_id += len(unique_names)

    return taxon_to_id_map


def create_final_mapping(taxon_to_id_map, sequential_mapping):
    final_mapping = {}

    for _, name_to_id_dict in taxon_to_id_map.items():
        for name, original_nid in name_to_id_dict.items():
            final_mapped_id = sequential_mapping.get(original_nid)
            final_mapping[name] = final_mapped_id
                
    return final_mapping


def add_NodeID_and_lvl(df, taxon_to_id_map, hierarchy):

    df['NodeID'] = -1
    df['lvl'] = -1

    for i, rank in enumerate(hierarchy):
        current_lvl = i + 2     # Skips root and kingdom

        mapping = taxon_to_id_map.get(rank, {})
        current_rank_ids = df[rank].map(mapping)
        mask = current_rank_ids.notna()
        df.loc[mask, 'NodeID'] = current_rank_ids[mask]
        df.loc[mask, 'lvl'] = current_lvl

    df['NodeID'] = df['NodeID'].astype(int)
    df['lvl'] = df['lvl'].astype(int)

    return df


def add_ParentID(df, taxon_to_id_map, hierarchy):

    df['ParentID'] = 1      # to set phylum parents to kingdom

    hierarchy_reversed = hierarchy[::-1]

    for rank in hierarchy_reversed[:-1]:  # Skip the last one (Phylum)
        # Get the mapping for the current rank
        mapping = taxon_to_id_map.get(rank, {})
        current_rank_ids = df[rank].map(mapping)
        
        # MASK: Rows where the current global_node_id is this rank
        is_current_rank_mask = (df['NodeID'] == current_rank_ids)
        
        # Identify the parent rank (the one immediately above it in the hierarchy)
        current_rank_idx = hierarchy.index(rank)
        parent_rank = hierarchy[current_rank_idx - 1]
        
        # Get the parent IDs for that rank
        parent_mapping = taxon_to_id_map.get(parent_rank, {})
        parent_ids = df[parent_rank].map(parent_mapping)

        df.loc[is_current_rank_mask, 'ParentID'] = parent_ids[is_current_rank_mask]

    df['ParentID'] = df['ParentID'].astype(int)
    df = df.sort_values(by=['ParentID', 'NodeID']).reset_index(drop=True)

    return df


def add_unknown_nodes(df, hierarchy, unknown_prior_per_rank):
    """
    Adds unknown nodes to the taxonomy
    df with columns: nid, pid, lvl, name

    adds unknown to the taxonomy as a new node under each parent.
    Each unknown node has a new id.
    Unknowns are added as new rows in the dataframe.
    """
    
    nid = df.index.max() + 1
    parents = np.sort(df["pid"].unique())
    parents = parents[2:]  # smallest two pids are root (0) and kingdom (1)
    parent_levels = df.loc[parents, "lvl"]

    lvl_to_rank = {i + 2: rank for i, rank in enumerate(hierarchy)}     # +2 because we start at the phylum level (lvl 2)
    unk_priors = [
        unknown_prior_per_rank.get(lvl_to_rank.get(lvl + 1), 0) 
        for lvl in parent_levels
    ]

    unk_df = pd.DataFrame({
        "nid": range(nid, nid + len(parents)),
        "pid": parents,
        "lvl": parent_levels + 1,
        "name": "unk",
        "prior": unk_priors
    }).set_index("nid")

    df = pd.concat([df, unk_df])
    return df


def build_taxonomy_dataframe(df, taxon_to_id_map, taxon_to_prior_map, hierarchy):
    tree_rows = []
    tree_rows.append({'nid': 0, 'pid': 0, 'lvl': 0, 'name': 'root', 'prior': 1.0})        # To add the root.
    tree_rows.append({'nid': 1, 'pid': 0, 'lvl': 1, 'name': 'kingdom', 'prior': 1.0})        # To add the kingdom.

    for i, rank in enumerate(hierarchy):
        current_level = i + 2
        
        if i == 0:      # Means we are at the Phylum level, which is the top of the hierarchy
            unique_phyla = df[rank].dropna().unique()
            for name in sorted(unique_phyla):
                node_id = taxon_to_id_map[rank][name]
                prior = taxon_to_prior_map[rank][name]
                tree_rows.append({
                    'nid': int(node_id),
                    'pid': 1,                 # parent ID for Phylum is kingdom
                    'lvl': current_level,
                    'name': name,
                    'prior': prior
                })
        else:
            # For subsequent ranks, the parent comes from the rank immediately above it
            parent_rank = hierarchy[i-1]
            
            # Identify unique Parent-Child name pairs
            unique_pairs = df[[parent_rank, rank]].dropna().drop_duplicates()
            
            # Sort by child name to keep the listing organized
            for _, row in unique_pairs.sort_values(by=rank).iterrows():
                parent_name = row[parent_rank]
                child_name = row[rank]
                
                # Map names to their Global Node IDs
                node_id = taxon_to_id_map[rank][child_name]
                parent_id = taxon_to_id_map[parent_rank][parent_name]
                prior = taxon_to_prior_map[rank][child_name]
                
                tree_rows.append({
                    'nid': int(node_id),
                    'pid': int(parent_id),
                    'lvl': current_level,
                    'name': child_name,
                    'prior': prior
                })
    
    return pd.DataFrame(tree_rows).sort_values(by=['pid', 'nid']).reset_index(drop=True)


def convert_to_taxonomy(df, hierarchy, data_dir):

    taxon_to_prior_map, unknown_prior_per_rank = assign_priors(df, hierarchy)
    max_seq_length = df['dna_barcode'].str.len().max()

    df = df.drop(columns=['dna_barcode'])
    df = df.drop_duplicates().reset_index(drop=True)
    taxon_to_id_map = assign_global_ids(df, hierarchy)

    df = add_NodeID_and_lvl(df, taxon_to_id_map, hierarchy)
    df = add_ParentID(df, taxon_to_id_map, hierarchy)
    df = build_taxonomy_dataframe(df, taxon_to_id_map, taxon_to_prior_map, hierarchy)

    df.set_index("nid", inplace=True)
    df = add_unknown_nodes(df, hierarchy, unknown_prior_per_rank)      

    df = df.reset_index()
    df = df.set_index("pid").sort_index()
    sequential_mapping = dict(zip(df["nid"], range(len(df["nid"]))))
    df["nid"] = df["nid"].map(sequential_mapping)
    df.index = df.index.map(sequential_mapping)

    prior = df["prior"].to_numpy()
    ranks = df["lvl"].to_numpy()
    names =df["name"].to_numpy()
    unk = (df["name"] == "unk").to_numpy()
    unk = np.expand_dims(unk, axis=1)   # expand dims to match no_refs shape in assign_sequences_to_taxonomy
    
    parents = df.index.to_numpy()
    parents = parents.astype(np.int32)
    paths = get_descendants(parents, len(df), ranks)
    parents[0] = -1
    
    unq = np.unique(parents)
    segments = np.searchsorted(unq, parents)

    name_to_id_map = create_final_mapping(taxon_to_id_map, sequential_mapping)

    np.savez(f"{data_dir}/taxonomy_temp.npz", 
             segments=segments, 
             parents=parents,
             unk=unk,
             ranks=ranks, 
             prior=prior, 
             paths=paths,
             names=names,
             name_to_id_map=name_to_id_map,
             max_seq_length=max_seq_length)

    print("Taxonomy created.")
    return

### ------------------------------------------------ Assigning Sequences to Taxonomy ------------------------------------------------
def standardize_barcode(df):
    df['dna_barcode'] = df['dna_barcode'].str.replace(r'[^ACGT]', 'N', regex=True, case=False)
    df['dna_barcode'] = df['dna_barcode'].str.upper()
    
    return df


def lil_to_csr(lil, shape):
    """
    Converts a list of lists to a csr matrix
    lil: list of lists where each list contains the column indices of the non-zero elements in the row
    shape: (n_rows, n_cols)
    """

    num_nonzero = 0
    indices = []
    indptr = [0]

    for row in lil:
        num_nonzero += len(row)
        indptr.append(num_nonzero)
        indices.extend(row)
    
    data = np.ones(num_nonzero, dtype=bool)
    csr = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=bool)
    return csr


def pack_seqs(dna_barcodes):

    ref_list = []
    ok_pos = []
    max_len = dna_barcodes.str.len().max()

    for seq in dna_barcodes:
        padded_seq = seq.ljust(max_len, 'N')
        seq_bits = get_seq_bits(padded_seq)
        ref_list.append(np.packbits(seq_bits[:4], axis=None))
        ok_pos.append(np.packbits(seq_bits[4], axis=None))

    return np.array(ref_list), np.array(ok_pos)


def assign_sequences_to_taxonomy(sequences_df, hierarchy, data_dir):
    taxonomy_dir = f"{data_dir}/taxonomy_temp.npz"
    taxonomy = np.load(taxonomy_dir, allow_pickle=True)
    name2id = taxonomy["name_to_id_map"].item()
    taxonomy_length = taxonomy["names"].shape[0]
    sequences_length = len(sequences_df)

    sequences_df = standardize_barcode(sequences_df)
    
    no_refs = np.ones((taxonomy_length, 1), dtype=bool)   # To track which nodes have no references
    has_refs = set()                        # To track which nodes have references
    node2seq = [[] for _ in range(taxonomy_length)]

    sequences_df.reset_index(drop=True, inplace=True)

    for idx, taxon in sequences_df.iterrows():
        curr_nids = [0,1]   # Add root and kingdom

        for rank in hierarchy:
            if taxon[rank] is not np.nan:
                curr_nids.append(name2id[taxon[rank]])

        for nid in curr_nids:
            has_refs.add(nid)
            node2seq[nid].append(idx)

    # mask out nodes that have references.
    no_refs[list(has_refs)] = False
    node_state = no_refs*np.logical_not(taxonomy["unk"])    # marks nodes that are known but Empty
    node_state = np.concatenate([node_state, np.logical_not(no_refs)], axis=1)  # concatenate to [N, 2], [known_empty, has_refs]

    node2seq = lil_to_csr(node2seq, (taxonomy_length, sequences_length))
    n2s_indices = node2seq.indices
    n2s_indptr = node2seq.indptr

    refs, ok_pos = pack_seqs(sequences_df["dna_barcode"])

    # update the taxonomy dictionary with the new data
    data_dict = {key: taxonomy[key] for key in taxonomy.files}
    data_dict['refs'] = refs
    data_dict['ok_pos'] = ok_pos
    data_dict['n2s_indices'] = n2s_indices
    data_dict['n2s_indptr'] = n2s_indptr
    data_dict['node_state'] = node_state

    taxonomy.close()
    np.savez(f"{data_dir}/taxonomy.npz", **data_dict)

    print("Sequences assigned.")

    return
### ------------------------------------------------ Creating Labels and Targets ------------------------------------------------
def get_list_of_targets(sequences_df, hierarchy, data_dir):
    taxonomy_dir = f"{data_dir}/taxonomy.npz"
    taxonomy = np.load(taxonomy_dir, allow_pickle=True)
    name_to_id_map = taxonomy["name_to_id_map"].item()
    taxonomy.close()

    # create training-targets
    sequences_df.reset_index(drop=True, inplace=True)
    list_of_targets = []
    for _, row in sequences_df.iterrows():
        row_ids = [-1] * 7
        row_ids[0] = 1
        for i, rank in enumerate(hierarchy):
            row_ids[i+1] = name_to_id_map.get(row[rank], -1)

        list_of_targets.append(row_ids)

    return list_of_targets


def save_training_targets(list_of_targets, data_dir):

    targets_df = pd.DataFrame(list_of_targets)
    targets_df_tr = targets_df.T
    targets_df_tr.to_csv(f'{data_dir}/train-targets.csv', index=True)

    print("Training targets saved.")

    return


def save_test_labels(list_of_targets, data_dir):
    targets_df = pd.DataFrame(list_of_targets)
    taxa_levels = ['kingdom_id', 'phylum_id', 'class_id', 'order_id', 'family_id', 'genus_id', 'species_id']
    targets_df.columns = taxa_levels
    targets_df.insert(0, 'root_id', 0)

    targets_df.to_csv(f'{data_dir}/test_labels.csv', index=False)

    print("Test labels saved.")

    return


### ------------------------------------------------ Main Function ------------------------------------------------
if __name__ == "__main__":

    data_dir = "datasets/mycoai/new_tax"
    # full_df, hierarchy = load_dataset(split="all")
    # full_df, hierarchy = mycoai_fasta_to_df(f"{data_dir}/trainset.fasta")  
    # full_df.to_csv(f"{data_dir}/trainset.csv", index=False)
    # Rainbows = remove_question_marks(full_df) 

    hierarchy = ["phylum", "class", "order", "family", "genus", "species"]
    full_df = pd.read_csv(f"{data_dir}/trainset_clean.csv")
    convert_to_taxonomy(full_df, hierarchy, data_dir)    
    
    train_sequences_df = pd.read_csv(f"{data_dir}/trainset_clean.csv")
    targets_df = train_sequences_df.copy()
  
    assign_sequences_to_taxonomy(train_sequences_df, hierarchy, data_dir)

    list_of_targets = get_list_of_targets(targets_df, hierarchy, data_dir)
    save_training_targets(list_of_targets, data_dir)
    save_test_labels(list_of_targets, data_dir)


# ------------------------------------------------ Picking Unknown Samples ------------------------------------------------
# def pick_test_sequence(taxon, grouped_remaining):
#     try:
#         # Get all available sequences for this genus that aren't in train
#         available = grouped_remaining.get_group(taxon)
#         # Sample 1 (randomly) from what's left
#         return available.sample(n=1)
#     except (KeyError, ValueError):
#         # Handle cases where a genus might not have a second sequence
#         return None

# def pick_unknown_samples(upper_level_taxa):

#     unknown_from_trainset = pd.read_csv("notebooks/unknown_from_trainset.csv")
#     train_df = unknown_from_trainset.sample(n=17735, random_state=42)
#     remaining_df = unknown_from_trainset.drop(train_df.index)
#     grouped_remaining = remaining_df.groupby(upper_level_taxa)

#     test_list = []
#     for taxon in train_df[upper_level_taxa]:
#         sibling = pick_test_sequence(taxon, grouped_remaining)
#         if sibling is not None:
#             test_list.append(sibling)

#     test_df = pd.concat(test_list).reset_index(drop=True)
#     train_df = train_df.reset_index(drop=True)

#     test_df.to_csv("notebooks/unknown_test.csv", index=False)
#     train_df.to_csv("notebooks/unknown_train.csv", index=False)



# if __name__ == "__main__":
#     pick_unknown_samples("order")