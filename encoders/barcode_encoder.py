from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from functools import partial
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from bioscan_dataset import CanadianInvertebrates


def load_dataset(split):
    dataset = CanadianInvertebrates(root="~/Datasets/bioscan/", target_type=["phylum", "class", "order", "family", "genus", "species"], split=split, download=False)
    df = dataset.metadata
    hierarchy = ["phylum", "class", "order", "family", "genus", "species"]
    df = df[hierarchy + ["dna_barcode"]]
    return df, hierarchy


def read_mapping(taxonomy_dir: str):
    taxonomy = np.load(taxonomy_dir, allow_pickle=True)
    name_to_id_map = taxonomy["name_to_id_map"].item()
    taxonomy.close()
    return name_to_id_map


class BarcodeBERT():
    def __init__(self, model_name: str = "bioscan-ml/BarcodeBERT", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.model.eval().to(self.device)

        self.name_to_nid = None
        self.base_embeddings = None
        self.labels_list = None


    # --------- Preprocessing functions for the 37k taxonomy dataset ---------
    def fasta_to_list(self, fasta_dir: str, class_level: str):
        """
        Convert a FASTA file to a list of sequences and labels.
        Used with the 37k taxonomy dataset.
        Returns:
            sequences: list[str]
            labels: list[str]
        """
        sequences = []
        labels = []

        f = open(fasta_dir)
        while True:
            curr = f.readline().strip('\n')
            label = curr.replace('|', '\t').split('\t')

            seq = f.readline().strip('\n')

            if not seq:
                break

            sequences.append(seq)
            if class_level == "species":
                labels.append(label[1])
            elif class_level == "genus":
                parts = label[1].split(',')
                label = parts[:6]
                label = ','.join(label)
                labels.append(label)
            elif class_level == "family":
                parts = label[1].split(',')
                label = parts[:5]
                label = ','.join(label)
                labels.append(label)
        return sequences, labels

    def read_node_ids(self, node_ids_dir: str):
        """
        Read node IDs from taxonomy.priors file.
        """
        nodeIDs_df = pd.read_csv(node_ids_dir, sep='\t', header=None, names = ["nid", "pid", "lvl", "name", "priors", "..."])        
        self.name_to_nid = dict(zip(nodeIDs_df['name'], nodeIDs_df['nid']))

    def names_to_node_ids(self, names: list[str]):
        """
        Map a list of taxonomic names to their corresponding node IDs.
        """
        return [
            self.name_to_nid[name]
            for name in tqdm(names, desc="Mapping names to node IDs")
        ]

    def indices_to_NodeIDS(self, indices):
        labels = []
        for i in range(len(indices)):
            labels.append(self.labels_list[indices[i].item()])

        return labels


    # --------- Generation of normalized embeddings (for external use) ---------
    def generate_embeddings(self, sequences: list[str], batch_size: int = 256):
        """
        Generate normalized embeddings for a list of sequences.

        Returns:
            jax.Array: Shape (len(sequences), 768), float32, L2-normalized.
            JAX-compatible so it can be used directly with jax.numpy and PROTAX.
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings", total=(len(sequences) + batch_size - 1) // batch_size, file=sys.stderr, dynamic_ncols=True):

                batch = sequences[i:i + batch_size]
                input_ids = []
                for seq in batch:
                    ids = self.tokenizer(seq, padding=True, return_tensors="pt")["input_ids"]
                    input_ids.append(ids.squeeze(0))

                input_ids = torch.stack(input_ids, dim=0).to(self.device)

                outputs = self.model(input_ids=input_ids)
                hidden_states = outputs.hidden_states[-1]
                features = hidden_states.mean(dim=1)
                embeddings.append(features)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings_norm = F.normalize(embeddings, dim=1)

        # Return JAX array so callers use the same framework
        return jnp.array(embeddings_norm.cpu().numpy())

    def save_embeddings(self, embeddings: jnp.ndarray, path: str):
        """Save embeddings to a .npz file."""
        np.savez_compressed(path, embeddings=np.asarray(embeddings))
        print(f"Saved embeddings to {path}")

    def load_embeddings(self, path: str):
        """Load embeddings from a .npz file."""
        print(f"Loading embeddings from {path}")
        return jnp.array(np.load(path)["embeddings"])


    # --------- Using raw model for classification ---------
    def load_base_references(self, embeddings_path: str, labels: list[str]):
        # if self.name_to_nid is None:
        #     raise ValueError("Node IDs need to be read first")

        self.base_embeddings = self.load_embeddings(embeddings_path)
        print(f"Loaded {len(self.base_embeddings)} base embeddings")
        # self.labels_list = self.names_to_node_ids(labels) # for finbol and bioscan
        self.labels_list = labels # for mycoai
        print(f"Loaded {len(self.labels_list)} labels")
    
    def get_top_k_similarities(
        self,
        query_embeddings: jnp.ndarray,
        k: int = 1,
        train_eval: bool = True,
        query_chunk_size: int = 256,
        loo_map: dict = None,
    ):
        if self.base_embeddings is None:
            raise ValueError("Base references need to be loaded first")

        n_queries = query_embeddings.shape[0]
        n_base = self.base_embeddings.shape[0]

        # A full (n_queries x n_base) similarity matrix on GPU can require tens of GiB
        # (XLA may need large workspace); chunk queries to cap peak memory.
        print(
            "Computing cosine similarities (chunked queries, "
            f"chunk_size={query_chunk_size}, queries={n_queries}, refs={n_base})..."
        )

        top_vals_chunks = []
        top_idx_chunks = []
        for start in range(0, n_queries, query_chunk_size):
            end = min(start + query_chunk_size, n_queries)
            chunk = query_embeddings[start:end]
            similarities = chunk @ self.base_embeddings.T

            if train_eval:
                local_rows = jnp.arange(end - start)
                global_cols = start + local_rows
                similarities = similarities.at[local_rows, global_cols].set(0.0)

            elif loo_map is not None:
                # loo_map is a Python dict: query_global_index -> base_ref_index_to_mask
                # Do the dict lookup on the host (Python), then apply a vectorized scatter update.
                rows_to_mask = []
                cols_to_mask = []
                for r, g in enumerate(range(start, end)):
                    c = loo_map.get(g, None)
                    if c is None:
                        continue
                    if 0 <= c < n_base:
                        rows_to_mask.append(r)
                        cols_to_mask.append(c)
                if len(rows_to_mask) > 0:
                    similarities = similarities.at[
                        jnp.asarray(rows_to_mask, dtype=jnp.int32),
                        jnp.asarray(cols_to_mask, dtype=jnp.int32),
                    ].set(0.0)

            tv, ti = jax.lax.top_k(similarities, k=k)
            top_vals_chunks.append(tv)
            top_idx_chunks.append(ti)

        top_values = jnp.concatenate(top_vals_chunks, axis=0)
        top_indices = jnp.concatenate(top_idx_chunks, axis=0)

        return top_indices, top_values


    def classify(self, query_embeddings: jnp.ndarray, class_level: str, train_eval: bool = True, query_chunk_size: int = 256, loo_map: dict = None):
        print("Getting predictions...")
        
        top_indices, top_values = self.get_top_k_similarities(
            query_embeddings, k=1, train_eval=train_eval, query_chunk_size=query_chunk_size, loo_map=loo_map
        )
        probs_list = top_values.flatten().tolist()
        prediction_list = self.indices_to_NodeIDS(top_indices)
        prediction_list = np.asarray(prediction_list, dtype=np.int64).reshape(-1, 1)

        # Create 7 zero columns + 1 prediction column (to match PROTAX output format)
        res = np.hstack([
            np.zeros((len(prediction_list), 7), dtype=np.int64),
            prediction_list
        ])
        df = pd.DataFrame(np.array(res))
        df = df.rename(columns={7: f"{class_level}_id"})
        df[f'{class_level}_prob'] = probs_list
        df.to_csv("results_BarcodeBERT.csv", index=False)