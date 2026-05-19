import torch
import torch.nn.functional as F

import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from bioscan_dataset import CanadianInvertebrates

from BarcodeMamba.utils.probing_utils import get_pretrained_barcodemamba
from BarcodeMamba.utils.ssm_dataset import get_tokenizer

def fasta_to_list(fasta_dir):
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
        labels.append(label[1])

    return sequences, labels

def load_dataset(split):
    dataset = CanadianInvertebrates(root="~/Datasets/bioscan/", target_type=["phylum", "class", "order", "family", "genus", "species"], split=split, download=False)
    df = dataset.metadata
    hierarchy = ["phylum", "class", "order", "family", "genus", "species"]
    df = df[hierarchy + ["dna_barcode"]]
    return df, hierarchy

def standardize_barcode(df):
    df['dna_barcode'] = df['dna_barcode'].str.replace(r'[^ACGT]', 'N', regex=True, case=False)
    df['dna_barcode'] = df['dna_barcode'].str.upper()
    
    return df

# ---------------- Generation of normalized embeddings (for external use) ----------------

def embed_dna_sequences(
    sequences: list[str],
    pretrained_run_dir: str,
    device: str = "cuda",
):

    ckpt_path, cfg, model = get_pretrained_barcodemamba(pretrained_run_dir)
    assert (
        model is not None and cfg is not None
    ), f"Failed to load model from {pretrained_run_dir}"
    # This script assumes a char-tokenizer checkpoint.
    tokenizer = get_tokenizer("char", cfg.tokenizer)
    tokenizer.pad_token = "N"
    model = model.to(device).eval()

    embs = []
    with torch.no_grad():
        for i in tqdm(range(len(sequences)), desc="Generating embeddings", total=len(sequences), file=sys.stderr, dynamic_ncols=True):
            seq = sequences[i]

            enc = tokenizer(
                seq,
                add_special_tokens=False,
                padding="max_length",
                max_length=2500,
                # max_length=1062,
                truncation=True,
            )
            input_ids = torch.tensor(
                enc["input_ids"], dtype=torch.long, device=device
            ).unsqueeze(0)  # [1, L]
            att_mask = torch.tensor(
                enc["attention_mask"], dtype=torch.long, device=device
            ).unsqueeze(0)  # [1, L]

            hs = model.get_hidden_states(input_ids)  # [1, L, d_model]

            mask = att_mask.to(hs.dtype)  # [1, L]
            seq_emb = (hs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(
                1
            ).unsqueeze(-1)

            embs.append(seq_emb.squeeze(0).detach().cpu())  # [d_model]

    embs = torch.stack(embs, dim=0)
    embs = F.normalize(embs, dim=1)

    return embs, ckpt_path  # [N, d_model]

if __name__ == "__main__":
    
    # Don't forget to change the max_length

    # sequences, labels = fasta_to_list("datasets/finbol/train.aln")

    train_dataset = pd.read_csv("datasets/mycoai/test_2/test2_clean.csv")
    # train_dataset = pd.read_csv("datasets/mycoai_full/unknown_data/unknown_test.csv")
    train_dataset = standardize_barcode(train_dataset)
    sequences = train_dataset["dna_barcode"].tolist()

    # train_dataset, hierarchy = load_dataset(split="train")
    # sequences = train_dataset["dna_barcode"].tolist()

    pretrained_run_dir = "BarcodeMamba/BarcodeMamba-dim768-layer6-char"
    embeddings, ckpt = embed_dna_sequences(sequences, pretrained_run_dir)
    print("ckpt:", ckpt)
    print("embeddings:", embeddings.shape)
    np.savez_compressed("BarcodeMamba/test_embeddings_2.npz", embeddings=np.asarray(embeddings))