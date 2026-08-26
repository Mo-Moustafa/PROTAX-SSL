from protax.classify import classify_file
from scripts.calibration import evaluate
from pathlib import Path
import json
import pickle
import sys

if __name__ == "__main__":
    
    protax_args = sys.argv
    data_dir, test_mode, model_dir, train_config, exp_details, id = protax_args[1:]

    dist_scaling = json.loads(train_config)["dist_scaling"]
    variant = json.loads(train_config)["model"]

    data_dir = Path(data_dir)
    tax_dir = data_dir / "taxonomy.npz"

    if variant == "og":
        base_dir = None
    else:
        base_dir = data_dir / "base_embeddings.npz"

    if test_mode == "train":
        if variant == "og":
            query_dir = data_dir / "train.aln"
        else:
            query_dir = data_dir / "train_embeddings.npz"

        labels_dir = data_dir / "train_labels.csv"
        train_eval = True
        res_dir = f"results_{id}.csv"

    else:
        if variant == "og":
            query_dir = data_dir / "test.aln"
        else:
            query_dir = data_dir / "test_embeddings.npz"
        
        labels_dir = data_dir / "test_labels.csv"
        train_eval = False
        res_dir = f"results_{id}_test.csv"

    
    classify_file(variant, query_dir, base_dir, model_dir, tax_dir, dist_scaling, id, train_eval)

    for class_level in ["species", "genus", "family"]:
        evaluate(res_dir, labels_dir, train_config, exp_details, class_level, tax_dir, id)



