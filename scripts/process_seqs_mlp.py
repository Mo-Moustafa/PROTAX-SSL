from protax.classify_hybrid_mlp import classify_file
from scripts.calibration import evaluate
from pathlib import Path
import json
import pickle

import sys

if __name__ == "__main__":
    
    protax_args = sys.argv
    data_dir, test_mode, model_dir, train_config, exp_details, id = protax_args[1:]

    data_dir = Path(data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    base_dir = data_dir / "base_embeddings.npz"

    if test_mode == "train":
        query_dir = data_dir / "train_embeddings.npz"
        labels_dir = data_dir / "train_labels.csv"
        train_eval = True
        red_dir = f"results_{id}.csv"
    else:
        query_dir = data_dir / "test_embeddings.npz"
        labels_dir = data_dir / "test_labels.csv"
        train_eval = False
        red_dir = f"results_{id}_test.csv"

    dist_scaling = json.loads(train_config)["dist_scaling"]
    
    classify_file(query_dir, base_dir, model_dir, tax_dir, dist_scaling, id, train_eval)

    # for class_level in ["species", "genus", "family"]:
    class_level = "species"
    evaluate(red_dir, labels_dir, train_config, exp_details, class_level, tax_dir, id)

    # else:
    #     i=2
    #     query_dir = data_dir / f"test_embeddings_{i}.npz"
    #     labels_dir = data_dir / f"test_labels_{i}.csv"
    #     loo_map = data_dir / f"loo_map_{i}.pkl"
    #     with open(loo_map, 'rb') as f:
    #         loo_map = pickle.load(f)
    #     train_eval = False
    #     red_dir = f"results_{id}_test.csv"

    #     dist_scaling = json.loads(train_config)["dist_scaling"]

    #     classify_file(query_dir, base_dir, model_dir, tax_dir, dist_scaling, id, train_eval, loo_map=loo_map)

    #     for class_level in ["species", "genus", "family"]:
    #         evaluate(red_dir, labels_dir, train_config, exp_details, class_level, tax_dir, id)
