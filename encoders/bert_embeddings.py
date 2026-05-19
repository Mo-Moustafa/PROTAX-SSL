from encoders.barcode_encoder import *
from scripts.bioscan_convert import *
import numpy as np
from scripts.calibration import evaluate
import pickle

# if __name__ == "__main__":
#     taxonomy_dir = "datasets/mycoai"
#     dataset = pd.read_csv(f"{taxonomy_dir}/test_10k.csv")
#     dataset = standardize_barcode(dataset)
#     sequences = dataset["dna_barcode"].tolist()
#     model = BarcodeBERT()
#     embeddings = model.generate_embeddings(sequences)
#     model.save_embeddings(embeddings, f"{taxonomy_dir}/benchmark_embeddings.npz")


# ---------------- Pipeline for the MycoAI dataset ----------------
if __name__ == "__main__":

    taxonomy_dir = "datasets/mycoai/bert"
    # class_level = "species"
    for class_level in ["genus"]:
    # train_dataset, _ = mycoai_fasta_to_df(f"{taxonomy_dir}/test3.fasta")
    # train_dataset = pd.read_csv(f"{taxonomy_dir}/unknown_test.csv")
    # train_dataset = standardize_barcode(train_dataset)
    # sequences = train_dataset["dna_barcode"].tolist()

        labels_df = pd.read_csv(f"{taxonomy_dir}/train_labels.csv")
        train_labels = labels_df[f"{class_level}_id"].tolist()

        model = BarcodeBERT()

        # run this first time only
    # train_embeddings = model.generate_embeddings(sequences)
    # model.save_embeddings(train_embeddings, f"{taxonomy_dir}/bbert_unknown_test.npz")

        model.load_base_references(f"{taxonomy_dir}/base_embeddings.npz", train_labels)

        # test_dataset, _ = mycoai_fasta_to_df(f"{taxonomy_dir}/test3.fasta")
        # test_dataset = standardize_barcode(test_dataset)

        # test_sequences = test_dataset["dna_barcode"].tolist()
        # test_embeddings = model.generate_embeddings(test_sequences)

        # model.save_embeddings(test_embeddings, f"{taxonomy_dir}/test_embeddings.npz")

        with open(f"{taxonomy_dir}/loo_map_2.pkl", "rb") as f:
            loo_map = pickle.load(f)

        test_embeddings = model.load_embeddings(f"{taxonomy_dir}/test_embeddings_2.npz")

        model.classify(test_embeddings, class_level, train_eval=False, loo_map=loo_map)
        evaluate(f"results_BarcodeBERT.csv", f"{taxonomy_dir}/test_labels_2.csv", "BarcodeBERT", "MycoAI Test", class_level, f"{taxonomy_dir}/taxonomy.npz", "")



# ---------------- Pipeline for the Bioscan dataset ----------------
# if __name__ == "__main__":

#     taxonomy_dir = "datasets/canadian_invertebrates/mamba"
#     class_level = "species"
    
#     train_dataset, _ = load_dataset(split="train")
#     sequences = train_dataset["dna_barcode"].tolist()
#     train_labels = train_dataset[class_level].tolist()

#     model = BarcodeBERT()
#     model.name_to_nid = read_mapping(f"{taxonomy_dir}/taxonomy.npz")
#     model.load_base_references(f"{taxonomy_dir}/base_embeddings.npz", train_labels)

#     # test_dataset, _ = load_dataset(split="test")
#     # test_sequences = test_dataset["dna_barcode"].tolist()
#     # test_embeddings = model.generate_embeddings(test_sequences)

#     # model.save_embeddings(test_embeddings, f"{taxonomy_dir}/test_embeddings.npz")
#     test_embeddings = model.load_embeddings(f"{taxonomy_dir}/train_embeddings.npz")

#     model.classify(test_embeddings, class_level, train_eval=True)
#     evaluate(f"results_BarcodeBERT.csv", f"{taxonomy_dir}/train_labels.csv", "BarcodeBERT", "Canadian Invertebrates", class_level, f"{taxonomy_dir}/taxonomy.npz", "")


# ---------------- Pipeline for the 37k taxonomy dataset ----------------
# if __name__ == "__main__":
    
#     model = BarcodeBERT()
#     model.read_node_ids("datasets/finbol/taxonomy.priors")
#     class_level = "species"
#     s_train, l_train = model.fasta_to_list("datasets/finbol/train.aln", class_level)
#     # run this first time only
#     # base_embeddings = model.generate_embeddings(s_train)
#     # model.save_embeddings(base_embeddings, "models/ref_db/train_test/train_embeddings.npz")

#     # run this for the first time only for the train-test splits
#     # s_test, l_test = model.fasta_to_list("models/ref_db/train_test/test.aln")
#     # query_embeddings = model.generate_embeddings(s_test)
#     # model.save_embeddings(query_embeddings, "models/ref_db/train_test/test_embeddings.npz")

#     model.load_base_references("datasets/finbol/mamba/base_embeddings.npz", l_train)
#     query_embeddings = model.load_embeddings("datasets/finbol/mamba/train_embeddings.npz")
#     model.classify(query_embeddings, class_level, train_eval=True)
#     evaluate(f"results_BarcodeBERT.csv", "datasets/finbol/train_labels.csv", "BarcodeBERT", "FinBOL", class_level, "datasets/finbol/taxonomy.npz", "")

