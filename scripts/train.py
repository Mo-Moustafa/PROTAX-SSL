import protax.model as model

from protax import protax_utils
import numpy as np
import jax
import jax.numpy as jnp
import time
import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt
import argparse
import json
import math
from functools import partial
from tqdm import tqdm
import sys
import joblib
from scripts.calibration import evaluate
from protax.classify import classify_file

def CE_loss(log_probs, y_ind, q_param, tree, train_with_q):

    row = jnp.take(log_probs, y_ind, axis=0)
    log_p_model = jnp.sum(row)

    if train_with_q:
        q = jnp.clip(jax.nn.sigmoid(q_param), 1e-7, 1.0 - 1e-7)
        log_prior = jnp.log(tree.prior[y_ind])
        log_p_mix = jnp.logaddexp(
            jnp.log(q) + log_prior,
            jnp.log(1.0 - q) + log_p_model,
        )
        return -log_p_mix

    return -log_p_model


def forward(X, tree, beta, segnum, y_ind, lvl, q_param, train_with_q):
    beta = jnp.take(beta, lvl, axis=0)
    log_probs = model.fill_log_bprob(X, beta, tree, segnum)
    return CE_loss(log_probs, y_ind, q_param, tree, train_with_q)


def make_vectorized_batch_ops(tree_for_lp, lvl, segnum):
    """
    vmap grad / forward over batch dimension. X differs per sample; tree_for_lp
    supplies segments/paths/prior for fill_log_bprob / CE_loss.
    """

    def _loss_beta(beta, x, y, q_param, train_with_q):
        beta_n = jnp.take(beta, lvl, axis=0)
        log_probs = model.fill_log_bprob(x, beta_n, tree_for_lp, segnum)
        return CE_loss(log_probs, y, q_param, tree_for_lp, train_with_q)

    def _single_beta_grad(beta, x, y, q_param, train_with_q):
        return jax.grad(_loss_beta, argnums=0)(beta, x, y, q_param, train_with_q)

    @partial(jax.jit, static_argnums=(4,))
    def batched_beta_grad(beta, X_batch, y_batch, q_param, train_with_q):
        return jax.vmap(_single_beta_grad, in_axes=(None, 0, 0, None, None))(
            beta, X_batch, y_batch, q_param, train_with_q
        )

    def _loss_q(q_param, x, y, beta, train_with_q):
        beta_n = jnp.take(beta, lvl, axis=0)
        log_probs = model.fill_log_bprob(x, beta_n, tree_for_lp, segnum)
        return CE_loss(log_probs, y, q_param, tree_for_lp, train_with_q)

    def _single_q_grad(q_param, x, y, beta, train_with_q):
        return jax.grad(_loss_q, argnums=0)(q_param, x, y, beta, train_with_q)

    @partial(jax.jit, static_argnums=(4,))
    def batched_q_grad(beta, X_batch, y_batch, q_param, train_with_q):
        return jax.vmap(_single_q_grad, in_axes=(None, 0, 0, None, None))(
            q_param, X_batch, y_batch, beta, train_with_q
        )

    @partial(jax.jit, static_argnums=(4,))
    def batched_forward(beta, X_batch, y_batch, q_param, train_with_q):
        def one(x, y):
            return forward(x, tree_for_lp, beta, segnum, y, lvl, q_param, train_with_q)

        return jax.vmap(one)(X_batch, y_batch)

    return batched_beta_grad, batched_q_grad, batched_forward

def lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
        train_with_q = False
        return lr, train_with_q
    else:
        lr = base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
        train_with_q = True
        return lr, train_with_q

def create_design_matrices(variant, base_dir, train_dir, tree, train_config, N, params):
    n2s = tree.node2seq
    node_state = tree.node_state
    dist_scaling = train_config["dist_scaling"]
    if dist_scaling == "power":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = joblib.load("models/scalings/gap_power_transformer.joblib")
    elif dist_scaling == "hybrid":
        pt_min = joblib.load("models/scalings/min_power_transformer.joblib")
        pt_gap = None
    else:
        pt_min = None
        pt_gap = None

    if variant in ("bert", "mamba", "hybrid_lin"):
        print("Reading Embeddings...")
        base_embeddings = np.load(base_dir)
        train_embeddings = np.load(train_dir)

        if variant in ("bert", "mamba"):
            base_embeddings = jnp.array(base_embeddings[variant])
            print("base embeddings.shape: ", base_embeddings.shape)
            base_length = int(base_embeddings.shape[0])

            train_embeddings = jnp.array(train_embeddings[variant])
            print("train embeddings.shape: ", train_embeddings.shape)
            R = int(train_embeddings.shape[0])
            
        else:
            bert_base = jnp.array(base_embeddings["bert"])
            mamba_base = jnp.array(base_embeddings["mamba"])
            assert bert_base.shape == mamba_base.shape
            base_length = int(bert_base.shape[0])

            bert_train = jnp.array(train_embeddings["bert"])
            mamba_train = jnp.array(train_embeddings["mamba"])
            assert bert_train.shape == mamba_train.shape
            R = int(bert_train.shape[0])

    elif variant == "og":
        seq_list, ok_list = protax_utils.read_refs(train_dir, padding_len=tree.max_seq_length)
        print("Read reference sequences successfully")
        R = int(seq_list.shape[0])
        base_length = int(tree.refs.shape[0])
    
    else:
        raise ValueError(f"Model {variant} not supported")

    X_all = None    
    for i in tqdm(range(R), desc="Computing design matrices", file=sys.stderr, dynamic_ncols=True, mininterval=5):
        # mask the tree for the current sample
        if i < base_length:
            new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
            tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)
        else:
            tree = tree._replace(node2seq=n2s, node_state=node_state)
        
        # compute the distance matrix based on the model
        if variant in ("bert", "mamba"):
            dists = model.cosine_dist(train_embeddings[i], base_embeddings)
            X = model.get_X(dists, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        elif variant in ("hybrid_lin"):
            dists_bert = model.cosine_dist(bert_train[i], bert_base)
            dists_mamba = model.cosine_dist(mamba_train[i], mamba_base)
            X = model.get_hybrid_X(dists_bert, dists_mamba, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        elif variant == "og":
            q_seq = seq_list[i]
            ok = ok_list[i]
            dists = model.p_dist(q_seq, ok, tree.refs, tree.ok_pos)
            X = model.get_X(dists, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        x_host = np.asarray(jax.device_get(X), dtype=np.float32)
        if X_all is None:
            n_nodes, F_dim = x_host.shape
            X_all = np.zeros((R, n_nodes, F_dim), dtype=np.float32)
        X_all[i] = x_host

    return X_all, R

def get_targ(target_dir):
    """
    Get node id for each reference sequence at lowest level
    NOTE: Node ids are 1-indexed, not 0-indexed (assumes first column to be an index)
    """
    targ = pd.read_csv(target_dir)
    targ = targ.to_numpy()[:, 1:].T

    res = np.zeros((targ.shape[0],), dtype=np.int32)

    for i in range(len(targ)):
        old = -1
        for j in range(targ.shape[1]):
            if targ[i][j] == -1:
                res[i] = old
            elif j == targ.shape[1] - 1:
                res[i] = targ[i][j]
            old = targ[i][j]

    return jnp.array(res)


def load_params(pdir, tdir):
    par_dir = Path(pdir)
    tax_dir = Path(tdir)

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    beta = par["beta"]
    sc = par["scalings"]
    # lvl = tax["node_layer"]
    lvl = tax["ranks"]

    if beta.shape[0] == 7:
        print("Padding beta to (8, 4)...")
        zero_row = np.zeros((1, 4)) # Use zeros, not ones
        beta = np.vstack([zero_row, beta])
        
    # 2. Fix scalings: Pad with [0, 1, 0, 1] if shape is (7, 4)
    if sc.shape[0] == 7:
        print("Padding scalings to (8, 4)...")
        padding_row = np.array([[0, 1, 0, 1]])
        sc = np.vstack([padding_row, sc])

    beta = jnp.array(beta)
    sc = jnp.array(sc)

    return beta, lvl, sc


def train(variant, tax_dir, base_dir, train_dir, targ_dir, scalings_dir, train_config, run_id, continue_training, q_perc=0.5, decay_lr=False):

    if not continue_training:
        print("Training from scratch")
        tree, params, N, segnum = protax_utils.read_model_jax(scalings_dir, tax_dir)

        key_beta = jax.random.PRNGKey(0)
        sigma_beta = 5.0
        if variant in ("bert", "mamba", "og"):
            beta = jax.random.normal(key_beta, (8, 4)) * sigma_beta
        elif variant == "hybrid_lin":
            beta = jax.random.normal(key_beta, (8, 6)) * sigma_beta

        q_param = jnp.asarray(-3.5, dtype=jnp.float32)
        _, lvl, sc = load_params(scalings_dir, tax_dir)

    else:
        print("Continuing training")
        tree, params, N, segnum = protax_utils.read_model_jax(f"models/model_{run_id}.npz", tax_dir)
        beta, lvl, sc = load_params(f"models/model_{run_id}.npz", tax_dir)
        initial_value = jnp.log(q_perc / (1 - q_perc))
        q_param = jnp.array(initial_value)

    q_mult = 0.1

    tree_for_lp = tree
    lvl = jnp.asarray(lvl, dtype=jnp.int32)
    targ = get_targ(targ_dir)
    targ_np = np.asarray(jax.device_get(targ), dtype=np.int32)
    batched_beta_grad, batched_q_grad, batched_forward = make_vectorized_batch_ops(tree_for_lp, lvl, segnum)

    X_all, R = create_design_matrices(variant, base_dir, train_dir, tree, train_config, N, params)

    base_lr = train_config["learning_rate"]
    l2 = float(train_config["l2"])
    train_with_q = False
    if decay_lr:
        warmup_ratio = 0.05
        steps_per_epoch = math.ceil(R / train_config["batch_size"])
        total_steps = steps_per_epoch * train_config["num_epochs"]
        warmup_steps = int(total_steps * warmup_ratio)
        step = 0
        lr, train_with_q = lr_schedule(step, total_steps, warmup_steps, base_lr)
    else:
        lr = base_lr

    print("Training model...")
    start_time = time.time()
    
    # lr_list = [lr, 0.5*lr, 2*lr]
    # run_id_og = run_id
    # for i in range (3):
    #     print("\n\n--------------------------------")
    #     print(f"Training model with lr: {lr_list[i]}")
    #     beta = jax.random.normal(key_beta, (8, 4)) * sigma_beta
    #     q_param = jnp.asarray(0.0, dtype=jnp.float32)

    #     train_with_q = False
    #     lr = lr_list[i]
    #     run_id = f"{run_id_og}_run_{i}"

    epoch_loss_hist = []

    for e in range(1, train_config["num_epochs"] + 1):
            
        # if e == 6:
            # classify_file(variant, data_dir / "test_embeddings.npz", base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=False, X_all=None)
            # evaluate(f"results_{run_id}.csv", data_dir / "test_labels.csv", "", exp_details, "species", tax_dir, run_id)
            # train_with_q = True
            
        #     lr = lr * 0.01
        #     print("\n--------------------------------")
        #     print("Decreasing lr to: ", lr)

        # elif e == 11:
        #     classify_file(variant, data_dir / "test_embeddings.npz", base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=False, X_all=None)
        #     evaluate(f"results_{run_id}.csv", data_dir / "test_labels.csv", "", exp_details, "species", tax_dir, run_id)

        #     lr = lr * 0.05
        #     print("\n--------------------------------")
        #     print("Decreasing lr to: ", lr)
        #     # train_with_q = True

        # elif e == 31:
        #     lr = lr * 0.1
        #     print("\n--------------------------------")
        #     print("Decreasing lr to: ", lr)


        print(f"epoch {e} / {train_config['num_epochs']}")

        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)

        traversal = list(range(R))
        random.shuffle(traversal)

        B = int(train_config["batch_size"])
        for start in tqdm(range(0, len(traversal), B), desc=f"epoch {e} batches", file=sys.stderr, dynamic_ncols=True, mininterval=5):
            
            if decay_lr:
                lr, train_with_q = lr_schedule(step, total_steps, warmup_steps, base_lr)
                step += 1

            chunk = traversal[start : start + B]
            idx = np.asarray(chunk, dtype=np.int32)
            X_batch = jnp.asarray(X_all[idx], dtype=jnp.float32)
            y_batch = jnp.asarray(targ_np[idx], dtype=jnp.int32)

            loss_sum += jnp.sum(batched_forward(beta, X_batch, y_batch, q_param, train_with_q))

            G_beta = batched_beta_grad(beta, X_batch, y_batch, q_param, train_with_q)
            
            if train_with_q:
                G_q = batched_q_grad(beta, X_batch, y_batch, q_param, train_with_q)
                q_step = jnp.clip(jnp.mean(G_q), -10.0, 10.0)
                q_param = q_param - (lr * q_mult * q_step)

            # L2 weight decay on beta (set train_config["l2"] > 0 to enable)
            beta = beta - lr * (jnp.mean(G_beta, axis=0) + l2 * beta)

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item())

        epoch_loss = loss_sum / float(R)
        epoch_loss_hist.append(float(epoch_loss))
        print("total loss: ", float(epoch_loss))
        
        # save checkpoint
        mf = Path(f"models/model_{run_id}.npz")
        np.savez_compressed(mf.resolve(), beta=np.array(beta), scalings=sc)        

    plt.plot(epoch_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(f"loss_curve_{run_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    end_time = time.time()
    print(f"Time taken to train: {(end_time - start_time) / 3600} hours")

    return X_all


def str2bool(s: str) -> bool:
    return s.lower() == "true"

if __name__ == "__main__":
    # parse config from command line
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--scalings_dir", type=str, help="Path to scalings")
    parser.add_argument('--tc', type=str, help='Training Hyperparameters')
    parser.add_argument("--exp", type=str, help="Experiment details")
    parser.add_argument("--id", type=str, help="ID of the run")
    parser.add_argument("--continue_training", type=str2bool, choices=[True, False], default=False)
    args = parser.parse_args()

    tc = json.loads(args.tc)
    variant = tc["model"]

    scalings_dir = Path(args.scalings_dir)
    data_dir = Path(args.data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    targ_dir = data_dir / "train-targets.csv"
    train_labels = data_dir / "train_labels.csv"
    if variant == "og":
        base_dir = None
        train_dir = data_dir / "train.aln"
    else:
        base_dir = data_dir / "base_embeddings.npz"
        train_dir = data_dir / "train_embeddings.npz"

    exp_details = args.exp
    run_id = args.id
    continue_training = args.continue_training

    X_all = train(variant, tax_dir, base_dir, train_dir, targ_dir, scalings_dir, tc, run_id, continue_training)
    
    classify_file(variant, train_dir, base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=True, X_all=X_all)
    for class_level in ["species", "genus", "family"]:
        evaluate(f"results_{run_id}.csv", train_labels, "", exp_details, class_level, tax_dir, run_id)