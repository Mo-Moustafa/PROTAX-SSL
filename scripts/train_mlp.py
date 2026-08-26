"""
Train hybrid + per-rank MLP (same data path as train_gd_hybrid.py).
Checkpoints: beta (8,6) zeros placeholder, scalings, and MLP parameters.
"""
import protax.model as model
import protax.model_mlp as mlp_model
from protax import protax_utils

import numpy as np
import jax
import jax.numpy as jnp

import time
import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json
from functools import partial

from tqdm import tqdm
import sys
import math
import joblib
from scripts.calibration import evaluate
from protax.classify_mlp import classify_file


def lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
        # train_with_q = False
        # return lr, train_with_q
    else:
        lr = base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
        # train_with_q = True
        # return lr, train_with_q
    return lr

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


def make_vectorized_batch_ops_mlp(tree_for_lp, lvl, segnum, q_mult):
    def _loss_with_mlp(mlp_params, x, y, q_param, train_with_q):
        log_probs = mlp_model.fill_log_bprob(x, mlp_params, tree_for_lp, segnum, lvl)
        return CE_loss(log_probs, y, q_param, tree_for_lp, train_with_q)

    def _batch_losses_no_q(mlp_params, X_batch, y_batch, q_param):
        one = lambda x, y: _loss_with_mlp(mlp_params, x, y, q_param, False)
        return jax.vmap(one)(X_batch, y_batch)

    def _batch_losses_with_q(mlp_params, X_batch, y_batch, q_param):
        one = lambda x, y: _loss_with_mlp(mlp_params, x, y, q_param, True)
        return jax.vmap(one)(X_batch, y_batch)

    @jax.jit
    def train_step_no_q(mlp_params, q_param, X_batch, y_batch, lr, l2):
        def mean_loss_mlp_only(mlp_local):
            return jnp.mean(_batch_losses_no_q(mlp_local, X_batch, y_batch, q_param))

        mean_loss, mlp_grad = jax.value_and_grad(mean_loss_mlp_only)(mlp_params)
        mlp_params_new = jax.tree_util.tree_map(lambda p, g: p - lr * (g + l2 * p), mlp_params, mlp_grad)
        batch_loss_sum = mean_loss * jnp.asarray(X_batch.shape[0], dtype=mean_loss.dtype)
        return mlp_params_new, q_param, batch_loss_sum

    @jax.jit
    def train_step_with_q(mlp_params, q_param, X_batch, y_batch, lr, l2):
        def mean_loss_mlp_q(mlp_local, q_local):
            return jnp.mean(_batch_losses_with_q(mlp_local, X_batch, y_batch, q_local))

        mean_loss, (mlp_grad, q_grad) = jax.value_and_grad(mean_loss_mlp_q, argnums=(0, 1))(mlp_params, q_param)
        q_step = jnp.clip(q_grad, -10.0, 10.0)
        q_param_new = q_param - (lr * q_mult * q_step)
        mlp_params_new = jax.tree_util.tree_map(lambda p, g: p - lr * (g + l2 * p), mlp_params, mlp_grad)
        batch_loss_sum = mean_loss * jnp.asarray(X_batch.shape[0], dtype=mean_loss.dtype)
        return mlp_params_new, q_param_new, batch_loss_sum

    return train_step_no_q, train_step_with_q


def create_design_matrices(base_dir, train_dir, tree, train_config, N, params):
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

    print("Reading Embeddings...")
    base_embeddings = np.load(base_dir)
    train_embeddings = np.load(train_dir)
            
    bert_base = jnp.array(base_embeddings["bert"])
    mamba_base = jnp.array(base_embeddings["mamba"])
    # assert bert_base.shape == mamba_base.shape
    base_length = int(bert_base.shape[0])

    bert_train = jnp.array(train_embeddings["bert"])
    mamba_train = jnp.array(train_embeddings["mamba"])
    # assert bert_train.shape == mamba_train.shape
    R = int(bert_train.shape[0])

    X_all = None    
    for i in tqdm(range(R), desc="Computing design matrices", file=sys.stderr, dynamic_ncols=True, mininterval=1000):
        # mask the tree for the current sample
        if i < base_length:
            new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
            tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)
        else:
            tree = tree._replace(node2seq=n2s, node_state=node_state)
        
        dists_bert = model.cosine_dist(bert_train[i], bert_base)
        dists_mamba = model.cosine_dist(mamba_train[i], mamba_base)
        X = model.get_hybrid_X(dists_bert, dists_mamba, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

        x_host = np.asarray(jax.device_get(X), dtype=np.float32)
        if X_all is None:
            n_nodes, F_dim = x_host.shape
            X_all = np.zeros((R, n_nodes, F_dim), dtype=np.float32)
        X_all[i] = x_host

    return X_all, R


def train(tax_dir, base_dir, train_dir, targ_dir, scalings_dir, train_config, run_id, continue_training, q_perc=0.5, decay_lr=False):

    start_time = time.time()

    mlp_hidden = int(train_config.get("mlp_hidden", 16))
    mlp_layers = int(train_config.get("mlp_layers", 1))

    q_mult = 1.0
    base_lr = train_config["learning_rate"]
    l2 = float(train_config["l2"])
    B = int(train_config["batch_size"])

    tree, N, segnum = protax_utils.read_tree(tax_dir)
    tree_for_lp = tree
    lvl = jnp.asarray(tree.ranks, dtype=jnp.int32)
    targ_jnp = protax_utils.get_targets(targ_dir)
    train_step_no_q, train_step_with_q = make_vectorized_batch_ops_mlp(tree_for_lp, lvl, segnum, q_mult)

    if not continue_training:
        params = protax_utils.read_model(scalings_dir, tree.ranks)
        key_mlp = jax.random.PRNGKey(0)
        sigma_mlp = 0.05
        mlp_params = mlp_model.init_hybrid_mlp_params(
            key_mlp,
            num_ranks=mlp_model.NUM_RANKS,
            in_dim=mlp_model.HYBRID_IN_DIM,
            hidden=mlp_hidden,
            num_layers=mlp_layers,
            scale=sigma_mlp,
        )
        q_param = jnp.asarray(-3.5, dtype=jnp.float32)

    else:
        print("Continuing training (hybrid MLP)")
        ckpt = f"models/model_{run_id}.npz"
        params = protax_utils.read_model(ckpt, tree.ranks)
        mlp_params = mlp_model.load_mlp_params_from_npz(ckpt)
        initial_value = jnp.log(q_perc / (1 - q_perc))
        q_param = jnp.array(initial_value)

    print("Creating design matrices...")
    X_all, R = create_design_matrices(base_dir, train_dir, tree, train_config, N, params)
    beta_placeholder = np.zeros((8, 6), dtype=np.float32)

    train_with_q = False
    if decay_lr:
        warmup_ratio = 0.1
        steps_per_epoch = math.ceil(R / B)
        total_steps = steps_per_epoch * train_config["num_epochs"]
        warmup_steps = int(total_steps * warmup_ratio)
        step = 0
        lr = lr_schedule(step, total_steps, warmup_steps, base_lr)
    else:
        lr = base_lr

    print("Training hybrid_mlp model...")
    epoch_loss_hist = []

    for e in tqdm(range(1, train_config["num_epochs"] + 1), desc=f"training model", file=sys.stderr, dynamic_ncols=True):

        print(f"epoch {e} / {train_config['num_epochs']}", flush=True)
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)

        traversal = np.random.permutation(R)
        for start in range(0, len(traversal), B):

            if decay_lr:
                lr = lr_schedule(step, total_steps, warmup_steps, base_lr)
                train_with_q = step >= warmup_steps
                step += 1

            idx = np.asarray(traversal[start : start + B], dtype=np.int32)
            X_batch = jnp.asarray(X_all[idx], dtype=jnp.float32)
            y_batch = jnp.take(targ_jnp, jnp.asarray(idx, dtype=jnp.int32), axis=0)

            if train_with_q:
                mlp_params, q_param, batch_loss_sum = train_step_with_q(mlp_params, q_param, X_batch, y_batch, lr, l2)
            else:
                mlp_params, q_param, batch_loss_sum = train_step_no_q(mlp_params, q_param, X_batch, y_batch, lr, l2)
            loss_sum += batch_loss_sum

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item(), flush=True)

        epoch_loss = loss_sum / float(R)
        epoch_loss_hist.append(float(epoch_loss))
        print("total loss: ", float(epoch_loss), flush=True)
        print("lr: ", lr, flush=True)

        mf = Path(f"models/model_{run_id}.npz")
        save_dict = {
            "beta": beta_placeholder,
            "scalings": np.array(params.sc_conc),
            "mlp_num_layers": np.array(len(mlp_params.hidden_W), dtype=np.int32),
            "mlp_hidden": np.array(mlp_hidden),
        }
        for li, (W, b) in enumerate(zip(mlp_params.hidden_W, mlp_params.hidden_b)):
            save_dict[f"mlp_hidden_W_{li}"] = np.array(W)
            save_dict[f"mlp_hidden_b_{li}"] = np.array(b)
        save_dict["mlp_out_W"] = np.array(mlp_params.out_W)
        save_dict["mlp_out_b"] = np.array(mlp_params.out_b)

        # Keep legacy keys for compatibility when depth=1.
        if len(mlp_params.hidden_W) == 1:
            save_dict["mlp_W"] = np.array(mlp_params.hidden_W[0])
            save_dict["mlp_b"] = np.array(mlp_params.hidden_b[0])
            save_dict["mlp_v"] = np.array(mlp_params.out_W)
            save_dict["mlp_c"] = np.array(mlp_params.out_b)

        np.savez_compressed(mf.resolve(), **save_dict)

    plt.plot(epoch_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (hybrid_mlp)")
    plt.grid(True)
    plt.savefig(f"loss_curve_{run_id}_hybrid_mlp.png", dpi=300, bbox_inches="tight")
    plt.close()

    end_time = time.time()
    print(f"Time taken to train: {(end_time - start_time) / 3600} hours")

    return X_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid + per-rank MLP model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--scalings_dir", type=str, help="Path to scalings / initial model npz")
    parser.add_argument("--tc", type=str, help="Training hyperparameters JSON")
    parser.add_argument("--id", type=str, help="ID of the run")
    parser.add_argument("--exp", type=str, help="Experiment details")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    base_dir = data_dir / "base_embeddings.npz"
    train_dir = data_dir / "train_embeddings.npz"
    train_labels = data_dir / "train_labels.csv"
    scalings_dir = Path(args.scalings_dir)
    exp_details = args.exp
    tc = json.loads(args.tc)
    run_id = args.id
    continue_training = tc["continue_training"]
    
    decay_lr = tc["decay_lr"]
    X_all = train(tax_dir, base_dir, train_dir, train_labels, scalings_dir, tc, run_id, continue_training, decay_lr=decay_lr)

    classify_file(train_dir, base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=True, X_all=X_all)
    for class_level in ["species", "genus", "family"]:
        evaluate(f"results_{run_id}.csv", train_labels, "", exp_details, class_level, tax_dir, run_id)