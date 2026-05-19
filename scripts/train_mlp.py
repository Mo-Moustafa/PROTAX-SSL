"""
Train hybrid + per-rank MLP (same data path as train_gd_hybrid.py).
Checkpoints: beta (8,6) zeros placeholder for read_model_jax), scalings, mlp_W/b/v/c.
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
import random
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
        return base_lr * (step / warmup_steps)
    else:
        return base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))

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


def make_vectorized_batch_ops_mlp(tree_for_lp, lvl, segnum):
    def forward_one(X, mlp_params, y_ind, q_param, train_with_q):
        log_probs = mlp_model.fill_log_bprob(X, mlp_params, tree_for_lp, segnum, lvl)
        return CE_loss(log_probs, y_ind, q_param, tree_for_lp, train_with_q)

    def _loss_mlp(mlp_params, x, y, q_param, train_with_q):
        return forward_one(x, mlp_params, y, q_param, train_with_q)

    def _single_mlp_grad(mlp_params, x, y, q_param, train_with_q):
        return jax.grad(_loss_mlp, argnums=0)(mlp_params, x, y, q_param, train_with_q)

    @partial(jax.jit, static_argnums=(4,))
    def batched_mlp_grad(mlp_params, X_batch, y_batch, q_param, train_with_q):
        return jax.vmap(_single_mlp_grad, in_axes=(None, 0, 0, None, None))(
            mlp_params, X_batch, y_batch, q_param, train_with_q
        )

    def _loss_q(q_param, x, y, mlp_params, train_with_q):
        return forward_one(x, mlp_params, y, q_param, train_with_q)

    def _single_q_grad(q_param, x, y, mlp_params, train_with_q):
        return jax.grad(_loss_q, argnums=0)(q_param, x, y, mlp_params, train_with_q)

    @partial(jax.jit, static_argnums=(4,))
    def batched_q_grad(mlp_params, X_batch, y_batch, q_param, train_with_q):
        return jax.vmap(_single_q_grad, in_axes=(None, 0, 0, None, None))(
            q_param, X_batch, y_batch, mlp_params, train_with_q
        )

    @partial(jax.jit, static_argnums=(4,))
    def batched_forward(mlp_params, X_batch, y_batch, q_param, train_with_q):
        def one(x, y):
            return forward_one(x, mlp_params, y, q_param, train_with_q)

        return jax.vmap(one)(X_batch, y_batch)

    return batched_mlp_grad, batched_q_grad, batched_forward


def get_targ(target_dir):
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
    lvl = tax["ranks"]

    if beta.shape[0] == 7:
        print("Padding beta to (8, 6)...")
        zero_row = np.zeros((1, 6))
        beta = np.vstack([zero_row, beta])

    if sc.shape[0] == 7:
        print("Padding scalings to (8, 4)...")
        padding_row = np.array([[0, 1, 0, 1]])
        sc = np.vstack([padding_row, sc])

    beta = jnp.array(beta)
    sc = jnp.array(sc)

    return beta, lvl, sc


def train(tax_dir, base_dir, train_dir, targ_dir, scalings_dir, train_config, run_id, continue_training, q_perc=0.5, stop_at=1000):
    mlp_hidden = int(train_config.get("mlp_hidden", 16))

    if not continue_training:
        print("Training from scratch (hybrid MLP)")
        tree, params, N, segnum = protax_utils.read_model_jax(scalings_dir, tax_dir)
        pkey = jax.random.PRNGKey(0)
        key_mlp, key_q = jax.random.split(pkey)
        mlp_params = mlp_model.init_hybrid_mlp_params(
            key_mlp,
            num_ranks=mlp_model.NUM_RANKS,
            in_dim=mlp_model.HYBRID_IN_DIM,
            hidden=mlp_hidden,
            scale=5.0,
        )
        q_param = jnp.asarray(0.0, dtype=jnp.float32)
        q_mult = 1.0
        _, lvl, sc = load_params(scalings_dir, tax_dir)

    else:
        print("Continuing training (hybrid MLP)")
        ckpt = f"models/model_{run_id}.npz"
        tree, params, N, segnum = protax_utils.read_model_jax(ckpt, tax_dir)
        mlp_params = mlp_model.load_mlp_params_from_npz(ckpt)
        _, lvl, sc = load_params(ckpt, tax_dir)
        initial_value = jnp.log(q_perc / (1 - q_perc))
        q_param = jnp.array(initial_value)
        q_mult = 1.0

    n2s = tree.node2seq
    node_state = tree.node_state
    targ = get_targ(targ_dir)

    base_lr = train_config["learning_rate"]
    l2 = float(train_config["l2"])
    train_with_q = train_config["train_with_q"]

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

    print("Reading base embeddings...")
    base_embeddings = np.load(base_dir)
    bert_base = jnp.array(base_embeddings["bert"])
    mamba_base = jnp.array(base_embeddings["mamba"])
    assert bert_base.shape == mamba_base.shape
    print("base embeddings.shape: ", bert_base.shape, mamba_base.shape)
    train_embeddings = np.load(train_dir)
    bert_train = jnp.array(train_embeddings["bert"])
    mamba_train = jnp.array(train_embeddings["mamba"])
    assert bert_train.shape == mamba_train.shape
    print("query embeddings.shape: ", bert_train.shape, mamba_train.shape)

    R = int(bert_train.shape[0])
    X_all = None
    for i in tqdm(range(R), desc="Computing design matrices", file=sys.stderr, dynamic_ncols=True, mininterval=5):
        dists_bert = model.cosine_dist(bert_train[i], bert_base)
        dists_mamba = model.cosine_dist(mamba_train[i], mamba_base)
        new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
        tree_m = tree._replace(node2seq=new_node2seq, node_state=new_node_state)

        X = model.get_hybrid_X(
            dists_bert,
            dists_mamba,
            tree_m,
            N,
            params.sc_mean,
            params.sc_var,
            dist_scaling,
            pt_min,
            pt_gap,
        )
        x_host = np.asarray(jax.device_get(X), dtype=np.float32)
        if X_all is None:
            n_nodes, F_dim = x_host.shape
            X_all = np.zeros((R, n_nodes, F_dim), dtype=np.float32)
        X_all[i] = x_host

    tree_for_lp = tree._replace(node2seq=n2s, node_state=node_state)
    lvl = jnp.asarray(lvl, dtype=jnp.int32)
    targ_np = np.asarray(jax.device_get(targ), dtype=np.int32)
    batched_mlp_grad, batched_q_grad, batched_forward = make_vectorized_batch_ops_mlp(tree_for_lp, lvl, segnum)

    print("Training hybrid_mlp model...")
    start_time = time.time()

    epoch_loss_hist = []

    warmup_ratio = 0.05
    steps_per_epoch = math.ceil(R / train_config["batch_size"])
    total_steps = steps_per_epoch * train_config["num_epochs"]
    warmup_steps = int(total_steps * warmup_ratio)
    step = 0
    lr = lr_schedule(step, total_steps, warmup_steps, base_lr)

    beta_placeholder = np.zeros((8, 6), dtype=np.float32)

    for e in range(1, train_config["num_epochs"] + 1):
        # if e > stop_at:
        #     break

        if e == 6:
            # classify_file(data_dir / "test_embeddings.npz", base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=False, X_all=None)
            # evaluate(f"results_{run_id}.csv", data_dir / "test_labels.csv", "", exp_details, "species", tax_dir, run_id)
            train_with_q = True

        if e == 6:
        #     lr = lr * 0.5
        #     print("\n--------------------------------")
        #     print("Decreasing lr to: ", lr)
            train_with_q = True
        # elif e == 21:
        #     lr = lr * 0.5
        #     print("\n--------------------------------")
        #     print("Decreasing lr to: ", lr)
        # elif e == 31:
        #     lr = lr * 0.5
        #     print("\n--------------------------------")
        #     print("Decreasing lr to: ", lr)

        print(f"epoch {e} / {train_config['num_epochs']}")

        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)

        traversal = list(range(R))
        random.shuffle(traversal)

        B = int(train_config["batch_size"])
        for start in tqdm(
            range(0, len(traversal), B),
            desc=f"epoch {e} batches",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=5,
        ):
            lr = lr_schedule(step, total_steps, warmup_steps, base_lr)
            step += 1
            chunk = traversal[start : start + B]
            idx = np.asarray(chunk, dtype=np.int32)
            X_batch = jnp.asarray(X_all[idx], dtype=jnp.float32)
            y_batch = jnp.asarray(targ_np[idx], dtype=jnp.int32)

            loss_sum += jnp.sum(batched_forward(mlp_params, X_batch, y_batch, q_param, train_with_q))

            G_mlp = batched_mlp_grad(mlp_params, X_batch, y_batch, q_param, train_with_q)
            if train_with_q:
                G_q = batched_q_grad(mlp_params, X_batch, y_batch, q_param, train_with_q)
            mean_G_mlp = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), G_mlp)
            mlp_params = jax.tree_util.tree_map(
                lambda p, g: p - lr * (g + l2 * p),
                mlp_params,
                mean_G_mlp,
            )
            if train_with_q:
                q_step = jnp.clip(jnp.mean(G_q), -10.0, 10.0)
                q_param = q_param - (lr * q_mult * q_step)

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item())

        epoch_loss = loss_sum / float(R)
        epoch_loss_hist.append(float(epoch_loss))
        print("total loss: ", float(epoch_loss))

        mf = Path(f"models/model_{run_id}.npz")
        np.savez_compressed(
            mf.resolve(),
            beta=beta_placeholder,
            scalings=np.array(sc),
            mlp_W=np.array(mlp_params.W),
            mlp_b=np.array(mlp_params.b),
            mlp_v=np.array(mlp_params.v),
            mlp_c=np.array(mlp_params.c),
            mlp_hidden=np.array(mlp_hidden),
        )

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


def str2bool(s: str) -> bool:
    return s.lower() == "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid + per-rank MLP model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--scalings_dir", type=str, help="Path to scalings / initial model npz")
    parser.add_argument("--tc", type=str, help="Training hyperparameters JSON")
    parser.add_argument("--id", type=str, help="ID of the run")
    parser.add_argument("--continue_training", type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--exp", type=str, help="Experiment details")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    base_dir = data_dir / "base_embeddings.npz"
    train_dir = data_dir / "train_embeddings.npz"
    targ_dir = data_dir / "train-targets.csv"
    train_labels = data_dir / "train_labels.csv"
    scalings_dir = Path(args.scalings_dir)
    exp_details = args.exp
    tc = json.loads(args.tc)
    run_id = args.id
    continue_training = args.continue_training

    X_all = train(tax_dir, base_dir, train_dir, targ_dir, scalings_dir, tc, run_id, continue_training)

    classify_file(train_dir, base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=True, X_all=X_all)
    for class_level in ["species", "genus", "family"]:
        evaluate(f"results_{run_id}.csv", train_labels, "", exp_details, class_level, tax_dir, run_id)
