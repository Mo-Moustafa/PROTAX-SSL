"""
Train hybrid + per-rank MLP (same data path as train_gd_hybrid.py).
Checkpoints: beta (8,6) zeros placeholder for read_model_jax), scalings, mlp_W/b/v/c.
"""
import protax.model_bert as model
import protax.model_hybrid_mlp as mlp_model
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

from tqdm import tqdm
import sys

import joblib


def CE_loss(log_probs, y_ind, q_param, tree, train_with_q, epoch):
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


def forward(X, tree, mlp_params, segnum, y_ind, ranks, q_param, train_with_q, epoch):
    log_probs = mlp_model.fill_log_bprob(X, mlp_params, tree, segnum, ranks)
    return CE_loss(log_probs, y_ind, q_param, tree, train_with_q, epoch)


f_grad_mlp = jax.jit(jax.grad(forward, argnums=(2)), static_argnums=(3, 7, 8))
f_grad_q = jax.jit(jax.grad(forward, argnums=(6)), static_argnums=(3, 7, 8))
forward_jit = jax.jit(forward, static_argnums=(3, 7, 8))


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


def train(
    tax_dir,
    base_dir,
    train_dir,
    targ_dir,
    scalings_dir,
    train_config,
    run_id,
    continue_training,
    q_perc=0.5,
    stop_at=1000,
):
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

    lr = train_config["learning_rate"]
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

    epoch_loss_hist = []

    print("Training hybrid_mlp model...")
    start_time = time.time()

    # Placeholder beta for checkpoint compatibility with read_model_jax (unused by this model).
    beta_placeholder = np.zeros((8, 6), dtype=np.float32)

    for e in range(1, train_config["num_epochs"] + 1):
        if e > stop_at:
            break

        print(f"epoch {e} / {train_config['num_epochs']}")

        mlp_grad_acc = None  # sum of MLP grads in current (partial) batch
        q_grad = 0
        loss_sum = 0
        batch_loss = 0
        batch_count = 0

        traversal = list(range(bert_train.shape[0]))
        random.shuffle(traversal)

        for i in tqdm(traversal, file=sys.stderr, dynamic_ncols=True, mininterval=5):
            batch_count += 1

            query_bert = bert_train[i]
            query_mamba = mamba_train[i]

            dists_bert = model.seq_dist(query_bert, bert_base)
            dists_mamba = model.seq_dist(query_mamba, mamba_base)

            if i < bert_base.shape[0]:
                new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
                tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)
            else:
                tree = tree._replace(node2seq=n2s, node_state=node_state)

            X = model.get_hybrid_X(
                dists_bert,
                dists_mamba,
                tree,
                N,
                params.sc_mean,
                params.sc_var,
                dist_scaling,
                pt_min,
                pt_gap,
            )

            g = f_grad_mlp(
                X,
                tree,
                mlp_params,
                segnum,
                targ.at[i].get(),
                lvl,
                q_param,
                train_with_q,
                e,
            )
            if mlp_grad_acc is None:
                mlp_grad_acc = g
            else:
                mlp_grad_acc = jax.tree_util.tree_map(lambda a, b: a + b, mlp_grad_acc, g)

            if train_with_q:
                q_grad += f_grad_q(
                    X,
                    tree,
                    mlp_params,
                    segnum,
                    targ.at[i].get(),
                    lvl,
                    q_param,
                    train_with_q,
                    e,
                )

            batch_loss += forward_jit(
                X,
                tree,
                mlp_params,
                segnum,
                targ.at[i].get(),
                lvl,
                q_param,
                train_with_q,
                e,
            )

            if batch_count >= train_config["batch_size"]:
                mlp_params = jax.tree_util.tree_map(
                    lambda p, g_: p - lr * (g_ / batch_count),
                    mlp_params,
                    mlp_grad_acc,
                )
                mlp_grad_acc = None

                if train_with_q:
                    q_grad = q_grad / batch_count
                    q_grad = jnp.clip(q_grad, -10.0, 10.0)
                    q_param = q_param - (lr * q_mult * q_grad)
                    q_grad = 0

                loss_sum += batch_loss
                batch_loss = 0
                batch_count = 0

        if batch_count > 0:
            if mlp_grad_acc is not None:
                mlp_params = jax.tree_util.tree_map(
                    lambda p, g_: p - lr * (g_ / batch_count),
                    mlp_params,
                    mlp_grad_acc,
                )
                mlp_grad_acc = None
            if train_with_q:
                q_grad = q_grad / batch_count
                q_grad = jnp.clip(q_grad, -10.0, 10.0)
                q_param = q_param - (lr * q_mult * q_grad)
                q_grad = 0
            loss_sum += batch_loss

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item())

        epoch_loss = loss_sum / bert_train.shape[0]
        epoch_loss_hist.append(epoch_loss)
        print("total loss: ", epoch_loss)

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


def str2bool(s: str) -> bool:
    return s.lower() == "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid + per-rank MLP model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--scalings_dir", type=str, help="Path to scalings / initial model npz")
    parser.add_argument("--tc", type=str, help="Training hyperparameters JSON")
    parser.add_argument("--id", type=str, help="ID of the run")
    parser.add_argument("--continue_training", type=str2bool, choices=[True, False], default=False)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    base_dir = data_dir / "base_embeddings.npz"
    train_dir = data_dir / "train_embeddings.npz"
    targ_dir = data_dir / "train-targets.csv"
    scalings_dir = Path(args.scalings_dir)

    tc = json.loads(args.tc)
    run_id = args.id
    continue_training = args.continue_training

    train(
        tax_dir,
        base_dir,
        train_dir,
        targ_dir,
        scalings_dir,
        tc,
        run_id,
        continue_training,
    )
