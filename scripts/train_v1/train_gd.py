import protax.model as model
from protax import protax_utils

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

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

from scripts.calibration import evaluate
from protax.classify import classify_file


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

def forward(X, tree, beta, segnum, y_ind, lvl, q_param, train_with_q, epoch):
    beta = jnp.take(beta, lvl, axis=0)
    log_probs = model.fill_log_bprob(X, beta, tree, segnum)
    return CE_loss(log_probs, y_ind, q_param, tree, train_with_q, epoch)

f_grad_beta = jax.jit(jax.grad(forward, argnums=(2)), static_argnums=(3, 7, 8))
f_grad_q = jax.jit(jax.grad(forward, argnums=(6)), static_argnums=(3, 7, 8))
forward_jit = jax.jit(forward, static_argnums=(3, 7, 8))


def get_targ(target_dir):
    """
    Get node id for each reference sequence at lowest level
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


def train(tax_dir, train_dir, targ_dir, scalings_dir, train_config, run_id, continue_training, q_perc=0.5, stop_at=100):

    if not continue_training:
        print("Training from scratch")
        tree, params, N, segnum = protax_utils.read_model_jax(scalings_dir, tax_dir)
        pkey = jax.random.PRNGKey(0)
        key_beta, key_q = jax.random.split(pkey)
        sigma_beta = 5.0
        # sigma_q = 0.1
        beta = jax.random.normal(key_beta, (8, 4)) * sigma_beta
        # q_param = jax.random.normal(key_q, ()) * sigma_q
        q_param = jnp.asarray(0.0, dtype=jnp.float32)
        q_mult = 1.0
        _, lvl, sc = load_params(scalings_dir, tax_dir)

    else:
        print("Continuing training")
        tree, params, N, segnum = protax_utils.read_model_jax(f"models/model_{run_id}.npz", tax_dir)
        beta, lvl, sc = load_params(f"models/model_{run_id}.npz", tax_dir)
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

    seq_list, ok_list = protax_utils.read_refs(train_dir, padding_len=tree.max_seq_length)
    print("Read reference sequences successfully")
    
    # params and node lvl
    epoch_loss_hist = []

    print("Training model...")
    start_time = time.time()

    for e in range(1, train_config["num_epochs"] + 1):
        if e > stop_at:
            break

        print(f"epoch {e} / {train_config['num_epochs']}")

        beta_grad = 0
        q_grad = 0
        loss_sum = 0
        batch_loss = 0
        batch_count = 0  # number of samples in current batch

        traversal = list(range(seq_list.shape[0]))
        random.shuffle(traversal)

        # minibatch
        for i in tqdm(traversal, file=sys.stderr, dynamic_ncols=True, mininterval=5):
            batch_count += 1
            # mask out tree
            q_seq = seq_list[i]
            ok = ok_list[i]

            if i < seq_list.shape[0]:
                new_node2seq, new_node_state = protax_utils.mask_n2s(n2s, node_state, i)
                tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)
            else:
                tree = tree._replace(node2seq=n2s, node_state=node_state)
            
            X = model.get_X(q_seq, ok, tree, N, params.sc_mean, params.sc_var, dist_scaling, pt_min, pt_gap)

            beta_grad += f_grad_beta(
                X,
                tree,
                beta,
                segnum,
                targ.at[i].get(),
                lvl,
                q_param,
                train_with_q,
                e,
            )

            if train_with_q:
                q_grad += f_grad_q(
                    X,
                    tree,
                    beta,
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
                beta,
                segnum,
                targ.at[i].get(),
                lvl,
                q_param,
                train_with_q,
                e,
            )

            # Update every batch_size samples (by count, not by value of i)
            if batch_count >= train_config["batch_size"]:
                # Use mean gradient so LR is batch-size invariant
                beta = beta - lr * (beta_grad / batch_count)
                beta_grad = 0

                if train_with_q:
                    q_grad = q_grad / batch_count
                    q_grad = jnp.clip(q_grad, -10.0, 10.0)
                    q_param = q_param - (lr * q_mult * q_grad)
                    q_grad = 0

                loss_sum += batch_loss
                batch_loss = 0
                batch_count = 0

        # Final update for remainder (last partial batch in the epoch)
        if batch_count > 0:
            beta = beta - lr * (beta_grad / batch_count)
            beta_grad = 0
            if train_with_q:
                q_grad = q_grad / batch_count
                q_grad = jnp.clip(q_grad, -10.0, 10.0)
                q_param = q_param - (lr * q_mult * q_grad)
                q_grad = 0
            loss_sum += batch_loss

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item())

        epoch_loss = loss_sum / seq_list.shape[0]
        epoch_loss_hist.append(epoch_loss)
        print("total loss: ", epoch_loss)

        # save checkpoint
        mf = Path(f"models/model_{run_id}.npz")
        np.savez_compressed(mf.resolve(), beta=np.array(beta), scalings=sc)

        # if e % 3 == 0:
        #     classify_file("models/ref_db/train_test/test.aln", f"models/model_{run_id}.npz", tax_dir, e)
        #     evaluate(f"results_{e}.csv", "labels_dir", "", "", e)
        
    plt.plot(epoch_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(f"loss_curve_{run_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    end_time = time.time()
    print(f"Time taken to train: {(end_time - start_time) / 3600} hours")



def str2bool(s: str) -> bool:
    return s.lower() == "true"

if __name__ == "__main__":
    # parse config from command line
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--scalings_dir", type=str, help="Path to scalings")
    parser.add_argument('--tc', type=str, help='Training Hyperparameters')
    parser.add_argument("--id", type=str, help="ID of the run")
    parser.add_argument("--continue_training", type=str2bool, choices=[True, False], default=False)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    train_dir = data_dir / "train.aln"
    targ_dir = data_dir / "train-targets.csv"
    scalings_dir = Path(args.scalings_dir)

    tc = json.loads(args.tc)
    run_id = args.id
    continue_training = args.continue_training

    train(tax_dir, train_dir, targ_dir, scalings_dir, tc, run_id, continue_training)