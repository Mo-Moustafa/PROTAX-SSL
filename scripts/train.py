import protax.model as model

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
import math
from functools import partial
from tqdm import tqdm
import sys
import joblib
from scripts.calibration import evaluate
from protax.classify import classify_file

def lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
    else:
        lr = base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))    
    
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

def make_vectorized_batch_ops(tree_for_lp, lvl, segnum, q_mult):
    """
    vmap grad / forward over batch dimension. X differs per sample; tree_for_lp
    supplies segments/paths/prior for fill_log_bprob / CE_loss.
    """

    def _loss_with_beta_n(beta_n, x, y, q_param, train_with_q):
        log_probs = model.fill_log_bprob(x, beta_n, tree_for_lp, segnum)
        return CE_loss(log_probs, y, q_param, tree_for_lp, train_with_q)

    def _batch_losses_no_q(beta, X_batch, y_batch, q_param):
        beta_n = jnp.take(beta, lvl, axis=0)
        one = lambda x, y: _loss_with_beta_n(beta_n, x, y, q_param, False)
        return jax.vmap(one)(X_batch, y_batch)

    def _batch_losses_with_q(beta, X_batch, y_batch, q_param):
        beta_n = jnp.take(beta, lvl, axis=0)
        one = lambda x, y: _loss_with_beta_n(beta_n, x, y, q_param, True)
        return jax.vmap(one)(X_batch, y_batch)

    @jax.jit
    def train_step_no_q(beta, q_param, X_batch, y_batch, lr, l2):
        def mean_loss_beta_only(beta_local):
            return jnp.mean(_batch_losses_no_q(beta_local, X_batch, y_batch, q_param))

        mean_loss, beta_grad = jax.value_and_grad(mean_loss_beta_only)(beta)
        beta_new = beta - lr * (beta_grad + l2 * beta)
        batch_loss_sum = mean_loss * jnp.asarray(X_batch.shape[0], dtype=mean_loss.dtype)
        return beta_new, q_param, batch_loss_sum, beta_grad

    @jax.jit
    def train_step_with_q(beta, q_param, X_batch, y_batch, lr, l2):
        def mean_loss_beta_q(beta_local, q_local):
            return jnp.mean(_batch_losses_with_q(beta_local, X_batch, y_batch, q_local))

        mean_loss, (beta_grad, q_grad) = jax.value_and_grad(mean_loss_beta_q, argnums=(0, 1))(beta, q_param)
        q_step = jnp.clip(q_grad, -10.0, 10.0)
        q_param_new = q_param - (lr * q_mult * q_step)
        beta_new = beta - lr * (beta_grad + l2 * beta)
        batch_loss_sum = mean_loss * jnp.asarray(X_batch.shape[0], dtype=mean_loss.dtype)
        return beta_new, q_param_new, batch_loss_sum, beta_grad

    return train_step_no_q, train_step_with_q


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
            # assert bert_base.shape == mamba_base.shape
            base_length = int(bert_base.shape[0])

            bert_train = jnp.array(train_embeddings["bert"])
            mamba_train = jnp.array(train_embeddings["mamba"])
            # assert bert_train.shape == mamba_train.shape
            R = int(bert_train.shape[0])

    elif variant == "og":
        seq_list, ok_list = protax_utils.read_refs(train_dir, padding_len=tree.max_seq_length)
        print("Read reference sequences successfully")
        R = int(seq_list.shape[0])
        base_length = int(tree.refs.shape[0])
    
    else:
        raise ValueError(f"Model {variant} not supported")

    X_all = None    
    for i in tqdm(range(R), desc="Computing design matrices", file=sys.stderr, dynamic_ncols=True, mininterval=1000):
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


def train(variant, tax_dir, base_dir, train_dir, targ_dir, scalings_dir, train_config, run_id, continue_training, q_perc=0.5, decay_lr=False):

    start_time = time.time()
    
    q_mult = 0.1                                # Change if heavier mislabeling regularization is desired
    base_lr = train_config["learning_rate"]
    l2 = float(train_config["l2"])
    B = int(train_config["batch_size"])

    tree, N, segnum = protax_utils.read_tree(tax_dir)
    tree_for_lp = tree
    lvl = jnp.asarray(tree.ranks, dtype=jnp.int32)
    targ_jnp = protax_utils.get_targets(targ_dir)
    train_step_no_q, train_step_with_q = make_vectorized_batch_ops(tree_for_lp, lvl, segnum, q_mult)

    if not continue_training:
        print("Training from scratch")
        params = protax_utils.read_model(scalings_dir, tree.ranks)
        key_beta = jax.random.PRNGKey(0)
        sigma_beta = 5.0
        if variant in ("bert", "mamba", "og"):
            beta = jax.random.normal(key_beta, (8, 4)) * sigma_beta
        elif variant == "hybrid_lin":
            beta = jax.random.normal(key_beta, (8, 6)) * sigma_beta

        q_param = jnp.asarray(0.0, dtype=jnp.float32)

    else:
        print("Continuing training")
        params = protax_utils.read_model(f"models/model_{run_id}.npz", tree.ranks)
        beta = params.beta_conc
        initial_value = jnp.log(q_perc / (1 - q_perc))
        q_param = jnp.array(initial_value)

    print("Creating design matrices...")
    X_all, R = create_design_matrices(variant, base_dir, train_dir, tree, train_config, N, params)
    
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

    print("Training started...")
    epoch_loss_hist = []

    for e in tqdm(range(1, train_config["num_epochs"] + 1), desc=f"training model", file=sys.stderr, dynamic_ncols=True):

        print(f"epoch {e} / {train_config['num_epochs']}", flush=True)
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        beta_grad = None

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
                beta, q_param, batch_loss_sum, beta_grad = train_step_with_q(beta, q_param, X_batch, y_batch, lr, l2)
            else:
                beta, q_param, batch_loss_sum, beta_grad = train_step_no_q(beta, q_param, X_batch, y_batch, lr, l2)
            loss_sum += batch_loss_sum

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item(), flush=True)

        epoch_loss = loss_sum / float(R)
        epoch_loss_hist.append(float(epoch_loss))
        print("total loss: ", float(epoch_loss), flush=True)
        # print("beta:\n", np.array(beta), flush=True)
        # print("beta_grad:\n", np.array(beta_grad), flush=True)
        
        # save checkpoint
        mf = Path(f"models/model_{run_id}.npz")
        np.savez_compressed(mf.resolve(), beta=np.array(beta), scalings=params.sc_conc)


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


if __name__ == "__main__":
    # parse config from command line
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--scalings_dir", type=str, help="Path to scalings")
    parser.add_argument('--tc', type=str, help='Training Hyperparameters')
    parser.add_argument("--exp", type=str, help="Experiment details")
    parser.add_argument("--id", type=str, help="ID of the run")
    args = parser.parse_args()

    tc = json.loads(args.tc)
    decay_lr = tc["decay_lr"]
    variant = tc["model"]

    scalings_dir = Path(args.scalings_dir)
    data_dir = Path(args.data_dir)
    tax_dir = data_dir / "taxonomy.npz"
    train_labels = data_dir / "train_labels.csv"
    if variant == "og":
        base_dir = None
        train_dir = data_dir / "train.aln"
    else:
        base_dir = data_dir / "base_embeddings.npz"
        train_dir = data_dir / "train_embeddings.npz"

    exp_details = args.exp
    run_id = args.id
    continue_training = tc["continue_training"]

    X_all = train(variant, tax_dir, base_dir, train_dir, train_labels, scalings_dir, tc, run_id, continue_training, decay_lr=decay_lr)
    classify_file(variant, train_dir, base_dir, Path(f"models/model_{run_id}.npz"), tax_dir, tc["dist_scaling"], run_id, train_eval=True, X_all=X_all)    
    for class_level in ["species", "genus", "family"]:
        evaluate(f"results_{run_id}.csv", train_labels, "", exp_details, class_level, tax_dir, run_id)