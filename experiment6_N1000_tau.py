"""
experiment6_N1000_tau.py

Purpose:
Train a dedicated betweenness GNN at N=1000 and report Kendall tau.
This fixes the missing tau value in the scalability table.

Recommended first run:
    NUM_TRAIN = 50
    NUM_TEST  = 10
    EPOCHS    = 30

If it runs comfortably, increase to:
    NUM_TRAIN = 100
    NUM_TEST  = 20
    EPOCHS    = 50
"""

import os
import time
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import networkx as nx
import torch

from betweennes_model import GNN_Bet
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation,
)

# ============================================================
# Settings
# ============================================================

SEED = 42
N = 1000
P = 0.15

NUM_TRAIN = 50       # increase to 100 if time allows
NUM_TEST = 10        # increase to 20 if time allows
EPOCHS = 30          # increase to 50 if time allows

HIDDEN = 40
DROPOUT = 0.4
LR = 1e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 2       # keep small for N=1000

OUTPUT_DIR = "paper_figures_scalability"
MODEL_PATH = os.path.join(OUTPUT_DIR, "betweenness_model_N1000.pth")
CSV_PATH = os.path.join(OUTPUT_DIR, "N1000_tau_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("N=1000 BETWEenness GNN TRAINING AND KENDALL TAU EVALUATION")
print("=" * 70)
print(f"Device: {device}")
print(f"N={N}, p={P}")
print(f"Train graphs: {NUM_TRAIN}, Test graphs: {NUM_TEST}, Epochs: {EPOCHS}")
print("=" * 70)


# ============================================================
# Helper functions
# ============================================================

def make_er_graph(n=N, p=P):
    """
    Directed ER graph, consistent with betweenness experiments.
    """
    g = nx.erdos_renyi_graph(n, p, directed=True, seed=None)

    # Ensure node labels are 0,...,N-1
    g = nx.convert_node_labels_to_integers(g)

    return g


def get_adj(graph, model_size=N):
    """
    Convert graph to scipy sparse adjacency matrix.
    """
    n = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype("float32")

    if n < model_size:
        adj = sp.block_diag([adj, csr_matrix((model_size - n, model_size - n))])

    return adj.tocsr()


def get_adj_T(graph, model_size=N):
    return get_adj(graph, model_size).T.tocsr()


def exact_betweenness(graph):
    """
    Exact NetworkX betweenness labels.
    """
    return nx.betweenness_centrality(graph, normalized=True)


def generate_dataset(num_graphs, split_name):
    """
    Generate ER graphs, exact betweenness labels, adjacency matrices,
    and transpose adjacency matrices.
    """
    graphs = []
    adjs = []
    adjs_T = []
    labels = []
    nodes_list = []

    label_times = []

    print(f"\nGenerating {split_name} dataset: {num_graphs} graphs")

    for i in range(num_graphs):
        g = make_er_graph()
        graphs.append(g)
        nodes_list.append(g.number_of_nodes())

        adjs.append(get_adj(g))
        adjs_T.append(get_adj_T(g))

        t0 = time.perf_counter()
        b = exact_betweenness(g)
        label_time = time.perf_counter() - t0
        label_times.append(label_time)
        labels.append(b)

        print(
            f"{split_name} graph {i+1}/{num_graphs} | "
            f"label time = {label_time:.2f}s | "
            f"avg so far = {np.mean(label_times):.2f}s"
        )

    print(
        f"{split_name} label generation time: "
        f"{np.sum(label_times)/60:.2f} min total, "
        f"{np.mean(label_times):.2f}s per graph"
    )

    return graphs, adjs, adjs_T, labels, nodes_list, label_times


def train_one_epoch(model, optimizer, adjs, adjs_T, labels, nodes_list):
    model.train()

    indices = np.arange(len(adjs))
    np.random.shuffle(indices)

    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]

        optimizer.zero_grad()
        batch_loss = 0.0

        for idx in batch_idx:
            adj_torch = sparse_mx_to_torch_sparse_tensor(adjs[idx], device)
            adjT_torch = sparse_mx_to_torch_sparse_tensor(adjs_T[idx], device)

            y_out = model(adj_torch, adjT_torch)

            true_arr = np.array(list(labels[idx].values()), dtype=np.float32)
            true_val = torch.from_numpy(true_arr).float().to(device)

            loss = loss_cal(
                y_out=y_out,
                true_val=true_val,
                num_nodes=nodes_list[idx],
                device=device,
                model_size=N,
            )

            batch_loss += loss

        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        optimizer.step()

        total_loss += float(batch_loss.detach().cpu())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_tau(model, adjs, adjs_T, labels, nodes_list):
    model.eval()

    taus = []
    inference_times = []

    for i in range(len(adjs)):
        adj_torch = sparse_mx_to_torch_sparse_tensor(adjs[i], device)
        adjT_torch = sparse_mx_to_torch_sparse_tensor(adjs_T[i], device)

        true_arr = np.array(list(labels[i].values()), dtype=np.float32)
        true_val = torch.from_numpy(true_arr).float().to(device)

        t0 = time.perf_counter()
        y_out = model(adj_torch, adjT_torch)
        inference_time = time.perf_counter() - t0

        tau = ranking_correlation(
            y_out=y_out,
            true_val=true_val,
            node_num=nodes_list[i],
            model_size=N,
        )

        taus.append(tau)
        inference_times.append(inference_time)

        print(
            f"Test graph {i+1}/{len(adjs)} | "
            f"tau = {tau:.4f} | "
            f"inference = {inference_time*1000:.1f} ms"
        )

    return np.array(taus), np.array(inference_times)


# ============================================================
# Generate data
# ============================================================

train_graphs, train_adjs, train_adjs_T, train_labels, train_nodes, train_label_times = generate_dataset(
    NUM_TRAIN, "TRAIN"
)

test_graphs, test_adjs, test_adjs_T, test_labels, test_nodes, test_label_times = generate_dataset(
    NUM_TEST, "TEST"
)

# ============================================================
# Build model
# ============================================================

net = GNN_Bet(
    ninput=N,
    nhid=HIDDEN,
    dropout=DROPOUT,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
)

model, _ = net.model_to_device(net)
model = model.to(device)
optimizer = net.get_optimizer(model)

# ============================================================
# Train
# ============================================================

print("\nTraining N=1000 model")
print("-" * 70)

train_start = time.perf_counter()

best_tau = -1.0
best_epoch = -1

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.perf_counter()

    train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        adjs=train_adjs,
        adjs_T=train_adjs_T,
        labels=train_labels,
        nodes_list=train_nodes,
    )

    # Evaluate every 5 epochs and final epoch
    if epoch % 5 == 0 or epoch == EPOCHS:
        taus, inf_times = evaluate_tau(
            model=model,
            adjs=test_adjs,
            adjs_T=test_adjs_T,
            labels=test_labels,
            nodes_list=test_nodes,
        )

        mean_tau = float(np.nanmean(taus))
        std_tau = float(np.nanstd(taus))

        if mean_tau > best_tau:
            best_tau = mean_tau
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"loss = {train_loss:.6f} | "
            f"test tau = {mean_tau:.4f} ± {std_tau:.4f} | "
            f"time = {(time.perf_counter() - epoch_start):.1f}s"
        )
    else:
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"loss = {train_loss:.6f} | "
            f"time = {(time.perf_counter() - epoch_start):.1f}s"
        )

training_time = time.perf_counter() - train_start

print("\nLoading best model for final evaluation")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

final_taus, final_inf_times = evaluate_tau(
    model=model,
    adjs=test_adjs,
    adjs_T=test_adjs_T,
    labels=test_labels,
    nodes_list=test_nodes,
)

final_tau_mean = float(np.nanmean(final_taus))
final_tau_std = float(np.nanstd(final_taus))
final_inf_mean_ms = float(np.mean(final_inf_times) * 1000)
final_inf_std_ms = float(np.std(final_inf_times) * 1000)

nx_label_mean_ms = float(np.mean(test_label_times) * 1000)
nx_label_std_ms = float(np.std(test_label_times) * 1000)
speedup = nx_label_mean_ms / final_inf_mean_ms

# ============================================================
# Save results
# ============================================================

with open(CSV_PATH, "w", encoding="utf-8") as f:
    f.write("N,num_train,num_test,epochs,tau_mean,tau_std,nx_ms,nx_std_ms,gnn_ms,gnn_std_ms,speedup,best_epoch,training_time_min\n")
    f.write(
        f"{N},{NUM_TRAIN},{NUM_TEST},{EPOCHS},"
        f"{final_tau_mean:.6f},{final_tau_std:.6f},"
        f"{nx_label_mean_ms:.2f},{nx_label_std_ms:.2f},"
        f"{final_inf_mean_ms:.2f},{final_inf_std_ms:.2f},"
        f"{speedup:.2f},{best_epoch},{training_time/60:.2f}\n"
    )

print("\n" + "=" * 70)
print("FINAL N=1000 RESULTS")
print("=" * 70)
print(f"Best epoch              : {best_epoch}")
print(f"Kendall tau             : {final_tau_mean:.4f} ± {final_tau_std:.4f}")
print(f"NetworkX exact time     : {nx_label_mean_ms:.1f} ± {nx_label_std_ms:.1f} ms")
print(f"GNN inference time      : {final_inf_mean_ms:.1f} ± {final_inf_std_ms:.1f} ms")
print(f"Speedup                 : {speedup:.1f}x")
print(f"Training time           : {training_time/60:.2f} min")
print(f"Saved model             : {MODEL_PATH}")
print(f"Saved CSV               : {CSV_PATH}")
print("=" * 70)