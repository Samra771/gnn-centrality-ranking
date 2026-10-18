"""
experiment7_ablation_betweenness.py

Purpose:
Compact ablation study for the betweenness GNN.

This version runs the depth ablation:
- Dual pathway + sum, L=3
- Dual pathway + sum, L=5
- Dual pathway + sum, L=7

It fixes:
1. Indentation error in train_one_epoch()
2. Indentation error in evaluate()
3. NaN issue caused by skipping all tau evaluations
4. Uses stable settings: epochs=30, lr=1e-3, batch_size=8
"""

import os
import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GNN_Layer, GNN_Layer_Init, MLP
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation,
)

# ============================================================
# Settings
# ============================================================

SEED = 42
N = 200
P = 0.15

NUM_TRAIN = 500
NUM_TEST_ER = 100
NUM_TEST_GRP = 100

EPOCHS = 30
HIDDEN = 40
DROPOUT = 0.4
LR = 1e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8

OUT_DIR = "paper_figures_ablation"
CSV_PATH = os.path.join(OUT_DIR, "ablation_depth_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BETWEENNESS GNN DEPTH ABLATION STUDY")
print("=" * 70)
print(f"Device: {device}")
print(f"N={N}, p={P}")
print(f"Train graphs: {NUM_TRAIN}")
print(f"ER test graphs: {NUM_TEST_ER}")
print(f"GRP test graphs: {NUM_TEST_GRP}")
print(f"Epochs: {EPOCHS}")
print("=" * 70)


# ============================================================
# Flexible ablation model
# ============================================================

class GNNBetAblation(nn.Module):
    def __init__(
        self,
        ninput,
        nhid,
        dropout,
        num_layers=5,
        dual_pathway=True,
        fusion="sum",
        learning_rate=1e-3,
        weight_decay=0.01,
    ):
        super().__init__()

        assert num_layers in [3, 5, 7], "Use num_layers in {3, 5, 7}"
        assert fusion in ["product", "sum", "concat"], "fusion must be product, sum, or concat"

        self.ninput = ninput
        self.nhid = nhid
        self.dropout = dropout
        self.num_layers = num_layers
        self.dual_pathway = dual_pathway
        self.fusion = fusion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.layers = nn.ModuleList()
        self.layers.append(GNN_Layer_Init(ninput, nhid))

        for _ in range(num_layers - 1):
            self.layers.append(GNN_Layer(nhid, nhid))

        self.score_layer = MLP(nhid, dropout)

        if fusion == "concat" and dual_pathway:
            self.concat_mlp = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        else:
            self.concat_mlp = None

    def pathway_forward(self, adj):
        layer_scores = []

        x = self.layers[0](adj)
        x = F.relu(x)
        x = F.normalize(x, p=2, dim=1)

        layer_scores.append(self.score_layer(x, self.dropout))

        for k in range(1, self.num_layers):
            x = self.layers[k](x, adj)
            x = F.relu(x)
            x = F.normalize(x, p=2, dim=1)

            layer_scores.append(self.score_layer(x, self.dropout))

        score = torch.stack(layer_scores, dim=0).sum(dim=0)
        return score

    def forward(self, adj1, adj2):
        score1 = self.pathway_forward(adj1)

        if not self.dual_pathway:
            return score1

        score2 = self.pathway_forward(adj2)

        if self.fusion == "product":
            return torch.mul(score1, score2)

        if self.fusion == "sum":
            return score1 + score2

        if self.fusion == "concat":
            x = torch.cat([score1, score2], dim=1)
            return self.concat_mlp(x)

        raise ValueError("Unknown fusion type")

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


# ============================================================
# Graph generation utilities
# ============================================================

def make_er_graph():
    g = nx.erdos_renyi_graph(N, P, directed=True)
    return nx.convert_node_labels_to_integers(g)


def make_grp_graph():
    """
    GRP graph for topology-shift evaluation.

    NetworkX gaussian_random_partition_graph is undirected.
    We convert it to a directed graph with reciprocal edges
    for compatibility with the directed betweenness model.
    """
    g = nx.gaussian_random_partition_graph(
        n=N,
        s=20,
        v=5,
        p_in=0.3,
        p_out=0.05,
        directed=False,
    )

    g = nx.convert_node_labels_to_integers(g)
    g = g.to_directed()

    return g


def get_adj(graph, model_size=N):
    adj = nx.adjacency_matrix(
        graph,
        nodelist=list(range(graph.number_of_nodes()))
    ).astype("float32")

    if graph.number_of_nodes() < model_size:
        pad = csr_matrix(
            (model_size - graph.number_of_nodes(),
             model_size - graph.number_of_nodes())
        )
        adj = sp.block_diag([adj, pad])

    return adj.tocsr()


def exact_betweenness(graph):
    return nx.betweenness_centrality(graph, normalized=True)


def generate_dataset(num_graphs, graph_type):
    adjs = []
    adjs_T = []
    labels = []
    nodes_list = []

    print(f"\nGenerating {num_graphs} {graph_type} graphs")

    for i in range(num_graphs):
        if graph_type == "ER":
            g = make_er_graph()
        elif graph_type == "GRP":
            g = make_grp_graph()
        else:
            raise ValueError("graph_type must be ER or GRP")

        adj = get_adj(g)

        adjs.append(adj)
        adjs_T.append(adj.T.tocsr())
        labels.append(exact_betweenness(g))
        nodes_list.append(g.number_of_nodes())

        if (i + 1) % 10 == 0:
            print(f"{graph_type}: {i + 1}/{num_graphs}")

    return adjs, adjs_T, labels, nodes_list


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(model, optimizer, train_adjs, train_adjs_T, train_labels, train_nodes):
    model.train()

    indices = np.arange(len(train_adjs))
    np.random.shuffle(indices)

    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]

        optimizer.zero_grad()
        batch_loss = 0.0
        valid_batch = True

        for idx in batch_idx:
            adj = sparse_mx_to_torch_sparse_tensor(train_adjs[idx], device)
            adj_T = sparse_mx_to_torch_sparse_tensor(train_adjs_T[idx], device)

            true_arr = np.array(
                list(train_labels[idx].values()),
                dtype=np.float32
            )
            true_val = torch.from_numpy(true_arr).float().to(device)

            y_out = model(adj, adj_T)

            if not torch.isfinite(y_out).all():
                valid_batch = False
                break

            loss = loss_cal(
                y_out=y_out,
                true_val=true_val,
                num_nodes=train_nodes[idx],
                device=device,
                model_size=N,
            )

            if not torch.isfinite(loss):
                valid_batch = False
                break

            batch_loss += loss

        if not valid_batch:
            print("Warning: non-finite batch skipped")
            continue

        batch_loss = batch_loss / len(batch_idx)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(batch_loss.detach().cpu())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, adjs, adjs_T, labels, nodes_list):
    model.eval()

    taus = []

    for i in range(len(adjs)):
        adj = sparse_mx_to_torch_sparse_tensor(adjs[i], device)
        adj_T = sparse_mx_to_torch_sparse_tensor(adjs_T[i], device)

        true_arr = np.array(
            list(labels[i].values()),
            dtype=np.float32
        )
        true_val = torch.from_numpy(true_arr).float().to(device)

        y_out = model(adj, adj_T)

        if not torch.isfinite(y_out).all():
            print(f"Warning: non-finite predictions on graph {i}")
            taus.append(np.nan)
            continue

        pred_np = y_out[:nodes_list[i]].detach().cpu().numpy().reshape(-1)

        if not np.isfinite(pred_np).all():
            print(f"Warning: prediction contains NaN or Inf on graph {i}")
            taus.append(np.nan)
            continue

        if np.std(pred_np) < 1e-12:
            print(f"Warning: constant predictions on graph {i}")
            taus.append(np.nan)
            continue

        tau = ranking_correlation(
            y_out=y_out,
            true_val=true_val,
            node_num=nodes_list[i],
            model_size=N,
        )

        taus.append(tau)

    valid_taus = [t for t in taus if np.isfinite(t)]

    if len(valid_taus) == 0:
        return np.nan, np.nan

    return float(np.mean(valid_taus)), float(np.std(valid_taus))


def run_variant(name, dual_pathway, fusion, num_layers, train_data, er_data, grp_data):
    train_adjs, train_adjs_T, train_labels, train_nodes = train_data
    er_adjs, er_adjs_T, er_labels, er_nodes = er_data
    grp_adjs, grp_adjs_T, grp_labels, grp_nodes = grp_data

    print("\n" + "=" * 70)
    print(f"Running variant: {name}")
    print("=" * 70)

    model = GNNBetAblation(
        ninput=N,
        nhid=HIDDEN,
        dropout=DROPOUT,
        num_layers=num_layers,
        dual_pathway=dual_pathway,
        fusion=fusion,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
    ).to(device)

    optimizer = model.get_optimizer()

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_adjs=train_adjs,
            train_adjs_T=train_adjs_T,
            train_labels=train_labels,
            train_nodes=train_nodes,
        )

        if epoch % 10 == 0 or epoch == EPOCHS:
            er_mean, er_std = evaluate(
                model,
                er_adjs,
                er_adjs_T,
                er_labels,
                er_nodes,
            )

            grp_mean, grp_std = evaluate(
                model,
                grp_adjs,
                grp_adjs_T,
                grp_labels,
                grp_nodes,
            )

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"loss={loss:.5f} | "
                f"ER tau={er_mean:.4f} | "
                f"GRP tau={grp_mean:.4f}"
            )

    total_time = time.perf_counter() - start_time

    er_mean, er_std = evaluate(
        model,
        er_adjs,
        er_adjs_T,
        er_labels,
        er_nodes,
    )

    grp_mean, grp_std = evaluate(
        model,
        grp_adjs,
        grp_adjs_T,
        grp_labels,
        grp_nodes,
    )

    return {
        "variant": name,
        "dual_pathway": dual_pathway,
        "fusion": fusion,
        "layers": num_layers,
        "er_tau_mean": er_mean,
        "er_tau_std": er_std,
        "grp_tau_mean": grp_mean,
        "grp_tau_std": grp_std,
        "training_time_min": total_time / 60.0,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_data = generate_dataset(NUM_TRAIN, "ER")
    er_test_data = generate_dataset(NUM_TEST_ER, "ER")
    grp_test_data = generate_dataset(NUM_TEST_GRP, "GRP")

    variants = [
        {
            "name": "Dual pathway + sum, L=3",
            "dual_pathway": True,
            "fusion": "sum",
            "num_layers": 3,
        },
        {
            "name": "Dual pathway + sum, L=5",
            "dual_pathway": True,
            "fusion": "sum",
            "num_layers": 5,
        },
        {
            "name": "Dual pathway + sum, L=7",
            "dual_pathway": True,
            "fusion": "sum",
            "num_layers": 7,
        },
    ]

    results = []

    for v in variants:
        res = run_variant(
            name=v["name"],
            dual_pathway=v["dual_pathway"],
            fusion=v["fusion"],
            num_layers=v["num_layers"],
            train_data=train_data,
            er_data=er_test_data,
            grp_data=grp_test_data,
        )

        results.append(res)

        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)

        print("\nCurrent results:")
        print(df)

    print("\n" + "=" * 70)
    print("FINAL DEPTH ABLATION RESULTS")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(CSV_PATH, index=False)

    print(f"\nSaved results to: {CSV_PATH}")