

import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau

# ============================================================
# Reproducibility
# ============================================================

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# Settings
# ============================================================

N = 200
P = 0.15
NUM_TEST_GRAPHS = 100   # use 50 if you are short on time
BRANDES_K_VALUES = [10, 20, 50]

OUTPUT_DIR = "./paper_figures_baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "brandes_topology_baseline.csv")

print("=" * 70)
print("BRANDES BASELINE ACROSS GRAPH TOPOLOGIES")
print("=" * 70)
print(f"N = {N}")
print(f"p = {P}")
print(f"Test graphs per topology = {NUM_TEST_GRAPHS}")
print(f"Brandes k values = {BRANDES_K_VALUES}")
print("=" * 70)


# ============================================================
# Graph generators
# ============================================================

def make_er_graph(seed=None):
    """
    Directed ER graph, consistent with the betweenness experiments.
    """
    g = nx.erdos_renyi_graph(N, P, directed=True, seed=seed)
    return nx.convert_node_labels_to_integers(g)


def make_ba_graph(seed=None):
    """
    BA graph converted to directed graph with reciprocal edges.
    """
    m = max(1, int(P * N / 2))
    g = nx.barabasi_albert_graph(N, m, seed=seed)
    g = nx.convert_node_labels_to_integers(g)
    return g.to_directed()


def make_grp_graph(seed=None):
    """
    GRP graph converted to directed graph with reciprocal edges.
    """
    g = nx.gaussian_random_partition_graph(
        n=N,
        s=20,
        v=5,
        p_in=0.3,
        p_out=0.05,
        directed=False,
        seed=seed,
    )
    g = nx.convert_node_labels_to_integers(g)
    return g.to_directed()


GRAPH_GENERATORS = {
    "ER": make_er_graph,
    "BA": make_ba_graph,
    "GRP": make_grp_graph,
}


# ============================================================
# Utility functions
# ============================================================

def centrality_to_array(cent_dict, n):
    arr = np.zeros(n, dtype=np.float32)
    for node, val in cent_dict.items():
        if 0 <= node < n:
            arr[node] = float(val)
    return arr


def kendall_tau(pred, truth):
    tau, _ = kendalltau(pred, truth)
    return float(tau) if np.isfinite(tau) else np.nan


def evaluate_one_graph(graph, graph_type, graph_id):
    """
    Evaluate random, degree, and Brandes baselines on one graph.
    """
    n = graph.number_of_nodes()

    rows = []

    # Exact betweenness ground truth
    t0 = time.perf_counter()
    exact_dict = nx.betweenness_centrality(graph, normalized=True)
    exact_time = (time.perf_counter() - t0) * 1000.0
    exact_arr = centrality_to_array(exact_dict, n)

    # Random ranking
    random_scores = np.random.rand(n)
    rows.append({
        "graph_type": graph_type,
        "graph_id": graph_id,
        "method": "Random Ranking",
        "k": "",
        "tau": kendall_tau(random_scores, exact_arr),
        "time_ms": 0.0,
        "exact_time_ms": exact_time,
    })

    # Degree centrality
    t0 = time.perf_counter()
    degree_dict = nx.degree_centrality(graph)
    degree_time = (time.perf_counter() - t0) * 1000.0
    degree_arr = centrality_to_array(degree_dict, n)

    rows.append({
        "graph_type": graph_type,
        "graph_id": graph_id,
        "method": "Degree Centrality",
        "k": "",
        "tau": kendall_tau(degree_arr, exact_arr),
        "time_ms": degree_time,
        "exact_time_ms": exact_time,
    })

    # Brandes approximation
    for k in BRANDES_K_VALUES:
        t0 = time.perf_counter()
        approx_dict = nx.betweenness_centrality(
            graph,
            k=k,
            normalized=True,
            seed=SEED + graph_id,
        )
        approx_time = (time.perf_counter() - t0) * 1000.0
        approx_arr = centrality_to_array(approx_dict, n)

        rows.append({
            "graph_type": graph_type,
            "graph_id": graph_id,
            "method": f"Brandes k={k}",
            "k": k,
            "tau": kendall_tau(approx_arr, exact_arr),
            "time_ms": approx_time,
            "exact_time_ms": exact_time,
        })

    return rows


# ============================================================
# Run experiment
# ============================================================

all_rows = []

for graph_type, generator in GRAPH_GENERATORS.items():
    print(f"\nEvaluating {graph_type} graphs")

    for i in range(NUM_TEST_GRAPHS):
        graph = generator(seed=SEED + i)
        rows = evaluate_one_graph(graph, graph_type, i)
        all_rows.extend(rows)

        if (i + 1) % 10 == 0:
            print(f"  {graph_type}: {i + 1}/{NUM_TEST_GRAPHS}")

raw_df = pd.DataFrame(all_rows)

# Save raw results
raw_csv = os.path.join(OUTPUT_DIR, "brandes_topology_baseline_raw.csv")
raw_df.to_csv(raw_csv, index=False)

# ============================================================
# Summary table
# ============================================================

summary_rows = []

for graph_type in ["ER", "BA", "GRP"]:
    for method in ["Random Ranking", "Degree Centrality", "Brandes k=10", "Brandes k=20", "Brandes k=50"]:
        subset = raw_df[
            (raw_df["graph_type"] == graph_type) &
            (raw_df["method"] == method)
        ]

        tau_values = subset["tau"].dropna().values
        time_values = subset["time_ms"].dropna().values

        summary_rows.append({
            "Graph": graph_type,
            "Method": method,
            "Tau mean": np.mean(tau_values),
            "Tau std": np.std(tau_values),
            "Time mean ms": np.mean(time_values),
            "Time std ms": np.std(time_values),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(CSV_PATH, index=False)

print("\n" + "=" * 70)
print("SUMMARY RESULTS")
print("=" * 70)

for graph_type in ["ER", "BA", "GRP"]:
    print(f"\n{graph_type}")
    sub = summary_df[summary_df["Graph"] == graph_type]
    for _, row in sub.iterrows():
        print(
            f"  {row['Method']:<18} "
            f"tau = {row['Tau mean']:.4f} ± {row['Tau std']:.4f} | "
            f"time = {row['Time mean ms']:.2f} ± {row['Time std ms']:.2f} ms"
        )

print("\nSaved raw results to:")
print(raw_csv)

print("\nSaved summary results to:")
print(CSV_PATH)
print("=" * 70)
