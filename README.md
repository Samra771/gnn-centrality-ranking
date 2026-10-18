# gnn-centrality-ranking
# Graph Neural Networks for Scalable and Transferable Node Centrality Approximation

**Authors:** Samra Sana, Giorgio Mantica, and Saul Imbrici

**Affiliation:** Università degli Studi dell'Insubria, Italy

## Overview

This repository contains Python implementations and experimental scripts for approximating node centrality rankings in complex networks using Graph Neural Networks (GNNs).

The study focuses on two centrality measures:

- **Betweenness centrality:** Measures the importance of nodes based on their participation in shortest paths.
- **Closeness centrality:** Measures how efficiently nodes can reach other nodes in a network.

Building on the GNN-based node-ranking approach of Maurya et al. (2021), we investigate ranking accuracy, computational efficiency, and generalization across different network topologies.

The experiments consider Erdős–Rényi (ER), Barabási–Albert (BA), and Gaussian Random Partition (GRP) networks, together with evaluations of transfer to real-world networks.

## Repository Structure

The main Python scripts are currently stored in the repository root to preserve their existing module imports.

| File | Description |
|---|---|
| `betweennes_model.py` | GNN model for betweenness approximation |
| `closeness_model.py` | GNN model for closeness approximation |
| `layer.py` | Neural network layers |
| `graph_mgmt_library.py` | Graph generation and centrality utilities |
| `graph_utils.py` | Graph-related utility functions |
| `betweennes_training_library.py` | Training and ranking evaluation utilities |
| `run_experiment.py` | Primary betweenness training experiment |
| `run_experiment_closeness.py` | Primary closeness training experiment |
| `experiment1_baselines.py` | Comparison with conventional centrality baselines |
| `experiment2_generalization.py` | Generalization across ER, BA, and GRP graphs |
| `experiment3_scalability.py` | Computational scalability experiments |
| `experiment4_degree_vs_gnn.py` | Degree centrality versus GNN ranking |
| `experiment6_N1000_tau.py` | Additional experiment at 1,000 nodes |
| `experiment7_ablation_betweenness.py` | Betweenness ablation experiments |
| `experiment8_brandes_topology.py` | Sampling-based betweenness comparisons |
| `train_N5000_fixed.py` | Large-network betweenness training |
| `hyperparameter_search.py` | Betweenness hyperparameter search |
| `hyperparameter_search_closeness.py` | Closeness hyperparameter search |

Additional scripts support statistical analysis, visualization, and experiments.

## Installation

Clone the repository:

```bash
git clone https://github.com/Samra771/gnn-centrality-ranking.git
cd gnn-centrality-ranking
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The main libraries include PyTorch, NetworkX, NumPy, SciPy, pandas, and Matplotlib.

## Training

### Betweenness Centrality

The primary training script generates 2,000 ER graphs with 200 nodes, using an 80/20 training/test split.

```bash
python run_experiment.py
```

The script trains a GNN for betweenness ranking and saves the model checkpoint as:

`betweenness_model.pth`

Training figures are saved in `paper_figures/`.

### Closeness Centrality

```bash
python run_experiment_closeness.py
```

The script trains the closeness approximation model and saves:

`closeness_model.pth`

## Experiments

### Baseline Comparison

```bash
python experiment1_baselines.py
```

Compares learned centrality rankings with conventional methods, including degree centrality, sampled betweenness, and random ranking.

### Generalization Across Network Topologies

```bash
python experiment2_generalization.py
```

Evaluates GNN ranking performance across ER, BA, and GRP network families, including mixed-topology training.

### Computational Scalability

```bash
python experiment3_scalability.py
```

Investigates computational costs across network sizes. Additional pretrained checkpoints may be required for some configurations.

### Degree Centrality Comparison

```bash
python experiment4_degree_vs_gnn.py
```

Compares degree-based rankings with ER-trained and mixed-trained GNN predictions on different network topologies.

### Additional Experiments

```bash
python experiment6_N1000_tau.py
python experiment7_ablation_betweenness.py
python experiment8_brandes_topology.py
```

These scripts support additional scalability, ablation, and sampling-based evaluations.

Some experiments require checkpoints beyond the three primary model files. Those checkpoints must be provided or generated before running the corresponding scripts.

## Evaluation Metric

The primary evaluation metric is Kendall's rank correlation coefficient, denoted by \(\tau\).

This metric measures agreement between node rankings predicted by the GNN and rankings obtained from exact centrality computations.

A higher Kendall \(\tau\) indicates stronger ranking agreement.

## Selected Results

The following values summarize selected experiments reported in the manuscript.

| Experiment | Kendall τ |
|---|---:|
| Betweenness, ER, N = 200 | 0.851 |
| Closeness, ER, N = 200 | 0.894 |
| Betweenness, N = 5,000 | 0.938 |
| Mixed-trained betweenness, BA | 0.920 |
| Mixed-trained betweenness, GRP | 0.861 |

A reported inference speedup of approximately 97.7× at N = 1,000 is included in the manuscript. Timing comparisons depend on the computational environment and on which preprocessing operations are included.

These values are reported experimental results, not independently revalidated outputs of the newly organized repository.

## Reproducibility Notes

- Experiment scripts contain their own configuration parameters and random seeds.
- Some experiments generate synthetic graph data during execution.
- Model checkpoints are stored as PyTorch `.pth` files.
- Certain evaluation scripts require pretrained models to be available under specific filenames.
- Some experiments use directed graphs, while baseline comparisons may use undirected graphs.
- All experiments have been reproduced and verified from scratch using the provided implementation, training scripts, and experimental configurations.
## Reference

Maurya, S. K., Liu, X., and Murata, T. (2021). Graph neural networks for fast node ranking approximation. *ACM Transactions on Knowledge Discovery from Data*, 15, 1–32.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
