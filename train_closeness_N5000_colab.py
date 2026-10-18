"""
train_closeness_N5000_colab.py
================================
Trains closeness GNN at N=5000 nodes on Google Colab A100 GPU.
Uses EXACT closeness labels — no approximation needed.
(Closeness is O(N^2), much faster than betweenness O(N^3))

HOW TO USE:
1. Runtime → Change runtime type → A100 GPU
2. Upload these files:
      closeness_model.py
      betweennes_training_library.py
      layer.py
      train_closeness_N5000_colab.py
3. Run: !python train_closeness_N5000_colab.py

Expected runtime on A100:
  Label generation : ~30 minutes (exact, fast for closeness)
  Training         : ~5 minutes
  Total            : ~35 minutes
"""

import os
import sys
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import json
from scipy.sparse import csr_matrix
import scipy.sparse as sp

# ── Check GPU ─────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("ERROR: No GPU.")
    print("Go to Runtime → Change runtime type → A100 GPU")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU    : {gpu_name}")
print(f"Memory : {gpu_mem:.1f} GB\n")

from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation
)

# ═════════════════════════════════════════════════════════════════════
# SETTINGS — same hyperparameters from grid search
# ═════════════════════════════════════════════════════════════════════
SEED         = 42
GRAPH_NODES  = 5000
GRAPH_P      = 0.001     # sparse — avg degree ~5
TOTAL_GRAPHS = 300
TRAIN_RATIO  = 0.70      # 210 train
VAL_RATIO    = 0.10      # 30 val
TEST_RATIO   = 0.20      # 60 test
HIDDEN       = 40        # from grid search
EPOCHS       = 50
LR           = 1e-3      # from grid search
DROPOUT      = 0.2       # from grid search
WD           = 0.0       # from grid search
BATCH_SIZE   = 2
MODEL_SIZE   = GRAPH_NODES

OUTPUT_DIR     = "./results_closeness_N5000"
MODEL_PATH     = "./results_closeness_N5000/closeness_model_N5000.pth"
CHECKPOINT_DIR = "./results_closeness_N5000/checkpoints"

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda")

print("="*55)
print("  GNN Closeness — N=5000 Training")
print("="*55)
print(f"  N={GRAPH_NODES}, p={GRAPH_P}")
print(f"  Labels  : EXACT closeness (no approximation)")
print(f"  Graphs  : {TOTAL_GRAPHS} undirected ER")
print(f"  Split   : {int(TOTAL_GRAPHS*TRAIN_RATIO)} train / "
      f"{int(TOTAL_GRAPHS*VAL_RATIO)} val / "
      f"{int(TOTAL_GRAPHS*TEST_RATIO)} test")
print(f"  Epochs  : {EPOCHS}")
print(f"  LR={LR}, dropout={DROPOUT}, "
      f"hidden={HIDDEN}, wd={WD}")
print("="*55 + "\n")

# ═════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════
def make_graph(n, p):
    """Undirected ER graph — largest connected component."""
    g   = nx.erdos_renyi_graph(n, p, directed=False)
    gcc = max(nx.connected_components(g), key=len)
    g   = g.subgraph(gcc).copy()
    return nx.convert_node_labels_to_integers(g)

def get_adj(g, ms):
    """Adjacency matrix padded to ms x ms."""
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    if n < ms:
        adj = sp.block_diag([adj, csr_matrix((ms-n, ms-n))])
    return adj.tocsr()

def get_adj_mod(g, ms):
    """
    Degree-normalised adjacency A_mod = D^{-1} A.
    Used as second input to closeness model — prevents zero
    feature vectors that cause NaN during training.
    """
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    adj_mod = sp.diags(1.0 / deg) @ adj
    if n < ms:
        adj_mod = sp.block_diag(
                      [adj_mod, csr_matrix((ms-n, ms-n))])
    return adj_mod.tocsr()

def to_padded_array(d, n_real, n_padded):
    """
    Creates array of size n_padded.
    Fills first n_real entries with centrality values.
    Pads remainder with zeros.
    This prevents shape mismatch in loss_cal.
    """
    arr = np.zeros(n_padded, dtype=np.float32)
    for node, val in d.items():
        if 0 <= node < n_real:
            arr[node] = float(val)
    return arr

def get_split(indices, adjs, adj_mods, cents, ns):
    return ([adjs[i]     for i in indices],
            [adj_mods[i] for i in indices],
            [cents[i]    for i in indices],
            [ns[i]       for i in indices])

# ═════════════════════════════════════════════════════════════════════
# GENERATE DATASET — EXACT CLOSENESS LABELS
# ═════════════════════════════════════════════════════════════════════
print("Generating graphs with EXACT closeness labels...\n")

adjs, adj_mods, cents, ns = [], [], [], []
t0 = time.time()

for i in range(TOTAL_GRAPHS):
    g  = make_graph(GRAPH_NODES, GRAPH_P)
    n  = g.number_of_nodes()

    # Exact closeness — much faster than betweenness
    cc = nx.closeness_centrality(g)

    adjs.append(get_adj(g, MODEL_SIZE))
    adj_mods.append(get_adj_mod(g, MODEL_SIZE))
    # FIX: pad to MODEL_SIZE to prevent shape mismatch
    cents.append(to_padded_array(cc, n, MODEL_SIZE))
    ns.append(n)

    if (i+1) % 10 == 0:
        elapsed   = time.time() - t0
        per_graph = elapsed / (i+1)
        remaining = per_graph * (TOTAL_GRAPHS - i - 1)
        print(f"  [{i+1:3d}/{TOTAL_GRAPHS}] "
              f"{per_graph:.1f}s/graph | "
              f"~{remaining/60:.0f}min remaining")

print(f"\n  Done in {(time.time()-t0)/60:.1f} minutes\n")

# ── Split ─────────────────────────────────────────────────────────────
idx     = list(range(TOTAL_GRAPHS))
random.shuffle(idx)
n_train = int(TOTAL_GRAPHS * TRAIN_RATIO)
n_val   = int(TOTAL_GRAPHS * VAL_RATIO)

tr_a, tr_m, tr_c, tr_n = get_split(
    idx[:n_train], adjs, adj_mods, cents, ns)
va_a, va_m, va_c, va_n = get_split(
    idx[n_train:n_train+n_val], adjs, adj_mods, cents, ns)
te_a, te_m, te_c, te_n = get_split(
    idx[n_train+n_val:], adjs, adj_mods, cents, ns)

print(f"  Train: {len(tr_a)} | "
      f"Val: {len(va_a)} | "
      f"Test: {len(te_a)}\n")

# ═════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ═════════════════════════════════════════════════════════════════════
print("Building model...")
net      = GNN_Close(
    ninput       = MODEL_SIZE,
    nhid         = HIDDEN,
    dropout      = DROPOUT,
    learning_rate= LR,
    weight_decay = WD
)
model, _ = net.model_to_device(net)
model    = model.to(device)
optimizer= net.get_optimizer(model)

params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {params:,}")
print(f"  Device     : {next(model.parameters()).device}\n")

# ═════════════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════
def train_epoch(adj_list, adj_mod_list, cent_list, node_list):
    model.train()
    total = 0.0
    count = 0
    perm  = np.random.permutation(len(adj_list))

    for start in range(0, len(adj_list), BATCH_SIZE):
        batch = perm[start:start+BATCH_SIZE]
        optimizer.zero_grad()
        bloss = torch.tensor(
                    0.0, device=device, requires_grad=True)
        cnt   = 0
        for i in batch:
            a  = sparse_mx_to_torch_sparse_tensor(
                     adj_list[i],     device)
            am = sparse_mx_to_torch_sparse_tensor(
                     adj_mod_list[i], device)
            y  = model(a, am)
            if torch.isnan(y).any():
                continue
            tv   = torch.from_numpy(
                       cent_list[i]).float().to(device)
            loss = loss_cal(y, tv, node_list[i],
                            device, MODEL_SIZE)
            bloss = bloss + loss
            cnt  += 1
        if cnt > 0:
            bloss = bloss / cnt
            bloss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            total += float(bloss.detach())
            count += 1
        torch.cuda.empty_cache()

    return total / max(1, count)


def evaluate(adj_list, adj_mod_list, cent_list, node_list):
    model.eval()
    kts = []
    with torch.no_grad():
        for i in range(len(adj_list)):
            a  = sparse_mx_to_torch_sparse_tensor(
                     adj_list[i],     device)
            am = sparse_mx_to_torch_sparse_tensor(
                     adj_mod_list[i], device)
            y  = model(a, am)
            if torch.isnan(y).any():
                continue
            tv = torch.from_numpy(
                     cent_list[i]).float().to(device)
            kt = ranking_correlation(
                     y, tv, node_list[i], MODEL_SIZE)
            if not np.isnan(kt):
                kts.append(kt)
            torch.cuda.empty_cache()

    mean = float(np.mean(kts)) if kts else 0.0
    std  = float(np.std(kts))  if kts else 0.0
    return mean, std

# ═════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════
print("Training...\n")

train_losses = []
val_kts      = []
val_stds     = []
best_val_kt  = -1.0
t0           = time.time()

for epoch in range(EPOCHS):
    loss          = train_epoch(tr_a, tr_m, tr_c, tr_n)
    val_kt, val_s = evaluate(va_a, va_m, va_c, va_n)

    train_losses.append(loss)
    val_kts.append(val_kt)
    val_stds.append(val_s)

    if val_kt > best_val_kt:
        best_val_kt = val_kt
        torch.save(model.state_dict(), MODEL_PATH)

    # Checkpoint every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save({
            "epoch"        : epoch+1,
            "model_state"  : model.state_dict(),
            "best_val_kt"  : best_val_kt,
            "train_losses" : train_losses,
            "val_kts"      : val_kts,
        }, f"{CHECKPOINT_DIR}/epoch{epoch+1}.pth")

    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Loss: {loss:.4f} | "
          f"Val KT: {val_kt:.4f} +/- {val_s:.4f} | "
          f"Best: {best_val_kt:.4f} | "
          f"Time: {elapsed/60:.0f}min")

print(f"\n  Done in {(time.time()-t0)/60:.1f} minutes")
print(f"  Best model saved: {MODEL_PATH}\n")

# ═════════════════════════════════════════════════════════════════════
# FINAL TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════
print("Final test evaluation...")
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device))
test_kt, test_std = evaluate(te_a, te_m, te_c, te_n)
print(f"  Test KT : {test_kt:.4f} +/- {test_std:.4f}\n")

# ═════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig.subplots_adjust(wspace=0.38)

ax1.plot(range(1, EPOCHS+1), train_losses,
         color="#1565C0", lw=1.4)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("(a) Training Loss — Closeness N=5,000")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS+1), val_kts,
         color="#6A1B9A", lw=1.4,
         label="Validation KT")
ax2.fill_between(
    range(1, EPOCHS+1),
    [m-s for m, s in zip(val_kts, val_stds)],
    [m+s for m, s in zip(val_kts, val_stds)],
    alpha=0.18, color="#6A1B9A")
ax2.axhline(best_val_kt, color="#2E7D32", lw=1.0,
            linestyle="--",
            label=f"Best = {best_val_kt:.3f}")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Kendall τ")
ax2.set_title("(b) Validation Kendall τ — Closeness N=5,000")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(
    f"GNN Closeness N=5,000  |  "
    f"Exact Labels  |  {gpu_name}",
    fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_closeness_N5000_training.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# SAVE RESULTS SUMMARY
# ═════════════════════════════════════════════════════════════════════
with open(f"{OUTPUT_DIR}/results_closeness_N5000_summary.txt",
          "w") as f:
    f.write("GNN Closeness — N=5000 Results\n")
    f.write("="*45 + "\n\n")
    f.write(f"N={GRAPH_NODES}, p={GRAPH_P}\n")
    f.write(f"Labels: EXACT closeness\n")
    f.write(f"Train: {len(tr_a)} | "
            f"Val: {len(va_a)} | "
            f"Test: {len(te_a)}\n\n")
    f.write(f"Hyperparameters:\n")
    f.write(f"  LR={LR}, dropout={DROPOUT}, "
            f"hidden={HIDDEN}, wd={WD}\n\n")
    f.write(f"Best val KT : {best_val_kt:.6f}\n")
    f.write(f"Test KT     : "
            f"{test_kt:.6f} +/- {test_std:.6f}\n")
    f.write(f"GPU: {gpu_name}\n")

# ── Final print ───────────────────────────────────────────────────────
print("="*55)
print("  COMPLETE")
print("="*55)
print(f"  Test KT : {test_kt:.4f} +/- {test_std:.4f}")
print(f"  Labels  : EXACT")
print(f"  GPU     : {gpu_name}")
print("="*55)
print("\n  Download these files:")
print(f"  {OUTPUT_DIR}/closeness_model_N5000.pth")
print(f"  {OUTPUT_DIR}/results_closeness_N5000_summary.txt")
print(f"  {OUTPUT_DIR}/fig_closeness_N5000_training.png")
