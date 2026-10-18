"""
Figure 3: Real-world validation of GNN-based centrality approximation

Panels:
a) 3D C. Elegans-like neural network
b) 3D Email-Eu-Core-like communication network
c) 3D Power-grid-like spatial network
d) Premium grouped bar plot for zero-shot transfer performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx

# =====================================================
# Output folder
# =====================================================
OUT = "figure3_real_world_validation_bar"
os.makedirs(OUT, exist_ok=True)

# =====================================================
# Colors
# =====================================================
C_BET = "#1565C0"       # blue: betweenness
C_CLOSE = "#E65100"     # orange: closeness
C_GREEN = "#2E7D32"     # C. Elegans
C_RED = "#B71C1C"       # Power Grid
C_GRAY = "#999999"
C_DARK = "#222222"

# =====================================================
# Style
# =====================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5,

    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#777777",

    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.color": "#555555",
    "ytick.color": "#555555",

    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.grid": True,
    "grid.color": "#EDEDED",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",

    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.facecolor": "white",
})

# =====================================================
# Helper functions
# =====================================================
def panel_label_2d(ax, letter, x=-0.10, y=1.08):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left"
    )


def panel_label_3d(ax, letter, x=-0.06, y=1.02):
    ax.text2D(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left"
    )


def clean_3d_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.grid(False)

    # Transparent panes
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))


def normalize_2d_pos(pos):
    """Normalize 2D layout to approximately [-0.5, 0.5]."""
    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes])

    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)

    arr = (arr - mins) / (maxs - mins + 1e-12)
    arr = arr - 0.5

    return {node: arr[i] for i, node in enumerate(nodes)}


def make_3d_from_2d(pos2d, z_mode="random", seed=0, scale_z=0.25):
    """Convert 2D graph coordinates into visually pleasing 3D coordinates."""
    rng = np.random.default_rng(seed)

    nodes = list(pos2d.keys())
    arr = np.array([pos2d[n] for n in nodes])

    if z_mode == "random":
        zvals = rng.normal(0, scale_z, len(nodes))
    elif z_mode == "layered":
        zvals = rng.choice([-scale_z, 0, scale_z], size=len(nodes))
    elif z_mode == "wave":
        zvals = scale_z * np.sin(np.linspace(0, 2 * np.pi, len(nodes)))
    else:
        zvals = np.zeros(len(nodes))

    pos3d = {}
    for i, n in enumerate(nodes):
        pos3d[n] = np.array([arr[i, 0], arr[i, 1], zvals[i]])

    return pos3d


def draw_network_3d(ax, G, pos3d, color, title):
    degrees = np.array([G.degree(n) for n in G.nodes()])
    node_sizes = 10 + 4.0 * degrees

    # Edges
    for u, v in G.edges():
        x = [pos3d[u][0], pos3d[v][0]]
        y = [pos3d[u][1], pos3d[v][1]]
        z = [pos3d[u][2], pos3d[v][2]]

        ax.plot(
            x, y, z,
            color="#BBBBBB",
            lw=0.45,
            alpha=0.55
        )

    xyz = np.array([pos3d[n] for n in G.nodes()])

    # Soft shadow layer
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2] - 0.015,
        s=node_sizes * 1.45,
        c="#EAEAEA",
        alpha=0.70,
        edgecolors="none"
    )

    # Main nodes
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=node_sizes,
        c=color,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.96
    )

    ax.set_title(title, fontsize=8, pad=2)
    clean_3d_axes(ax)

    try:
        ax.set_box_aspect([1, 1, 0.55])
    except Exception:
        pass


# =====================================================
# Build representative 3D networks
# =====================================================
np.random.seed(4)

# -----------------------------------------------------
# Panel a: C. Elegans-like compact neural network
# -----------------------------------------------------
G_ce = nx.watts_strogatz_graph(85, 6, 0.18, seed=1)

pos_ce_2d = nx.spring_layout(G_ce, seed=2, k=0.23)
pos_ce_2d = normalize_2d_pos(pos_ce_2d)

pos_ce_3d = make_3d_from_2d(
    pos_ce_2d,
    z_mode="random",
    seed=1,
    scale_z=0.18
)

# -----------------------------------------------------
# Panel b: Email-Eu-Core-like modular communication graph
# -----------------------------------------------------
sizes = [20, 20, 20, 20]
probs = [
    [0.18,  0.025, 0.020, 0.020],
    [0.025, 0.18,  0.025, 0.020],
    [0.020, 0.025, 0.18,  0.025],
    [0.020, 0.020, 0.025, 0.18],
]

G_email = nx.stochastic_block_model(sizes, probs, seed=3)

cluster_centers = [
    (-0.35,  0.28),
    ( 0.35,  0.28),
    (-0.35, -0.28),
    ( 0.35, -0.28)
]

pos_email_3d = {}
node_start = 0
rng = np.random.default_rng(7)

for block_id, size in enumerate(sizes):
    sub_nodes = list(range(node_start, node_start + size))
    subG = G_email.subgraph(sub_nodes)

    sub_pos = nx.spring_layout(subG, seed=block_id + 5, k=0.40)
    sub_pos = normalize_2d_pos(sub_pos)

    cx, cy = cluster_centers[block_id]

    for n in sub_nodes:
        x, y = sub_pos[n]
        z = 0.12 * rng.normal()

        pos_email_3d[n] = np.array([
            0.23 * x + cx,
            0.20 * y + cy,
            z
        ])

    node_start += size

# -----------------------------------------------------
# Panel c: Power-grid-like elongated spatial network
# -----------------------------------------------------
G_power = nx.random_geometric_graph(120, radius=0.14, seed=5)

if not nx.is_connected(G_power):
    largest_cc = max(nx.connected_components(G_power), key=len)
    G_power = G_power.subgraph(largest_cc).copy()

raw_pos = nx.get_node_attributes(G_power, "pos")

pos_power_3d = {}
for n, (x, y) in raw_pos.items():
    pos_power_3d[n] = np.array([
        1.80 * (x - 0.5),
        0.55 * (y - 0.5),
        0.10 * np.sin(8 * x) + 0.02 * rng.normal()
    ])

# =====================================================
# Real-world validation values
# =====================================================
networks = ["C. Elegans", "Email-Eu-Core", "Power Grid"]
nodes = [131, 986, 4941]

betweenness_tau = np.array([0.6543, 0.7823, 0.6027])
closeness_tau   = np.array([0.2650, 0.7029, 0.1080])

# =====================================================
# Figure canvas
# =====================================================
fig = plt.figure(figsize=(10.8, 6.9))

gs = GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.0, 1.15],
    hspace=0.38,
    wspace=0.28
)

# =====================================================
# Top row: 3D networks
# =====================================================
ax_a = fig.add_subplot(gs[0, 0], projection="3d")
draw_network_3d(
    ax_a,
    G_ce,
    pos_ce_3d,
    C_GREEN,
    "C. Elegans-like neural network"
)
ax_a.view_init(elev=20, azim=-60)
panel_label_3d(ax_a, "a")

ax_b = fig.add_subplot(gs[0, 1], projection="3d")
draw_network_3d(
    ax_b,
    G_email,
    pos_email_3d,
    C_BET,
    "Email-Eu-Core-like communication network"
)
ax_b.view_init(elev=18, azim=-55)
panel_label_3d(ax_b, "b")

ax_c = fig.add_subplot(gs[0, 2], projection="3d")
draw_network_3d(
    ax_c,
    G_power,
    pos_power_3d,
    C_RED,
    "Power-grid-like spatial network"
)
ax_c.view_init(elev=18, azim=-70)
panel_label_3d(ax_c, "c")

# =====================================================
# Bottom row: Premium grouped bar plot
# =====================================================
ax_d = fig.add_subplot(gs[1, :])
panel_label_2d(ax_d, "d", x=-0.035, y=1.06)

x = np.arange(len(networks))
width = 0.34

# Soft background performance bands
ax_d.axhspan(0.0, 0.30, color="#FFF3E0", alpha=0.35, zorder=0)
ax_d.axhspan(0.30, 0.60, color="#F7F7F7", alpha=0.55, zorder=0)
ax_d.axhspan(0.60, 1.00, color="#EAF4EA", alpha=0.30, zorder=0)

# Bars
bars_bet = ax_d.bar(
    x - width / 2,
    betweenness_tau,
    width=width,
    color=C_BET,
    edgecolor="white",
    linewidth=0.8,
    label="Betweenness",
    zorder=3
)

bars_close = ax_d.bar(
    x + width / 2,
    closeness_tau,
    width=width,
    color=C_CLOSE,
    edgecolor="white",
    linewidth=0.8,
    label="Closeness",
    zorder=3
)

# Value labels
for bar in bars_bet:
    h = bar.get_height()
    ax_d.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.025,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=C_BET
    )

for bar in bars_close:
    h = bar.get_height()
    ax_d.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.025,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=C_CLOSE
    )

# Reference lines from synthetic ER test
ax_d.axhline(
    0.851,
    color=C_BET,
    linestyle="--",
    linewidth=0.9,
    alpha=0.75,
    zorder=1
)

ax_d.axhline(
    0.894,
    color=C_CLOSE,
    linestyle="--",
    linewidth=0.9,
    alpha=0.75,
    zorder=1
)

ax_d.text(
    2.55,
    0.851 + 0.012,
    "ER betweenness test = 0.851",
    color=C_BET,
    fontsize=6.7,
    ha="right",
    va="bottom"
)

ax_d.text(
    2.55,
    0.894 + 0.012,
    "ER closeness test = 0.894",
    color=C_CLOSE,
    fontsize=6.7,
    ha="right",
    va="bottom"
)

# Transfer gap labels
for i, (b, c) in enumerate(zip(betweenness_tau, closeness_tau)):
    gap = b - c

    ax_d.plot(
        [i, i],
        [c, b],
        color="#777777",
        linewidth=0.8,
        alpha=0.75,
        zorder=2
    )

    ax_d.text(
        i,
        (b + c) / 2,
        f"gap={gap:.2f}",
        fontsize=6.5,
        color="#555555",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="#CCCCCC",
            linewidth=0.5,
            alpha=0.9
        )
    )

# X labels
xticklabels = [
    "C. Elegans\nN=131",
    "Email-Eu-Core\nN=986",
    "Power Grid\nN=4,941"
]

ax_d.set_xticks(x)
ax_d.set_xticklabels(xticklabels)

# Axis labels and limits
ax_d.set_ylabel("Kendall $\\tau$")
ax_d.set_ylim(0, 1.02)
ax_d.set_xlim(-0.55, len(networks) - 0.45)

ax_d.set_title(
    "Zero-shot transfer to real-world networks",
    fontsize=8,
    pad=4
)

# Clean grid
ax_d.grid(axis="y", color="#E6E6E6", linewidth=0.6)
ax_d.grid(axis="x", visible=False)
ax_d.set_axisbelow(True)

# Legend
ax_d.legend(
    loc="upper left",
    ncol=2,
    frameon=False,
    fontsize=7,
    handlelength=1.0,
    columnspacing=1.2
)

# Interpretation note
ax_d.text(
    0.01,
    -0.22,
    "Betweenness centrality transfers more reliably across real-world networks, while closeness is more sensitive to topology shift.",
    transform=ax_d.transAxes,
    fontsize=7.3,
    color=C_DARK,
    ha="left",
    va="center"
)

# =====================================================
# Overall title
# =====================================================
fig.suptitle(
    "Real-world validation of GNN-based centrality approximation",
    fontsize=10,
    fontweight="bold",
    y=0.98
)

# =====================================================
# Save
# =====================================================
fig.savefig(os.path.join(OUT, "fig3_real_world_validation_bar.png"))
fig.savefig(os.path.join(OUT, "fig3_real_world_validation_bar.pdf"))
fig.savefig(os.path.join(OUT, "fig3_real_world_validation_bar.svg"))

plt.close(fig)

print("Saved files:")
print(os.path.join(OUT, "fig3_real_world_validation_bar.png"))
print(os.path.join(OUT, "fig3_real_world_validation_bar.pdf"))
print(os.path.join(OUT, "fig3_real_world_validation_bar.svg"))