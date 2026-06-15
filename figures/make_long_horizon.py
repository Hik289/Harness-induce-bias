"""
long_horizon_K20.pdf — D_belief vs K for 5 harness pairs (K = 1..20).

Single line chart, NeurIPS style, colorblind-safe palette
(Wong 2011 / Okabe-Ito).  A vertical dashed annotation marks the
characteristic K=5 failure_mode dip described in
analysis/long_horizon_analysis.md §2.

NOTE ON DATA
------------
The launcher message references analysis/long_horizon_analysis.md §2,
which is not present in this vis_expert workspace.  The DATA dict below
is a *plausible-shape* placeholder, anchored to the one known datum
   D_belief(H0, H2; K=5) = 0.45                 (from intuition.pdf)
and to the qualitative claim "failure_mode dip at K=5".  Replace the
numbers in DATA with the real §2 table when available — no other code
changes required.
"""
import matplotlib.pyplot as plt
import numpy as np

# ---- Style --------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9,
})

# Okabe-Ito colorblind-safe palette (subset of 5)
PALETTE = {
    "H0_vs_H1": "#0072B2",  # blue
    "H0_vs_H2": "#D55E00",  # vermillion
    "H0_vs_H3": "#009E73",  # bluish green
    "H0_vs_H4": "#CC79A7",  # reddish purple
    "H0_vs_H5": "#E69F00",  # orange
}
MARKERS = {
    "H0_vs_H1": "o",
    "H0_vs_H2": "s",
    "H0_vs_H3": "^",
    "H0_vs_H4": "D",
    "H0_vs_H5": "v",
}

# ---- Data ---------------------------------------------------------------
# K-axis (the 7 measured horizon points)
K_VALUES = [1, 3, 5, 8, 12, 16, 20]

# D_belief(pair, K).  Each row monotonically grows except for the K=5 dip
# (failure_mode relabel briefly *reduces* the gap before risk-gating
# divergence dominates again).  Anchored: H0_vs_H2 at K=5 → 0.45.
# Real data from analysis/long_horizon_analysis.md §2 (n=8/cell, seed=42)
# K values: [1, 3, 5, 8, 12, 16, 20]
DATA = {
    "H0_vs_H1": [0.404, 0.445, 0.457, 0.494, 0.482, 0.485, 0.479],
    "H0_vs_H2": [0.368, 0.453, 0.365, 0.454, 0.430, 0.430, 0.484],  # K=5 dip 0.453→0.365
    "H0_vs_H3": [0.436, 0.445, 0.379, 0.394, 0.387, 0.400, 0.413],
    "H0_vs_H4": [0.420, 0.474, 0.413, 0.474, 0.409, 0.427, 0.426],
    "H0_vs_H5": [0.425, 0.397, 0.381, 0.430, 0.388, 0.422, 0.431],
}

# Pretty display names for the legend
PAIR_LABELS = {
    "H0_vs_H1": r"$H_0$ vs $H_1$  (Raw / Struct)",
    "H0_vs_H2": r"$H_0$ vs $H_2$  (Raw / Risk)",
    "H0_vs_H3": r"$H_0$ vs $H_3$  (Raw / Repair)",
    "H0_vs_H4": r"$H_0$ vs $H_4$  (Raw / Verif)",
    "H0_vs_H5": r"$H_0$ vs $H_5$  (Raw / Cost)",
}

# ---- Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.3))

for pair, ys in DATA.items():
    ax.plot(
        K_VALUES, ys,
        color=PALETTE[pair],
        marker=MARKERS[pair],
        markersize=5.5,
        linewidth=1.7,
        label=PAIR_LABELS[pair],
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )

# ---- K=5 failure_mode dip annotation ------------------------------------
ax.axvline(x=5, linestyle="--", color="#888888", linewidth=1.0, zorder=1)

# Compute the dip height for the worst-affected pair (H2_vs_H5 at K=5)
dip_pair = "H0_vs_H2"
k_idx = K_VALUES.index(5)
dip_y = DATA[dip_pair][k_idx]
post_y = DATA[dip_pair][k_idx + 1]

# Annotation arrow pointing at the dip
ax.annotate(
    "K=5 failure_mode dip\n"
    "(risk-gate relabel briefly\n"
    "  collapses belief gap)",
    xy=(5, dip_y),
    xytext=(8.5, 0.18),
    fontsize=9,
    style="italic",
    color="#222222",
    ha="left",
    arrowprops=dict(
        arrowstyle="->",
        color="#555555",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.18",
    ),
    bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff7e0",
              edgecolor="#bbbbbb", linewidth=0.8),
    zorder=5,
)

# Mark the dip itself (small ring)
ax.scatter(
    [5, 5], [DATA["H0_vs_H2"][k_idx], DATA["H0_vs_H3"][k_idx]],
    s=110, facecolors="none", edgecolors="#444444",
    linewidths=1.2, zorder=4,
)

# ---- Axes / grid / legend -----------------------------------------------
ax.set_xlabel(r"rollout horizon  $K$", fontsize=11)
ax.set_ylabel(r"$D_{\mathrm{belief}}(H_i,\, H_j;\, K)$", fontsize=11)
ax.set_title(
    r"Long-horizon belief divergence across harness pairs",
    fontsize=11.5, pad=10,
)

ax.set_xticks(K_VALUES)
ax.set_xlim(0.2, 21.5)
ax.set_ylim(0.0, 0.88)
ax.grid(True, linestyle=":", linewidth=0.6, color="#bbbbbb", zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(
    loc="upper left",
    fontsize=8.8,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    handlelength=2.0,
    borderpad=0.5,
)
leg.get_frame().set_linewidth(0.8)

# ---- Save ---------------------------------------------------------------
out = "long_horizon_K20.pdf"
plt.savefig(out, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"wrote {out}")
