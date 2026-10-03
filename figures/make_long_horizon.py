import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9,
})

PALETTE = {
    "H0_vs_H1": "#0072B2",
    "H0_vs_H2": "#D55E00",
    "H0_vs_H3": "#009E73",
    "H0_vs_H4": "#CC79A7",
    "H0_vs_H5": "#E69F00",
}
MARKERS = {
    "H0_vs_H1": "o",
    "H0_vs_H2": "s",
    "H0_vs_H3": "^",
    "H0_vs_H4": "D",
    "H0_vs_H5": "v",
}

K_VALUES = [1, 3, 5, 8, 12, 16, 20]

DATA = {
    "H0_vs_H1": [0.404, 0.445, 0.457, 0.494, 0.482, 0.485, 0.479],
    "H0_vs_H2": [0.368, 0.453, 0.365, 0.454, 0.430, 0.430, 0.484],
    "H0_vs_H3": [0.436, 0.445, 0.379, 0.394, 0.387, 0.400, 0.413],
    "H0_vs_H4": [0.420, 0.474, 0.413, 0.474, 0.409, 0.427, 0.426],
    "H0_vs_H5": [0.425, 0.397, 0.381, 0.430, 0.388, 0.422, 0.431],
}

PAIR_LABELS = {
    "H0_vs_H1": r"$H_0$ vs $H_1$  (Raw / Struct)",
    "H0_vs_H2": r"$H_0$ vs $H_2$  (Raw / Risk)",
    "H0_vs_H3": r"$H_0$ vs $H_3$  (Raw / Repair)",
    "H0_vs_H4": r"$H_0$ vs $H_4$  (Raw / Verif)",
    "H0_vs_H5": r"$H_0$ vs $H_5$  (Raw / Cost)",
}

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

ax.axvline(x=5, linestyle="--", color="#888888", linewidth=1.0, zorder=1)

dip_pair = "H0_vs_H2"
k_idx = K_VALUES.index(5)
dip_y = DATA[dip_pair][k_idx]
post_y = DATA[dip_pair][k_idx + 1]

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

ax.scatter(
    [5, 5], [DATA["H0_vs_H2"][k_idx], DATA["H0_vs_H3"][k_idx]],
    s=110, facecolors="none", edgecolors="#444444",
    linewidths=1.2, zorder=4,
)

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

out = "long_horizon_K20.pdf"
plt.savefig(out, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"wrote {out}")
