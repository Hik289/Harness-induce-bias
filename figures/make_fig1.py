"""Regenerate fig1_overview.pdf with clean, readable layout."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})

K = np.array([1, 3, 5, 8])
x_pos = np.linspace(0.1, 0.9, 4)

# ─── Panel A: Phenomenon ───────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('(A)  Same task + same LLM, different harnesses', fontsize=8, fontweight='bold', pad=4)

# Input box
ax.add_patch(plt.FancyBboxPatch((0.25, 0.85), 0.50, 0.10,
    boxstyle='round,pad=0.02', fc='#f3f4f6', ec='#6b7280', lw=1))
ax.text(0.50, 0.905, 'Same task  /  Same LLM', ha='center', va='center', fontsize=7.5)

# Arrows down to two harness boxes
ax.annotate('', xy=(0.22, 0.73), xytext=(0.38, 0.85),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='#374151'))
ax.annotate('', xy=(0.78, 0.73), xytext=(0.62, 0.85),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='#374151'))

# H0 box
ax.add_patch(plt.FancyBboxPatch((0.03, 0.60), 0.35, 0.14,
    boxstyle='round,pad=0.02', fc='#dbeafe', ec='#2563eb', lw=1.2))
ax.text(0.205, 0.675, 'H0  raw harness', ha='center', va='center',
        fontsize=7.5, color='#1e40af', fontweight='bold')

# H2 box
ax.add_patch(plt.FancyBboxPatch((0.62, 0.60), 0.35, 0.14,
    boxstyle='round,pad=0.02', fc='#ffedd5', ec='#ea580c', lw=1.2))
ax.text(0.795, 0.675, 'H2  risk-gated', ha='center', va='center',
        fontsize=7.5, color='#9a3412', fontweight='bold')

# Diverging belief trajectories (schematic lines)
ax_inner = ax.inset_axes([0.05, 0.05, 0.90, 0.50])
ax_inner.set_xlim(0.5, 8.5); ax_inner.set_ylim(0.10, 0.35)
ax_inner.spines['top'].set_visible(False); ax_inner.spines['right'].set_visible(False)
ax_inner.set_xticks(K); ax_inner.set_xticklabels([f'K={k}' for k in K], fontsize=6.5)
ax_inner.set_yticks([]); ax_inner.set_ylabel(r'$D_\mathrm{growth}$', fontsize=7)
d_h0 = np.array([0.156, 0.200, 0.257, 0.254])
d_h2 = np.array([0.142, 0.179, 0.154, 0.158])
ax_inner.plot(K, d_h0, '-o', color='#2563eb', lw=2, ms=4, label='H0 raw')
ax_inner.plot(K, d_h2, '-s', color='#ea580c', lw=2, ms=4, label='H2 risk-gated')
ax_inner.legend(fontsize=6, loc='upper left', frameon=False)
ax_inner.annotate('1.65× growth\n(K=1→K=5)', xy=(5, 0.257), xytext=(6.2, 0.280),
                  fontsize=6, color='#1e40af',
                  arrowprops=dict(arrowstyle='->', lw=0.8, color='#1e40af'))

# ─── Panel B: BIWM Alignment ──────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('(B)  Cross-harness alignment (\u200bBIWM)', fontsize=8, fontweight='bold', pad=4)

# BIWM box
ax.add_patch(plt.FancyBboxPatch((0.15, 0.75), 0.70, 0.16,
    boxstyle='round,pad=0.02', fc='#dcfce7', ec='#16a34a', lw=1.5))
ax.text(0.50, 0.845, 'BIWM  (canonicalise / log / align)', ha='center', va='center',
        fontsize=7.5, color='#14532d', fontweight='bold')

# Alignment chart
ax_inner2 = ax.inset_axes([0.05, 0.05, 0.90, 0.58])
ax_inner2.set_xlim(0.5, 8.5)
ax_inner2.set_ylim(0.13, 0.20)
ax_inner2.spines['top'].set_visible(False); ax_inner2.spines['right'].set_visible(False)
ax_inner2.set_xticks(K); ax_inner2.set_xticklabels([f'K={k}' for k in K], fontsize=6.5)
ax_inner2.set_yticks([0.14, 0.16, 0.18, 0.20])
ax_inner2.set_yticklabels(['0.14','0.16','0.18','0.20'], fontsize=6)
ax_inner2.set_ylabel(r'$D_\mathrm{growth}$', fontsize=7)
naive = np.array([0.1712, 0.1834, 0.1747, 0.1835])
aligned = np.array([0.1588, 0.1777, 0.1546, 0.1470])
ax_inner2.plot(K, naive, '-o', color='#6b7280', lw=2, ms=4, label='Naive mean (H1–H5)')
ax_inner2.plot(K, aligned, '-o', color='#16a34a', lw=2, ms=4, label='Aligned (BIWM)')
ax_inner2.fill_between(K, aligned, naive, alpha=0.15, color='#16a34a')
for ki, na, al in zip(K, naive, aligned):
    ax_inner2.annotate(f'{al-na:+.3f}', xy=(ki, (na+al)/2),
                       ha='center', va='center', fontsize=5.5, color='#166534')
ax_inner2.legend(fontsize=6, loc='upper left', frameon=False)
ax_inner2.set_title('Gap: −0.012 (K=1) → −0.037 (K=8)', fontsize=7,
                    color='#166534', pad=2)

plt.tight_layout(pad=0.5)
fig.savefig('fig1_overview.pdf', bbox_inches='tight', dpi=200)
print("fig1_overview.pdf written")
