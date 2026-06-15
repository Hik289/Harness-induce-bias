import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# Figure 1 conceptual overview
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.axis('off')
# panels
for x0, title in [(0.03, 'A. Same task, different harnesses'), (0.52, 'B. Protocol-layer alignment')]:
    ax.add_patch(plt.Rectangle((x0, 0.08), 0.45, 0.84, fill=False, lw=1.2, color='0.25'))
    ax.text(x0+0.225, 0.88, title, ha='center', va='center', weight='bold')
# left panel boxes
ax.text(0.255, 0.78, 'Same task + same LLM', ha='center', bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', ec='0.3'))
ax.annotate('', xy=(0.16,0.64), xytext=(0.23,0.73), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(0.35,0.64), xytext=(0.28,0.73), arrowprops=dict(arrowstyle='->'))
ax.text(0.16, 0.60, 'H0 raw\nopen branch', ha='center', bbox=dict(boxstyle='round,pad=0.25', fc='#dbeafe', ec='#2563eb'))
ax.text(0.35, 0.60, 'H2 risk-gated\nblocked branch', ha='center', bbox=dict(boxstyle='round,pad=0.25', fc='#ffedd5', ec='#ea580c'))
K=[1,3,5,8]; x=np.linspace(0.10,0.42,4); y1=[0.38,0.42,0.52,0.45]; y2=[0.40,0.58,0.80,0.46]
ax.plot(x,y1, '-o', color='#2563eb', lw=2, ms=3); ax.plot(x,y2, '-o', color='#ea580c', lw=2, ms=3)
for xi,ki in zip(x,K): ax.text(xi,0.24,f'K={ki}',ha='center',fontsize=7)
ax.text(0.255,0.18,'growth-D separates K-amplified drift\nfrom on-arrival interface shift',ha='center',fontsize=8)
ax.text(0.255,0.105,'steak/tool analogy: tool changes belief about same object',ha='center',fontsize=7,color='0.35')
# right panel
ax.text(0.745, 0.78, 'Multiple harness views', ha='center', bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', ec='0.3'))
ax.annotate('', xy=(0.745,0.63), xytext=(0.745,0.73), arrowprops=dict(arrowstyle='->'))
ax.text(0.745, 0.58, 'BIWM\ncanonicalize + log blocked/repair/verify\nshadow + align', ha='center', bbox=dict(boxstyle='round,pad=0.25', fc='#dcfce7', ec='#16a34a'))
ax.annotate('', xy=(0.745,0.40), xytext=(0.745,0.51), arrowprops=dict(arrowstyle='->'))
x2=np.linspace(0.58,0.91,4); naive=[0.1712,0.1834,0.1747,0.1835]; aligned=[0.1588,0.1777,0.1546,0.1470]
# rescale
yn=0.22+np.array(naive)*1.2; ya=0.22+np.array(aligned)*1.2
ax.plot(x2,yn,'-o',color='#6b7280',lw=2,ms=3,label='mean Hx'); ax.plot(x2,ya,'-o',color='#16a34a',lw=2,ms=3,label='aligned')
for xi,ki in zip(x2,K): ax.text(xi,0.205,f'{ki}',ha='center',fontsize=7)
ax.text(0.745,0.15,'D_growth gap: -0.012 → -0.037',ha='center',fontsize=8,weight='bold',color='#166534')
ax.text(0.745,0.105,'cross-harness voting cancels part of individual drift',ha='center',fontsize=7,color='0.35')
plt.tight_layout()
fig.savefig('fig1_overview.pdf', bbox_inches='tight')

# Phase 1 horizon plot
fig, ax = plt.subplots(figsize=(4.8,3.0))
K=np.array([1,3,5,8])
series={
'H0-H1 structured':[0.156,0.200,0.257,0.254],
'H0-H2 risk-gated':[0.142,0.179,0.154,0.158],
'H0-H4 verify-selective':[0.190,0.208,0.151,0.231],
}
for name,y in series.items(): ax.plot(K,y,'-o',label=name,lw=2,ms=4)
ax.set_xlabel('Rollout horizon K'); ax.set_ylabel(r'$D_{growth}$'); ax.set_xticks(K); ax.grid(alpha=.25); ax.legend(fontsize=7)
fig.savefig('phase1_growth_horizon.pdf', bbox_inches='tight')

# Alignment figure
fig, ax = plt.subplots(figsize=(4.8,3.0))
naiveDg=np.array([0.1712,0.1834,0.1747,0.1835]); alignDg=np.array([0.1588,0.1777,0.1546,0.1470])
ax.plot(K,naiveDg,'-o',label='Mean naive H0-vs-Hx',lw=2,color='#6b7280')
ax.plot(K,alignDg,'-o',label='BIWM cross-harness aligned',lw=2,color='#16a34a')
for x,y,gap in zip(K,alignDg,alignDg-naiveDg): ax.text(x,y-0.008,f'{gap:+.3f}',ha='center',fontsize=7,color='#166534')
ax.set_xlabel('Rollout horizon K'); ax.set_ylabel(r'$D_{growth}$'); ax.set_xticks(K); ax.grid(alpha=.25); ax.legend(fontsize=7)
fig.savefig('biwm_alignment_growth.pdf', bbox_inches='tight')
