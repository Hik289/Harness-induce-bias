"""
intuition.pdf — Steak analogy intuition figure.

Two side-by-side panels: same steak (= same task + same LLM),
different harness (knife&fork vs. risk-gated chopsticks) → different belief.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Polygon, Rectangle, Circle

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Palette
STEAK_OUTER = "#7a2e1f"   # seared crust
STEAK_INNER = "#b8523a"   # medium-rare
PLATE       = "#fafafa"
PLATE_EDGE  = "#444444"
GOOD_FILL   = "#d9ead3"   # light green
BAD_FILL    = "#f4cccc"   # light red
NEUTRAL     = "#fff2cc"   # light yellow
EDGE        = "#111111"


def draw_steak(ax, cx, cy, w=2.6, h=1.45):
    """Cartoon steak on a plate."""
    # plate
    plate = Ellipse((cx, cy - 0.15), w * 1.55, h * 1.30,
                    facecolor=PLATE, edgecolor=PLATE_EDGE, linewidth=1.2)
    ax.add_patch(plate)
    # steak outer (seared)
    steak = Ellipse((cx, cy), w, h,
                    facecolor=STEAK_OUTER, edgecolor="#3a1a10", linewidth=1.4)
    ax.add_patch(steak)
    # inner medium-rare
    inner = Ellipse((cx, cy), w * 0.72, h * 0.62,
                    facecolor=STEAK_INNER, edgecolor="none")
    ax.add_patch(inner)
    # tiny marbling marks
    for (dx, dy) in [(-0.4, 0.15), (0.3, -0.1), (0.0, 0.25), (-0.2, -0.3)]:
        ax.add_patch(Ellipse((cx + dx, cy + dy), 0.12, 0.05,
                             facecolor="#e0a78a", edgecolor="none"))


def draw_knife_fork(ax, cx, cy):
    """Simple knife + fork icon."""
    # fork (left)
    ax.add_patch(Rectangle((cx - 0.55, cy - 0.6), 0.10, 1.2,
                           facecolor="#888", edgecolor=EDGE, linewidth=0.8))
    for i, dx in enumerate([-0.62, -0.55, -0.48]):
        ax.add_patch(Rectangle((dx + cx + 0.04, cy + 0.45), 0.03, 0.30,
                               facecolor="#888", edgecolor=EDGE, linewidth=0.5))
    # knife (right)
    ax.add_patch(Rectangle((cx + 0.45, cy - 0.6), 0.10, 0.7,
                           facecolor="#888", edgecolor=EDGE, linewidth=0.8))
    blade = Polygon(
        [(cx + 0.42, cy + 0.10), (cx + 0.58, cy + 0.10),
         (cx + 0.62, cy + 0.75), (cx + 0.46, cy + 0.75)],
        facecolor="#cccccc", edgecolor=EDGE, linewidth=0.8,
    )
    ax.add_patch(blade)


def draw_chopsticks(ax, cx, cy):
    """Two chopsticks crossed, with a small 'gate' lock symbol."""
    # chopstick 1
    ax.add_patch(Polygon(
        [(cx - 0.55, cy - 0.65), (cx - 0.48, cy - 0.65),
         (cx + 0.20, cy + 0.75), (cx + 0.13, cy + 0.75)],
        facecolor="#c49a6c", edgecolor=EDGE, linewidth=0.8,
    ))
    # chopstick 2
    ax.add_patch(Polygon(
        [(cx + 0.55, cy - 0.65), (cx + 0.48, cy - 0.65),
         (cx - 0.20, cy + 0.75), (cx - 0.13, cy + 0.75)],
        facecolor="#c49a6c", edgecolor=EDGE, linewidth=0.8,
    ))
    # tiny lock/gate badge
    ax.add_patch(Circle((cx + 0.55, cy + 0.55), 0.18,
                        facecolor="#cc0000", edgecolor=EDGE, linewidth=0.8))
    ax.text(cx + 0.55, cy + 0.55, "!", ha="center", va="center",
            fontsize=10, weight="bold", color="white")


def panel(ax, title, harness_name, harness_drawer, belief_text,
          belief_fill, prob_text, prob_color):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top label
    ax.add_patch(FancyBboxPatch(
        (1.0, 10.6), 8.0, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=NEUTRAL, edgecolor=EDGE, linewidth=1.2,
    ))
    ax.text(5.0, 11.10, "Same Task  +  Same LLM",
            ha="center", va="center", fontsize=11, weight="bold")
    ax.text(5.0, 10.78, title,
            ha="center", va="center", fontsize=9.5, style="italic",
            color="#444")

    # The steak (same on both sides)
    draw_steak(ax, cx=5.0, cy=8.8, w=3.2, h=1.7)
    ax.text(5.0, 7.55,
            "same coding task / bug",
            ha="center", va="center", fontsize=9, style="italic", color="#333")

    # Harness box
    ax.add_patch(FancyBboxPatch(
        (1.2, 5.2), 7.6, 1.7,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="#eef3fb", edgecolor=EDGE, linewidth=1.2,
    ))
    ax.text(2.6, 6.55, harness_name,
            ha="center", va="center", fontsize=10.5, weight="bold")
    harness_drawer(ax, cx=2.6, cy=5.85)
    ax.text(6.3, 6.05,
            "→  observe / act / verify",
            ha="left", va="center", fontsize=9.5, color="#222", style="italic")

    # Arrow down
    a = FancyArrowPatch((5.0, 5.15), (5.0, 4.30),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.6, color="#333")
    ax.add_patch(a)

    # Belief box
    ax.add_patch(FancyBboxPatch(
        (0.8, 1.8), 8.4, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=belief_fill, edgecolor=EDGE, linewidth=1.3,
    ))
    ax.text(5.0, 3.85, "LLM belief", ha="center", va="center",
            fontsize=10.5, weight="bold")
    ax.text(5.0, 3.20, belief_text,
            ha="center", va="center", fontsize=10, color="#111")
    ax.text(5.0, 2.30, prob_text,
            ha="center", va="center", fontsize=12, weight="bold",
            color=prob_color)


# ---- Build figure --------------------------------------------------------
FIG_W, FIG_H = 11.5, 7.0
fig = plt.figure(figsize=(FIG_W, FIG_H))

ax_left  = fig.add_axes([0.02, 0.18, 0.42, 0.78])
ax_right = fig.add_axes([0.56, 0.18, 0.42, 0.78])

panel(
    ax_left,
    title="(steak / coding task is identical on both sides)",
    harness_name="Harness  H0\n(knife & fork)",
    harness_drawer=draw_knife_fork,
    belief_text='“steak is tender, task looks doable”',
    belief_fill=GOOD_FILL,
    prob_text="P(success)  =  0.85",
    prob_color="#2e7d32",
)

panel(
    ax_right,
    title="(steak / coding task is identical on both sides)",
    harness_name="Harness  H2\n(risk-gated chopsticks)",
    harness_drawer=draw_chopsticks,
    belief_text='relabeled: “policy_violation”\n“the meat resists the tool”',
    belief_fill=BAD_FILL,
    prob_text="P(success)  =  0.28",
    prob_color="#b00020",
)

# ---- Center divider + caption -------------------------------------------
# Vertical separator with center label
ax_mid = fig.add_axes([0.44, 0.20, 0.12, 0.76])
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 10)
ax_mid.axis("off")
# dashed vertical line
ax_mid.plot([0.5, 0.5], [0.5, 9.5], linestyle="--", color="#888", linewidth=1.2)
# center label
ax_mid.add_patch(FancyBboxPatch(
    (0.02, 4.3), 0.96, 1.4,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor="#ffffff", edgecolor=EDGE, linewidth=1.3,
))
ax_mid.text(0.5, 5.30, "Harness  =", ha="center", va="center",
            fontsize=10, weight="bold")
ax_mid.text(0.5, 4.80, "the tool that\nshapes the belief",
            ha="center", va="center", fontsize=9, style="italic")

# Bottom caption banner
ax_cap = fig.add_axes([0.0, 0.0, 1.0, 0.14])
ax_cap.set_xlim(0, 10)
ax_cap.set_ylim(0, 2)
ax_cap.axis("off")
ax_cap.add_patch(FancyBboxPatch(
    (0.4, 0.20), 9.2, 1.55,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor="#eef3fb", edgecolor=EDGE, linewidth=1.2,
))
ax_cap.text(5.0, 1.20,
            r"$D_{\mathrm{belief}}(H_0,\, H_2;\, K=5)  =  0.45$",
            ha="center", va="center", fontsize=13, weight="bold")
ax_cap.text(5.0, 0.55,
            "→  the tool diverges the dinner impression  "
            "(same steak, different belief about it).",
            ha="center", va="center", fontsize=10, style="italic",
            color="#222")

# ---- Save ---------------------------------------------------------------
out = "intuition.pdf"
plt.savefig(out, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"wrote {out}")
