"""
Generate a focused PDF explaining Approach V: Temporal Self-Attention Dynamic Dilation.
One main flowchart + key math formulas.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

# ── colour palette ──────────────────────────────────────────────────────────
C_BG    = '#F7F9FC'
C_BLUE  = '#2A6EBB'
C_TEAL  = '#1A8A7A'
C_ORG   = '#D4750A'
C_RED   = '#B03A2E'
C_PURP  = '#6C3483'
C_GRAY  = '#5D6D7E'
C_LT    = '#EAF2FF'
C_LT2   = '#E8F8F5'
C_LT3   = '#FEF9E7'
C_LT4   = '#F9EBEA'
C_LT5   = '#F5EEF8'

def fbox(ax, xy, w, h, color, lc, text, tsize=11, bold=False,
         tc='white', radius=0.012, zorder=3, lw=1.8):
    """Draw a rounded box with centred text."""
    x, y = xy
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f'round,pad=0.01,rounding_size={radius}',
                       facecolor=color, edgecolor=lc, linewidth=lw, zorder=zorder)
    ax.add_patch(b)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=tsize, color=tc, fontweight=weight, zorder=zorder+1,
            linespacing=1.4)

def arrow(ax, x0, y0, x1, y1, color=C_GRAY, lw=2.0, zorder=2,
          arrowstyle='->', mutation_scale=18):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=mutation_scale),
                zorder=zorder)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE  ── single A4-landscape figure
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.set_facecolor(C_BG)
ax.axis('off')

# ── Title ─────────────────────────────────────────────────────────────────
ax.text(8, 10.45, 'Approach V · Temporal Self-Attention Dynamic Dilation',
        ha='center', va='center', fontsize=18, fontweight='bold', color=C_BLUE)
ax.text(8, 10.0,
        'How per-position expected offset  $\\bar{d}_t = \\sum_{\\delta}\\, \\delta \\cdot a_{t,\\delta}$'
        '  adaptively modulates the convolution kernel',
        ha='center', va='center', fontsize=12.5, color=C_GRAY)

# thin divider
ax.axhline(9.70, xmin=0.04, xmax=0.96, color=C_BLUE, lw=1.2, alpha=0.4)

# ── SOURCE LABELS (top-right) ──────────────────────────────────────────────
ax.text(15.6, 10.45, '① Dynamic Conv\n(Chen et al., CVPR 2020)', ha='right',
        va='center', fontsize=7.5, color=C_ORG,
        bbox=dict(boxstyle='round,pad=0.3', fc='#FEF5E7', ec=C_ORG, lw=1))
ax.text(15.6, 9.90, '② Non-Local NN\n(Wang et al., CVPR 2018)', ha='right',
        va='center', fontsize=7.5, color=C_TEAL,
        bbox=dict(boxstyle='round,pad=0.3', fc='#E8F8F5', ec=C_TEAL, lw=1))

# ══════════════════════════════════════════════════════════════════════════
# MAIN FLOW  (left → right, y ≈ 7.0 as the spine)
# ══════════════════════════════════════════════════════════════════════════
spine_y = 6.70

# Step positions along x-axis
xs = [1.35, 3.80, 6.30, 9.00, 11.60, 14.10]
labels_main = [
    'Input\nSequence\n$\\mathbf{x} \\in \\mathbb{R}^{T \\times C}$',
    'Project to\n$Q, K, V$',
    'Pairwise\nSimilarity\n(Non-Local)',
    'Softmax\nAttention Weights\n$a_{t,\\delta}$',
    'Expected\nDilation\n$\\bar{d}_t$',
    'Adaptive\nConv Output\n$y_t$',
]
box_colors = [C_BLUE, C_TEAL, C_TEAL, C_PURP, C_ORG, C_RED]
box_lts    = [C_LT,   C_LT2,  C_LT2,  C_LT5,  C_LT3, C_LT4]
box_lcs    = [C_BLUE, C_TEAL, C_TEAL, C_PURP, C_ORG, C_RED]

BW, BH = 2.1, 1.15
for i, (x, lab, bc, lc) in enumerate(zip(xs, labels_main, box_colors, box_lcs)):
    fbox(ax, (x, spine_y), BW, BH, bc, lc, lab,
         tsize=9.5, bold=True, tc='white', radius=0.025)

# arrows between boxes
for i in range(len(xs)-1):
    arrow(ax, xs[i]+BW/2, spine_y, xs[i+1]-BW/2, spine_y, color=C_GRAY, lw=2.4)

# ── Step numbers ──────────────────────────────────────────────────────────
step_labels = ['❶','❷','❸','❹','❺']
for i, x in enumerate(xs[:-1]):
    mx = (xs[i]+BW/2 + xs[i+1]-BW/2) / 2
    ax.text(mx, spine_y + 0.78, step_labels[i], ha='center', va='center',
            fontsize=12, color=C_GRAY, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════
# FORMULA PANEL  (below the main flow)
# Three formula boxes
# ══════════════════════════════════════════════════════════════════════════
fy = 4.60   # centre of formula row

formula_blocks = [
    # (x-centre, title, formula-lines, bg, ec, source)
    (2.90, '❷ Query / Key Projection',
     [r'$Q = \mathbf{x} W_Q,\quad K = \mathbf{x} W_K,\quad V = \mathbf{x} W_V$',
      r'$W_Q, W_K, W_V \in \mathbb{R}^{C \times d_k}$'],
     '#E8F8F5', C_TEAL, '② Non-Local NN'),
    (7.60, '❸ – ❹  Attention over Dilation Offsets',
     [r'$e_{t,\delta} = \dfrac{q_t \cdot k_{t+\delta}^{\top}}{\sqrt{d_k}}$',
      r'$a_{t,\delta} = \dfrac{\exp(e_{t,\delta})}{\sum_{\delta^{\prime}} \exp(e_{t,\delta^{\prime}})}$'],
     '#F5EEF8', C_PURP, '② Non-Local NN'),
    (12.60, '❺  Expected Dilation  &  Kernel Modulation',
     [r'$\bar{d}_t \;=\; \sum_{\delta}\, \delta \cdot a_{t,\delta}$',
      r'$y_t \;=\; \sum_{\delta}\, a_{t,\delta}\; W_{\delta}\; x_{t+\delta}$'],
     '#FEF9E7', C_ORG, '① Dynamic Conv'),
]

for cx, title, lines, bg, ec, src in formula_blocks:
    # outer box
    bw, bh = 4.50, 2.30
    rect = FancyBboxPatch((cx - bw/2, fy - bh/2), bw, bh,
                          boxstyle='round,pad=0.02,rounding_size=0.03',
                          facecolor=bg, edgecolor=ec, linewidth=2.0, zorder=3)
    ax.add_patch(rect)
    # title
    ax.text(cx, fy + bh/2 - 0.22, title, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=ec, zorder=4)
    ax.axhline(fy + bh/2 - 0.42, xmin=(cx - bw/2)/16, xmax=(cx + bw/2)/16,
               color=ec, lw=0.8, alpha=0.6)
    # source tag
    ax.text(cx + bw/2 - 0.08, fy + bh/2 - 0.20, src,
            ha='right', va='center', fontsize=6.5, color=ec,
            fontstyle='italic', zorder=4)
    # formulas
    n = len(lines)
    for j, ln in enumerate(lines):
        ypos = fy - 0.08 + (n - 1 - j) * 0.62 - 0.22
        ax.text(cx, ypos, ln, ha='center', va='center',
                fontsize=12.5, color='#1A1A1A', zorder=4)

# vertical arrows from spine to formula panels
for (cx, *_), bsx in zip(formula_blocks, [3.80, 6.30/9.00, 11.60]):
    # use the matching spine x
    pass
# draw arrows: spine box bottom → formula box top
connections = [
    (xs[1], xs[2], 2.90),   # step ❷ → formula 1
    (xs[3], xs[3], 7.60),   # step ❸/❹ → formula 2
    (xs[4], xs[4], 12.60),  # step ❺ → formula 3
]
for sx, _, fx in connections:
    arrow(ax, fx, spine_y - BH/2, fx, fy + 1.16,
          color='#AAAAAA', lw=1.5)

# ══════════════════════════════════════════════════════════════════════════
# KEY INSIGHT BOX  (bottom)
# ══════════════════════════════════════════════════════════════════════════
ins_y = 2.35
ins_w, ins_h = 13.5, 1.60
rect2 = FancyBboxPatch((8 - ins_w/2, ins_y - ins_h/2), ins_w, ins_h,
                        boxstyle='round,pad=0.02,rounding_size=0.03',
                        facecolor='#EAF2FF', edgecolor=C_BLUE, linewidth=2.0, zorder=3)
ax.add_patch(rect2)
ax.text(8, ins_y + 0.45, '⚑  Key Insight: Why This Achieves Adaptive Dilation',
        ha='center', va='center', fontsize=11, fontweight='bold', color=C_BLUE, zorder=4)
ax.text(8, ins_y - 0.05,
        r'Each time-step $t$ computes its own attention distribution $\{a_{t,\delta}\}$ over a discrete set of offsets '
        r'$\delta \in \{1,2,4,8,\ldots\}$.',
        ha='center', va='center', fontsize=9.8, color='#1A1A1A', zorder=4)
ax.text(8, ins_y - 0.50,
        r'The expected offset  $\bar{d}_t = \sum_\delta \delta \cdot a_{t,\delta}$  is a '
        r'content-dependent soft dilation — no fixed receptive field, '
        r'fully differentiable end-to-end.',
        ha='center', va='center', fontsize=9.8, color='#1A1A1A', zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# MINI VISUALISATION: bar chart of a_{t,δ} and d̄_t arrow  (bottom-right)
# ══════════════════════════════════════════════════════════════════════════
axi = fig.add_axes([0.74, 0.07, 0.22, 0.16])
offsets = [1, 2, 4, 8, 16]
# example attention weights (peaked at δ=4)
weights = np.array([0.06, 0.15, 0.45, 0.25, 0.09])
bars = axi.bar(offsets, weights, color=[C_PURP]*5, edgecolor='white',
               width=[0.5, 0.8, 1.5, 2.5, 4], align='center', zorder=3)
bars[2].set_facecolor(C_ORG)          # highlight peak
d_bar = np.sum(np.array(offsets) * weights)
axi.axvline(d_bar, color=C_RED, lw=2.2, ls='--', zorder=4, label=f'$\\bar{{d}}_t={d_bar:.1f}$')
axi.set_xlabel(r'Dilation offset $\delta$', fontsize=8)
axi.set_ylabel(r'$a_{t,\delta}$', fontsize=8)
axi.set_title(r'Example attention over offsets', fontsize=8, pad=3)
axi.set_xticks(offsets)
axi.legend(fontsize=8, loc='upper right')
axi.set_facecolor('#FAFAFA')
axi.tick_params(labelsize=7)
for sp in ['top','right']: axi.spines[sp].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════
# Fixed vs Adaptive comparison panel (bottom-left)
# ══════════════════════════════════════════════════════════════════════════
axc = fig.add_axes([0.04, 0.07, 0.22, 0.16])
T = np.arange(20)
# fixed dilation
fixed_d  = np.ones(20) * 4
# adaptive dilation (varies with content)
np.random.seed(42)
adapt_d  = 2 + 6*np.abs(np.sin(T * 0.45)) + np.random.randn(20)*0.3
adapt_d  = np.clip(adapt_d, 1, 12)
axc.plot(T, fixed_d,  color=C_GRAY,  lw=2,   ls='--', label='Fixed $d$=4')
axc.plot(T, adapt_d,  color=C_ORG,   lw=2.0, label=r'Adaptive $\bar{d}_t$')
axc.fill_between(T, fixed_d, adapt_d, alpha=0.15, color=C_ORG)
axc.set_xlabel('Time step $t$', fontsize=8)
axc.set_ylabel('Dilation', fontsize=8)
axc.set_title('Fixed vs. Adaptive Dilation', fontsize=8, pad=3)
axc.legend(fontsize=7.5)
axc.set_facecolor('#FAFAFA')
axc.tick_params(labelsize=7)
for sp in ['top','right']: axc.spines[sp].set_visible(False)

# ── footer ────────────────────────────────────────────────────────────────
ax.text(8, 0.30,
        'Sources:  ① Chen et al., "Dynamic Convolution: Attention over Convolution Kernels," CVPR 2020  ·  '
        '② Wang et al., "Non-local Neural Networks," CVPR 2018',
        ha='center', va='center', fontsize=7.8, color=C_GRAY, style='italic')

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════
out_path = '/home/user/sssdtcn/Approach_V_Dynamic_Dilation.pdf'
with PdfPages(out_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight', facecolor=C_BG)
plt.close(fig)
print(f'Saved → {out_path}')
