"""
Figure 4: Diffusion-based Time Series Imputation Process.
v6: Exact template layout with proper flow.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

ROWS, COLS = 5, 5
CS = 0.22
GW = COLS * CS
GH = ROWS * CS

# Mask: 1 = observed, 0 = missing
MASK = np.array([
    [1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
])

# Colors
C_OBS  = '#6699CC'
C_MASK = '#9E9E9E'
C_WH   = '#FFFFFF'
C_BD   = '#444444'
C_ARR  = '#2C3E50'
C_DASH = '#6699CC'
C_MBG  = '#E8E8E8'
C_MBD  = '#888888'


def draw_grid(ax, x0, y0, mask, mode):
    for r in range(ROWS):
        for c in range(COLS):
            x = x0 + c * CS
            y = y0 + (ROWS - 1 - r) * CS
            v = mask[r, c]

            if mode == 'data':
                fc = C_OBS if v else C_WH
            elif mode == 'mask':
                fc = C_MASK if v else C_WH
            elif mode == 'masked':
                fc = C_OBS if v else C_WH
            elif mode == 'full':
                fc = C_OBS
            elif mode == 'imputed':
                fc = C_OBS if v else C_WH
            else:
                fc = C_WH

            ax.add_patch(Rectangle((x, y), CS, CS, fc=fc, ec=C_BD, lw=0.4, zorder=2))

            if mode == 'imputed' and v == 0:
                ax.add_patch(Circle((x + CS/2, y + CS/2), CS * 0.35,
                    fc='none', ec=C_OBS, lw=1.0, ls=(0, (2, 2)), zorder=3))


def draw_op(ax, x, y, symbol, fs=10, r=0.15):
    ax.add_patch(Circle((x, y), r, fc='white', ec=C_BD, lw=1.0, zorder=3))
    ax.text(x, y, symbol, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#333', zorder=4)


def arrow(ax, x1, y1, x2, y2, c=C_ARR, lw=1.0, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->,head_width=0.12,head_length=0.08',
                        color=c, lw=lw, ls=ls), zorder=2)


def line(ax, x1, y1, x2, y2, c=C_ARR, lw=1.0, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2,
            solid_capstyle='round')


# ══════════════════════════════════════════════════════════════
#  PRECISE TEMPLATE LAYOUT
#
#  TOP:    [Z] -------- ε -------- [Z_imp]
#           ~          ↑ ↓          ↑
#  BOT:    [M] → ⊙ → [Z⊙M] → [Ẑ] → ⊙ → ⊕
#                                  (1-M)  ↓
#                                      [final]
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.set_xlim(-0.6, 8.0)
ax.set_ylim(-1.2, 4.5)
ax.axis('off')
ax.set_aspect('equal')

# ── Positions ──
x1 = 0.0            # Z, M
x2 = 2.0            # Z⊙M
x3 = 3.8            # Ẑ (prediction)
x4 = 6.0            # Z_imp (top), final (bottom)

y_top = 2.8
y_bot = 0.8

yc_t = y_top + GH/2
yc_b = y_bot + GH/2

# ══════════════════════════════════════════════════════════════
#  GRIDS
# ══════════════════════════════════════════════════════════════

# Left column
draw_grid(ax, x1, y_top, MASK, 'data')      # Z
draw_grid(ax, x1, y_bot, MASK, 'mask')      # M

# Middle
draw_grid(ax, x2, y_bot, MASK, 'masked')    # Z⊙M

# Right-middle
draw_grid(ax, x3, y_bot, MASK, 'full')      # Ẑ (prediction, all blue)

# Right column
draw_grid(ax, x4, y_top, MASK, 'imputed')   # Z_imp (top)
draw_grid(ax, x4, y_bot, MASK, 'full')      # Final output (bottom)

# ══════════════════════════════════════════════════════════════
#  ε MODEL (top center)
# ══════════════════════════════════════════════════════════════
eps_w = GW * 0.9
eps_h = GH * 0.6
eps_x = (x2 + x3) / 2 + GW/2 - eps_w/2
eps_y = y_top + 0.15
ax.add_patch(FancyBboxPatch((eps_x, eps_y), eps_w, eps_h,
             boxstyle='round,pad=0.06', fc=C_MBG, ec=C_MBD, lw=1.2, zorder=2))
ax.text(eps_x + eps_w/2, eps_y + eps_h/2, 'ε', ha='center', va='center',
        fontsize=18, fontweight='bold', fontstyle='italic', color='#333', zorder=3)

# ══════════════════════════════════════════════════════════════
#  OPERATORS
# ══════════════════════════════════════════════════════════════

# ⊙ between M and Z⊙M
op1_x = (x1 + GW + x2) / 2
op1_y = yc_b
draw_op(ax, op1_x, op1_y, '⊙')

# ⊙(1-M) after Ẑ
op2_x = x3 + GW + 0.35
op2_y = yc_b
draw_op(ax, op2_x, op2_y, '⊙')
ax.text(op2_x, op2_y - 0.28, '(1−M)', ha='center', fontsize=6.5,
        color='#555', fontstyle='italic')

# ⊕ combining results
op3_x = op2_x + 0.55
op3_y = yc_b
draw_op(ax, op3_x, op3_y, '⊕', fs=9)

# ══════════════════════════════════════════════════════════════
#  ARROWS
# ══════════════════════════════════════════════════════════════
G = 0.06

# --- Main flow (bottom row) ---

# 1. Z down to ⊙
line(ax, x1 + GW/2, y_top, x1 + GW/2, op1_y + 0.35)
arrow(ax, x1 + GW/2, op1_y + 0.35, op1_x - 0.12, op1_y + 0.08)

# 2. M → ⊙
arrow(ax, x1 + GW + G, yc_b, op1_x - 0.15 - G, yc_b)

# 3. ⊙ → Z⊙M
arrow(ax, op1_x + 0.15 + G, op1_y, x2 - G, yc_b)

# 4. Z⊙M up to ε (L-shaped)
line(ax, x2 + GW/2, y_bot + GH, x2 + GW/2, eps_y + eps_h/2)
arrow(ax, x2 + GW/2, eps_y + eps_h/2, eps_x - G, eps_y + eps_h/2)

# 5. ε down to Ẑ (L-shaped)
line(ax, eps_x + eps_w, eps_y + eps_h/2, x3 + GW/2, eps_y + eps_h/2)
arrow(ax, x3 + GW/2, eps_y + eps_h/2, x3 + GW/2, y_bot + GH + G)

# 6. Ẑ → ⊙(1-M)
arrow(ax, x3 + GW + G, yc_b, op2_x - 0.15 - G, op2_y)

# 7. ⊙(1-M) → ⊕
arrow(ax, op2_x + 0.15 + G, op2_y, op3_x - 0.15 - G, op3_y)

# 8. ⊕ → final (down)
arrow(ax, op3_x, op3_y - 0.15 - G, op3_x, y_bot + GH + G, c=C_ARR)
line(ax, op3_x, y_bot + GH + G, x4 + GW/2, y_bot + GH + G)
arrow(ax, x4 + GW/2, y_bot + GH + G, x4 + GW/2, y_bot + GH + 0.01)

# --- Top dashed flow (keep observed) ---

# 9. Z (top-right) to ε (dashed)
dash_y1 = y_top + GH + 0.15
line(ax, x1 + GW, yc_t, x1 + GW + 0.08, dash_y1, c=C_DASH, ls='--')
line(ax, x1 + GW + 0.08, dash_y1, eps_x - 0.1, dash_y1, c=C_DASH, ls='--')
arrow(ax, eps_x - 0.1, dash_y1, eps_x - G, eps_y + eps_h - 0.08, c=C_DASH, ls='--')

# 10. ε to Z_imp (dashed)
dash_y2 = eps_y + eps_h + 0.12
line(ax, eps_x + eps_w, eps_y + eps_h - 0.08, eps_x + eps_w + 0.1, dash_y2, c=C_DASH, ls='--')
line(ax, eps_x + eps_w + 0.1, dash_y2, x4 + GW + 0.1, dash_y2, c=C_DASH, ls='--')
arrow(ax, x4 + GW + 0.1, dash_y2, x4 + GW + G, yc_t, c=C_DASH, ls='--')

# ══════════════════════════════════════════════════════════════
#  LABELS
# ══════════════════════════════════════════════════════════════

# ~ between Z and M
ax.text(x1 + GW/2, (y_top + y_bot + GH) / 2, '~',
        ha='center', va='center', fontsize=14, color='#555', fontweight='bold')

# K, L dimensions
ky = y_bot - 0.20
ax.annotate('', xy=(x1, ky), xytext=(x1 + GW, ky),
    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.6))
ax.text(x1 + GW/2, ky - 0.16, 'K', ha='center', fontsize=8,
        fontstyle='italic', fontweight='bold', color='#666')

lx = x1 - 0.14
ax.annotate('', xy=(lx, y_bot), xytext=(lx, y_bot + GH),
    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.6))
ax.text(lx - 0.16, y_bot + GH/2, 'L', ha='center', va='center',
        fontsize=8, fontstyle='italic', fontweight='bold', color='#666')

# ══════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════
lgx = 1.2
lgy = -0.65
sw = 0.20

items = [
    (C_OBS, None, 'Observed data'),
    (C_MASK, None, 'Mask'),
    (None, 'imp', 'Imputed data'),
    (C_MBG, 'eps', 'ε  Model'),
]

curr_x = lgx
for fc, sp_type, txt in items:
    if sp_type == 'imp':
        ax.add_patch(Rectangle((curr_x, lgy), sw, sw, fc='white', ec=C_BD, lw=0.4, zorder=3))
        ax.add_patch(Circle((curr_x + sw/2, lgy + sw/2), sw * 0.38,
                            fc='none', ec=C_OBS, lw=0.9, ls=(0, (2, 2)), zorder=4))
    elif sp_type == 'eps':
        ax.add_patch(FancyBboxPatch((curr_x, lgy), sw, sw,
                     boxstyle='round,pad=0.02', fc=C_MBG, ec=C_MBD, lw=0.5, zorder=3))
        ax.text(curr_x + sw/2, lgy + sw/2, 'ε', ha='center', va='center',
                fontsize=9, fontweight='bold', fontstyle='italic', color='#333', zorder=4)
    else:
        ax.add_patch(Rectangle((curr_x, lgy), sw, sw, fc=fc, ec=C_BD, lw=0.4, zorder=3))

    ax.text(curr_x + sw + 0.08, lgy + sw/2, txt, va='center', fontsize=7.5, color='#333', zorder=3)
    curr_x += sw + 0.08 + len(txt) * 0.058 + 0.32

# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════
fig.tight_layout()
out = '/home/user/sssdtcn/Figures4paper'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}/fig4_imputation_process.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 4 (v6) saved.')
