"""
Figure 4: Diffusion-based Time Series Imputation Process.
Exact template replication - only color changed from blue to orange.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION (matching template exactly)
# ══════════════════════════════════════════════════════════════

ROWS, COLS = 5, 5
CS = 0.16
GW = COLS * CS
GH = ROWS * CS

# Mask pattern
MASK = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
])

# Colors - Claude orange instead of blue
C_OBS  = '#E8853D'   # Claude orange (was blue)
C_MASK = '#808080'   # Gray
C_WH   = '#FFFFFF'   # White
C_BD   = '#000000'   # Black border
C_ARR  = '#4A90D9'   # Blue arrows
C_MBG  = '#D0D0D0'   # Model background
C_MBD  = '#888888'   # Model border


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

            ax.add_patch(Rectangle((x, y), CS, CS, fc=fc, ec=C_BD, lw=0.3, zorder=2))

            if mode == 'imputed' and v == 0:
                ax.add_patch(Circle((x + CS/2, y + CS/2), CS * 0.30,
                    fc='none', ec=C_OBS, lw=0.7, ls=(0, (2, 2)), zorder=3))


def draw_op(ax, x, y, symbol, fs=9, r=0.11):
    ax.add_patch(Circle((x, y), r, fc='white', ec=C_BD, lw=0.8, zorder=3))
    ax.text(x, y, symbol, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#000', zorder=4)


def arrow(ax, x1, y1, x2, y2, c=C_ARR, lw=0.7, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->,head_width=0.06,head_length=0.05',
                        color=c, lw=lw, ls=ls), zorder=2)


def line(ax, x1, y1, x2, y2, c=C_ARR, lw=0.7, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2, solid_capstyle='round')


# ══════════════════════════════════════════════════════════════
#  EXACT TEMPLATE LAYOUT - 7 grids + 3 operators + ε + legend
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7.0, 5.5))
ax.set_xlim(-0.4, 5.8)
ax.set_ylim(-1.3, 3.8)
ax.axis('off')
ax.set_aspect('equal')

# ══════════════════════════════════════════════════════════════
#  POSITIONS (matching template)
# ══════════════════════════════════════════════════════════════

# Left column
x_data = 0.0
y_data = 2.4      # Grid 1: original data (top-left)
y_mask = 1.2      # Grid 2: mask (below data)

# Middle column
x_zm = 1.6        # Grid 3: Z⊙M
y_zm = 1.2

x_mid2 = 2.5      # Grid 4: middle-right full grid
y_mid2 = 1.2

# Right column - top (two imputed grids stacked)
x_imp = 4.0
y_imp1 = 2.65     # Grid 5: imputed top
y_imp2 = 1.75     # Grid 6: imputed bottom

# Right column - bottom
x_final = 4.0
y_final = 0.0     # Grid 7: final output (full)

# ε model position
eps_x = 2.0
eps_y = 2.9
eps_w = 0.7
eps_h = 0.5

# Operators
op1_x = 1.15      # ⊙ (left, between mask and Z⊙M)
op1_y = 1.6

op2_x = 4.85      # ⊙ (right side)
op2_y = 1.4

op3_x = 4.85      # + (below ⊙)
op3_y = 0.7

# ══════════════════════════════════════════════════════════════
#  DRAW 7 GRIDS
# ══════════════════════════════════════════════════════════════

# Grid 1: Original data (top-left)
draw_grid(ax, x_data, y_data, MASK, 'data')

# Grid 2: Mask (left, below data)
draw_grid(ax, x_data, y_mask, MASK, 'mask')

# Grid 3: Z⊙M (middle-left)
draw_grid(ax, x_zm, y_zm, MASK, 'masked')

# Grid 4: Full grid (middle-right)
draw_grid(ax, x_mid2, y_mid2, MASK, 'full')

# Grid 5: Imputed (right-top-upper)
draw_grid(ax, x_imp, y_imp1, MASK, 'imputed')

# Grid 6: Imputed (right-top-lower)
draw_grid(ax, x_imp, y_imp2, MASK, 'imputed')

# Grid 7: Final output (right-bottom, full)
draw_grid(ax, x_final, y_final, MASK, 'full')

# ══════════════════════════════════════════════════════════════
#  ε MODEL BOX
# ══════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch((eps_x, eps_y), eps_w, eps_h,
             boxstyle='round,pad=0.03', fc=C_MBG, ec=C_MBD, lw=0.8, zorder=2))
ax.text(eps_x + eps_w/2, eps_y + eps_h/2, 'ε', ha='center', va='center',
        fontsize=14, fontweight='bold', fontstyle='italic', color='#000', zorder=3)

# ══════════════════════════════════════════════════════════════
#  3 OPERATORS
# ══════════════════════════════════════════════════════════════
draw_op(ax, op1_x, op1_y, '⊙')
draw_op(ax, op2_x, op2_y, '⊙')
draw_op(ax, op3_x, op3_y, '+')

# ══════════════════════════════════════════════════════════════
#  ARROWS (matching template)
# ══════════════════════════════════════════════════════════════
G = 0.04

# --- Dashed path (top): data → ε → imputed ---
# From data top-right corner, up and right to ε
line(ax, x_data + GW, y_data + GH/2 + 0.1, x_data + GW + 0.1, y_data + GH + 0.15, c=C_ARR, ls='--')
line(ax, x_data + GW + 0.1, y_data + GH + 0.15, eps_x - 0.05, y_data + GH + 0.15, c=C_ARR, ls='--')
arrow(ax, eps_x - 0.05, y_data + GH + 0.15, eps_x, eps_y + eps_h/2, c=C_ARR, ls='--')

# From ε to imputed grids (dashed)
line(ax, eps_x + eps_w, eps_y + eps_h/2, eps_x + eps_w + 0.1, y_data + GH + 0.15, c=C_ARR, ls='--')
line(ax, eps_x + eps_w + 0.1, y_data + GH + 0.15, x_imp + GW + 0.1, y_data + GH + 0.15, c=C_ARR, ls='--')
# Arrow down to top imputed grid
arrow(ax, x_imp + GW + 0.1, y_data + GH + 0.15, x_imp + GW, y_imp1 + GH/2, c=C_ARR, ls='--')

# Dashed line between two imputed grids
line(ax, x_imp + GW + 0.15, y_imp1 + GH/2, x_imp + GW + 0.15, y_imp2 + GH/2, c=C_ARR, ls='--')

# --- Solid paths ---
# Data down to ⊙
line(ax, x_data + GW/2, y_data, x_data + GW/2, op1_y + 0.2, c=C_ARR)
arrow(ax, x_data + GW/2, op1_y + 0.2, op1_x - 0.08, op1_y + 0.05, c=C_ARR)

# Mask right to ⊙
arrow(ax, x_data + GW + G, y_mask + GH/2, op1_x - 0.11 - G, op1_y, c=C_ARR)

# ⊙ to Z⊙M
arrow(ax, op1_x + 0.11 + G, op1_y, x_zm - G, y_zm + GH/2, c=C_ARR)

# Z⊙M up to ε
line(ax, x_zm + GW/2, y_zm + GH, x_zm + GW/2, eps_y + eps_h/2, c=C_ARR)
arrow(ax, x_zm + GW/2, eps_y + eps_h/2, eps_x - G, eps_y + eps_h/2, c=C_ARR)

# ε to middle-right grid (down)
line(ax, eps_x + eps_w, eps_y + eps_h/2, x_mid2 + GW/2, eps_y + eps_h/2, c=C_ARR)
arrow(ax, x_mid2 + GW/2, eps_y + eps_h/2, x_mid2 + GW/2, y_mid2 + GH + G, c=C_ARR)

# Middle-right to ⊙ (long horizontal line)
arrow(ax, x_mid2 + GW + G, y_mid2 + GH/2, op2_x - 0.11 - G, op2_y, c=C_ARR)

# ⊙ to + (dashed)
arrow(ax, op2_x, op2_y - 0.11 - G, op3_x, op3_y + 0.11 + G, c=C_ARR, ls='--')

# From imputed area down to +
line(ax, x_imp + GW + 0.15, y_imp2 + GH/2, x_imp + GW + 0.15, op3_y + 0.3, c=C_ARR, ls='--')
arrow(ax, x_imp + GW + 0.15, op3_y + 0.3, op3_x + 0.11 + G, op3_y, c=C_ARR, ls='--')

# + to final grid
line(ax, op3_x, op3_y - 0.11 - G, op3_x, y_final + GH + 0.1, c=C_ARR)
line(ax, op3_x, y_final + GH + 0.1, x_final + GW/2, y_final + GH + 0.1, c=C_ARR)
arrow(ax, x_final + GW/2, y_final + GH + 0.1, x_final + GW/2, y_final + GH, c=C_ARR)

# ══════════════════════════════════════════════════════════════
#  LABELS
# ══════════════════════════════════════════════════════════════

# ~ between data and mask
ax.text(x_data + GW/2, (y_data + y_mask + GH) / 2, '~',
        ha='center', va='center', fontsize=11, color='#000', fontweight='bold')

# L dimension (vertical, left of mask)
lx = x_data - 0.12
ax.annotate('', xy=(lx, y_mask), xytext=(lx, y_mask + GH),
    arrowprops=dict(arrowstyle='<->', color='#000', lw=0.5))
ax.text(lx - 0.12, y_mask + GH/2, 'L', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#000')

# K dimension (horizontal, below mask)
ky = y_mask - 0.12
ax.annotate('', xy=(x_data, ky), xytext=(x_data + GW, ky),
    arrowprops=dict(arrowstyle='<->', color='#000', lw=0.5))
ax.text(x_data + GW/2, ky - 0.12, 'K', ha='center', fontsize=9,
        fontstyle='italic', color='#000')

# ══════════════════════════════════════════════════════════════
#  LEGEND BOX (exactly as template)
# ══════════════════════════════════════════════════════════════
lgx = 1.4
lgy = -0.15
lg_w = 1.15
lg_h = 0.80
sw = 0.12

# Legend box
ax.add_patch(Rectangle((lgx, lgy), lg_w, lg_h, fc='white', ec='#000', lw=0.5, zorder=3))

# Legend items (4 rows)
row_h = 0.18
items = [
    (lgx + 0.06, lgy + lg_h - 0.20, C_OBS, None, 'Observed data'),
    (lgx + 0.06, lgy + lg_h - 0.38, C_MASK, None, 'Mask'),
    (lgx + 0.06, lgy + lg_h - 0.56, None, 'imp', 'Imputed data'),
    (lgx + 0.06, lgy + lg_h - 0.74, None, 'eps', 'ε  Model'),
]

for ix, iy, fc, sp_type, txt in items:
    if sp_type == 'imp':
        ax.add_patch(Rectangle((ix, iy), sw, sw, fc='white', ec=C_BD, lw=0.3, zorder=4))
        ax.add_patch(Circle((ix + sw/2, iy + sw/2), sw * 0.32,
                            fc='none', ec=C_OBS, lw=0.6, ls=(0, (2, 2)), zorder=5))
    elif sp_type == 'eps':
        ax.add_patch(FancyBboxPatch((ix, iy), sw, sw,
                     boxstyle='round,pad=0.01', fc=C_MBG, ec=C_MBD, lw=0.4, zorder=4))
        ax.text(ix + sw/2, iy + sw/2, 'ε', ha='center', va='center',
                fontsize=7, fontweight='bold', fontstyle='italic', color='#000', zorder=5)
    else:
        ax.add_patch(Rectangle((ix, iy), sw, sw, fc=fc, ec=C_BD, lw=0.3, zorder=4))

    ax.text(ix + sw + 0.05, iy + sw/2, txt, va='center', fontsize=7, color='#000', zorder=4)

# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════
fig.tight_layout()
out = '/home/user/sssdtcn/Figures4paper'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}/fig4_imputation_process.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 4 saved - 7 grids, 3 operators, orange color.')
