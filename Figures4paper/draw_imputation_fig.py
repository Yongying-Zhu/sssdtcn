"""
Figure 4: Diffusion-based Time Series Imputation Process.
v7: Exact template replication with Claude orange color.
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
CS = 0.18
GW = COLS * CS
GH = ROWS * CS

# Mask pattern (matching template)
MASK = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
])

# Colors - Claude orange instead of blue
C_OBS  = '#E8853D'   # Claude orange
C_MASK = '#808080'   # Gray
C_WH   = '#FFFFFF'   # White
C_BD   = '#000000'   # Black border
C_ARR  = '#4A90D9'   # Blue arrows (like template)
C_MBG  = '#D0D0D0'   # Model background gray
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

            ax.add_patch(Rectangle((x, y), CS, CS, fc=fc, ec=C_BD, lw=0.4, zorder=2))

            if mode == 'imputed' and v == 0:
                ax.add_patch(Circle((x + CS/2, y + CS/2), CS * 0.30,
                    fc='none', ec=C_OBS, lw=0.8, ls=(0, (2, 2)), zorder=3))


def draw_op(ax, x, y, symbol, fs=10, r=0.12):
    ax.add_patch(Circle((x, y), r, fc='white', ec=C_BD, lw=0.8, zorder=3))
    ax.text(x, y, symbol, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#000', zorder=4)


def arrow(ax, x1, y1, x2, y2, c=C_ARR, lw=0.8, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->,head_width=0.08,head_length=0.06',
                        color=c, lw=lw, ls=ls), zorder=2)


def line(ax, x1, y1, x2, y2, c=C_ARR, lw=0.8, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2,
            solid_capstyle='round')


# ══════════════════════════════════════════════════════════════
#  LAYOUT (exact template replication)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.set_xlim(-0.5, 6.0)
ax.set_ylim(-1.0, 4.0)
ax.axis('off')
ax.set_aspect('equal')

# Positions (matching template layout)
x1 = 0.0            # Left column (Z, M)
x2 = 1.8            # Z⊙M
x3 = 3.6            # Right column (imputed top, full bottom)

y_top = 2.4         # Top row
y_mid = 1.2         # Middle row
y_bot = 0.0         # Bottom row

# ══════════════════════════════════════════════════════════════
#  GRIDS
# ══════════════════════════════════════════════════════════════

# Left column
draw_grid(ax, x1, y_top, MASK, 'data')       # Z (top-left)
draw_grid(ax, x1, y_mid, MASK, 'mask')       # M (middle-left)

# Middle column
draw_grid(ax, x2, y_mid, MASK, 'masked')     # Z⊙M

# Right column
draw_grid(ax, x3, y_top, MASK, 'imputed')    # Imputed (top-right)
draw_grid(ax, x3, y_bot, MASK, 'full')       # Full output (bottom-right)

# ══════════════════════════════════════════════════════════════
#  ε MODEL BOX
# ══════════════════════════════════════════════════════════════
eps_w = GW * 0.8
eps_h = GH * 0.55
eps_x = (x1 + x3) / 2 + GW/2 - eps_w/2
eps_y = y_top + 0.35
ax.add_patch(FancyBboxPatch((eps_x, eps_y), eps_w, eps_h,
             boxstyle='round,pad=0.04', fc=C_MBG, ec=C_MBD, lw=1.0, zorder=2))
ax.text(eps_x + eps_w/2, eps_y + eps_h/2, 'ε', ha='center', va='center',
        fontsize=14, fontweight='bold', fontstyle='italic', color='#000', zorder=3)

# ══════════════════════════════════════════════════════════════
#  OPERATORS
# ══════════════════════════════════════════════════════════════

# ⊙ between left column and Z⊙M
op1_x = (x1 + GW + x2) / 2
op1_y = y_mid + GH/2
draw_op(ax, op1_x, op1_y, '⊙')

# ⊙ on right side (for 1-M multiplication)
op2_x = x3 + GW + 0.25
op2_y = y_mid + GH/2
draw_op(ax, op2_x, op2_y, '⊙')

# ⊕ below ⊙
op3_x = op2_x
op3_y = y_mid - 0.15
draw_op(ax, op3_x, op3_y, '+')

# ══════════════════════════════════════════════════════════════
#  ARROWS (matching template exactly)
# ══════════════════════════════════════════════════════════════
G = 0.05

# --- Dashed path (top): Z → ε → imputed ---
dash_y = y_top + GH + 0.12
# Z to ε (dashed)
line(ax, x1 + GW, y_top + GH/2 + 0.15, x1 + GW + 0.08, dash_y, c=C_ARR, ls='--')
line(ax, x1 + GW + 0.08, dash_y, eps_x - 0.05, dash_y, c=C_ARR, ls='--')
arrow(ax, eps_x - 0.05, dash_y, eps_x, eps_y + eps_h/2, c=C_ARR, ls='--')

# ε to imputed (dashed)
line(ax, eps_x + eps_w, eps_y + eps_h/2, eps_x + eps_w + 0.05, dash_y, c=C_ARR, ls='--')
line(ax, eps_x + eps_w + 0.05, dash_y, x3 + GW + 0.08, dash_y, c=C_ARR, ls='--')
arrow(ax, x3 + GW + 0.08, dash_y, x3 + GW, y_top + GH/2 + 0.15, c=C_ARR, ls='--')

# --- Solid path (main flow) ---
# Z down toward ⊙
line(ax, x1 + GW/2, y_top, x1 + GW/2, op1_y + 0.25, c=C_ARR)
arrow(ax, x1 + GW/2, op1_y + 0.25, op1_x - 0.10, op1_y + 0.05, c=C_ARR)

# M right to ⊙
arrow(ax, x1 + GW + G, op1_y, op1_x - 0.12 - G, op1_y, c=C_ARR)

# ⊙ to Z⊙M
arrow(ax, op1_x + 0.12 + G, op1_y, x2 - G, op1_y, c=C_ARR)

# Z⊙M to right side (continues to ⊙)
arrow(ax, x2 + GW + G, op1_y, op2_x - 0.12 - G, op2_y, c=C_ARR)

# Upper part: from Z⊙M area up to ε, then to ⊙
line(ax, x2 + GW/2, y_mid + GH, x2 + GW/2, eps_y + eps_h/2, c=C_ARR)
arrow(ax, x2 + GW/2, eps_y + eps_h/2, eps_x - G, eps_y + eps_h/2, c=C_ARR)

line(ax, eps_x + eps_w, eps_y + eps_h/2, op2_x, eps_y + eps_h/2, c=C_ARR)
arrow(ax, op2_x, eps_y + eps_h/2, op2_x, op2_y + 0.12 + G, c=C_ARR)

# ⊙ to ⊕ (dashed for observed part)
arrow(ax, op2_x, op2_y - 0.12 - G, op3_x, op3_y + 0.12 + G, c=C_ARR, ls='--')

# ⊕ down to bottom grid
arrow(ax, op3_x, op3_y - 0.12 - G, op3_x, y_bot + GH + G, c=C_ARR)
line(ax, op3_x, y_bot + GH + G, x3 + GW/2, y_bot + GH + G, c=C_ARR)
arrow(ax, x3 + GW/2, y_bot + GH + G, x3 + GW/2, y_bot + GH, c=C_ARR)

# ══════════════════════════════════════════════════════════════
#  LABELS
# ══════════════════════════════════════════════════════════════

# ~ between Z and M
ax.text(x1 + GW/2, (y_top + y_mid + GH) / 2, '~',
        ha='center', va='center', fontsize=12, color='#000', fontweight='bold')

# K dimension (horizontal)
ky = y_mid - 0.18
ax.annotate('', xy=(x1, ky), xytext=(x1 + GW, ky),
    arrowprops=dict(arrowstyle='<->', color='#000', lw=0.5))
ax.text(x1 + GW/2, ky - 0.12, 'K', ha='center', fontsize=9,
        fontstyle='italic', color='#000')

# L dimension (vertical)
lx = x1 - 0.12
ax.annotate('', xy=(lx, y_mid - 0.35), xytext=(lx, y_mid + GH + 0.35),
    arrowprops=dict(arrowstyle='<->', color='#000', lw=0.5))
ax.text(lx - 0.12, y_mid + GH/2, 'L', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#000')

# ══════════════════════════════════════════════════════════════
#  LEGEND (exactly as template - boxed legend)
# ══════════════════════════════════════════════════════════════
lgx = 1.35
lgy = -0.05
lg_w = 1.30
lg_h = 0.85
sw = 0.14

# Legend box
ax.add_patch(Rectangle((lgx, lgy), lg_w, lg_h, fc='white', ec='#000', lw=0.5, zorder=3))

# Legend items
items = [
    (lgx + 0.08, lgy + lg_h - 0.22, C_OBS, None, 'Observed data'),
    (lgx + 0.08, lgy + lg_h - 0.42, C_MASK, None, 'Mask'),
    (lgx + 0.08, lgy + lg_h - 0.62, None, 'imp', 'Imputed data'),
    (lgx + 0.08, lgy + lg_h - 0.82, None, 'eps', 'ε  Model'),
]

for ix, iy, fc, sp_type, txt in items:
    if sp_type == 'imp':
        ax.add_patch(Rectangle((ix, iy), sw, sw, fc='white', ec=C_BD, lw=0.4, zorder=4))
        ax.add_patch(Circle((ix + sw/2, iy + sw/2), sw * 0.32,
                            fc='none', ec=C_OBS, lw=0.7, ls=(0, (2, 2)), zorder=5))
    elif sp_type == 'eps':
        ax.add_patch(FancyBboxPatch((ix, iy), sw, sw,
                     boxstyle='round,pad=0.01', fc=C_MBG, ec=C_MBD, lw=0.4, zorder=4))
        ax.text(ix + sw/2, iy + sw/2, 'ε', ha='center', va='center',
                fontsize=7, fontweight='bold', fontstyle='italic', color='#000', zorder=5)
    else:
        ax.add_patch(Rectangle((ix, iy), sw, sw, fc=fc, ec=C_BD, lw=0.4, zorder=4))

    ax.text(ix + sw + 0.06, iy + sw/2, txt, va='center', fontsize=8, color='#000', zorder=4)

# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════
fig.tight_layout()
out = '/home/user/sssdtcn/Figures4paper'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}/fig4_imputation_process.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 4 (v7 - orange) saved.')
