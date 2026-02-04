"""
Figure 4: Diffusion-based Time Series Imputation Process.
v2: Claude orange palette, uniform grid/box sizes, compact reference-style layout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np

# ── Grid config ──
ROWS, COLS = 5, 4           # L×K grid
CS = 0.28                    # cell size
GW = COLS * CS               # grid width  = 1.12
GH = ROWS * CS               # grid height = 1.40

# Observation mask (1=observed, 0=missing)  ~40 % missing
OBS = np.array([
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
])

# ── Claude orange palette ──
C_OBS   = '#E8853D'    # observed data (Claude warm orange)
C_OBS_L = '#F5D5B8'    # light orange for imputed circle fill hint
C_MASK  = '#9E9E9E'    # mask cells (neutral grey)
C_WHITE = '#FFFFFF'
C_BD    = '#666666'     # grid border
C_ARR   = '#2C3E50'    # arrows (dark)
C_MBG   = '#F0E2D0'    # model box bg (warm cream)
C_MBD   = '#C4956A'    # model box border (warm brown)


def grid(ax, x0, y0, mask_arr, mode='data'):
    """Draw L×K grid. mode: data/mask/masked/output/imputed"""
    for r in range(ROWS):
        for c in range(COLS):
            x = x0 + c * CS
            y = y0 + (ROWS - 1 - r) * CS
            v = mask_arr[r, c]
            if mode in ('data', 'masked'):
                fc = C_OBS if v else C_WHITE
            elif mode == 'mask':
                fc = C_MASK if v else C_WHITE
            elif mode == 'output':
                fc = C_OBS
            elif mode == 'imputed':
                fc = C_OBS if v else C_WHITE
            else:
                fc = C_WHITE
            ax.add_patch(Rectangle((x, y), CS, CS, fc=fc, ec=C_BD,
                                   lw=0.5, zorder=2))
            if mode == 'imputed' and v == 0:
                ax.add_patch(Circle((x + CS/2, y + CS/2), CS * 0.30,
                                    fc='none', ec=C_OBS, lw=1.0,
                                    ls=(0, (2, 2)), zorder=3))


def OP(ax, x, y, txt, fs=11, r=0.18):
    """Circled operator symbol."""
    ax.add_patch(Circle((x, y), r, fc='white', ec=C_BD, lw=1.0, zorder=3))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#333', zorder=4)


def AR(ax, x1, y1, x2, y2, lw=1.2, c=C_ARR):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.18,head_length=0.13',
                                color=c, lw=lw, mutation_scale=11), zorder=2)


def LN(ax, x1, y1, x2, y2, lw=1.0, c=C_ARR, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2,
            solid_capstyle='round')


# ══════════════════════════════════════════════════════════════
#  LAYOUT  –  compact 2-row arrangement matching reference
#
#  Row 1 (top):   [Z]                              [Z_imp]
#                   ~                                  ↑
#  Row 2 (bot):   [M]  ⊙  [Z⊙M]  →  [ε_θ]  →  [Ẑ]  ⊙ ⊕
#
#  L,K labels on Z;  legend bottom-centre
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(-0.6, 8.8)
ax.set_ylim(-1.4, 6.0)
ax.axis('off')
ax.set_aspect('equal')

G = 0.08   # arrow gap

# ── y positions ──
y_top = 3.9           # top-row grid base (Z, Z_imp)
y_bot = 1.0           # bottom-row grid base (M, Z⊙M, ε, Ẑ)
yc_top = y_top + GH/2
yc_bot = y_bot + GH/2

# ── x positions (tight, uniform spacing) ──
x_Z   = 0.0           # Z grid (top-left)
x_M   = 0.0           # M grid (bottom-left)
x_op1 = GW + 0.35     # ⊙ (mask × data)
x_ZM  = x_op1 + 0.50  # Z⊙M grid
x_eps = x_ZM + GW + 0.45   # ε_θ model box left edge
x_Zh  = x_eps + GW + 0.55  # Ẑ grid
x_op2 = x_Zh + GW + 0.30   # ⊙ (1-M)
x_op3 = x_op2 + 0.45       # ⊕
x_Zi  = x_op3 + 0.45       # Z_imp grid (top-right)

# ════════════════════════════════════
#  Draw all 5 grids (SAME SIZE)
# ════════════════════════════════════
grid(ax, x_Z,  y_top, OBS, 'data')
grid(ax, x_M,  y_bot, OBS, 'mask')
grid(ax, x_ZM, y_bot, OBS, 'masked')
grid(ax, x_Zh, y_bot, OBS, 'output')
grid(ax, x_Zi, y_top, OBS, 'imputed')

# ════════════════════════════════════
#  Model box ε_θ  (SAME HEIGHT as grid)
# ════════════════════════════════════
mw = GW + 0.08         # same width as grid + tiny padding
mh = GH                # same height as grid
mx = x_eps
my = y_bot
ax.add_patch(FancyBboxPatch((mx, my), mw, mh,
             boxstyle='round,pad=0.10', fc=C_MBG, ec=C_MBD,
             lw=1.3, zorder=2))
mcx = mx + mw/2
mcy = my + mh/2
ax.text(mcx, mcy + 0.08, 'ε', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#333', zorder=3,
        fontstyle='italic')
ax.text(mcx + 0.22, mcy - 0.18, 'θ', ha='center', va='center',
        fontsize=10, color='#666', zorder=3, fontstyle='italic')

# ════════════════════════════════════
#  Grid labels
# ════════════════════════════════════
lfs = 8; lc = '#444'
def lbl_above(x0, y0, txt):
    ax.text(x0 + GW/2, y0 + GH + 0.12, txt, ha='center', fontsize=lfs,
            color=lc, fontstyle='italic')
def lbl_below(x0, y0, txt):
    ax.text(x0 + GW/2, y0 - 0.15, txt, ha='center', fontsize=lfs,
            color=lc, fontstyle='italic', va='top')

lbl_above(x_Z, y_top, 'Z  (observed + missing)')
lbl_below(x_M, y_bot, 'M  (mask)')
lbl_below(x_ZM, y_bot, 'Z ⊙ M')
lbl_below(x_Zh, y_bot, 'Ẑ  (prediction)')
lbl_above(x_Zi, y_top, 'Z_imp  (imputed)')

# ── "~" between Z and M ──
ax.text(x_Z + GW/2, (y_top + y_bot + GH) / 2,
        '~', ha='center', va='center', fontsize=18, color='#555',
        fontweight='bold')

# ════════════════════════════════════
#  L, K dimension labels on Z grid
# ════════════════════════════════════
lx = x_Z - 0.12
AR(ax, lx, y_top + GH - 0.04, lx, y_top + 0.04, c='#666', lw=0.8)
ax.text(lx - 0.14, yc_top, 'L', ha='center', va='center',
        fontsize=10, fontstyle='italic', fontweight='bold', color='#666')
ky = y_top - 0.08
AR(ax, x_Z + 0.04, ky, x_Z + GW - 0.04, ky, c='#666', lw=0.8)
ax.text(x_Z + GW/2, ky - 0.16, 'K', ha='center', va='center',
        fontsize=10, fontstyle='italic', fontweight='bold', color='#666')

# ════════════════════════════════════
#  ARROWS  (bottom-row main flow)
# ════════════════════════════════════

# M → ⊙
OP(ax, x_op1, yc_bot, '⊙')
AR(ax, x_M + GW + G, yc_bot, x_op1 - 0.18 - G, yc_bot)

# Z → ⊙  (data feeds into element-wise product from above)
LN(ax, x_Z + GW/2, y_top, x_Z + GW/2, yc_bot + GH/2 + 0.10)
AR(ax, x_Z + GW/2, yc_bot + GH/2 + 0.10, x_op1, yc_bot + 0.18 + G)

# ⊙ → Z⊙M
AR(ax, x_op1 + 0.18 + G, yc_bot, x_ZM - G, yc_bot)

# Z⊙M → ε_θ
AR(ax, x_ZM + GW + G, yc_bot, mx - G, mcy)

# ε_θ → Ẑ
AR(ax, mx + mw + G, mcy, x_Zh - G, yc_bot)

# Ẑ → ⊙(1-M)
OP(ax, x_op2, yc_bot, '⊙')
AR(ax, x_Zh + GW + G, yc_bot, x_op2 - 0.18 - G, yc_bot)
ax.text(x_op2, yc_bot - 0.30, '(1−M)', ha='center',
        fontsize=7, color='#555', fontstyle='italic')

# ⊕ (combine)
OP(ax, x_op3, yc_bot + 0.65, '⊕', fs=10)

# ⊙(1-M) → ⊕  (upward)
AR(ax, x_op2, yc_bot + 0.18 + G, x_op3 - 0.05, yc_bot + 0.65 - 0.18 - G)

# ════════════════════════════════════
#  M⊙Z path: keep observed values
#  Z grid right → dashed across top → down to ⊕
# ════════════════════════════════════
route_y = y_top + GH + 0.35
LN(ax, x_Z + GW, yc_top, x_Z + GW + 0.08, route_y,
   c='#B86E2A', lw=0.9, ls='--')
LN(ax, x_Z + GW + 0.08, route_y, x_op3, route_y,
   c='#B86E2A', lw=0.9, ls='--')
AR(ax, x_op3, route_y, x_op3, yc_bot + 0.65 + 0.18 + G,
   c='#B86E2A', lw=0.9)
ax.text((x_Z + GW + 0.15 + x_op3) / 2, route_y + 0.14,
        'M ⊙ Z  (keep observed)', ha='center', fontsize=6.5,
        color='#B86E2A', fontstyle='italic')

# ⊕ → Z_imp
AR(ax, x_op3 + 0.18 + G, yc_bot + 0.65, x_Zi - G, yc_top)

# ════════════════════════════════════
#  Legend (bottom-centre, compact)
# ════════════════════════════════════
lgx = 1.8; lgy = -0.55
lgw = 4.5; lgh = 0.90
ax.add_patch(FancyBboxPatch((lgx, lgy), lgw, lgh,
             boxstyle='round,pad=0.06', fc='#FAFAFA', ec='#AAA',
             lw=0.7, zorder=2))
sw = 0.22  # swatch size
items = [
    (lgx + 0.18, lgy + 0.52, C_OBS,  None,  'Observed data'),
    (lgx + 0.18, lgy + 0.15, C_MASK, None,  'Mask'),
    (lgx + 2.30, lgy + 0.52, None,   'imp', 'Imputed data'),
    (lgx + 2.30, lgy + 0.15, C_MBG,  'eps', 'ε  Model'),
]
for ix, iy, fc, special, txt in items:
    if special == 'imp':
        ax.add_patch(Rectangle((ix, iy), sw, sw, fc='white', ec=C_BD,
                               lw=0.5, zorder=3))
        ax.add_patch(Circle((ix + sw/2, iy + sw/2), sw * 0.35,
                            fc='none', ec=C_OBS, lw=0.9,
                            ls=(0, (2, 2)), zorder=4))
    elif special == 'eps':
        ax.add_patch(FancyBboxPatch((ix, iy), sw, sw,
                     boxstyle='round,pad=0.02', fc=C_MBG, ec=C_MBD,
                     lw=0.7, zorder=3))
        ax.text(ix + sw/2, iy + sw/2, 'ε', ha='center', va='center',
                fontsize=9, fontweight='bold', fontstyle='italic',
                color='#333', zorder=4)
    else:
        ax.add_patch(Rectangle((ix, iy), sw, sw, fc=fc, ec=C_BD,
                               lw=0.5, zorder=3))
    ax.text(ix + sw + 0.10, iy + sw/2, txt, va='center', fontsize=8,
            color='#333', zorder=3)

# ════════════════════════════════════
#  Equation
# ════════════════════════════════════
ax.text(4.1, -1.1,
        r'$\mathbf{Z}_{\mathrm{imp}} = M \odot Z \;+\;'
        r' (1\!-\!M) \odot \hat{Z}_\theta$',
        ha='center', fontsize=10, color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.20', fc='#fafafa', ec='#bdc3c7',
                  lw=0.6))

fig.tight_layout()
out = '/home/user/sssdtcn/Figures4paper'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}/fig4_imputation_process.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 4 (v2) saved.')
