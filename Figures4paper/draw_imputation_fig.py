"""
Figure 4: Diffusion-based Time Series Imputation Process.
Style: grid-based visualisation matching published paper reference.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np

# ── Grid config ──
ROWS, COLS = 6, 5          # L=6 timesteps, K=5 sensors
CS = 0.34                   # cell size
GW = COLS * CS              # grid width  = 1.70
GH = ROWS * CS              # grid height = 2.04

# Observation mask (1=observed, 0=missing)  ~33 % missing
OBS = np.array([
    [1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
])

# ── Colours ──
C_OBS   = '#4A7FB5'   # observed data  (steel blue)
C_MASK  = '#A0A0A0'   # mask cells     (grey)
C_WHITE = '#FFFFFF'    # missing / empty
C_BD    = '#555555'    # grid border
C_ARR   = '#333333'    # arrows
C_MBG   = '#DCDCDC'    # model box background
C_MBD   = '#888888'    # model box border


# ── Helper: draw a grid ─────────────────────────────────────
def grid(ax, x0, y0, mask_arr, mode='data', label=None):
    """
    mode:
      'data'    – blue=observed, white=missing
      'mask'    – grey=observed, white=missing
      'masked'  – same as data (Z⊙M)
      'output'  – all cells blue (model prediction)
      'imputed' – blue=originally observed, dotted ○ = imputed
    """
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
                                   lw=0.55, zorder=2))

            # dotted circle for imputed positions
            if mode == 'imputed' and v == 0:
                ax.add_patch(Circle((x + CS/2, y + CS/2), CS * 0.28,
                                    fc='none', ec=C_OBS, lw=1.1,
                                    ls=(0, (2.5, 2)), zorder=3))

    # optional label below grid
    if label:
        ax.text(x0 + GW/2, y0 - 0.25, label, ha='center', va='top',
                fontsize=9.5, fontweight='bold', color='#333')


# ── Helper: circled operator ─────────────────────────────────
def op(ax, x, y, txt, fs=13, r=0.22):
    ax.add_patch(Circle((x, y), r, fc='white', ec=C_BD, lw=1.2, zorder=3))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#333', zorder=4)


# ── Helper: arrow ────────────────────────────────────────────
def ar(ax, x1, y1, x2, y2, lw=1.3, c=C_ARR):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.22,head_length=0.16',
                                color=c, lw=lw, mutation_scale=12), zorder=2)


def ln(ax, x1, y1, x2, y2, lw=1.2, c=C_ARR, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2,
            solid_capstyle='round')


# ══════════════════════════════════════════════════════════════
#  Main figure
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-0.8, 13.5)
ax.set_ylim(-1.8, 8.2)
ax.axis('off')
ax.set_aspect('equal')

# ── Row positions ──
y_top = 5.4          # top-row grid base
y_bot = 1.6          # bottom-row grid base
yc_top = y_top + GH/2
yc_bot = y_bot + GH/2

# ── x positions ──
x_Z   = 0.3          # Z grid (top-left)
x_M   = 0.3          # M grid (bottom-left)
x_op1 = 2.7          # ⊙  (mask operation)
x_ZM  = 3.5          # Z⊙M grid (bottom-mid)
x_mod = 6.5          # model box centre
x_Zh  = 8.2          # Ẑ grid (bottom-right)
x_op2 = 10.4         # ⊙  (combine)
x_op3 = 11.0         # ⊕  (add)
x_Zi  = 11.5         # Z_imputed grid (top-right)

# ════════════════════════════════════════════
#  Draw grids
# ════════════════════════════════════════════
grid(ax, x_Z,  y_top, OBS, 'data',    None)
grid(ax, x_M,  y_bot, OBS, 'mask',    None)
grid(ax, x_ZM, y_bot, OBS, 'masked',  None)
grid(ax, x_Zh, y_bot, OBS, 'output',  None)
grid(ax, x_Zi, y_top, OBS, 'imputed', None)

# ── Grid labels ──
lfs = 9; lc = '#444'
ax.text(x_Z  + GW/2, y_top + GH + 0.18, 'Z  (observed + missing)',
        ha='center', fontsize=lfs, color=lc, fontstyle='italic')
ax.text(x_M  + GW/2, y_bot - 0.20, 'M  (mask)',
        ha='center', fontsize=lfs, color=lc, fontstyle='italic', va='top')
ax.text(x_ZM + GW/2, y_bot - 0.20, 'Z ⊙ M',
        ha='center', fontsize=lfs, color=lc, fontstyle='italic', va='top')
ax.text(x_Zh + GW/2, y_bot - 0.20, 'Ẑ  (prediction)',
        ha='center', fontsize=lfs, color=lc, fontstyle='italic', va='top')
ax.text(x_Zi + GW/2, y_top + GH + 0.18, 'Z_imp  (imputed)',
        ha='center', fontsize=lfs, color=lc, fontstyle='italic')

# ── "~" between Z and M ──
ax.text(x_Z + GW/2, (y_top + y_bot + GH) / 2, '~',
        ha='center', va='center', fontsize=22, color='#555',
        fontweight='bold')

# ════════════════════════════════════════════
#  Model box  ε_θ
# ════════════════════════════════════════════
mw, mh = 1.8, 2.6
mx = x_mod - mw/2
my = yc_bot - 0.2
ax.add_patch(FancyBboxPatch((mx, my), mw, mh,
             boxstyle='round,pad=0.15', fc=C_MBG, ec=C_MBD,
             lw=1.5, zorder=2))
ax.text(x_mod, my + mh/2 + 0.15, 'ε', ha='center', va='center',
        fontsize=28, fontweight='bold', color='#333', zorder=3,
        fontstyle='italic')
ax.text(x_mod, my + mh/2 - 0.45, 'θ', ha='center', va='center',
        fontsize=14, color='#555', zorder=3, fontstyle='italic')

# ════════════════════════════════════════════
#  Arrows & connections
# ════════════════════════════════════════════
G = 0.12

# ── Bottom row flow ──
# M → ⊙
op(ax, x_op1, yc_bot, '⊙')
ar(ax, x_M + GW + G, yc_bot, x_op1 - 0.22 - G, yc_bot)

# ⊙ → Z⊙M
ar(ax, x_op1 + 0.22 + G, yc_bot, x_ZM - G, yc_bot)

# Z⊙M → model
ar(ax, x_ZM + GW + G, yc_bot, mx - G, yc_bot + 0.3)

# Z → ⊙ (downward: observed data goes into ⊙)
ln(ax, x_Z + GW/2 + 0.4, y_top, x_Z + GW/2 + 0.4, yc_bot + 0.8)
ar(ax, x_Z + GW/2 + 0.4, yc_bot + 0.8, x_op1, yc_bot + 0.22 + G)

# model → Ẑ
ar(ax, mx + mw + G, yc_bot + 0.3, x_Zh - G, yc_bot)

# ── Top row: Z → model (via dashed arrow showing conceptual flow) ──
ar(ax, x_Z + GW + G, yc_top, mx - G, my + mh - 0.3, lw=1.0, c='#888')

# model → Z_imp (dashed to show conceptual)
ar(ax, mx + mw + G, my + mh - 0.3, x_Zi - G, yc_top, lw=1.0, c='#888')

# ── Combine operations on the right ──
# Ẑ → ⊙(1-M)
op(ax, x_op2, yc_bot, '⊙')
ar(ax, x_Zh + GW + G, yc_bot, x_op2 - 0.22 - G, yc_bot)
ax.text(x_op2, yc_bot - 0.40, '(1−M)',
        ha='center', fontsize=8, color='#555', fontstyle='italic')

# ⊕  (combine observed + imputed)
op(ax, x_op3, yc_top - 0.6, '⊕', fs=12)

# ⊙(1-M) → ⊕
ar(ax, x_op2, yc_bot + 0.22 + G, x_op3, yc_top - 0.6 - 0.22 - G)

# M⊙Z (observed portion) → ⊕ from top-left Z grid
# Draw L-shape: from Z grid right edge → across top → down to ⊕
route_y = y_top + GH + 0.55
ln(ax, x_Z + GW, yc_top + 0.3, x_Z + GW + 0.15, route_y,
   c='#3498db', lw=1.0, ls='--')
ln(ax, x_Z + GW + 0.15, route_y, x_op3, route_y,
   c='#3498db', lw=1.0, ls='--')
ar(ax, x_op3, route_y, x_op3, yc_top - 0.6 + 0.22 + G,
   c='#3498db', lw=1.0)
ax.text((x_Z + GW + x_op3) / 2, route_y + 0.18, 'M ⊙ Z  (keep observed)',
        ha='center', fontsize=7.5, color='#3498db', fontstyle='italic')

# ⊕ → Z_imputed
ar(ax, x_op3 + 0.22 + G, yc_top - 0.6, x_Zi - G, yc_top - 0.3)

# ════════════════════════════════════════════
#  L, K dimension labels (on the Z grid)
# ════════════════════════════════════════════
# L arrow (vertical, left of grid)
lx = x_Z - 0.15
ar(ax, lx, y_top + GH - 0.05, lx, y_top + 0.05, c='#666', lw=0.9)
ax.text(lx - 0.15, yc_top, 'L', ha='center', va='center',
        fontsize=11, fontstyle='italic', fontweight='bold', color='#666')

# K arrow (horizontal, below grid)
ky = y_top - 0.10
ar(ax, x_Z + 0.05, ky, x_Z + GW - 0.05, ky, c='#666', lw=0.9)
ax.text(x_Z + GW/2, ky - 0.22, 'K', ha='center', va='center',
        fontsize=11, fontstyle='italic', fontweight='bold', color='#666')

# ════════════════════════════════════════════
#  Legend box (bottom-right)
# ════════════════════════════════════════════
lgx, lgy = 8.0, -0.3
lgw, lgh = 4.8, 1.2
ax.add_patch(FancyBboxPatch((lgx, lgy), lgw, lgh,
             boxstyle='round,pad=0.08', fc='#FAFAFA', ec='#999',
             lw=0.8, zorder=2))

# legend items
s = 0.28  # swatch size
items = [
    (lgx + 0.25, lgy + 0.75, C_OBS,  None,  'Observed data'),
    (lgx + 0.25, lgy + 0.30, C_MASK, None,  'Mask'),
    (lgx + 2.55, lgy + 0.75, None,   'imp', 'Imputed data'),
    (lgx + 2.55, lgy + 0.30, C_MBG,  'eps', 'ε   Model'),
]
for ix, iy, fc, special, txt in items:
    if special == 'imp':
        ax.add_patch(Rectangle((ix, iy), s, s, fc='white', ec=C_BD, lw=0.6, zorder=3))
        ax.add_patch(Circle((ix + s/2, iy + s/2), s * 0.35,
                            fc='none', ec=C_OBS, lw=1.0,
                            ls=(0, (2.5, 2)), zorder=4))
    elif special == 'eps':
        ax.add_patch(FancyBboxPatch((ix, iy), s, s,
                     boxstyle='round,pad=0.03', fc=C_MBG, ec=C_MBD,
                     lw=0.8, zorder=3))
        ax.text(ix + s/2, iy + s/2, 'ε', ha='center', va='center',
                fontsize=10, fontweight='bold', fontstyle='italic',
                color='#333', zorder=4)
    else:
        ax.add_patch(Rectangle((ix, iy), s, s, fc=fc, ec=C_BD, lw=0.6, zorder=3))

    ax.text(ix + s + 0.15, iy + s/2, txt, va='center', fontsize=9,
            color='#333', zorder=3)

# ════════════════════════════════════════════
#  Equation at the very bottom
# ════════════════════════════════════════════
ax.text(6.5, -1.35,
        r'$\mathbf{Z}_{\mathrm{imp}} = M \odot Z \;+\; (1-M) \odot \hat{Z}_\theta$',
        ha='center', fontsize=12, color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.25', fc='#fafafa', ec='#bdc3c7', lw=0.7))

# ── Title ──
ax.set_title('Diffusion-based Conditional Imputation Process',
             fontsize=13, fontweight='bold', pad=14)

fig.tight_layout()
out = '/home/user/sssdtcn/Figures4paper'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}/fig4_imputation_process.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 4 saved.')
