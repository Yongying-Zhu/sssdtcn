"""
Figure 4: Diffusion-based Time Series Imputation Process.
v3: Compact vertical layout (no centre whitespace), L-shaped arrow routing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np

# ── Grid config ──
ROWS, COLS = 5, 4
CS = 0.28
GW = COLS * CS    # 1.12
GH = ROWS * CS    # 1.40

OBS = np.array([
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
])

# ── Claude orange palette ──
C_OBS  = '#E8853D';  C_MASK = '#9E9E9E';  C_WH = '#FFFFFF'
C_BD   = '#666666';  C_ARR  = '#2C3E50'
C_MBG  = '#F0E2D0';  C_MBD  = '#C4956A'


def grid(ax, x0, y0, m, mode):
    for r in range(ROWS):
        for c in range(COLS):
            x, y = x0 + c*CS, y0 + (ROWS-1-r)*CS
            v = m[r, c]
            if mode in ('data','masked'): fc = C_OBS if v else C_WH
            elif mode == 'mask':          fc = C_MASK if v else C_WH
            elif mode == 'output':        fc = C_OBS
            elif mode == 'imputed':       fc = C_OBS if v else C_WH
            else:                         fc = C_WH
            ax.add_patch(Rectangle((x,y),CS,CS,fc=fc,ec=C_BD,lw=0.5,zorder=2))
            if mode=='imputed' and v==0:
                ax.add_patch(Circle((x+CS/2,y+CS/2),CS*0.30,
                    fc='none',ec=C_OBS,lw=1.0,ls=(0,(2,2)),zorder=3))

def OP(ax,x,y,t,fs=11,r=0.16):
    ax.add_patch(Circle((x,y),r,fc='white',ec=C_BD,lw=1.0,zorder=3))
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,
            fontweight='bold',color='#333',zorder=4)

def AR(ax,x1,y1,x2,y2,lw=1.1,c=C_ARR):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->,head_width=0.16,head_length=0.12',
                        color=c,lw=lw,mutation_scale=10),zorder=2)

def LN(ax,x1,y1,x2,y2,lw=1.0,c=C_ARR,ls='-'):
    ax.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=ls,zorder=2,
            solid_capstyle='round')

# ══════════════════════════════════════════════════════════════
#  COMPACT LAYOUT  –  two tight rows, minimal whitespace
#
#  Row 1 (top):   [Z]       [ε_θ]      [Z_imp]
#                  ~  ↘       ↑↓        ↗  ↑
#  Row 2 (bot):   [M]  ⊙  [Z⊙M]      [Ẑ]  ⊙ ⊕
#
#  Gap between rows ≈ 0.45  (no dead space)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.set_xlim(-0.55, 6.5)
ax.set_ylim(-1.2, 5.4)
ax.axis('off')
ax.set_aspect('equal')

G = 0.07

# ── Row positions (tight!) ──
y1 = 3.10       # row-1 grid base  (Z, ε, Z_imp)
y2 = 1.25       # row-2 grid base  (M, Z⊙M, Ẑ)
# gap = y1 - (y2 + GH) = 3.10 - 2.65 = 0.45
yc1 = y1 + GH/2   # 3.80
yc2 = y2 + GH/2   # 1.95

# ── x positions ──
x_Z   = 0.0                        # Z  (top-left)
x_M   = 0.0                        # M  (bot-left, below Z)
x_op1 = GW + 0.30                  # ⊙  at 1.42
x_ZM  = x_op1 + 0.42               # Z⊙M  at 1.84
x_eps = x_ZM + GW + 0.30           # ε left edge  at 3.26
x_Zh  = x_eps + GW + 0.40          # Ẑ  at 4.78
x_op2 = x_Zh + GW + 0.25           # ⊙(1-M)  at 6.15
x_Zi  = 4.8                        # Z_imp  (top-right, above Ẑ area)

# ═══════════════ GRIDS (all same size) ═══════════════
grid(ax, x_Z,  y1, OBS, 'data')
grid(ax, x_M,  y2, OBS, 'mask')
grid(ax, x_ZM, y2, OBS, 'masked')
grid(ax, x_Zh, y2, OBS, 'output')
grid(ax, x_Zi, y1, OBS, 'imputed')

# ═══════════════ ε_θ MODEL BOX (same size as grid) ═══════════════
mw, mh = GW + 0.06, GH
ax.add_patch(FancyBboxPatch((x_eps, y1), mw, mh,
             boxstyle='round,pad=0.08', fc=C_MBG, ec=C_MBD,
             lw=1.2, zorder=2))
mcx = x_eps + mw/2
ax.text(mcx, yc1+0.05, 'ε', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#333', fontstyle='italic', zorder=3)
ax.text(mcx+0.18, yc1-0.22, 'θ', ha='center', va='center',
        fontsize=9, color='#666', fontstyle='italic', zorder=3)

# ═══════════════ LABELS ═══════════════
lfs = 7.5; lc = '#444'
ax.text(x_Z +GW/2, y1+GH+0.12, 'Z (observed+missing)',
        ha='center',fontsize=lfs,color=lc,fontstyle='italic')
ax.text(x_Zi+GW/2, y1+GH+0.12, 'Z_imp (imputed)',
        ha='center',fontsize=lfs,color=lc,fontstyle='italic')
ax.text(x_M +GW/2, y2-0.12, 'M (mask)',
        ha='center',fontsize=lfs,color=lc,fontstyle='italic',va='top')
ax.text(x_ZM+GW/2, y2-0.12, 'Z ⊙ M',
        ha='center',fontsize=lfs,color=lc,fontstyle='italic',va='top')
ax.text(x_Zh+GW/2, y2-0.12, 'Ẑ (prediction)',
        ha='center',fontsize=lfs,color=lc,fontstyle='italic',va='top')

# ── "~" between Z and M ──
ax.text(x_Z+GW/2, (y1+y2+GH)/2, '~', ha='center', va='center',
        fontsize=16, color='#555', fontweight='bold')

# ── L, K labels ──
lx = x_Z - 0.10
AR(ax, lx, y1+GH-0.04, lx, y1+0.04, c='#888', lw=0.7)
ax.text(lx-0.12, yc1, 'L', ha='center', va='center',
        fontsize=9, fontstyle='italic', fontweight='bold', color='#888')
ky = y2 - 0.28
AR(ax, x_M+0.04, ky, x_M+GW-0.04, ky, c='#888', lw=0.7)
ax.text(x_M+GW/2, ky-0.15, 'K', ha='center', fontsize=9,
        fontstyle='italic', fontweight='bold', color='#888')

# ═══════════════ ARROWS ═══════════════

# ── Row 2 horizontal flow ──
# M → ⊙
OP(ax, x_op1, yc2, '⊙')
AR(ax, x_M+GW+G, yc2, x_op1-0.16-G, yc2)

# Z → ⊙  (data feeds from above into masking op)
LN(ax, x_Z+GW/2, y1, x_Z+GW/2, yc2+0.40)
AR(ax, x_Z+GW/2, yc2+0.40, x_op1, yc2+0.16+G)

# ⊙ → Z⊙M
AR(ax, x_op1+0.16+G, yc2, x_ZM-G, yc2)

# Z⊙M → ε_θ  (upward from row2 to row1)
mid_x_zm = x_ZM + GW/2
LN(ax, mid_x_zm, y2+GH, mid_x_zm, y1-G-0.02)
AR(ax, mid_x_zm, y1-G-0.02, x_eps+0.10, y1-G)

# ε_θ → Ẑ  (downward from row1 to row2)
mid_x_zh = x_Zh + GW/2
LN(ax, x_eps+mw-0.10, y1-G, mid_x_zh, y1-G-0.02)
AR(ax, mid_x_zh, y1-G-0.02, mid_x_zh, y2+GH+G)

# Ẑ → ⊙(1-M)
OP(ax, x_op2, yc2, '⊙')
AR(ax, x_Zh+GW+G, yc2, x_op2-0.16-G, yc2)
ax.text(x_op2, yc2-0.28, '(1−M)', ha='center',
        fontsize=6.5, color='#555', fontstyle='italic')

# ── ⊕ (combine) ──
op3_x = x_op2
op3_y = yc2 + 0.65
OP(ax, op3_x, op3_y, '⊕', fs=9)

# ⊙(1-M) → ⊕  (upward)
AR(ax, x_op2, yc2+0.16+G, op3_x, op3_y-0.16-G)

# ── ⊕ → Z_imp  (L-shaped: right → up → left into Z_imp) ──
route_r = x_Zi + GW + 0.22          # right-side routing column
LN(ax, op3_x+0.16, op3_y, route_r, op3_y)       # horizontal right
LN(ax, route_r, op3_y, route_r, yc1)              # vertical up
AR(ax, route_r, yc1, x_Zi+GW+G, yc1)             # arrow left into Z_imp

# ── M⊙Z (keep observed): dashed path from Z across top to ⊕ ──
route_top = y1 + GH + 0.30
LN(ax, x_Z+GW, yc1+0.15, x_Z+GW+0.06, route_top,
   c='#B86E2A', lw=0.8, ls='--')
LN(ax, x_Z+GW+0.06, route_top, op3_x, route_top,
   c='#B86E2A', lw=0.8, ls='--')
AR(ax, op3_x, route_top, op3_x, op3_y+0.16+G,
   c='#B86E2A', lw=0.8)
ax.text((x_Z+GW+0.15+op3_x)/2, route_top+0.12,
        'M ⊙ Z (keep observed)', ha='center', fontsize=6,
        color='#B86E2A', fontstyle='italic')

# ═══════════════ LEGEND ═══════════════
lgx, lgy = 2.0, -0.25
lgw, lgh = 3.2, 0.80
ax.add_patch(FancyBboxPatch((lgx,lgy),lgw,lgh,
             boxstyle='round,pad=0.05',fc='#FAFAFA',ec='#AAA',lw=0.6,zorder=2))
sw = 0.20
items = [
    (lgx+0.15, lgy+0.45, C_OBS,  None,  'Observed data'),
    (lgx+0.15, lgy+0.12, C_MASK, None,  'Mask'),
    (lgx+1.65, lgy+0.45, None,   'imp', 'Imputed data'),
    (lgx+1.65, lgy+0.12, C_MBG,  'eps', 'ε  Model'),
]
for ix, iy, fc, sp, txt in items:
    if sp == 'imp':
        ax.add_patch(Rectangle((ix,iy),sw,sw,fc='white',ec=C_BD,lw=0.4,zorder=3))
        ax.add_patch(Circle((ix+sw/2,iy+sw/2),sw*0.35,
                            fc='none',ec=C_OBS,lw=0.8,ls=(0,(2,2)),zorder=4))
    elif sp == 'eps':
        ax.add_patch(FancyBboxPatch((ix,iy),sw,sw,
                     boxstyle='round,pad=0.02',fc=C_MBG,ec=C_MBD,lw=0.6,zorder=3))
        ax.text(ix+sw/2,iy+sw/2,'ε',ha='center',va='center',
                fontsize=8,fontweight='bold',fontstyle='italic',color='#333',zorder=4)
    else:
        ax.add_patch(Rectangle((ix,iy),sw,sw,fc=fc,ec=C_BD,lw=0.4,zorder=3))
    ax.text(ix+sw+0.08,iy+sw/2,txt,va='center',fontsize=7.5,color='#333',zorder=3)

# ═══════════════ FORMULA ═══════════════
ax.text(3.5, -0.85,
        r'$\mathbf{Z}_{\mathrm{imp}} = M \odot Z + (1\!-\!M) \odot \hat{Z}_\theta$',
        ha='center', fontsize=9, color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.18',fc='#fafafa',ec='#bdc3c7',lw=0.5))

fig.tight_layout()
out = '/home/user/sssdtcn/Figures4paper'
for ext in ('png','pdf'):
    fig.savefig(f'{out}/fig4_imputation_process.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig 4 (v3) saved.')
