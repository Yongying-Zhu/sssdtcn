"""
Architecture diagrams for Implicit-Explicit Diffusion Model paper.
v4: LEFT-TO-RIGHT horizontal flow matching reference paper style.
    - No text/dashed-border overlap
    - Labels clearly positioned in empty space
    - Colored background shading for groups
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np

# ── Colors ───────────────────────────────────────────────────
C_IN   = '#D5E8D4'; C_IN_E  = '#82B366'   # green
C_PR   = '#DAE8FC'; C_PR_E  = '#6C8EBF'   # blue
C_CV   = '#FFE6CC'; C_CV_E  = '#D79B00'   # orange
C_S4   = '#D5E8FC'; C_S4_E  = '#3A7BBF'   # indigo-blue
C_FU   = '#F8CECC'; C_FU_E  = '#B85450'   # red
C_RE   = '#E1D5E7'; C_RE_E  = '#9673A6'   # purple
C_TI   = '#FFF2CC'; C_TI_E  = '#D6B656'   # yellow
C_MK   = '#D5F5E3'; C_MK_E  = '#27AE60'   # mint
C_NM   = '#FFF2CC'; C_NM_E  = '#D6B656'   # yellow-norm
C_AC   = '#E8F8F5'; C_AC_E  = '#1ABC9C'   # teal
C_DR   = '#F2F3F4'; C_DR_E  = '#95A5A6'   # grey
C_WH   = '#FFFFFF'; C_BK    = '#2C3E50'

# Shading backgrounds
BG_IMP = '#FFF8E1'  # warm yellow for implicit region
BG_EXP = '#E3F2FD'  # light blue for explicit region

def B(ax, cx, cy, w, h, t, fc, ec, fs=9, fw='bold', tc='#1a1a1a',
      style='round,pad=0.06', lw=1.3):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle=style,
                 fc=fc,ec=ec,lw=lw,zorder=3))
    ax.text(cx,cy,t,ha='center',va='center',fontsize=fs,
            fontweight=fw,color=tc,zorder=4,linespacing=1.15)

def CC(ax, cx, cy, r, t, fc=C_WH, ec=C_BK, fs=10, lw=1.3):
    ax.add_patch(plt.Circle((cx,cy),r,fc=fc,ec=ec,lw=lw,zorder=3))
    ax.text(cx,cy,t,ha='center',va='center',fontsize=fs,
            fontweight='bold',color='#1a1a1a',zorder=4)

def AR(ax, x1,y1,x2,y2, c=C_BK, lw=1.2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->,head_width=0.2,head_length=0.15',
                        color=c,lw=lw,mutation_scale=12),zorder=2)

def LN(ax, x1,y1,x2,y2, c=C_BK, lw=1.2, ls='-'):
    ax.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=ls,zorder=2,
            solid_capstyle='round')

def shaded_rect(ax, x, y, w, h, fc, ec='none', alpha=0.35, lw=0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',
                 fc=fc,ec=ec,lw=lw,alpha=alpha,zorder=0))


# ════════════════════════════════════════════════════════════════
#  Figure 1  –  Overall Architecture  (LEFT → RIGHT)
# ════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1.0, 5.5)
    ax.axis('off')

    H = 0.50; bw = 1.6  # box height, default width
    yc = 2.2             # center y for main flow
    yt = 3.5             # top branch (implicit)
    yb = 0.9             # bottom branch (explicit)

    # ── x positions (left to right) ──
    x_in    = 0.6
    x_ipr   = 2.5;  x_me = 2.5   # input proj (top) / mask embed (bottom)
    x_add1  = 4.2                 # + (proj + mask)
    x_imp   = 6.0                 # implicit module
    x_exp   = 6.0                 # explicit module
    x_fu    = 8.2                 # gated fusion
    x_add2  = 9.8                 # + (time)
    x_te    = 9.8                 # time embed (above)
    x_res   = 11.2                # residual blocks
    x_op    = 12.6                # output proj
    x_out   = 14.0                # output

    # ── shaded background regions ──
    shaded_rect(ax, 4.8, yt-H*0.7-0.15, 2.4, H*1.4+0.3, BG_IMP, alpha=0.4)
    shaded_rect(ax, 4.8, yb-H*0.7-0.15, 2.4, H*1.4+0.3, BG_EXP, alpha=0.4)

    # ── boxes ──
    B(ax, x_in,  yc, 1.4, H, 'Input\n(x, mask, t)', C_IN, C_IN_E, fs=8)

    B(ax, x_ipr, yt-0.15, 1.5, H, 'Input\nProjection', C_PR, C_PR_E, fs=8)
    B(ax, x_me,  yb+0.15, 1.5, H, 'Mask\nEmbedding',   C_MK, C_MK_E, fs=8)

    CC(ax, x_add1, yc, 0.18, '+')

    B(ax, x_imp, yt-0.15, 1.8, H*1.2, 'Implicit Module\n(Dilated Conv)', C_CV, C_CV_E, fs=8)
    B(ax, x_exp, yb+0.15, 1.8, H*1.2, 'Explicit Module\n(S4 Layer ×2)',  C_S4, C_S4_E, fs=8)

    B(ax, x_fu,  yc, 1.5, H*1.1, 'Gated\nFusion', C_FU, C_FU_E, fs=9)

    CC(ax, x_add2, yc, 0.18, '+')
    B(ax, x_te, yc+1.1, 1.2, H*0.9, 'Time\nEmbed', C_TI, C_TI_E, fs=8)

    B(ax, x_res, yc, 1.4, H*1.1, 'Residual\nBlocks ×6', C_RE, C_RE_E, fs=8)
    B(ax, x_op,  yc, 1.1, H, 'Output\nProj', C_PR, C_PR_E, fs=8)
    B(ax, x_out, yc, 0.8, H, 'Output', C_IN, C_IN_E, fs=8)

    # ── arrows (all left→right or vertical, no crossing) ──

    # Input forks up/down
    fork_x = x_in + 1.4/2 + 0.15
    LN(ax, x_in+1.4/2, yc, fork_x, yc)
    LN(ax, fork_x, yc, fork_x, yt-0.15)
    AR(ax, fork_x, yt-0.15, x_ipr-1.5/2, yt-0.15)
    LN(ax, fork_x, yc, fork_x, yb+0.15)
    AR(ax, fork_x, yb+0.15, x_me-1.5/2, yb+0.15)

    # Input Proj → (+)
    AR(ax, x_ipr+1.5/2, yt-0.15, x_add1, yc+0.18)
    # Mask Embed → (+)
    AR(ax, x_me+1.5/2, yb+0.15, x_add1, yc-0.18)

    # (+) splits to branches
    split_x = x_add1 + 0.18 + 0.15
    LN(ax, x_add1+0.18, yc, split_x, yc)
    LN(ax, split_x, yc, split_x, yt-0.15)
    AR(ax, split_x, yt-0.15, x_imp-1.8/2, yt-0.15)
    LN(ax, split_x, yc, split_x, yb+0.15)
    AR(ax, split_x, yb+0.15, x_exp-1.8/2, yb+0.15)

    # Branches → Gated Fusion
    merge_x = x_fu - 1.5/2 - 0.15
    LN(ax, x_imp+1.8/2, yt-0.15, merge_x, yt-0.15)
    LN(ax, merge_x, yt-0.15, merge_x, yc)
    AR(ax, merge_x, yc, x_fu-1.5/2, yc)

    LN(ax, x_exp+1.8/2, yb+0.15, merge_x, yb+0.15)
    LN(ax, merge_x, yb+0.15, merge_x, yc)

    # Fusion → (+time)
    AR(ax, x_fu+1.5/2, yc, x_add2-0.18, yc)
    # Time Embed → (+time)   (from above, clean vertical)
    AR(ax, x_te, yc+1.1-H*0.9/2, x_add2, yc+0.18)
    # (+time) → Residual → Output Proj → Output
    AR(ax, x_add2+0.18, yc, x_res-1.4/2, yc)
    AR(ax, x_res+1.4/2, yc, x_op-1.1/2, yc)
    AR(ax, x_op+1.1/2, yc, x_out-0.8/2, yc)

    # ── labels (placed clearly in empty space, no overlap) ──
    ax.text(x_imp, yt-0.15+H*1.2/2+0.18, 'Implicit Branch', ha='center',
            fontsize=7.5, color='#E65100', fontstyle='italic', fontweight='bold')
    ax.text(x_exp, yb+0.15-H*1.2/2-0.18, 'Explicit Branch', ha='center',
            fontsize=7.5, color='#1565C0', fontstyle='italic', fontweight='bold')

    ax.set_title('(a) Overall Architecture of Implicit-Explicit Diffusion Model',
                 fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'/home/user/sssdtcn/Figures4paper/fig1_overall_architecture.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Fig 1 done')


# ════════════════════════════════════════════════════════════════
#  Figure 2  –  Implicit Module  (LEFT → RIGHT, sequential chain)
# ════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(-0.3, 15.2)
    ax.set_ylim(-1.0, 4.5)
    ax.axis('off')

    H = 0.42
    yc = 1.8   # center line

    # ── y positions within each block (top→bottom) ──
    y_conv = yc + 0.58
    y_gelu = yc
    y_drop = yc - 0.58
    y_plus = y_drop - 0.38  # (+) below dropout
    y_route = y_plus - 0.25  # routing level between blocks

    # ── x positions ──
    x_in  = 0.5
    x_li  = 1.9
    conv_step = 1.65
    x_c0 = 3.7
    dilations = [1, 2, 4, 8, 16, 32]
    x_convs = [x_c0 + i * conv_step for i in range(6)]
    x_lo  = x_convs[-1] + conv_step + 0.1
    x_out = x_lo + 1.5
    bw = 1.15  # block width

    # ── shaded background ──
    shaded_rect(ax, x_c0-bw/2-0.15, y_plus-0.3,
                x_convs[-1]-x_c0+bw+0.3, y_conv-y_plus+0.50/2+0.55,
                BG_IMP, alpha=0.35)

    # ── Input / Linear In ──
    B(ax, x_in, yc, 0.9, H, 'Input', C_IN, C_IN_E, fs=9)
    B(ax, x_li, yc, 1.1, H, 'Linear\n(D→H)', C_PR, C_PR_E, fs=8)
    AR(ax, x_in+0.9/2, yc, x_li-1.1/2, yc)

    # ── 6 Conv blocks ──
    for i, (xc, d) in enumerate(zip(x_convs, dilations)):
        # Conv (orange)
        B(ax, xc, y_conv, bw, 0.48, f'Conv d={d}', C_CV, C_CV_E, fs=7.5)
        # GELU (teal)
        B(ax, xc, y_gelu, bw, 0.38, 'GELU', C_AC, C_AC_E, fs=7, fw='normal')
        # Dropout (grey)
        B(ax, xc, y_drop, bw, 0.38, 'Dropout', C_DR, C_DR_E, fs=7, fw='normal')

        # Internal arrows (top → bottom)
        AR(ax, xc, y_conv-0.48/2, xc, y_gelu+0.38/2)
        AR(ax, xc, y_gelu-0.38/2, xc, y_drop+0.38/2)

        # (+) circle
        CC(ax, xc, y_plus, 0.12, '+', fs=7)
        AR(ax, xc, y_drop-0.38/2, xc, y_plus+0.12)

        # Residual bypass on right side (outside block, no crossing)
        rx = xc + bw/2 + 0.12
        rt = y_conv + 0.48/2 + 0.03
        LN(ax, xc+bw/2, rt, rx, rt, c='#95a5a6', lw=0.9)
        LN(ax, rx, rt, rx, y_plus, c='#95a5a6', lw=0.9)
        AR(ax, rx, y_plus, xc+0.12, y_plus, c='#95a5a6', lw=0.9)

        # alpha label (above block, clear space)
        ax.text(xc, y_conv+0.48/2+0.12, f'α{chr(8321+i)}',
                fontsize=7, color='#c0392c', ha='center', fontstyle='italic')

    # ── Inter-block connections (L-shaped, clean right-angle paths) ──

    # Linear → first block: horizontal at yc, then up to Conv top
    LN(ax, x_li+1.1/2, yc, x_convs[0]-bw/2-0.08, yc)
    LN(ax, x_convs[0]-bw/2-0.08, yc, x_convs[0]-bw/2-0.08, y_conv)
    AR(ax, x_convs[0]-bw/2-0.08, y_conv, x_convs[0]-bw/2, y_conv)

    # Block i (+) → Block i+1 Conv: down to route_y, right, up to Conv
    for i in range(len(dilations)-1):
        x_from = x_convs[i]
        x_to   = x_convs[i+1]
        # (+) down to routing level
        LN(ax, x_from, y_plus-0.12, x_from, y_route)
        # right to next block x
        LN(ax, x_from, y_route, x_to-bw/2-0.08, y_route)
        # up to Conv top level
        LN(ax, x_to-bw/2-0.08, y_route, x_to-bw/2-0.08, y_conv)
        AR(ax, x_to-bw/2-0.08, y_conv, x_to-bw/2, y_conv)

    # Last block (+) → Linear Out: down to route_y, right, up to Linear
    x_last = x_convs[-1]
    LN(ax, x_last, y_plus-0.12, x_last, y_route)
    LN(ax, x_last, y_route, x_lo-1.1/2-0.08, y_route)
    LN(ax, x_lo-1.1/2-0.08, y_route, x_lo-1.1/2-0.08, yc)
    AR(ax, x_lo-1.1/2-0.08, yc, x_lo-1.1/2, yc)

    # ── Linear Out / Output ──
    B(ax, x_lo, yc, 1.1, H, 'Linear\n(H→H)', C_PR, C_PR_E, fs=8)
    B(ax, x_out, yc, 0.9, H, 'Output', C_IN, C_IN_E, fs=9)
    AR(ax, x_lo+1.1/2, yc, x_out-0.9/2, yc)

    # ── Global skip (below everything, clearly separated) ──
    skip_y = y_route - 0.4
    LN(ax, x_li, yc-H/2, x_li, skip_y, c='#3498db', lw=1.0, ls='--')
    LN(ax, x_li, skip_y, x_lo, skip_y, c='#3498db', lw=1.0, ls='--')
    LN(ax, x_lo, skip_y, x_lo, yc-H/2, c='#3498db', lw=1.0, ls='--')
    # (+) at Linear Out for global skip
    CC(ax, x_lo-0.25, yc-H/2-0.05, 0.10, '+', fs=6)

    ax.text((x_li+x_lo)/2, skip_y-0.15, 'Global Residual Connection',
            ha='center', fontsize=7.5, color='#3498db', fontstyle='italic')

    # ── Title label above conv region (clear of alpha labels) ──
    ax.text((x_convs[0]+x_convs[-1])/2, y_conv+0.48/2+0.38,
            'Multi-scale Dilated Causal Convolution (d = 1, 2, 4, 8, 16, 32)',
            ha='center', fontsize=8.5, color='#E65100', fontweight='bold')

    # Formula at bottom
    ax.text((x_convs[0]+x_convs[-1])/2, skip_y-0.45,
            r'$h_i = \alpha_i \cdot f_i(h_{i-1}) + (1-\alpha_i) \cdot h_{i-1}$',
            ha='center', fontsize=9, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', fc='#fafafa', ec='#bdc3c7', lw=0.7))

    ax.set_title('(b) Implicit Module: Multi-scale Dilated Causal Convolution',
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'/home/user/sssdtcn/Figures4paper/fig2_implicit_module.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Fig 2 done')


# ════════════════════════════════════════════════════════════════
#  Figure 3  –  Explicit Module  (reference-style: left module + right S4 detail)
# ════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.8, 5.5)
    ax.axis('off')

    H = 0.45

    # =============================================
    # LEFT PANEL: Module overview (left→right)
    # Following reference: Input → S4 → (+) → S4 → Output
    # =============================================
    yc_l = 2.5  # center y

    x_lin  = 0.5
    x_s41  = 2.3
    x_add1 = 3.6
    x_s42  = 4.9
    x_lout = 6.5

    # Shaded background for explicit module
    shaded_rect(ax, 1.4, yc_l-1.1, 4.4, 2.2, BG_EXP, alpha=0.35)

    B(ax, x_lin, yc_l, 1.0, H, 'Input', C_IN, C_IN_E, fs=9)

    # S4 Block 1 (tall box like reference)
    B(ax, x_s41, yc_l, 1.2, H*2.2, 'S4\nBlock', C_S4, C_S4_E, fs=9)
    ax.text(x_s41, yc_l-H*2.2/2-0.15, '1', ha='center', fontsize=8,
            color='#1565C0', fontweight='bold')

    CC(ax, x_add1, yc_l, 0.15, '+', fs=8)

    # S4 Block 2
    B(ax, x_s42, yc_l, 1.2, H*2.2, 'S4\nBlock', C_S4, C_S4_E, fs=9)
    ax.text(x_s42, yc_l-H*2.2/2-0.15, '2', ha='center', fontsize=8,
            color='#1565C0', fontweight='bold')

    B(ax, x_lout, yc_l, 1.2, H, 'Output', C_IN, C_IN_E, fs=9)

    # Arrows (left → right)
    AR(ax, x_lin+1.0/2, yc_l, x_s41-1.2/2, yc_l)
    AR(ax, x_s41+1.2/2, yc_l, x_add1-0.15, yc_l)
    AR(ax, x_add1+0.15, yc_l, x_s42-1.2/2, yc_l)
    AR(ax, x_s42+1.2/2, yc_l, x_lout-1.2/2, yc_l)

    # Residual skip from Input to (+) (below, no crossing)
    skip_y_l = yc_l - 1.3
    LN(ax, x_lin+1.0/2, yc_l-H/2, x_lin+1.0/2, skip_y_l, c='#3498db', lw=1.0)
    LN(ax, x_lin+1.0/2, skip_y_l, x_add1, skip_y_l, c='#3498db', lw=1.0)
    AR(ax, x_add1, skip_y_l, x_add1, yc_l-0.15, c='#3498db', lw=1.0)
    ax.text((x_lin+1.0/2+x_add1)/2, skip_y_l-0.15, 'skip',
            ha='center', fontsize=7, color='#3498db', fontstyle='italic')

    # Module label (clearly below the shaded region)
    ax.text((x_s41+x_s42)/2, yc_l+H*2.2/2+0.25,
            'Explicit Information Extraction Module',
            ha='center', fontsize=9, color='#1565C0', fontweight='bold')

    # =============================================
    # SEPARATOR (dashed line connecting to detail)
    # =============================================
    sep_x = 7.5
    # Dashed curved line from S4 Block to detail
    LN(ax, x_s42+1.2/2+0.1, yc_l+H*2.2/2, sep_x, 4.5, c='#95a5a6', lw=0.8, ls='--')
    LN(ax, sep_x, 4.5, 8.2, 4.5, c='#95a5a6', lw=0.8, ls='--')

    # =============================================
    # RIGHT PANEL: S4 Block detail (BOTTOM → TOP, matching reference)
    # Reference order: S4 Layer → GELU → Dropout → (+skip) → LayerNorm
    # =============================================
    rx = 9.8; rw = 1.8

    ry_s4   = 0.5    # bottom: S4 Layer
    ry_gelu = 1.35
    ry_drop = 2.1
    ry_add  = 2.85   # (+) with skip
    ry_ln   = 3.7    # top: LayerNorm

    # Yellow/orange background (like reference)
    shaded_rect(ax, rx-rw/2-0.25, ry_s4-H/2-0.25, rw+0.5, ry_ln-ry_s4+H+0.5,
                '#FFF8E1', ec='#D6B656', alpha=0.5)
    # Border
    ax.add_patch(FancyBboxPatch((rx-rw/2-0.25, ry_s4-H/2-0.25),
                 rw+0.5, ry_ln-ry_s4+H+0.5,
                 boxstyle='round,pad=0.08', fc='none', ec='#D6B656',
                 lw=1.0, ls='-', zorder=1))

    B(ax, rx, ry_s4,   rw, H, 'S4 Layer', C_S4, C_S4_E, fs=9)
    B(ax, rx, ry_gelu, rw*0.7, H*0.8, 'GELU', C_AC, C_AC_E, fs=8, fw='normal')
    B(ax, rx, ry_drop, rw*0.8, H*0.8, 'Dropout', C_DR, C_DR_E, fs=8, fw='normal')
    CC(ax, rx, ry_add, 0.14, '+', fs=8)
    B(ax, rx, ry_ln,   rw, H, 'LayerNorm', C_NM, C_NM_E, fs=9)

    # Arrows (bottom → top)
    AR(ax, rx, ry_s4+H/2,       rx, ry_gelu-H*0.8/2)
    AR(ax, rx, ry_gelu+H*0.8/2, rx, ry_drop-H*0.8/2)
    AR(ax, rx, ry_drop+H*0.8/2, rx, ry_add-0.14)
    AR(ax, rx, ry_add+0.14,     rx, ry_ln-H/2)

    # Skip connection on RIGHT side (outside border, no crossing)
    skx = rx + rw/2 + 0.55
    LN(ax, rx+rw/2, ry_s4, skx, ry_s4, c='#2c3e50', lw=1.0)
    LN(ax, skx, ry_s4, skx, ry_add, c='#2c3e50', lw=1.0)
    AR(ax, skx, ry_add, rx+0.14, ry_add, c='#2c3e50', lw=1.0)

    # "Skip" label (clearly to the right, no overlap with border)
    ax.text(skx+0.15, (ry_s4+ry_add)/2, 'Skip', fontsize=8.5,
            color='#2c3e50', rotation=90, va='center', ha='left', fontweight='bold')

    # "S4 Block" label clearly below the bordered region
    ax.text(rx, ry_s4-H/2-0.42, 'S4 Block', ha='center', fontsize=10,
            color='#D6B656', fontweight='bold')

    # ── SSM equation at very bottom, clearly separated ──
    ax.text(6.0, -0.55,
            r'$\mathbf{s}_{t+1} = \bar{A}\,\mathbf{s}_t + \bar{B}\,\mathbf{x}_t'
            r',\quad \mathbf{y}_t = C\,\mathbf{s}_t + D\,\mathbf{x}_t$',
            ha='center', fontsize=9.5, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', fc='#fafafa', ec='#bdc3c7', lw=0.7))

    ax.set_title('(c) Explicit Module: Structured State Space (S4) Layer',
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'/home/user/sssdtcn/Figures4paper/fig3_explicit_module.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Fig 3 done')


if __name__ == '__main__':
    print('Generating v4 figures (horizontal, reference-style)...')
    fig1(); fig2(); fig3()
    print('Done – all saved to Figures4paper/')
