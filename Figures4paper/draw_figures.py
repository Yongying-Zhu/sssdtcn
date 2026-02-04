"""
Architecture diagrams for Implicit-Explicit Diffusion Model paper.
v6: Fix merge routing horizontal segments, Output Proj spacing, conv entry arrow stems.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colors ───────────────────────────────────────────────────
C_IN   = '#D5E8D4'; C_IN_E  = '#82B366'
C_PR   = '#DAE8FC'; C_PR_E  = '#6C8EBF'
C_CV   = '#FFE6CC'; C_CV_E  = '#D79B00'
C_S4   = '#D5E8FC'; C_S4_E  = '#3A7BBF'
C_FU   = '#F8CECC'; C_FU_E  = '#B85450'
C_RE   = '#E1D5E7'; C_RE_E  = '#9673A6'
C_TI   = '#FFF2CC'; C_TI_E  = '#D6B656'
C_MK   = '#D5F5E3'; C_MK_E  = '#27AE60'
C_NM   = '#FFF2CC'; C_NM_E  = '#D6B656'
C_AC   = '#E8F8F5'; C_AC_E  = '#1ABC9C'
C_DR   = '#F2F3F4'; C_DR_E  = '#95A5A6'
C_WH   = '#FFFFFF'; C_BK    = '#2C3E50'
BG_IMP = '#FFF8E1'
BG_EXP = '#E3F2FD'

def B(ax, cx, cy, w, h, t, fc, ec, fs=9, fw='bold', tc='#1a1a1a'):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle='round,pad=0.06',fc=fc,ec=ec,lw=1.3,zorder=3))
    ax.text(cx,cy,t,ha='center',va='center',fontsize=fs,
            fontweight=fw,color=tc,zorder=4,linespacing=1.15)

def CC(ax, cx, cy, r, t, fc=C_WH, ec=C_BK, fs=10, lw=1.3):
    ax.add_patch(plt.Circle((cx,cy),r,fc=fc,ec=ec,lw=lw,zorder=3))
    ax.text(cx,cy,t,ha='center',va='center',fontsize=fs,
            fontweight='bold',color='#1a1a1a',zorder=4)

def AR(ax, x1,y1,x2,y2, c=C_BK, lw=1.2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.18',
                        color=c,lw=lw,mutation_scale=13),zorder=2)

def LN(ax, x1,y1,x2,y2, c=C_BK, lw=1.2, ls='-'):
    ax.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=ls,zorder=2,
            solid_capstyle='round')

def SR(ax, x, y, w, h, fc, alpha=0.35):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',
                 fc=fc,ec='none',lw=0,alpha=alpha,zorder=0))


# ════════════════════════════════════════════════════════════════
#  Figure 1  –  Overall Architecture  (LEFT → RIGHT)
#  Fixes: all arrows long enough, proper gaps between box edge and arrowhead
# ════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(17, 6))
    ax.set_xlim(-0.3, 17.0)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    H = 0.55
    yc = 2.3
    yt = 3.7
    yb = 0.9

    # x positions — widened for visible arrows everywhere
    x_in   = 0.7
    x_ipr  = 3.0;  x_me = 3.0
    x_add1 = 5.0
    x_imp  = 7.2;  x_exp = 7.2
    x_fu   = 10.2
    x_add2 = 11.7
    x_te   = 11.7
    x_res  = 13.2
    x_op   = 14.9
    x_out  = 16.3

    # Box widths
    w_in = 1.4; w_pr = 1.6; w_br = 2.0; w_fu = 1.5; w_te = 1.2
    w_res = 1.5; w_op = 1.2; w_out = 0.9

    # shaded backgrounds
    SR(ax, x_imp-w_br/2-0.15, yt-H*0.65-0.1, w_br+0.3, H*1.3+0.2, BG_IMP, 0.4)
    SR(ax, x_exp-w_br/2-0.15, yb-H*0.65-0.1, w_br+0.3, H*1.3+0.2, BG_EXP, 0.4)

    # boxes
    B(ax, x_in,  yc,  w_in, H, 'Input\n(x, mask, t)', C_IN, C_IN_E, fs=8)
    B(ax, x_ipr, yt,  w_pr, H, 'Input\nProjection',   C_PR, C_PR_E, fs=8)
    B(ax, x_me,  yb,  w_pr, H, 'Mask\nEmbedding',     C_MK, C_MK_E, fs=8)
    CC(ax, x_add1, yc, 0.20, '+')
    B(ax, x_imp, yt,  w_br, H*1.2, 'Implicit Module\n(Dilated Conv)', C_CV, C_CV_E, fs=8)
    B(ax, x_exp, yb,  w_br, H*1.2, 'Explicit Module\n(S4 Layer ×2)',  C_S4, C_S4_E, fs=8)
    B(ax, x_fu,  yc,  w_fu, H*1.2, 'Gated\nFusion',  C_FU, C_FU_E, fs=9)
    CC(ax, x_add2, yc, 0.20, '+')
    B(ax, x_te,  yc+1.3, w_te, H*0.9, 'Time\nEmbed', C_TI, C_TI_E, fs=8)
    B(ax, x_res, yc,  w_res, H*1.1, 'Residual\nBlocks ×6', C_RE, C_RE_E, fs=8)
    B(ax, x_op,  yc,  w_op, H, 'Output\nProj',       C_PR, C_PR_E, fs=8)
    B(ax, x_out, yc,  w_out, H, 'Output',             C_IN, C_IN_E, fs=8)

    # ── ARROWS ──
    G = 0.08

    # Input → fork
    fork_x = x_in + w_in/2 + 0.3
    LN(ax, x_in+w_in/2, yc, fork_x, yc)
    LN(ax, fork_x, yc, fork_x, yt)
    AR(ax, fork_x, yt, x_ipr-w_pr/2-G, yt)
    LN(ax, fork_x, yc, fork_x, yb)
    AR(ax, fork_x, yb, x_me-w_pr/2-G, yb)

    # Input Proj → (+) via L-shape
    route1 = x_add1 - 0.35
    LN(ax, x_ipr+w_pr/2, yt, route1, yt)
    LN(ax, route1, yt, route1, yc+0.35)
    AR(ax, route1, yc+0.35, x_add1-0.14, yc+0.14)
    # Mask Embed → (+) via L-shape
    LN(ax, x_me+w_pr/2, yb, route1, yb)
    LN(ax, route1, yb, route1, yc-0.35)
    AR(ax, route1, yc-0.35, x_add1-0.14, yc-0.14)

    # (+) → split to branches
    split_x = x_add1 + 0.20 + 0.30
    LN(ax, x_add1+0.20, yc, split_x, yc)
    LN(ax, split_x, yc, split_x, yt)
    AR(ax, split_x, yt, x_imp-w_br/2-G, yt)
    LN(ax, split_x, yc, split_x, yb)
    AR(ax, split_x, yb, x_exp-w_br/2-G, yb)

    # Branches → Gated Fusion  (KEY FIX: visible horizontal segment first)
    merge_x = x_imp + w_br/2 + 0.7   # 0.7 past module right edge
    LN(ax, x_imp+w_br/2, yt, merge_x, yt)   # horizontal from Implicit right
    LN(ax, merge_x, yt, merge_x, yc)          # vertical down to center
    AR(ax, merge_x, yc, x_fu-w_fu/2-G, yc)   # arrow into Fusion
    LN(ax, x_exp+w_br/2, yb, merge_x, yb)   # horizontal from Explicit right
    LN(ax, merge_x, yb, merge_x, yc)          # vertical up to center

    # Fusion → (+time)
    AR(ax, x_fu+w_fu/2+G, yc, x_add2-0.20-G, yc)
    # Time Embed → (+time)
    AR(ax, x_te, yc+1.3-H*0.9/2-G, x_add2, yc+0.20+G)
    # (+time) → Residual
    AR(ax, x_add2+0.20+G, yc, x_res-w_res/2-G, yc)
    # Residual → Output Proj (properly spaced)
    AR(ax, x_res+w_res/2+G, yc, x_op-w_op/2-G, yc)
    # Output Proj → Output
    AR(ax, x_op+w_op/2+G, yc, x_out-w_out/2-G, yc)

    # labels
    ax.text(x_imp, yt+H*1.2/2+0.2, 'Implicit Branch', ha='center',
            fontsize=7.5, color='#E65100', fontstyle='italic', fontweight='bold')
    ax.text(x_exp, yb-H*1.2/2-0.2, 'Explicit Branch', ha='center',
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
#  Figure 2  –  Implicit Module
#  Fixes: wider gaps for internal arrows, Output fully visible, more spacing
# ════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.set_xlim(-0.3, 18.0)
    ax.set_ylim(-1.2, 4.8)
    ax.axis('off')

    H = 0.45
    yc = 2.0
    y_conv = yc + 0.70
    y_gelu = yc
    y_drop = yc - 0.70
    y_plus = y_drop - 0.45
    y_route = y_plus - 0.30
    G = 0.06

    x_in = 0.5; x_li = 2.0
    conv_step = 1.85
    x_c0 = 4.0
    dilations = [1, 2, 4, 8, 16, 32]
    x_convs = [x_c0 + i * conv_step for i in range(6)]
    x_lo = x_convs[-1] + conv_step + 0.3
    x_out = x_lo + 1.8
    bw = 1.2
    rt_off = 0.40  # routing column offset from box left edge (visible stem)

    # shaded background
    SR(ax, x_c0-bw/2-0.2, y_plus-0.35,
       x_convs[-1]-x_c0+bw+0.4, y_conv-y_plus+0.48/2+0.65, BG_IMP, 0.35)

    # Input / Linear
    B(ax, x_in, yc, 1.0, H, 'Input', C_IN, C_IN_E, fs=9)
    B(ax, x_li, yc, 1.2, H, 'Linear\n(D→H)', C_PR, C_PR_E, fs=8)
    AR(ax, x_in+1.0/2+G, yc, x_li-1.2/2-G, yc)

    # Conv blocks
    for i, (xc, d) in enumerate(zip(x_convs, dilations)):
        B(ax, xc, y_conv, bw, 0.50, f'Conv d={d}', C_CV, C_CV_E, fs=7.5)
        B(ax, xc, y_gelu, bw, 0.40, 'GELU', C_AC, C_AC_E, fs=7, fw='normal')
        B(ax, xc, y_drop, bw, 0.40, 'Dropout', C_DR, C_DR_E, fs=7, fw='normal')

        # Internal arrows
        AR(ax, xc, y_conv-0.50/2-G, xc, y_gelu+0.40/2+G)
        AR(ax, xc, y_gelu-0.40/2-G, xc, y_drop+0.40/2+G)

        # (+) circle
        CC(ax, xc, y_plus, 0.13, '+', fs=7)
        AR(ax, xc, y_drop-0.40/2-G, xc, y_plus+0.13+G)

        # Residual bypass right side
        rx = xc + bw/2 + 0.15
        rt = y_conv + 0.50/2 + 0.04
        LN(ax, xc+bw/2, rt, rx, rt, c='#95a5a6', lw=0.9)
        LN(ax, rx, rt, rx, y_plus, c='#95a5a6', lw=0.9)
        AR(ax, rx, y_plus, xc+0.13+G, y_plus, c='#95a5a6', lw=0.9)

        # alpha label
        ax.text(xc, y_conv+0.50/2+0.15, f'α{chr(8321+i)}',
                fontsize=7, color='#c0392c', ha='center', fontstyle='italic')

    # ── Inter-block connections (KEY FIX: visible arrow stem) ──
    # Linear → first block: route column well left of Conv box
    rt_col = x_convs[0] - bw/2 - rt_off
    LN(ax, x_li+1.2/2, yc, rt_col, yc)
    LN(ax, rt_col, yc, rt_col, y_conv)
    AR(ax, rt_col, y_conv, x_convs[0]-bw/2-G, y_conv)

    # Between blocks: visible horizontal stem before arrowhead
    for i in range(5):
        xf = x_convs[i]; xt = x_convs[i+1]
        rt_col = xt - bw/2 - rt_off
        LN(ax, xf, y_plus-0.13, xf, y_route)
        LN(ax, xf, y_route, rt_col, y_route)
        LN(ax, rt_col, y_route, rt_col, y_conv)
        AR(ax, rt_col, y_conv, xt-bw/2-G, y_conv)

    # Last block → Linear Out
    xl = x_convs[-1]
    rt_col_lo = x_lo - 1.2/2 - rt_off
    LN(ax, xl, y_plus-0.13, xl, y_route)
    LN(ax, xl, y_route, rt_col_lo, y_route)
    LN(ax, rt_col_lo, y_route, rt_col_lo, yc)
    AR(ax, rt_col_lo, yc, x_lo-1.2/2-G, yc)

    # Linear Out / Output (wider Output box to prevent clipping)
    B(ax, x_lo, yc, 1.2, H, 'Linear\n(H→H)', C_PR, C_PR_E, fs=8)
    B(ax, x_out, yc, 1.2, H, 'Output', C_IN, C_IN_E, fs=9)
    AR(ax, x_lo+1.2/2+G, yc, x_out-1.2/2-G, yc)

    # Global skip
    skip_y = y_route - 0.45
    LN(ax, x_li, yc-H/2, x_li, skip_y, c='#3498db', lw=1.0, ls='--')
    LN(ax, x_li, skip_y, x_lo, skip_y, c='#3498db', lw=1.0, ls='--')
    LN(ax, x_lo, skip_y, x_lo, yc-H/2, c='#3498db', lw=1.0, ls='--')

    ax.text((x_li+x_lo)/2, skip_y-0.18, 'Global Residual Connection',
            ha='center', fontsize=7.5, color='#3498db', fontstyle='italic')

    # Title
    ax.text((x_convs[0]+x_convs[-1])/2, y_conv+0.50/2+0.42,
            'Multi-scale Dilated Causal Convolution (d = 1, 2, 4, 8, 16, 32)',
            ha='center', fontsize=8.5, color='#E65100', fontweight='bold')

    # Formula
    ax.text((x_convs[0]+x_convs[-1])/2, skip_y-0.5,
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
#  Figure 3  –  Explicit Module
#  Fixes: "1"/"2" labels NOT overlapping boxes, remove meaningless dashed
#         lines, "S4 Block" label NOT overlapping border, proper arrows
# ════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.0, 5.8)
    ax.axis('off')

    H = 0.50
    G = 0.06

    # ── LEFT PANEL ──
    yc_l = 2.5
    x_lin = 0.6; x_s41 = 2.5; x_add1 = 3.9; x_s42 = 5.3; x_lout = 7.0

    SR(ax, x_s41-0.75, yc_l-1.15, x_s42-x_s41+1.5, 2.3, BG_EXP, 0.35)

    B(ax, x_lin, yc_l, 1.0, H, 'Input', C_IN, C_IN_E, fs=9)

    # S4 Block 1
    s4h = H*2.0
    B(ax, x_s41, yc_l, 1.2, s4h, 'S4\nBlock 1', C_S4, C_S4_E, fs=9)

    CC(ax, x_add1, yc_l, 0.16, '+', fs=8)

    # S4 Block 2
    B(ax, x_s42, yc_l, 1.2, s4h, 'S4\nBlock 2', C_S4, C_S4_E, fs=9)

    B(ax, x_lout, yc_l, 1.2, H, 'Output', C_IN, C_IN_E, fs=9)

    # Arrows (left → right, proper gaps)
    AR(ax, x_lin+1.0/2+G, yc_l, x_s41-1.2/2-G, yc_l)
    AR(ax, x_s41+1.2/2+G, yc_l, x_add1-0.16-G, yc_l)
    AR(ax, x_add1+0.16+G, yc_l, x_s42-1.2/2-G, yc_l)
    AR(ax, x_s42+1.2/2+G, yc_l, x_lout-1.2/2-G, yc_l)

    # Skip from Input to (+) below
    skip_y_l = yc_l - 1.4
    LN(ax, x_lin, yc_l-H/2, x_lin, skip_y_l, c='#3498db', lw=1.0)
    LN(ax, x_lin, skip_y_l, x_add1, skip_y_l, c='#3498db', lw=1.0)
    AR(ax, x_add1, skip_y_l, x_add1, yc_l-0.16-G, c='#3498db', lw=1.0)
    ax.text((x_lin+x_add1)/2, skip_y_l-0.18, 'skip',
            ha='center', fontsize=7, color='#3498db', fontstyle='italic')

    # Module label ABOVE the shaded region (no overlap)
    ax.text((x_s41+x_s42)/2, yc_l+s4h/2+0.25,
            'Explicit Information Extraction Module',
            ha='center', fontsize=9, color='#1565C0', fontweight='bold')

    # ── RIGHT PANEL: S4 Block detail (bottom → top) ──
    rx = 10.2; rw = 2.0

    ry_s4   = 0.3
    ry_gelu = 1.25
    ry_drop = 2.1
    ry_add  = 3.0
    ry_ln   = 3.9

    # Border box (sized precisely, label below with gap)
    bdr_x = rx - rw/2 - 0.3
    bdr_y = ry_s4 - H/2 - 0.25
    bdr_w = rw + 0.6
    bdr_h = ry_ln - ry_s4 + H + 0.5
    SR(ax, bdr_x, bdr_y, bdr_w, bdr_h, '#FFF8E1', 0.5)
    ax.add_patch(FancyBboxPatch((bdr_x, bdr_y), bdr_w, bdr_h,
                 boxstyle='round,pad=0.08', fc='none', ec='#D6B656',
                 lw=1.0, zorder=1))

    B(ax, rx, ry_s4,   rw,     H,     'S4 Layer',  C_S4, C_S4_E, fs=9)
    B(ax, rx, ry_gelu, rw*0.7, H*0.8, 'GELU',      C_AC, C_AC_E, fs=8, fw='normal')
    B(ax, rx, ry_drop, rw*0.8, H*0.8, 'Dropout',    C_DR, C_DR_E, fs=8, fw='normal')
    CC(ax, rx, ry_add, 0.15, '+', fs=8)
    B(ax, rx, ry_ln,   rw,     H,     'LayerNorm',  C_NM, C_NM_E, fs=9)

    # Arrows (bottom → top, proper gaps)
    AR(ax, rx, ry_s4+H/2+G,       rx, ry_gelu-H*0.8/2-G)
    AR(ax, rx, ry_gelu+H*0.8/2+G, rx, ry_drop-H*0.8/2-G)
    AR(ax, rx, ry_drop+H*0.8/2+G, rx, ry_add-0.15-G)
    AR(ax, rx, ry_add+0.15+G,     rx, ry_ln-H/2-G)

    # Skip on RIGHT side
    skx = rx + rw/2 + 0.5
    LN(ax, rx+rw/2, ry_s4, skx, ry_s4, c='#2c3e50', lw=1.0)
    LN(ax, skx, ry_s4, skx, ry_add, c='#2c3e50', lw=1.0)
    AR(ax, skx, ry_add, rx+0.15+G, ry_add, c='#2c3e50', lw=1.0)
    ax.text(skx+0.15, (ry_s4+ry_add)/2, 'Skip', fontsize=8.5,
            color='#2c3e50', rotation=90, va='center', ha='left', fontweight='bold')

    # "S4 Block" label clearly BELOW the border (with proper gap)
    ax.text(rx, bdr_y-0.2, 'S4 Block', ha='center', fontsize=10,
            color='#D6B656', fontweight='bold')

    # SSM equation
    ax.text(5.5, -0.7,
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
    print('Generating v6 figures (merge routing, arrow stems, Output spacing)...')
    fig1(); fig2(); fig3()
    print('Done – all saved to Figures4paper/')
