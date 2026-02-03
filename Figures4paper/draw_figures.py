"""
Architecture diagrams for Implicit-Explicit Diffusion Model paper.
v3: Tight layout, ZERO crossing arrows, clean compact design.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colors (draw.io / paper style) ──────────────────────────
C_IN   = '#D5E8D4'; C_IN_E   = '#82B366'   # green  input/output
C_PR   = '#DAE8FC'; C_PR_E   = '#6C8EBF'   # blue   projection
C_CV   = '#FFE6CC'; C_CV_E   = '#D79B00'   # orange conv
C_S4   = '#D5E8FC'; C_S4_E   = '#3A7BBF'   # indigo S4
C_FU   = '#F8CECC'; C_FU_E   = '#B85450'   # red    fusion
C_RE   = '#E1D5E7'; C_RE_E   = '#9673A6'   # purple residual
C_TI   = '#FFF2CC'; C_TI_E   = '#D6B656'   # yellow time
C_MK   = '#D5F5E3'; C_MK_E   = '#27AE60'   # mint   mask
C_NM   = '#FFF2CC'; C_NM_E   = '#D6B656'   # yellow norm
C_AC   = '#E8F8F5'; C_AC_E   = '#1ABC9C'   # teal   activation
C_DR   = '#F2F3F4'; C_DR_E   = '#95A5A6'   # grey   dropout
C_WH   = '#FFFFFF'; C_BK     = '#2C3E50'   # white / dark

def B(ax, cx, cy, w, h, t, fc, ec, fs=9, fw='bold', tc='#1a1a1a'):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle='round,pad=0.06', fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(cx,cy,t, ha='center',va='center',fontsize=fs,
            fontweight=fw, color=tc, zorder=4)

def C(ax, cx, cy, r, t, fc=C_WH, ec=C_BK, fs=10):
    ax.add_patch(plt.Circle((cx,cy),r, fc=fc,ec=ec,lw=1.3,zorder=3))
    ax.text(cx,cy,t, ha='center',va='center',fontsize=fs,
            fontweight='bold',color='#1a1a1a',zorder=4)

def A(ax, x1,y1,x2,y2, c=C_BK, lw=1.2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=c,lw=lw,mutation_scale=11),zorder=2)

def L(ax, x1,y1,x2,y2, c=C_BK, lw=1.2, ls='-'):
    ax.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=ls,zorder=2,solid_capstyle='round')

def DR(ax, cx,cy,w,h, lab='', ec='#7f8c8d', lw=0.9, fs=8):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle='round,pad=0.1', fc='none',ec=ec,lw=lw,ls='--',zorder=1))
    if lab:
        ax.text(cx,cy+h/2+0.06,lab,ha='center',va='bottom',
                fontsize=fs,fontstyle='italic',color=ec,fontweight='bold')

# ════════════════════════════════════════════════════════════
#  Figure 1  –  Overall Architecture  (bottom → top, compact)
# ════════════════════════════════════════════════════════════
def fig1():
    fig,ax = plt.subplots(figsize=(6.5,8.5))
    ax.set_xlim(-3.2,3.8); ax.set_ylim(-0.2,8.5); ax.axis('off')

    W=2.2; H=0.44; s=0.62    # box width, height, step

    # y positions  (bottom → top)
    y_in   = 0.0
    y_fork = 0.55             # fork line from Input
    y_ipr  = 1.3;  y_me = 1.3
    y_add  = 2.15             # + circle (proj + mask)
    y_sp   = 2.75             # split to branches
    y_br   = 3.55             # branch boxes
    y_mg   = 4.35             # merge line going to fusion
    y_fu   = 4.75             # gated fusion
    y_ta   = 5.55             # + circle (time)
    y_rs   = 6.25             # residual blocks
    y_op   = 6.95             # output proj
    y_out  = 7.65             # output

    xc=0; xL=-1.3; xR=1.3; xT=3.0  # center, left, right, time-embed x

    # ── boxes ──
    B(ax, xc, y_in,  W,  H, 'Input (x, mask, t)', C_IN, C_IN_E)
    B(ax, xL, y_ipr, 1.7,H, 'Input Projection',   C_PR, C_PR_E, fs=8)
    B(ax, xR, y_me,  1.7,H, 'Mask Embedding',      C_MK, C_MK_E, fs=8)
    C(ax, xc, y_add, 0.16, '+')
    B(ax, xL, y_br, 1.7, H*1.3, 'Implicit Module\n(Dilated Conv)',  C_CV, C_CV_E, fs=7.5)
    B(ax, xR, y_br, 1.7, H*1.3, 'Explicit Module\n(S4 Layer ×2)',   C_S4, C_S4_E, fs=7.5)
    B(ax, xc, y_fu, W,  H, 'Gated Fusion',    C_FU, C_FU_E)
    C(ax, xc, y_ta, 0.16, '+')
    B(ax, xT, y_ta, 1.0, H, 'Time\nEmbed', C_TI, C_TI_E, fs=7)
    B(ax, xc, y_rs, W,  H, 'Residual Blocks ×6', C_RE, C_RE_E)
    B(ax, xc, y_op, W,  H, 'Output Projection',   C_PR, C_PR_E)
    B(ax, xc, y_out,W,  H, 'Output',              C_IN, C_IN_E)

    # dashed group around branch modules only (not fusion)
    dr_cy = y_br
    dr_h  = H*1.3 + 0.25
    DR(ax, xc, dr_cy, 4.2, dr_h, '', ec='#7f8c8d', fs=7)
    ax.text(xc, dr_cy - dr_h/2 - 0.06, 'Dual-Branch Feature Extraction',
            ha='center', va='top', fontsize=7, fontstyle='italic',
            color='#7f8c8d', fontweight='bold')

    # ── arrows (all go UP or horizontal, no crossing) ──

    # Input forks to Input Proj (left) and Mask Embed (right)
    L(ax, xc, y_in+H/2, xc, y_fork)
    L(ax, xc, y_fork, xL, y_fork)
    A(ax, xL, y_fork, xL, y_ipr-H/2)
    L(ax, xc, y_fork, xR, y_fork)
    A(ax, xR, y_fork, xR, y_me-H/2)

    # Input Proj and Mask Embed → + circle
    A(ax, xL, y_ipr+H/2, xc-0.16, y_add)
    A(ax, xR, y_me+H/2,  xc+0.16, y_add)

    # + circle → split → branches
    L(ax, xc, y_add+0.16, xc, y_sp)
    L(ax, xc, y_sp, xL, y_sp);  A(ax, xL, y_sp, xL, y_br-H*1.3/2)
    L(ax, xc, y_sp, xR, y_sp);  A(ax, xR, y_sp, xR, y_br-H*1.3/2)

    # branches → merge → fusion  (go up vertically, then horizontal into fusion)
    L(ax, xL, y_br+H*1.3/2, xL, y_mg)
    A(ax, xL, y_mg, xc-W/2, y_fu)     # horizontal left→center
    L(ax, xR, y_br+H*1.3/2, xR, y_mg)
    A(ax, xR, y_mg, xc+W/2, y_fu)     # horizontal right→center

    # fusion → + time
    A(ax, xc, y_fu+H/2, xc, y_ta-0.16)
    # time embed → + (horizontal from right, outside all other paths)
    A(ax, xT-1.0/2, y_ta, xc+0.16, y_ta)

    # + time → residual → output proj → output
    A(ax, xc, y_ta+0.16, xc, y_rs-H/2)
    A(ax, xc, y_rs+H/2,  xc, y_op-H/2)
    A(ax, xc, y_op+H/2,  xc, y_out-H/2)

    # subtle labels
    ax.text(xL, y_sp+0.1, 'implicit', ha='center',fontsize=6.5,color='#95a5a6',fontstyle='italic')
    ax.text(xR, y_sp+0.1, 'explicit', ha='center',fontsize=6.5,color='#95a5a6',fontstyle='italic')

    ax.set_title('(a) Overall Architecture of Implicit-Explicit Diffusion Model',
                 fontsize=11.5, fontweight='bold', pad=8)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'/home/user/sssdtcn/Figures4paper/fig1_overall_architecture.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Fig 1 ✓')


# ════════════════════════════════════════════════════════════
#  Figure 2  –  Implicit Module  (sequential top → bottom)
# ════════════════════════════════════════════════════════════
def fig2():
    fig,ax = plt.subplots(figsize=(6, 12))
    ax.set_xlim(-2.5,3.8); ax.set_ylim(-1.2,14.5); ax.axis('off')

    W=2.6; H=0.42
    xc=0.0

    # ── top section ──
    y = 14.0
    y_in = y;    y -= 0.55
    y_li = y;    y -= 0.50
    y_tr = y;    y -= 0.45

    B(ax, xc, y_in, 1.5, H, 'Input (B,L,D)', C_IN, C_IN_E, fs=9)
    B(ax, xc, y_li, W, H, 'Linear (D → H)', C_PR, C_PR_E, fs=9)
    B(ax, xc, y_tr, 1.3, 0.32, 'Transpose', C_DR, C_DR_E, fs=7.5, fw='normal')
    A(ax, xc, y_in-H/2, xc, y_li+H/2)
    A(ax, xc, y_li-H/2, xc, y_tr+0.32/2)

    # ── 6 conv blocks (each: Conv → GELU → Drop, with right-side residual) ──
    dilations = [1, 2, 4, 8, 16, 32]
    prev_bot = y_tr - 0.32/2
    block_step = 1.75   # total space per block

    conv_top_y = None   # record first conv top for global skip

    for i, d in enumerate(dilations):
        cy = prev_bot - 0.30   # top of this conv box (gap from previous)
        cy_conv = cy
        cy_act  = cy - 0.52   # GELU+Drop combined
        cy_add  = cy - 0.90   # residual add circle

        if i == 0:
            conv_top_y = cy_conv + H/2

        # Conv box
        B(ax, xc, cy_conv, W, H,
          f'CausalConv1d (k=5, d={d})', C_CV, C_CV_E, fs=8)

        # GELU + Dropout (single compact box)
        B(ax, xc, cy_act, 1.6, 0.30,
          'GELU → Dropout', C_AC, C_AC_E, fs=7, fw='normal')

        # Residual add
        C(ax, xc, cy_add, 0.14, '+', fs=8)

        # Internal arrows
        A(ax, xc, prev_bot, xc, cy_conv+H/2)
        A(ax, xc, cy_conv-H/2, xc, cy_act+0.30/2)
        A(ax, xc, cy_act-0.30/2, xc, cy_add+0.14)

        # Right-side residual bypass (outside main flow, no crossing)
        rx = W/2 + 0.30
        rt = cy_conv + H/2 + 0.04
        rb = cy_add
        L(ax, xc+W/2, rt, rx, rt, c='#95a5a6', lw=0.9)    # horiz out
        L(ax, rx, rt, rx, rb, c='#95a5a6', lw=0.9)          # vert down
        A(ax, rx, rb, xc+0.14, rb, c='#95a5a6', lw=0.9)     # horiz into +

        # alpha label
        ax.text(rx+0.12, (rt+rb)/2, f'α{chr(8321+i)}',
                fontsize=6.5, color='#c0392c', va='center', fontstyle='italic')

        prev_bot = cy_add - 0.14

    # ── bottom section ──
    y_tr2  = prev_bot - 0.30
    y_lo   = y_tr2 - 0.48
    y_radd = y_lo - 0.52
    y_out  = y_radd - 0.55

    B(ax, xc, y_tr2, 1.3, 0.32, 'Transpose', C_DR, C_DR_E, fs=7.5, fw='normal')
    B(ax, xc, y_lo,  W, H, 'Linear (H → H)', C_PR, C_PR_E, fs=9)
    C(ax, xc, y_radd, 0.16, '+')
    B(ax, xc, y_out, 1.5, H, 'Output (B,L,H)', C_IN, C_IN_E, fs=9)

    A(ax, xc, prev_bot, xc, y_tr2+0.32/2)
    A(ax, xc, y_tr2-0.32/2, xc, y_lo+H/2)
    A(ax, xc, y_lo-H/2, xc, y_radd+0.16)
    A(ax, xc, y_radd-0.16, xc, y_out+H/2)

    # Global residual skip on LEFT side (no crossing!)
    sx = -W/2 - 0.45
    L(ax, xc-W/2, y_li, sx, y_li, c='#3498db', lw=1.0, ls='--')
    L(ax, sx, y_li, sx, y_radd, c='#3498db', lw=1.0, ls='--')
    A(ax, sx, y_radd, xc-0.16, y_radd, c='#3498db', lw=1.0)
    ax.text(sx-0.06, (y_li+y_radd)/2, 'skip', fontsize=7,
            color='#3498db', rotation=90, va='center', ha='right', fontstyle='italic')

    # Formula
    ax.text(xc, y_out-H/2-0.22,
            r'$h_i = \alpha_i \cdot f_i(h_{i-1}) + (1{-}\alpha_i) \cdot h_{i-1}$',
            ha='center', fontsize=8.5, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', fc='#fafafa', ec='#bdc3c7', lw=0.7))

    ax.set_title('(b) Implicit Module: Multi-scale Dilated Causal Convolution',
                 fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'/home/user/sssdtcn/Figures4paper/fig2_implicit_module.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Fig 2 ✓')


# ════════════════════════════════════════════════════════════
#  Figure 3  –  Explicit Module  (left stack + right detail)
# ════════════════════════════════════════════════════════════
def fig3():
    fig,ax = plt.subplots(figsize=(10.5,7))
    ax.set_xlim(-0.5,10.5); ax.set_ylim(-0.6,7.2); ax.axis('off')

    H=0.40

    # ── LEFT: S4 stack (bottom → top) ──
    lx=1.6; lw=1.8
    ly_in=0.8; ly_s1=2.1; ly_s2=3.6; ly_out=4.9

    B(ax, lx, ly_in,  lw, H,     'Input Features',   C_IN, C_IN_E, fs=9)
    B(ax, lx, ly_s1,  lw, H*1.5, 'S4 Block 1',       C_S4, C_S4_E, fs=10)
    B(ax, lx, ly_s2,  lw, H*1.5, 'S4 Block 2',       C_S4, C_S4_E, fs=10)
    B(ax, lx, ly_out, lw, H,     'Explicit Features', C_IN, C_IN_E, fs=9)

    A(ax, lx, ly_in+H/2,    lx, ly_s1-H*1.5/2)
    A(ax, lx, ly_s1+H*1.5/2,lx, ly_s2-H*1.5/2)
    A(ax, lx, ly_s2+H*1.5/2,lx, ly_out-H/2)

    DR(ax, lx, (ly_in+ly_out)/2, lw+0.7,
       ly_out-ly_in+H+0.5, 'Explicit Module', ec='#3A7BBF', fs=9)

    # ── separator ──
    L(ax, 3.5, -0.3, 3.5, 6.9, c='#bdc3c7', lw=0.7, ls=':')

    # ── RIGHT: S4 Block detail (top → bottom) ──
    rx=7.0; rw=2.2

    # y coords (top → bottom)
    ry_in  = 6.4
    ry_pr  = 5.65
    ry_ssm = 4.8
    ry_op  = 3.95
    ry_ge  = 3.25
    ry_dr  = 2.7
    ry_ln  = 2.05
    ry_add = 1.35
    ry_out = 0.6

    ax.text(rx, 6.95, 'S4 Block Detail', ha='center', fontsize=11,
            fontweight='bold', color='#1a237e')

    B(ax, rx, ry_in,  rw,     H,     'Input (B, L, D)',  C_IN, C_IN_E)
    B(ax, rx, ry_pr,  rw,     H,     'Input Proj (D→S)', C_PR, C_PR_E)
    B(ax, rx, ry_ssm, rw+0.2, H*1.35,'S4 Recurrence\n(HiPPO, Multi-Head)', C_S4, C_S4_E, fs=8.5)
    B(ax, rx, ry_op,  rw,     H,     'Output Proj (S→D)',C_PR, C_PR_E)
    B(ax, rx, ry_ge,  1.1,    0.32,  'GELU',             C_AC, C_AC_E, fs=8, fw='normal')
    B(ax, rx, ry_dr,  1.2,    0.32,  'Dropout',          C_DR, C_DR_E, fs=8, fw='normal')
    B(ax, rx, ry_ln,  rw,     H,     'LayerNorm',        C_NM, C_NM_E)
    C(ax, rx, ry_add, 0.15, '+', fs=9)
    B(ax, rx, ry_out, rw,     H,     'Output (B, L, D)', C_IN, C_IN_E)

    # all arrows go straight down (zero crossing)
    A(ax, rx, ry_in-H/2,       rx, ry_pr+H/2)
    A(ax, rx, ry_pr-H/2,       rx, ry_ssm+H*1.35/2)
    A(ax, rx, ry_ssm-H*1.35/2, rx, ry_op+H/2)
    A(ax, rx, ry_op-H/2,       rx, ry_ge+0.32/2)
    A(ax, rx, ry_ge-0.32/2,    rx, ry_dr+0.32/2)
    A(ax, rx, ry_dr-0.32/2,    rx, ry_ln+H/2)
    A(ax, rx, ry_ln-H/2,       rx, ry_add+0.15)
    A(ax, rx, ry_add-0.15,     rx, ry_out+H/2)

    # skip / residual on RIGHT side (no crossing)
    skx = rx + rw/2 + 0.4
    L(ax, rx+rw/2, ry_in, skx, ry_in, c='#3498db', lw=1.0)
    L(ax, skx, ry_in, skx, ry_add,     c='#3498db', lw=1.0)
    A(ax, skx, ry_add, rx+0.15, ry_add, c='#3498db', lw=1.0)
    ax.text(skx+0.08, (ry_in+ry_add)/2, 'Residual', fontsize=7,
            color='#3498db', rotation=90, va='center', ha='left', fontstyle='italic')

    # annotations on LEFT side (no crossing)
    ax.text(rx-rw/2-0.1, ry_pr,       'D=128',   fontsize=7, ha='right', color='#7f8c8d')
    ax.text(rx-rw/2-0.1, ry_ssm+0.10, 'S=256',   fontsize=7, ha='right', color='#7f8c8d')
    ax.text(rx-rw/2-0.1, ry_ssm-0.10, '2 heads',fontsize=7, ha='right', color='#7f8c8d')

    # dashed border
    DR(ax, rx, (ry_out+ry_in)/2, rw+1.3,
       ry_in-ry_out+H+0.5, '', ec='#3A7BBF', lw=0.8)

    # equation
    ax.text(5.0, -0.35,
            r'$\mathbf{s}_{t+1}=\bar{A}\mathbf{s}_t+\bar{B}\mathbf{x}_t'
            r',\;\;\mathbf{y}_t=C\mathbf{s}_t+D\mathbf{x}_t$',
            ha='center', fontsize=9.5, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', fc='#fafafa', ec='#bdc3c7', lw=0.7))

    ax.set_title('(c) Explicit Module: Structured State Space (S4) Layer',
                 fontsize=11.5, fontweight='bold', pad=8)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'/home/user/sssdtcn/Figures4paper/fig3_explicit_module.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Fig 3 ✓')


if __name__ == '__main__':
    print('Generating v3 figures...')
    fig1(); fig2(); fig3()
    print('Done – all saved to Figures4paper/')
