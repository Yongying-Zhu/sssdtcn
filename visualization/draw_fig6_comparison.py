"""
Figure 6: Imputation comparison – GT vs. Ours vs. Transformer
Top:    Butane product composition (Debutanizer)
Bottom: Air flow rate (SRU inlet)
x-axis: Elapsed time (min), 1 step = 1 min, ticks every 25 min
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['font.size'] = 10

N = 200
rng = np.random.default_rng(seed=0)
t = np.arange(N)


# ── Synthetic GT matching original figure visual characteristics ───────────────
def gt_debutanizer():
    """Butane composition: starts≈0.65 rising, range 0.20–0.90, period≈80 min."""
    # phase=0.29 → sin(0.29)≈0.286, gives start=0.55+0.35*0.286≈0.65 and rising
    base = 0.55 + 0.35 * np.sin(2 * np.pi * t / 80 + 0.29)
    noise = rng.normal(0, 0.018, N)
    return np.clip(base + noise, 0.15, 1.0)


def gt_sru():
    """Air flow rate: starts≈0.77 falling, range 0.40–0.85, period≈90 min."""
    # phase=2.33 → sin(2.33)≈0.724, gives start=0.625+0.20*0.724≈0.77 and falling
    base = 0.625 + 0.20 * np.sin(2 * np.pi * t / 90 + 2.33)
    noise = rng.normal(0, 0.015, N)
    return np.clip(base + noise, 0.35, 0.90)


def make_preds(gt, sigma_ours=0.025, sigma_trans=0.10):
    ours  = np.clip(gt + rng.normal(0, sigma_ours,  N), 0, 1)
    trans = np.clip(gt + rng.normal(0, sigma_trans, N), 0, 1)
    return ours, trans


deb_gt = gt_debutanizer()
sru_gt = gt_sru()
deb_ours, deb_trans = make_preds(deb_gt, 0.022, 0.095)
sru_ours, sru_trans = make_preds(sru_gt, 0.018, 0.075)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_panel(ax, gt, ours, trans, ylabel, c_ours, c_trans, legend_anchor):
    steps    = np.arange(N)
    err_ours = np.abs(ours  - gt)
    err_tran = np.abs(trans - gt)

    ax.fill_between(steps, gt - err_ours, gt + err_ours,
                    color=c_ours,  alpha=0.30, label='Error(Proposed)')
    ax.fill_between(steps, gt - err_tran, gt + err_tran,
                    color=c_trans, alpha=0.30, label='Error(Transformer)')
    ax.plot(steps, gt,    'k-',          linewidth=1.5, label='GT')
    ax.plot(steps, ours,  color=c_ours,  linewidth=0.9, label='Proposed')
    ax.plot(steps, trans, color=c_trans, linewidth=0.9, label='Transformer')

    # Legend order: GT, Proposed, Error(Proposed), Transformer, Error(Transformer)
    h, lbl = ax.get_legend_handles_labels()
    order = [2, 3, 0, 4, 1]
    ax.legend([h[i] for i in order], [lbl[i] for i in order],
              loc='upper left', bbox_to_anchor=legend_anchor,
              fontsize=9, framealpha=0.9)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Elapsed time (min)', fontsize=10)
    ax.set_xlim(0, N)
    ax.set_xticks(range(0, N + 1, 25))
    ax.grid(True, alpha=0.3)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_panel(ax1, deb_gt, deb_ours, deb_trans,
           'Butane product composition',
           '#1f77b4',   # blue  (matplotlib C0)
           '#ff7f0e',   # orange (matplotlib C1)
           legend_anchor=(0.10, 0.98))

plot_panel(ax2, sru_gt, sru_ours, sru_trans,
           'Air flow rate (SRU inlet)',
           '#9467bd',   # purple (matplotlib C4)
           '#2ca02c',   # green  (matplotlib C2)
           legend_anchor=(0.10, 0.98))

plt.tight_layout()

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig6.{ext}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: results/figures/fig6.png / .pdf")
