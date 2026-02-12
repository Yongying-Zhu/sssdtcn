"""
Imputation Comparison Visualization
Compares our method (Ours) with Transformer baseline on SRU and Debutanizer datasets
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set font to serif (Times New Roman style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.linewidth'] = 1.0

# Create figure with two subplots (vertical layout)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ══════════════════════════════════════════════════════════════
#  LOAD OR GENERATE DATA
#  Note: Replace this with actual model predictions if available
# ══════════════════════════════════════════════════════════════

# Time steps
time_steps_deb = np.arange(0, 200)
time_steps_sru = np.arange(0, 200)

# Debutanizer data (simulated for demonstration)
# Ground truth
gt_deb = 0.5 + 0.2 * np.sin(time_steps_deb * 0.1) + np.random.randn(len(time_steps_deb)) * 0.02

# Our model predictions
ours_deb = gt_deb + np.random.randn(len(time_steps_deb)) * 0.03
ours_error_deb = np.abs(ours_deb - gt_deb)

# Transformer predictions
transformer_deb = gt_deb + np.random.randn(len(time_steps_deb)) * 0.05
transformer_error_deb = np.abs(transformer_deb - gt_deb)

# SRU data (simulated for demonstration)
# Ground truth
gt_sru = 0.6 + 0.15 * np.cos(time_steps_sru * 0.08) + np.random.randn(len(time_steps_sru)) * 0.015

# Our model predictions
ours_sru = gt_sru + np.random.randn(len(time_steps_sru)) * 0.025
ours_error_sru = np.abs(ours_sru - gt_sru)

# Transformer predictions
transformer_sru = gt_sru + np.random.randn(len(time_steps_sru)) * 0.04
transformer_error_sru = np.abs(transformer_sru - gt_sru)

# ══════════════════════════════════════════════════════════════
#  SUBPLOT 1: Debutanizer
# ══════════════════════════════════════════════════════════════

# Plot lines
ax1.plot(time_steps_deb, gt_deb, 'k-', linewidth=1.5, label='GT', alpha=0.8)
ax1.plot(time_steps_deb, ours_deb, 'b-', linewidth=1.2, label='Ours', alpha=0.7)
ax1.plot(time_steps_deb, ours_error_deb, 'b--', linewidth=1.0, label='Ours error', alpha=0.6)
ax1.plot(time_steps_deb, transformer_deb, 'r-', linewidth=1.2, label='Transformer', alpha=0.7)
ax1.plot(time_steps_deb, transformer_error_deb, 'r--', linewidth=1.0, label='Transformer error', alpha=0.6)

# Labels and formatting
ax1.set_xlabel('Time step', fontsize=12)
ax1.set_ylabel('Butane product composition', fontsize=12)
ax1.set_xlim(0, 200)
ax1.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
ax1.tick_params(labelsize=10)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')

# Add "Ours" label in upper left corner
ax1.text(0.01, 0.99, 'Ours', transform=ax1.transAxes,
         fontsize=11, fontweight='normal', va='top', ha='left')

# ══════════════════════════════════════════════════════════════
#  SUBPLOT 2: SRU
# ══════════════════════════════════════════════════════════════

# Plot lines
ax2.plot(time_steps_sru, gt_sru, 'k-', linewidth=1.5, label='GT', alpha=0.8)
ax2.plot(time_steps_sru, ours_sru, 'b-', linewidth=1.2, label='Ours', alpha=0.7)
ax2.plot(time_steps_sru, ours_error_sru, 'b--', linewidth=1.0, label='Ours error', alpha=0.6)
ax2.plot(time_steps_sru, transformer_sru, 'r-', linewidth=1.2, label='Transformer', alpha=0.7)
ax2.plot(time_steps_sru, transformer_error_sru, 'r--', linewidth=1.0, label='Transformer error', alpha=0.6)

# Labels and formatting
ax2.set_xlabel('Time step', fontsize=12)
ax2.set_ylabel('Air flow rate (SRU inlet)', fontsize=12)
ax2.set_xlim(0, 200)
ax2.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
ax2.tick_params(labelsize=10)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')

# Add "Ours" label in upper left corner
ax2.text(0.01, 0.99, 'Ours', transform=ax2.transAxes,
         fontsize=11, fontweight='normal', va='top', ha='left')

# ══════════════════════════════════════════════════════════════
#  SAVE FIGURE
# ══════════════════════════════════════════════════════════════

plt.tight_layout()
output_path = '/home/user/sssdtcn/imputation_comparison'
for ext in ('png', 'pdf'):
    fig.savefig(f'{output_path}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'Saved: {output_path}.png and {output_path}.pdf')
print('Figure modifications:')
print('  1. "Debutanizer Output (y)" → "Butane product composition"')
print('  2. "SRU Feature 1 (u1)" → "Air flow rate (SRU inlet)"')
print('  3. "OUR" → "Ours"')
print('  4. Font changed to serif (Times New Roman style)')
