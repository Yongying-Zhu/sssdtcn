"""
Debutanizer Dataset Multi-scale Periodicity Visualization
Feature: u6 (selected for strongest periodicity - autocorr=0.276)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════
#  LOAD DATA - Select u6 (best periodicity)
# ══════════════════════════════════════════════════════════════

data = np.loadtxt('/home/user/sssdtcn/debutanizer_data.txt', skiprows=5)
# u6 is column index 5 (0-indexed: u1=0, u2=1, ..., u6=5)
feature = data[:, 5]
feature_name = 'u6 (Tray temperature)'
n_samples = len(feature)

# Time parameters
samples_per_hour = 60
samples_per_period = 480  # 8 hours

# ══════════════════════════════════════════════════════════════
#  CREATE FIGURE
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 7.5))

ax1 = fig.add_axes([0.08, 0.58, 0.40, 0.35])
ax2 = fig.add_axes([0.56, 0.58, 0.40, 0.35])
ax3 = fig.add_axes([0.08, 0.10, 0.88, 0.38])

C_BLUE = '#4A90D9'
C_GRAY = '#888888'
C_ORANGE = '#E8853D'

# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════

fig.text(0.50, 0.96, f'Feature {feature_name} in Debutanizer',
         fontsize=13, fontweight='bold', ha='center')

# ══════════════════════════════════════════════════════════════
#  (a) LONG-TERM TREND
# ══════════════════════════════════════════════════════════════

time_hours = np.arange(n_samples) / samples_per_hour

ax1.plot(time_hours, feature, color=C_BLUE, linewidth=0.8, alpha=0.95)

# Shifted version to show periodicity
shift = samples_per_period
feature_shifted = np.roll(feature, -shift)
time_shifted = time_hours[:-shift]
ax1.plot(time_shifted, feature_shifted[:len(time_shifted)],
         color=C_GRAY, linewidth=0.7, alpha=0.6)

ax1.set_ylabel('Feature value', fontsize=10)
ax1.set_xlim(0, 40)
ax1.set_ylim(0, 1.05)

xticks = np.arange(0, 41, 8)
ax1.set_xticks(xticks)
ax1.set_xticklabels([str(int(x)) for x in xticks], fontsize=9)
ax1.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  (b) SHORT-TERM CYCLE
# ══════════════════════════════════════════════════════════════

n_show = samples_per_period
feature_period = feature[:n_show]
time_period = np.arange(n_show) / samples_per_hour

ax2.plot(time_period, feature_period, color=C_ORANGE, linewidth=1.0)

ax2.set_ylabel('Feature value', fontsize=10)
ax2.set_xlim(0, 8)

xticks_b = np.arange(0, 9, 1)
ax2.set_xticks(xticks_b)
ax2.set_xticklabels([str(int(x)) for x in xticks_b], fontsize=9)
ax2.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  (c) INTRA-PERIOD PATTERN
# ══════════════════════════════════════════════════════════════

n_periods = n_samples // samples_per_period
time_minutes = np.arange(samples_per_period)

period_colors = ['#4A90D9', '#E8853D', '#2ECC71', '#E74C3C', '#9B59B6']
period_labels = ['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5']

for i in range(min(5, n_periods)):
    start_idx = i * samples_per_period
    end_idx = start_idx + samples_per_period
    if end_idx <= n_samples:
        y_period = feature[start_idx:end_idx]
        ax3.plot(time_minutes, y_period, color=period_colors[i],
                 linewidth=0.9, alpha=0.85, label=period_labels[i])

ax3.set_ylabel('Feature value', fontsize=10)
ax3.set_xlim(0, samples_per_period)

hour_ticks = np.arange(0, samples_per_period + 1, 60)
hour_labels = [f'{i//60}:00' for i in hour_ticks]
ax3.set_xticks(hour_ticks)
ax3.set_xticklabels(hour_labels, fontsize=8)

ax3.legend(loc='upper left', fontsize=8, ncol=1, framealpha=0.9, handlelength=1.5)
ax3.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  SUBPLOT LABELS - FIXED (no duplicate, no extra space)
# ══════════════════════════════════════════════════════════════

# (a) - single complete label
ax1.text(0.5, -0.16, '(a) Time interval (hour)', transform=ax1.transAxes,
         fontsize=10, ha='center')

# (b) - with colored "hour" highlighted (include closing paren)
ax2.text(0.5, -0.16, '(b) Time interval (', transform=ax2.transAxes,
         fontsize=10, ha='right')
ax2.text(0.5, -0.16, 'hour)', transform=ax2.transAxes,
         fontsize=10, ha='left', color=C_ORANGE, fontweight='bold')

# (c) - with colored "minute" highlighted (include closing paren)
ax3.text(0.44, -0.13, '(c) Time interval (', transform=ax3.transAxes,
         fontsize=10, ha='right')
ax3.text(0.44, -0.13, 'minute)', transform=ax3.transAxes,
         fontsize=10, ha='left', color=C_ORANGE, fontweight='bold')

# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

out_path = '/home/user/sssdtcn/Figures4paper/fig_debutanizer_periodicity'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out_path}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out_path}.png/pdf')
print(f'Feature selected: u6 (highest periodicity score)')
