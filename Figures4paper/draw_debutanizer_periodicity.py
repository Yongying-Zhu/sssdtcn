"""
Debutanizer Dataset Multi-scale Periodicity Visualization
Feature: u7 (selected for strongest periodicity - autocorr=0.424)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════
#  LOAD DATA - Select u7 (best periodicity, autocorr=0.424)
# ══════════════════════════════════════════════════════════════

data = np.loadtxt('/home/user/sssdtcn/debutanizer_data.txt', skiprows=5)
# u7 is column index 6 (0-indexed: u1=0, u2=1, ..., u7=6)
feature = data[:, 6]
feature_name = 'u7 (Butane concentration)'
n_samples = len(feature)

# Time parameters
samples_per_hour = 60
samples_per_day = samples_per_hour * 24  # 1440 samples per day
samples_per_period = 480  # 8 hours for period overlay

# ══════════════════════════════════════════════════════════════
#  CREATE FIGURE
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 7.5))

ax1 = fig.add_axes([0.08, 0.58, 0.40, 0.35])
ax2 = fig.add_axes([0.56, 0.58, 0.40, 0.35])
ax3 = fig.add_axes([0.08, 0.10, 0.88, 0.38])

C_GRAY = '#888888'
C_ORANGE = '#E8853D'

# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════

fig.text(0.50, 0.96, f'Feature {feature_name} in Debutanizer',
         fontsize=13, fontweight='bold', ha='center')

# ══════════════════════════════════════════════════════════════
#  (a) LONG-TERM TREND - in DAYS (no blue line, only gray)
# ══════════════════════════════════════════════════════════════

time_days = np.arange(n_samples) / samples_per_day

# Only gray line showing entire dataset trend
ax1.plot(time_days, feature, color=C_GRAY, linewidth=0.8, alpha=0.9)

ax1.set_ylabel('Feature value', fontsize=10)
ax1.set_xlim(0, 5)  # Extended to 0-5 days as requested
ax1.set_ylim(0, 1.05)

# X-axis: 0-5 days
xticks_a = np.arange(0, 6, 1)  # 0, 1, 2, 3, 4, 5
ax1.set_xticks(xticks_a)
ax1.set_xticklabels([str(int(x)) for x in xticks_a], fontsize=9)
ax1.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  (b) SHORT-TERM CYCLE - Extended to 24 hours
# ══════════════════════════════════════════════════════════════

# Show 24 hours (or full data if less) to see periodic patterns
n_show_b = min(24 * samples_per_hour, n_samples)  # 24 hours
feature_period = feature[:n_show_b]
time_hours = np.arange(n_show_b) / samples_per_hour

ax2.plot(time_hours, feature_period, color=C_ORANGE, linewidth=0.9)

ax2.set_ylabel('Feature value', fontsize=10)
ax2.set_xlim(0, n_show_b / samples_per_hour)

xticks_b = np.arange(0, n_show_b / samples_per_hour + 1, 4)
ax2.set_xticks(xticks_b)
ax2.set_xticklabels([str(int(x)) for x in xticks_b], fontsize=9)
ax2.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  (c) INTRA-PERIOD PATTERN - Extended x-axis with time labels
# ══════════════════════════════════════════════════════════════

n_periods = n_samples // samples_per_period
time_minutes = np.arange(samples_per_period)

period_colors = ['#4A90D9', '#E8853D', '#2ECC71', '#E74C3C']
period_labels = ['Period 1', 'Period 2', 'Period 3', 'Period 4']

for i in range(min(4, n_periods)):
    start_idx = i * samples_per_period
    end_idx = start_idx + samples_per_period
    if end_idx <= n_samples:
        y_period = feature[start_idx:end_idx]
        ax3.plot(time_minutes, y_period, color=period_colors[i],
                 linewidth=0.9, alpha=0.85, label=period_labels[i])

ax3.set_ylabel('Feature value', fontsize=10)
ax3.set_xlim(0, samples_per_period)

# Extended x-axis with specific time labels (HH:MM format)
hour_ticks = np.arange(0, samples_per_period + 1, 60)  # Every hour
hour_labels = [f'{i//60}:00' for i in hour_ticks]
ax3.set_xticks(hour_ticks)
ax3.set_xticklabels(hour_labels, fontsize=8)

ax3.legend(loc='upper left', fontsize=8, ncol=1, framealpha=0.9, handlelength=1.5)
ax3.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  SUBPLOT LABELS
# ══════════════════════════════════════════════════════════════

# (a) - Long-term in DAYS
ax1.text(0.5, -0.16, '(a) Time interval (day)', transform=ax1.transAxes,
         fontsize=10, ha='center')

# (b) - Short-term with colored "hour"
ax2.text(0.5, -0.16, '(b) Time interval (', transform=ax2.transAxes,
         fontsize=10, ha='right')
ax2.text(0.5, -0.16, 'hour)', transform=ax2.transAxes,
         fontsize=10, ha='left', color=C_ORANGE, fontweight='bold')

# (c) - Intra-period with colored "minute"
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
print(f'Feature selected: u7 (highest periodicity, autocorr=0.424)')
