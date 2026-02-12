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
#  CREATE FIGURE WITH AESTHETIC SETTINGS
# ══════════════════════════════════════════════════════════════

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#333333'

fig = plt.figure(figsize=(10, 7.5))

ax1 = fig.add_axes([0.08, 0.58, 0.40, 0.35])
ax2 = fig.add_axes([0.56, 0.58, 0.40, 0.35])
ax3 = fig.add_axes([0.08, 0.10, 0.88, 0.38])

C_GRAY = '#666666'
C_ORANGE = '#E8853D'

# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════

fig.text(0.50, 0.96, 'Butane Concentration in Debutanizer Column',
         fontsize=14, fontweight='bold', ha='center', color='#222222')

# ══════════════════════════════════════════════════════════════
#  (a) LONG-TERM TREND - in DAYS
# ══════════════════════════════════════════════════════════════

time_days = np.arange(n_samples) / samples_per_day

ax1.plot(time_days, feature, color=C_GRAY, linewidth=0.7, alpha=0.9)

ax1.set_ylabel('Butane concentration', fontsize=10)
ax1.set_xlim(0, 2)  # ~1.66 days data
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

xticks_a = np.arange(0, 2.5, 0.5)
ax1.set_xticks(xticks_a)
ax1.set_xticklabels([f'{x:.1f}' for x in xticks_a], fontsize=9)
ax1.tick_params(labelsize=9, direction='out', length=3)

# ══════════════════════════════════════════════════════════════
#  (b) SHORT-TERM CYCLE - 24 hours
# ══════════════════════════════════════════════════════════════

n_show_b = min(24 * samples_per_hour, n_samples)
feature_period = feature[:n_show_b]
time_hours = np.arange(n_show_b) / samples_per_hour

ax2.plot(time_hours, feature_period, color=C_ORANGE, linewidth=0.8)

ax2.set_ylabel('Butane concentration', fontsize=10)
ax2.set_xlim(0, n_show_b / samples_per_hour)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

xticks_b = np.arange(0, n_show_b / samples_per_hour + 1, 4)
ax2.set_xticks(xticks_b)
ax2.set_xticklabels([str(int(x)) for x in xticks_b], fontsize=9)
ax2.tick_params(labelsize=9, direction='out', length=3)

# ══════════════════════════════════════════════════════════════
#  (c) INTRA-PERIOD PATTERN OVERLAY
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

ax3.set_ylabel('Butane concentration', fontsize=10)
ax3.set_xlim(0, samples_per_period)
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

hour_ticks = np.arange(0, samples_per_period + 1, 60)
hour_labels = [f'{i//60}:00' for i in hour_ticks]
ax3.set_xticks(hour_ticks)
ax3.set_xticklabels(hour_labels, fontsize=9)

ax3.legend(loc='upper left', fontsize=8, ncol=1, framealpha=0.95,
           edgecolor='#cccccc', handlelength=1.5, labelspacing=0.3)
ax3.tick_params(labelsize=9, direction='out', length=3)

# ══════════════════════════════════════════════════════════════
#  SUBPLOT LABELS
# ══════════════════════════════════════════════════════════════

ax1.text(0.5, -0.16, '(a) Time interval (day)', transform=ax1.transAxes,
         fontsize=10, ha='center')

ax2.text(0.5, -0.16, '(b) Time interval (hour)', transform=ax2.transAxes,
         fontsize=10, ha='center')

ax3.text(0.44, -0.13, '(c) Time interval (minute)', transform=ax3.transAxes,
         fontsize=10, ha='center')

# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

out_path = '/home/user/sssdtcn/Figures4paper/fig_debutanizer_periodicity'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out_path}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out_path}.png/pdf')
print(f'Feature selected: u7 (highest periodicity, autocorr=0.424)')
