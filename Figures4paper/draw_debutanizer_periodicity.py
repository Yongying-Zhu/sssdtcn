"""
Debutanizer Dataset Multi-scale Periodicity Visualization
Shows that the feature to be imputed has periodic patterns at different time scales.
Follows the template style exactly.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

data = np.loadtxt('/home/user/sssdtcn/debutanizer_data.txt', skiprows=5)
# Select y (target variable - butane content)
y = data[:, 7]
n_samples = len(y)

# Time parameters (assuming 1-minute sampling)
samples_per_hour = 60
samples_per_period = 480  # 8 hours = one operational period

# ══════════════════════════════════════════════════════════════
#  CREATE FIGURE (exact template layout)
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 7.5))

# Layout matching template
ax1 = fig.add_axes([0.08, 0.58, 0.40, 0.35])   # (a) top-left
ax2 = fig.add_axes([0.56, 0.58, 0.40, 0.35])   # (b) top-right
ax3 = fig.add_axes([0.08, 0.10, 0.88, 0.38])   # (c) bottom full width

# Colors matching template
C_BLUE = '#4A90D9'    # Long-term (blue)
C_GRAY = '#888888'    # Reference (gray)
C_ORANGE = '#E8853D'  # Short-term (orange)

# ══════════════════════════════════════════════════════════════
#  TITLE with Long-term / Short-term labels (like template)
# ══════════════════════════════════════════════════════════════

fig.text(0.30, 0.96, 'Butane content in Debutanizer', fontsize=12, fontweight='bold', ha='center')
fig.text(0.72, 0.96, 'Long-term', fontsize=11, fontweight='bold', color=C_BLUE, ha='center')
fig.text(0.85, 0.96, 'Short-term', fontsize=11, fontweight='bold', color=C_ORANGE, ha='center')

# ══════════════════════════════════════════════════════════════
#  (a) LONG-TERM TREND - Shows multiple periods
# ══════════════════════════════════════════════════════════════

# Show entire dataset with hour-based x-axis
time_hours = np.arange(n_samples) / samples_per_hour
y_scaled = y.copy()

# Plot main line (blue)
ax1.plot(time_hours, y_scaled, color=C_BLUE, linewidth=0.8, alpha=0.95)

# Plot shifted version to show periodicity (gray)
shift = samples_per_period
y_shifted = np.roll(y_scaled, -shift)
time_shifted = time_hours[:-shift]
ax1.plot(time_shifted, y_shifted[:len(time_shifted)], color=C_GRAY, linewidth=0.7, alpha=0.6)

ax1.set_ylabel('Butane content', fontsize=10)
ax1.set_xlim(0, 40)

# X-axis ticks as period markers
period_hours = 8
xticks = np.arange(0, 41, period_hours)
ax1.set_xticks(xticks)
xticklabels = [f'P{i+1}' for i in range(len(xticks))]
xticklabels[0] = '0'
ax1.set_xticklabels(xticklabels, fontsize=9)

ax1.tick_params(labelsize=9)
ax1.set_ylim(0, 1.05)

# ══════════════════════════════════════════════════════════════
#  (b) SHORT-TERM CYCLE - One period as continuous line (like template)
# ══════════════════════════════════════════════════════════════

# Show one complete period (like one week in template)
n_show = samples_per_period  # 8 hours
y_period = y[:n_show]
time_period = np.arange(n_show) / samples_per_hour

ax2.plot(time_period, y_period, color=C_ORANGE, linewidth=1.0)

ax2.set_ylabel('Butane content', fontsize=10)
ax2.set_xlim(0, 8)

# X-axis ticks as hours
xticks_b = np.arange(0, 9, 1)
ax2.set_xticks(xticks_b)
ax2.set_xticklabels(['0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h'], fontsize=9)

ax2.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  (c) INTRA-PERIOD PATTERN - Multiple periods overlaid (like weekdays)
# ══════════════════════════════════════════════════════════════

n_periods = n_samples // samples_per_period
time_minutes = np.arange(samples_per_period)

# Period colors matching template weekday colors
period_colors = ['#4A90D9', '#E8853D', '#2ECC71', '#E74C3C', '#9B59B6', '#34495E', '#F39C12']
period_labels = ['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5']

for i in range(min(5, n_periods)):
    start_idx = i * samples_per_period
    end_idx = start_idx + samples_per_period
    if end_idx <= n_samples:
        y_period = y[start_idx:end_idx]
        ax3.plot(time_minutes, y_period, color=period_colors[i],
                 linewidth=0.9, alpha=0.85, label=period_labels[i])

ax3.set_ylabel('Butane content', fontsize=10)
ax3.set_xlim(0, samples_per_period)

# X-axis: time within period (like 00:00-23:00 in template)
hour_ticks = np.arange(0, samples_per_period + 1, 60)
hour_labels = [f'{i//60}:00' for i in hour_ticks]
ax3.set_xticks(hour_ticks)
ax3.set_xticklabels(hour_labels, fontsize=8)

# Legend (matching template - inside plot area)
ax3.legend(loc='upper left', fontsize=8, ncol=1, framealpha=0.9,
           handlelength=1.5)

ax3.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════
#  SUBPLOT LABELS (a), (b), (c) - matching template style
# ══════════════════════════════════════════════════════════════

ax1.text(0.5, -0.18, '(a) Time interval (period)', transform=ax1.transAxes,
         fontsize=10, ha='center')
ax2.text(0.5, -0.18, '(b) Time interval (hours)', transform=ax2.transAxes,
         fontsize=10, ha='center', color=C_ORANGE)
ax3.text(0.5, -0.15, '(c) Time interval (minutes)', transform=ax3.transAxes,
         fontsize=10, ha='center', color=C_ORANGE)

# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

out_path = '/home/user/sssdtcn/Figures4paper/fig_debutanizer_periodicity'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out_path}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out_path}.png/pdf')
