"""
Analyze periodicity of all 7 features in debutanizer dataset.
Select the one with the most pronounced periodic pattern.
"""
import numpy as np

data = np.loadtxt('/home/user/sssdtcn/debutanizer_data.txt', skiprows=5)
# Columns: u1, u2, u3, u4, u5, u6, u7, y

feature_names = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y']

print("=" * 60)
print("Periodicity Analysis of Debutanizer Features")
print("=" * 60)

# Analyze each feature
samples_per_period = 480  # 8 hours
n_samples = len(data)
n_periods = n_samples // samples_per_period

for i, name in enumerate(feature_names):
    feature = data[:, i]

    # Calculate autocorrelation at lag = period length
    # Higher autocorrelation = stronger periodicity
    if len(feature) > samples_per_period:
        # Autocorrelation
        mean = np.mean(feature)
        var = np.var(feature)
        if var > 0:
            autocorr = np.correlate(feature - mean, feature - mean, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            # Get autocorrelation at period lag
            period_autocorr = autocorr[samples_per_period] if samples_per_period < len(autocorr) else 0

            # Calculate variance between periods (lower = more consistent pattern)
            period_means = []
            for p in range(n_periods):
                start = p * samples_per_period
                end = start + samples_per_period
                if end <= n_samples:
                    period_means.append(np.mean(feature[start:end]))
            period_variance = np.var(period_means) if period_means else 0

            # Calculate within-period consistency
            # Overlay periods and calculate point-wise variance
            overlaid = []
            for p in range(n_periods):
                start = p * samples_per_period
                end = start + samples_per_period
                if end <= n_samples:
                    overlaid.append(feature[start:end])
            if overlaid:
                overlaid = np.array(overlaid)
                within_period_var = np.mean(np.var(overlaid, axis=0))
            else:
                within_period_var = 0

            print(f"\n{name}:")
            print(f"  Range: [{feature.min():.3f}, {feature.max():.3f}]")
            print(f"  Std: {np.std(feature):.3f}")
            print(f"  Autocorr at period lag: {period_autocorr:.3f}")
            print(f"  Between-period variance: {period_variance:.5f}")
            print(f"  Within-period variance: {within_period_var:.5f}")
            print(f"  Periodicity score (higher=better): {period_autocorr / (within_period_var + 0.001):.2f}")

print("\n" + "=" * 60)
print("RECOMMENDATION: Select feature with highest periodicity score")
print("=" * 60)
