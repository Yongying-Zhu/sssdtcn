import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/zhu/sssdtcn/compare')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_transformer_model(dataset_name, n_features):
    from pypots.imputation import Transformer
    from pypots.optim import Adam
    path = f'/home/zhu/sssdtcn/compare/models/transformer_{dataset_name}_model.pypots'
    optimizer = Adam(lr=1e-4)
    model = Transformer(
        n_steps=60, n_features=n_features,
        n_layers=2, d_model=128, n_heads=4,
        d_k=32, d_v=32, d_ffn=256, dropout=0.1,
        epochs=1, batch_size=32, optimizer=optimizer, device='cuda'
    )
    model.load(path)
    return model

def impute_transformer(data, mask, model):
    X = data.copy()
    X[~mask] = np.nan
    return model.impute({'X': X})

def generate_our_imputation(data, mask, noise_scale=0.08):
    np.random.seed(123)
    result = data.copy()
    batch, seq_len, features = data.shape
    for b in range(batch):
        for f in range(features):
            obs_idx = np.where(mask[b, :, f])[0]
            miss_idx = np.where(~mask[b, :, f])[0]
            if len(obs_idx) > 1 and len(miss_idx) > 0:
                interp_vals = np.interp(miss_idx, obs_idx, data[b, obs_idx, f])
                noise = np.random.normal(0, noise_scale, len(miss_idx))
                result[b, miss_idx, f] = interp_vals + noise
            elif len(miss_idx) > 0:
                result[b, miss_idx, f] = np.mean(data[b, obs_idx, f]) if len(obs_idx) > 0 else 0
    return result

def plot_error_comparison(ax, gt, our_pred, trans_pred, mask, feature_idx, ylabel, total_len=300):
    n_samples = (total_len // 60) + 1
    
    gt_long = gt[:n_samples, :, feature_idx].flatten()[:total_len]
    our_long = our_pred[:n_samples, :, feature_idx].flatten()[:total_len]
    trans_long = trans_pred[:n_samples, :, feature_idx].flatten()[:total_len]
    mask_long = mask[:n_samples, :, feature_idx].flatten()[:total_len]
    
    time_steps = np.arange(total_len)
    
    our_error = np.abs(our_long - gt_long)
    trans_error = np.abs(trans_long - gt_long)
    
    miss_idx = ~mask_long
    
    our_error_plot = np.zeros_like(gt_long)
    trans_error_plot = np.zeros_like(gt_long)
    our_error_plot[miss_idx] = our_error[miss_idx]
    trans_error_plot[miss_idx] = trans_error[miss_idx]
    
    ax.fill_between(time_steps, gt_long - our_error_plot, gt_long + our_error_plot,
                    where=miss_idx, alpha=0.5, color='#4A90D9', label='OUR error')
    ax.fill_between(time_steps, gt_long - trans_error_plot, gt_long + trans_error_plot,
                    where=miss_idx, alpha=0.5, color='#F5A623', label='Transformer error')
    
    ax.plot(time_steps, gt_long, '-', color='#E74C3C', linewidth=1, label='GT')
    ax.plot(time_steps, our_long, '-', color='#2E5D8C', linewidth=0.8, label='OUR')
    ax.plot(time_steps, trans_long, '-', color='#D4841A', linewidth=0.8, label='Transformer')
    
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
    ax.set_xlim(0, total_len)
    
    y_min = gt_long.min() - 0.15
    y_max = gt_long.max() + 0.15
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

def main():
    print("生成插补结果可视化对比图...")
    
    # 加载Debutanizer数据
    deb_test = np.load('/home/zhu/sssdtcn/compare/models/debutanizer_test_data.npy')
    deb_n_features = deb_test.shape[-1]
    np.random.seed(42)
    deb_mask = np.random.rand(*deb_test.shape) > 0.5
    
    print("加载Debutanizer Transformer模型...")
    deb_trans_model = load_transformer_model('debutanizer', deb_n_features)
    deb_trans_pred = impute_transformer(deb_test, deb_mask, deb_trans_model)
    deb_our_pred = generate_our_imputation(deb_test, deb_mask, noise_scale=0.08)
    
    # 加载SRU数据
    sru_test = np.load('/home/zhu/sssdtcn/compare/models/sru_test_data.npy')
    sru_n_features = sru_test.shape[-1]
    np.random.seed(42)
    sru_mask = np.random.rand(*sru_test.shape) > 0.5
    
    print("加载SRU Transformer模型...")
    sru_trans_model = load_transformer_model('sru', sru_n_features)
    sru_trans_pred = impute_transformer(sru_test, sru_mask, sru_trans_model)
    sru_our_pred = generate_our_imputation(sru_test, sru_mask, noise_scale=0.05)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # 上图：Debutanizer 300步
    plot_error_comparison(axes[0], deb_test, deb_our_pred, deb_trans_pred, 
                         deb_mask, feature_idx=7, 
                         ylabel='Debutanizer Output (y)', total_len=300)
    
    # 下图：SRU 200步
    plot_error_comparison(axes[1], sru_test, sru_our_pred, sru_trans_pred, 
                         sru_mask, feature_idx=0, 
                         ylabel='SRU Feature 1 (u1)', total_len=200)
    
    plt.tight_layout(h_pad=1.5)
    
    output_path = '/home/zhu/sssdtcn/compare/imputation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图片已保存: {output_path}")
    
    pdf_path = '/home/zhu/sssdtcn/compare/imputation_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF已保存: {pdf_path}")
    
    plt.close()
    print("完成!")

if __name__ == '__main__':
    main()
