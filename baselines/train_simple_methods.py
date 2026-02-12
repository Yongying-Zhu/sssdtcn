"""
简单插补方法：Median 和 Last
这些方法不需要训练，但为保持一致性，保存为"模型"文件
"""
import numpy as np
import pickle
import os
import sys
sys.path.append('/home/zhu/sssdtcn/compare')
from data_utils import load_sru_data, load_debutanizer_data, prepare_data

class MedianImputer:
    """中位数插补器"""
    def __init__(self):
        self.medians = None
    
    def fit(self, data):
        # data: [N, seq_len, features]
        # 计算每个特征的中位数
        flat_data = data.reshape(-1, data.shape[-1])
        self.medians = np.nanmedian(flat_data, axis=0)
        return self
    
    def transform(self, data, mask):
        # 用中位数填充缺失值
        result = data.copy()
        for f in range(data.shape[-1]):
            result[:, :, f] = np.where(mask[:, :, f], data[:, :, f], self.medians[f])
        return result
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'medians': self.medians}, f)

class LastImputer:
    """前向填充插补器 (Last Observation Carried Forward)"""
    def __init__(self):
        self.global_means = None  # 用于处理序列开头的缺失
    
    def fit(self, data):
        flat_data = data.reshape(-1, data.shape[-1])
        self.global_means = np.nanmean(flat_data, axis=0)
        return self
    
    def transform(self, data, mask):
        result = data.copy()
        batch, seq_len, features = data.shape
        for b in range(batch):
            for f in range(features):
                last_valid = self.global_means[f]
                for t in range(seq_len):
                    if mask[b, t, f]:
                        last_valid = data[b, t, f]
                        result[b, t, f] = data[b, t, f]
                    else:
                        result[b, t, f] = last_valid
        return result
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'global_means': self.global_means}, f)

def train_simple_methods(dataset_name, data_path, selected_features=None):
    """训练简单方法"""
    print(f"\n{'='*50}")
    print(f"Training simple methods on {dataset_name}")
    print(f"{'='*50}")
    
    # 加载数据
    if dataset_name == 'sru':
        data = load_sru_data(data_path, selected_features)
    else:
        data = load_debutanizer_data(data_path)
    
    # 准备数据
    train_data, val_data, test_data, scaler = prepare_data(data)
    
    # 保存scaler
    scaler_path = f'/home/zhu/sssdtcn/compare/models/{dataset_name}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # 训练Median
    print("\nTraining Median imputer...")
    median_imputer = MedianImputer()
    median_imputer.fit(train_data)
    median_path = f'/home/zhu/sssdtcn/compare/models/median_{dataset_name}_model.pkl'
    median_imputer.save(median_path)
    print(f"Median model saved to {median_path}")
    
    # 训练Last
    print("\nTraining Last imputer...")
    last_imputer = LastImputer()
    last_imputer.fit(train_data)
    last_path = f'/home/zhu/sssdtcn/compare/models/last_{dataset_name}_model.pkl'
    last_imputer.save(last_path)
    print(f"Last model saved to {last_path}")
    
    # 保存测试数据供后续评估使用
    test_path = f'/home/zhu/sssdtcn/compare/models/{dataset_name}_test_data.npy'
    np.save(test_path, test_data)
    print(f"Test data saved to {test_path}")
    
    print(f"\nSimple methods training completed for {dataset_name}!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['sru', 'debutanizer'])
    args = parser.parse_args()
    
    if args.dataset == 'sru':
        train_simple_methods('sru', '/home/zhu/sssdtcn/SRU_data.txt', [0, 2])
    else:
        train_simple_methods('debutanizer', '/home/zhu/sssdtcn/debutanizer_data.txt')
