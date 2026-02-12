"""
使用PyPOTS库训练深度学习插补方法 (适配PyPOTS 1.1)
支持: BRITS, SAITS, Transformer
"""
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/zhu/sssdtcn/compare')
from data_utils import load_sru_data, load_debutanizer_data, prepare_data, create_missing_mask

from pypots.imputation import SAITS, BRITS, Transformer
from pypots.optim import Adam

def prepare_pypots_data(train_data, val_data, missing_ratio=0.5):
    """准备PyPOTS格式的数据"""
    train_mask = create_missing_mask(train_data, missing_ratio)
    train_X = train_data.copy()
    train_X[~train_mask] = np.nan
    
    val_mask = create_missing_mask(val_data, 0.5)
    val_X = val_data.copy()
    val_X[~val_mask] = np.nan
    
    return {'X': train_X}, {'X': val_X, 'X_ori': val_data}

def train_saits(train_dict, val_dict, n_features, dataset_name, epochs=300):
    """训练SAITS模型"""
    print("\n" + "="*50)
    print("Training SAITS...")
    print("="*50)
    
    saving_path = f'/home/zhu/sssdtcn/compare/models/saits_{dataset_name}'
    os.makedirs(saving_path, exist_ok=True)
    
    optimizer = Adam(lr=1e-4, weight_decay=1e-5)
    
    model = SAITS(
        n_steps=60,
        n_features=n_features,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_k=32,
        d_v=32,
        d_ffn=256,
        dropout=0.1,
        epochs=epochs,
        batch_size=32,
        patience=20,
        optimizer=optimizer,
        device='cuda',
        saving_path=saving_path,
        model_saving_strategy='best'
    )
    
    model.fit(train_dict, val_dict)
    model_path = f'/home/zhu/sssdtcn/compare/models/saits_{dataset_name}_model.pypots'
    model.save(model_path)
    print(f"SAITS model saved to {model_path}")
    return model

def train_brits(train_dict, val_dict, n_features, dataset_name, epochs=300):
    """训练BRITS模型"""
    print("\n" + "="*50)
    print("Training BRITS...")
    print("="*50)
    
    saving_path = f'/home/zhu/sssdtcn/compare/models/brits_{dataset_name}'
    os.makedirs(saving_path, exist_ok=True)
    
    optimizer = Adam(lr=1e-4, weight_decay=1e-5)
    
    model = BRITS(
        n_steps=60,
        n_features=n_features,
        rnn_hidden_size=128,
        epochs=epochs,
        batch_size=32,
        patience=20,
        optimizer=optimizer,
        device='cuda',
        saving_path=saving_path,
        model_saving_strategy='best'
    )
    
    model.fit(train_dict, val_dict)
    model_path = f'/home/zhu/sssdtcn/compare/models/brits_{dataset_name}_model.pypots'
    model.save(model_path)
    print(f"BRITS model saved to {model_path}")
    return model

def train_transformer(train_dict, val_dict, n_features, dataset_name, epochs=300):
    """训练Transformer模型"""
    print("\n" + "="*50)
    print("Training Transformer...")
    print("="*50)
    
    saving_path = f'/home/zhu/sssdtcn/compare/models/transformer_{dataset_name}'
    os.makedirs(saving_path, exist_ok=True)
    
    optimizer = Adam(lr=1e-4, weight_decay=1e-5)
    
    model = Transformer(
        n_steps=60,
        n_features=n_features,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_k=32,
        d_v=32,
        d_ffn=256,
        dropout=0.1,
        epochs=epochs,
        batch_size=32,
        patience=20,
        optimizer=optimizer,
        device='cuda',
        saving_path=saving_path,
        model_saving_strategy='best'
    )
    
    model.fit(train_dict, val_dict)
    model_path = f'/home/zhu/sssdtcn/compare/models/transformer_{dataset_name}_model.pypots'
    model.save(model_path)
    print(f"Transformer model saved to {model_path}")
    return model

def train_all_pypots_methods(dataset_name, data_path, selected_features=None, epochs=300):
    """训练所有PyPOTS方法"""
    print(f"\n{'#'*60}")
    print(f"# Training PyPOTS methods on {dataset_name.upper()}")
    print(f"{'#'*60}")
    
    if dataset_name == 'sru':
        data = load_sru_data(data_path, selected_features)
    else:
        data = load_debutanizer_data(data_path)
    
    n_features = data.shape[1]
    print(f"Number of features: {n_features}")
    
    train_data, val_data, test_data, scaler = prepare_data(data)
    
    scaler_path = f'/home/zhu/sssdtcn/compare/models/{dataset_name}_scaler.pkl'
    if not os.path.exists(scaler_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    test_path = f'/home/zhu/sssdtcn/compare/models/{dataset_name}_test_data.npy'
    if not os.path.exists(test_path):
        np.save(test_path, test_data)
    
    train_dict, val_dict = prepare_pypots_data(train_data, val_data, missing_ratio=0.5)
    
    train_saits(train_dict, val_dict, n_features, dataset_name, epochs)
    train_brits(train_dict, val_dict, n_features, dataset_name, epochs)
    train_transformer(train_dict, val_dict, n_features, dataset_name, epochs)
    
    print(f"\n{'#'*60}")
    print(f"# All PyPOTS methods training completed for {dataset_name.upper()}!")
    print(f"{'#'*60}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['sru', 'debutanizer'])
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    
    if args.dataset == 'sru':
        train_all_pypots_methods('sru', '/home/zhu/sssdtcn/SRU_data.txt', [0, 2], args.epochs)
    else:
        train_all_pypots_methods('debutanizer', '/home/zhu/sssdtcn/debutanizer_data.txt', None, args.epochs)
