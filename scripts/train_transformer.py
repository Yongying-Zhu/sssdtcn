"""
Transformer训练脚本
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append('/home/zhu/sssdtcn')
from universal_data_loader import get_dataloaders
from transformer_model import TransformerImputer

def train_transformer(config, dataset_name):
    """训练Transformer"""
    print("="*80)
    print(f"Training Transformer - {dataset_name} Dataset")
    print("="*80)
    print()
    
    # 加载数据
    train_loader, val_loader, _ = get_dataloaders(
        config['data_path'],
        config['sequence_length'],
        config['batch_size'],
        config['scaler_path']
    )
    
    # 创建模型
    model = TransformerImputer(
        input_dim=config['num_sensors'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(config['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    best_val_loss = float('inf')
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        # 训练
        model.train()
        train_losses = []
        
        for x in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False):
            x = x.to(config['device'])
            mask = torch.rand_like(x) > 0.3  # 30%缺失率训练
            
            optimizer.zero_grad()
            loss = model.compute_loss(x, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 验证
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(config['device'])
                mask = torch.rand_like(x) > 0.3
                
                loss = model.compute_loss(x, mask)
                val_losses.append(loss.item())
        
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, f"{config['save_dir']}/best_model.pth")
            print("  ✓ Best model saved")
    
    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    config = {
        'data_path': '/home/zhu/sssdtcn/hydraulic_1hz_clean.csv',
        'sequence_length': 60,
        'num_sensors': 7,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'scaler_path': '/home/zhu/sssdtcn/scaler_1hz.pkl',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': '/home/zhu/sssdtcn/baselines/transformer/checkpoints_1hz'
    }
    
    train_transformer(config, "1Hz")
