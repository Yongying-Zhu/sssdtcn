"""通用训练脚本"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

from universal_data_loader import get_dataloaders

sys.path.append('/home/zhu/sssdtcn')
from implicit_explicit_diffusion import ImplicitExplicitDiffusionModel

def train_epoch(model, train_loader, optimizer, config, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs}')
    
    for x in pbar:
        x = x.to(config.device)
        mask_ratio = np.random.uniform(0.2, 0.8)
        mask = torch.rand_like(x) > mask_ratio
        
        optimizer.zero_grad()
        loss = model.compute_loss(x, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(config.device)
            mask = torch.rand_like(x) > 0.5
            loss = model.compute_loss(x, mask)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_model(config, dataset_name):
    print(f"{'='*80}")
    print(f"{dataset_name}数据集训练")
    print(f"{'='*80}")
    
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_path, config.sequence_length, config.batch_size, config.scaler_path
    )
    
    print(f"\n创建模型...")
    model = ImplicitExplicitDiffusionModel(
        num_sensors=config.num_sensors,
        sequence_length=config.sequence_length,
        s4_state_dim=config.s4_state_dim,
        conv_channels=config.conv_channels,
        conv_kernel_size=config.conv_kernel_size,
        conv_dilation_rates=config.conv_dilation_rates,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        diffusion_steps=config.diffusion_steps,
        noise_schedule=config.noise_schedule,
        use_mask_embedding=config.use_mask_embedding,
        mask_embed_dim=config.mask_embed_dim
    ).to(config.device)
    
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    writer = SummaryWriter(config.log_dir)
    
    print(f"\n开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config, epoch)
        val_loss = validate(model, val_loader, config)
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, f"{config.save_dir}/best_model.pth")
            print(f"  ✓ 最佳模型已保存")
        
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, f"{config.save_dir}/checkpoint_epoch_{epoch}.pth")
    
    writer.close()
    print(f"\n训练完成！最佳Val Loss: {best_val_loss:.4f}")
    return best_val_loss
