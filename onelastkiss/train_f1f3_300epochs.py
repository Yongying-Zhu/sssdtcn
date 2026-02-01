import os
import sys
sys.path.insert(0, '/home/user/sssdtcn')
import torch
from torch.optim import AdamW
from tqdm import tqdm
import json

from models.implicit_explicit_diffusion import ImplicitExplicitDiffusionModel
from onelastkiss.data_loader_f1f3 import load_data_f1f3, create_random_mask

class Config:
    data_path = '/home/user/sssdtcn/SRU_data.txt'
    num_features = 2
    sequence_length = 60
    batch_size = 32
    train_ratio = 0.7
    val_ratio = 0.1
    hidden_dim = 128
    embedding_dim = 256
    num_residual_layers = 6
    dropout = 0.15
    num_epochs = 300
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = 'cuda:0'
    save_dir = './checkpoints/sru_f1f3'

config = Config()

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_data in pbar:
        batch_data = batch_data.to(device)
        batch_size = batch_data.size(0)
        
        mask, masked_data = create_random_mask(batch_data, missing_ratio=0.5)
        mask = mask.to(device)
        masked_data = masked_data.to(device)
        
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
        predicted_data = model(masked_data, timesteps, mask)
        
        mask_missing = (mask == 0)
        loss = ((predicted_data - batch_data) ** 2 * mask_missing.float()).sum() / mask_missing.sum()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)
            
            mask, masked_data = create_random_mask(batch_data, missing_ratio=0.5)
            mask = mask.to(device)
            masked_data = masked_data.to(device)
            
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            predicted_data = model(masked_data, timesteps, mask)
            
            mask_missing = (mask == 0)
            loss = ((predicted_data - batch_data) ** 2 * mask_missing.float()).sum() / mask_missing.sum()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    print("="*70)
    print("Training Diffusion Model - SRU Dataset")
    print("Features: 2 (Feature 1 and Feature 3 only)")
    print(f"Epochs: {config.num_epochs}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print("="*70)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, scaler, num_features = load_data_f1f3(
        config.data_path, config.sequence_length, config.batch_size, 
        config.train_ratio, config.val_ratio
    )
    
    print("\nCreating model...")
    model = ImplicitExplicitDiffusionModel(
        input_dim=num_features, hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_residual_layers=config.num_residual_layers,
        dropout=config.dropout
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    os.makedirs(config.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {config.num_epochs} epochs...\n")
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_loss = validate(model, val_loader, config.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'scaler': scaler
            }, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"  -> Best model saved (val_loss: {val_loss:.6f})")
    
    log_data = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'config': {
            'num_features': config.num_features,
            'num_epochs': config.num_epochs,
            'hidden_dim': config.hidden_dim
        }
    }
    with open(os.path.join(config.save_dir, 'training_log.json'), 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved: {config.save_dir}/best_model.pth")

if __name__ == '__main__':
    main()
