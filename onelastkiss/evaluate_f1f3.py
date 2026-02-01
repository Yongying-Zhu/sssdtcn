import os
import sys
sys.path.insert(0, '/home/user/sssdtcn')
import torch
import numpy as np
from onelastkiss.data_loader_f1f3 import load_data_f1f3, create_random_mask
from models.implicit_explicit_diffusion import ImplicitExplicitDiffusionModel

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
    device = 'cuda:0'
    save_dir = './checkpoints/sru_f1f3'
    missing_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

config = Config()

def evaluate_model(model, test_loader, device, missing_ratio):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)
            
            mask, masked_data = create_random_mask(batch_data, missing_ratio)
            mask = mask.to(device)
            masked_data = masked_data.to(device)
            
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            predicted_data = model(masked_data, timesteps, mask)
            
            mask_missing = (mask == 0)
            pred_missing = predicted_data[mask_missing].cpu().numpy()
            target_missing = batch_data[mask_missing].cpu().numpy()
            
            all_predictions.extend(pred_missing)
            all_targets.extend(target_missing)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    return mae, rmse

def main():
    print("="*70)
    print("Evaluating Model - SRU Dataset (Feature 1 & 3)")
    print("="*70)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, scaler, num_features = load_data_f1f3(
        config.data_path, config.sequence_length, config.batch_size,
        config.train_ratio, config.val_ratio
    )
    
    print("\nLoading model...")
    model = ImplicitExplicitDiffusionModel(
        input_dim=num_features, hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_residual_layers=config.num_residual_layers,
        dropout=config.dropout
    ).to(config.device)
    
    checkpoint = torch.load(os.path.join(config.save_dir, 'best_model.pth'), 
                           map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    print("\nEvaluating on different missing ratios...")
    results = []
    
    for ratio in config.missing_ratios:
        print(f"\nMissing Ratio: {ratio}")
        mae, rmse = evaluate_model(model, test_loader, config.device, ratio)
        results.append({
            'missing_ratio': ratio,
            'mae': mae,
            'rmse': rmse
        })
        print(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    
    output_file = os.path.join(config.save_dir, 'evaluation_results.txt')
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Evaluation Results - SRU Dataset (Feature 1 & 3 Only)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: Implicit-Explicit Diffusion Model\n")
        f.write(f"Features: Feature 1 (u1) and Feature 3 (u3)\n")
        f.write(f"Hidden Dim: {config.hidden_dim}\n")
        f.write(f"Test Sequences: {len(test_loader.dataset)}\n\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Missing Ratio':<15} {'MAE':<15} {'RMSE':<15}\n")
        f.write("-"*70 + "\n")
        
        for res in results:
            f.write(f"{res['missing_ratio']:<15.1f} {res['mae']:<15.6f} {res['rmse']:<15.6f}\n")
        
        f.write("-"*70 + "\n\n")
        avg_mae = np.mean([r['mae'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        f.write(f"Average MAE:  {avg_mae:.6f}\n")
        f.write(f"Average RMSE: {avg_rmse:.6f}\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for res in results:
        print(f"Missing Ratio {res['missing_ratio']:.1f} - MAE: {res['mae']:.6f}, RMSE: {res['rmse']:.6f}")
    print(f"\nAverage MAE:  {avg_mae:.6f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print("="*70)

if __name__ == '__main__':
    main()
