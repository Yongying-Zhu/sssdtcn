import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

log_file = 'checkpoints/sru_f1f3/training_log.json'

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    plt.figure(figsize=(14, 7))
    plt.plot(train_loss, label='Training Loss', linewidth=2.5, color='#1f77b4', alpha=0.9)
    plt.plot(val_loss, label='Validation Loss', linewidth=2.5, color='#ff7f0e', alpha=0.9)
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=16, fontweight='bold')
    plt.title('SRU Dataset (Feature 1 & 3) - Training Curve (300 Epochs)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=14, loc='upper right', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    output_path = 'checkpoints/sru_f1f3/loss_curve_300epochs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print("\n" + "="*70)
    print("✅ Loss curve saved successfully!")
    print(f"   Location: {output_path}")
    print(f"   Total epochs: {len(train_loss)}")
    print(f"   Final train loss: {train_loss[-1]:.6f}")
    print(f"   Final val loss: {val_loss[-1]:.6f}")
    print(f"   Best val loss: {min(val_loss):.6f} (Epoch {val_loss.index(min(val_loss))+1})")
    print("="*70 + "\n")
else:
    print(f"❌ Training log not found: {log_file}")
