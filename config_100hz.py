"""100Hz数据集配置（加速优化版）"""
import torch

class Config100Hz:
    data_path = "/home/zhu/sssdtcn/hydraulic_100hz_clean.csv"
    sequence_length = 6000
    num_sensors = 7
    
    # S4层（保持容量）
    s4_state_dim = 256
    s4_dropout = 0.1
    
    # 卷积（保持容量）
    conv_channels = 64
    conv_kernel_size = 3
    conv_dilation_rates = [1, 2, 4, 8]
    
    # Mask Embedding
    use_mask_embedding = True
    mask_embed_dim = 16
    
    # 扩散（轻微减少）
    diffusion_steps = 150       # 200→150 (加速25%)
    noise_schedule = "cosine"
    
    # 模型（保持容量）
    hidden_dim = 128
    num_layers = 4
    
    # 训练（关键优化）
    batch_size = 18             # 16→18 (平衡速度和显存)
    learning_rate = 1e-4
    num_epochs = 40             # 150→40 (总时间大幅减少)
    
    # 硬件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4
    
    # 保存
    save_dir = "/home/zhu/sssdtcn/checkpoints_100hz"
    log_dir = "/home/zhu/sssdtcn/logs_100hz"
    scaler_path = "/home/zhu/sssdtcn/scaler_100hz.pkl"

config = Config100Hz()
