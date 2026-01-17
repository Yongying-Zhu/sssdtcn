"""1Hz数据集配置（最终优化版）"""
import torch

class Config1Hz:
    # 数据（已过滤VS1常值传感器）
    data_path = "/home/zhu/sssdtcn/hydraulic_1hz_clean.csv"
    sequence_length = 60
    num_sensors = 7  # CE, CP, SE, TS1, TS2, TS3, TS4
    
    # S4层（适度增强）
    s4_state_dim = 160          # 128→160 (+25%)
    s4_dropout = 0.1
    
    # 卷积（适度增强）
    conv_channels = 40          # 32→40 (+25%)
    conv_kernel_size = 3
    conv_dilation_rates = [1, 2, 4, 8]  # 增加感受野
    
    # Mask Embedding
    use_mask_embedding = True
    mask_embed_dim = 20         # 16→20
    
    # 扩散
    diffusion_steps = 200
    noise_schedule = "cosine"
    
    # 模型
    hidden_dim = 80             # 64→80 (+25%)
    num_layers = 4              # 3→4（更深）
    
    # 训练
    batch_size = 32             # 序列短，可以用大batch
    learning_rate = 8e-5        # 更稳定的学习率
    num_epochs = 180
    
    # 硬件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4
    
    # 保存
    save_dir = "/home/zhu/sssdtcn/checkpoints_1hz"
    log_dir = "/home/zhu/sssdtcn/logs_1hz"
    scaler_path = "/home/zhu/sssdtcn/scaler_1hz.pkl"

config = Config1Hz()
