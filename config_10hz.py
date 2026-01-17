"""10Hz数据集训练配置"""
import torch
import os

class Config10Hz:
    data_path = "/home/zhu/sssdtcn/hydraulic_10hz_clean.csv"
    sequence_length = 600
    num_sensors = 2
    
    # S4层
    s4_state_dim = 192  # 中等序列长度
    s4_dropout = 0.1
    
    # 卷积
    conv_channels = 48
    conv_kernel_size = 3
    conv_dilation_rates = [1, 2, 4, 8]
    
    # Mask Embedding
    use_mask_embedding = True
    mask_embed_dim = 24
    
    # 扩散
    diffusion_steps = 200
    noise_schedule = "cosine"
    
    # 模型
    hidden_dim = 96
    num_layers = 3
    
    # 训练
    batch_size = 24  # 中等序列
    learning_rate = 1e-4
    num_epochs = 150
    
    # 硬件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4
    
    # 保存
    save_dir = "/home/zhu/sssdtcn/checkpoints_10hz"
    log_dir = "/home/zhu/sssdtcn/logs_10hz"
    scaler_path = "/home/zhu/sssdtcn/scaler_10hz.pkl"

sensor_file = "/home/zhu/sssdtcn/hydraulic_10hz_clean_sensors.txt"
if os.path.exists(sensor_file):
    with open(sensor_file) as f:
        Config10Hz.num_sensors = len([l for l in f if l.strip()])

config = Config10Hz()
