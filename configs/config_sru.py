"""
配置文件 - SRU数据集
"""

class ConfigSRU:
    # 数据配置
    data_path = '/home/user/sssdtcn/SRU_data.txt'
    dataset_name = 'SRU'
    num_features = 7  # u1-u5, y1-y2

    # 序列配置
    sequence_length = 60
    batch_size = 32
    train_ratio = 0.7
    val_ratio = 0.1

    # 模型配置
    hidden_dim = 128
    embedding_dim = 256
    num_residual_layers = 6
    dropout = 0.15

    # 训练配置
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = 'cuda:0'

    # 早停配置
    early_stop_patience = 10
    early_stop_threshold = 0.005

    # 评估配置
    missing_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # 模型保存
    save_dir = './checkpoints/sru'

    # Transformer配置（用于融合）
    transformer_model_path = './checkpoints/sru/transformer_best.pth'
