"""
配置文件 - Debutanizer数据集（消融实验）
"""

class ConfigDebutanizer:
    # 数据配置
    data_path = '/home/user/sssdtcn/debutanizer_data.txt'
    dataset_name = 'Debutanizer'
    num_features = 8  # u1-u7, y

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
    save_dir = './checkpoints'
