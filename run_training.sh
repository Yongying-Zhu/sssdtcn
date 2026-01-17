#!/bin/bash

# ========================================
# 快速训练脚本
# ========================================

cd /home/zhu/sssdtcn
source AnYujin/bin/activate

echo "=========================================="
echo "隐式显式扩散模型 - 训练"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  - 设备: A40 (CUDA:0)"
echo "  - 数据: LD2011_2014.txt"
echo "  - 缺失率: 20%"
echo "  - 训练轮数: 100"
echo ""
echo "开始训练..."
echo "=========================================="

# 清理旧结果
rm -rf logs/* results/* checkpoints/* 2>/dev/null || true

# 重新创建目录
mkdir -p logs results/figures results/metrics checkpoints

# 训练
CUDA_VISIBLE_DEVICES=0 python train.py

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - 训练曲线: logs/training_curve.png"
echo "  - 最佳模型: checkpoints/best_model.pt"
echo "  - TensorBoard: tensorboard --logdir=logs"
echo ""
echo "下一步: 运行评估脚本"
echo "  ./run_evaluation.sh"
echo "=========================================="
