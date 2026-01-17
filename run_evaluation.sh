#!/bin/bash

# ========================================
# 快速评估脚本
# ========================================

cd /home/zhu/sssdtcn
source AnYujin/bin/activate

echo "=========================================="
echo "隐式显式扩散模型 - 评估"
echo "=========================================="
echo ""
echo "测试配置:"
echo "  - 设备: A40 (CUDA:0)"
echo "  - 缺失率: 20%, 30%, 40%, 50%, 60%, 70%, 80%"
echo ""
echo "开始评估..."
echo "=========================================="

# 检查模型是否存在
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo ""
    echo "错误: 找不到训练好的模型!"
    echo "请先运行训练脚本: ./run_training.sh"
    echo ""
    exit 1
fi

# 评估
CUDA_VISIBLE_DEVICES=0 python evaluate.py

echo ""
echo "=========================================="
echo "评估完成!"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - 指标CSV: results/metrics/results_all_missing_rates.csv"
echo "  - 插补可视化: results/figures/imputation_mrXX.png"
echo "  - 指标曲线: results/figures/metrics_vs_missing_rate.png"
echo ""
echo "打开CSV文件:"
echo "  cat results/metrics/results_all_missing_rates.csv"
echo "=========================================="
