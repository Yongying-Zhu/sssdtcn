#!/bin/bash
# ==============================================================================
# 消融实验脚本（并行训练） - Debutanizer数据集
# ==============================================================================
#
# 注意: 此脚本会同时训练两个模型，需要足够的GPU显存
# 如果显存不足，请使用 run_ablation.sh（顺序训练）
#
# ==============================================================================

cd /home/user/sssdtcn/Ablation

mkdir -p logs
mkdir -p checkpoints/single_scale
mkdir -p checkpoints/explicit_only

echo "=========================================="
echo "消融实验开始（并行模式）"
echo "数据集: Debutanizer"
echo "=========================================="

# 并行训练两个模型
echo ""
echo "同时启动两个模型训练..."
echo ""

# 启动单尺度模型（GPU 0）
CUDA_VISIBLE_DEVICES=0 nohup python train_single_scale.py > logs/train_single_scale.log 2>&1 &
SINGLE_PID=$!
echo "[1] 单尺度模型 PID: $SINGLE_PID (GPU 0)"
echo "    日志: logs/train_single_scale.log"

# 启动仅显式模型（GPU 1，如果只有一个GPU则等待）
CUDA_VISIBLE_DEVICES=0 nohup python train_explicit_only.py > logs/train_explicit_only.log 2>&1 &
EXPLICIT_PID=$!
echo "[2] 仅显式模型 PID: $EXPLICIT_PID (GPU 0)"
echo "    日志: logs/train_explicit_only.log"

echo ""
echo "=========================================="
echo "训练进程已启动"
echo "=========================================="
echo ""
echo "查看训练进度:"
echo "  tail -f logs/train_single_scale.log"
echo "  tail -f logs/train_explicit_only.log"
echo ""
echo "等待训练完成后运行评估:"
echo "  python evaluate_ablation.py"
