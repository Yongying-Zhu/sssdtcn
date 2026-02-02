#!/bin/bash
# ==============================================================================
# 消融实验脚本 - Debutanizer数据集
# ==============================================================================
#
# 实验设置：
#   1. 单尺度 + 显式特征: 使用dilation=1的因果卷积 + S4Layer
#   2. 仅显式特征: 仅使用S4Layer，移除因果卷积模块
#
# 对比基准：
#   - 完整模型（多尺度扩张卷积 + S4Layer）
#
# ==============================================================================

# 切换到Ablation目录
cd /home/user/sssdtcn/Ablation

# 创建日志目录
mkdir -p logs
mkdir -p checkpoints/single_scale
mkdir -p checkpoints/explicit_only

echo "=========================================="
echo "消融实验开始"
echo "数据集: Debutanizer"
echo "=========================================="

# 训练单尺度+显式特征模型
echo ""
echo "[1/2] 训练单尺度 + 显式特征模型..."
echo "模型: SingleScaleDiffusionModel"
echo "特点: dilation=1 (无多尺度), 保留S4Layer"
echo ""

nohup python train_single_scale.py > logs/train_single_scale.log 2>&1 &
SINGLE_PID=$!
echo "单尺度模型训练进程 PID: $SINGLE_PID"
echo "日志: logs/train_single_scale.log"

# 等待单尺度模型训练完成
wait $SINGLE_PID
echo "单尺度模型训练完成!"

# 训练仅显式特征模型
echo ""
echo "[2/2] 训练仅显式特征模型..."
echo "模型: ExplicitOnlyDiffusionModel"
echo "特点: 无扩张卷积, 仅使用S4Layer"
echo ""

nohup python train_explicit_only.py > logs/train_explicit_only.log 2>&1 &
EXPLICIT_PID=$!
echo "仅显式模型训练进程 PID: $EXPLICIT_PID"
echo "日志: logs/train_explicit_only.log"

# 等待仅显式模型训练完成
wait $EXPLICIT_PID
echo "仅显式模型训练完成!"

# 评估所有模型
echo ""
echo "=========================================="
echo "评估消融实验模型"
echo "=========================================="
python evaluate_ablation.py

echo ""
echo "=========================================="
echo "消融实验完成!"
echo "=========================================="
echo ""
echo "结果文件: checkpoints/ablation_results.xlsx"
echo ""
echo "模型保存位置:"
echo "  - 单尺度模型: checkpoints/single_scale/single_scale_best.pth"
echo "  - 仅显式模型: checkpoints/explicit_only/explicit_only_best.pth"
