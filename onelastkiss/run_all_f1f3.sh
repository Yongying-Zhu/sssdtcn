#!/bin/bash

echo "========================================"
echo "Step 1: Training (300 epochs)"
echo "========================================"
python train_f1f3_300epochs.py

echo ""
echo "========================================"
echo "Step 2: Plotting loss curve"
echo "========================================"
python plot_loss_f1f3.py

echo ""
echo "========================================"
echo "Step 3: Evaluation on test set"
echo "========================================"
python evaluate_f1f3.py

echo ""
echo "========================================"
echo "ALL COMPLETED!"
echo "========================================"
echo "Results:"
echo "  - Model: checkpoints/sru_f1f3/best_model.pth"
echo "  - Loss curve: checkpoints/sru_f1f3/loss_curve_300epochs.png"
echo "  - Evaluation results: checkpoints/sru_f1f3/evaluation_results.txt"
echo "========================================"
