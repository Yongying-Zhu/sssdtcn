# An Implicit-Explicit Diffusion Model for Industrial Data Imputation

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

---

<a name="english"></a>
## 📖 English Documentation

### Overview

This repository implements **an Implicit-Explicit Diffusion Model** for **time series imputation** in industrial process control systems. The model architecture is called **SSSDTCN**, which integrates:

- 🔹 **Structured State Space Models (S4)**: Efficiently model long-range dependencies and temporal dynamics
- 🔹 **Dilated Temporal Convolutions**: Multi-scale causal convolutions capture patterns at different time scales
- 🔹 **Implicit-Explicit Fusion**: Combines implicit feature extraction (dilated convolutions) with explicit modeling (state space models)
- 🔹 **Diffusion-based Imputation**: Probabilistic diffusion process for robust missing value estimation

### 🎯 Key Features

✅ **Multi-scale Temporal Modeling**: Captures both short-term trends and long-term patterns
✅ **State Space Models**: Efficient handling of long sequences with S4 layers
✅ **Diffusion Framework**: Robust uncertainty quantification
✅ **Industrial Datasets**: Evaluated on Debutanizer and SRU (Sulfur Recovery Unit) datasets
✅ **Comprehensive Baselines**: Comparison with 7+ state-of-the-art methods

---

### 📁 Project Structure

```
sssdtcn/
├── 📂 data/                        # Datasets
│   ├── debutanizer_data.txt       # Debutanizer dataset (Butane composition)
│   └── SRU_data.txt                # SRU dataset (Air flow rate)
│
├── 📂 models/                      # Core model architectures
│   ├── implicit_explicit_diffusion.py  # Main SSSDTCN model
│   ├── diffusion_core.py               # Diffusion process module
│   ├── dilated_causal_conv.py          # Multi-scale dilated convolutions
│   ├── s4_layer.py                     # S4 state space layer
│   ├── mask_embedding.py               # Mask embedding module
│   └── transformer_model.py            # Transformer baseline
│
├── 📂 configs/                     # Configuration files
│   ├── config_debutanizer.py      # Debutanizer dataset config
│   └── config_sru.py               # SRU dataset config
│
├── 📂 scripts/                     # Training & evaluation scripts
│   ├── train_diffusion.py         # Train main SSSDTCN model
│   ├── train_transformer.py       # Train Transformer baseline
│   ├── train_universal.py         # Universal training script
│   ├── evaluate_fusion.py         # Evaluate SSSDTCN model
│   ├── evaluate_transformer.py    # Evaluate Transformer
│   └── data_loader.py             # Data loading utilities
│
├── 📂 baselines/                   # Baseline comparison experiments
│   ├── train_all.py               # 🌟 Train all baseline methods
│   ├── evaluate_all.py            # 🌟 Evaluate all methods → Excel results
│   ├── plot_comparison.py         # 🌟 Generate comparison figures
│   ├── train_simple_methods.py    # Median, Last observation
│   ├── train_pypots_methods.py    # SAITS, BRITS, Transformer
│   ├── train_mrnn_custom.py       # M-RNN baseline
│   └── train_gpvae_custom.py      # GP-VAE baseline
│
├── 📂 ablation/                    # Ablation study experiments
│   ├── train_single_scale.py      # Single-scale convolution ablation
│   ├── train_explicit_only.py     # Explicit-only (S4 only) ablation
│   ├── evaluate_ablation.py       # Evaluate ablation models
│   ├── run_ablation.sh            # 🌟 Run all ablation experiments
│   └── models/                     # Ablation model variants
│
├── 📂 visualization/               # Paper figure generation
│   ├── draw_figures.py            # Model architecture diagrams
│   ├── draw_imputation_fig.py     # Imputation result visualization
│   ├── draw_debutanizer_periodicity.py  # Periodicity analysis
│   ├── analyze_periodicity.py     # Periodicity analysis script
│   ├── compute_cost_analysis.py   # Computational cost analysis
│   └── create_notation_table.py   # Notation table generation
│
├── 📂 results/                     # Experimental results (generated)
│   ├── figures/                    # Output figures (PNG, PDF)
│   ├── tables/                     # Result tables (Excel)
│   └── checkpoints/                # Trained model checkpoints
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
└── 📄 __init__.py                  # Package initialization
```

---

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone repository
git clone https://github.com/Yongying-Zhu/sssdtcn.git
cd sssdtcn

# Install dependencies
pip install -r requirements.txt
```

#### 2. Training Main Model (SSSDTCN)

```bash
# Train on Debutanizer dataset
python scripts/train_diffusion.py --config configs/config_debutanizer.py

# Train on SRU dataset
python scripts/train_diffusion.py --config configs/config_sru.py
```

#### 3. Training Baseline Methods

```bash
# Train all baseline methods (Median, SAITS, BRITS, M-RNN, GP-VAE, Transformer)
cd baselines
python train_all.py --dataset debutanizer --epochs 300
python train_all.py --dataset sru --epochs 300
```

#### 4. Evaluation (20%-80% Missing Rates)

```bash
# Evaluate SSSDTCN model
python scripts/evaluate_fusion.py --dataset debutanizer

# Evaluate all baseline methods and generate Excel results
cd baselines
python evaluate_all.py --dataset debutanizer --missing_rates 20,30,40,50,60,70,80
```

#### 5. Ablation Study

```bash
# Run ablation experiments
cd ablation
bash run_ablation.sh

# Or run in parallel
bash run_ablation_parallel.sh
```

#### 6. Generate Figures

```bash
# Generate comparison figures (like in paper)
cd baselines
python plot_comparison.py --dataset debutanizer

# Generate architecture diagrams
cd visualization
python draw_figures.py

# Generate periodicity analysis
python draw_debutanizer_periodicity.py
```

---

### 📊 Datasets

| Dataset | Description | Features | Sampling Rate | Time Steps |
|---------|-------------|----------|---------------|------------|
| **Debutanizer** | Butane product composition control | 7 variables | 1 min | 2394 |
| **SRU** | Sulfur Recovery Unit air flow control | 6 variables | 1 min | 10081 |

Both datasets are from real industrial process control systems.

---

### 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{sssdtcn2024,
  title={An Implicit-Explicit Diffusion Model for Industrial Data Imputation},
  author={Yongying Zhu},
  journal={arXiv preprint},
  year={2024}
}
```

---

### 📧 Contact

**Author**: Yongying Zhu
**GitHub**: [https://github.com/Yongying-Zhu](https://github.com/Yongying-Zhu)

---

### 📜 License

This project is licensed under the MIT License.

---

<a name="中文"></a>
## 📖 中文文档

### 概述

本仓库实现了用于工业过程控制系统中**时间序列插补**的**隐式-显式扩散模型**。模型架构称为**SSSDTCN**，融合了：

- 🔹 **结构化状态空间模型（S4）**：高效建模长期依赖关系和时序动态
- 🔹 **扩张时序卷积**：多尺度因果卷积捕捉不同时间尺度的模式
- 🔹 **隐式-显式融合**：结合隐式特征提取（扩张卷积）与显式建模（状态空间模型）
- 🔹 **基于扩散的插补**：概率扩散过程用于鲁棒的缺失值估计

### 🎯 核心特性

✅ **多尺度时序建模**：同时捕捉短期趋势和长期模式
✅ **状态空间模型**：使用S4层高效处理长序列
✅ **扩散框架**：鲁棒的不确定性量化
✅ **工业数据集**：在Debutanizer和SRU（硫回收装置）数据集上评估
✅ **全面的基线对比**：与7+种最先进方法对比

---

### 📁 项目结构

详见上方英文版项目结构说明。

---

### 🚀 快速开始

#### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Yongying-Zhu/sssdtcn.git
cd sssdtcn

# 安装依赖
pip install -r requirements.txt
```

#### 2. 训练主模型 (SSSDTCN)

```bash
# 在Debutanizer数据集上训练
python scripts/train_diffusion.py --config configs/config_debutanizer.py

# 在SRU数据集上训练
python scripts/train_diffusion.py --config configs/config_sru.py
```

#### 3. 训练基线方法

```bash
# 训练所有基线方法（Median, SAITS, BRITS, M-RNN, GP-VAE, Transformer）
cd baselines
python train_all.py --dataset debutanizer --epochs 300
python train_all.py --dataset sru --epochs 300
```

#### 4. 评估（20%-80%缺失率）

```bash
# 评估SSSDTCN模型
python scripts/evaluate_fusion.py --dataset debutanizer

# 评估所有基线方法并生成Excel结果
cd baselines
python evaluate_all.py --dataset debutanizer --missing_rates 20,30,40,50,60,70,80
```

#### 5. 消融实验

```bash
# 运行消融实验
cd ablation
bash run_ablation.sh

# 或并行运行
bash run_ablation_parallel.sh
```

#### 6. 生成图表

```bash
# 生成对比图（论文中的图）
cd baselines
python plot_comparison.py --dataset debutanizer

# 生成架构图
cd visualization
python draw_figures.py

# 生成周期性分析图
python draw_debutanizer_periodicity.py
```

---

### 📊 数据集说明

| 数据集 | 描述 | 特征数 | 采样率 | 时间步 |
|---------|-------------|----------|---------------|------------|
| **Debutanizer** | 丁烷产品组成控制 | 7个变量 | 1分钟 | 2394 |
| **SRU** | 硫回收装置空气流速控制 | 6个变量 | 1分钟 | 10081 |

两个数据集均来自真实的工业过程控制系统。

---

### 📝 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{sssdtcn2024,
  title={An Implicit-Explicit Diffusion Model for Industrial Data Imputation},
  author={朱永英},
  journal={arXiv预印本},
  year={2024}
}
```

---

### 📧 联系方式

**作者**：朱永英
**GitHub**：[https://github.com/Yongying-Zhu](https://github.com/Yongying-Zhu)

---

### 📜 许可证

本项目采用MIT许可证。

---

**Happy Coding! 祝您使用愉快！** 🎉
