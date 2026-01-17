# SSSDTCN - State Space Structured Deep Temporal Convolution Network

Time series prediction project for hydraulic system monitoring using deep learning models.

## 📁 Project Structure

```
sssdtcn/
├── train_universal.py          # Universal training script
├── train.py                     # Main training script
├── evaluate.py                  # Model evaluation
├── evaluate_universal.py        # Universal evaluation
├── evaluate_enhanced.py         # Enhanced evaluation
├── run_1hz.py                   # Run script for 1Hz frequency
├── run_10hz.py                  # Run script for 10Hz frequency
├── run_100hz.py                 # Run script for 100Hz frequency
├── run_training.sh              # Training bash script
├── run_evaluation.sh            # Evaluation bash script
├── models/                      # Model architectures
├── baselines/                   # Baseline models
├── utils/                       # Utility functions
├── config_1hz.py               # Configuration for 1Hz
├── config_10hz.py              # Configuration for 10Hz
└── AnYujin/                    # Custom modules
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training
```bash
# For different frequencies
python run_1hz.py
python run_10hz.py
python run_100hz.py

# Or use bash script
bash run_training.sh
```

### Evaluation
```bash
python evaluate.py
# Or
bash run_evaluation.sh
```

## 📊 Data

This project works with hydraulic system sensor data at different sampling frequencies:
- 1Hz clean sensors
- 10Hz clean sensors
- 100Hz clean sensors

## 🔧 Configuration

Edit `config_*.py` files to adjust:
- Model hyperparameters
- Training settings
- Data paths
- Evaluation metrics

## 📝 Models

The project includes implementations of:
- S4 Layer (State Space Sequence Models)
- Transformer-based models
- Implicit-Explicit Diffusion models
- Mask Embedding techniques

## 📄 License

[Add your license here]

## 👤 Author

Yongying Zhu

## 🙏 Acknowledgments

[Add acknowledgments if any]
