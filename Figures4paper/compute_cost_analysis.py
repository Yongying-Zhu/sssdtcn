"""
Computational Cost Analysis for Implicit-Explicit Diffusion Model
Generates TABLE VII style output for Debutanizer and SRU datasets
"""
import sys
sys.path.insert(0, '/home/user/sssdtcn')

import torch
import torch.nn as nn
import time
import numpy as np

# Import model and configs
from implicit_explicit_diffusion import ImplicitExplicitDiffusionModel
from configs.config_debutanizer import ConfigDebutanizer
from configs.config_sru import ConfigSRU


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params):
    """Format parameter count as readable string"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f} G"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f} M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f} K"
    else:
        return str(num_params)


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def measure_training_time(model, config, device, num_iterations=100):
    """Measure training time for specified iterations"""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Prepare dummy data
    batch_size = config.batch_size
    seq_len = config.sequence_length
    input_dim = config.num_features

    # Warmup
    for _ in range(10):
        x = torch.randn(batch_size, seq_len, input_dim).to(device)
        timesteps = torch.randint(0, 200, (batch_size,)).to(device)
        mask = torch.ones(batch_size, seq_len, input_dim).to(device)
        observed = torch.randn(batch_size, seq_len, input_dim).to(device)

        optimizer.zero_grad()
        output = model(x, timesteps, mask, observed)
        loss = output.mean()
        loss.backward()
        optimizer.step()

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    start_time = time.time()
    for _ in range(num_iterations):
        x = torch.randn(batch_size, seq_len, input_dim).to(device)
        timesteps = torch.randint(0, 200, (batch_size,)).to(device)
        mask = torch.ones(batch_size, seq_len, input_dim).to(device)
        observed = torch.randn(batch_size, seq_len, input_dim).to(device)

        optimizer.zero_grad()
        output = model(x, timesteps, mask, observed)
        loss = output.mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    return elapsed


def measure_inference_time(model, config, device, num_samples=100, diffusion_steps=200):
    """Measure inference time for imputation"""
    model = model.to(device)
    model.eval()

    batch_size = config.batch_size
    seq_len = config.sequence_length
    input_dim = config.num_features

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(batch_size, seq_len, input_dim).to(device)
            timesteps = torch.randint(0, 200, (batch_size,)).to(device)
            mask = torch.ones(batch_size, seq_len, input_dim).to(device)
            observed = torch.randn(batch_size, seq_len, input_dim).to(device)
            _ = model(x, timesteps, mask, observed)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure full diffusion inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(batch_size, seq_len, input_dim).to(device)
            mask = torch.ones(batch_size, seq_len, input_dim).to(device)
            observed = torch.randn(batch_size, seq_len, input_dim).to(device)

            # Simulate diffusion inference (multiple denoising steps)
            for t in range(diffusion_steps - 1, -1, -10):  # DDIM-style skip
                timesteps = torch.full((batch_size,), t, device=device)
                x = model(x, timesteps, mask, observed)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    return elapsed


def main():
    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}

    # ========== Debutanizer ==========
    print("=" * 50)
    print("Analyzing Debutanizer Dataset...")
    print("=" * 50)

    config_deb = ConfigDebutanizer()
    model_deb = ImplicitExplicitDiffusionModel(
        input_dim=config_deb.num_features,
        hidden_dim=config_deb.hidden_dim,
        embedding_dim=config_deb.embedding_dim,
        num_residual_layers=config_deb.num_residual_layers,
        dropout=config_deb.dropout
    )

    num_params_deb = count_parameters(model_deb)
    print(f"Number of parameters: {num_params_deb:,} ({format_params(num_params_deb)})")

    # Training time (100 iterations)
    train_time_deb = measure_training_time(model_deb, config_deb, device, num_iterations=100)
    print(f"Training time (100 iter): {format_time(train_time_deb)}")

    # Inference time
    inference_time_deb = measure_inference_time(model_deb, config_deb, device, num_samples=10)
    print(f"Inference time (10 samples): {format_time(inference_time_deb)}")

    results['Debutanizer'] = {
        'params': num_params_deb,
        'params_str': format_params(num_params_deb),
        'train_time': train_time_deb,
        'train_time_str': format_time(train_time_deb),
        'inference_time': inference_time_deb,
        'inference_time_str': format_time(inference_time_deb)
    }

    # Clear GPU memory
    del model_deb
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ========== SRU ==========
    print()
    print("=" * 50)
    print("Analyzing SRU Dataset...")
    print("=" * 50)

    config_sru = ConfigSRU()
    model_sru = ImplicitExplicitDiffusionModel(
        input_dim=config_sru.num_features,
        hidden_dim=config_sru.hidden_dim,
        embedding_dim=config_sru.embedding_dim,
        num_residual_layers=config_sru.num_residual_layers,
        dropout=config_sru.dropout
    )

    num_params_sru = count_parameters(model_sru)
    print(f"Number of parameters: {num_params_sru:,} ({format_params(num_params_sru)})")

    # Training time (100 iterations)
    train_time_sru = measure_training_time(model_sru, config_sru, device, num_iterations=100)
    print(f"Training time (100 iter): {format_time(train_time_sru)}")

    # Inference time
    inference_time_sru = measure_inference_time(model_sru, config_sru, device, num_samples=10)
    print(f"Inference time (10 samples): {format_time(inference_time_sru)}")

    results['SRU'] = {
        'params': num_params_sru,
        'params_str': format_params(num_params_sru),
        'train_time': train_time_sru,
        'train_time_str': format_time(train_time_sru),
        'inference_time': inference_time_sru,
        'inference_time_str': format_time(inference_time_sru)
    }

    # ========== Print TABLE VII Style Output ==========
    print()
    print("=" * 70)
    print("TABLE VII")
    print("THE COMPUTATION COST ON DEBUTANIZER AND SRU")
    print("=" * 70)
    print(f"{'Dataset':<15} {'#Parameter':<15} {'Training time (s/100 iter)':<28} {'Inference time':<15}")
    print("-" * 70)
    for dataset, data in results.items():
        print(f"{dataset:<15} {data['params_str']:<15} {data['train_time_str']:<28} {data['inference_time_str']:<15}")
    print("-" * 70)

    return results


if __name__ == '__main__':
    results = main()
