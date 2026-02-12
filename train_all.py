"""
主训练脚本 - 训练所有方法
"""
import subprocess
import sys
import os

def run_command(cmd, log_file):
    """运行命令并记录日志"""
    print(f"\nRunning: {cmd}")
    print(f"Log file: {log_file}")
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT
        )
        process.wait()
    return process.returncode

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['sru', 'debutanizer', 'all'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--methods', type=str, default='all', 
                        help='Comma-separated list of methods or "all"')
    args = parser.parse_args()
    
    datasets = ['sru', 'debutanizer'] if args.dataset == 'all' else [args.dataset]
    
    base_dir = '/home/zhu/sssdtcn/compare'
    log_dir = f'{base_dir}/logs'
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# Processing dataset: {dataset.upper()}")
        print(f"{'#'*70}")
        
        # 1. 简单方法 (Median, Last)
        if args.methods == 'all' or 'simple' in args.methods.lower():
            cmd = f"python {base_dir}/train_simple_methods.py --dataset {dataset}"
            run_command(cmd, f"{log_dir}/simple_{dataset}.log")
        
        # 2. PyPOTS方法 (SAITS, BRITS, Transformer, 可能包括MRNN和GPVAE)
        if args.methods == 'all' or 'pypots' in args.methods.lower():
            cmd = f"python {base_dir}/train_pypots_methods.py --dataset {dataset} --epochs {args.epochs}"
            run_command(cmd, f"{log_dir}/pypots_{dataset}.log")
        
        # 3. 自定义M-RNN (如果PyPOTS不支持)
        if args.methods == 'all' or 'mrnn' in args.methods.lower():
            cmd = f"python {base_dir}/train_mrnn_custom.py --dataset {dataset} --epochs {args.epochs}"
            run_command(cmd, f"{log_dir}/mrnn_custom_{dataset}.log")
        
        # 4. 自定义GP-VAE (如果PyPOTS不支持)
        if args.methods == 'all' or 'gpvae' in args.methods.lower():
            cmd = f"python {base_dir}/train_gpvae_custom.py --dataset {dataset} --epochs {args.epochs}"
            run_command(cmd, f"{log_dir}/gpvae_custom_{dataset}.log")
    
    print("\n" + "="*70)
    print("All training completed!")
    print("="*70)
    print(f"\nModel files saved in: {base_dir}/models/")
    print(f"Log files saved in: {log_dir}/")

if __name__ == '__main__':
    main()
