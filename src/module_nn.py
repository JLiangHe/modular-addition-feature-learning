#!/usr/bin/env python3
"""
Module NN: Neural Network Training Wrapper
==========================================

A flexible wrapper for training neural networks on modular arithmetic tasks.
Supports command-line parameter overrides for easy batch experimentation.

Usage Examples:
    # Use default config
    python module_nn.py
    
    # Override specific parameters
    python module_nn.py --p 17 --lr 0.01 --num_epochs 10000
    
    # Run batch experiments on init_type, optimizer, and act_type (16 total combinations)
    python module_nn.py --experiments
    
    # Run batch experiments with custom parameters
    python module_nn.py --experiments --p 17 --num_epochs 3000
    
    # Dry run to see configuration
    python module_nn.py --dry_run --p 23 --lr 0.001
    
    # Multiple parameters for single experiment
    python module_nn.py --p 23 --lr 0.001 --d_mlp 256 --act_type ReLU --seed 42

Bash Script Example:
    # Run experiments for different primes
    for p in 17 23 31; do
        python module_nn.py --experiments --p $p --num_epochs 3000
    done
"""

import argparse
import sys
from collections import deque

from utils import *
from nnTrainer import Trainer


def parse_arguments():
    """Parse command line arguments with support for config overrides"""
    parser = argparse.ArgumentParser(
        description='Neural Network Training for Modular Arithmetic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data parameters
    parser.add_argument('--p', type=int, help='Prime number for modular arithmetic')
    parser.add_argument('--d_vocab', type=int, help='Vocabulary size (defaults to p)')
    parser.add_argument('--fn_name', type=str, choices=['add', 'subtract', 'x2xyy2'], help='Function to learn')
    parser.add_argument('--frac_train', type=float, help='Fraction of data for training')
    parser.add_argument('--batch_style', type=str, help='Batch processing style')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, help='Model embedding dimensionality')
    parser.add_argument('--d_mlp', type=int, help='MLP layer dimensionality')
    parser.add_argument('--act_type', type=str, choices=['ReLU', 'GeLU', 'Quad', 'Id'], help='Activation function')
    parser.add_argument('--embed_type', type=str, choices=['one_hot', 'learned'], help='Embedding type')
    parser.add_argument('--init_type', type=str, choices=['random', 'single-freq'], help='Weight initialization')
    parser.add_argument('--init_scale', type=float, help='Scale factor for weight initialization')
    parser.add_argument('--freq_num', type=int, help='Number of frequencies for single-freq init')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--optimizer', type=str, choices=['AdamW', 'SGD'], help='Optimizer')
    parser.add_argument('--stopping_thresh', type=float, help='Early stopping threshold')
    parser.add_argument('--save_models', type=bool, help='Whether to save models')
    parser.add_argument('--save_every', type=int, help='Save frequency (epochs)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Special flags
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--dry_run', action='store_true', help='Print config and exit without training')
    
    return parser.parse_args()


def override_config(config_dict, args):
    """Override config values with command line arguments"""
    # Flatten the nested config for easier access
    flat_config = {}
    
    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten_dict(v, parent_key)
            else:
                flat_config[k] = v
    
    flatten_dict(config_dict)
    
    # Override with command line arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name in flat_config:
            flat_config[arg_name] = arg_value
            print(f"Override: {arg_name} = {arg_value}")
    
    # Reconstruct nested structure
    result = {'data': {}, 'model': {}, 'training': {}}
    
    # Data parameters
    data_params = ['p', 'd_vocab', 'fn_name', 'frac_train', 'batch_style']
    for param in data_params:
        if param in flat_config:
            result['data'][param] = flat_config[param]
    
    # Model parameters  
    model_params = ['d_model', 'd_mlp', 'act_type', 'embed_type', 'init_type', 'init_scale', 'freq_num']
    for param in model_params:
        if param in flat_config:
            result['model'][param] = flat_config[param]
    
    # Training parameters
    training_params = ['num_epochs', 'lr', 'weight_decay', 'optimizer', 'stopping_thresh', 
                      'save_models', 'save_every', 'seed']
    for param in training_params:
        if param in flat_config:
            result['training'][param] = flat_config[param]
    
    return result


def run_experiment(config_dict):
    """Run the training experiment with given configuration"""
    print("="*80)
    print("MODULAR ARITHMETIC NEURAL NETWORK TRAINING")
    print("="*80)
    
    # Create config object
    pipeline_config = Config(config_dict)
    print(f"Configuration loaded successfully")
    print(f"Device: {pipeline_config.device}")
    print(f"Prime p: {pipeline_config.p}")
    print(f"Vocabulary size: {pipeline_config.d_vocab}")
    print(f"Model dimensions: d_model={pipeline_config.d_model}, d_mlp={pipeline_config.d_mlp}")
    print(f"Function: {pipeline_config.fn_name}")
    print(f"Activation: {pipeline_config.act_type}")
    print(f"Seed: {pipeline_config.seed}")
    print(f"Init scale: {pipeline_config.init_scale}")
    print(f"Learning rate: {pipeline_config.lr}")
    print("-" * 80)
    
    # Initialize trainer
    world = Trainer(config=pipeline_config)
    print(f'Run name: {world.run_name}')
    world.initial_save_if_appropriate()
    
    # Training variables
    recent_test_loss = deque(maxlen=2)
    save_point = 0
    
    print(f"Starting training for {pipeline_config.num_epochs} epochs...")
    print("-" * 80)
    
    # Training loop
    for epoch in range(pipeline_config.num_epochs):
        # Perform a training step and get train/test losses
        train_loss, test_loss = world.do_a_training_step(epoch)
        
        # Stop training if test loss falls below the threshold
        if test_loss.item() < pipeline_config.stopping_thresh:
            print(f"Early stopping at epoch {epoch}: test loss {test_loss.item():.6f} < {pipeline_config.stopping_thresh}")
            break
        
        # Save model state if it's time to do so
        if pipeline_config.is_it_time_to_save(epoch=epoch):
            world.save_epoch(epoch=epoch, local_save=True)
    
    # Save final model state after training is complete
    print("-" * 80)
    print("Training completed! Saving final model...")
    world.post_training_save(save_optimizer_and_scheduler=True)
    print(f"Final train loss: {world.train_losses[-1]:.6f}")
    print(f"Final test loss: {world.test_losses[-1]:.6f}")
    print(f"Final train accuracy: {world.train_accs[-1]:.4f}")
    print(f"Final test accuracy: {world.test_accs[-1]:.4f}")
    print("="*80)
    
    return world


def run_batch_experiments(base_config):
    """Run batch experiments on init_type, optimizer, and act_type"""
    print("="*80)
    print("BATCH EXPERIMENTS: init_type, optimizer, act_type")
    print("="*80)
    
    results = []
    experiment_count = 0
    
    # Test parameters
    init_types = ['random', 'single-freq']
    optimizers = ['AdamW', 'SGD']
    act_types = ['ReLU', 'GeLU', 'Quad', 'Id']
    
    total_experiments = len(init_types) * len(optimizers) * len(act_types)
    print(f"Running {total_experiments} experiments...")
    print("-" * 80)
    
    for init_type in init_types:
        for optimizer in optimizers:
            for act_type in act_types:
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                print(f"Configuration: init_type={init_type}, optimizer={optimizer}, act_type={act_type}")
                print("-" * 50)
                
                # Create experiment config
                exp_config = base_config.copy()
                exp_config['model']['init_type'] = init_type
                exp_config['training']['optimizer'] = optimizer
                exp_config['model']['act_type'] = act_type
                
                # Use different seeds for each experiment
                exp_config['training']['seed'] = 1024 + experiment_count
                
                # Reduce epochs for faster batch testing
                exp_config['training']['num_epochs'] = min(exp_config['training']['num_epochs'], 5000)
                
                try:
                    # Run the experiment
                    trainer = run_experiment(exp_config)
                    
                    # Collect results
                    result = {
                        'experiment': experiment_count,
                        'init_type': init_type,
                        'optimizer': optimizer,
                        'act_type': act_type,
                        'seed': exp_config['training']['seed'],
                        'final_train_loss': trainer.train_losses[-1],
                        'final_test_loss': trainer.test_losses[-1],
                        'final_train_acc': trainer.train_accs[-1],
                        'final_test_acc': trainer.test_accs[-1],
                        'run_name': trainer.run_name
                    }
                    results.append(result)
                    
                    print(f"✓ Experiment {experiment_count} completed successfully")
                    print(f"  Final test accuracy: {result['final_test_acc']:.4f}")
                    
                except Exception as e:
                    print(f"✗ Experiment {experiment_count} failed: {str(e)}")
                    results.append({
                        'experiment': experiment_count,
                        'init_type': init_type,
                        'optimizer': optimizer,
                        'act_type': act_type,
                        'seed': exp_config['training']['seed'],
                        'error': str(e)
                    })
                
                print("-" * 50)
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH EXPERIMENTS SUMMARY")
    print("="*80)
    
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        print("\nTop 5 Results by Test Accuracy:")
        print("-" * 50)
        sorted_results = sorted(successful_results, key=lambda x: x['final_test_acc'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. Test Acc: {result['final_test_acc']:.4f} | "
                  f"init_type={result['init_type']}, optimizer={result['optimizer']}, "
                  f"act_type={result['act_type']}")
        
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'Exp':<3} {'Init':<11} {'Opt':<5} {'Act':<4} {'Train Acc':<9} {'Test Acc':<8} {'Train Loss':<10} {'Test Loss':<9}")
        print("-" * 80)
        for result in sorted_results:
            print(f"{result['experiment']:<3} "
                  f"{result['init_type']:<11} "
                  f"{result['optimizer']:<5} "
                  f"{result['act_type']:<4} "
                  f"{result['final_train_acc']:<9.4f} "
                  f"{result['final_test_acc']:<8.4f} "
                  f"{result['final_train_loss']:<10.6f} "
                  f"{result['final_test_loss']:<9.6f}")
    
    if failed_results:
        print(f"\nFailed Experiments:")
        for result in failed_results:
            print(f"Exp {result['experiment']}: {result['init_type']}, {result['optimizer']}, {result['act_type']} - {result['error']}")
    
    print("="*80)
    return results


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load base configuration
    if args.config:
        # Load custom config file
        import yaml
        with open(args.config, 'r') as f:
            configs = yaml.safe_load(f)
    else:
        # Load default config
        configs = read_config()
    
    # Override with command line arguments
    final_config = override_config(configs, args)
    
    if args.dry_run:
        print("DRY RUN - Configuration that would be used:")
        print("-" * 50)
        import yaml
        print(yaml.dump(final_config, default_flow_style=False, indent=2))
        return
    
    # Run single experiment
    trainer = run_experiment(final_config)
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {trainer.save_dir}/{trainer.run_name}")
    return trainer


if __name__ == "__main__":
    main()