"""
Part 2.2 Warmup Script
Tests if the model can memorize and reproduce a simple string
"""
import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from train import Trainer, prepare_data
from inference import TextGenerator


def run_operation_test(operation, prime, seed):
    """
    Run a test for an operation and a prime
    """
    print("=" * 60)
    print("Testing operation:", operation, "with prime:", prime)
    print("=" * 60)
    
    # Configuration for operations test, aligned with paper specifications
    config = {
        # Data
        'batch_size': 16,
        'block_size': 32,
        'mask_until_equals': True,
        'char_to_idx': None,  # Will be set based on data
        
        # Model
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'n_ffwd': 512,
        'dropout': 0.0,
        'bias': True,
        'vocab_size': None,  # Will be set based on data
        
        # Training
        'learning_rate': 1e-3,
        'max_iters': 100000,
        'weight_decay': 1.0,
        'beta1': 0.9,
        'beta2': 0.98,
        
        # Evaluation
        'eval_interval': 200,
        'log_interval': 100,
        'eval_iters': 10,
        
        # System
        'seed': seed,
        'out_dir': f'out_operation_test/double_layer/{operation}_{prime}_seed{seed}',
    }
    
    # Prepare data
    data_train =f"data/train_{operation}_{prime}.txt"
    data_test = f"data/test_{operation}_{prime}.txt"
    with open(data_train, 'r') as f:
        train_info = prepare_data(f.read())
    with open(data_test, 'r') as f:
        test_info = prepare_data(f.read())
    config['vocab_size'] = train_info['vocab_size']
    config['char_to_idx'] = train_info['char_to_idx']
    
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Characters: {train_info['chars']}")
    
    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Save tokenizer info
    with open(os.path.join(config['out_dir'], 'tokenizer.pkl'), 'wb') as f:
        pickle.dump({
            'char_to_idx': train_info['char_to_idx'],
            'idx_to_char': train_info['idx_to_char'],
            'chars': train_info['chars']
        }, f)
    
    # Convert to numpy array
    train_data = np.array(train_info['data'])
    test_data = np.array(test_info['data'])
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train the model
    print("\nStarting training on prime:", prime + " with operation:", operation)
    history = trainer.train(train_data, test_data)

    # Evaluate on test set
    test_loss = trainer.estimate_loss(train_data, test_data)
    print(f"Raw test loss dict: {test_loss}")
    
    return config['out_dir'], history



def plot_training_curves(output_dir, history, title="Training Loss"):
    """
    Plot training curves
    """
    plt.figure(figsize=(10, 6))
    
    iterations = list(range(0, len(history['train_losses']) * history['config']['eval_interval'], 
                           history['config']['eval_interval']))
    
    plt.plot(iterations, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    
    if history['val_losses']:
        plt.plot(iterations, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see loss going to zero
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Run addition and subtraction operations for primes 97 and 113 using a single-layer model
    """
    print("Starting operation tests...")
    
    seeds = [1337, 1338, 1339]
    operations = ['addition', 'subtraction']
    primes = ['97', '113']
    
    results = []

    # Run experiments for all combinations
    for operation in operations:
        for prime in primes:
            for seed in seeds:
                print(f"\nRunning {operation.capitalize()} mod {prime} with seed {seed}")
                dir_path, history = run_operation_test(operation=operation, prime=prime, seed=seed)
                results.append((operation, prime, seed, dir_path, history))
    
    # Plot training curves
    for operation, prime, seed, dir_path, history in results:
        title = f"{operation.capitalize()} {prime} Training Curve (seed={seed})"
        plot_training_curves(dir_path, history, title)
    
    print("\n" + "=" * 60)
    print("Operation check completed!")
    print("=" * 60)
    
    print("\nSUMMARY:")
    for operation in operations:
        for prime in primes:
            print(f"{operation.capitalize()} {prime} Final Losses:")
            for op, p, seed, dir_path, history in results:
                if op == operation and p == prime:
                    print(f"- Seed {seed}: Final train loss = {history['train_losses'][-1]:.6f}, output dir = {dir_path}")
    
    print("\nOUTPUT DIRECTORIES:")
    for op, p, seed, dir_path, _ in results:
        print(f"- {dir_path}/: {op.capitalize()} {p} (single_layer) experiment (seed={seed})")
    print("- training_curve.png: Training curves for each operation")


if __name__ == "__main__":
    main()