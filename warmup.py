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
        'batch_size': 1,  # Small batch for memorization
        'block_size': 32,
        'mask_until_equals': True,
        'char_to_idx': None,  # Will be set based on data
        
        # Model - single layer
        'n_layer': 1,
        'n_head': 4,
        'n_embd': 128,
        'n_ffwd': 512,
        'dropout': 0.0,
        'bias': True,
        'vocab_size': None,  # Will be set based on data
        
        # Training
        'learning_rate': 1e-3,
        'max_iters': 1000,
        'weight_decay': 1.0,
        'beta1': 0.9,
        'beta2': 0.98,
        
        # Evaluation
        'eval_interval': 200,
        'log_interval': 100,
        'eval_iters': 10,
        
        # System
        'seed': seed,
        'out_dir': f'out_operation_test/{operation}_{prime}_seed{seed}',
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
    history = trainer.train(train_data)

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
    Run all operations for 2.1
    """
    print("Starting operation tests...")
    
    # Run addition 97
    seeds = [1337, 1338, 1339]
    results = []

    for seed in seeds:
        print(f"\nRunning Addition mod 97 with seed {seed}")
        dir_path, history = run_operation_test(operation='addition', prime='97', seed=seed)
        results.append((seed, dir_path, history))
    # Run addition 113
    # dir2, history2 = run_operation_test(operation='addition', prime='113')

    # Run subtraction 97
    # dir3, history3 = run_operation_test(operation='subtraction', prime='97')
    # Run subtraction 113
    # dir4, history4 = run_operation_test(operation='subtraction', prime='113')

    # Run division 97
    # dir5, history5 = run_operation_test(operation='division', prime='97')
    # Run division 113
    # dir6, history6 = run_operation_test(operation='division', prime='113')
    
    # Plot training curve for second experiment
    for seed, dir_path, history in results:
        title = f"Addition 97 Training Curve (seed={seed})"
        plot_training_curves(dir_path, history, title)
    # plot_training_curves(dir2, history2, "Addition 113 Training Curve")
    # plot_training_curves(dir3, history3, "Subtraction 97 Training Curve")
    # plot_training_curves(dir4, history4, "Subtraction 113 Training Curve")
    # plot_training_curves(dir5, history5, "Division 97 Training Curve")  
    # plot_training_curves(dir6, history6, "Division 113 Training Curve")
    
    print("\n" + "=" * 60)
    print("Operation check completed!")
    print("=" * 60)
    
    print("\nSUMMARY:")
    print("Addition 97 Final Losses:")
    for seed, dir_path, history in results:
        print(f"- Seed {seed}: Final train loss = {history['train_losses'][-1]:.6f}, output dir = {dir_path}")
    # print(f"2. Addition 113 final loss: {history2['train_losses'][-1]:.6f}")
    # print(f"3. Subtraction 97 final loss: {history3['train_losses'][-1]:.6f}")
    # print(f"4. Subtraction 113 final loss: {history4['train_losses'][-1]:.6f}")
    # print(f"5. Division 97 final loss: {history5['train_losses'][-1]:.6f}") 
    # print(f"6. Division 113 final loss: {history6['train_losses'][-1]:.6f}")
    
    for seed, dir_path, _ in results:
        print(f"- {dir_path}/: Addition 97 experiment (seed={seed})")
    # print(f"- {dir2}/: Addition 113 experiment")
    # print(f"- {dir3}/: Subtraction 97 experiment")
    # print(f"- {dir4}/: Subtraction 113 experiment")
    # print(f"- {dir5}/: Division 97 experiment")
    # print(f"- {dir6}/: Division 113 experiment")
    print("- training_curve.png: Training curves for each operation")
    


if __name__ == "__main__":
    main()