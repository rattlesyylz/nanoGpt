"""
Sanity check script for Part 1.5
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


def run_memorization_test():
    """
    Test 1: Model should memorize "I love machine learning" perfectly
    """
    print("=" * 60)
    print("SANITY CHECK 1: Memorization Test")
    print("=" * 60)
    
    # Configuration for memorization test
    config = {
        # Data
        'batch_size': 1,  # Small batch for memorization
        'block_size': 32,
        
        # Model - single layer as specified
        'n_layer': 1,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': True,
        'vocab_size': None,  # Will be set based on data
        
        # Training - more aggressive for memorization
        'learning_rate': 1e-3,
        'max_iters': 2000,
        'weight_decay': 0.0,  # No regularization for memorization
        'beta1': 0.9,
        'beta2': 0.95,
        
        # Evaluation
        'eval_interval': 200,
        'log_interval': 100,
        'eval_iters': 10,
        
        # System
        'seed': 1337,
        'out_dir': 'out_sanity_check',
    }
    
    # The target string to memorize
    text = "I love machine learning"
    print(f"Target text: '{text}'")
    
    # Prepare data
    data_info = prepare_data(text)
    config['vocab_size'] = data_info['vocab_size']
    
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Characters: {data_info['chars']}")
    
    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Save tokenizer info
    with open(os.path.join(config['out_dir'], 'tokenizer.pkl'), 'wb') as f:
        pickle.dump({
            'char_to_idx': data_info['char_to_idx'],
            'idx_to_char': data_info['idx_to_char'],
            'chars': data_info['chars']
        }, f)
    
    # Convert to numpy array
    train_data = np.array(data_info['data'])
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train the model
    print("\nStarting memorization training...")
    history = trainer.train(train_data)
    
    # Check if loss went to near zero
    final_loss = history['train_losses'][-1]
    print(f"\nFinal training loss: {final_loss:.6f}")
    
    if final_loss < 0.01:
        print("✅ SUCCESS: Loss went close to zero!")
    else:
        print("❌ WARNING: Loss did not converge to zero. May need more training or different hyperparameters.")
    
    # Test generation
    print("\n" + "-" * 40)
    print("TESTING GENERATION:")
    print("-" * 40)
    
    model_path = os.path.join(config['out_dir'], 'final_model.pt')
    generator = TextGenerator(model_path)
    
    # Test unconditional generation
    print("\nUnconditional generation (should produce the memorized text):")
    for i in range(3):  # Try multiple times
        result = generator.sample_unconditional(max_tokens=len(text), temperature=0.1)
        if result.strip() == text:
            print(f"✅ Attempt {i+1}: Perfect match!")
        else:
            print(f"❌ Attempt {i+1}: '{result.strip()}' (expected: '{text}')")
    
    return config['out_dir'], history


def run_masked_loss_test():
    """
    Test 2: Train with loss masked on first 3 tokens
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 2: Masked Loss Test")
    print("=" * 60)
    
    # Configuration for masked loss test
    config = {
        # Data
        'batch_size': 1,
        'block_size': 32,
        'mask_first_tokens': 3,  # Mask first 3 tokens
        
        # Model
        'n_layer': 1,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': True,
        'vocab_size': None,
        
        # Training
        'learning_rate': 1e-3,
        'max_iters': 2000,
        'weight_decay': 0.0,
        'beta1': 0.9,
        'beta2': 0.95,
        
        # Evaluation
        'eval_interval': 200,
        'log_interval': 100,
        'eval_iters': 10,
        
        # System
        'seed': 42,  # Different seed
        'out_dir': 'out_masked_loss',
    }
    
    text = "I love machine learning"
    print(f"Target text: '{text}'")
    print(f"Masking first {config['mask_first_tokens']} tokens from loss computation")
    
    # Prepare data
    data_info = prepare_data(text)
    config['vocab_size'] = data_info['vocab_size']
    
    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Save tokenizer info
    with open(os.path.join(config['out_dir'], 'tokenizer.pkl'), 'wb') as f:
        pickle.dump({
            'char_to_idx': data_info['char_to_idx'],
            'idx_to_char': data_info['idx_to_char'],
            'chars': data_info['chars']
        }, f)
    
    # Convert to numpy array
    train_data = np.array(data_info['data'])
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Create loss mask
    loss_mask = torch.ones(config['batch_size'], config['block_size'])
    loss_mask[:, :config['mask_first_tokens']] = 0
    loss_mask = loss_mask.to(trainer.device)
    
    print(f"\nLoss mask shape: {loss_mask.shape}")
    print(f"Loss mask (first few positions): {loss_mask[0, :10].tolist()}")
    
    # Train with masked loss
    print("\nStarting masked loss training...")
    history = trainer.train(train_data, loss_mask=loss_mask)
    
    print(f"\nFinal training loss: {history['train_losses'][-1]:.6f}")
    
    # Test generation
    print("\n" + "-" * 40)
    print("TESTING GENERATION WITH MASKED LOSS:")
    print("-" * 40)
    
    model_path = os.path.join(config['out_dir'], 'final_model.pt')
    generator = TextGenerator(model_path)
    
    # The model should struggle with the first few characters since loss was masked
    print("\nUnconditional generation (may struggle with first few chars due to masking):")
    for i in range(3):
        result = generator.sample_unconditional(max_tokens=len(text), temperature=0.1)
        print(f"Attempt {i+1}: '{result.strip()}'")
    
    # Test with different starting prompts
    test_prompts = ["I", "I ", "I l"]
    print("\nTesting with different prompts:")
    for prompt in test_prompts:
        result = generator.complete_text(prompt, max_new_tokens=15, temperature=0.1)
        print(f"Prompt '{prompt}' -> Result: '{result}'")
    
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
        plt.plot(iterations, history['val_losses'], 'r-', label='Test Loss', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see loss going to zero
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()


def compare_results(dir1, history1, dir2, history2):
    """
    Compare results from both experiments
    """
    print("\n" + "=" * 60)
    print("COMPARISON OF RESULTS")
    print("=" * 60)
    
    print(f"\nExperiment 1 (Normal): Final loss = {history1['train_losses'][-1]:.6f}")
    print(f"Experiment 2 (Masked): Final loss = {history2['train_losses'][-1]:.6f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training curves comparison
    plt.subplot(2, 1, 1)
    
    iterations1 = list(range(0, len(history1['train_losses']) * history1['config']['eval_interval'], 
                            history1['config']['eval_interval']))
    iterations2 = list(range(0, len(history2['train_losses']) * history2['config']['eval_interval'], 
                            history2['config']['eval_interval']))
    
    plt.plot(iterations1, history1['train_losses'], 'b-', label='Normal Training', linewidth=2)
    plt.plot(iterations2, history2['train_losses'], 'r-', label='Masked Loss Training', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Final losses bar chart
    plt.subplot(2, 1, 2)
    experiments = ['Normal', 'Masked Loss']
    final_losses = [history1['train_losses'][-1], history2['train_losses'][-1]]
    
    plt.bar(experiments, final_losses, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Final Loss')
    plt.title('Final Training Loss Comparison')
    plt.yscale('log')
    
    for i, v in enumerate(final_losses):
        plt.text(i, v, f'{v:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sanity_check_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Run all sanity checks
    """
    print("Starting Sanity Checks for Part 1.5")
    print("This will test if the training infrastructure works correctly")
    
    # Run memorization test
    dir1, history1 = run_memorization_test()
    
    # Plot training curve for first experiment
    plot_training_curves(dir1, history1, "Memorization Test - Training Loss")
    
    # Run masked loss test
    dir2, history2 = run_masked_loss_test()
    
    # Plot training curve for second experiment
    plot_training_curves(dir2, history2, "Masked Loss Test - Training Loss")
    
    # Compare results
    compare_results(dir1, history1, dir2, history2)
    
    print("\n" + "=" * 60)
    print("SANITY CHECKS COMPLETED!")
    print("=" * 60)
    
    print("\nSUMMARY:")
    print(f"1. Normal memorization final loss: {history1['train_losses'][-1]:.6f}")
    print(f"2. Masked loss final loss: {history2['train_losses'][-1]:.6f}")
    
    print("\nFILES GENERATED:")
    print(f"- {dir1}/: Normal memorization experiment")
    print(f"- {dir2}/: Masked loss experiment") 
    print("- sanity_check_comparison.png: Comparison plot")
    
    if history1['train_losses'][-1] < 0.01:
        print("\n SUCCESS")
    else:
        print("\WARNING: Model may need more training to fully memorize.")
    
    print("\nTo test inference manually, run:")
    print(f"python inference.py test {dir1}/final_model.pt")
    print(f"python inference.py test {dir2}/final_model.pt")


if __name__ == "__main__":
    main()