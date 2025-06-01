"""
Training script for GPT model
"""
import os
import json
import time
import math
import pickle
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import GPT, GPTConfig


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class Trainer:
    def __init__(self, config):
        self.config = config
        
        # Set random seeds for reproducibility
        torch.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model configuration
        model_config = GPTConfig(
            block_size=config['block_size'],
            vocab_size=config['vocab_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            dropout=config['dropout'],
            bias=config['bias']
        )
        
        # Initialize model
        self.model = GPT(model_config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.model.configure_optimizers(
            weight_decay=config['weight_decay'],
            learning_rate=config['learning_rate'],
            betas=(config['beta1'], config['beta2']),
            device_type='cuda' if self.device.type == 'cuda' else 'cpu'
        )
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        os.makedirs(config['out_dir'], exist_ok=True)
        
        # Save config
        with open(os.path.join(config['out_dir'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_batch(self, data, batch_size, device):
        """Generate a batch of data"""
        # Ensure block_size doesn't exceed data length
        effective_block_size = min(self.config['block_size'], len(data) - 1)
        
        ix = torch.randint(len(data) - effective_block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+effective_block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+effective_block_size]).astype(np.int64)) for i in ix])
        
        if device.type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    def estimate_loss(self, train_data, val_data):
        """Estimate loss on train and validation sets"""
        out = {}
        self.model.eval()
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                X, Y = self.get_batch(data, self.config['batch_size'], self.device)
                with torch.no_grad():
                    logits = self.model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_step(self, X, Y, loss_mask=None):
        """Single training step"""
        logits = self.model(X)
        
        if loss_mask is not None:
            # Apply loss mask (e.g., for masking first few tokens)
            batch_size, seq_len = Y.shape
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')
            loss = loss.view(batch_size, seq_len)
            
            # Resize loss_mask if needed
            if loss_mask.shape[1] != seq_len:
                # Create a new mask of the right size
                effective_mask = loss_mask[:, :seq_len] if loss_mask.shape[1] > seq_len else torch.ones((batch_size, seq_len), device=loss_mask.device)
                if loss_mask.shape[1] < seq_len:
                    effective_mask[:, :loss_mask.shape[1]] = loss_mask
            else:
                effective_mask = loss_mask
                
            loss = loss * effective_mask
            loss = loss.sum() / effective_mask.sum()
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
        
        return loss, logits
    
    def train(self, train_data, val_data=None, loss_mask=None):
        """Main training loop"""
        print(f"Starting training for {self.config['max_iters']} iterations...")
        
        # Training loop
        train_losses = []
        val_losses = []
        
        t0 = time.time()
        raw_model = self.model
        running_mfu = -1.0
        
        for iter_num in range(self.config['max_iters']):
            self.iter_num = iter_num
            
            # Evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.config['eval_interval'] == 0 or iter_num == self.config['max_iters'] - 1:
                if val_data is not None:
                    losses = self.estimate_loss(train_data, val_data)
                    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    train_losses.append(losses['train'].item())
                    val_losses.append(losses['val'].item())
                    
                    # Save checkpoint if validation loss improved
                    if losses['val'] < self.best_val_loss:
                        self.best_val_loss = losses['val']
                        if iter_num > 0:
                            self.save_checkpoint('best_model.pt')
                else:
                    # No validation data, just compute train loss
                    X, Y = self.get_batch(train_data, self.config['batch_size'], self.device)
                    with torch.no_grad():
                        loss, _ = self.train_step(X, Y, loss_mask)
                    print(f"step {iter_num}: train loss {loss:.4f}")
                    train_losses.append(loss.item())
            
            # Sample a batch of data
            X, Y = self.get_batch(train_data, self.config['batch_size'], self.device)
            
            # Forward pass
            loss, logits = self.train_step(X, Y, loss_mask)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.config['log_interval'] == 0:
                lossf = loss.item()
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses if val_data is not None else [],
            'config': self.config
        }
        
        with open(os.path.join(self.config['out_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model.config,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        print(f"Saving checkpoint to {os.path.join(self.config['out_dir'], filename)}")
        torch.save(checkpoint, os.path.join(self.config['out_dir'], filename))


def prepare_data(text, vocab_size=None):
    """Prepare text data for training"""
    # Simple character-level tokenization
    chars = sorted(list(set(text)))
    if vocab_size is not None:
        chars = chars[:vocab_size]
    
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the text
    data = [char_to_idx[ch] for ch in text if ch in char_to_idx]
    
    return {
        'data': data,
        'vocab_size': vocab_size,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'chars': chars
    }


def main():
    # Default configuration
    config = {
        # Data
        'batch_size': 4,
        'block_size': 32,
        
        # Model
        'n_layer': 1,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': True,
        'vocab_size': None,  # Will be set based on data
        
        # Training
        'learning_rate': 3e-4,
        'max_iters': 1000,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        
        # Evaluation
        'eval_interval': 100,
        'log_interval': 10,
        'eval_iters': 20,
        
        # System
        'seed': 1337,
        'out_dir': 'out',
    }
    
    # For sanity check - single string memorization
    text = "I love machine learning"
    
    # Prepare data
    data_info = prepare_data(text)
    config['vocab_size'] = data_info['vocab_size']
    
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Data length: {len(data_info['data'])}")
    print(f"Characters: {data_info['chars']}")
    
    # Save tokenizer info
    os.makedirs(config['out_dir'], exist_ok=True)
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
    
    # For sanity check with loss masking on first 3 tokens
    # Create loss mask (1 where we want to compute loss, 0 where we don't)
    loss_mask = None
    if config.get('mask_first_tokens', 0) > 0:
        mask_tokens = config['mask_first_tokens']
        loss_mask = torch.ones(config['batch_size'], config['block_size'])
        loss_mask[:, :mask_tokens] = 0  # Mask first few tokens
        loss_mask = loss_mask.to(trainer.device)
    
    # Train the model
    history = trainer.train(train_data, loss_mask=loss_mask)
    
    print("Training completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()