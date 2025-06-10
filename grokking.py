import os
import time
import numpy as np
import json
import pickle
import torch
from train import Trainer
from model import GPTConfig, GPT
from train import prepare_data

# === Experiment Settings ===
prime = 97
operation = "division"
max_iters = 150_000
log_interval = 500
save_interval = 5000
out_dir = f"out_grokking/{operation}_{prime}"
os.makedirs(out_dir, exist_ok=True)

# === Load Dataset ===
with open(f"data/train_{operation}_{prime}.txt", "r") as f:
    train_text = f.read()

with open(f"data/val_{operation}_{prime}.txt", "r") as f:
    val_text = f.read()
# Prepare training and validation data
train_info = prepare_data(train_text)
val_info = prepare_data(val_text)

train_data = np.array(train_info['data'])
val_data = np.array(val_info['data'])

vocab_size = train_info['vocab_size']
char_to_idx = train_info['char_to_idx']
idx_to_char = train_info['idx_to_char']
chars = train_info['chars']


# === Save tokenizer ===
with open(os.path.join(out_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'chars': chars
    }, f)

# === GPT Model Configuration ===
model_args = dict(
    n_layer=2,
    n_head=4,
    n_embd=128,
    block_size=32,
    bias=True,
    vocab_size=vocab_size,
    dropout=0.0,
)
model = GPT(GPTConfig(**model_args))

# === Training Configuration ===
config = {
    "seed": 1337,
    "block_size": 32,
    "n_layer": 2,
    "n_head": 4,
    "n_embd": 128,
    "dropout": 0.0,
    "bias": True,
    "vocab_size": vocab_size,
    #"model": model,
    "train_data": train_data,
    "val_data": val_data,
    "out_dir": out_dir,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "max_iters": max_iters,
    "weight_decay": 1.0,
    "beta1": 0.9,
    "beta2": 0.98,
    "mask_until_equals": True,
    "eval_interval": log_interval,
    "log_interval": log_interval,
    "eval_iters": 10,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
}
 
config["char_to_idx"] = char_to_idx


trainer = Trainer(config)
trainer.train(train_data, val_data)

# === Save training history ===
history_path = os.path.join(out_dir, "training_history.json")
config_to_save = {k: v for k, v in config.items() if k not in ("train_data", "val_data")}
with open(history_path, "w") as f:
    # Remove train_data and val_data before saving config
    json.dump({
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "config": config_to_save
    }, f, indent=2)


# === Save final model ===
ckpt_path = os.path.join(out_dir, "final_model.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"Final model saved to {ckpt_path}")