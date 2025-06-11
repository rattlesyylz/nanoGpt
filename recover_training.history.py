import json
import os
import torch
import numpy as np

from train import Trainer
from model import GPTConfig, GPT

# Path to the saved model
out_dir = "out_grokking/division_97"
checkpoint_path = os.path.join(out_dir, "final_model.pt")

# --- FIX: explicitly allow GPTConfig for torch.load ---
import torch.serialization
torch.serialization.add_safe_globals([GPTConfig])

# Load checkpoint with weights_only=False to get full training state
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Recover training info
config = checkpoint.get('config', {})
train_losses = checkpoint.get('train_losses', [])
val_losses = checkpoint.get('val_losses', [])

# Filter out non-JSON-serializable values
serializable_config = {
    k: v for k, v in config.items()
    if not isinstance(v, (np.ndarray, list, dict))
}

# Save as training history
output_path = os.path.join(out_dir, "training_history.json")
with open(output_path, "w") as f:
    json.dump({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": serializable_config,
    }, f, indent=2)

print(f"Recovered training history saved to: {output_path}")
