import json
import matplotlib.pyplot as plt
import os

# Path to your training history
history_path = os.path.join("out_sanity_check", "training_history.json")

# Load training history
with open(history_path, "r") as f:
    history = json.load(f)

train_losses = history["train_losses"]
eval_interval = history["config"]["eval_interval"]
steps = [i * eval_interval for i in range(len(train_losses))]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, marker='o', label="Train Loss")
plt.yscale("log")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Sanity Check)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
