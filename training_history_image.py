import json
import matplotlib.pyplot as plt

# Load training history
with open("out_grokking/division_97/training_history.json") as f:
    history = json.load(f)

eval_interval = history['config']['eval_interval']
iters = [i * eval_interval for i in range(len(history['train_losses']))]

# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(iters, history['train_losses'], label="Train Loss")
plt.plot(iters, history['val_losses'], label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.title("Grokking Curve on Modular Division (p=97)")
plt.legend()
plt.grid(True)
plt.savefig("out_grokking/division_97/grokking_curve.png")
plt.show()
