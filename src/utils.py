import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, eval_interval, save_path="plots/loss_curve.png"):
    steps = [i * eval_interval for i in range(len(train_losses))]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Validation Loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(save_path, dpi=300)

    # Show plot
    plt.show()

    # Free memory (important for long runs)
    plt.close()


def format_num(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)