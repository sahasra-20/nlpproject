# visualize_training.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(csv_path="training_history.csv", save_path="training_plots.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run training first!")
        return

    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("Error: training_history.csv is empty.")
        return

    # Set up the style
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Loss Plot
    ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', color='#e74c3c', linewidth=2)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. Accuracy Plot
    ax2.plot(df['epoch'], df['train_acc'] * 100, 'o-', label='Train Acc', color='#3498db', linewidth=2)
    ax2.plot(df['epoch'], df['val_acc'] * 100, 'o-', label='Val Acc', color='#2ecc71', linewidth=2)
    ax2.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Training plots saved to {save_path}")
    
    # Show the final results in terminal too
    last_row = df.iloc[-1]
    print("\nFinal Results:")
    print(f"  Epoch      : {int(last_row['epoch'])}")
    print(f"  Train Loss : {last_row['train_loss']:.4f}")
    print(f"  Train Acc  : {last_row['train_acc']*100:.2f}%")
    print(f"  Val Acc    : {last_row['val_acc']*100:.2f}%")

if __name__ == "__main__":
    plot_training_results()
