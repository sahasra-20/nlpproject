import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from collections import Counter

def visualize_training_history(history_file='training_history.csv'):
    """Plot training and validation metrics over epochs."""
    if not os.path.exists(history_file):
        print(f"File {history_file} not found. You need to run train.py first to generate history.")
        return

    df = pd.read_csv(history_file)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Accuracy
    ax1.plot(df['epoch'], df['train_acc'] * 100, label='Train Accuracy', marker='o', color='blue')
    if 'val_acc' in df.columns:
        ax1.plot(df['epoch'], df['val_acc'] * 100, label='Validation Accuracy', marker='o', color='orange')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot Loss
    ax2.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', color='red')
    ax2.set_title('Training Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_plot.png', dpi=300)
    print("Saved training history plot to 'training_history_plot.png'")
    # Try to show it, might fail if no display
    try:
        plt.show()
    except:
        pass

def visualize_dataset(data_file='data.csv'):
    """Analyze and plot dataset statistics."""
    if not os.path.exists(data_file):
        print(f"File {data_file} not found.")
        return
        
    print(f"Loading {data_file} for analysis...")
    df = pd.read_csv(data_file).dropna(subset=['question', 'answer'])
    
    # Word lengths
    df['q_len'] = df['question'].astype(str).apply(lambda x: len(x.split()))
    df['a_len'] = df['answer'].astype(str).apply(lambda x: len(x.split()))
    
    print(f"Total valid Q&A pairs: {len(df)}")
    print(f"Average Question Length: {df['q_len'].mean():.2f} words")
    print(f"Average Answer Length: {df['a_len'].mean():.2f} words")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Length Distributions
    sns.histplot(df['q_len'], bins=50, color='skyblue', label='Questions', ax=ax1, kde=True)
    sns.histplot(df['a_len'], bins=50, color='salmon', label='Answers', ax=ax1, kde=True)
    ax1.set_title('Distribution of Sequence Lengths')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(0, 150) # Assuming most are < 150 words
    ax1.legend()
    
    # Word frequencies
    all_text = " ".join(df['question'].astype(str) + " " + df['answer'].astype(str)).lower()
    words = all_text.split()
    
    # Filter common stop words for better visualization
    stop_words = {'the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'it', 'on', 'with', 'are', 'how', 'what', 'can'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    top_words = Counter(filtered_words).most_common(20)
    words_list, counts_list = zip(*top_words)
    
    # Plot Word Frequencies
    sns.barplot(x=list(counts_list), y=list(words_list), ax=ax2, palette='viridis')
    ax2.set_title('Top 20 Most Frequent Words (Excluding basic stopwords)')
    ax2.set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300)
    print("Saved dataset statistics to 'dataset_statistics.png'")
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    import sys
    # Don't use interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    print("--- NLP Data & Training Visualizer ---")
    visualize_dataset()
    print("\n--------------------------------------\n")
    visualize_training_history()
