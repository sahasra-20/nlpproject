# train.py
# Person 5: Training loop and evaluation
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
try:
    from transformer import TransformerQA
except ImportError:
    # Fallback if transformer is completely broken
    TransformerQA = None

from data_loader import get_dataloader

def evaluate_accuracy(model, dataloader, device):
    """Basic evaluation utility to check accuracy of the model on the dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]
            logits = model(src, decoder_input)
            
            preds = logits.argmax(dim=-1)
            # Mask out padding
            pad_mask = (target != 0)
            correct += ((preds == target) & pad_mask).sum().item()
            total += pad_mask.sum().item()
            
    return correct / max(total, 1)

def train_pipeline():
    print("--- Starting Training Pipeline ---")
    if TransformerQA is None:
        print("TransformerQA failed to import. Halting.")
        return
        
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Embeddings
    print("Loading pre-trained BERT embeddings...")
    import numpy as np
    import os
    try:
        embeddings = np.load("models/embeddings_bert_static.npy")
        print("Successfully loaded pre-trained embeddings.")
    except FileNotFoundError:
        print("Embeddings not found! (Did pretrained_bert finish?) Falling back to random initialization.")
        embeddings = None
    
    # Init model
    model = TransformerQA(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        embeddings=embeddings
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # Load real data
    print("Loading actual dataset from data.csv...")
    dataloader, tokenizer = get_dataloader('data.csv', config)
    
    print(f"Training on device: {device} for {config.epochs} epochs")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        batches = 0
        
        for batch_src, batch_tgt in dataloader:
            batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
            
            optimizer.zero_grad()
            
            decoder_input = batch_tgt[:, :-1]
            target = batch_tgt[:, 1:]
            logits = model(batch_src, decoder_input)
            
            # Flatten for loss
            loss = criterion(logits.reshape(-1, config.vocab_size), target.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
            if batches % 50 == 0:
                print(f"  [Epoch {epoch+1}] Batch {batches} | Loss: {loss.item():.4f}")
            
        acc = evaluate_accuracy(model, dataloader, device)
        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")
        
    # Save model
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    train_pipeline()
