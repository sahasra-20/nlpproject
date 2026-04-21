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

def get_dummy_data(config, num_samples=100):
    """Generates mock data since Person 1 (preprocess) is missing."""
    src = torch.randint(3, config.vocab_size, (num_samples, 15))
    tgt = torch.randint(3, config.vocab_size, (num_samples, 20))
    # inject start/end tokens to mock realistic targets
    tgt[:, 0] = config.start_token_id
    tgt[:, -1] = config.end_token_id
    return src, tgt

def evaluate_accuracy(model, src, tgt, config):
    """Basic evaluation utility to check accuracy of the model on a mock batch."""
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)
        # logits: (batch, seq_len, vocab_size)
        preds = logits.argmax(dim=-1)
        correct = (preds == tgt).sum().item()
        total = tgt.numel()
    return correct / total

def train_pipeline():
    print("--- Starting Training Pipeline ---")
    if TransformerQA is None:
        print("TransformerQA failed to import. Halting.")
        return
        
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init model
    model = TransformerQA(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # Dummy data
    print("Generating mock data (Person 1 preprocess is pending)...")
    src, tgt = get_dummy_data(config, num_samples=config.batch_size * 5)
    src, tgt = src.to(device), tgt.to(device)
    
    print(f"Training on device: {device} for {config.epochs} epochs")
    
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(src, tgt)
        
        # Flatten for loss
        # logits: (B, S, V) -> (B*S, V)
        loss = criterion(logits.view(-1, config.vocab_size), tgt.view(-1))
        
        loss.backward()
        optimizer.step()
        
        acc = evaluate_accuracy(model, src, tgt, config)
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {loss.item():.4f} | Mock Acc: {acc*100:.2f}%")
        
    # Save model
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    train_pipeline()
