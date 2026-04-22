# inference.py
# Person 5: Inference, sample generation, text logic
import torch
from config import Config
try:
    from transformer import TransformerQA
except ImportError:
    TransformerQA = None

from data_loader import SimpleTokenizer

def generate_answer(question_text):
    print(f"--- Inference ---")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if TransformerQA is None:
        print("TransformerQA unavailable.")
        return
        
    import pandas as pd
    tokenizer = SimpleTokenizer(config)
    # Ensure vocabulary exists
    import os
    if not os.path.exists(tokenizer.vocab_file):
        df = pd.read_csv('data.csv').dropna(subset=['question', 'answer'])
        tokenizer.build_vocab(df)
    else:
        tokenizer.build_vocab(None)
    
    import numpy as np
    try:
        embeddings = np.load("models/embeddings_bert_static.npy")
    except FileNotFoundError:
        embeddings = None
        
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
    
    try:
        model.load_state_dict(torch.load(config.model_save_path, map_location=device))
        print(f"Loaded trained model from {config.model_save_path}")
    except FileNotFoundError:
        print("No trained model found. Using untrained weights for generating random sample.")
        
    model.eval()
    
    # Tokenize
    src_ids = torch.tensor([tokenizer.encode(question_text)], device=device)
    
    print(f"Q: {question_text}")
    print(f"Tokenized Q: {src_ids.tolist()[0]}")
    
    # Generate
    with torch.no_grad():
        try:
            # We use beam search = 1 (greedy)
            result_ids = model.generate(src_ids, tokenizer, max_length=20, beam_width=1)
            # The 'generate' in transformer.py returns list of ids
            if isinstance(result_ids, tuple):
                result_ids = result_ids[0] # Just in case
            
            gen_text = tokenizer.decode(result_ids)
            print(f"A: {gen_text}")
            return gen_text
        except Exception as e:
            print(f"Generation failed: {e}")
            return "Error generating response."

if __name__ == "__main__":
    generate_answer("How to treat black gram seeds?")
