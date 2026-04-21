# inference.py
# Person 5: Inference, sample generation, text logic
import torch
from config import Config
try:
    from transformer import TransformerQA
except ImportError:
    TransformerQA = None

class MockTokenizer:
    """Mock Tokenizer since Person 1's tokenizer is missing."""
    def __init__(self, config):
        self.config = config
    
    def encode(self, text):
        # Fake encode
        return [self.config.start_token_id, 10, 20, 30, self.config.end_token_id]
        
    def decode(self, ids):
        # Fake decode
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        words = []
        for i in ids:
            if i == self.config.pad_token_id: continue
            if i == self.config.start_token_id: words.append("<SOS>")
            elif i == self.config.end_token_id: words.append("<EOS>")
            else: words.append(f"word_{i}")
        return " ".join(words)

def generate_answer(question_text):
    print(f"--- Inference ---")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if TransformerQA is None:
        print("TransformerQA unavailable.")
        return
        
    tokenizer = MockTokenizer(config)
    
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
