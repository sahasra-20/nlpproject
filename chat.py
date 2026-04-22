# chat.py
import torch
import os
import pandas as pd
from config import Config
from data_loader import SimpleTokenizer

try:
    from transformer import TransformerQA
except ImportError:
    print("TransformerQA unavailable.")
    exit(1)

def main():
    print("========================================")
    print("      Agricultural QA Chatbot           ")
    print("========================================")
    print("Initializing system... (please wait)")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    tokenizer = SimpleTokenizer(config)
    if not os.path.exists(tokenizer.vocab_file):
        print("Vocabulary not found. Building from data.csv...")
        df = pd.read_csv('data.csv').dropna(subset=['question', 'answer'])
        tokenizer.build_vocab(df)
    else:
        tokenizer.build_vocab(None)
        
    # 2. Init Model
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
    
    # 3. Load Weights
    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path, map_location=device))
        print(f"Loaded trained weights from '{config.model_save_path}'!")
    else:
        print(f"WARNING: '{config.model_save_path}' not found!")
        print("The model is untrained. Responses will be random garbage.")
        
    model.eval()
    
    print("\nChatbot is ready! Type 'quit' or 'exit' to stop.")
    print("-" * 40)
    
    # 4. Interactive Loop
    while True:
        try:
            question = input("\nFarmer > ")
            if question.strip().lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            if not question.strip():
                continue
                
            # Tokenize
            src_ids = torch.tensor([tokenizer.encode(question)], device=device)
            
            # Generate
            with torch.no_grad():
                result_ids = model.generate(src_ids, tokenizer, max_length=20, beam_width=1)
                if isinstance(result_ids, tuple):
                    result_ids = result_ids[0]
                
                # --- ADD THIS LINE ---
                print(f"[Debug] Raw IDs: {result_ids}")
                
                answer = tokenizer.decode(result_ids)
                print(f"Bot    > {answer}")
            # # Generate
            # with torch.no_grad():
            #     result_ids = model.generate(src_ids, tokenizer, max_length=20, beam_width=1)
            #     if isinstance(result_ids, tuple):
            #         result_ids = result_ids[0]
                
            #     answer = tokenizer.decode(result_ids)
            #     print(f"Bot    > {answer}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
