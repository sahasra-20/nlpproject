import torch
import pandas as pd
from tqdm import tqdm
import evaluate

from config import Config
try:
    from transformer import TransformerQA
except ImportError:
    TransformerQA = None
from data_loader import SimpleTokenizer

def run_evaluation():
    print("--- Starting Evaluation on 5% Validation Set ---")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if TransformerQA is None:
        print("TransformerQA unavailable. Exiting.")
        return
        
    tokenizer = SimpleTokenizer(config)
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
        print("No trained model found! Performance will be zero/random.")
        
    model.eval()
    
    from data_loader import get_dataloaders
    
    print("Loading 5% Validation dataset from data.csv...")
    try:
        _, _, tokenizer = get_dataloaders('data.csv', config, train_ratio=0.95)
        # We need the raw text. Let's load the CSV again and use the same random seed to split.
        df = pd.read_csv('data.csv').dropna(subset=['question', 'answer'])
        
        # Manual split using the same logic as get_dataloaders to get validation rows
        train_size = int(len(df) * 0.95)
        val_size = len(df) - train_size
        
        # We need the exact indices used by PyTorch's random_split
        dataset_indices = list(range(len(df)))
        generator = torch.Generator().manual_seed(42)
        # Using torch.randperm to get the exact same split indices
        indices = torch.randperm(len(df), generator=generator).tolist()
        val_indices = indices[train_size:]
        
        eval_df = df.iloc[val_indices].reset_index(drop=True)
        # Optionally limit size for speed
        eval_df = eval_df.head(800) # Match the size of AI71 benchmark for fair comparison
        
        print(f"Dataset loaded. Total validation examples for evaluation: {len(eval_df)}")
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        return
        
    try:
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return
    
    predictions = []
    references = []
    rouge_references = []
    
    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        question = row['question']
        reference_answer = row['answer']
            
        src_ids = torch.tensor([tokenizer.encode(question)], device=device)
        
        with torch.no_grad():
            try:
                result_ids = model.generate(src_ids, tokenizer, max_length=50, beam_width=1)
                if isinstance(result_ids, tuple):
                    result_ids = result_ids[0]
                gen_text = tokenizer.decode(result_ids)
            except Exception as e:
                print(f"Gen error: {e}")
                gen_text = ""
                
        predictions.append(gen_text)
        references.append([reference_answer]) # BLEU expects a list of references for each prediction
        rouge_references.append(reference_answer) # ROUGE usually expects string
        
    print("\nComputing metrics...")
    
    # Ensure at least one prediction has text so BLEU doesn't crash with zero division
    has_text = any(len(p.strip()) > 0 for p in predictions)
    if not has_text:
        predictions[0] = "empty" # Dummy to avoid float division by zero
        
    # Compute metrics
    try:
        # evaluate BLEU requires references as list of lists of strings
        bleu_results = bleu_metric.compute(predictions=predictions, references=references)
        print("\n=== BLEU ===")
        for k, v in bleu_results.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Failed to compute BLEU: {e}")
        
    try:
        rouge_results = rouge_metric.compute(predictions=predictions, references=rouge_references)
        print("\n=== ROUGE ===")
        for k, v in rouge_results.items():
             print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"Failed to compute ROUGE: {e}")
        
    print("\nExample generations:")
    for i in range(min(3, len(predictions))):
        print(f"Q: {eval_df.iloc[i]['question']}")
        print(f"Ref: {rouge_references[i]}")
        print(f"Pred: {predictions[i]}")
        print("-" * 30)

if __name__ == "__main__":
    run_evaluation()
