# main.py
import os
import sys
import json
import torch
import pandas as pd
import math

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_sep():
    print("-" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def show_walkthrough():
    print_header("STEP 0 — PIPELINE WALKTHROUGH")
    try:
        from config import Config
        from data_loader import SimpleTokenizer
        from transformer import TransformerQA

        config = Config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEMO_Q = "how to control stem borer in paddy?"

        print(f"Target Question: {DEMO_Q}")
        print_sep()

        # 1. Tokenizer
        tokenizer = SimpleTokenizer(config)
        if not os.path.exists(tokenizer.vocab_file):
            df = pd.read_csv("data.csv").dropna(subset=["question", "answer"])
            tokenizer.build_vocab(df)
        
        ids = tokenizer.encode(DEMO_Q)
        print(f"Tokens: {ids}")

        # 2. Model Init
        model = TransformerQA(
            vocab_size=config.vocab_size, hidden_dim=config.hidden_dim,
            num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads, ff_dim=config.ff_dim, max_seq_len=config.rag_max_seq_len
        ).to(device)

        if os.path.exists(config.model_save_path):
            state = torch.load(config.model_save_path, map_location=device)
            model.load_state_dict({k:v for k,v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}, strict=False)
            print("Loaded existing weights for walkthrough.")
        
        # 3. Generate
        model.eval()
        with torch.no_grad():
            _, answer = model.generate(torch.tensor([ids]).to(device), tokenizer, max_length=40)
        print(f"Generated: {answer}")
        print_sep()
        print("✓ Walkthrough finished.")

    except Exception as e:
        print(f"Walkthrough Error: {e}")

def step_kb():
    print_header("STEP 1 — KNOWLEDGE BASE")
    from rag_knowledge_base import build_knowledge_base
    chunks = build_knowledge_base(verbose=True)
    print(f"✓ Created {len(chunks):,} chunks.")

def step_train():
    print_header("STEP 2 — TRAINING")
    if os.path.exists("data.csv"):
        count = len(pd.read_csv("data.csv"))
        print(f"Dataset detected: {count:,} rows.")
    from train import train_pipeline
    train_pipeline()

def step_index():
    print_header("STEP 3 — RAG INDEXING")
    if not os.path.exists("rag_chunks.json"):
        print("  rag_chunks.json not found — skipping index build.")
        return

    from config import Config
    from data_loader import SimpleTokenizer
    from transformer import TransformerQA
    from rag_retriever import RAGRetriever

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = SimpleTokenizer(config)
    if not os.path.exists(tokenizer.vocab_file):
        print(f"  {tokenizer.vocab_file} not found — run training first.")
        return

    # Load model
    model = TransformerQA(
        vocab_size          = config.vocab_size,
        hidden_dim          = config.hidden_dim,
        num_encoder_layers  = config.num_encoder_layers,
        num_decoder_layers  = config.num_decoder_layers,
        num_heads           = config.num_heads,
        ff_dim              = config.ff_dim,
        dropout             = 0.0,
        max_seq_len         = config.rag_max_seq_len,
        pad_token_id        = config.pad_token_id,
        start_token_id      = config.start_token_id,
        end_token_id        = config.end_token_id,
    ).to(device)

    if os.path.exists(config.model_save_path):
        state     = torch.load(config.model_save_path, map_location=device)
        own_state = model.state_dict()
        compat    = {k: v for k, v in state.items()
                     if k in own_state and v.shape == own_state[k].shape}
        model.load_state_dict(compat, strict=False)
        print(f"  Loaded {len(compat)}/{len(state)} tensors from checkpoint")

    # Build the index
    retriever = RAGRetriever(model, tokenizer, config, device)
    retriever.build_index(verbose=True)
    print("  ✓ RAG FAISS index built and saved.")

def step_inference():
    print_header("STEP 4 — RAG INFERENCE DEMO")
    demo_questions = [
        "how to control stem borer in paddy?",
        "nutrient management for cotton crop",
        "weed management in wheat field",
    ]

    try:
        from rag_inference import load_model_and_tokenizer, build_retriever, answer_with_rag
        from config import Config

        config   = Config()
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_model_and_tokenizer(config, device)
        retriever        = build_retriever(model, tokenizer, config, device)

        for q in demo_questions:
            result = answer_with_rag(
                q, model, tokenizer, retriever, config, device,
                beam_width=4, verbose=True
            )
            print(f"\n  Q: {q}")
            print(f"  A: {result['answer']}")
            print(f"  RAG used: {result['used_rag']}  |  chunks: {len(result['context'])}")
            print()

    except Exception as e:
        print(f"  RAG inference error: {e}")
        print("  Falling back to plain inference …")
        try:
            from inference import generate_answer
            for q in demo_questions[:1]:
                generate_answer(q)
        except Exception as e2:
            print(f"  Plain inference also failed: {e2}")

def step_viz():
    print_header("STEP 5 — VISUALIZATION")
    from visualize_training import plot_training_results
    try:
        plot_training_results()
    except Exception as e:
        print(f"Visualization error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_header("AGRICULTURAL QA SYSTEM — FULL ORCHESTRATOR")
    
    # Run steps in order
    show_walkthrough()
    
    if not os.path.exists("rag_chunks.json"):
        step_kb()
        
    step_train()
    step_index()
    step_inference()
    step_viz()
    
    print_header("PIPELINE COMPLETE")
