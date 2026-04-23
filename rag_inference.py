"""
rag_inference.py
────────────────
RAG-augmented answer generation.

Usage:
    python rag_inference.py "how to control stem borer in paddy?"

How it works:
1. Load trained TransformerQA model
2. Load RAG index (build first with: python rag_retriever.py)
3. Retrieve top-k context chunks for the question
4. Prepend context as: [CTX] chunk1 [CTX] chunk2 [Q] question
5. Encode augmented input → decode answer autoregressively
"""

import sys
import os
import json
from typing import List, Dict
import torch

# ── Imports ────────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from config       import Config
from data_loader  import SimpleTokenizer
from transformer  import TransformerQA
from rag_retriever import RAGRetriever, format_rag_input, CHUNK_FILE, INDEX_FILE, VECTORS_FILE


# ── Setup ──────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: Config, device: torch.device):
    tokenizer = SimpleTokenizer(config)
    if os.path.exists("vocab.json"):
        with open("vocab.json") as f:
            tokenizer.word2id = json.load(f)
        tokenizer.id2word = {int(v): k for k, v in tokenizer.word2id.items()}
    else:
        raise FileNotFoundError("vocab.json not found — run training first.")

    import numpy as np
    dummy_embeddings = np.zeros((config.vocab_size, config.hidden_dim), dtype=np.float32)

    model = TransformerQA(
        vocab_size          = config.vocab_size,
        hidden_dim          = config.hidden_dim,
        num_encoder_layers  = config.num_encoder_layers,
        num_decoder_layers  = config.num_decoder_layers,
        num_heads           = config.num_heads,
        ff_dim              = config.ff_dim,
        dropout             = 0.0,   # no dropout at inference
        max_seq_len         = config.rag_max_seq_len,  # wider window for RAG
        pad_token_id        = config.pad_token_id,
        start_token_id      = config.start_token_id,
        end_token_id        = config.end_token_id,
        embeddings          = dummy_embeddings,
    ).to(device)

    if os.path.exists(config.model_save_path):
        state     = torch.load(config.model_save_path, map_location=device)
        own_state = model.state_dict()
        # Only load tensors whose shape matches the current model.
        # Gracefully handles vocab-size mismatches across checkpoints.
        compatible = {
            k: v for k, v in state.items()
            if k in own_state and v.shape == own_state[k].shape
        }
        skipped = [k for k in state if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(f"[RAG-Inference] Loaded {len(compatible)}/{len(state)} weight tensors from {config.model_save_path}")
        if skipped:
            print(f"[RAG-Inference] Skipped (shape mismatch — retrain needed): {skipped}")
    else:
        print(f"[WARNING] No saved model at {config.model_save_path} — using random weights.")

    model.eval()
    return model, tokenizer


def build_retriever(model, tokenizer, config, device) -> RAGRetriever:
    """Load existing index or build a new one from rag_chunks.json."""
    retriever = RAGRetriever(
        model,
        tokenizer,
        config,
        device,
        chunk_file      = CHUNK_FILE,
        min_similarity  = config.rag_min_similarity,
        top_k           = config.rag_top_k,
    )

    if os.path.exists(INDEX_FILE) or os.path.exists(VECTORS_FILE):
        retriever.load_index()
    elif os.path.exists(CHUNK_FILE):
        print("[RAG-Inference] Index not found — building now (one-time, ~1-2 min) …")
        retriever.build_index()
    else:
        raise FileNotFoundError(
            f"{CHUNK_FILE} not found — run:  python rag_knowledge_base.py"
        )

    return retriever


# ── Tokenise augmented input ───────────────────────────────────────────────────

def tokenise_rag_input(
    question:      str,
    context_texts: List[str],
    tokenizer:     SimpleTokenizer,
    config:        Config,
) -> torch.Tensor:
    """
    Build the augmented encoder token sequence.

    Format: [CTX] chunk1 [CTX] chunk2 [Q] question
    If context_texts is empty, returns the plain question.
    """
    raw = format_rag_input(question, context_texts)
    tokens = raw.lower().split()

    # Look up token IDs (handles [CTX] and [Q] which are in word2id)
    ids = [
        tokenizer.word2id.get(tok, config.unk_token_id)
        for tok in tokens
    ]

    # Truncate / pad to rag_max_seq_len
    max_len = config.rag_max_seq_len
    ids = ids[:max_len]
    ids += [config.pad_token_id] * (max_len - len(ids))

    return torch.tensor([ids], dtype=torch.long)


# ── Main answer function ───────────────────────────────────────────────────────

@torch.no_grad()
def answer_with_rag(
    question:    str,
    model:       TransformerQA,
    tokenizer:   SimpleTokenizer,
    retriever:   RAGRetriever,
    config:      Config,
    device:      torch.device,
    beam_width:  int   = 4,
    temperature: float = 1.0,
    verbose:     bool  = True,
) -> dict:
    """
    Generate an answer using RAG-augmented encoder input.

    Returns dict with:
        answer      — decoded answer string
        context     — list of retrieved chunk texts
        scores      — cosine similarity scores for each chunk
        used_rag    — True if context was prepended
    """
    # 1. Retrieve
    ctx_chunks, scores = retriever.retrieve(question)
    used_rag = len(ctx_chunks) > 0
    ctx_texts = [c["text"] for c in ctx_chunks]

    if verbose:
        if used_rag:
            print(f"\n[RAG] Retrieved {len(ctx_texts)} chunk(s):")
            for txt, sc in zip(ctx_texts, scores):
                print(f"  [{sc:.3f}] {txt[:100]} …")
        else:
            print("[RAG] No context retrieved above threshold — answering from memory.")

    # 2. Build augmented encoder input
    src_ids = tokenise_rag_input(question, ctx_texts, tokenizer, config).to(device)

    # 3. Temporarily widen the model's positional encoding if needed
    model.eval()
    ids, text = model.generate(
        src_ids,
        tokenizer,
        max_length         = 80,
        beam_width         = beam_width,
        temperature        = temperature,
        repetition_penalty = 1.3,
    )

    return {
        "answer":   text,
        "context":  ctx_texts,
        "scores":   scores,
        "used_rag": used_rag,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
               "how to manage stem borer in paddy?"

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(config, device)
    retriever        = build_retriever(model, tokenizer, config, device)

    result = answer_with_rag(
        question,
        model, tokenizer, retriever, config, device,
        beam_width=4,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print("=" * 60)
