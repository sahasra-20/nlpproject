"""
Contextual Embeddings via BERT / Sentence-BERT
===============================================
Unlike Word2Vec / FastText / GloVe (static — one vector per word),
BERT produces CONTEXTUAL vectors: the same word gets a different vector
depending on the sentence it appears in.

This file gives you two strategies:

  Strategy A — Static BERT token embeddings
  ------------------------------------------
  Extract BERT's raw token embedding table (the nn.Embedding layer from
  its vocabulary).  These are NOT contextual — they're just the lookup
  table before the transformer layers.  Suitable as a drop-in replacement
  for Word2Vec because the shape is (vocab_size, hidden_dim).

  Strategy B — Mean-pooled sentence embeddings  (recommended for QnA)
  --------------------------------------------------------------------
  Run all Q&A pairs through BERT / Sentence-BERT and store mean-pooled
  CLS representations.  Then map each tokenizer token to its average
  contextualised vector across the corpus.  More expensive but gives
  richer agricultural semantics.

Install:
  pip install transformers torch sentence-transformers

Models used:
  - 'bert-base-uncased'                       (~440 MB)  general English
  - 'sentence-transformers/all-MiniLM-L6-v2'  (~90 MB)   fast, good for QnA
  - 'sentence-transformers/all-mpnet-base-v2'  (~420 MB)  best quality
"""

import numpy as np
import os
import torch


# ── Strategy A: Static BERT token embedding table ─────────────────────────────

def load_bert_static_embeddings(
    tokenizer,
    bert_model_name: str = "bert-base-uncased",
    embeddings_path: str = "models/embeddings_bert_static.npy",
    project_to_dim: int | None = 256,
) -> np.ndarray:
    """
    Extract BERT's static token embedding table and align it to your
    custom BPE tokenizer's vocabulary.

    Because BERT has its own vocab (~30k), and your BPE tokenizer has a
    different vocab (~8k), this function aligns them by surface string.
    Tokens not found in BERT's vocab get random init.

    Args:
        tokenizer       : your custom BPE tokenizer.
        bert_model_name : any HuggingFace model name or local path.
        embeddings_path : output .npy path.
        project_to_dim  : if set, project BERT's 768-dim vectors down to
                          this size (must match your transformer hidden_dim).
                          Set to None to keep original 768.

    Returns:
        np.ndarray shape (vocab_size, project_to_dim or 768).
    """
    from transformers import BertModel, BertTokenizer as HFBertTokenizer

    print(f"Loading BERT model: {bert_model_name}")
    hf_tokenizer = HFBertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    bert_model.eval()

    # BERT's own embedding table: shape (bert_vocab_size, 768)
    bert_embed_table = bert_model.embeddings.word_embeddings.weight.data.cpu().numpy()
    bert_vocab = hf_tokenizer.get_vocab()          # { word_piece : bert_token_id }
    bert_dim = bert_embed_table.shape[1]           # 768

    print(f"BERT vocab size : {len(bert_vocab):,}")
    print(f"BERT embed dim  : {bert_dim}")

    vocab_size = tokenizer.get_vocab_size()
    raw_matrix = np.zeros((vocab_size, bert_dim), dtype=np.float32)

    hit, oov = 0, 0

    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id]).strip()
        except Exception:
            token_str = ""

        if not token_str:
            continue

        # Try direct lookup in BERT's vocab (BERT uses WordPiece)
        # Also try the '##' continuation prefix used by WordPiece
        bert_id = bert_vocab.get(token_str) \
               or bert_vocab.get(token_str.lower()) \
               or bert_vocab.get(f"##{token_str.lower()}")

        if bert_id is not None:
            raw_matrix[token_id] = bert_embed_table[bert_id]
            hit += 1
        else:
            # Decompose via BERT's own tokenizer and average subpiece vectors
            pieces = hf_tokenizer.tokenize(token_str)
            piece_ids = hf_tokenizer.convert_tokens_to_ids(pieces)
            if piece_ids:
                raw_matrix[token_id] = bert_embed_table[piece_ids].mean(axis=0)
                hit += 1
            else:
                raw_matrix[token_id] = np.random.uniform(
                    -0.02, 0.02, bert_dim
                ).astype(np.float32)
                oov += 1

    total = vocab_size
    print(f"\nStatic BERT embedding coverage:")
    print(f"  Aligned  : {hit:>6} / {total}  ({100*hit/total:.1f}%)")
    print(f"  Random   : {oov:>6} / {total}  ({100*oov/total:.1f}%)")

    # Optional linear projection 768 → project_to_dim
    if project_to_dim and project_to_dim != bert_dim:
        print(f"  Projecting {bert_dim} to {project_to_dim} dims...")
        # Simple random projection (preserves approximate distances)
        proj = np.random.randn(bert_dim, project_to_dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        final_matrix = raw_matrix @ proj
    else:
        final_matrix = raw_matrix

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, final_matrix)
    print(f"Saved to {embeddings_path}  shape={final_matrix.shape}")

    return final_matrix


# ── Strategy B: Contextual mean-pooled embeddings ─────────────────────────────

def load_sbert_contextual_embeddings(
    tokenizer,
    corpus_path: str,
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embeddings_path: str = "models/embeddings_sbert.npy",
    batch_size: int = 64,
    project_to_dim: int | None = 256,
) -> np.ndarray:
    """
    For every token in your vocabulary, collect all sentences from the
    agricultural corpus that contain that token, run them through
    Sentence-BERT, and average the token-level hidden states.

    This gives a genuine contextual representation tuned to your corpus.

    NOTE: This is the most expensive option (~minutes on GPU, ~hours on CPU
    for a large corpus).  Run once and save the .npy file.

    Args:
        tokenizer       : your custom BPE tokenizer.
        corpus_path     : path to agri_corpus.txt (one sentence per line).
        sbert_model_name: any sentence-transformers model.
        embeddings_path : output .npy path.
        batch_size      : sentences per forward pass.
        project_to_dim  : project hidden states to this dim (match hidden_dim).

    Returns:
        np.ndarray shape (vocab_size, project_to_dim or sbert_hidden_dim).
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading SBERT model: {sbert_model_name}")
    hf_tok = AutoTokenizer.from_pretrained(sbert_model_name)
    model  = AutoModel.from_pretrained(sbert_model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"Running on: {device}")

    hidden_dim = model.config.hidden_size

    # Load corpus sentences
    with open(corpus_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Corpus sentences: {len(sentences):,}")

    vocab_size = tokenizer.get_vocab_size()

    # Accumulators: sum of hidden states + count per token_id
    embed_sum   = np.zeros((vocab_size, hidden_dim), dtype=np.float64)
    embed_count = np.zeros(vocab_size, dtype=np.int64)

    def mean_pool(token_embeddings, attention_mask):
        """Average non-padding token hidden states."""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i: i + batch_size]

        encoded = hf_tok(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # token_embeddings: (batch, seq_len, hidden_dim)
        token_embeds = output.last_hidden_state.cpu().numpy()
        # Also re-encode with YOUR tokenizer to find which token_ids appear
        for sent_idx, sentence in enumerate(batch_sentences):
            my_token_ids = tokenizer.encode(sentence)
            seq_len = min(len(my_token_ids), token_embeds.shape[1])

            for pos in range(seq_len):
                tid = my_token_ids[pos]
                if 0 <= tid < vocab_size:
                    embed_sum[tid]   += token_embeds[sent_idx, pos]
                    embed_count[tid] += 1

        if (i // batch_size + 1) % 20 == 0:
            print(f"  Processed {i + len(batch_sentences):,} / {len(sentences):,} sentences")

    # Average accumulated vectors; random-init for unseen tokens
    final_matrix = np.zeros((vocab_size, hidden_dim), dtype=np.float32)
    seen, unseen = 0, 0
    for tid in range(vocab_size):
        if embed_count[tid] > 0:
            final_matrix[tid] = (embed_sum[tid] / embed_count[tid]).astype(np.float32)
            seen += 1
        else:
            final_matrix[tid] = np.random.uniform(-0.02, 0.02, hidden_dim).astype(np.float32)
            unseen += 1

    print(f"\nSBERT contextual embedding coverage:")
    print(f"  Seen in corpus : {seen:>6} / {vocab_size}  ({100*seen/vocab_size:.1f}%)")
    print(f"  Random init    : {unseen:>6} / {vocab_size}  ({100*unseen/vocab_size:.1f}%)")

    # Optional projection
    if project_to_dim and project_to_dim != hidden_dim:
        print(f"  Projecting {hidden_dim} to {project_to_dim} dims...")
        proj = np.random.randn(hidden_dim, project_to_dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        final_matrix = (final_matrix @ proj).astype(np.float32)

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, final_matrix)
    print(f"Saved to {embeddings_path}  shape={final_matrix.shape}")

    return final_matrix


# ── usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import SimpleTokenizer
    from config import Config
    import pandas as pd
    
    config = Config()
    tokenizer = SimpleTokenizer(config)
    
    # Load DF to build vocab
    print("Building vocabulary from data.csv...")
    df = pd.read_csv("data.csv").dropna(subset=["question", "answer"])
    tokenizer.build_vocab(df)

    # ── Option A: fast static BERT embeddings ─────────────────────────────────
    embeddings = load_bert_static_embeddings(
        tokenizer=tokenizer,
        bert_model_name="bert-base-uncased",
        embeddings_path="models/embeddings_bert_static.npy",
        project_to_dim=256,     # match your transformer hidden_dim
    )

    print(f"\nFinal matrix: {embeddings.shape}")
    print("Done. Use the saved .npy path in transformer.py")
