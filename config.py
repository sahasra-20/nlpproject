# config.py
# Configuration parameters for the Agricultural Transformer QA Project

class Config:
    # ── Model architecture ────────────────────────────────────────────────────
    # Scaled up for 130k-row corpus.
    # Rule: hidden_dim / num_heads = 64 (head_dim stays constant).
    # Old → New:  vocab 30k→50k, dim 256→384, layers 2→3, heads 4→6, ff 512→1024
    vocab_size         = 50000   # covers full 130k-row token distribution
    hidden_dim         = 384     # ↑ from 256; head_dim = 384/6 = 64 (unchanged)
    num_encoder_layers = 3       # ↑ from 2
    num_decoder_layers = 3       # ↑ from 2
    num_heads          = 6       # ↑ from 4; 384/6 = 64 per head
    ff_dim             = 1024    # ↑ from 512; ~2.7× hidden_dim
    dropout            = 0.15    # ↑ from 0.1; regularise the larger model

    # Sequence lengths
    src_max_len = 80     # ↑ from 64 — longer questions in bigger corpus
    tgt_max_len = 100    # ↑ from 72 — longer answers
    max_seq_len = 80     # backward-compat alias

    # ── Tokenizer special tokens ──────────────────────────────────────────────
    pad_token_id   = 0   # <PAD>
    start_token_id = 1   # <SOS>
    end_token_id   = 2   # <EOS>
    unk_token_id   = 3   # <UNK>
    # IDs 4 and 5 are reserved for RAG separators [CTX] and [Q]

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size = 32      # User requested 32
    epochs     = 15      # ↓ from 25 — more data per epoch; fewer passes needed
    lr         = 1e-4

    # ── File paths ────────────────────────────────────────────────────────────
    model_save_path = "transformer_qa.pth"

    # ── Embedding source switch ────────────────────────────────────────────────
    # "bert"     → load BERT-aligned static embeddings
    # "word2vec" → load Word2Vec embeddings trained on agri corpus
    # "random"   → Xavier random init — use this first; switch after stable
    embedding_mode    = "bert"
    bert_emb_path     = "models/embeddings_bert_static.npy"
    word2vec_emb_path = "models/embeddings_word2vec.npy"

    # ── RAG settings ──────────────────────────────────────────────────────────
    rag_train_ratio    = 0.30   # ↑ from 0.20 — 30% of train examples get [CTX]
    rag_max_seq_len    = 320    # ↑ from 200  — 5 chunks×~50tok + sep + question
    rag_min_similarity = 0.50   # ↓ from 0.55 — slightly more permissive retrieval
    rag_top_k          = 5      # ↑ from 3    — more context chunks per query
