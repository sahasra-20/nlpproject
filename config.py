
# Configuration parameters for the Agricultural Transformer QA Project

class Config:
   
    vocab_size         = 50000   
    hidden_dim         = 384     
    num_encoder_layers = 4    
    num_decoder_layers = 4       
    num_heads          = 6       
    ff_dim             = 1024    # Size of the feedforward network in the transformer layers 
    dropout            = 0.15    

    # Sequence lengths
    src_max_len = 80     #  # Max length for input questions(encoder input)
    tgt_max_len = 100   # Max length for output answers(decoder target)
    max_seq_len = 80      # Controls what model can handle internally (affects positional encodings and attention masks)

    # ── Tokenizer special tokens ──────────────────────────────────────────────
    pad_token_id   = 0   # <PAD>
    start_token_id = 1   # <SOS>
    end_token_id   = 2   # <EOS>
    unk_token_id   = 3   # <UNK>
    # IDs 4 and 5 are reserved for RAG separators [CTX] and [Q]

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size = 32      
    epochs     = 15      #
    lr         = 1e-4

    # ── File paths ────────────────────────────────────────────────────────────
    model_save_path = "transformer_qa.pth"

    # ── Embedding source switch ────────────────────────────────────────────────
    # "bert"     → loads BERT-aligned static embeddings
    # "word2vec" → loads Word2Vec embeddings trained on agri corpus
    # "random"   → Xavier random init 
    embedding_mode    = "bert"
    bert_emb_path     = "models/embeddings_bert_static.npy"
    word2vec_emb_path = "models/embeddings_word2vec.npy"

    # ── RAG settings ──────────────────────────────────────────────────────────
    rag_train_ratio    = 0.30   # Proportion of training epochs to dedicate to RAG fine-tuning (the rest is direct generation)
    rag_max_seq_len    = 320   # Max total length for RAG input (question + retrieved passages)
    rag_min_similarity = 0.50   # Minimum cosine similarity for retrieved passages to be included in RAG input
    rag_top_k          = 5      # Max number of passages to retrieve for RAG input
