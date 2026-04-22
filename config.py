# config.py
# Person 5: Configuration parameters for the QA Transformer Project

class Config:
    # Model parameters
    vocab_size = 8000
    hidden_dim = 256
    num_encoder_layers = 2
    num_decoder_layers = 2
    num_heads = 4
    ff_dim = 512
    dropout = 0.1
    max_seq_len = 100
    
    # Tokenizer tokens (based on transformer.py)
    pad_token_id = 0
    start_token_id = 1
    end_token_id = 2
    unk_token_id = 3

    # Training parameters
    batch_size = 32
    epochs = 25
    lr = 1e-4

    # File paths
    model_save_path = "transformer_qa.pth"
    
    # Runtime flags
    use_word2vec = False
