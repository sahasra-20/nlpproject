# inference.py
# Standard (non-RAG) inference — asks the trained model a question and returns an answer.
import os
import json
import torch
import numpy as np
from config import Config
from data_loader import SimpleTokenizer

try:
    from transformer import TransformerQA
except ImportError:
    TransformerQA = None


def _load_model_and_tokenizer(config: Config, device: torch.device):
    """Load tokenizer from vocab.json and model from checkpoint."""
    tokenizer = SimpleTokenizer(config)

    if os.path.exists(tokenizer.vocab_file):
        with open(tokenizer.vocab_file) as f:
            tokenizer.word2id = json.load(f)
        tokenizer.id2word = {int(v): k for k, v in tokenizer.word2id.items()}
        # Ensure RAG tokens are always present
        for tok, tid in (("[CTX]", 4), ("[Q]", 5)):
            if tok not in tokenizer.word2id:
                tokenizer.word2id[tok] = tid
                tokenizer.id2word[tid]  = tok
    else:
        import pandas as pd
        df = pd.read_csv("data.csv").dropna(subset=["question", "answer"])
        tokenizer.build_vocab(df)

    # Load embedding matrix if available
    emb_path = getattr(config, "bert_emb_path", "models/embeddings_bert_static.npy")
    if config.embedding_mode == "word2vec":
        emb_path = getattr(config, "word2vec_emb_path", "models/embeddings_word2vec.npy")

    # Always use dummy embeddings if real ones aren't found to prevent weight tying
    embeddings = np.zeros((config.vocab_size, config.hidden_dim), dtype=np.float32)
    if config.embedding_mode != "random" and os.path.exists(emb_path):
        embeddings = np.load(emb_path)

    # Always use rag_max_seq_len (200) so the model handles RAG-augmented inputs
    model = TransformerQA(
        vocab_size          = config.vocab_size,
        hidden_dim          = config.hidden_dim,
        num_encoder_layers  = config.num_encoder_layers,
        num_decoder_layers  = config.num_decoder_layers,
        num_heads           = config.num_heads,
        ff_dim              = config.ff_dim,
        dropout             = 0.0,          # no dropout at inference
        max_seq_len         = config.rag_max_seq_len,
        pad_token_id        = config.pad_token_id,
        start_token_id      = config.start_token_id,
        end_token_id        = config.end_token_id,
        embeddings          = embeddings,
    ).to(device)

    if os.path.exists(config.model_save_path):
        state = torch.load(config.model_save_path, map_location=device)
        # Drop PE buffers (deterministic, safe to skip on any seq-len mismatch)
        state.pop("pos_encoding.pe", None)
        state.pop("decoder.pos_encoding.pe", None)
        # Skip any key whose shape doesn't match the current model
        own_state   = model.state_dict()
        compatible  = {k: v for k, v in state.items()
                       if k in own_state and v.shape == own_state[k].shape}
        skipped     = [k for k in state if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(f"Loaded model: {len(compatible)}/{len(state)} tensors from {config.model_save_path}")
        if skipped:
            print(f"  (skipped shape-mismatched keys — retrain to fix): {skipped}")
    else:
        print(f"No checkpoint at {config.model_save_path} — using untrained weights.")

    model.eval()
    return model, tokenizer


def _pad_ids(ids: list, max_len: int, pad_id: int) -> list:
    """Truncate to max_len then right-pad with pad_id."""
    ids = ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def generate_answer(question_text: str, beam_width: int = 4) -> str:
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TransformerQA is None:
        return "TransformerQA unavailable."

    model, tokenizer = _load_model_and_tokenizer(config, device)

    # Tokenise + pad to src_max_len
    raw_ids = tokenizer.encode(question_text)
    padded  = _pad_ids(raw_ids, config.src_max_len, config.pad_token_id)
    src_ids = torch.tensor([padded], dtype=torch.long, device=device)

    print(f"Q: {question_text}")

    ids, text = model.generate(
        src_ids, tokenizer,
        max_length         = config.tgt_max_len,
        beam_width         = beam_width,
        temperature        = 1.0,
        repetition_penalty = 1.3,
    )
    print(f"A: {text}")
    return text


if __name__ == "__main__":
    generate_answer("How to treat black gram seeds?")
