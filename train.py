
# Training loop 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import math
from config import Config

try:
    from transformer import TransformerQA
except ImportError:
    TransformerQA = None

from data_loader import SimpleTokenizer, get_dataloaders, get_rag_dataloaders


# ── Embedding loader ───────────────────────────────────────────────────────────

def load_embeddings(config: Config):
    # returns a (vocab_size, hidden_dim) numpy array or None for random init
    mode = getattr(config, "embedding_mode", "bert")

    if mode == "random":
        print("[Embeddings] Using random Xavier initialization.")
        return None

    if mode == "word2vec":
        path = getattr(config, "word2vec_emb_path", "models/embeddings_word2vec.npy")
        if os.path.exists(path):
            emb = np.load(path)
            print(f"[Embeddings] Loaded Word2Vec embeddings from {path} {emb.shape}")
            return emb
        print(f"[Embeddings] Word2Vec cache not found. Training on agri corpus …")
        try:
            from word2vec import create_word2vec_embeddings
            import pandas as pd
            df = pd.read_csv("data.csv").dropna(subset=["question", "answer"])
            corpus = "\n".join(
                (row["question"] + " " + row["answer"])
                for _, row in df.iterrows()
            )
            from data_loader import SimpleTokenizer
            tok = SimpleTokenizer(config)
            tok.build_vocab(df)
            os.makedirs("models", exist_ok=True)
            emb, _ = create_word2vec_embeddings(
                tok, corpus,
                backend         = "custom",
                embedding_dim   = config.hidden_dim,
                embeddings_path = path,
                num_epochs      = 5,
                window_size     = 5,
                neg_samples     = 15,
            )
            print(f"[Embeddings] Word2Vec trained and saved → {path} {emb.shape}")
            return emb
        except Exception as e:
            print(f"[Embeddings] Word2Vec training failed: {e}. Falling back to random init.")
            return None

    # default: "bert"
    path = getattr(config, "bert_emb_path", "models/embeddings_bert_static.npy")
    expected_shape = (config.vocab_size, config.hidden_dim)
    if os.path.exists(path):
        emb = np.load(path)
        if emb.shape != expected_shape:
            print(
                f"[Embeddings] STALE cache detected: file shape {emb.shape} "
                f"!= expected {expected_shape}.\n"
                f"[Embeddings] Deleting stale file and regenerating …"
            )
            os.remove(path)
            # Fall through to regeneration below
        else:
            print(f"[Embeddings] Loaded BERT embeddings from {path} {emb.shape}")
            return emb

    print(f"[Embeddings] BERT file not found at {path}. Regenerating from BERT …")
    try:
        from pretrained_bert import load_bert_static_embeddings
        from data_loader import SimpleTokenizer
        import pandas as pd
        df  = pd.read_csv("data.csv").dropna(subset=["question", "answer"])
        tok = SimpleTokenizer(config)
        tok.build_vocab(df)
        os.makedirs("models", exist_ok=True)
        emb = load_bert_static_embeddings(
            tokenizer       = tok,
            embeddings_path = path,
            project_to_dim  = config.hidden_dim,
        )
        print(f"[Embeddings] Regenerated BERT embeddings {emb.shape}")
        return emb
    except Exception as e:
        print(f"[Embeddings] Regeneration failed: {e}. Falling back to random init.")
        return None


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def load_checkpoint(model, path: str, device):
    
    """
    Load saved weights safely.
    - Drops PositionalEncoding buffers (deterministic, shape can change with RAG).
    - Skips ANY key whose tensor shape doesn't match the current model, so
      vocab_size / hidden_dim changes never cause a RuntimeError.
    """
    if not os.path.exists(path):
        return
    state = torch.load(path, map_location=device, weights_only=True)

    # Always drop positional-encoding buffers — they're deterministic math
    state.pop("pos_encoding.pe", None)
    state.pop("decoder.pos_encoding.pe", None)

    # Filter: keep only keys that exist in the current model AND have matching shape
    model_state = model.state_dict()
    compatible, skipped = {}, []
    for k, v in state.items():
        if k not in model_state:
            skipped.append(f"{k} (not in model)")
        elif v.shape != model_state[k].shape:
            skipped.append(
                f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_state[k].shape)}"
            )
        else:
            compatible[k] = v

    if skipped:
        print(f"[Checkpoint] Skipped {len(skipped)} incompatible key(s):")
        for s in skipped:
            print(f"  ✗ {s}")
        print("[Checkpoint] Model will use fresh weights for skipped keys.")

    model.load_state_dict(compatible, strict=False)
    print(f"[Checkpoint] Loaded {len(compatible)}/{len(state)} tensors from {path}")


# ── RAG retriever helper ────────────────────────────────────────────────────────

def _try_load_retriever(model, tokenizer, config, device):
    """
    Try to load the FAISS/numpy RAG index.
    Returns a RAGRetriever on success, None if the index doesn't exist yet.
    """
    try:
        from rag_retriever import RAGRetriever, CHUNK_FILE, INDEX_FILE, VECTORS_FILE
    except ImportError as e:
        print(f"[RAG] rag_retriever import failed ({e}) — skipping RAG.")
        return None

    chunk_ok = os.path.exists(CHUNK_FILE)
    index_ok = os.path.exists(INDEX_FILE) or os.path.exists(VECTORS_FILE)

    if not chunk_ok:
        print("[RAG] rag_chunks.json not found. Run: python rag_knowledge_base.py")
        return None
    if not index_ok:
        print("[RAG] RAG index not found. Run: python rag_retriever.py")
        return None

    try:
        retriever = RAGRetriever(model, tokenizer, config, device)
        retriever.load_index(verbose=False)
        n = (retriever.index.ntotal
             if retriever.index is not None
             else len(retriever.vectors))
        print(f"[RAG] Retriever ready — {n:,} chunks. "
              f"{config.rag_train_ratio*100:.0f}% of train batches will use [CTX] context.")
        return retriever
    except Exception as e:
        print(f"[RAG] Failed to load retriever ({e}) — training without RAG.")
        return None


def _maybe_resample(train_loader):
    """Re-shuffle which examples get RAG context at the start of each epoch."""
    ds = train_loader.dataset   # may be Subset wrapping RAGQADataset
    inner = getattr(ds, 'dataset', ds)
    if hasattr(inner, 'resample'):
        inner.resample()


# ── Accuracy helper ───────────────────────────────────────────────────────────

def evaluate_accuracy(model, dataloader, device, config):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            target        = tgt[:, 1:]
            logits        = model(src, decoder_input)
            preds         = logits.argmax(dim=-1)
            pad_mask      = (target != config.pad_token_id)
            correct += ((preds == target) & pad_mask).sum().item()
            total   += pad_mask.sum().item()
    return correct / max(total, 1)


# ── Main training pipeline ────────────────────────────────────────────────────

def train_pipeline():
    print("--- Starting Training Pipeline ---")
    if TransformerQA is None:
        print("TransformerQA failed to import. Halting.")
        return

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load embeddings (bert / word2vec / random)
    embeddings = load_embeddings(config)

    # 2. Build model
    #    Use rag_max_seq_len (320) for PE so it can handle full RAG inputs.
    model = TransformerQA(
        vocab_size          = config.vocab_size,
        hidden_dim          = config.hidden_dim,
        num_encoder_layers  = config.num_encoder_layers,
        num_decoder_layers  = config.num_decoder_layers,
        num_heads           = config.num_heads,
        ff_dim              = config.ff_dim,
        dropout             = config.dropout,
        max_seq_len         = config.rag_max_seq_len,
        pad_token_id        = config.pad_token_id,
        start_token_id      = config.start_token_id,
        end_token_id        = config.end_token_id,
        embeddings          = embeddings,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Build tokenizer + vocab BEFORE retriever (retriever needs the tokenizer)
    print("Building vocabulary ...")
    tokenizer = SimpleTokenizer(config)
    _df = pd.read_csv("data.csv").dropna(subset=["question", "answer"])
    tokenizer.build_vocab(_df)
    del _df

    # 4. Load checkpoint (safe shape-filtered load)
    load_checkpoint(model, config.model_save_path, device)

    # 5. Try to attach RAG retriever (requires rag_chunks.json + rag_index.faiss)
    retriever = _try_load_retriever(model, tokenizer, config, device)

    # 6. Build dataloaders
    #    • With retriever  → RAGQADataset (30 % get [CTX] prefix each epoch)
    #    • Without         → plain QADataset
    print("Loading dataset (95/5 split) …")
    if retriever is not None:
        train_loader, val_loader, _ = get_rag_dataloaders(
            "data.csv", config,
            retriever   = retriever,
            train_ratio = 0.95,
            rag_ratio   = config.rag_train_ratio,
        )
    else:
        train_loader, val_loader, _ = get_dataloaders(
            "data.csv", config, train_ratio=0.95
        )

    print(f"Train samples : {len(train_loader.dataset):,}  |  "
          f"Val samples : {len(val_loader.dataset):,}  |  "
          f"Batches/epoch : {len(train_loader)}")

    # 7. Optimizer + warmup-cosine scheduler
    optimizer      = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    total_batches  = len(train_loader) * config.epochs
    # Warmup over ~5 % of total steps (scales with dataset size)
    warmup_steps   = max(500, total_batches // 20)

    def warmup_cosine_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps + 1)   # 0 → 1
        progress = (step - warmup_steps) / max(1, total_batches - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine → 5%

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    print(f"Device: {device}  |  Epochs: {config.epochs}  |  "
          f"Warmup steps: {warmup_steps}  |  Peak LR: {config.lr:.1e}")

    # 8. Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index    = config.pad_token_id,
        label_smoothing = 0.1,
    )

    history = []

    for epoch in range(config.epochs):
        # Reshuffle which examples get RAG context this epoch
        _maybe_resample(train_loader)

        model.train()
        total_loss = batches = train_correct = train_total = 0

        for batch_src, batch_tgt in train_loader:
            batch_src = batch_src.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()
            decoder_input = batch_tgt[:, :-1]
            target        = batch_tgt[:, 1:]
            logits        = model(batch_src, decoder_input)

            loss = criterion(logits.reshape(-1, config.vocab_size), target.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds       = logits.argmax(dim=-1)
                pad_mask    = (target != config.pad_token_id)
                train_correct += ((preds == target) & pad_mask).sum().item()
                train_total   += pad_mask.sum().item()

            total_loss += loss.item()
            batches    += 1

            if batches % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"  [Epoch {epoch+1}] Batch {batches:>4} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"GradNorm: {grad_norm:.3f}"
                )

        train_acc = train_correct / max(train_total, 1)
        val_acc   = evaluate_accuracy(model, val_loader, device, config)
        avg_loss  = total_loss / max(batches, 1)
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        history.append({
            "epoch": epoch + 1, "train_loss": avg_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })
        pd.DataFrame(history).to_csv("training_history.csv", index=False)

    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved -> {config.model_save_path}")
    print("--- Training Complete ---")


if __name__ == "__main__":
    train_pipeline()
