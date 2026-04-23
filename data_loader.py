import os
import json
import numpy as np
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizer:
    def __init__(self, config):
        self.config = config
        self.word2id = {
            "<PAD>": config.pad_token_id,   # 0
            "<SOS>": config.start_token_id,  # 1
            "<EOS>": config.end_token_id,    # 2
            "<UNK>": config.unk_token_id,    # 3
            # RAG separator tokens — always in vocab
            "[CTX]": 4,
            "[Q]":   5,
        }
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_file = "vocab.json"
        
    def get_vocab_size(self):
        return self.config.vocab_size
        
    def build_vocab(self, dataframe):
        """Builds vocabulary from a pandas DataFrame containing 'question' and 'answer' columns."""
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r') as f:
                cached = json.load(f)
            # Validate: if the saved vocab is smaller than config.vocab_size,
            # it was built with an old config and must be rebuilt.
            if len(cached) >= self.config.vocab_size:
                print(f"Loading existing vocabulary from {self.vocab_file} "
                      f"(size={len(cached)})")
                self.word2id = cached
                self.id2word = {int(v): k for k, v in self.word2id.items()}
                # Always ensure RAG separator tokens are present
                for tok, tid in (("[CTX]", 4), ("[Q]", 5)):
                    if tok not in self.word2id:
                        self.word2id[tok] = tid
                        self.id2word[tid] = tok
                return
            else:
                print(f"[Vocab] Cached vocab size {len(cached)} < "
                      f"config.vocab_size {self.config.vocab_size}. "
                      f"Rebuilding …")
                os.remove(self.vocab_file)

        print("Building new vocabulary from dataset...")
        all_text = " ".join(dataframe['question'].astype(str) + " " + dataframe['answer'].astype(str))
        tokens = all_text.lower().split()
        
        # Count frequencies
        word_counts = Counter(tokens)
        
        # We reserve 6 spots: PAD, SOS, EOS, UNK, [CTX], [Q]
        max_words = self.config.vocab_size - 6
        most_common = word_counts.most_common(max_words)
        
        for idx, (word, _) in enumerate(most_common):
            # Start assigning IDs after the 6 reserved special tokens
            token_id = idx + 6
            self.word2id[word] = token_id
            self.id2word[token_id] = word
            
        # Save for inference
        with open(self.vocab_file, 'w') as f:
            json.dump(self.word2id, f)
        print(f"Saved vocabulary of size {len(self.word2id)} to {self.vocab_file}")

    def encode(self, text):
        """Converts a string to a list of token IDs."""
        tokens = str(text).lower().split()
        return [self.word2id.get(token, self.config.unk_token_id) for token in tokens]

    def decode(self, ids):
        """Converts a list/tensor of token IDs to a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        words = []
        for idx in ids:
            if idx == self.config.pad_token_id:
                continue
            word = self.id2word.get(idx, "<UNK>")
            words.append(word)
        return " ".join(words)


class QADataset(Dataset):
    def __init__(self, csv_file, tokenizer, config):
        self.df = pd.read_csv(csv_file).dropna(subset=['question', 'answer'])
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_text = row['question']
        a_text = row['answer']
        
        # Encode
        q_ids = self.tokenizer.encode(q_text)
        a_ids = [self.config.start_token_id] + self.tokenizer.encode(a_text) + [self.config.end_token_id]
        
        # Pad / Truncate Question (Source) — use src_max_len (64)
        src_len = getattr(self.config, 'src_max_len', self.config.max_seq_len)
        if len(q_ids) > src_len:
            q_ids = q_ids[:src_len]
        else:
            q_ids = q_ids + [self.config.pad_token_id] * (src_len - len(q_ids))

        # Pad / Truncate Answer (Target) — use tgt_max_len (72)
        tgt_len = getattr(self.config, 'tgt_max_len', self.config.max_seq_len)
        if len(a_ids) > tgt_len:
            a_ids = a_ids[:tgt_len]
        else:
            a_ids = a_ids + [self.config.pad_token_id] * (tgt_len - len(a_ids))
            
        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(a_ids, dtype=torch.long)


from torch.utils.data import random_split

def get_dataloaders(csv_file, config, train_ratio=0.95):
    tokenizer = SimpleTokenizer(config)
    
    # Load DF to build vocab
    df = pd.read_csv(csv_file).dropna(subset=['question', 'answer'])
    tokenizer.build_vocab(df)
    
    dataset = QADataset(csv_file, tokenizer, config)
    
    # Calculate sizes
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, tokenizer


# ── RAG-augmented dataset ─────────────────────────────────────────────────────

class RAGQADataset(Dataset):
    """
    Drop-in replacement for QADataset that optionally prepends retrieved
    context chunks to the encoder input.

    For Phase 3 curriculum training (rag_train_ratio=0.20):
    • 20 % of examples get:  [CTX] chunk1 [CTX] chunk2 [Q] question
    • 80 % of examples get:  question  (unchanged)

    If `retriever` is None, behaves identically to QADataset.

    The longer RAG sequence is padded/truncated to `rag_max_seq_len`
    (default 200) instead of `max_seq_len` (default 60).
    """

    CTX_TOKEN = "[CTX]"
    Q_TOKEN   = "[Q]"

    def __init__(
        self,
        csv_file:   str,
        tokenizer:  SimpleTokenizer,
        config,
        retriever=None,          # RAGRetriever instance or None
        rag_ratio:  float = 0.20,
        seed:       int   = 42,
    ):
        self.df        = pd.read_csv(csv_file).dropna(subset=["question", "answer"])
        self.tokenizer = tokenizer
        self.config    = config
        self.retriever = retriever
        self.rag_ratio = rag_ratio
        self.rng       = np.random.default_rng(seed)

        # Decide once per epoch which examples will receive RAG context.
        # Re-calling resample() shuffles the 20 % selection each epoch.
        self._rag_flags: np.ndarray = np.zeros(len(self.df), dtype=bool)
        self._resample()

    def _resample(self) -> None:
        """Randomly mark rag_ratio fraction of examples for RAG augmentation."""
        n = len(self.df)
        self._rag_flags[:] = False
        rag_n = int(n * self.rag_ratio)
        chosen = self.rng.choice(n, size=rag_n, replace=False)
        self._rag_flags[chosen] = True

    def __len__(self) -> int:
        return len(self.df)

    def _encode_and_pad(self, ids: list[int], max_len: int) -> list[int]:
        if len(ids) > max_len:
            return ids[:max_len]
        return ids + [self.config.pad_token_id] * (max_len - len(ids))

    def resample(self) -> None:
        """Public alias — call once per epoch to re-shuffle which examples get RAG context."""
        self._resample()

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        q_text = str(row["question"])
        a_text = str(row["answer"])

        # ── encoder input ──────────────────────────────────────────────────────
        use_rag = self.retriever is not None and self._rag_flags[idx]

        # IMPORTANT: when a retriever is attached, ALL items must be padded to
        # rag_max_seq_len — even non-RAG ones — so the DataLoader can stack
        # them into a single batch tensor (all rows must have equal length).
        if self.retriever is not None:
            final_seq_len = self.config.rag_max_seq_len
        else:
            final_seq_len = getattr(self.config, 'src_max_len', self.config.max_seq_len)

        if use_rag:
            ctx_texts = self.retriever.retrieve_text(q_text)
            if ctx_texts:
                # Build: [CTX] chunk1 [CTX] chunk2 [Q] question
                parts = []
                for chunk in ctx_texts:
                    parts.append(self.CTX_TOKEN)
                    parts.extend(str(chunk).lower().split())
                parts.append(self.Q_TOKEN)
                parts.extend(str(q_text).lower().split())
                q_ids = [
                    self.tokenizer.word2id.get(tok, self.config.unk_token_id)
                    for tok in parts
                ]
            else:
                # No useful context retrieved — fall back to plain question
                q_ids = self.tokenizer.encode(q_text)
        else:
            q_ids = self.tokenizer.encode(q_text)

        q_ids = self._encode_and_pad(q_ids, final_seq_len)

        # ── decoder target ─────────────────────────────────────────────────────
        tgt_len = getattr(self.config, 'tgt_max_len', self.config.max_seq_len)
        a_ids = (
            [self.config.start_token_id]
            + self.tokenizer.encode(a_text)
            + [self.config.end_token_id]
        )
        a_ids = self._encode_and_pad(a_ids, tgt_len)

        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(a_ids, dtype=torch.long)


def get_rag_dataloaders(
    csv_file:    str,
    config,
    retriever=None,
    train_ratio: float = 0.95,
    rag_ratio:   float = 0.20,
):
    """
    Returns dataloaders where:
      • train → RAGQADataset with retriever  (20% get [CTX] prefix)
      • val   → plain QADataset, no RAG      (clean accuracy measurement)

    Both splits use the SAME deterministic shuffle so results are comparable
    with non-RAG training runs.
    """
    from torch.utils.data import Subset

    tokenizer = SimpleTokenizer(config)
    df = pd.read_csv(csv_file).dropna(subset=["question", "answer"])
    tokenizer.build_vocab(df)

    n          = len(df)
    train_size = int(n * train_ratio)

    # Deterministic shuffle (same seed as get_dataloaders)
    generator = torch.Generator().manual_seed(42)
    all_indices   = torch.randperm(n, generator=generator).tolist()
    train_indices = all_indices[:train_size]
    val_indices   = all_indices[train_size:]

    # Train: RAG-augmented
    train_ds = RAGQADataset(
        csv_file, tokenizer, config,
        retriever=retriever,
        rag_ratio=rag_ratio,
    )

    # Val: always clean — no retriever, standard QADataset
    val_ds = QADataset(csv_file, tokenizer, config)

    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, val_loader, tokenizer
