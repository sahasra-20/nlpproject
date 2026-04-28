import os
import json
import numpy as np
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class SimpleTokenizer:
    def __init__(self, config):
        self.config = config
        self.vocab_file = "vocab_bpe.json"
        self.tokenizer = None
        
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "[CTX]", "[Q]"]
        self._word2id = {}
        self._id2word = {}
        
    @property
    def word2id(self):
        if not self._word2id and self.tokenizer:
            self._word2id = self.tokenizer.get_vocab()
        return self._word2id

    @property
    def id2word(self):
        if not self._id2word and self.tokenizer:
            self._id2word = {v: k for k, v in self.tokenizer.get_vocab().items()}
        return self._id2word

    def get_vocab_size(self):
        return self.config.vocab_size
        
    def build_vocab(self, dataframe):
        """Builds BPE vocabulary from a pandas DataFrame containing 'question' and 'answer' columns."""
        if os.path.exists(self.vocab_file):
            self.tokenizer = Tokenizer.from_file(self.vocab_file)
            if self.tokenizer.get_vocab_size() >= self.config.vocab_size:
                print(f"Loading existing BPE vocabulary from {self.vocab_file} "
                      f"(size={self.tokenizer.get_vocab_size()})")
                return
            else:
                print(f"[Vocab] Cached vocab size {self.tokenizer.get_vocab_size()} < "
                      f"config.vocab_size {self.config.vocab_size}. "
                      f"Rebuilding ...")
                os.remove(self.vocab_file)

        print("Building new BPE vocabulary from dataset...")
        
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=self.special_tokens
        )
        
        # Generator for dataset strings
        def get_training_corpus():
            for i in range(0, len(dataframe), 1000):
                chunk = dataframe.iloc[i:i+1000]
                for text in chunk['question'].astype(str).tolist() + chunk['answer'].astype(str).tolist():
                    yield text
                
        self.tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        self.tokenizer.save(self.vocab_file)
        print(f"Saved BPE vocabulary of size {self.tokenizer.get_vocab_size()} to {self.vocab_file}")

    def encode(self, text):
        """Converts a string to a list of token IDs."""
        if not self.tokenizer:
            if os.path.exists(self.vocab_file):
                self.tokenizer = Tokenizer.from_file(self.vocab_file)
            else:
                return []
        return self.tokenizer.encode(str(text)).ids

    def decode(self, ids):
        """Converts a list/tensor of token IDs to a string."""
        if not self.tokenizer:
            if os.path.exists(self.vocab_file):
                self.tokenizer = Tokenizer.from_file(self.vocab_file)
            else:
                return ""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        # Remove pad tokens before decoding
        ids = [i for i in ids if i != self.config.pad_token_id]
        return self.tokenizer.decode(ids)


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
        self.df        = pd.read_csv(csv_file).dropna(subset=["question", "answer"])# remove rows with missing Q&A pairs  
        self.tokenizer = tokenizer
        self.config    = config
        self.retriever = retriever
        # A reference to retriever object, typically an instance of a RAGRetriever,imported from rag_retriever.py in this project
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
