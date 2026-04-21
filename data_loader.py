import os
import json
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizer:
    def __init__(self, config):
        self.config = config
        self.word2id = {
            "<PAD>": config.pad_token_id,
            "<SOS>": config.start_token_id,
            "<EOS>": config.end_token_id,
            "<UNK>": config.unk_token_id
        }
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_file = "vocab.json"
        
    def get_vocab_size(self):
        return self.config.vocab_size
        
    def build_vocab(self, dataframe):
        """Builds vocabulary from a pandas DataFrame containing 'question' and 'answer' columns."""
        if os.path.exists(self.vocab_file):
            print(f"Loading existing vocabulary from {self.vocab_file}")
            with open(self.vocab_file, 'r') as f:
                self.word2id = json.load(f)
            self.id2word = {int(v): k for k, v in self.word2id.items()}
            return

        print("Building new vocabulary from dataset...")
        all_text = " ".join(dataframe['question'].astype(str) + " " + dataframe['answer'].astype(str))
        tokens = all_text.lower().split()
        
        # Count frequencies
        word_counts = Counter(tokens)
        
        # We reserve 4 spots for PAD, SOS, EOS, UNK
        max_words = self.config.vocab_size - 4
        most_common = word_counts.most_common(max_words)
        
        for idx, (word, _) in enumerate(most_common):
            # Start assigning IDs after the 4 reserved special tokens
            token_id = idx + 4
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
        
        # Pad / Truncate Question (Source)
        if len(q_ids) > self.config.max_seq_len:
            q_ids = q_ids[:self.config.max_seq_len]
        else:
            q_ids = q_ids + [self.config.pad_token_id] * (self.config.max_seq_len - len(q_ids))
            
        # Pad / Truncate Answer (Target)
        if len(a_ids) > self.config.max_seq_len:
            a_ids = a_ids[:self.config.max_seq_len]
        else:
            a_ids = a_ids + [self.config.pad_token_id] * (self.config.max_seq_len - len(a_ids))
            
        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(a_ids, dtype=torch.long)


def get_dataloader(csv_file, config):
    tokenizer = SimpleTokenizer(config)
    
    # Load DF to build vocab
    df = pd.read_csv(csv_file).dropna(subset=['question', 'answer'])
    tokenizer.build_vocab(df)
    
    dataset = QADataset(csv_file, tokenizer, config)
    
    # Drop last to avoid batch size mismatches during training loops if dataset isn't perfectly divisible
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    return dataloader, tokenizer
