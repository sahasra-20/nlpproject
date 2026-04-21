"""
word2vec.py — Unified Embedding Module for Agriculture QnA
===========================================================
One file, four backends:

  "custom"    — Train Word2Vec Skip-gram from scratch on your agri corpus
                (original implementation, no external deps beyond PyTorch)

  "fasttext"  — Facebook pretrained FastText vectors
                Best for OOV agri terms (subword n-gram fallback)
                Needs: downloaded .vec file (~2 GB)

  "gensim"    — Google News Word2Vec via gensim
                Best for general English semantics
                Needs: pip install gensim  (auto-downloads ~1.6 GB)

  "bert"      — BERT / Sentence-BERT contextual embeddings
                Best quality, context-aware
                Needs: pip install transformers sentence-transformers

Entry point used by main.py / train.py:
  embeddings, model = create_word2vec_embeddings(tokenizer, corpus_text, backend="custom")

All backends return the same thing:
  embeddings  — np.ndarray  shape (vocab_size, embedding_dim)
  model       — the trained Word2VecSkipGram model if backend="custom", else None
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from io import open as io_open
import os


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND 1 — Custom Word2Vec (Skip-gram + Negative Sampling)
# ═══════════════════════════════════════════════════════════════════════════════

class Word2VecDataset(Dataset):
    """Dataset for Word2Vec training with subsampling (Mikolov et al.)"""

    def __init__(self, tokenizer, corpus_text, window_size=5,
                 subsample_threshold=1e-3, special_token_ids=[0, 1, 2, 3]):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.special_token_ids = set(special_token_ids)

        # ── Tokenize corpus ──────────────────────────────────────────────────
        self.word_freq = Counter()
        raw_sentences = corpus_text.strip().split('\n')

        tokenized_sentences = []
        for sentence in raw_sentences:
            sent_tokens = tokenizer.encode(sentence)
            clean_tokens = [t for t in sent_tokens if t not in self.special_token_ids]
            if len(clean_tokens) > 1:
                tokenized_sentences.append(clean_tokens)
                for t in clean_tokens:
                    self.word_freq[t] += 1

        total_words = sum(self.word_freq.values())
        print(f"Total valid tokens: {total_words}")

        # ── Subsampling probabilities (Mikolov et al.) ───────────────────────
        # P(keep) = (sqrt(z/t) + 1) * (t/z)  where z = word frequency fraction
        keep_probs = {}
        for word_id, count in self.word_freq.items():
            fraction = count / total_words
            keep_probs[word_id] = (
                (np.sqrt(fraction / subsample_threshold) + 1)
                * (subsample_threshold / fraction)
            )

        # ── Negative sampling distribution (freq^0.75) ───────────────────────
        vocab_size = tokenizer.get_vocab_size()
        neg_sampling_dist = np.zeros(vocab_size)
        for word_id, freq in self.word_freq.items():
            neg_sampling_dist[word_id] = freq ** 0.75
        for st in self.special_token_ids:
            if st < vocab_size:
                neg_sampling_dist[st] = 0.0
        neg_sampling_dist /= neg_sampling_dist.sum()
        self.neg_sampling_dist = torch.from_numpy(neg_sampling_dist).float()

        # ── Build training pairs ──────────────────────────────────────────────
        self.training_pairs = []
        for sent_tokens in tokenized_sentences:
            for i, target in enumerate(sent_tokens):
                if keep_probs.get(target, 1.0) < np.random.rand():
                    continue
                dynamic_window = np.random.randint(1, self.window_size + 1)
                start = max(0, i - dynamic_window)
                end = min(len(sent_tokens), i + dynamic_window + 1)
                for j in range(start, end):
                    if i != j:
                        self.training_pairs.append((target, sent_tokens[j]))

        print(f"Training pairs after subsampling: {len(self.training_pairs)}")

        # Pre-convert to tensors for fast __getitem__
        self.targets  = torch.tensor([p[0] for p in self.training_pairs], dtype=torch.long)
        self.contexts = torch.tensor([p[1] for p in self.training_pairs], dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx], self.contexts[idx]


class Word2VecSkipGram(nn.Module):
    """Skip-gram model using Batch Matrix Multiplication"""

    def __init__(self, vocab_size, embedding_dim=256):
        super().__init__()
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim

        self.target_embedding  = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True)

        initrange = 1.0 / embedding_dim
        nn.init.uniform_(self.target_embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.context_embedding.weight)

    def forward(self, target, context, neg_samples):
        # target, context : (B,)    neg_samples : (B, K)
        target_emb  = self.target_embedding(target).unsqueeze(2)      # (B, D, 1)
        context_emb = self.context_embedding(context).unsqueeze(1)    # (B, 1, D)

        pos_score = torch.bmm(context_emb, target_emb).squeeze()      # (B,)
        pos_loss  = -torch.log(torch.sigmoid(pos_score) + 1e-10)

        neg_emb    = self.context_embedding(neg_samples)               # (B, K, D)
        neg_scores = torch.bmm(neg_emb, -target_emb).squeeze()        # (B, K)
        neg_loss   = -torch.log(torch.sigmoid(neg_scores) + 1e-10).sum(dim=1)

        return torch.mean(pos_loss + neg_loss)

    def get_embeddings(self):
        return self.target_embedding.weight.data.cpu().numpy()


class Word2VecTrainer:
    def __init__(self, model, neg_dist, neg_samples_count=15, device='cpu'):
        self.model             = model.to(device)
        self.device            = device
        self.neg_dist          = neg_dist.to(device)
        self.neg_samples_count = neg_samples_count
        # SparseAdam is faster for sparse=True embedding layers
        self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.01)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch_idx, (target, context) in enumerate(dataloader):
            target  = target.to(self.device)
            context = context.to(self.device)
            B       = target.size(0)

            neg_samples = torch.multinomial(
                self.neg_dist, B * self.neg_samples_count, replacement=True
            ).view(B, self.neg_samples_count)

            self.optimizer.zero_grad()
            loss = self.model(target, context, neg_samples)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

        return total_loss / len(dataloader)


def _train_custom(tokenizer, corpus_text, embedding_dim, num_epochs,
                  embeddings_path, batch_size, window_size, neg_samples):
    """Train Word2Vec from scratch. Returns (embeddings, model)."""
    dataset    = Word2VecDataset(tokenizer, corpus_text,
                                 window_size=window_size,
                                 special_token_ids=[0, 1, 2, 3])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model   = Word2VecSkipGram(tokenizer.get_vocab_size(), embedding_dim)
    trainer = Word2VecTrainer(model, dataset.neg_sampling_dist,
                              neg_samples_count=neg_samples, device=device)

    print(f"Training custom Word2Vec on {device}...")
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    embeddings = model.get_embeddings()
    np.save(embeddings_path, embeddings)
    print(f"Saved → {embeddings_path}  shape={embeddings.shape}")
    return embeddings, model


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND 2 — Pretrained FastText
# ═══════════════════════════════════════════════════════════════════════════════

def _load_vec_file(vec_path, max_vectors=None):
    """Parse a FastText .vec file → {word: np.ndarray}"""
    vectors = {}
    with io_open(vec_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        n_total, dim = map(int, f.readline().split())
        print(f"FastText file: {n_total:,} vectors, dim={dim}")
        for i, line in enumerate(f):
            if max_vectors and i >= max_vectors:
                break
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == dim:
                vectors[word] = vec
    print(f"Loaded {len(vectors):,} FastText vectors.")
    return vectors


def _subword_fallback(word, ft_vectors, dim, n_min=3, n_max=6):
    """Average character n-gram vectors for OOV words (FastText style)."""
    ngram_vecs = []
    padded = f"<{word}>"
    for n in range(n_min, n_max + 1):
        for start in range(len(padded) - n + 1):
            gram = padded[start: start + n]
            if gram in ft_vectors:
                ngram_vecs.append(ft_vectors[gram])
    if not ngram_vecs:
        return None
    return np.mean(ngram_vecs, axis=0).astype(np.float32)


def _load_fasttext(tokenizer, vec_path, embeddings_path, max_vectors=None):
    """Build embedding matrix from FastText .vec file."""
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(
            f"FastText .vec file not found: {vec_path}\n"
            "Download from:\n"
            "  https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip\n"
            "  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/crawl-300d-2M.vec.zip\n"
            "Unzip and pass the path as fasttext_vec_path."
        )

    ft_vectors = _load_vec_file(vec_path, max_vectors=max_vectors)
    dim        = next(iter(ft_vectors.values())).shape[0]
    vocab_size = tokenizer.get_vocab_size()
    matrix     = np.zeros((vocab_size, dim), dtype=np.float32)

    hit, oov_sub, oov_rand = 0, 0, 0

    for tid in range(vocab_size):
        try:
            token = tokenizer.decode([tid]).strip()
        except Exception:
            token = ""
        if not token:
            continue

        if token in ft_vectors:
            matrix[tid] = ft_vectors[token]
            hit += 1
        elif token.lower() in ft_vectors:
            matrix[tid] = ft_vectors[token.lower()]
            hit += 1
        else:
            fb = _subword_fallback(token.lower(), ft_vectors, dim)
            if fb is not None:
                matrix[tid] = fb
                oov_sub += 1
            else:
                matrix[tid] = np.random.uniform(-1/dim, 1/dim, dim).astype(np.float32)
                oov_rand += 1

    total = vocab_size
    print(f"\nFastText coverage:")
    print(f"  Direct hit   : {hit:>6}/{total} ({100*hit/total:.1f}%)")
    print(f"  Subword hit  : {oov_sub:>6}/{total} ({100*oov_sub/total:.1f}%)")
    print(f"  Random init  : {oov_rand:>6}/{total} ({100*oov_rand/total:.1f}%)")

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, matrix)
    print(f"Saved → {embeddings_path}  shape={matrix.shape}")
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND 3 — Pretrained Gensim / Google News Word2Vec
# ═══════════════════════════════════════════════════════════════════════════════

def _find_gensim_vector(word, model):
    """Try raw → lower → title → phrase variants before giving up."""
    for variant in [word, word.lower(), word.title(), word.replace(" ", "_")]:
        if variant in model:
            return model[variant].astype(np.float32)
    return None


def _load_gensim(tokenizer, source, embeddings_path,
                 finetune=False, corpus_path="agri_corpus.txt"):
    """Build embedding matrix from Google News Word2Vec via gensim."""
    try:
        from gensim.models import KeyedVectors
        import gensim.downloader as api
    except ImportError:
        raise ImportError("Run: pip install gensim")

    if os.path.isfile(source):
        print(f"Loading Word2Vec binary: {source}")
        kv = KeyedVectors.load_word2vec_format(source, binary=True)
    else:
        print(f"Downloading via gensim.downloader: '{source}'")
        kv = api.load(source)
    print(f"Loaded {len(kv):,} vectors, dim={kv.vector_size}")

    dim        = kv.vector_size
    vocab_size = tokenizer.get_vocab_size()
    matrix     = np.zeros((vocab_size, dim), dtype=np.float32)
    hit, oov   = 0, 0

    for tid in range(vocab_size):
        try:
            token = tokenizer.decode([tid]).strip()
        except Exception:
            token = ""
        if not token:
            continue
        vec = _find_gensim_vector(token, kv)
        if vec is not None:
            matrix[tid] = vec
            hit += 1
        else:
            matrix[tid] = np.random.uniform(-0.5/dim, 0.5/dim, dim).astype(np.float32)
            oov += 1

    total = vocab_size
    print(f"\nGensim Word2Vec coverage:")
    print(f"  Direct hit  : {hit:>6}/{total} ({100*hit/total:.1f}%)")
    print(f"  Random init : {oov:>6}/{total} ({100*oov/total:.1f}%)")

    # ── Optional fine-tune on agri corpus ─────────────────────────────────────
    if finetune and os.path.isfile(corpus_path):
        from gensim.models import Word2Vec as GensimW2V
        print(f"Fine-tuning on agri corpus: {corpus_path}")
        sentences  = [l.strip().split() for l in open(corpus_path, encoding="utf-8") if l.strip()]
        agri_model = GensimW2V(sentences=sentences, vector_size=dim,
                               window=5, min_count=1, workers=4, epochs=5)
        refined = 0
        for tid in range(vocab_size):
            try:
                token = tokenizer.decode([tid]).strip()
            except Exception:
                token = ""
            if token and token in agri_model.wv:
                matrix[tid] = agri_model.wv[token].astype(np.float32)
                refined += 1
        print(f"  Refined {refined} tokens via agri fine-tuning.")

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, matrix)
    print(f"Saved → {embeddings_path}  shape={matrix.shape}")
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND 4 — BERT / Sentence-BERT
# ═══════════════════════════════════════════════════════════════════════════════

def _load_bert_static(tokenizer, bert_model_name, embeddings_path, project_to_dim):
    """Extract BERT's token embedding table and align to your BPE vocab."""
    try:
        from transformers import BertModel, BertTokenizer as HFTok
    except ImportError:
        raise ImportError("Run: pip install transformers")

    print(f"Loading BERT: {bert_model_name}")
    hf_tok     = HFTok.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    bert_model.eval()

    bert_table = bert_model.embeddings.word_embeddings.weight.data.cpu().numpy()
    bert_vocab = hf_tok.get_vocab()
    bert_dim   = bert_table.shape[1]
    print(f"BERT vocab: {len(bert_vocab):,}  dim: {bert_dim}")

    vocab_size = tokenizer.get_vocab_size()
    raw        = np.zeros((vocab_size, bert_dim), dtype=np.float32)
    hit, oov   = 0, 0

    for tid in range(vocab_size):
        try:
            token = tokenizer.decode([tid]).strip()
        except Exception:
            token = ""
        if not token:
            continue

        bert_id = (bert_vocab.get(token)
                   or bert_vocab.get(token.lower())
                   or bert_vocab.get(f"##{token.lower()}"))

        if bert_id is not None:
            raw[tid] = bert_table[bert_id]
            hit += 1
        else:
            pieces     = hf_tok.tokenize(token)
            piece_ids  = hf_tok.convert_tokens_to_ids(pieces)
            if piece_ids:
                raw[tid] = bert_table[piece_ids].mean(axis=0)
                hit += 1
            else:
                raw[tid] = np.random.uniform(-0.02, 0.02, bert_dim).astype(np.float32)
                oov += 1

    total = vocab_size
    print(f"\nStatic BERT coverage:")
    print(f"  Aligned : {hit:>6}/{total} ({100*hit/total:.1f}%)")
    print(f"  Random  : {oov:>6}/{total} ({100*oov/total:.1f}%)")

    # Project 768 → project_to_dim if needed
    if project_to_dim and project_to_dim != bert_dim:
        print(f"  Projecting {bert_dim} → {project_to_dim}...")
        proj   = np.random.randn(bert_dim, project_to_dim).astype(np.float32)
        proj  /= np.linalg.norm(proj, axis=0, keepdims=True)
        matrix = raw @ proj
    else:
        matrix = raw

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, matrix)
    print(f"Saved → {embeddings_path}  shape={matrix.shape}")
    return matrix


def _load_bert_contextual(tokenizer, corpus_path, sbert_model_name,
                          embeddings_path, batch_size, project_to_dim):
    """Run corpus through Sentence-BERT and average token-level hidden states."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Run: pip install transformers sentence-transformers")

    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Run preprocess.py first to generate agri_corpus.txt."
        )

    print(f"Loading SBERT: {sbert_model_name}")
    hf_tok = AutoTokenizer.from_pretrained(sbert_model_name)
    model  = AutoModel.from_pretrained(sbert_model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"Running on: {device}")

    hidden_dim = model.config.hidden_size

    with open(corpus_path, encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]
    print(f"Corpus sentences: {len(sentences):,}")

    vocab_size  = tokenizer.get_vocab_size()
    embed_sum   = np.zeros((vocab_size, hidden_dim), dtype=np.float64)
    embed_count = np.zeros(vocab_size, dtype=np.int64)

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i: i + batch_size]
        enc   = hf_tok(batch, padding=True, truncation=True,
                       max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        token_embeds = out.last_hidden_state.cpu().numpy()

        for s_idx, sent in enumerate(batch):
            my_ids  = tokenizer.encode(sent)
            seq_len = min(len(my_ids), token_embeds.shape[1])
            for pos in range(seq_len):
                tid = my_ids[pos]
                if 0 <= tid < vocab_size:
                    embed_sum[tid]   += token_embeds[s_idx, pos]
                    embed_count[tid] += 1

        if (i // batch_size + 1) % 20 == 0:
            print(f"  Processed {i+len(batch):,}/{len(sentences):,} sentences")

    final  = np.zeros((vocab_size, hidden_dim), dtype=np.float32)
    seen, unseen = 0, 0
    for tid in range(vocab_size):
        if embed_count[tid] > 0:
            final[tid] = (embed_sum[tid] / embed_count[tid]).astype(np.float32)
            seen += 1
        else:
            final[tid] = np.random.uniform(-0.02, 0.02, hidden_dim).astype(np.float32)
            unseen += 1

    total = vocab_size
    print(f"\nSBERT contextual coverage:")
    print(f"  Seen   : {seen:>6}/{total} ({100*seen/total:.1f}%)")
    print(f"  Random : {unseen:>6}/{total} ({100*unseen/total:.1f}%)")

    if project_to_dim and project_to_dim != hidden_dim:
        print(f"  Projecting {hidden_dim} → {project_to_dim}...")
        proj  = np.random.randn(hidden_dim, project_to_dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        final = (final @ proj).astype(np.float32)

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, final)
    print(f"Saved → {embeddings_path}  shape={final.shape}")
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# Projection helper (shared across pretrained backends)
# ═══════════════════════════════════════════════════════════════════════════════

def _project_if_needed(matrix, target_dim, save_path):
    """Random orthogonal projection if matrix dim != target_dim."""
    current_dim = matrix.shape[1]
    if current_dim == target_dim:
        return matrix
    print(f"Projecting embeddings {current_dim} → {target_dim}...")
    proj = np.random.randn(current_dim, target_dim).astype(np.float32)
    if current_dim >= target_dim:
        proj, _ = np.linalg.qr(proj)
    proj      = proj[:current_dim, :target_dim]
    projected = (matrix.astype(np.float32) @ proj)
    np.save(save_path, projected)
    print(f"Saved projected → {save_path}  shape={projected.shape}")
    return projected


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT  (called by main.py / train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def create_word2vec_embeddings(
    tokenizer,
    corpus_text,
    backend: str = "custom",
    embedding_dim: int = 256,
    embeddings_path: str = "models/embeddings.npy",

    # ── custom backend ────────────────────────────────────────────────────────
    num_epochs: int   = 5,
    batch_size: int   = 2048,
    window_size: int  = 5,
    neg_samples: int  = 15,

    # ── fasttext backend ──────────────────────────────────────────────────────
    fasttext_vec_path: str       = "vectors/wiki-news-300d-1M.vec",
    fasttext_max_vectors: int    = 500_000,

    # ── gensim backend ────────────────────────────────────────────────────────
    gensim_source: str           = "word2vec-google-news-300",
    gensim_finetune: bool        = False,
    corpus_path: str             = "agri_corpus.txt",

    # ── bert backend ──────────────────────────────────────────────────────────
    bert_model_name: str         = "bert-base-uncased",
    bert_strategy: str           = "static",      # "static" | "contextual"
    sbert_model_name: str        = "sentence-transformers/all-MiniLM-L6-v2",
    bert_batch_size: int         = 64,
):
    """
    Unified embedding creation. Called the same way regardless of backend.

    Args:
        tokenizer      : your custom BPE tokenizer
        corpus_text    : raw corpus string (used by 'custom' backend;
                         ignored by pretrained backends)
        backend        : "custom" | "fasttext" | "gensim" | "bert"
        embedding_dim  : output vector dimension (must match transformer hidden_dim)
        embeddings_path: where to save the .npy file

        --- custom ---
        num_epochs, batch_size, window_size, neg_samples

        --- fasttext ---
        fasttext_vec_path     : path to downloaded .vec file
        fasttext_max_vectors  : max entries to load (saves RAM)

        --- gensim ---
        gensim_source         : local .bin path OR gensim downloader name
        gensim_finetune       : continue training on agri corpus
        corpus_path           : path to agri_corpus.txt

        --- bert ---
        bert_model_name       : HuggingFace model name (static strategy)
        bert_strategy         : "static" (fast) or "contextual" (best quality)
        sbert_model_name      : Sentence-BERT model (contextual strategy)
        bert_batch_size       : sentences per forward pass

    Returns:
        embeddings  — np.ndarray shape (vocab_size, embedding_dim)
        model       — Word2VecSkipGram if backend="custom", else None
    """
    backend = backend.lower().strip()

    # ── custom ────────────────────────────────────────────────────────────────
    if backend == "custom":
        embeddings, model = _train_custom(
            tokenizer, corpus_text, embedding_dim, num_epochs,
            embeddings_path, batch_size, window_size, neg_samples
        )
        return embeddings, model

    # ── fasttext ──────────────────────────────────────────────────────────────
    elif backend == "fasttext":
        embeddings = _load_fasttext(
            tokenizer, fasttext_vec_path, embeddings_path,
            max_vectors=fasttext_max_vectors
        )
        embeddings = _project_if_needed(embeddings, embedding_dim, embeddings_path)
        return embeddings, None

    # ── gensim ────────────────────────────────────────────────────────────────
    elif backend == "gensim":
        embeddings = _load_gensim(
            tokenizer, gensim_source, embeddings_path,
            finetune=gensim_finetune, corpus_path=corpus_path
        )
        embeddings = _project_if_needed(embeddings, embedding_dim, embeddings_path)
        return embeddings, None

    # ── bert ──────────────────────────────────────────────────────────────────
    elif backend == "bert":
        if bert_strategy == "static":
            embeddings = _load_bert_static(
                tokenizer, bert_model_name, embeddings_path,
                project_to_dim=embedding_dim
            )
        elif bert_strategy == "contextual":
            embeddings = _load_bert_contextual(
                tokenizer, corpus_path, sbert_model_name,
                embeddings_path, bert_batch_size,
                project_to_dim=embedding_dim
            )
        else:
            raise ValueError(f"Unknown bert_strategy '{bert_strategy}'. Use 'static' or 'contextual'.")
        return embeddings, None

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose: 'custom', 'fasttext', 'gensim', 'bert'."
        )
