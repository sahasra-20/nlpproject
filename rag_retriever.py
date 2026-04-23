"""
rag_retriever.py
─────────────────────
Retrieval engine for the RAG pipeline.

How it works
────────────
1. Load chunks from rag_chunks.json (built by rag_knowledge_base.py).
2. Encode each chunk with YOUR OWN trained TransformerQA encoder:
      • tokenize the chunk text
      • run through encoder → mean-pool non-padding token representations
      • L2-normalise to unit sphere
3. Store all vectors in a FAISS flat-IP index (cosine similarity via
   normalised dot product).
4. At query time, encode the question the same way and retrieve the
   top-k chunks whose cosine similarity exceeds `min_similarity` (0.55).

No external embedding model is needed — your 256-dim encoder is the
only component used.

Usage (standalone):
    python rag_retriever.py
"""

from typing import List, Optional, Union, Dict
import json
import os
import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[RAG] FAISS not installed — falling back to numpy cosine search.")


CHUNK_FILE   = "rag_chunks.json"
INDEX_FILE   = "rag_index.faiss"
VECTORS_FILE = "rag_vectors.npy"     # kept for debugging / inspection

# ── Retrieval threshold (tunable) ─────────────────────────────────────────────
MIN_SIMILARITY  = 0.55   # below this → skip retrieval, answer from memory
TOP_K           = 3      # how many chunks to prepend to the encoder input


# ── Encoder helper ─────────────────────────────────────────────────────────────

class ChunkEncoder:
    """
    Wraps the TransformerQA encoder to embed arbitrary text chunks.

    encode_text(text) → numpy array shape (hidden_dim,)
    """

    def __init__(self, model, tokenizer, config, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config
        self.device    = device

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Tokenise → pad/truncate → encode → mean-pool non-pad tokens → L2-norm.
        Returns a 1-D float32 numpy array of shape (hidden_dim,).
        """
        ids = self.tokenizer.encode(text)
        # Use src_max_len (64) — same length the encoder was trained on
        seq_len = getattr(self.config, 'src_max_len', self.config.max_seq_len)
        ids = ids[:seq_len]
        ids = ids + [self.config.pad_token_id] * (seq_len - len(ids))

        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        # src_mask: 1 for real tokens, 0 for padding
        src_mask = (src != self.config.pad_token_id).float()

        # Run encoder: (1, seq_len, hidden_dim)
        self.model.eval()
        encoder_out, _ = self.model.encode(src)  # (1, seq_len, hidden_dim)

        # Mean-pool over non-padding positions
        mask_expanded = src_mask.unsqueeze(-1)            # (1, seq_len, 1)
        sum_vec   = (encoder_out * mask_expanded).sum(1)  # (1, hidden_dim)
        count     = mask_expanded.sum(1).clamp(min=1e-9)  # (1, 1)
        mean_vec  = (sum_vec / count).squeeze(0)           # (hidden_dim,)

        # L2-normalise
        norm = mean_vec.norm().clamp(min=1e-9)
        unit = (mean_vec / norm).cpu().numpy().astype(np.float32)
        return unit

    def encode_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode a list of texts, returns (N, hidden_dim) float32 array."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenise & pad
            ids_list = []
            for t in batch:
                seq_len = getattr(self.config, 'src_max_len', self.config.max_seq_len)
                ids = self.tokenizer.encode(t)[:seq_len]
                ids += [self.config.pad_token_id] * (seq_len - len(ids))
                ids_list.append(ids)

            src = torch.tensor(ids_list, dtype=torch.long, device=self.device)
            src_mask = (src != self.config.pad_token_id).float()

            self.model.eval()
            with torch.no_grad():
                enc_out, _ = self.model.encode(src)  # (B, seq_len, D)

            mask_exp = src_mask.unsqueeze(-1)
            sum_vec  = (enc_out * mask_exp).sum(1)
            count    = mask_exp.sum(1).clamp(min=1e-9)
            mean_vec = (sum_vec / count).cpu().numpy().astype(np.float32)

            # L2-normalise each row
            norms = np.linalg.norm(mean_vec, axis=1, keepdims=True).clip(min=1e-9)
            mean_vec /= norms
            all_vecs.append(mean_vec)

        return np.vstack(all_vecs)   # (N, hidden_dim)


# ── Main retriever class ────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Retrieves the top-k most relevant chunks for a farmer's question.

    Workflow:
        retriever = RAGRetriever(model, tokenizer, config, device)
        retriever.build_index()          # one-time, persists to disk
        # — or —
        retriever.load_index()           # load a previously built index

        chunks, scores = retriever.retrieve("how to manage stem borer in paddy?")
    """

    def __init__(
        self,
        model,
        tokenizer,
        config,
        device,
        chunk_file:     str   = CHUNK_FILE,
        index_file:     str   = INDEX_FILE,
        min_similarity: float = None,   # None → read from config.rag_min_similarity
        top_k:          int   = None,   # None → read from config.rag_top_k
    ):
        self.encoder    = ChunkEncoder(model, tokenizer, config, device)
        self.chunk_file = chunk_file
        self.index_file = index_file
        # Prefer explicit args; fall back to config values; then module defaults
        self.min_sim = (
            min_similarity
            if min_similarity is not None
            else getattr(config, "rag_min_similarity", MIN_SIMILARITY)
        )
        self.top_k = (
            top_k
            if top_k is not None
            else getattr(config, "rag_top_k", TOP_K)
        )

        self.chunks: list = []   # raw chunk dicts
        self.index  = None             # FAISS index  (or numpy fallback)
        self.vectors: Optional[np.ndarray] = None  # (N, D) fallback matrix

    # ── Build ──────────────────────────────────────────────────────────────────
    def build_index(self, verbose: bool = True) -> None:
        """
        Embed all chunks with the encoder and build a FAISS index.
        Results are written to INDEX_FILE and VECTORS_FILE.
        """
        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        texts = [c["text"] for c in self.chunks]
        if verbose:
            print(f"[RAG] Encoding {len(texts):,} chunks …")

        vecs = self.encoder.encode_batch(texts, batch_size=256)
        np.save(VECTORS_FILE, vecs)

        dim = vecs.shape[1]

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(dim)   # inner product on unit vectors = cosine
            index.add(vecs)
            faiss.write_index(index, self.index_file)
            self.index = index
            if verbose:
                print(f"[RAG] FAISS index built ({index.ntotal:,} vectors, dim={dim})")
        else:
            self.vectors = vecs
            if verbose:
                print(f"[RAG] Numpy fallback index built ({len(vecs):,} vectors, dim={dim})")

    # ── Load ───────────────────────────────────────────────────────────────────
    def load_index(self, verbose: bool = True) -> None:
        """Load previously built index and chunk list from disk."""
        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if FAISS_AVAILABLE and os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            if verbose:
                print(f"[RAG] Loaded FAISS index ({self.index.ntotal:,} vectors)")
        elif os.path.exists(VECTORS_FILE):
            self.vectors = np.load(VECTORS_FILE)
            if verbose:
                print(f"[RAG] Loaded numpy vectors ({len(self.vectors):,} vectors)")
        else:
            raise FileNotFoundError(
                "No index found. Run build_index() first."
            )

    # ── Query ──────────────────────────────────────────────────────────────────
    def retrieve(self, question: str) -> tuple[list[dict], list[float]]:
        """
        Retrieve the top-k chunks most similar to `question`.

        Returns:
            (chunks, scores)   — empty lists if no chunk passes min_similarity.

        Chunks below MIN_SIMILARITY (0.55) are dropped so the model falls
        back to training-memory answers when retrieval is unreliable.
        """
        q_vec = self.encoder.encode_text(question)  # (D,)
        q_vec = q_vec[np.newaxis, :]                # (1, D)

        if FAISS_AVAILABLE and self.index is not None:
            scores, idxs = self.index.search(q_vec, self.top_k)
            scores = scores[0].tolist()
            idxs   = idxs[0].tolist()
        else:
            # Numpy fallback: cosine = dot(q, V) since both are L2-normed
            sims   = (self.vectors @ q_vec.T).squeeze(-1)  # (N,)
            idxs   = np.argsort(sims)[::-1][: self.top_k].tolist()
            scores = [float(sims[i]) for i in idxs]

        # Filter by similarity threshold
        results, kept_scores = [], []
        for idx, score in zip(idxs, scores):
            if score >= self.min_sim and 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
                kept_scores.append(round(float(score), 4))

        return results, kept_scores

    def retrieve_text(self, question: str) -> list[str]:
        """Convenience: return just the chunk text strings."""
        chunks, _ = self.retrieve(question)
        return [c["text"] for c in chunks]


# ── Context formatting for the encoder input ───────────────────────────────────

CTX_SEP = "[CTX]"   # special separator token added to vocabulary
Q_SEP   = "[Q]"


def format_rag_input(question: str, context_chunks: list[str]) -> str:
    """
    Produce the augmented encoder input string:

        [CTX] chunk_1 [CTX] chunk_2 [Q] farmer question

    If context_chunks is empty, returns the bare question.
    """
    if not context_chunks:
        return question

    ctx_part = f" {CTX_SEP} ".join(context_chunks)
    return f"{CTX_SEP} {ctx_part} {Q_SEP} {question}"


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # ── Check dependencies ─────────────────────────────────────────────────────
    if not os.path.exists(CHUNK_FILE):
        print(f"[ERROR] {CHUNK_FILE} not found — run rag_knowledge_base.py first.")
        sys.exit(1)

    # ── Minimal stub model for testing without GPU ─────────────────────────────
    sys.path.insert(0, ".")
    from config import Config
    from data_loader import SimpleTokenizer
    from transformer import TransformerQA

    config    = Config()
    tokenizer = SimpleTokenizer(config)

    if not os.path.exists(tokenizer.vocab_file):
        print(f"Warning: {tokenizer.vocab_file} not found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransformerQA(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
    ).to(device)

    # Load saved weights if available
    model_path = config.model_save_path
    if os.path.exists(model_path):
        state      = torch.load(model_path, map_location=device)
        own_state  = model.state_dict()
        # Only load keys whose shape exactly matches the current model.
        # This gracefully handles vocab-size changes between checkpoints.
        compatible = {
            k: v for k, v in state.items()
            if k in own_state and v.shape == own_state[k].shape
        }
        skipped = [k for k in state if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(f"[RAG] Loaded {len(compatible)}/{len(state)} weight tensors from {model_path}")
        if skipped:
            print(f"[RAG] Skipped (shape mismatch — retrain needed): {skipped}")
    else:
        print(f"[RAG] No saved model found at {model_path} — using random weights.")

    # ── Build or load index ────────────────────────────────────────────────────
    retriever = RAGRetriever(model, tokenizer, config, device)

    if os.path.exists(INDEX_FILE) or os.path.exists(VECTORS_FILE):
        print("[RAG] Loading existing index …")
        retriever.load_index()
    else:
        print("[RAG] Building index (this takes a while) …")
        retriever.build_index()

    # ── Test retrieval ─────────────────────────────────────────────────────────
    test_questions = [
        "how to control stem borer in paddy?",
        "nutrient management for banana",
        "weed management in cotton",
    ]

    for q in test_questions:
        chunks, scores = retriever.retrieve(q)
        print(f"\nQ: {q}")
        if chunks:
            for c, s in zip(chunks, scores):
                print(f"  [{s:.3f}] {c['text'][:120]} …")
        else:
            print("  (no chunks above threshold — answer from memory)")

    # ── Format RAG input ───────────────────────────────────────────────────────
    ctx = retriever.retrieve_text(test_questions[0])
    rag_input = format_rag_input(test_questions[0], ctx)
    print(f"\nRAG encoder input (truncated):\n{rag_input[:300]} …")
    print("\nRAG retriever: OK")
