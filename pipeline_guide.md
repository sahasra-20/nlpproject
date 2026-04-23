# pipeline_guide.md — Agricultural Transformer QA
# Complete explanation of every stage, parameter, and run step.
# Keep this file open when debugging; it references every other file.

## Architecture at a Glance

```
Farmer question (raw text)
        │
[SimpleTokenizer]      ← data_loader.py
  word → int IDs (vocab_size=20000)
        │
[Embedding (256-dim)]  ← transformer.py _embed()
  + sqrt scaling (random only)
        │
[PositionalEncoding]   ← transformer_encoder.py
  sinusoidal, max 200 positions
        │
[Encoder ×2 layers]    ← transformer_encoder.py
  MultiHeadAttention (4 heads, head_dim=64)
  + NaN guard + FeedForward + LayerNorm
        │                    ↑
        │         [RAG Retriever]     ← rag_retriever.py
        │         cosine search over 50k+ chunks
        │         threshold: sim > 0.55
        │         prepend: [CTX] chunk [Q] question
        │
[Decoder ×2 layers]    ← transformer_decoder.py
  Self-Attention (causal)
  Cross-Attention (attends to encoder output)
  FeedForward + LayerNorm
        │
[Output Projection]
  Linear(256 → 20000) → logits
        │
Answer text
```

---

## File Map

| File | Role |
|------|------|
| `config.py` | Single source of truth for all hyperparameters |
| `data_loader.py` | Tokenizer + QADataset + RAGQADataset |
| `transformer_encoder.py` | Encoder attention, PE, FFN, Layer |
| `transformer_decoder.py` | Decoder attention, causal masks, Layer |
| `transformer.py` | TransformerQA: combines encoder+decoder, generate() |
| `word2vec.py` | 4-backend embedding trainer (custom/fasttext/gensim/bert) |
| `pretrained_bert.py` | BERT embedding extraction → models/embeddings_bert_static.npy |
| `train.py` | AdamW + Noam scheduler + label smoothing training loop |
| `inference.py` | Standard (non-RAG) inference |
| `rag_knowledge_base.py` | CSV → rag_chunks.json |
| `rag_retriever.py` | FAISS/numpy index + cosine retrieval |
| `rag_inference.py` | RAG-augmented inference |
| `pipeline_guide.md` | THIS FILE |

---

## Stage 1 — Tokenizer (`data_loader.py`)

Word-level, lowercase. Vocabulary built from `data.csv` questions + answers.

### Special token IDs (always fixed — never shift these)

| ID | Token  | Role |
|----|--------|------|
| 0  | PAD    | Padding — zero gradient, ignored by loss |
| 1  | SOS    | Start decoder input sequence |
| 2  | EOS    | Stop generation when predicted |
| 3  | UNK    | Out-of-vocabulary word |
| 4  | [CTX]  | RAG: separates retrieved context chunks |
| 5  | [Q]    | RAG: separates context from question |
| 6+ | words  | Top-(vocab_size−6) by corpus frequency |

### UNK-rate fix
Old vocab had 8,000 entries → 57% of tokens were UNK.
`vocab_size` raised to **20,000** which covers all tokens with freq≥2 in the corpus.

**After changing vocab_size: delete vocab.json, retrain.**

### Sequence lengths
- `src_max_len = 64` — encoder input (farmer question, max ~60 words)
- `tgt_max_len = 72` — decoder target (answer + SOS/EOS)
- `rag_max_seq_len = 200` — encoder input with RAG context prepended

---

## Stage 2 — Embeddings (`transformer.py`, `word2vec.py`, `pretrained_bert.py`)

### Switch: `config.embedding_mode`

**"bert"** (default)
- File: `models/embeddings_bert_static.npy`
- Shape: `(20000, 256)` — BERT's 768-dim table projected to 256
- Generate once: `python pretrained_bert.py`

**"word2vec"**
- File: `models/embeddings_word2vec.npy`
- Auto-trains if file missing (Skip-gram, window=5, neg_samples=15, 5 epochs)
- Pure PyTorch, no gensim needed
- Switch: `embedding_mode = "word2vec"` in config.py

**"random"**
- Xavier normal init (std = 1/√256)
- Good for ablation, not for production

### Scaling rule
```python
if not self.use_word2vec:      # random init only
    emb = emb * math.sqrt(self.hidden_dim)   # = 16.0
```
Pretrained vectors are already normalized — scaling would destroy positional encoding signal.

### Weight tying
- Random: `output_projection.weight = embedding.weight` — forces consistency, saves params
- Pretrained: separate Xavier-initialized projection (prevents interference)

---

## Stage 3 — Encoder (`transformer_encoder.py`)

2 × TransformerEncoderLayer:
1. Multi-Head Self-Attention (4 heads)
   - `scores = QK^T / sqrt(64)` (head_dim=64)
   - Padding mask applied: `masked_fill(mask==0, -inf)`
   - `nan_to_num(softmax, nan=0.0)` — critical: all-padding rows → all-inf → NaN without this
2. Dropout + LayerNorm (post-norm residual)
3. FeedForward: Linear(256→512) → ReLU → Dropout → Linear(512→256)
4. Dropout + LayerNorm

Mask flow:
```
src_ids → (B, S)
  create_padding_mask → (B, S)  [1=real, 0=pad]
  .unsqueeze(1).unsqueeze(1) → (B, 1, 1, S)
  → encoder self-attention mask
  → saved as src_mask_4d for cross-attention in decoder
```

---

## Stage 4 — Decoder (`transformer_decoder.py`)

2 × TransformerDecoderLayer:
1. Masked Self-Attention — sees only past tokens (causal mask)
2. Cross-Attention — Q=decoder state, K,V=encoder output
3. FeedForward

### Mask combination
```python
causal  = tril(ones(T, T))               # (T, T) — prevents future peek
tgt_pad = (tgt_ids != 0).bool()          # (B, T) — ignores padding
tgt_mask = causal * tgt_pad.unsqueeze(1) # (B, T, T)
```
Both self-attention and cross-attention have `nan_to_num` guards.

---

## Stage 5 — Training (`train.py`)

### Optimizer: AdamW
`lr=1e-4, weight_decay=0.01` — decoupled weight decay prevents L2 regularization
from being absorbed by Adam's adaptive rates.

### Scheduler: Noam
```
lr_scale = hidden_dim^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
warmup_steps = 4000
```
LR rises for 4000 steps (preventing early instability), then decays.
Peak LR ≈ `1e-4 × hidden_dim^(-0.5) × warmup^(0.5)` ≈ 0.0005

### Loss: CrossEntropyLoss
- `ignore_index=0` — PAD positions don't count
- `label_smoothing=0.1` — softens target: 90% true class, 10% spread; reduces overconfidence

### Gradient clipping: `max_norm=1.0`
Clips gradient 2-norm to 1.0 — prevents explosion especially in early warmup.

### Split: 95/5, seed=42, deterministic

---

## Stage 6 — Inference (`inference.py`)

```
python inference.py
```

What happens:
1. Load vocab.json → SimpleTokenizer
2. Load embedding matrix from config.embedding_mode path
3. Build TransformerQA with max_seq_len=200 (handles RAG-augmented inputs too)
4. Load checkpoint with strict=False, popping pos_encoding.pe (PE is deterministic,
   safe to drop when max_seq_len changes between runs)
5. Tokenize + pad question to src_max_len=64
6. Beam search decode (beam_width=4, length_penalty^0.6, rep_penalty=1.3)
7. Decode token IDs → text

---

## Stage 7 — RAG (`rag_knowledge_base.py`, `rag_retriever.py`, `rag_inference.py`)

### What RAG fixes
The model has 6.7M parameters — it can't memorize 50,000 expert Q&A pairs.
RAG lets it look up relevant knowledge at query time instead of relying on training memory.

### Step A: Knowledge Base
```
python rag_knowledge_base.py
```
Each CSV row → one chunk:
```
"Crop: cotton. Topic: pest. Q: how to control bollworm A: spray indoxacarb..."
```
Chunks > 200 words are split with 30-word overlap.
Output: `rag_chunks.json` (~50k entries)

### Step B: FAISS Index
```
python rag_retriever.py
```
1. Encode each chunk with the trained encoder (mean-pool, L2-normalize → 256-dim unit vector)
2. `faiss.IndexFlatIP` (inner product on unit vectors = cosine similarity)
3. Save: `rag_index.faiss`, `rag_vectors.npy` (numpy fallback if FAISS not installed)

### Step C: RAG Inference
```
python rag_inference.py "how to manage stem borer in paddy?"
```
1. Encode question → 256-dim unit vector
2. FAISS: top-3 cosine neighbors
3. Filter: keep only chunks with similarity > 0.55
4. Build encoder input:
   ```
   [CTX] chunk1 [CTX] chunk2 [Q] farmer question
   ```
5. Tokenize to rag_max_seq_len=200, encode, decode

### Why 0.55 threshold?
- Cosine similarity on unit vectors: 0.0 = orthogonal (random), 1.0 = identical
- 0.55 means "at least moderately related"
- Below this: irrelevant context → model answers from memory (no context prepended)
- Above this: grounded answer using retrieved evidence

### RAG Training (20% curriculum)
`RAGQADataset` randomly marks 20% of examples per epoch to receive RAG context.
The model learns when context helps and when to ignore it.
Val set is ALWAYS clean (no RAG) — measures generalization on plain questions.

---

## Full Run Order (Fresh Start)

```powershell
# 1. Delete stale vocab (vocab_size changed 12000 → 20000)
del vocab.json

# 2. Generate BERT embedding matrix (~5 min, one-time)
python pretrained_bert.py

# 3. Train the model (~hours, 25 epochs on 50k rows)
python train.py

# 4. Test plain inference
python inference.py

# 5. Build RAG knowledge base (~30 seconds)
python rag_knowledge_base.py

# 6. Build FAISS index (~2-5 minutes — uses trained encoder)
python rag_retriever.py

# 7. RAG inference
python rag_inference.py "how to control stem borer in paddy?"
```

Optional — Word2Vec instead of BERT:
```powershell
# In config.py: embedding_mode = "word2vec"
# On first train.py run, word2vec trains automatically from data.csv
python train.py
```

---

## Parameter Table

| Parameter | Value | Justification |
|-----------|-------|---------------|
| vocab_size | 20000 | Covers full corpus, eliminates 57% UNK |
| hidden_dim | 256 | Good capacity/speed balance for 50k row dataset |
| num_encoder_layers | 2 | Standard for ~6M param models |
| num_decoder_layers | 2 | Symmetric |
| num_heads | 4 | head_dim=64 (standard for small transformers) |
| ff_dim | 512 | 2× hidden_dim (paper uses 4×, 2× fits in less VRAM) |
| dropout | 0.1 | Standard transformer value |
| src_max_len | 64 | Farmer questions ≤30 words; 64 gives buffer |
| tgt_max_len | 72 | KCC answers ≤50 words + SOS/EOS |
| rag_max_seq_len | 200 | 3 chunks×~50 tokens + separators + question |
| batch_size | 32 | Standard for this model size |
| epochs | 25 | AgriQA domain converges in ~15-20 epochs |
| lr | 1e-4 | Noam peak; actual peak ~0.0005 |
| warmup_steps | 4000 | From "Attention Is All You Need" |
| weight_decay | 0.01 | AdamW regularization |
| label_smoothing | 0.1 | Reduces overconfidence |
| grad_clip | 1.0 | Prevents explosion in warmup |
| rag_min_similarity | 0.55 | Rejects irrelevant context |
| rag_top_k | 3 | 3×50 tokens fits in 200-token window |
| rag_train_ratio | 0.20 | 20% RAG examples — enough signal |

---

## Known Gotchas

1. **vocab.json mismatch** — vocab_size changed (12000→20000). Old vocab.json has 8000
   entries. Delete it; build_vocab() will create a fresh 20000-entry file.

2. **Embedding shape assertion** — transformer.py line 44 asserts shape matches
   (vocab_size, hidden_dim). If you change vocab_size, regenerate embeddings FIRST,
   then train. Loading old 8000×256 embeddings with new vocab_size=20000 will crash.

3. **PE buffer in checkpoint** — PositionalEncoding stores `pe` as a buffer in state_dict.
   If max_seq_len differs between saved model and current config, shape mismatch occurs.
   Solution (already applied): inference.py and rag_inference.py pop `pos_encoding.pe`
   before load_state_dict. PE is deterministic (pure math), so dropping it is safe.

4. **FAISS not installed** — rag_retriever.py auto-detects and falls back to numpy.
   Numpy fallback is exact but slower for large indices.
   Install: `pip install faiss-cpu`

5. **RAG index staleness** — If you retrain the model, rebuild the FAISS index too.
   The index encodes chunks with the OLD model's encoder. Mismatched encoders give
   bad retrieval. Always: `python rag_retriever.py` after `python train.py`.

6. **SOS/EOS not in inference** — inference.py adds SOS via generate(). The tokenizer's
   encode() should NOT add SOS/EOS — that's the training data_loader's job.
   decode() skips PAD, but does NOT strip EOS/SOS — generate() handles that.
