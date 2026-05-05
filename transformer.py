import math
import torch
import torch.nn as nn
import numpy as np
from transformer_encoder import TransformerEncoder, PositionalEncoding
from transformer_decoder import TransformerDecoder, create_causal_mask, create_padding_mask


class TransformerQA(nn.Module):
    # Input:  Tokenized farmer question  e.g. [34, 891, 12, 445, 203]
    # Output: Generated answer text      e.g. "Black gram seeds should be treated..."
    
    def __init__(
        self,
        vocab_size,
        hidden_dim=384,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=6,
        ff_dim=1024,
        # there is a matrix of size (256 × 512) which expands features \
        # next activation adds non-linearity W2 (512×256): compresses backreturns to the standard size so layers stack cleanly
        dropout=0.15,
        # Dropout is applied per element in each vector [256 values] words randomly become 0 with a prob of 0.1
        embeddings=None,
        pad_token_id=0,
        start_token_id=1,
        end_token_id=2,
        max_seq_len=80
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_seq_len     = max_seq_len
        
        # Embedding layer
        if embeddings is not None:
            # Use pretrained embeddings

            assert embeddings.shape == (vocab_size, hidden_dim), (
                f"Word2Vec embedding shape {embeddings.shape} "
                f"must match (vocab_size={vocab_size}, hidden_dim={hidden_dim})"
            )
            # assert is used to check if a condition is true while debugging

            self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(embeddings).float(), # our embedding matrix
                padding_idx=pad_token_id,  # pad token's embedding stays zero, no gradient
                freeze=False               # allow fine-tuning
            )
            self.use_word2vec = True

            # row index must match your tokenizer’s word index.

        else:
            # Random initialization
            self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

            nn.init.normal_(self.embedding.weight, mean=0, std=hidden_dim ** -0.5)
            # N(0,1/sqrt(dim)) 
            # Reset padding embedding to zero (it will not receive gradient anyway)
            with torch.no_grad():
                self.embedding.weight[pad_token_id].fill_(0)
            self.use_word2vec = False
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        # (hidden_dim) → (vocab_size)
        # scores = logits
        
        if not self.use_word2vec:
            self.output_projection.weight = self.embedding.weight


        # Initialize output projection separately (if not tied)
        if self.use_word2vec:
            nn.init.xavier_uniform_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
            # If using Word2Vec/pretrained bert → separate initialization

        self.dropout = nn.Dropout(dropout)

    def _embed(self, token_ids):

        # Args:
        #     token_ids: (batch, seq_len) integer tensor

        # Returns:
        #     (batch, seq_len, hidden_dim) float tensor with position information

        # token IDs → meaningful vectors + position info

        emb = self.embedding(token_ids)
        # Always scale embeddings so positional encodings don't dominate, to Keep embedding magnitude comparable to positional encoding
        emb = emb * math.sqrt(self.hidden_dim)
        emb = self.pos_encoding(emb)
    
        emb = self.dropout(emb)
        # Regularization ,Randomly zeroes some values
        # emb: (batch, seq_len, hidden_dim)

        return emb
    
    def encode(self, src_ids, src_mask=None):
        """
        Encode source (question)
        
        Args:
            src_ids: (batch_size, src_seq_len)
            src_mask: (batch_size, src_seq_len)
        
        Returns:
            encoder_output: (batch_size, src_seq_len, hidden_dim)
        """
        # Embedding + positional encoding
        src_emb = self._embed(src_ids)
        
        # Create padding mask
        if src_mask is None:
            src_mask = create_padding_mask(src_ids, self.pad_token_id) # 1 → real tokens,0 → padding With mask model ignores padding
        
        # Expand mask for multi-head attention
        src_mask_4d = src_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, src_len)
        # unsqueeze(dim) adds a new dimension of size 1 at position dim

        # Attention scores # Q @ Kᵀ scores: (32, 6, 80, 80)(dimension_) so 4d mask

        encoder_output = self.encoder(src_emb, mask=src_mask_4d)
        
        return encoder_output,src_mask_4d
    
    def decode(self, tgt_ids, encoder_output, src_mask_4d):
        tgt_seq_len = tgt_ids.shape[1]
        tgt_emb = self._embed(tgt_ids)

    # ── Causal mask (future masking) ─────────────────────
        causal = create_causal_mask(tgt_seq_len, device=tgt_ids.device)
        causal = causal.unsqueeze(0)  # (1, tgt_len, tgt_len)

    # ── Padding mask ────────────────────────────────────
        tgt_pad = create_padding_mask(tgt_ids, self.pad_token_id)  # (batch, tgt_len)

    # Combine masks
        tgt_mask = causal * tgt_pad.unsqueeze(1)  # (batch, tgt_len, tgt_len)

    # Expand for multi-head attention
        tgt_mask_4d = tgt_mask.unsqueeze(1)  # (batch, 1, tgt_len, tgt_len)

    # Decode
        decoder_output = self.decoder(
        tgt_emb,
        encoder_output,
        self_mask=tgt_mask_4d,
        cross_mask=src_mask_4d
        )

    # Project to vocab
        logits = self.output_projection(decoder_output)

        return logits
    
    def forward(self, src_ids, tgt_ids, src_mask=None):
        """
        
        Args:
            src_ids: (batch_size, src_seq_len)
            tgt_ids: (batch_size, tgt_seq_len)
            src_mask: (batch_size, src_seq_len)
            tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len)
        
        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size)
        """
        encoder_output, src_mask_4d = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, encoder_output, src_mask_4d)
        return logits


    @torch.no_grad()
    def generate(
        self,
        src_ids,
        tokenizer,
        max_length=50,
        beam_width=1,
        temperature=1.0,
        repetition_penalty=1.3
    ):
        """
        Generates answer
        
        Args:
            src_ids: (1, src_seq_len) or list of token ids
            tokenizer: tokenizer instance
            max_length: maximum sequence length
            beam_width: 1 for greedy, >1 for beam search
            temperature: temperature for sampling
        
        Returns:
            generated_ids: list of token ids
            generated_text: string
        """
        if isinstance(src_ids, list):
            src_ids = torch.tensor([src_ids], dtype=torch.long).to(next(self.parameters()).device)
        
        if src_ids.dim() == 1:
            src_ids = src_ids.unsqueeze(0)
        
        device = next(self.parameters()).device
        src_ids = src_ids.to(device)

        self.eval()

        encoder_output, src_mask_4d = self.encode(src_ids)


        if beam_width == 1:
            ids = self._greedy_decode(
                encoder_output, src_mask_4d,
                max_length, temperature, repetition_penalty, device
            )
        else:
            ids = self._beam_search(
                encoder_output, src_mask_4d,
                max_length, beam_width, repetition_penalty, device
            )
        
        text = tokenizer.decode(ids)
        return ids, text


    def _greedy_decode(
       self, encoder_output, src_mask_4d,
                max_length, temperature, repetition_penalty, device
    ):

        generated = [self.start_token_id]  # Start with <SOS>

        for step in range(max_length):
            # Convert generated ids to tensor
            tgt_ids = torch.tensor([generated], dtype=torch.long, device=device)
            # tgt_ids: (1, current_len)

            # Get logits for all positions — we only need the LAST position
            logits = self.decode(tgt_ids, encoder_output, src_mask_4d)
            # logits: (1, current_len, vocab_size)

            # Take logits for the LAST position only — this is the next token prediction
            next_logits = logits[0, -1, :].clone()
            # next_logits: (vocab_size,)

            # Apply repetition penalty to already-generated tokens
            # WHY? Transformers tend to repeat themselves without this.
            # Agricultural answers often have the same word in dataset
            # multiple times — model can get stuck in a loop.
            for prev_token in set(generated):
                if next_logits[prev_token] > 0:
                    next_logits[prev_token] /= repetition_penalty
                else:
                    next_logits[prev_token] *= repetition_penalty
            # This makes already-generated tokens less likely, not impossible

            # Prevent generating EOS at the very first step
            if step == 0:
                next_logits[self.end_token_id] = float('-inf')

            # temperature < 1.0: sharper distribution (more confident, less diverse)
            # temperature > 1.0: flatter distribution (more random, more diverse)
            # temperature = 1.0: unchanged
            if temperature != 1.0:
                next_logits = next_logits / temperature
                # Sample from distribution (for temp != 1.0)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy: just take the argmax
                next_token = next_logits.argmax().item()

            # Check for end token
            # FIX: original code removed end_token and continued — BUG
            # Correct: break immediately when end_token is generated
            if next_token == self.end_token_id:
                break

            generated.append(next_token)

        # Remove <SOS> from the beginning before returning
        result_ids = [t for t in generated[1:] if t != self.start_token_id]

        # # Decode token ids back to text using tokenizer
        # generated_text = tokenizer.decode(result_ids)

        # return result_ids, generated_text
        return result_ids

    def _beam_search(self, encoder_output, src_mask_4d,
                     max_length, beam_width, repetition_penalty, device):
        beams = [(0.0, [self.start_token_id])]
        # beams = [(score, sequence)]
        completed_beams = []

        for step in range(max_length):
            if not beams:
                break  # all beams completed

            all_candidates = []

            for log_prob, ids in beams:
                # If this beam already ended, move to completed
                if ids[-1] == self.end_token_id:
                    completed_beams.append((log_prob, ids))
                    continue

                # Get next token logits for this beam
                tgt = torch.tensor([ids], dtype=torch.long, device=device)
                logits = self.decode(tgt, encoder_output, src_mask_4d)
                next_logits = logits[0, -1, :].clone()
                # next_logits: (vocab_size,)

                # Apply repetition penalty
                for prev_token in set(ids):
                    if next_logits[prev_token] > 0:
                        next_logits[prev_token] /= repetition_penalty
                    else:
                        next_logits[prev_token] *= repetition_penalty

                # Prevent generating EOS at the very first step
                if step == 0:
                    next_logits[self.end_token_id] = float('-inf')

                # Convert to log probabilities
                log_probs = torch.log_softmax(next_logits, dim=-1)

                # Take top beam_width candidates for THIS beam
                top_log_probs, top_ids = log_probs.topk(beam_width)

                for next_log_prob, next_token_id in zip(top_log_probs, top_ids):
                    candidate_log_prob = log_prob + next_log_prob.item()
                    candidate_ids      = ids + [next_token_id.item()]
                    all_candidates.append((candidate_log_prob, candidate_ids))

            if not all_candidates:
                break

            # Sort all candidates by score, keep top beam_width
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_width]

        # Add remaining active beams to completed
        completed_beams.extend(beams)

        if not completed_beams:
            # Fallback: return empty
            return [], ""

        # Score by length-normalized log probability

        # Longer sequences accumulate more log probs (more negative).
        # Without normalization, beam search always prefers shorter answers.
        # Dividing by length^0.6 gives a fair comparison.
        
        def length_penalty(beam):
            log_prob, ids = beam
            length = max(len(ids) - 1, 1)  # exclude <SOS>
            return log_prob / (length ** 0.6)

        best_beam = max(completed_beams, key=length_penalty)
        best_ids  = best_beam[1]

        # Remove <SOS> and <EOS> tokens
        result_ids = [
            t for t in best_ids
            if t not in (self.start_token_id, self.end_token_id)
        ]

        return result_ids


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math
    from config import Config
    
    cfg = Config()
    B, S_src, S_tgt = 4, 15, 20
    V = cfg.vocab_size
    D = cfg.hidden_dim

    print("=" * 55)
    print("TransformerQA - self test")
    print("=" * 55)

    # ── Random init ──────────────────────────────────────────────
    model = TransformerQA(
        vocab_size=V, 
        hidden_dim=D,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim
    )

    src = torch.randint(3, V, (B, S_src))
    tgt = torch.randint(3, V, (B, S_tgt))
    src[:, 12:] = 0   # simulate padding

    logits = model(src, tgt)
    assert logits.shape == (B, S_tgt, V), f"Wrong shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"
    print(f"forward(): {src.shape}, {tgt.shape} to {logits.shape}  OK")

    # ── Word2Vec init ─────────────────────────────────────────────
    w2v = np.random.randn(V, D).astype(np.float32)
    w2v /= np.linalg.norm(w2v, axis=1, keepdims=True) + 1e-8

    model_w2v = TransformerQA(
        vocab_size=V, 
        hidden_dim=D,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        embeddings=w2v
    )
    logits2   = model_w2v(src, tgt)
    assert not torch.isnan(logits2).any(), "NaN with Word2Vec init"
    print(f"Word2Vec init:  OK")

    # ── src_mask flow (critical bug fix) ─────────────────────────
    enc_out, src_mask_4d = model.encode(src)
    assert src_mask_4d.shape == (B, 1, 1, S_src)
    dec_logits = model.decode(tgt, enc_out, src_mask_4d)
    assert not torch.isnan(dec_logits).any(), "NaN — cross-attention broken"
    print(f"src_mask flow:  OK  (cross-attention working)")

    # ── Greedy generation ─────────────────────────────────────────
    class FakeTok:
        def decode(self, ids): return f"[{len(ids)} tokens]"

    single_src = torch.randint(3, V, (1, S_src))
    ids, text = model.generate(single_src, FakeTok(), max_length=15, beam_width=1)
    assert model.start_token_id not in ids, "<start> should be removed"
    print(f"Greedy:         OK  ids={ids[:4]}...")

    # ── Beam search ───────────────────────────────────────────────
    ids_b, text_b = model.generate(single_src, FakeTok(), max_length=15, beam_width=4)
    print(f"Beam search:    OK  ids={ids_b[:4]}...")

    total = sum(p.numel() for p in model_w2v.parameters())
    print(f"\nTotal parameters: {total:,}")
    print("=" * 55)
    print("All tests passed. Ready for training.")
