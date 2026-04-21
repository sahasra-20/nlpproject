
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"


        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.attn_dropout = nn.Dropout(dropout)

        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(w.weight)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, hidden_dim)
            key: (batch_size, seq_len_k, hidden_dim)
            value: (batch_size, seq_len_v, hidden_dim)
            mask: (batch_size, seq_len_q, seq_len_k) or None

        Returns:
            output: (batch_size, seq_len_q, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # Project
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax attention weights
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        # Weighted values
        out = torch.matmul(attn, V)                     # (batch, heads, seq_q, head_dim)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_q, self.hidden_dim)
        out = self.W_o(out)                             # (batch, seq_q, hidden_dim)

        return out, attn


class FeedForwardNetwork(nn.Module):
    """Feedforward network: Linear -> ReLU -> Linear - DECODER COMPONENT"""

    def __init__(self, hidden_dim, ff_dim=512, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)

        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer - DECODER FOCUS"""

    def __init__(self, hidden_dim, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()

        self.self_attn   = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ff_dim, dropout)

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.norm3   = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: (batch_size, tgt_seq_len, hidden_dim)
            encoder_output: (batch_size, src_seq_len, hidden_dim)
            self_mask: causal mask for self-attention
            cross_mask: mask for cross-attention

        Returns:
            output: (batch_size, tgt_seq_len, hidden_dim)
        """
        self_out, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_out))

        cross_out, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_out))


        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder: Stack of decoder layers - DECODER FOCUS"""

    def __init__(self, hidden_dim, num_layers=2, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: (batch_size, tgt_seq_len, hidden_dim)
            encoder_output: (batch_size, src_seq_len, hidden_dim)
            self_mask: causal mask for self-attention
            cross_mask: mask for cross-attention

        Returns:
            output: (batch_size, tgt_seq_len, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)

        return x


def create_causal_mask(seq_len, device):
    """Create causal mask (lower triangular matrix) - DECODER UTILITY"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(device)
    return mask


def create_padding_mask(seq, pad_token_id=0):
    """Create padding mask - DECODER UTILITY"""
    return (seq != pad_token_id).bool()


if __name__ == "__main__":
    B, S_src, S_tgt, D = 32, 20, 30, 256

    enc_out = torch.randn(B, S_src, D)
    x       = torch.randn(B, S_tgt, D)

    # Source padding mask (last 5 positions are padding)
    src_ids = torch.randint(1, 8000, (B, S_src))
    src_ids[:, 15:] = 0
    cross_mask = create_padding_mask(src_ids).unsqueeze(1).unsqueeze(1)  # (B,1,1,S_src)

    # Causal mask for target
    causal    = create_causal_mask(S_tgt, device=x.device)
    self_mask = causal.unsqueeze(0).unsqueeze(0)                         # (1,1,S_tgt,S_tgt)

    dec = TransformerDecoder(D, num_layers=2, num_heads=4, ff_dim=512)
    out = dec(x, enc_out, self_mask, cross_mask)

    assert out.shape == (B, S_tgt, D)
    assert not torch.isnan(out).any(), "NaN in decoder output"
    print(f"TransformerDecoder: {x.shape} to {out.shape}  OK")

    # Verify cross-attention shape: (batch, heads, tgt_len, src_len) — rectangular
    attn_module = dec.layers[0].cross_attn
    q  = torch.randn(B, S_tgt, D)
    kv = torch.randn(B, S_src, D)
    _, w = attn_module(q, kv, kv)
    assert w.shape == (B, 4, S_tgt, S_src), f"Wrong cross-attn shape: {w.shape}"
    print(f"Cross-attn weights: {w.shape}  OK  (batch, heads, tgt_len, src_len)")
    print(f"Parameters: {sum(p.numel() for p in dec.parameters()):,}")