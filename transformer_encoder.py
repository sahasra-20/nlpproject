
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

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
        batch_size = query.shape[0]

        # Linear projections
        Q = self.query(query)  # (batch_size, seq_len_q, hidden_dim)
        K = self.key(key)      # (batch_size, seq_len_k, hidden_dim)
        V = self.value(value)  # (batch_size, seq_len_v, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        # Guard against NaN that arises when every position in a row is masked
        # (all -inf scores): softmax(-inf,...,-inf) = NaN → replace with 0.
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)

        # Reshape back
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, -1, self.hidden_dim)

        # Output projection
        output = self.fc_out(attended_values)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer - ENCODER COMPONENT"""

    def __init__(self, hidden_dim, max_seq_len=100):
        super().__init__()

        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term = 10000^(2i/d)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float) *
            -(math.log(10000.0) / hidden_dim)
        )

        # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        pe[:, 0::2] = torch.sin(position * div_term)

        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)

        Returns:
            x + positional_encoding: (batch_size, seq_len, hidden_dim)
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class FeedForwardNetwork(nn.Module):
    """Feedforward network: Linear -> ReLU -> Linear - ENCODER COMPONENT"""

    def __init__(self, hidden_dim, ff_dim=512, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

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


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer - ENCODER FOCUS"""

    def __init__(self, hidden_dim, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: attention mask

        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)

        # Feedforward with residual and layer norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder: Stack of encoder layers - ENCODER FOCUS"""

    def __init__(self, hidden_dim, num_layers=2, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: attention mask

        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x, mask)

        return x


if __name__ == "__main__":
    # Test encoder components
    batch_size = 2
    seq_len = 10
    hidden_dim = 256

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Test attention
    attn = MultiHeadAttention(hidden_dim, num_heads=4)
    output, weights = attn(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Test encoder
    encoder = TransformerEncoder(hidden_dim, num_layers=2, num_heads=4)
    enc_output = encoder(x)
    print(f"Encoder output shape: {enc_output.shape}")

    print("Encoder components working independently!")