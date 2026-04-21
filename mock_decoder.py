# mock_decoder.py
# Person 5: Mock Decoder logic to replace missing Person 4 code
import torch
import torch.nn as nn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        # Mock layer does essentially a linear pass and normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        # We just pass x to enc_out shape for mock behavior
        # Keep tensors matching so it doesn't crash during backprop
        return self.norm(x)

class TransformerDecoder(nn.Module):
    """Placeholder Decoder mimicking Person 4's component."""
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, tgt_emb, encoder_output, self_mask=None, cross_mask=None):
        x = tgt_emb
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return x

def create_causal_mask(seq_len, device):
    """Mock causal mask: lower triangular matrix."""
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    return mask

def create_padding_mask(seq_ids, pad_id):
    """Mock padding mask."""
    # True where it is NOT padding
    return (seq_ids != pad_id).bool()
