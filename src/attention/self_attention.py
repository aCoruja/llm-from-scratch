"""
Self-Attention (Single-Head)
=============================
Projeta a entrada X em Query, Key e Value através de matrizes treináveis W_q, W_k, W_v.
"""
import torch
import torch.nn as nn
from src.attention.scaled_dot_product import scaled_dot_product_attention


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x, mask=None, scale=True):
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        
        context_vecs, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, scale=scale
        )
        return self.out_proj(context_vecs), weights


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 4, 16)
    sa = SelfAttention(d_in=16, d_out=32)
    out, weights = sa(x)
    
    print("=== Demonstração: Self-Attention ===")
    print(f"Entrada: {tuple(x.shape)}")
    print(f"Saída:   {tuple(out.shape)}")
    print(f"Pesos:   {tuple(weights.shape)}")
