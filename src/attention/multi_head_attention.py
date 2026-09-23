"""
Multi-Head Attention (MHA)
===========================
Divide as representações em h cabeças de dimensão d_k = d_out / h.
Executa a atenção em paralelo e recombina o resultado com uma projeção linear final.
"""
import torch
import torch.nn as nn
from src.attention.scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out precisa ser divisível por num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x, is_causal=True):
        b, num_tokens, _ = x.shape

        # Projeta e divide nas cabeças:
        # (b, tokens, d_out) -> (b, tokens, num_heads, head_dim) -> (b, num_heads, tokens, head_dim)
        Q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        mask = self.mask[:num_tokens, :num_tokens].unsqueeze(0).unsqueeze(0) if is_causal else None

        context_vecs, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_layer=self.dropout
        )

        # Concatena as cabeças de volta: (b, num_heads, tokens, head_dim) -> (b, tokens, d_out)
        context_vecs = context_vecs.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vecs), weights


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 4, 16)
    mha = MultiHeadAttention(d_in=16, d_out=32, context_length=4, num_heads=4)
    out, weights = mha(x)
    
    print("=== Demonstração: Multi-Head Attention ===")
    print(f"Shape Saída: {tuple(out.shape)}")
    print(f"Shape Pesos (Batch, Heads, Tokens, Tokens): {tuple(weights.shape)}")
    print(f"Parâmetros totais da camada: {sum(p.numel() for p in mha.parameters())}")
