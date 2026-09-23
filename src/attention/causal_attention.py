"""
Causal Attention
=================
Auto-atenção com máscara triangular inferior, fundamental para modelos autoregressivos como o GPT.
"""
import torch
import torch.nn as nn
from src.attention.scaled_dot_product import scaled_dot_product_attention


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Buffer com a máscara triangular (não é parâmetro treinável)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Recorta a máscara para o tamanho real da sequência
        mask = self.mask[:num_tokens, :num_tokens].unsqueeze(0)
        
        context_vecs, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_layer=self.dropout
        )
        return self.out_proj(context_vecs), weights


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(1, 4, 16)
    causal = CausalAttention(d_in=16, d_out=16, context_length=4)
    out, weights = causal(x)
    
    print("=== Demonstração: Causal Attention ===")
    print("Matriz de pesos causal (triangular inferior):")
    print(weights[0].detach().cpu().numpy().round(3))
